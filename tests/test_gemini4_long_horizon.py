from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eval_runner import AgentRegistry, execute_evaluation_run
from hamster_wheel import ModelHamsterWheel, WheelStageConfig
import hamster_wheel
from indirect_input_classifier import build_default_template_library
from live_adapters import (
    FREE_GEMINI_MODELS,
    PRIORITY_GEMINI_MODELS,
    LiveHttpAgent,
    LiveProviderConfig,
    ProviderKind,
    build_gemini_endpoint_url,
    discover_live_agents,
    extract_gemini_parts_text,
    infer_architecture_family,
    is_multimodal_model,
    is_vertex_express_endpoint,
)
from live_providers import _parse_gemini, discover_live_agents as discover_provider_specs
from response_classifier import ResponseOutcome
import run_live_eval
from template_library import BenignTemplateLibrary
from tool_runtime import LocalMcpServer, render_diagnostic_chart_png


class _MockJsonResp:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def test_gemini4_priority_models_detection_and_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    assert PRIORITY_GEMINI_MODELS == ("gemini-4-carbon", "gemini-4-neon", "gemini-4-argon")

    for model_id in (*PRIORITY_GEMINI_MODELS, "gemini-4-hydrogen", "eriktonon/gemma4-uncensored:latest"):
        assert is_multimodal_model(model_id) is True

    assert is_multimodal_model("qwen2.5-coder:7b") is False
    assert infer_architecture_family(ProviderKind.GEMINI, "gemini-4-carbon") == "gemini-4-thinking-multimodal"

    monkeypatch.setenv("GEMINI_API_KEY", "test-gem4-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    class _OllamaTagsSession:
        def get(self, url: str, timeout: float = 1.5):
            if url.endswith("/api/tags"):
                return _MockJsonResp(
                    {
                        "models": [
                            {"name": "llama3.1:latest"},
                            {"name": "qwen3.8-27b-longctx:latest"},
                            {"name": "eriktonon/deepseek-v4.1-flash-1017:9b"},
                        ]
                    }
                )
            return _MockJsonResp({})

    discovered = discover_live_agents(session=_OllamaTagsSession())
    gem_ids = [ag.model_id for ag in discovered if ag.config.provider == ProviderKind.GEMINI]
    ollama_ids = [ag.model_id for ag in discovered if ag.config.provider == ProviderKind.OLLAMA]

    assert gem_ids == ["gemini-4-carbon", "gemini-4-neon", "gemini-4-argon"]
    assert ollama_ids[0] == "qwen3.8-27b-longctx:latest"

    specs = {s.provider: s for s in discover_provider_specs(timeout=0.2)}
    assert specs["gemini"].model == "gemini-4-neon"


def test_gemini4_thinking_config_thought_separation_and_tool_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gem4-key")
    captured_payloads: list[dict[str, Any]] = []
    call_counter = {"n": 0}

    class _GeminiMockSession:
        def post(self, url: str, headers=None, json=None, timeout=None):
            captured_payloads.append(json or {})
            call_counter["n"] += 1
            # Round 1: function call paired with thought block + thoughtSignature
            if json and "tools" in json and call_counter["n"] == 1:
                return _MockJsonResp(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [
                                        {"thought": True, "text": "Internal plan step"},
                                        {
                                            "functionCall": {
                                                "name": "convert_temperature",
                                                "args": {"value": 25, "from_unit": "celsius", "to_unit": "fahrenheit"},
                                            },
                                            "thoughtSignature": "sig-gem4-xyz",
                                        },
                                    ],
                                }
                            }
                        ],
                        "usageMetadata": {"promptTokenCount": 40, "candidatesTokenCount": 12},
                    }
                )
            # Round 2 / generate / multimodal: thought summary + visible strict JSON answer
            return _MockJsonResp(
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {"thought": True, "text": "Thinking process should stay separated"},
                                    {"text": '{"selected_tool":"convert_temperature","result":77.0}'},
                                ],
                            }
                        }
                    ],
                    "usageMetadata": {"promptTokenCount": 55, "candidatesTokenCount": 18},
                }
            )

    for model_id, level in (
        ("gemini-4-carbon", "high"),
        ("gemini-4-neon", "xhigh"),
        ("gemini-4-argon", "medium"),
    ):
        call_counter["n"] = 0
        captured_payloads.clear()
        agent = LiveHttpAgent(
            LiveProviderConfig(
                provider=ProviderKind.GEMINI,
                model_id=model_id,
                model_version="live-gem4",
                thinking_level=level,
            ),
            session=_GeminiMockSession(),
        )
        # 1) Tool loop preserves thoughtSignature in round-2 contents and filters thought=True from final_response
        mcp = LocalMcpServer(include_indirect_perturbations=False, live_weather=False, live_threat_intel=False)
        loop = agent.run_tool_loop(
            system_prompt="Call convert_temperature and return JSON.",
            user_prompt="Convert 25C to F.",
            mcp_server=mcp,
            tool_names=["convert_temperature"],
        )
        assert captured_payloads[0]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": level}
        model_turn_parts = captured_payloads[1]["contents"][1]["parts"]
        assert any(p.get("thoughtSignature") == "sig-gem4-xyz" for p in model_turn_parts)
        assert json.loads(loop.final_response)["result"] == 77.0

        # 2) Real-PNG multimodal probe routes cleanly to Gemini 4
        mm = agent.run_multimodal_probe(
            system_prompt="Return JSON.",
            user_prompt="Describe chart.",
            png_bytes=render_diagnostic_chart_png(["Carbon: 99%", "Neon: 98%"]),
        )
        assert "Thinking process" not in mm.response_text
        assert "77.0" in mm.response_text

    # Verify live_providers._parse_gemini separates thought=True parts cleanly
    parsed_text, _ = _parse_gemini(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"thought": True, "text": "Internal CoT trace"},
                            {"text": '{"status":"ok"}'},
                        ]
                    }
                }
            ]
        },
        "gemini",
    )
    assert parsed_text == '{"status":"ok"}'
    assert extract_gemini_parts_text([{"thought": True, "text": "Only thought"}]) == "Only thought"


def test_long_horizon_suites_and_gemini_priority_clis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lib = BenignTemplateLibrary()
    crescendo = lib.get("long_horizon_crescendo_drift").render()
    many_shot = lib.get("long_horizon_many_shot_context").render()

    assert len(crescendo.user_turns) == 6
    assert len(many_shot.user_turns) == 5

    agent = AgentRegistry().get("policy-assistant@v2.0-hardened")
    eval_run = execute_evaluation_run(
        run_id="long-horizon-smoke",
        agent=agent,
        test_cases=[crescendo, many_shot],
    )
    assert len(eval_run.records) == 2
    assert all(
        r.outcome in (ResponseOutcome.SAFE_REFUSAL, ResponseOutcome.SAFE_REDIRECT)
        for r in eval_run.records
    )

    # Verify wheel battery supports long-horizon + CLI flags dispatch all 3 PRIORITY_GEMINI_MODELS
    monkeypatch.setenv("GEMINI_API_KEY", "test-gem4-key")
    monkeypatch.setattr("requests.Session.get", lambda self, url, **kw: _MockJsonResp({"name": "models/gemini-4"}))
    monkeypatch.setattr("requests.Session.post", lambda self, url, **kw: _MockJsonResp({
        "candidates": [{"content": {"parts": [{"text": '{"status":"safe","summary":"boundary preserved"}'}]}}]
    }))

    rc_eval = run_live_eval.main(
        [
            "--gemini-priority",
            "--long-horizon",
            "--thinking-level",
            "medium",
            "--judge-mode",
            "policy_anchor",
            "--output-dir",
            str(tmp_path / "live"),
        ]
    )
    assert rc_eval == 0
    saved_live = list((tmp_path / "live").glob("*.jsonl"))
    assert len(saved_live) == 3
    names = " ".join(p.name for p in saved_live)
    assert "gemini-4-carbon" in names and "gemini-4-neon" in names and "gemini-4-argon" in names

    wheel_dir = tmp_path / "wheel_runs"
    rc_wheel = hamster_wheel.main(
        [
            "--gemini-priority",
            "--long-horizon",
            "--skip-indirect",
            "--output-dir",
            str(wheel_dir),
        ]
    )
    assert rc_wheel == 0
    wheel_reports = ModelHamsterWheel.load_saved_reports(wheel_dir)
    assert {r.model_id for r in wheel_reports} == set(PRIORITY_GEMINI_MODELS)
    assert all(r.total_cases_executed >= 4 for r in wheel_reports)

    # Also ensure indirect baseline template TPL-01 + long-horizon can coexist without conflict
    tpl01 = build_default_template_library().get("TPL-01-ROLE-TOOL")
    assert tpl01.template_id == "TPL-01-ROLE-TOOL"
    # Verify WheelStageConfig custom long-horizon subset directly
    lh_cfg = WheelStageConfig(
        include_indirect_paired=False,
        include_benign_suite=True,
        benign_template_ids=("long_horizon_crescendo_drift", "long_horizon_many_shot_context"),
    )
    assert len(ModelHamsterWheel.build_battery(lh_cfg)) == 2


def test_vertex_express_routing_health_tool_loop_and_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert FREE_GEMINI_MODELS == ("gemini-4-hydrogen", "gemini-3.8-flash", "gemini-3.6-flash")
    assert is_vertex_express_endpoint(ProviderKind.VERTEX_EXPRESS) is True
    assert is_vertex_express_endpoint(ProviderKind.GEMINI, "https://aiplatform.googleapis.com") is True
    assert is_vertex_express_endpoint(ProviderKind.GEMINI, "https://generativelanguage.googleapis.com") is False

    assert (
        build_gemini_endpoint_url(
            ProviderKind.GEMINI,
            "https://generativelanguage.googleapis.com",
            "gemini-4-hydrogen",
            "generateContent",
        )
        == "https://generativelanguage.googleapis.com/v1beta/models/gemini-4-hydrogen:generateContent"
    )
    assert (
        build_gemini_endpoint_url(
            ProviderKind.VERTEX_EXPRESS,
            "https://aiplatform.googleapis.com",
            "gemini-4-carbon",
            "generateContent",
        )
        == "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-4-carbon:generateContent"
    )
    assert (
        build_gemini_endpoint_url(
            ProviderKind.VERTEX_EXPRESS,
            "https://aiplatform.googleapis.com/v1beta1",
            "gemini-4-neon",
            "countTokens",
        )
        == "https://aiplatform.googleapis.com/v1beta1/publishers/google/models/gemini-4-neon:countTokens"
    )
    assert (
        build_gemini_endpoint_url(
            ProviderKind.GEMINI,
            "https://aiplatform.googleapis.com",
            "gemini-4-argon",
            "generateContent",
        )
        == "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-4-argon:generateContent"
    )

    monkeypatch.setenv("VERTEX_API_KEY", "vtx-test-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    posted: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    class _VertexMockSession:
        def get(self, url: str, timeout: float = 1.5, **kwargs):
            return _MockJsonResp({"models": []})

        def post(self, url: str, headers=None, json=None, timeout=None):
            posted.append((url, dict(headers or {}), dict(json or {})))
            if url.endswith(":countTokens"):
                return _MockJsonResp({"totalTokens": 3})
            if json and "tools" in json and sum(1 for u, _, _ in posted if u.endswith(":generateContent")) == 1:
                return _MockJsonResp(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "functionCall": {
                                                "name": "convert_temperature",
                                                "args": {"value": 25, "from_unit": "celsius", "to_unit": "fahrenheit"},
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                )
            return _MockJsonResp(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"thought": True, "text": "Vertex CoT"},
                                    {"text": '{"selected_tool":"convert_temperature","result":77.0}'},
                                ]
                            }
                        }
                    ],
                    "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 8},
                }
            )

    discovered = discover_live_agents(session=_VertexMockSession())
    vertex_ids = [ag.model_id for ag in discovered if ag.config.provider == ProviderKind.VERTEX_EXPRESS]
    assert vertex_ids == ["gemini-4-carbon", "gemini-4-neon", "gemini-4-argon"]

    agent = LiveHttpAgent(
        LiveProviderConfig(
            provider=ProviderKind.VERTEX_EXPRESS,
            model_id="gemini-4-carbon",
            model_version="live-vertex4",
            thinking_level="high",
        ),
        session=_VertexMockSession(),
    )
    ok, msg = agent.health_check()
    assert ok is True
    assert "countTokens" in msg
    assert posted[0][0] == "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-4-carbon:countTokens"
    assert posted[0][1]["x-goog-api-key"] == "vtx-test-key"

    mcp = LocalMcpServer(include_indirect_perturbations=False, live_weather=False, live_threat_intel=False)
    loop = agent.run_tool_loop(
        system_prompt="Use tool.",
        user_prompt="Convert 25C to F.",
        mcp_server=mcp,
        tool_names=["convert_temperature"],
    )
    assert json.loads(loop.final_response)["result"] == 77.0
    gen_url, gen_headers, gen_body = posted[1]
    assert gen_url == "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-4-carbon:generateContent"
    assert gen_headers["x-goog-api-key"] == "vtx-test-key"
    assert gen_body["systemInstruction"]["parts"][0]["text"] == "Use tool."
    assert "system_instruction" not in gen_body
    assert gen_body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "HIGH"}

    mm = agent.run_multimodal_probe(
        system_prompt="Return JSON.",
        user_prompt="Read chart.",
        png_bytes=render_diagnostic_chart_png(["Carbon: 99%"]),
    )
    assert "77.0" in mm.response_text
    assert posted[-1][0].endswith("/v1/publishers/google/models/gemini-4-carbon:generateContent")


def test_gemini_free_and_vertex_clis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_urls: list[str] = []
    monkeypatch.setenv("GEMINI_API_KEY", "free-gem-key")
    monkeypatch.setenv("VERTEX_API_KEY", "vtx-cli-key")

    def _mock_get(self, url: str, **kw):
        seen_urls.append(url)
        return _MockJsonResp({"name": "models/gemini-free"})

    def _mock_post(self, url: str, **kw):
        seen_urls.append(url)
        if url.endswith(":countTokens"):
            return _MockJsonResp({"totalTokens": 2})
        return _MockJsonResp(
            {"candidates": [{"content": {"parts": [{"text": '{"status":"safe","summary":"free tier ok"}'}]}}]}
        )

    monkeypatch.setattr("requests.Session.get", _mock_get)
    monkeypatch.setattr("requests.Session.post", _mock_post)

    # 1) --gemini-free on run_live_eval + hamster_wheel
    rc_eval = run_live_eval.main(
        [
            "--gemini-free",
            "--long-horizon",
            "--judge-mode",
            "policy_anchor",
            "--output-dir",
            str(tmp_path / "free_live"),
        ]
    )
    assert rc_eval == 0
    free_live_files = " ".join(p.name for p in (tmp_path / "free_live").glob("*.jsonl"))
    for model_id in FREE_GEMINI_MODELS:
        assert model_id in free_live_files

    wheel_free_dir = tmp_path / "free_wheel"
    rc_wheel = hamster_wheel.main(
        [
            "--gemini-free",
            "--long-horizon",
            "--skip-indirect",
            "--output-dir",
            str(wheel_free_dir),
        ]
    )
    assert rc_wheel == 0
    assert {r.model_id for r in ModelHamsterWheel.load_saved_reports(wheel_free_dir)} == set(FREE_GEMINI_MODELS)

    # 2) --provider vertex_express --gemini-priority hits aiplatform countTokens & publishers/google generateContent
    seen_urls.clear()
    rc_vtx = run_live_eval.main(
        [
            "--provider",
            "vertex_express",
            "--gemini-priority",
            "--suite",
            "benign",
            "--templates",
            "long_horizon_crescendo_drift",
            "--judge-mode",
            "policy_anchor",
            "--thinking-level",
            "low",
            "--output-dir",
            str(tmp_path / "vtx_live"),
        ]
    )
    assert rc_vtx == 0
    assert any("aiplatform.googleapis.com/v1/publishers/google/models/gemini-4-carbon:countTokens" in u for u in seen_urls)
    assert any("aiplatform.googleapis.com/v1/publishers/google/models/gemini-4-neon:generateContent" in u for u in seen_urls)


def test_backend_overview_and_vertex_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket
    import threading
    import time
    import requests
    import uvicorn
    from backend_server import create_backend_app

    monkeypatch.setenv("VERTEX_API_KEY", "vtx-web-key")
    real_post = requests.Session.post

    def _selective_post(self, url: str, **kw):
        if "aiplatform.googleapis.com" in url:
            assert url.endswith(":countTokens")
            return _MockJsonResp({"totalTokens": 2})
        return real_post(self, url, **kw)

    monkeypatch.setattr("requests.Session.post", _selective_post)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])

    app = create_backend_app(
        runs_dir=tmp_path / "wb_runs",
        corpus_dir=tmp_path / "wb_corpus",
        live_weather=False,
        discover_live=False,
    )
    server = uvicorn.Server(uvicorn.Config(app=app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(50):
            try:
                if requests.get(f"{base}/health", timeout=0.5).status_code == 200:
                    break
            except Exception:
                time.sleep(0.1)

        ov = requests.get(f"{base}/api/overview", timeout=3.0).json()
        assert ov["free_gemini_models"] == list(FREE_GEMINI_MODELS)
        assert ov["priority_gemini_models"] == list(PRIORITY_GEMINI_MODELS)

        h_res = requests.post(
            f"{base}/api/agents/health",
            json={"provider": "vertex_express", "model_id": "gemini-4-neon", "thinking_level": "low"},
            timeout=3.0,
        )
        assert h_res.status_code == 200 and h_res.json()["ok"] is True

        reg_res = requests.post(
            f"{base}/api/agents/register",
            json={"provider": "vertex_express", "model_id": "gemini-4-carbon", "model_version": "vtx-test"},
            timeout=3.0,
        )
        assert reg_res.status_code == 200
        assert reg_res.json()["agent"]["provider"] == "vertex_express"
    finally:
        server.should_exit = True
        thread.join(timeout=3.0)


