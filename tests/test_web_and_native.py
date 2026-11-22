"""End-to-end integration tests for the native Rust binary, Starlette/Uvicorn backend, WebSockets, and web UI."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time

import pytest
import requests
import uvicorn
from websockets.exceptions import ConnectionClosedOK
from websockets.sync.client import connect as ws_connect

from agent_comm_robustness import (
    AgentMessage,
    FramingMode,
    HardenedBoundaryGate,
    MessageEncoder,
    PayloadConfig,
    Performative,
    Provenance,
    sign_observed_claim,
)
from backend_server import (
    NaiveBaselineJsonParser,
    _build_wire_variants,
    _find_native_binary,
    create_backend_app,
)


def _run_native_batch(
    binary: Path,
    encoded_list: list,
    framing: FramingMode,
    cfg: PayloadConfig,
    *,
    require_sig: bool = True,
    reject_special: bool = True,
) -> list[dict]:
    stdin_payload = "".join(
        base64.b64encode(enc.raw_bytes).decode("ascii") + "\n" for enc in encoded_list
    ).encode("ascii")
    proc = subprocess.run(
        [
            str(binary),
            "wire-batch",
            "--framing",
            framing.value,
            "--max-wire-bytes",
            str(cfg.max_wire_bytes),
            "--max-claim-chars",
            str(cfg.max_claim_chars),
            "--max-nesting-depth",
            str(cfg.max_nesting_depth),
            "--require-signed-observed",
            "true" if require_sig else "false",
            "--reject-special-tokens",
            "true" if reject_special else "false",
            "--oracle-key",
            cfg.oracle_key.decode("utf-8"),
            "--stdin",
            "--json",
        ],
        input=stdin_payload,
        capture_output=True,
        timeout=15.0,
        check=True,
    )
    return json.loads(proc.stdout.decode("utf-8"))


class _LiveBackendHandle:
    def __init__(self, base_url: str, ws_base_url: str, session: requests.Session) -> None:
        self.base_url = base_url
        self.ws_base_url = ws_base_url
        self.session = session

    def get(self, path: str, **kwargs) -> requests.Response:
        return self.session.get(f"{self.base_url}{path}", timeout=15.0, **kwargs)

    def post(self, path: str, **kwargs) -> requests.Response:
        return self.session.post(f"{self.base_url}{path}", timeout=30.0, **kwargs)

    def delete(self, path: str, **kwargs) -> requests.Response:
        return self.session.delete(f"{self.base_url}{path}", timeout=15.0, **kwargs)

    def drain_ws_job(self, job_id: str) -> dict:
        final_msg: dict = {}
        with ws_connect(f"{self.ws_base_url}/ws/jobs/{job_id}", open_timeout=10.0, close_timeout=10.0) as ws:
            while True:
                try:
                    raw = ws.recv(timeout=20.0)
                except ConnectionClosedOK:
                    break
                msg = json.loads(raw)
                final_msg = msg
                if msg.get("event") == "snapshot" and msg.get("status") in ("completed", "failed"):
                    break
                if msg.get("event") in ("job_complete", "job_failed"):
                    break
        return final_msg


@pytest.fixture()
def live_server(tmp_path: Path):
    app = create_backend_app(
        runs_dir=tmp_path / "runs",
        corpus_dir=tmp_path / "corpus",
        live_weather=False,
        discover_live=False,
    )
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(32)
    port = sock.getsockname()[1]

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    ws_base_url = f"ws://127.0.0.1:{port}"
    session = requests.Session()

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            r = session.get(f"{base_url}/health", timeout=1.0)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.05)
    else:
        server.should_exit = True
        thread.join(timeout=3.0)
        sock.close()
        pytest.fail("Timed out waiting for live uvicorn backend server to start")

    try:
        yield _LiveBackendHandle(base_url, ws_base_url, session)
    finally:
        session.close()
        server.should_exit = True
        thread.join(timeout=5.0)
        try:
            sock.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 1. Native Rust Binary Tests & Python/Rust Gate Parity
# ---------------------------------------------------------------------------


def test_native_binary_present_and_version_json():
    binary = _find_native_binary()
    assert binary is not None and binary.is_file(), "Compiled native work121 binary not found"
    proc = subprocess.run([str(binary), "version"], capture_output=True, text=True, timeout=5.0, check=True)
    info = json.loads(proc.stdout)
    assert info["name"] == "work121-native"
    assert "version" in info


@pytest.mark.parametrize("framing", [FramingMode.RAW, FramingMode.LENGTH_PREFIXED, FramingMode.NDJSON])
def test_native_wire_batch_100pct_parity_across_framings(framing: FramingMode):
    binary = _find_native_binary()
    assert binary is not None
    cfg, variants = _build_wire_variants(framing=framing, include_benign_edges=True, include_controls=True, seed=42)
    encoder = MessageEncoder(framing=framing)
    gate = HardenedBoundaryGate(
        oracle_key=cfg.oracle_key,
        framing=framing,
        require_signed_observed=True,
        max_wire_bytes=cfg.max_wire_bytes,
        max_claim_chars=cfg.max_claim_chars,
        max_nesting_depth=cfg.max_nesting_depth,
        reject_special_tokens=True,
    )
    encoded_list = list(encoder.encode_stream(variants))
    native_results = _run_native_batch(binary, encoded_list, framing, cfg)

    assert len(native_results) == len(encoded_list)
    mismatches = []
    for enc, nres in zip(encoded_list, native_results):
        py_res = gate.parse_bytes(enc.raw_bytes)
        if nres["accepted"] != py_res.accepted or nres["error_code"] != py_res.error_code:
            mismatches.append((enc.variant.case_id, py_res.accepted, py_res.error_code, nres))
    assert not mismatches, f"Python/Rust parity mismatches for {framing.value}: {mismatches[:5]}"


def test_native_wire_check_empty_stream_and_toggled_flags():
    binary = _find_native_binary()
    assert binary is not None
    cfg = PayloadConfig(framing=FramingMode.RAW, seed=42)
    gate_permissive = HardenedBoundaryGate(
        oracle_key=cfg.oracle_key,
        framing=FramingMode.RAW,
        require_signed_observed=False,
        reject_special_tokens=False,
    )

    proc_empty = subprocess.run(
        [str(binary), "wire-check", "--framing", "raw", "--stdin", "--json"],
        input=b"",
        capture_output=True,
        timeout=5.0,
        check=False,
    )
    res_empty = json.loads(proc_empty.stdout.decode("utf-8"))
    py_empty = gate_permissive.parse_bytes(b"")
    assert res_empty["accepted"] is False
    assert res_empty["error_code"] == py_empty.error_code == "empty_wire_bytes"

    msg = {
        "msg_id": "m-permissive",
        "sender": "agent_a",
        "recipient": "agent_b",
        "round_idx": 1,
        "performative": "propose",
        "claim": "Literal token <|im_start|> mentioned in documentation",
        "answer": 7,
        "confidence": 0.9,
        "provenance": "observed",
        "constraints": [],
        "evidence_refs": [],
    }
    raw = json.dumps(msg).encode("utf-8")
    proc_perm = subprocess.run(
        [
            str(binary),
            "wire-check",
            "--framing",
            "raw",
            "--require-signed-observed",
            "false",
            "--reject-special-tokens",
            "false",
            "--stdin",
            "--json",
        ],
        input=raw,
        capture_output=True,
        timeout=5.0,
        check=False,
    )
    res_perm = json.loads(proc_perm.stdout.decode("utf-8"))
    py_perm = gate_permissive.parse_bytes(raw)
    assert res_perm["accepted"] == py_perm.accepted is True
    assert res_perm["error_code"] is None


def test_weak_baseline_diverges_while_hardened_and_native_reject_attacks():
    cfg, variants = _build_wire_variants(framing=FramingMode.RAW, include_benign_edges=True, include_controls=True, seed=42)
    encoder = MessageEncoder(framing=FramingMode.RAW)
    weak = NaiveBaselineJsonParser(FramingMode.RAW)
    hardened = HardenedBoundaryGate(oracle_key=cfg.oracle_key, framing=FramingMode.RAW)
    encoded = list(encoder.encode_stream(variants))
    weak_accepted_bad = [
        v.case_id
        for v, enc in zip(variants, encoded)
        if not v.expected_accepted and weak.parse_bytes(enc.raw_bytes).accepted
    ]
    hardened_accepted_bad = [
        v.case_id
        for v, enc in zip(variants, encoded)
        if not v.expected_accepted and hardened.parse_bytes(enc.raw_bytes).accepted
    ]
    assert len(weak_accepted_bad) > 0
    assert hardened_accepted_bad == []


# ---------------------------------------------------------------------------
# 2. Frontend Assets, HTTP REST & Real WebSocket Integration Tests
# ---------------------------------------------------------------------------


def test_frontend_static_assets_and_all_routes_wired(live_server: _LiveBackendHandle):
    resp_html = live_server.get("/")
    assert resp_html.status_code == 200
    html = resp_html.text

    for asset in (
        "/assets/styles.css",
        "/assets/api.js",
        "/assets/charts.js",
        "/assets/app.js",
        "/assets/pages_wire.js",
        "/assets/pages_eval.js",
        "/assets/pages_analysis.js",
    ):
        r = live_server.get(asset)
        assert r.status_code == 200, f"Asset {asset} returned {r.status_code}"
        assert len(r.text) > 100

    for bin_asset in ("/assets/hero_artwork.png", "/assets/work121_art.ico"):
        rb = live_server.get(bin_asset)
        assert rb.status_code == 200 and len(rb.content) > 10_000

    pages_eval_js = live_server.get("/assets/pages_eval.js").text
    assert "btn-onetap-launch" in pages_eval_js and "hero_artwork.png" in pages_eval_js

    nav_routes = re.findall(r'data-route="([a-z0-9-]+)"', html)
    assert len(nav_routes) == 10
    app_js = live_server.get("/assets/app.js").text
    for route in nav_routes:
        pattern = rf'(?:["\']?{re.escape(route)}["\']?\s*:)'
        assert re.search(pattern, app_js), f"Route {route} missing from app.js router"


def test_overview_and_agents_and_templates(live_server: _LiveBackendHandle):
    ov = live_server.get("/api/overview").json()
    assert ov["status"] == "ok"
    assert ov["native_binary"] is not None
    assert ov["counts"]["benign_templates"] >= 5
    assert ov["counts"]["indirect_templates"] >= 5
    assert ov["counts"]["wire_cases"] >= 50

    agents = live_server.get("/api/agents").json()
    keys = {a["agent_key"] for a in agents["agents"]}
    assert "policy-assistant@v1.0-baseline" in keys
    assert "policy-assistant@v2.0-hardened" in keys

    tpls = live_server.get("/api/templates/benign").json()["templates"]
    first_id = tpls[0]["template_id"]
    rendered = live_server.post("/api/templates/benign/render", json={"template_id": first_id}).json()
    assert rendered["template_id"] == first_id
    assert len(rendered["user_turns"]) >= 1


def test_wire_lab_api_parity_and_sign_observed(live_server: _LiveBackendHandle):
    catalog = live_server.get("/api/wire/catalog?framing=length_prefixed").json()
    assert catalog["total"] >= 50
    assert catalog["framing"] == "length_prefixed"

    suite = live_server.post(
        "/api/wire/suite/run",
        json={
            "framing": "length_prefixed",
            "include_benign_edges": True,
            "include_controls": True,
            "save_jsonl": True,
        },
    ).json()
    assert suite["native_parity"] is not None
    assert suite["native_parity"]["available"] is True
    assert suite["native_parity"]["parity_pct"] == 100.0
    assert suite["hardened_summary"]["passed_cases"] == suite["hardened_summary"]["total_cases"]
    assert suite["weak_baseline_summary"]["passed_cases"] < suite["hardened_summary"]["total_cases"]
    assert suite["saved_jsonl"] is not None

    signed = live_server.post(
        "/api/wire/sign-observed",
        json={"framing": "length_prefixed", "claim": "Measured value is 42", "answer": 42},
    ).json()
    assert signed["oracle_sig"]
    assert signed["frame_dissection"]["length_matches"] is True

    inspected = live_server.post(
        "/api/wire/inspect",
        json={"framing": "length_prefixed", "base64_bytes": signed["raw_b64"]},
    ).json()
    assert inspected["hardened_gate"]["accepted"] is True
    assert inspected["native_rust_gate"]["accepted"] is True
    assert inspected["native_rust_gate"]["error_code"] is None


def test_taxonomy_mcp_and_corpus_roundtrip(live_server: _LiveBackendHandle):
    tax = live_server.get("/api/taxonomy").json()
    assert len(tax["templates"]) >= 5
    tid = tax["templates"][0]["template_id"]

    tr = live_server.post(
        "/api/taxonomy/translate",
        json={"template_id": tid, "target_protocol": "anthropic_messages", "use_baseline": False},
    ).json()
    assert tr["target_protocol"] == "anthropic_messages"
    assert tr["wire_byte_length"] > 0

    hm = live_server.get("/api/taxonomy/handoff-matrix").json()
    assert len(hm["matrix"]) == 25

    exp_md = live_server.get("/api/taxonomy/export/markdown")
    assert exp_md.status_code == 200
    assert tid in exp_md.text

    tools = live_server.get("/api/mcp/tools?mode=perturbed").json()
    assert len(tools["tools"]) >= 3

    call_res = live_server.post(
        "/api/mcp/call",
        json={"tool_name": tools["tools"][0]["name"], "arguments": {}, "mode": "clean"},
    ).json()
    assert "tool_name" in call_res

    mm = live_server.post(
        "/api/mcp/multimodal",
        json={"title": "Test Chart", "labels": ["Alpha: 10ms", "Beta: 20ms"]},
    ).json()
    assert mm["byte_length"] > 100
    assert len(mm["image_b64"]) > 100

    save_res = live_server.post(
        "/api/corpus",
        json={"filename": "test_note.md", "content": "# Test\nBenign corpus document."},
    )
    assert save_res.status_code == 200
    docs = live_server.get("/api/corpus").json()["documents"]
    assert any(d["filename"] == "test_note.md" for d in docs)
    del_res = live_server.delete("/api/corpus/test_note.md")
    assert del_res.status_code == 200


def test_eval_websocket_stats_diff_reports_and_owasp(live_server: _LiveBackendHandle):
    start = live_server.post(
        "/api/eval/run",
        json={
            "agent_keys": ["policy-assistant@v1.0-baseline", "policy-assistant@v2.0-hardened"],
            "suite_mode": "benign",
            "scoring_engine": "heuristic",
        },
    ).json()
    assert start["status"] == "running"
    job_id = start["job_id"]

    done = live_server.drain_ws_job(job_id)
    assert done["status"] == "completed"

    runs_res = live_server.get("/api/runs").json()
    assert len(runs_res["runs"]) == 2
    run_a = runs_res["runs"][0]["run_id"]
    run_b = runs_res["runs"][1]["run_id"]

    detail = live_server.get(f"/api/runs/{run_a}").json()
    assert len(detail["records"]) >= 5

    stats = live_server.post("/api/runs/stats", json={"run_a_id": run_a, "run_b_id": run_b}).json()
    assert "fisher" in stats and "mcnemar" in stats and "per_case_matrix" in stats

    diff = live_server.post("/api/runs/diff", json={"baseline_run_id": run_a, "candidate_run_id": run_b}).json()
    assert "compared_cases" in diff and "case_diffs" in diff and diff["compared_cases"] >= 5

    tmatrix = live_server.get("/api/runs/template-matrix").json()
    assert "matrix" in tmatrix

    rep_md = live_server.get("/api/reports/export/markdown")
    assert rep_md.status_code == 200
    assert "policy-assistant@v2.0-hardened" in rep_md.text
    for fmt in ("latex", "runs_csv", "templates_csv", "manifest"):
        r_exp = live_server.get(f"/api/reports/export/{fmt}")
        assert r_exp.status_code == 200 and len(r_exp.text) > 10

    ow_cat = live_server.get("/api/owasp/catalog").json()
    assert len(ow_cat["catalog"]) >= 3
    ow_audit = live_server.get("/api/owasp/audit").json()
    assert "remediation" in ow_audit
    ow_val = live_server.post(
        "/api/owasp/validate",
        json={
            "untrusted_text": "Plain telemetry summary: CPU 42%.",
            "agent_message": '{"msg_id":"m1","sender":"a","recipient":"b","provenance":"derived","claim":"ok"}',
        },
    ).json()
    assert "untrusted_isolation" in ow_val and "inter_agent_envelope" in ow_val


def test_relay_and_hamster_wheel_end_to_end(live_server: _LiveBackendHandle):
    scenarios = live_server.get("/api/relay/scenarios").json()["scenarios"]
    assert len(scenarios) >= 1
    sid = scenarios[0]["scenario_id"]

    relay_res = live_server.post(
        "/api/relay/run",
        json={
            "scenario_id": sid,
            "agent_a_key": "policy-assistant@v1.0-baseline",
            "agent_b_key": "policy-assistant@v2.0-hardened",
            "source_protocol": "openai_chat_completions",
            "target_protocol": "huggingface_chatml",
            "ablation": True,
        },
    ).json()
    assert len(relay_res["results"]) == 2
    traces = live_server.get("/api/relay/traces").json()["traces"]
    assert len(traces) >= 2

    wheel_start = live_server.post(
        "/api/wheel/start",
        json={
            "agent_key": "policy-assistant@v2.0-hardened",
            "include_indirect_paired": True,
            "include_benign_suite": True,
            "include_relay_ablation": False,
            "cycles": 1,
        },
    ).json()
    wheel_done = live_server.drain_ws_job(wheel_start["job_id"])
    assert wheel_done["status"] == "completed"

    reports = live_server.get("/api/wheel/reports").json()["reports"]
    assert len(reports) >= 1
    assert reports[0]["agent_key"] == "policy-assistant@v2.0-hardened"


def test_one_tap_launcher_desktop_and_home_screen_shortcuts() -> None:
    ps1_path = Path(__file__).resolve().parents[1] / "one_tap.ps1"
    assert ps1_path.is_file(), f"one_tap.ps1 missing at {ps1_path}"

    if sys.platform != "win32":
        pytest.skip("one_tap.ps1 desktop and start menu shortcut installation requires Windows")

    # Verify PowerShell script can install to Desktop and Home Screen without error
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(ps1_path),
        "-InstallHomeShortcut",
        "-NoBrowser",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "Installed shortcut:" in res.stdout

    # Verify shortcut exists in Desktop and Start Menu Programs (Home screen)
    appdata = Path.home() / "AppData" / "Roaming"
    start_menu_programs = appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    desktop = Path.home() / "Desktop"
    onedrive_desktop = Path.home() / "OneDrive" / "Desktop"

    for name in ("work121 One-Tap Launch.lnk", "work121 Workbench.lnk"):
        home_shortcut = start_menu_programs / name
        assert home_shortcut.is_file(), f"Home screen / Start Menu shortcut missing at {home_shortcut}"
        assert home_shortcut.stat().st_size > 0
        assert (
            desktop / name
        ).is_file() or (
            onedrive_desktop / name
        ).is_file(), f"Desktop shortcut {name} must exist in Desktop or OneDrive Desktop"

    ico_path = Path(__file__).resolve().parents[1] / "web" / "work121_art.ico"
    assert ico_path.is_file() and ico_path.stat().st_size > 1000


def test_python_and_native_signed_observed_without_sig_enforcement_and_32char_token():
    binary = _find_native_binary()
    assert binary is not None
    cfg = PayloadConfig(framing=FramingMode.RAW, seed=42)
    gate = HardenedBoundaryGate(
        oracle_key=cfg.oracle_key,
        framing=FramingMode.RAW,
        require_signed_observed=False,
        reject_special_tokens=True,
    )

    msg = AgentMessage(
        msg_id="m-obs-signed-opt",
        sender="oracle",
        recipient="worker",
        round_idx=1,
        performative=Performative.PROPOSE,
        claim="Verified deterministic output is 42",
        answer=42,
        confidence=1.0,
        provenance=Provenance.OBSERVED,
        constraints=["integer_only"],
        evidence_refs=["oracle:unit"],
    )
    msg.oracle_sig = sign_observed_claim(msg, cfg.oracle_key)
    raw_signed = json.dumps(msg.to_dict()).encode("utf-8")

    py_signed = gate.parse_bytes(raw_signed)
    assert py_signed.accepted is True
    assert py_signed.error_code is None

    proc_signed = subprocess.run(
        [
            str(binary),
            "wire-check",
            "--framing",
            "raw",
            "--require-signed-observed",
            "false",
            "--reject-special-tokens",
            "true",
            "--oracle-key",
            cfg.oracle_key.decode("utf-8"),
            "--stdin",
            "--json",
        ],
        input=raw_signed,
        capture_output=True,
        timeout=5.0,
        check=False,
    )
    native_signed = json.loads(proc_signed.stdout.decode("utf-8"))
    assert native_signed["accepted"] is True
    assert native_signed["error_code"] is None

    tok_32 = "<|" + ("x" * 28) + "|>"
    assert len(tok_32) == 32
    msg_tok = AgentMessage(
        msg_id="m-tok-32",
        sender="attacker",
        recipient="worker",
        round_idx=1,
        performative=Performative.PROPOSE,
        claim=f"Injected {tok_32} boundary token",
        answer=42,
        confidence=0.9,
        provenance=Provenance.DERIVED,
        constraints=[],
        evidence_refs=[],
    )
    raw_tok = json.dumps(msg_tok.to_dict()).encode("utf-8")

    py_tok = gate.parse_bytes(raw_tok)
    assert py_tok.accepted is False
    assert py_tok.error_code == "special_token_delimiter"

    proc_tok = subprocess.run(
        [
            str(binary),
            "wire-check",
            "--framing",
            "raw",
            "--require-signed-observed",
            "false",
            "--reject-special-tokens",
            "true",
            "--stdin",
            "--json",
        ],
        input=raw_tok,
        capture_output=True,
        timeout=5.0,
        check=False,
    )
    native_tok = json.loads(proc_tok.stdout.decode("utf-8"))
    assert native_tok["accepted"] is False
    assert native_tok["error_code"] == "special_token_delimiter"


def test_backend_favicon_validation_and_unique_eval_runs(live_server: _LiveBackendHandle):
    fav = live_server.get("/favicon.ico")
    assert fav.status_code == 200
    assert len(fav.content) > 0

    bad_json = live_server.post(
        "/api/eval/run",
        data="{not-json",
        headers={"Content-Type": "application/json"},
    )
    assert bad_json.status_code == 400
    assert "error" in bad_json.json()

    non_obj = live_server.post("/api/wire/inspect", json=["not", "an", "object"])
    assert non_obj.status_code == 400
    assert "error" in non_obj.json()

    unknown_agent = live_server.post(
        "/api/eval/run",
        json={"agent_keys": ["missing-agent@v999"], "suite_mode": "benign"},
    )
    assert unknown_agent.status_code == 404
    assert "error" in unknown_agent.json()

    partial_diff = live_server.post(
        "/api/runs/diff",
        json={"baseline_jsonl": "", "candidate_jsonl": ""},
    )
    assert partial_diff.status_code == 400
    assert "error" in partial_diff.json()

    bad_hex = live_server.post(
        "/api/wire/inspect",
        json={
            "framing": "raw",
            "raw_text": "hello",
            "mutation": {
                "kind": "replace_bytes",
                "params": {"data": "zz-invalid-hex", "data_encoding": "hex"},
            },
        },
    )
    assert bad_hex.status_code == 400
    assert "error" in bad_hex.json()

    bad_limit = live_server.post(
        "/api/wire/inspect",
        json={"framing": "raw", "raw_text": "{}", "max_wire_bytes": 0},
    )
    assert bad_limit.status_code == 400

    # Two rapid eval runs for the same agent must not collide or overwrite each other
    r1 = live_server.post(
        "/api/eval/run",
        json={"agent_keys": ["policy-assistant@v2.0-hardened"], "suite_mode": "benign"},
    ).json()
    d1 = live_server.drain_ws_job(r1["job_id"])
    assert d1["status"] == "completed"

    r2 = live_server.post(
        "/api/eval/run",
        json={"agent_keys": ["policy-assistant@v2.0-hardened"], "suite_mode": "benign"},
    ).json()
    d2 = live_server.drain_ws_job(r2["job_id"])
    assert d2["status"] == "completed"

    runs_list = live_server.get("/api/runs").json()["runs"]
    assert len(runs_list) >= 2
    run_ids = [r["run_id"] for r in runs_list]
    assert len(set(run_ids)) == len(run_ids)

