from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Optional

from eval_runner import AgentRegistry, TargetAgent, execute_evaluation_run
from indirect_input_classifier import (
    IndirectInputClassifier,
    ProtocolFormat,
    build_default_template_library,
)
from live_adapters import (
    FREE_GEMINI_MODELS,
    PRIORITY_GEMINI_MODELS,
    EmbeddingAlignmentScorer,
    HybridSemanticJudge,
    LiveHttpAgent,
    LiveLlmSemanticJudge,
    LiveProviderConfig,
    ProviderKind,
    list_ollama_models,
    resolve_api_key,
)
from live_providers import (
    LiveHttpAgent as ProviderHttpAgent,
    PROVIDER_CHOICES as LIVE_PROVIDER_CHOICES,
    discover_live_agents as discover_provider_specs,
    resolve_spec as resolve_provider_spec,
)
from regression_tracker import JsonlRunStore, RegressionComparator
from report_generator import EmpiricalReportGenerator
from response_classifier import ResponsePatternClassifier, SusceptibilityMapper
from template_library import BenignTemplateLibrary
from tool_runtime import HttpMcpClient


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_live_eval",
        description="Run live cross-architecture indirect-input evaluations and export academic artifacts.",
    )
    provider_choices = list(
        dict.fromkeys(
            [
                "ollama",
                "openai_compatible",
                "anthropic",
                "gemini",
                "vertex_express",
                "simulated",
                *LIVE_PROVIDER_CHOICES,
            ]
        )
    )
    parser.add_argument(
        "--provider",
        choices=provider_choices,
        default="ollama",
        help="Target model provider (default: ollama).",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model ID (e.g., qwen2.5-coder:7b, llama3.1:latest, gpt-4o-mini, claude-3-5-sonnet-latest, gemini-4-hydrogen, gemini-4-neon).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional list of multiple model IDs to benchmark sequentially with automatic pairwise Fisher/McNemar significance testing.",
    )
    parser.add_argument(
        "--model-version",
        default="live",
        help="Model version label recorded in JSONL artifacts (default: live).",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="Optional custom API base URL (e.g., http://127.0.0.1:11434 or vLLM/LiteLLM endpoint).",
    )
    parser.add_argument(
        "--protocol",
        choices=[p.value for p in ProtocolFormat],
        default=None,
        help="Optional target wire protocol to translate all templates into before execution.",
    )
    parser.add_argument(
        "--all-protocols",
        action="store_true",
        help="Sweep all 5 wire protocols (OpenAI, Anthropic, Gemini, ChatML, Encoder-Decoder Flat) to measure translation boundary degradation.",
    )
    parser.add_argument(
        "--suite",
        choices=["indirect", "benign", "all"],
        default="indirect",
        help="Template suite to run: indirect (9 cross-protocol templates), benign (6 single/multi-turn templates), or all.",
    )
    parser.add_argument(
        "--templates",
        nargs="*",
        default=None,
        help="Optional subset of template IDs to execute.",
    )
    parser.add_argument(
        "--template-id",
        default="",
        help="Single BenignTemplateLibrary template ID to run with paired baseline control.",
    )
    parser.add_argument(
        "--paired",
        action="store_true",
        help="Include clean paired baseline controls to compute paired_regression_rate.",
    )
    parser.add_argument(
        "--no-invariant-judge",
        action="store_true",
        help="Use refusal/drift heuristics instead of IndirectInvariantJudge.",
    )
    parser.add_argument(
        "--judge-mode",
        choices=["hybrid", "policy_anchor", "heuristic"],
        default="hybrid",
        help="Judge backend: hybrid (invariants + optional LLM + PolicyAnchor), policy_anchor (offline deterministic), or heuristic.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional explicit JSONL output file path.",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_runs",
        help="Directory for JSONL runs and summary reports (default: eval_runs).",
    )
    parser.add_argument(
        "--compare-with",
        default="",
        help="Optional path to a baseline .jsonl file for automatic RegressionComparator diffing.",
    )
    parser.add_argument(
        "--export-docs",
        default="",
        help="Export academic Markdown, LaTeX (.tex), and JSON taxonomy artifacts to the given directory and exit.",
    )
    parser.add_argument(
        "--generate-report",
        default="",
        help="Generate live empirical Markdown + LaTeX paper report from --output-dir into the given directory and exit.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered live provider specifications as JSON and exit.",
    )
    parser.add_argument(
        "--list-ollama",
        action="store_true",
        help="List locally available Ollama models and exit.",
    )
    parser.add_argument(
        "--mcp-url",
        default="",
        help="Optional live MCP HTTP JSON-RPC URL (e.g., http://127.0.0.1:8775/mcp) to run real multi-round tool calls on tool templates.",
    )
    parser.add_argument(
        "--embed-model",
        default="",
        help="Optional Ollama embedding model (e.g., nomic-embed-text:latest) for real dense-embedding persona_alignment_score & drift_slope.",
    )
    parser.add_argument(
        "--judge-model",
        default="",
        help="Optional live Ollama judge model (e.g., llama3.1:latest or qwen2.5-coder:7b) for HybridSemanticJudge on open-ended/multi-turn cases.",
    )
    parser.add_argument(
        "--gemini-priority",
        action="store_true",
        help="Benchmark Gemini 4 priority models sequentially (gemini-4-carbon, gemini-4-neon, gemini-4-argon).",
    )
    parser.add_argument(
        "--gemini-free",
        action="store_true",
        help="Benchmark $0 Gemini Free Tier models sequentially (gemini-4-hydrogen, gemini-3.8-flash, gemini-3.6-flash).",
    )
    parser.add_argument(
        "--long-horizon",
        action="store_true",
        help="Focus benign/multi-turn suite on long-horizon Crescendo & Many-Shot context templates.",
    )
    parser.add_argument(
        "--thinking-level",
        choices=["low", "medium", "high", "xhigh"],
        default=None,
        help="Optional Gemini 4 thinkingConfig level (low, medium, high, xhigh).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request HTTP timeout in seconds (default: 120).",
    )
    return parser


def _resolve_agent(args: argparse.Namespace) -> TargetAgent:
    if args.provider == "simulated":
        registry = AgentRegistry()
        key = args.model or "gateway-relay@v1.0-flat-adapter"
        return registry.get(key)

    provider = ProviderKind(args.provider)
    model_id = args.model.strip()
    if not model_id:
        if provider == ProviderKind.OLLAMA:
            installed = list_ollama_models(args.base_url or "http://127.0.0.1:11434")
            if not installed:
                raise SystemExit("No local Ollama models found. Start Ollama or specify --model.")
            model_id = "qwen2.5-coder:7b" if "qwen2.5-coder:7b" in installed else installed[0]
        else:
            raise SystemExit(f"--model is required when --provider={provider.value}.")

    mcp_backend = HttpMcpClient(args.mcp_url) if args.mcp_url.strip() else None
    clean_mcp_backend = None
    if args.mcp_url.strip():
        clean_url = (
            args.mcp_url.rstrip("/") + "/clean"
            if args.mcp_url.rstrip("/").endswith("/mcp")
            else args.mcp_url
        )
        clean_mcp_backend = HttpMcpClient(clean_url)

    scorer = (
        EmbeddingAlignmentScorer(model_id=args.embed_model.strip())
        if args.embed_model.strip()
        else None
    )

    agent = LiveHttpAgent(
        LiveProviderConfig(
            provider=provider,
            model_id=model_id,
            model_version=args.model_version,
            base_url=args.base_url,
            timeout_seconds=args.timeout,
            thinking_level=getattr(args, "thinking_level", None),
        ),
        mcp_backend=mcp_backend,
        clean_mcp_backend=clean_mcp_backend,
        alignment_scorer=scorer,
    )
    ok, msg = agent.health_check()
    if not ok:
        raise SystemExit(f"Health check failed for {provider.value}/{model_id}: {msg}")
    print(f"[health] {msg}")
    return agent


def _run_benign_provider_cli(args: argparse.Namespace) -> int:
    benign_lib = BenignTemplateLibrary()
    if args.template_id:
        cases = [benign_lib.get(args.template_id).render()]
    elif args.templates:
        cases = [benign_lib.get(tid).render() for tid in args.templates]
    else:
        cases = [tpl.render() for tpl in benign_lib.list_templates()]

    if args.provider == "simulated":
        agent: TargetAgent = _resolve_agent(args)
    else:
        spec = resolve_provider_spec(
            args.provider,
            model=args.model.strip() or None,
            base_url=args.base_url.strip() or None,
        )
        agent = ProviderHttpAgent(spec, timeout=args.timeout)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = agent.model_id.replace("/", "_").replace(":", "_")
    run_id = f"{safe_model}-{agent.model_version}-{timestamp}"
    result = execute_evaluation_run(
        run_id=run_id,
        agent=agent,
        test_cases=cases,
        library=benign_lib,
    )
    out_path = Path(args.output) if args.output else Path(args.output_dir) / f"{run_id}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    JsonlRunStore().write_records(out_path, result.records, append=False)
    print(f"[done] Saved {len(result.records)} JSONL records to {out_path}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list:
        specs = discover_provider_specs(timeout=min(float(args.timeout), 1.0))
        print(json.dumps([spec.to_dict() for spec in specs], indent=2))
        return 0

    if args.list_ollama:
        models = list_ollama_models(args.base_url or "http://127.0.0.1:11434")
        if not models:
            print("No local Ollama models detected.")
            return 1
        for name in models:
            print(name)
        return 0

    if args.template_id or args.output or args.provider not in {
        "ollama",
        "openai_compatible",
        "anthropic",
        "gemini",
        "vertex_express",
        "simulated",
    }:
        return _run_benign_provider_cli(args)

    library: IndirectInputClassifier = build_default_template_library()

    if args.export_docs:
        out_dir = Path(args.export_docs)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "indirect_input_taxonomy.md").write_text(
            library.export_academic_markdown(), encoding="utf-8"
        )
        (out_dir / "indirect_input_taxonomy.tex").write_text(
            library.export_latex_table(), encoding="utf-8"
        )
        (out_dir / "indirect_input_templates.json").write_text(
            library.export_json(), encoding="utf-8"
        )
        print(f"Exported Markdown, LaTeX, and JSON documentation to {out_dir.resolve()}")
        return 0

    if args.generate_report:
        runs_dir = Path(args.output_dir)
        rep_dir = Path(args.generate_report)
        bundle = EmpiricalReportGenerator.export_research_bundle(runs_dir, rep_dir)
        print(
            "Generated reproducible research bundle: "
            + ", ".join(str(p.resolve()) for p in bundle.values())
        )
        return 0

    benign_lib = BenignTemplateLibrary()
    protocols_to_run: list[Optional[ProtocolFormat]] = (
        list(ProtocolFormat)
        if args.all_protocols
        else [ProtocolFormat(args.protocol) if args.protocol else None]
    )
    if args.gemini_priority or args.gemini_free:
        if args.provider == "ollama" and not args.model.strip():
            args.provider = "gemini"
        models_to_run = []
        if args.gemini_priority:
            models_to_run.extend(PRIORITY_GEMINI_MODELS)
        if args.gemini_free:
            models_to_run.extend(FREE_GEMINI_MODELS)
        models_to_run = list(dict.fromkeys(models_to_run))
    else:
        models_to_run = list(args.models) if args.models else [args.model]

    long_horizon_ids = (
        "long_horizon_crescendo_drift",
        "long_horizon_many_shot_context",
        "drift_tone_shift_multiturn",
        "drift_scope_creep_multiturn",
    )
    effective_suite = "all" if (args.long_horizon and args.suite == "indirect") else args.suite

    def _build_cases(target_protocol: Optional[ProtocolFormat]) -> list:
        built = []
        if effective_suite in ("indirect", "all"):
            if args.templates:
                valid_ids = {t.template_id for t in library.list_templates()}
                ind_tpls = [library.get(tid) for tid in args.templates if tid in valid_ids]
            else:
                ind_tpls = library.list_templates()
            for tpl in ind_tpls:
                if args.paired:
                    base_c, ind_c = tpl.to_paired_test_cases(protocol=target_protocol)
                    built.extend([base_c, ind_c])
                else:
                    built.append(
                        tpl.to_rendered_test_case(
                            test_case_id=f"indirect-{tpl.template_id}",
                            protocol=target_protocol,
                        )
                    )
        if effective_suite in ("benign", "all"):
            if args.long_horizon and not args.templates:
                ben_tpls = [benign_lib.get(tid) for tid in long_horizon_ids]
            elif args.templates:
                valid_ben = {t.template_id for t in benign_lib.list_templates()}
                ben_tpls = [benign_lib.get(tid) for tid in args.templates if tid in valid_ben]
            else:
                ben_tpls = benign_lib.list_templates()
            for tpl in ben_tpls:
                built.append(tpl.render(test_case_id=f"benign-{tpl.template_id}"))
        return built

    if args.no_invariant_judge or args.judge_mode == "heuristic":
        semantic_judge = None
        judge_id = "heuristic-baseline"
    elif args.judge_mode == "policy_anchor":
        from semantic_judge import PolicyAnchorSemanticJudge

        semantic_judge = PolicyAnchorSemanticJudge()
        judge_id = "policy-anchor-judge"
    elif args.provider != "simulated":
        llm_judge = None
        if args.judge_model.strip():
            judge_provider = ProviderKind(args.provider)
            judge_agent = LiveHttpAgent(
                LiveProviderConfig(
                    provider=judge_provider,
                    model_id=args.judge_model.strip(),
                    model_version="live-judge",
                    base_url=args.base_url or "",
                    api_key=resolve_api_key(judge_provider),
                    temperature=0.0,
                    max_tokens=256,
                    timeout_seconds=args.timeout,
                )
            )
            llm_judge = LiveLlmSemanticJudge(judge_agent)
        semantic_judge = HybridSemanticJudge(llm_judge=llm_judge)
        judge_id = (
            f"hybrid-judge:{args.judge_model.strip()}"
            if llm_judge is not None
            else "hybrid-invariant-anchor-judge"
        )
    else:
        semantic_judge = None
        judge_id = "heuristic-baseline"

    classifier = ResponsePatternClassifier(semantic_judge=semantic_judge)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = JsonlRunStore()

    from statistical_analysis import compare_runs_statistically

    completed_runs: list[tuple[str, list]] = []

    for model_choice in models_to_run:
        for proto in protocols_to_run:
            cases = _build_cases(proto)
            if not cases:
                raise SystemExit("No matching test templates selected.")

            run_args = argparse.Namespace(**vars(args))
            run_args.model = model_choice
            if args.all_protocols and proto is not None:
                run_args.model_version = f"{args.model_version}-proto-{proto.value}"
            agent = _resolve_agent(run_args)

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            safe_model = agent.model_id.replace("/", "_").replace(":", "_")
            proto_tag = (
                f"-{proto.value}"
                if (proto is not None and proto.value not in agent.model_version)
                else ""
            )
            run_id = f"{safe_model}-{agent.model_version}{proto_tag}-{timestamp}"

            print(
                f"[run] Starting '{run_id}' | cases={len(cases)} | paired={args.paired} | "
                f"protocol={proto.value if proto else 'native'} | judge={judge_id}"
            )

            def _on_case(idx: int, total: int, record) -> None:
                sigs = ", ".join(record.signals[:3])
                print(
                    f"  [{idx:02d}/{total:02d}] {record.test_case_id:<36} -> "
                    f"{record.outcome.value:<17} (conf={record.confidence:.2f}, tok={record.response_tokens.response_token_count}) [{sigs}]"
                )

            result = execute_evaluation_run(
                run_id=run_id,
                agent=agent,
                test_cases=cases,
                classifier=classifier,
                judge_id=judge_id,
                on_case_complete=_on_case,
            )

            jsonl_path = out_dir / f"{run_id}.jsonl"
            store.write_records(jsonl_path, result.records, append=False)
            completed_runs.append((run_id, list(result.records)))

            profiles = SusceptibilityMapper().build_map(result.classifications)
            summary = {
                "run_id": run_id,
                "agent_key": result.agent_key,
                "protocol": proto.value if proto else "native",
                "jsonl_path": str(jsonl_path),
                "total_cases": len(result.records),
                "profiles": [
                    {
                        "model_id": p.model_id,
                        "architecture_family": p.architecture_family,
                        "category": p.category.value,
                        "sample_count": p.sample_count,
                        "failure_rate": p.failure_rate,
                        "erosion_rate": p.erosion_rate,
                        "paired_regression_rate": p.paired_regression_rate,
                        "risk_tier": p.risk_tier,
                        "top_signals": list(p.top_signals),
                    }
                    for p in profiles
                ],
            }

            if args.compare_with:
                baseline_records = store.load_records(Path(args.compare_with))
                diff_report = RegressionComparator().compare(baseline_records, result.records)
                stats_report = compare_runs_statistically(baseline_records, result.records)
                summary["regression_diff"] = diff_report.to_dict()
                summary["statistical_significance"] = stats_report.to_dict()
                diff_path = out_dir / f"{run_id}_diff.json"
                diff_path.write_text(json.dumps(diff_report.to_dict(), indent=2), encoding="utf-8")
                print(
                    f"[diff] Saved regression diff to {diff_path} | "
                    f"Fisher p={stats_report.fisher.p_value:.4f}, McNemar p={stats_report.mcnemar.p_value:.4f}"
                )

            summary_path = out_dir / f"{run_id}_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

            print(f"\n[done] Saved JSONL records to {jsonl_path}")
            print(f"[done] Saved summary report to {summary_path}")
            for p in profiles:
                print(
                    f"  category={p.category.value:<28} n={p.sample_count} "
                    f"fail={p.failure_rate:.2f} erosion={p.erosion_rate:.2f} "
                    f"paired_reg={p.paired_regression_rate}"
                )

    if len(completed_runs) >= 2:
        pairwise_stats = []
        for i in range(len(completed_runs)):
            for j in range(i + 1, len(completed_runs)):
                stat_rep = compare_runs_statistically(completed_runs[i][1], completed_runs[j][1])
                pairwise_stats.append(stat_rep.to_dict())
                print(
                    f"[stats] {stat_rep.run_a_label} vs {stat_rep.run_b_label} | "
                    f"fail_a={stat_rep.fisher.rate_a:.2f} fail_b={stat_rep.fisher.rate_b:.2f} | "
                    f"Fisher p={stat_rep.fisher.p_value:.4f} | McNemar p={stat_rep.mcnemar.p_value:.4f}"
                )
        batch_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stats_path = out_dir / f"batch_significance_{batch_ts}.json"
        stats_path.write_text(json.dumps(pairwise_stats, indent=2), encoding="utf-8")
        print(f"[stats] Saved pairwise significance matrix to {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
