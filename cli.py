from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Optional, Sequence

from eval_runner import AgentRegistry, execute_evaluation_run
from owasp_defensive_auditor import OwaspDefensiveCatalog
from regression_tracker import (
    ChangeStatus,
    JsonlRunStore,
    RegressionComparator,
)
from response_classifier import (
    ClassificationResult,
    PatternCategory,
    ResponsePatternClassifier,
    SusceptibilityMapper,
)
from semantic_judge import PolicyAnchorSemanticJudge
from template_library import BenignTemplateLibrary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="work121",
        description="Defensive LLM response-pattern evaluation, OWASP auditing, and CI regression gating.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser(
        "run",
        help="Execute benign edge-case validation templates against registered target agents.",
    )
    p_run.add_argument(
        "--agent",
        action="append",
        dest="agents",
        help="Target agent key (e.g. policy-assistant@v2.0-hardened). Repeatable; defaults to all.",
    )
    p_run.add_argument(
        "--category",
        choices=[c.value for c in PatternCategory],
        default=None,
        help="Optional filter for a single validation category.",
    )
    p_run.add_argument(
        "--out-dir",
        type=Path,
        default=Path("eval_runs"),
        help="Directory to write JSONL run artifacts.",
    )
    p_run.add_argument(
        "--use-anchor-judge",
        action="store_true",
        help="Use PolicyAnchorSemanticJudge instead of heuristic-only classification.",
    )
    p_run.add_argument(
        "--redact",
        action="store_true",
        help="Redact raw templates/payloads/responses while keeping SHA-256 hashes.",
    )

    p_diff = sub.add_parser(
        "diff",
        help="Compare baseline and candidate JSONL runs and enforce CI regression gates.",
    )
    p_diff.add_argument("--baseline", type=Path, required=True, help="Baseline run JSONL path.")
    p_diff.add_argument("--candidate", type=Path, required=True, help="Candidate run JSONL path.")
    p_diff.add_argument(
        "--drift-tol",
        type=float,
        default=-0.05,
        help="Negative drift-slope delta threshold flagged as a regression (default: -0.05).",
    )
    p_diff.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional file path to write the full JSON diff report.",
    )
    p_diff.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with code 2 if any test case regressed.",
    )
    p_diff.add_argument(
        "--fail-on-template-drift",
        action="store_true",
        help="Exit with code 3 if any prompt template or input SHA-256 changed.",
    )

    p_audit = sub.add_parser(
        "audit-owasp",
        help="Generate an OWASP LLM Top 10 defensive remediation report from a JSONL run.",
    )
    p_audit.add_argument("--run-file", type=Path, required=True, help="Serialized JSONL run path.")
    p_audit.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional output path for the JSON remediation report.",
    )

    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    library = BenignTemplateLibrary()
    registry = AgentRegistry()
    store = JsonlRunStore()

    cat_filter = PatternCategory(args.category) if args.category else None
    templates = library.list_templates(category=cat_filter)
    cases = [tpl.render(test_case_id=f"default-{tpl.template_id}") for tpl in templates]

    agent_keys = args.agents or registry.list_keys()
    judge = PolicyAnchorSemanticJudge() if args.use_anchor_judge else None
    clf = ResponsePatternClassifier(semantic_judge=judge)
    judge_id = "policy-anchor-judge" if judge else "heuristic-baseline"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    written_files: list[str] = []

    for key in agent_keys:
        agent = registry.get(key)
        run_id = f"{agent.model_id}-{agent.model_version}-{timestamp}"
        result = execute_evaluation_run(
            run_id=run_id,
            agent=agent,
            test_cases=cases,
            classifier=clf,
            judge_id=judge_id,
            judge_version="1.0.0",
        )
        out_path = args.out_dir / f"{run_id}.jsonl"
        store.write_records(
            out_path,
            result.records,
            append=False,
            redact_raw_content=args.redact,
        )
        written_files.append(str(out_path))

    print(json.dumps({"cases_per_agent": len(cases), "artifacts": written_files}, indent=2))
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    store = JsonlRunStore()
    baseline = store.load_records(args.baseline)
    candidate = store.load_records(args.candidate)

    report = RegressionComparator(drift_regression_tolerance=args.drift_tol).compare(
        baseline, candidate
    )
    payload = report.to_dict()

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    regressions = sum(1 for d in report.case_diffs if d.status == ChangeStatus.REGRESSION)
    improvements = sum(1 for d in report.case_diffs if d.status == ChangeStatus.IMPROVEMENT)
    summary = {
        "baseline": payload["baseline"],
        "candidate": payload["candidate"],
        "compared_cases": report.compared_cases,
        "regressions": regressions,
        "improvements": improvements,
        "template_or_input_drift": len(report.mismatched_template_ids),
    }
    print(json.dumps(summary, indent=2))

    if args.fail_on_template_drift and report.mismatched_template_ids:
        return 3
    if args.fail_on_regression and regressions > 0:
        return 2
    return 0


def _cmd_audit_owasp(args: argparse.Namespace) -> int:
    store = JsonlRunStore()
    records = store.load_records(args.run_file)
    classifications = [
        ClassificationResult(
            transcript_id=r.test_case_id,
            model_id=r.model_id,
            architecture_family=r.architecture_family,
            category=r.category,
            outcome=r.outcome,
            confidence=r.confidence,
            signals=r.signals,
            drift_slope=r.drift_slope,
        )
        for r in records
    ]
    profiles = SusceptibilityMapper().build_map(classifications)
    recommendations = OwaspDefensiveCatalog.generate_remediation_plan(profiles)

    output = [
        {
            "model_id": rec.model_id,
            "architecture_family": rec.architecture_family,
            "pattern_category": rec.pattern_category.value,
            "owasp_category": rec.owasp_category.value,
            "risk_tier": rec.risk_tier,
            "failure_rate": rec.failure_rate,
            "erosion_rate": rec.erosion_rate,
            "mechanism_summary": rec.mechanism_summary,
            "recommended_controls": list(rec.recommended_controls),
        }
        for rec in recommendations
    ]

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps(output, indent=2))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "diff":
        return _cmd_diff(args)
    if args.command == "audit-owasp":
        return _cmd_audit_owasp(args)
    parser.error(f"Unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
