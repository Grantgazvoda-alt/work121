"""Run paired baseline and perturbed cases against a live model endpoint.

Writes one JSONL artifact through ``JsonlRunStore``. Provider failures are
recorded as indeterminate classifications; they are not dropped.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from eval_runner import execute_evaluation_run
from live_providers import (
    PROVIDER_CHOICES,
    LiveHttpAgent,
    LiveProviderError,
    discover_live_agents,
    resolve_spec,
)
from regression_tracker import JsonlRunStore
from template_library import BenignTemplateLibrary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate paired benign templates against a live model HTTP endpoint.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print configured providers as JSON lines and exit.",
    )
    parser.add_argument("--provider", choices=PROVIDER_CHOICES)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--output", type=Path, default=Path("eval_runs/live.jsonl"))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--template-id",
        action="append",
        dest="template_ids",
        default=None,
        help="Limit the run to one template id. Repeat to select several.",
    )
    parser.add_argument("--run-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        for spec in discover_live_agents():
            print(json.dumps(spec.to_dict(), sort_keys=True))
        return 0
    if not args.provider:
        parser.error("--provider is required unless --list is set")
    try:
        spec = resolve_spec(args.provider, model=args.model, base_url=args.base_url)
        agent = LiveHttpAgent(spec, timeout=args.timeout)
    except LiveProviderError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    library = BenignTemplateLibrary()
    if args.template_ids:
        cases = [library.get(template_id).render() for template_id in args.template_ids]
    else:
        cases = [template.render() for template in library.list_templates()]
    run_id = args.run_id or f"{agent.model_id}-{agent.model_version}-live"
    result = execute_evaluation_run(
        run_id=run_id,
        agent=agent,
        test_cases=cases,
        library=library,
        judge_id="boundary-judge",
        judge_version="1.0.0",
    )
    written = JsonlRunStore().write_records(args.output, result.records, append=False)
    counts = Counter(record.outcome.value for record in result.records)
    summary = " ".join(f"{name}={counts[name]}" for name in sorted(counts))
    print(f"wrote {written} records to {args.output}")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
