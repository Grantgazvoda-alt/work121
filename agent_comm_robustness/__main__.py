from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .benign_templates import BenignEdgeCategory, BenignEdgeGenerator, BenignTemplateConfig
from .encoder import MessageEncoder
from .payloads import PayloadCategory, PayloadConfig, PayloadGenerator
from .protocol import FramingMode, HardenedBoundaryGate
from .runner import SuiteRunner, format_report


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent-comm-robustness",
        description="Run the robustness validation suite against the agent wire parser.",
    )
    p.add_argument(
        "--framing",
        choices=[m.value for m in FramingMode],
        default=FramingMode.RAW.value,
        help="Wire framing mode (default: raw).",
    )
    p.add_argument(
        "--categories",
        nargs="+",
        choices=[c.value for c in PayloadCategory],
        default=None,
        help="Optional subset of payload categories to execute.",
    )
    p.add_argument(
        "--include-benign-edges",
        action="store_true",
        help="Include the parameterized benign edge-case template library.",
    )
    p.add_argument(
        "--benign-only",
        action="store_true",
        help="Run only the benign edge-case template library (tests false rejections).",
    )
    p.add_argument(
        "--benign-categories",
        nargs="+",
        choices=[c.value for c in BenignEdgeCategory],
        default=None,
        help="Optional subset of benign edge-case categories.",
    )
    p.add_argument("--max-wire-bytes", type=int, default=32_768)
    p.add_argument("--max-claim-chars", type=int, default=8_192)
    p.add_argument("--max-nesting-depth", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-controls",
        action="store_true",
        help="Skip positive boundary control messages.",
    )
    p.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Path to write per-message JSONL parser response logs.",
    )
    p.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Path to write the aggregate JSON summary report.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Stream per-case parser responses to stderr.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    framing = FramingMode(args.framing)
    categories = tuple(PayloadCategory(c) for c in args.categories) if args.categories else None
    benign_cats = (
        tuple(BenignEdgeCategory(c) for c in args.benign_categories)
        if args.benign_categories
        else None
    )

    config = PayloadConfig(
        framing=framing,
        max_wire_bytes=args.max_wire_bytes,
        max_claim_chars=args.max_claim_chars,
        max_nesting_depth=args.max_nesting_depth,
        seed=args.seed,
        include_positive_controls=not args.no_controls,
        categories=categories,
    )

    benign_cfg = BenignTemplateConfig(
        oracle_key=config.oracle_key,
        framing=framing,
        max_wire_bytes=config.max_wire_bytes,
        max_claim_chars=config.max_claim_chars,
        max_nesting_depth=config.max_nesting_depth,
        seed=config.seed,
        categories=benign_cats,
    )
    benign_gen = BenignEdgeGenerator(benign_cfg)

    if args.benign_only:
        variants = benign_gen.generate()
    else:
        if args.include_benign_edges or benign_cats is not None:
            config.custom_templates.extend(benign_gen.as_payload_templates())
        variants = PayloadGenerator(config).generate()

    encoder = MessageEncoder(framing=framing)
    gate = HardenedBoundaryGate(
        oracle_key=config.oracle_key,
        framing=framing,
        max_wire_bytes=config.max_wire_bytes,
        max_claim_chars=config.max_claim_chars,
        max_nesting_depth=config.max_nesting_depth,
    )

    runner = SuiteRunner(parser=gate, encoder=encoder, strict_error_codes=True)
    summary, records = runner.run(variants, jsonl_path=args.output_jsonl)

    print(format_report(summary, records))

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return 0 if summary.failed_cases == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
