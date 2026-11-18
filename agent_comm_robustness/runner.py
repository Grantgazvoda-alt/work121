from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .encoder import EncodedMessage, MessageEncoder
from .payloads import PayloadVariant
from .protocol import FramingMode, ParseResult, WireParser


@dataclass(frozen=True)
class CaseExecutionRecord:
    """Structured audit log entry for a single evaluated wire message."""

    case_id: str
    category: str
    description: str
    framing: str
    byte_length: int
    wire_sha256: str
    expected_accepted: bool
    actual_accepted: bool
    expected_error_code: str | None
    actual_error_code: str | None
    actual_error_detail: str | None
    passed: bool
    elapsed_us: float
    tags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SuiteSummary:
    """Aggregate benchmark metrics across the full suite run."""

    framing: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    negative_total: int
    negative_rejected: int
    false_accepts: int
    positive_total: int
    positive_accepted: int
    false_rejects: int
    unhandled_exceptions: int
    by_category: dict[str, dict[str, int]]
    error_code_histogram: dict[str, int]
    mean_latency_us: float
    p95_latency_us: float
    max_latency_us: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SuiteRunner:
    """Iterates through encoded payloads, evaluates a WireParser, and logs responses."""

    def __init__(
        self,
        parser: WireParser,
        encoder: MessageEncoder | None = None,
        *,
        strict_error_codes: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        self.parser = parser
        self.encoder = encoder or MessageEncoder(FramingMode.RAW)
        self.strict_error_codes = strict_error_codes
        self.logger = logger or logging.getLogger("agent_comm_robustness.runner")

    def _evaluate_one(self, encoded: EncodedMessage) -> tuple[CaseExecutionRecord, bool]:
        v = encoded.variant
        crashed = False
        start_ns = time.perf_counter_ns()
        try:
            result: ParseResult = self.parser.parse_bytes(encoded.raw_bytes)
        except Exception as exc:  # Defensive isolation for third-party parsers under test
            crashed = True
            result = ParseResult(
                accepted=False,
                message=None,
                error_code="unhandled_exception",
                error_detail=f"{type(exc).__name__}: {exc}",
            )
        elapsed_us = (time.perf_counter_ns() - start_ns) / 1_000.0

        outcome_matches = result.accepted == v.expected_accepted
        code_matches = (
            True
            if (not self.strict_error_codes or v.expected_error_code is None)
            else (result.error_code == v.expected_error_code)
        )
        passed = (not crashed) and outcome_matches and code_matches

        record = CaseExecutionRecord(
            case_id=v.case_id,
            category=v.category.value,
            description=v.description,
            framing=encoded.framing.value,
            byte_length=encoded.byte_length,
            wire_sha256=encoded.sha256,
            expected_accepted=v.expected_accepted,
            actual_accepted=result.accepted,
            expected_error_code=v.expected_error_code,
            actual_error_code=result.error_code,
            actual_error_detail=result.error_detail,
            passed=passed,
            elapsed_us=round(elapsed_us, 3),
            tags=list(v.tags),
        )
        return record, crashed

    def run(
        self,
        variants: Iterable[PayloadVariant],
        *,
        jsonl_path: str | Path | None = None,
    ) -> tuple[SuiteSummary, list[CaseExecutionRecord]]:
        records: list[CaseExecutionRecord] = []
        latencies: list[float] = []
        by_category: dict[str, dict[str, int]] = defaultdict(
            lambda: {"total": 0, "passed": 0, "failed": 0}
        )
        error_hist: Counter[str] = Counter()

        neg_total = neg_rejected = false_accepts = 0
        pos_total = pos_accepted = false_rejects = 0
        unhandled = 0

        out_fp = None
        try:
            if jsonl_path is not None:
                path = Path(jsonl_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                out_fp = path.open("w", encoding="utf-8", newline="\n")

            for encoded in self.encoder.encode_stream(variants):
                rec, crashed = self._evaluate_one(encoded)
                records.append(rec)
                latencies.append(rec.elapsed_us)

                if crashed:
                    unhandled += 1

                cat_stats = by_category[rec.category]
                cat_stats["total"] += 1
                if rec.passed:
                    cat_stats["passed"] += 1
                else:
                    cat_stats["failed"] += 1

                if rec.actual_error_code is not None:
                    error_hist[rec.actual_error_code] += 1

                if rec.expected_accepted:
                    pos_total += 1
                    if rec.actual_accepted:
                        pos_accepted += 1
                    else:
                        false_rejects += 1
                else:
                    neg_total += 1
                    if rec.actual_accepted:
                        false_accepts += 1
                    else:
                        neg_rejected += 1

                status = "PASS" if rec.passed else "FAIL"
                self.logger.log(
                    logging.INFO if rec.passed else logging.WARNING,
                    "[%s] %-42s bytes=%-6d accepted=%-5s code=%-34s (expected=%s) %.1fus",
                    status,
                    rec.case_id,
                    rec.byte_length,
                    rec.actual_accepted,
                    rec.actual_error_code,
                    rec.expected_error_code,
                    rec.elapsed_us,
                )

                if out_fp is not None:
                    out_fp.write(json.dumps(rec.to_dict(), sort_keys=True) + "\n")
        finally:
            if out_fp is not None:
                out_fp.close()

        total = len(records)
        passed_count = sum(1 for r in records if r.passed)
        failed_count = total - passed_count
        sorted_lat = sorted(latencies)
        mean_lat = sum(sorted_lat) / total if total else 0.0
        p95_idx = min(total - 1, int(0.95 * (total - 1))) if total else 0
        p95_lat = sorted_lat[p95_idx] if total else 0.0
        max_lat = sorted_lat[-1] if total else 0.0

        summary = SuiteSummary(
            framing=self.encoder.framing.value,
            total_cases=total,
            passed_cases=passed_count,
            failed_cases=failed_count,
            pass_rate=round(passed_count / total, 4) if total else 1.0,
            negative_total=neg_total,
            negative_rejected=neg_rejected,
            false_accepts=false_accepts,
            positive_total=pos_total,
            positive_accepted=pos_accepted,
            false_rejects=false_rejects,
            unhandled_exceptions=unhandled,
            by_category=dict(by_category),
            error_code_histogram=dict(error_hist.most_common()),
            mean_latency_us=round(mean_lat, 3),
            p95_latency_us=round(p95_lat, 3),
            max_latency_us=round(max_lat, 3),
        )
        return summary, records


def format_report(summary: SuiteSummary, records: list[CaseExecutionRecord]) -> str:
    """Render a concise human-readable report suitable for terminal or CI logs."""
    lines = [
        f"=== Agent Communication Robustness Suite ({summary.framing}) ===",
        (
            f"Total: {summary.total_cases} | Passed: {summary.passed_cases} | "
            f"Failed: {summary.failed_cases} | Pass rate: {summary.pass_rate:.2%}"
        ),
        (
            f"Negative cases: {summary.negative_rejected}/{summary.negative_total} rejected "
            f"(false accepts: {summary.false_accepts}) | "
            f"Positive controls: {summary.positive_accepted}/{summary.positive_total} accepted "
            f"(false rejects: {summary.false_rejects})"
        ),
        (
            f"Unhandled parser exceptions: {summary.unhandled_exceptions} | "
            f"Latency (us): mean={summary.mean_latency_us:.1f}, "
            f"p95={summary.p95_latency_us:.1f}, max={summary.max_latency_us:.1f}"
        ),
        "",
        "Category Breakdown:",
    ]
    for cat, stats in sorted(summary.by_category.items()):
        lines.append(
            f"  - {cat:<18} total={stats['total']:<3} "
            f"passed={stats['passed']:<3} failed={stats['failed']}"
        )

    failures = [r for r in records if not r.passed]
    if failures:
        lines.append("")
        lines.append("Failures:")
        for r in failures:
            lines.append(
                f"  * {r.case_id}: actual_accepted={r.actual_accepted} "
                f"(expected={r.expected_accepted}), actual_code={r.actual_error_code!r} "
                f"(expected={r.expected_error_code!r}) — {r.actual_error_detail}"
            )
    return "\n".join(lines)
