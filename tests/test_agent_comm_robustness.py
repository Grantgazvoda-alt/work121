from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Allow running directly without installing into site-packages
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_comm_robustness import (
    BenignEdgeCategory,
    BenignEdgeGenerator,
    BenignTemplateConfig,
    FramingMode,
    HardenedBoundaryGate,
    MessageEncoder,
    ParseResult,
    PayloadCategory,
    PayloadConfig,
    PayloadGenerator,
    PayloadTemplate,
    PayloadVariant,
    SuiteRunner,
    make_valid_derived,
)


class _NaiveJsonParser:
    """Intentionally weak baseline parser to verify SuiteRunner detects false accepts."""

    def parse_bytes(self, raw: bytes) -> ParseResult:
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            return ParseResult(False, None, "parse_error", str(exc))
        if isinstance(data, dict) and "claim" in data:
            return ParseResult(True, None, None, None)
        return ParseResult(False, None, "invalid_shape", "missing claim")


class _CrashingParser:
    """Simulates a parser that raises an unhandled exception on malformed input."""

    def parse_bytes(self, raw: bytes) -> ParseResult:
        raise RuntimeError("simulated parser crash")


class TestAgentCommRobustnessSuite(unittest.TestCase):
    def test_full_suite_across_all_framing_modes(self) -> None:
        for framing in FramingMode:
            with self.subTest(framing=framing.value):
                cfg = PayloadConfig(framing=framing)
                variants = PayloadGenerator(cfg).generate()
                encoder = MessageEncoder(framing=framing)
                gate = HardenedBoundaryGate(
                    oracle_key=cfg.oracle_key,
                    framing=framing,
                    max_wire_bytes=cfg.max_wire_bytes,
                    max_claim_chars=cfg.max_claim_chars,
                    max_nesting_depth=cfg.max_nesting_depth,
                )
                summary, records = SuiteRunner(gate, encoder, strict_error_codes=True).run(variants)

                self.assertGreaterEqual(summary.total_cases, 65)
                self.assertEqual(
                    summary.failed_cases,
                    0,
                    f"Failures in {framing.value}: {[r for r in records if not r.passed]}",
                )
                self.assertEqual(summary.false_accepts, 0)
                self.assertEqual(summary.false_rejects, 0)
                self.assertEqual(summary.unhandled_exceptions, 0)

    def test_custom_threshold_scaling(self) -> None:
        cfg = PayloadConfig(
            max_wire_bytes=4096,
            max_claim_chars=1024,
            max_nesting_depth=4,
        )
        gate = HardenedBoundaryGate(
            oracle_key=cfg.oracle_key,
            max_wire_bytes=cfg.max_wire_bytes,
            max_claim_chars=cfg.max_claim_chars,
            max_nesting_depth=cfg.max_nesting_depth,
        )
        summary, records = SuiteRunner(gate, MessageEncoder()).run(PayloadGenerator(cfg).generate())
        self.assertEqual(summary.failed_cases, 0, [r for r in records if not r.passed])

    def test_custom_template_registration(self) -> None:
        def _custom_factory(cfg: PayloadConfig, rng) -> list[PayloadVariant]:
            return [
                PayloadVariant(
                    case_id="custom_negative_confidence_edge",
                    category=PayloadCategory.SCHEMA_TYPE,
                    description="User-supplied custom template test case",
                    expected_accepted=False,
                    expected_error_code="invalid_confidence",
                    payload=make_valid_derived(confidence=-42.0),
                    tags=("custom",),
                )
            ]

        cfg = PayloadConfig(categories=(PayloadCategory.SCHEMA_TYPE,), include_positive_controls=False)
        gen = PayloadGenerator(cfg)
        gen.register_template(
            PayloadTemplate("custom_edge", PayloadCategory.SCHEMA_TYPE, _custom_factory)
        )
        variants = gen.generate()
        self.assertIn("custom_negative_confidence_edge", [v.case_id for v in variants])

        gate = HardenedBoundaryGate(oracle_key=cfg.oracle_key)
        summary, _ = SuiteRunner(gate, MessageEncoder()).run(variants)
        self.assertEqual(summary.failed_cases, 0)

    def _quiet_logger(self) -> logging.Logger:
        logger = logging.getLogger("test_quiet_runner")
        logger.handlers = [logging.NullHandler()]
        logger.propagate = False
        return logger

    def test_runner_detects_false_accepts_in_weak_parser(self) -> None:
        cfg = PayloadConfig(include_positive_controls=False)
        variants = PayloadGenerator(cfg).generate()
        summary, _ = SuiteRunner(
            _NaiveJsonParser(),
            MessageEncoder(),
            strict_error_codes=False,
            logger=self._quiet_logger(),
        ).run(variants)
        self.assertGreater(summary.false_accepts, 25)
        self.assertGreater(summary.failed_cases, 25)

    def test_runner_isolates_unhandled_parser_exceptions(self) -> None:
        cfg = PayloadConfig(categories=(PayloadCategory.POSITIVE_CONTROL,))
        variants = PayloadGenerator(cfg).generate()
        summary, records = SuiteRunner(
            _CrashingParser(),
            MessageEncoder(),
            logger=self._quiet_logger(),
        ).run(variants)
        self.assertEqual(summary.unhandled_exceptions, len(variants))
        self.assertTrue(all(r.actual_error_code == "unhandled_exception" for r in records))
        self.assertEqual(summary.failed_cases, len(variants))

    def test_jsonl_audit_log_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "responses.jsonl"
            summary_path = Path(tmpdir) / "summary.json"

            env = dict(os.environ)
            env["PYTHONPATH"] = str(REPO_ROOT)
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "agent_comm_robustness",
                    "--framing",
                    "length_prefixed",
                    "--output-jsonl",
                    str(jsonl_path),
                    "--summary-json",
                    str(summary_path),
                ],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
            self.assertTrue(jsonl_path.exists())
            self.assertTrue(summary_path.exists())

            lines = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(len(lines), summary_data["total_cases"])
            self.assertEqual(summary_data["failed_cases"], 0)
            self.assertEqual(summary_data["framing"], "length_prefixed")
            for row in lines:
                self.assertIn("wire_sha256", row)
                self.assertEqual(len(row["wire_sha256"]), 64)
                self.assertIn("elapsed_us", row)
                self.assertTrue(row["passed"])

    def test_benign_edge_templates_across_all_framings(self) -> None:
        for framing in FramingMode:
            with self.subTest(framing=framing.value):
                bcfg = BenignTemplateConfig(framing=framing)
                variants = BenignEdgeGenerator(bcfg).generate()
                self.assertGreaterEqual(len(variants), 50)
                self.assertTrue(all(v.expected_accepted for v in variants))

                gate = HardenedBoundaryGate(
                    oracle_key=bcfg.oracle_key,
                    framing=framing,
                    max_wire_bytes=bcfg.max_wire_bytes,
                    max_claim_chars=bcfg.max_claim_chars,
                    max_nesting_depth=bcfg.max_nesting_depth,
                )
                summary, records = SuiteRunner(
                    gate, MessageEncoder(framing=framing), strict_error_codes=True
                ).run(variants)
                self.assertEqual(
                    summary.false_rejects,
                    0,
                    f"False rejects in {framing.value}: {[r for r in records if not r.passed]}",
                )
                self.assertEqual(summary.failed_cases, 0)
                self.assertEqual(summary.positive_accepted, summary.total_cases)

    def test_benign_templates_integrate_with_payload_generator(self) -> None:
        bgen = BenignEdgeGenerator(
            BenignTemplateConfig(
                categories=(
                    BenignEdgeCategory.UNICODE_LOCALE,
                    BenignEdgeCategory.LENGTH_DEPTH_BOUNDARY,
                )
            )
        )
        cfg = PayloadConfig(
            max_wire_bytes=4096,
            max_claim_chars=1024,
            max_nesting_depth=5,
            custom_templates=bgen.as_payload_templates(),
        )
        variants = PayloadGenerator(cfg).generate()
        gate = HardenedBoundaryGate(
            oracle_key=cfg.oracle_key,
            max_wire_bytes=cfg.max_wire_bytes,
            max_claim_chars=cfg.max_claim_chars,
            max_nesting_depth=cfg.max_nesting_depth,
        )
        summary, records = SuiteRunner(gate, MessageEncoder()).run(variants)
        self.assertEqual(summary.failed_cases, 0, [r for r in records if not r.passed])
        self.assertGreaterEqual(summary.positive_total, 10)
        self.assertEqual(summary.false_accepts, 0)
        self.assertEqual(summary.false_rejects, 0)

    def test_cli_benign_only_mode(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "agent_comm_robustness",
                "--framing",
                "ndjson",
                "--benign-only",
            ],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
        self.assertIn("false rejects: 0", proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
