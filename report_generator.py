from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Optional

from indirect_input_classifier import _escape_latex
from owasp_defensive_auditor import OwaspDefensiveCatalog
from regression_tracker import EvalArtifactRecord, JsonlRunStore
from response_classifier import (
    ClassificationResult,
    ResponseOutcome,
    SusceptibilityMapper,
)
from statistical_analysis import (
    build_template_robustness_matrix,
    paired_perturbation_mcnemar,
)


@dataclass(frozen=True)
class RunMetricsRow:
    """Aggregated empirical metrics for one persisted JSONL evaluation run."""

    run_id: str
    model_id: str
    model_version: str
    architecture_family: str
    total_cases: int
    non_baseline_cases: int
    safe_rate: float
    failure_rate: float
    paired_regression_rate: Optional[float]
    paired_mcnemar_p: Optional[float]
    mean_latency_ms: Optional[float]
    mean_completion_tokens: float
    mean_alignment_score: Optional[float]
    mean_drift_slope: Optional[float]
    mcp_tool_calls: int
    top_failure_signals: tuple[str, ...]


@dataclass(frozen=True)
class RelayMetricsRow:
    """Aggregated metrics for one persisted 2-hop cross-model relay trace."""

    relay_id: str
    scenario_id: str
    agent_a: str
    agent_b: str
    handoff: str
    defense_policy: str
    relay_status: str
    origin_fact_retention: float
    downstream_schema_valid: bool
    perturbation_propagated: bool
    total_latency_ms: float


class EmpiricalReportGenerator:
    """Synthesizes live persisted evaluation artifacts into Markdown and LaTeX research reports."""

    @classmethod
    def collect_run_metrics(cls, runs_dir: Path = Path("eval_runs")) -> list[RunMetricsRow]:
        if not runs_dir.exists():
            return []
        store = JsonlRunStore()
        rows: list[RunMetricsRow] = []

        for path in sorted(runs_dir.glob("*.jsonl")):
            try:
                records = store.load_records(path)
            except ValueError:
                continue
            if not records:
                continue
            rows.append(cls._summarize_records(records))
        return rows

    @classmethod
    def _summarize_records(cls, records: list[EvalArtifactRecord]) -> RunMetricsRow:
        by_case = {r.test_case_id: r for r in records}
        non_base = [r for r in records if not r.extra_metadata.get("is_baseline", False)]
        eval_set = non_base or records
        n = len(eval_set)

        safe_outcomes = (ResponseOutcome.SAFE_REFUSAL, ResponseOutcome.SAFE_REDIRECT)
        safe_count = sum(1 for r in eval_set if r.outcome in safe_outcomes)
        fail_count = sum(1 for r in eval_set if r.outcome == ResponseOutcome.BOUNDARY_FAILURE)

        paired_total = 0
        paired_regs = 0
        for r in non_base:
            base_id = r.extra_metadata.get("baseline_test_case_id")
            if base_id and base_id in by_case:
                base_rec = by_case[base_id]
                if base_rec.outcome in safe_outcomes:
                    paired_total += 1
                    if r.outcome in (ResponseOutcome.PARTIAL_EROSION, ResponseOutcome.BOUNDARY_FAILURE):
                        paired_regs += 1
        paired_reg = round(paired_regs / paired_total, 4) if paired_total > 0 else None

        latencies = [
            float(r.extra_metadata["latency_ms"])
            for r in records
            if isinstance(r.extra_metadata.get("latency_ms"), (int, float))
        ]
        tokens = [r.response_tokens.response_token_count for r in records]
        align_vals = [
            float(score)
            for r in records
            for score in (r.extra_metadata.get("turn_alignment_scores") or [])
            if isinstance(score, (int, float))
        ]
        slopes = [r.drift_slope for r in records if r.drift_slope is not None]
        mcp_calls = sum(len(r.extra_metadata.get("tool_traces") or []) for r in records)

        fail_sig_counts: dict[str, int] = {}
        for r in eval_set:
            if r.outcome in (ResponseOutcome.BOUNDARY_FAILURE, ResponseOutcome.PARTIAL_EROSION):
                for sig in r.signals:
                    fail_sig_counts[sig] = fail_sig_counts.get(sig, 0) + 1
        top_fail_sigs = tuple(
            k for k, _ in sorted(fail_sig_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
        )

        mcnemar_res = paired_perturbation_mcnemar(records) if paired_total > 0 else None
        first = records[0]
        return RunMetricsRow(
            run_id=first.run_id,
            model_id=first.model_id,
            model_version=first.model_version,
            architecture_family=first.architecture_family,
            total_cases=len(records),
            non_baseline_cases=len(non_base),
            safe_rate=round(safe_count / max(1, n), 4),
            failure_rate=round(fail_count / max(1, n), 4),
            paired_regression_rate=paired_reg,
            paired_mcnemar_p=mcnemar_res.p_value if mcnemar_res is not None else None,
            mean_latency_ms=round(sum(latencies) / len(latencies), 1) if latencies else None,
            mean_completion_tokens=round(sum(tokens) / max(1, len(tokens)), 1),
            mean_alignment_score=(
                round(sum(align_vals) / len(align_vals), 4) if align_vals else None
            ),
            mean_drift_slope=round(sum(slopes) / len(slopes), 4) if slopes else None,
            mcp_tool_calls=mcp_calls,
            top_failure_signals=top_fail_sigs,
        )

    @classmethod
    def collect_relay_metrics(cls, runs_dir: Path = Path("eval_runs")) -> list[RelayMetricsRow]:
        relay_dir = runs_dir / "relays"
        if not relay_dir.exists():
            return []
        rows: list[RelayMetricsRow] = []
        for path in sorted(relay_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if "hop_1" not in data or "hop_2" not in data:
                continue
            pol = data.get("defense_policy") or {}
            pol_flags = [k for k, v in pol.items() if v]
            rows.append(
                RelayMetricsRow(
                    relay_id=str(data.get("relay_id", path.stem)),
                    scenario_id=str(data.get("scenario_id", "")),
                    agent_a=str(data["hop_1"].get("agent_key", "")),
                    agent_b=str(data["hop_2"].get("agent_key", "")),
                    handoff=f"{data.get('source_protocol', '')} -> {data.get('target_protocol', '')}",
                    defense_policy="+".join(pol_flags) if pol_flags else "raw_passthrough",
                    relay_status=str(data.get("relay_status", "")),
                    origin_fact_retention=float(data.get("origin_fact_retention", 0.0)),
                    downstream_schema_valid=bool(data.get("downstream_schema_valid", False)),
                    perturbation_propagated=bool(data.get("perturbation_propagated", False)),
                    total_latency_ms=float(data.get("total_latency_ms", 0.0)),
                )
            )
        return rows

    @classmethod
    def render_markdown_report(cls, runs_dir: Path = Path("eval_runs")) -> str:
        run_rows = cls.collect_run_metrics(runs_dir)
        relay_rows = cls.collect_relay_metrics(runs_dir)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines = [
            "# Empirical Cross-Model Interoperability & Indirect-Input Evaluation Report",
            "",
            f"*Generated automatically from live persisted artifacts in `{runs_dir}` ({now}).*",
            "",
            "## 1. Cross-Model Indirect-Input Robustness Results",
            "",
        ]
        if run_rows:
            lines.extend(
                [
                    "| Model | Version | Architecture | Cases | Safe Rate | Failure Rate | Paired Reg. | McNemar p | Mean Latency (ms) | Mean Embed Align | Drift Slope | Live MCP Calls | Top Failure Signals |",
                    "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
                ]
            )
            for r in run_rows:
                pr = f"{r.paired_regression_rate:.2f}" if r.paired_regression_rate is not None else "—"
                mp = f"{r.paired_mcnemar_p:.4f}" if r.paired_mcnemar_p is not None else "—"
                lat = f"{r.mean_latency_ms:.0f}" if r.mean_latency_ms is not None else "—"
                aln = f"{r.mean_alignment_score:.3f}" if r.mean_alignment_score is not None else "—"
                slp = f"{r.mean_drift_slope:+.3f}" if r.mean_drift_slope is not None else "—"
                sigs = ", ".join(r.top_failure_signals) if r.top_failure_signals else "None"
                lines.append(
                    f"| `{r.model_id}` | `{r.model_version}` | {r.architecture_family} | "
                    f"{r.total_cases} | {r.safe_rate:.2f} | {r.failure_rate:.2f} | {pr} | {mp} | "
                    f"{lat} | {aln} | {slp} | {r.mcp_tool_calls} | {sigs} |"
                )
        else:
            lines.append("_No JSONL evaluation runs found._")

        lines.extend(["", "## 2. Live 2-Hop Cross-Model Relay Traces", ""])
        if relay_rows:
            lines.extend(
                [
                    "| Relay ID | Scenario | Agent A | Agent B | Handoff | Status | Fact Ret. | Schema Valid | Perturb. Prop. | Latency (ms) |",
                    "| :--- | :--- | :--- | :--- | :--- | :--- | ---: | :--- | :--- | ---: |",
                ]
            )
            for rr in relay_rows:
                lines.append(
                    f"| `{rr.relay_id}` | `{rr.scenario_id}` | `{rr.agent_a}` | `{rr.agent_b}` | "
                    f"`{rr.handoff}` | **{rr.relay_status}** | {rr.origin_fact_retention:.2f} | "
                    f"{'YES' if rr.downstream_schema_valid else 'NO'} | "
                    f"{'YES' if rr.perturbation_propagated else 'NO'} | {rr.total_latency_ms:.0f} |"
                )
        else:
            lines.append("_No 2-hop relay traces found._")

        lines.extend(["", "## 3. Automated Empirical Highlights", ""])
        if run_rows:
            best = max(run_rows, key=lambda x: (x.safe_rate, -(x.failure_rate)))
            lines.append(
                f"- **Highest Adherence Run:** `{best.model_id}@{best.model_version}` "
                f"(safe rate `{best.safe_rate:.2f}`, failure rate `{best.failure_rate:.2f}` across {best.total_cases} cases)."
            )
            reg_runs = [r for r in run_rows if r.paired_regression_rate and r.paired_regression_rate > 0]
            for rr in reg_runs:
                lines.append(
                    f"- **Paired Perturbation Regression Detected:** `{rr.model_id}@{rr.model_version}` "
                    f"exhibited `paired_regression_rate={rr.paired_regression_rate:.2f}` "
                    f"(signals: {', '.join(rr.top_failure_signals)})."
                )
            total_mcp = sum(r.mcp_tool_calls for r in run_rows)
            lines.append(f"- **Live MCP Tool Invocations Recorded:** `{total_mcp}` JSON-RPC 2.0 tool executions across persisted runs.")

        remediation_plan = cls.collect_remediation_plan(runs_dir)
        lines.extend(["", "## 4. OWASP LLM Top 10 (2025) Defensive Remediation Plan", ""])
        if remediation_plan:
            lines.extend(
                [
                    "| Model | Pattern Category | OWASP Mapping | Risk Tier | Failure Rate | Recommended Defensive Controls |",
                    "| :--- | :--- | :--- | :--- | ---: | :--- |",
                ]
            )
            for rec in remediation_plan:
                controls = "; ".join(rec.recommended_controls)
                lines.append(
                    f"| `{rec.model_id}` | `{rec.pattern_category.value}` | {rec.owasp_category.value} | "
                    f"**{rec.risk_tier}** | {rec.failure_rate:.2f} | {controls} |"
                )
        else:
            lines.append("_No susceptibility profiles available for OWASP remediation mapping._")

        tpl_matrix = build_template_robustness_matrix(runs_dir)
        lines.extend(["", "## 5. Per-Template Cross-Model Robustness Breakdown", ""])
        if tpl_matrix:
            lines.extend(
                [
                    "| Template ID | Category | Trials | Models | Safe Rate | Failure/Erosion Rate | Top Failure Signals |",
                    "| :--- | :--- | ---: | ---: | ---: | ---: | :--- |",
                ]
            )
            for row in tpl_matrix:
                lines.append(
                    f"| `{row['template_id']}` | `{row['category']}` | {row['trials']} | "
                    f"{row['models_tested']} | {row['safe_rate']:.2f} | {row['failure_rate']:.2f} | "
                    f"{row['top_failure_signals']} |"
                )
        else:
            lines.append("_No per-template empirical trials available yet._")
        return "\n".join(lines) + "\n"

    @classmethod
    def collect_remediation_plan(cls, runs_dir: Path = Path("eval_runs")) -> list:
        if not runs_dir.exists():
            return []
        store = JsonlRunStore()
        classifications: list[ClassificationResult] = []
        for path in sorted(runs_dir.glob("*.jsonl")):
            try:
                records = store.load_records(path)
            except ValueError:
                continue
            for rec in records:
                base_case_id = rec.extra_metadata.get("baseline_test_case_id")
                classifications.append(
                    ClassificationResult(
                        transcript_id=f"{rec.run_id}:{rec.test_case_id}",
                        model_id=rec.model_id,
                        architecture_family=rec.architecture_family,
                        category=rec.category,
                        outcome=rec.outcome,
                        confidence=rec.confidence,
                        signals=rec.signals,
                        drift_slope=rec.drift_slope,
                        baseline_transcript_id=(
                            f"{rec.run_id}:{base_case_id}" if base_case_id else None
                        ),
                        is_baseline=bool(rec.extra_metadata.get("is_baseline", False)),
                    )
                )
        if not classifications:
            return []
        profiles = SusceptibilityMapper().build_map(classifications)
        return OwaspDefensiveCatalog.generate_remediation_plan(profiles)

    @classmethod
    def render_latex_empirical_table(cls, runs_dir: Path = Path("eval_runs")) -> str:
        run_rows = cls.collect_run_metrics(runs_dir)
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{llrrrrrr}",
            r"\toprule",
            r"\textbf{Model} & \textbf{Version} & \textbf{Cases} & \textbf{Safe} & "
            r"\textbf{Fail} & \textbf{Paired Reg.} & \textbf{Embed Align.} & \textbf{MCP Calls} \\",
            r"\midrule",
        ]
        for r in run_rows:
            pr = f"{r.paired_regression_rate:.2f}" if r.paired_regression_rate is not None else "--"
            aln = f"{r.mean_alignment_score:.3f}" if r.mean_alignment_score is not None else "--"
            lines.append(
                f"\\texttt{{{_escape_latex(r.model_id)}}} & "
                f"\\texttt{{{_escape_latex(r.model_version)}}} & "
                f"{r.total_cases} & {r.safe_rate:.2f} & {r.failure_rate:.2f} & "
                f"{pr} & {aln} & {r.mcp_tool_calls} \\\\"
            )
        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{Live empirical cross-model robustness, paired perturbation regression, dense embedding alignment, and MCP tool-call telemetry.}",
                r"\label{tab:empirical-results}",
                r"\end{table*}",
            ]
        )
        return "\n".join(lines) + "\n"

    @classmethod
    def export_research_bundle(
        cls,
        runs_dir: Path = Path("eval_runs"),
        out_dir: Path = Path("eval_runs/reports"),
    ) -> dict[str, Path]:
        """Export a reproducible research artifact bundle (Markdown, LaTeX, CSVs, SHA-256 manifest)."""
        import csv
        import hashlib
        import sys

        out_dir.mkdir(parents=True, exist_ok=True)

        md_path = out_dir / "empirical_report.md"
        tex_path = out_dir / "empirical_results.tex"
        runs_csv_path = out_dir / "empirical_summary.csv"
        tpl_csv_path = out_dir / "per_template_robustness.csv"
        manifest_path = out_dir / "reproducibility_manifest.json"

        md_path.write_text(cls.render_markdown_report(runs_dir), encoding="utf-8")
        tex_path.write_text(cls.render_latex_empirical_table(runs_dir), encoding="utf-8")

        run_rows = cls.collect_run_metrics(runs_dir)
        with runs_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "run_id",
                    "model_id",
                    "model_version",
                    "architecture_family",
                    "total_cases",
                    "non_baseline_cases",
                    "safe_rate",
                    "failure_rate",
                    "paired_regression_rate",
                    "paired_mcnemar_p",
                    "mean_latency_ms",
                    "mean_completion_tokens",
                    "mean_alignment_score",
                    "mean_drift_slope",
                    "mcp_tool_calls",
                    "top_failure_signals",
                ],
            )
            writer.writeheader()
            for r in run_rows:
                writer.writerow(
                    {
                        "run_id": r.run_id,
                        "model_id": r.model_id,
                        "model_version": r.model_version,
                        "architecture_family": r.architecture_family,
                        "total_cases": r.total_cases,
                        "non_baseline_cases": r.non_baseline_cases,
                        "safe_rate": r.safe_rate,
                        "failure_rate": r.failure_rate,
                        "paired_regression_rate": r.paired_regression_rate,
                        "paired_mcnemar_p": r.paired_mcnemar_p,
                        "mean_latency_ms": r.mean_latency_ms,
                        "mean_completion_tokens": r.mean_completion_tokens,
                        "mean_alignment_score": r.mean_alignment_score,
                        "mean_drift_slope": r.mean_drift_slope,
                        "mcp_tool_calls": r.mcp_tool_calls,
                        "top_failure_signals": "; ".join(r.top_failure_signals),
                    }
                )

        tpl_rows = build_template_robustness_matrix(runs_dir)
        with tpl_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "template_id",
                    "category",
                    "trials",
                    "models_tested",
                    "safe_rate",
                    "failure_rate",
                    "top_failure_signals",
                ],
            )
            writer.writeheader()
            for row in tpl_rows:
                writer.writerow(row)

        def _sha256(p: Path) -> str:
            return hashlib.sha256(p.read_bytes()).hexdigest()

        artifact_hashes: dict[str, str] = {}
        if runs_dir.exists():
            for p in sorted(runs_dir.glob("*.jsonl")):
                artifact_hashes[p.name] = _sha256(p)
            relay_dir = runs_dir / "relays"
            if relay_dir.exists():
                for p in sorted(relay_dir.glob("*.json")):
                    artifact_hashes[f"relays/{p.name}"] = _sha256(p)

        manifest = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version.split()[0],
            "runs_dir": str(runs_dir),
            "run_count": len(run_rows),
            "relay_count": len(cls.collect_relay_metrics(runs_dir)),
            "template_count": len(tpl_rows),
            "artifact_sha256": artifact_hashes,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return {
            "markdown": md_path,
            "latex": tex_path,
            "runs_csv": runs_csv_path,
            "templates_csv": tpl_csv_path,
            "manifest": manifest_path,
        }
