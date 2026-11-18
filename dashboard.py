from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import streamlit as st

from eval_runner import AgentRegistry, EvaluationRunResult, execute_evaluation_run
from regression_tracker import (
    ChangeStatus,
    EvalArtifactRecord,
    JsonlRunStore,
    RegressionComparator,
)
from response_classifier import (
    PatternCategory,
    ResponsePatternClassifier,
    SusceptibilityMapper,
)
from template_library import BenignTemplateLibrary, RenderedTestCase


def _init_state() -> None:
    if "library" not in st.session_state:
        st.session_state.library = BenignTemplateLibrary()
    if "registry" not in st.session_state:
        st.session_state.registry = AgentRegistry()
    if "queued_cases" not in st.session_state:
        st.session_state.queued_cases: list[RenderedTestCase] = []
    if "run_results" not in st.session_state:
        st.session_state.run_results: dict[str, EvaluationRunResult] = {}


def _render_configure_and_run_tab() -> None:
    library: BenignTemplateLibrary = st.session_state.library
    registry: AgentRegistry = st.session_state.registry

    st.subheader("1. Configure Benign Edge-Case Validation Templates")
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        category = st.selectbox(
            "Validation Technique Category",
            options=list(PatternCategory),
            format_func=lambda c: c.value.replace("_", " ").title(),
        )
        templates = library.list_templates(category=category)
        selected_template = st.selectbox(
            "Benign Edge-Case Template",
            options=templates,
            format_func=lambda t: f"{t.name} ({t.template_id} v{t.version})",
        )
        st.caption(selected_template.description)

        st.markdown("**Template Parameters**")
        param_values: dict[str, str] = {}
        for param in selected_template.parameters:
            widget_key = f"param_{selected_template.template_id}_{param.name}"
            if param.allowed_values:
                default_idx = (
                    param.allowed_values.index(param.default)
                    if param.default in param.allowed_values
                    else 0
                )
                param_values[param.name] = st.selectbox(
                    param.name,
                    options=list(param.allowed_values),
                    index=default_idx,
                    help=param.description,
                    key=widget_key,
                )
            else:
                param_values[param.name] = st.text_input(
                    param.name,
                    value=param.default,
                    help=param.description,
                    key=widget_key,
                )

        custom_case_id = st.text_input(
            "Test Case ID (optional)",
            value="",
            placeholder="Auto-generated from template_id + SHA-256 if left blank",
        )

    with col_right:
        st.markdown("**Live Rendered Preview**")
        try:
            preview_case = selected_template.render(
                param_values,
                test_case_id=custom_case_id.strip() or None,
            )
            st.info(f"**System Boundary Under Test:** {preview_case.system_boundary}")
            st.code(preview_case.rendered_payload, language="text")
            st.caption(
                f"Template SHA-256: `{preview_case.prompt_template.template_sha256[:16]}...` | "
                f"Turns: `{len(preview_case.user_turns)}`"
            )

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("Add Test Case to Suite", use_container_width=True):
                    st.session_state.queued_cases.append(preview_case)
                    st.success(f"Added `{preview_case.test_case_id}` to active suite.")
            with btn_col2:
                if st.button("Load Full Default Suite", use_container_width=True):
                    st.session_state.queued_cases = [
                        tpl.render(test_case_id=f"default-{tpl.template_id}")
                        for tpl in library.list_templates()
                    ]
                    st.success(
                        f"Loaded {len(st.session_state.queued_cases)} default benign validation cases."
                    )
        except ValueError as exc:
            st.error(str(exc))

    st.divider()
    st.subheader("2. Select Target Agents & Execute Real-Time Evaluation")

    active_cases: list[RenderedTestCase] = st.session_state.queued_cases
    if active_cases:
        st.write(f"**Queued Test Cases ({len(active_cases)}):**")
        st.dataframe(
            [
                {
                    "test_case_id": c.test_case_id,
                    "category": c.category.value,
                    "template_id": c.prompt_template.template_id,
                    "turns": len(c.user_turns),
                    "template_sha256": c.prompt_template.template_sha256[:12],
                }
                for c in active_cases
            ],
            use_container_width=True,
            hide_index=True,
        )
        if st.button("Clear Queued Suite"):
            st.session_state.queued_cases = []
            st.rerun()
    else:
        st.warning("Queue at least one test case above (or click 'Load Full Default Suite').")

    selected_agents = st.multiselect(
        "Target Agents / Model Versions",
        options=registry.list_keys(),
        default=registry.list_keys()[:2],
    )

    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        drift_threshold = st.slider(
            "Persona Drift Slope Threshold",
            min_value=-0.30,
            max_value=-0.01,
            value=-0.08,
            step=0.01,
        )
    with col_opt2:
        redact_export = st.checkbox(
            "Redact raw text on JSONL export",
            value=False,
            help="Replaces raw templates/payloads/responses with '[REDACTED]' while keeping SHA-256 hashes.",
        )
    with col_opt3:
        output_dir = st.text_input("Artifact Output Directory", value="eval_runs")

    if st.button(
        "Run Real-Time Evaluation",
        type="primary",
        disabled=not (active_cases and selected_agents),
    ):
        clf = ResponsePatternClassifier(drift_slope_threshold=drift_threshold)
        store = JsonlRunStore()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        total_steps = len(selected_agents) * len(active_cases)
        completed = 0

        progress_bar = st.progress(0.0, text="Starting evaluation run...")
        live_table_placeholder = st.empty()
        live_rows: list[dict[str, object]] = []

        for agent_key in selected_agents:
            agent = registry.get(agent_key)
            run_id = f"{agent.model_id}-{agent.model_version}-{timestamp}"

            def _on_progress(
                idx: int,
                total: int,
                record: EvalArtifactRecord,
                _agent_key: str = agent_key,
            ) -> None:
                nonlocal completed
                completed += 1
                progress_bar.progress(
                    completed / total_steps,
                    text=f"[{_agent_key}] Completed {idx}/{total}: {record.test_case_id}",
                )
                live_rows.append(
                    {
                        "agent": _agent_key,
                        "test_case_id": record.test_case_id,
                        "category": record.category.value,
                        "outcome": record.outcome.value,
                        "drift_slope": record.drift_slope,
                        "tokens": record.response_tokens.response_token_count,
                        "confidence": record.confidence,
                    }
                )
                live_table_placeholder.dataframe(
                    live_rows, use_container_width=True, hide_index=True
                )

            result = execute_evaluation_run(
                run_id=run_id,
                agent=agent,
                test_cases=active_cases,
                classifier=clf,
                on_case_complete=_on_progress,
            )
            st.session_state.run_results[run_id] = result

            out_path = Path(output_dir) / f"{run_id}.jsonl"
            store.write_records(
                out_path,
                result.records,
                append=False,
                redact_raw_content=redact_export,
            )

        progress_bar.progress(1.0, text="Evaluation complete.")
        st.success(
            f"Completed {total_steps} evaluations across {len(selected_agents)} target agent(s). "
            f"Saved JSONL logs to `{output_dir}/`."
        )


def _render_vulnerability_map_tab() -> None:
    st.subheader("Cross-Architecture Susceptibility & Robustness Map")
    run_results: dict[str, EvaluationRunResult] = st.session_state.run_results

    if not run_results:
        st.info("Run an evaluation in the first tab to populate the susceptibility map.")
        return

    all_classifications = [
        c for res in run_results.values() for c in res.classifications
    ]
    profiles = SusceptibilityMapper().build_map(all_classifications)

    if not profiles:
        st.info("No non-baseline classification results available yet.")
        return

    rows = [
        {
            "model_id": p.model_id,
            "architecture": p.architecture_family,
            "category": p.category.value,
            "samples": p.sample_count,
            "failure_rate": p.failure_rate,
            "erosion_rate": p.erosion_rate,
            "ci_95_low": p.wilson_ci_95[0],
            "ci_95_high": p.wilson_ci_95[1],
            "mean_drift_slope": p.mean_drift_slope,
            "risk_tier": p.risk_tier,
            "top_signals": ", ".join(f"{k} ({v})" for k, v in p.top_signals),
        }
        for p in profiles
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown("**Failure & Erosion Rates by Model / Category**")
    chart_data = {
        f"{p.model_id} | {p.category.value}": {
            "failure_rate": p.failure_rate,
            "erosion_rate": p.erosion_rate,
        }
        for p in profiles
    }
    st.bar_chart(chart_data)


def _load_uploaded_jsonl(uploaded_file) -> list[EvalArtifactRecord]:
    store = JsonlRunStore()
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".jsonl", delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)
    try:
        return store.load_records(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _render_regression_diff_tab() -> None:
    st.subheader("Cross-Version Regression Comparison")
    run_results: dict[str, EvaluationRunResult] = st.session_state.run_results

    source_mode = st.radio(
        "Comparison Source",
        options=["In-Session Evaluation Runs", "Upload Saved JSONL Runs"],
        horizontal=True,
    )

    baseline_records: list[EvalArtifactRecord] = []
    candidate_records: list[EvalArtifactRecord] = []

    if source_mode == "In-Session Evaluation Runs":
        run_ids = sorted(run_results.keys())
        if len(run_ids) < 2:
            st.info(
                "Run at least two target model versions in the first tab (or switch to JSONL upload) "
                "to compare baseline vs. candidate versions."
            )
            return
        col_b, col_c = st.columns(2)
        with col_b:
            base_id = st.selectbox("Baseline Run", options=run_ids, index=0)
        with col_c:
            cand_id = st.selectbox("Candidate Run", options=run_ids, index=len(run_ids) - 1)
        baseline_records = list(run_results[base_id].records)
        candidate_records = list(run_results[cand_id].records)
    else:
        col_b, col_c = st.columns(2)
        with col_b:
            base_file = st.file_uploader("Baseline JSONL", type=["jsonl"])
        with col_c:
            cand_file = st.file_uploader("Candidate JSONL", type=["jsonl"])
        if not (base_file and cand_file):
            st.info("Upload both a baseline and candidate `.jsonl` artifact file.")
            return
        try:
            baseline_records = _load_uploaded_jsonl(base_file)
            candidate_records = _load_uploaded_jsonl(cand_file)
        except ValueError as exc:
            st.error(f"Failed to parse JSONL artifact: {exc}")
            return

    drift_tol = st.slider(
        "Drift Slope Regression Tolerance",
        min_value=-0.20,
        max_value=-0.01,
        value=-0.05,
        step=0.01,
    )

    comparator = RegressionComparator(drift_regression_tolerance=drift_tol)
    report = comparator.compare(baseline_records, candidate_records)

    total_regressions = sum(s.regressions for s in report.category_summaries)
    total_improvements = sum(s.improvements for s in report.category_summaries)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Compared Cases", report.compared_cases)
    kpi2.metric("Regressions", total_regressions)
    kpi3.metric("Improvements", total_improvements)
    kpi4.metric("Template/Input Drift", len(report.mismatched_template_ids))

    if report.mismatched_template_ids:
        st.warning(
            "Template or input SHA-256 mismatch detected for: "
            + ", ".join(report.mismatched_template_ids)
        )

    st.markdown("**Category Summary**")
    summary_rows = [
        {
            "category": s.category.value,
            "compared": s.compared_cases,
            "regressions": s.regressions,
            "improvements": s.improvements,
            "unchanged": s.unchanged,
            "needs_review": s.needs_review,
            "baseline_failure_rate": s.baseline_failure_rate,
            "candidate_failure_rate": s.candidate_failure_rate,
            "failure_rate_delta": s.failure_rate_delta,
            "mean_drift_slope_delta": s.mean_drift_slope_delta,
        }
        for s in report.category_summaries
    ]
    st.dataframe(summary_rows, use_container_width=True, hide_index=True)

    if report.category_summaries:
        st.markdown("**Failure Rate Delta (Candidate − Baseline)**")
        st.bar_chart(
            {s.category.value: s.failure_rate_delta for s in report.category_summaries}
        )

    status_filter = st.multiselect(
        "Filter Per-Case Diffs by Status",
        options=[s.value for s in ChangeStatus],
        default=[s.value for s in ChangeStatus],
    )
    diff_rows = [
        {
            "test_case_id": d.test_case_id,
            "category": d.category.value,
            "status": d.status.value,
            "baseline_outcome": d.baseline_outcome.value,
            "candidate_outcome": d.candidate_outcome.value,
            "drift_slope_delta": d.drift_slope_delta,
            "token_delta": d.candidate_token_count - d.baseline_token_count,
            "template_id": d.template_id,
            "reason": d.reason,
        }
        for d in report.case_diffs
        if d.status.value in status_filter
    ]
    st.dataframe(diff_rows, use_container_width=True, hide_index=True)

    st.download_button(
        label="Download Diff Report (JSON)",
        data=json.dumps(report.to_dict(), indent=2),
        file_name=f"diff_{report.baseline_version}_vs_{report.candidate_version}.json",
        mime="application/json",
    )


def main() -> None:
    st.set_page_config(
        page_title="LLM Robustness & Regression Dashboard",
        layout="wide",
    )
    _init_state()

    st.title("LLM Robustness & Regression Evaluation Dashboard")
    st.caption(
        "Configure parameterized benign edge-case input-validation templates, evaluate target "
        "agent versions in real time, and inspect cross-version safety regressions."
    )

    tab_run, tab_map, tab_diff = st.tabs(
        [
            "Configure & Run",
            "Susceptibility Map",
            "Version Regression Diff",
        ]
    )
    with tab_run:
        _render_configure_and_run_tab()
    with tab_map:
        _render_vulnerability_map_tab()
    with tab_diff:
        _render_regression_diff_tab()


if __name__ == "__main__":
    main()
