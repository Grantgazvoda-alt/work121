# work121 — LLM Response-Pattern Classification & Regression Tracking

A defensive evaluation toolkit for analyzing LLM response robustness across architectures, running parameterized benign edge-case input-validation suites, and tracking safety regressions across model versions.

## Modules

- **`response_classifier.py`**:
  - `ResponsePatternClassifier`: Classifies collected single- and multi-turn evaluation transcripts across `indirect_adversarial_suffix`, `persona_drift_escalation`, and `semantic_frame_injection` categories, with a pluggable `SemanticJudge` protocol.
  - `SusceptibilityMapper`: Aggregates per-model failure/erosion rates, Wilson 95% confidence intervals, paired baseline regression rates, and multi-turn drift slopes.
- **`regression_tracker.py`**:
  - `EvalArtifactRecord` & `JsonlRunStore`: Deterministic JSONL serialization capturing the exact prompt template (with SHA-256 fingerprint), rendered evaluation input, response token metadata, semantic-judge drift slope, and outcome—plus optional `redact_raw_content=True` export mode.
  - `RegressionComparator`: Cross-version regression diffing with SHA-256 template/input pinning (`TEMPLATE_OR_INPUT_DRIFT` detection) and continuous drift-slope regression thresholds.
- **`template_library.py`**:
  - `BenignTemplateLibrary` & `ValidationTemplate`: Parameterized library of benign edge-case input-validation templates categorized by technique, testing harmless functional boundaries (instruction-data separation, schema adherence, domain scope, and multi-turn tone stability).
- **`eval_runner.py`**:
  - `TargetAgent` protocol, `AgentRegistry`, `SimulatedAgentProfile`, and `execute_evaluation_run` for streaming progress callbacks and wiring custom model adapters into the evaluation pipeline.
- **`dashboard.py`**:
  - Streamlit frontend wrapper for configuring template parameters, selecting target agent versions, running real-time evaluations, inspecting the cross-architecture susceptibility map, and visualizing version-over-version regression diffs.

## Running Tests

```bash
pytest
```

## Launching the Streamlit Dashboard

```bash
pip install streamlit
streamlit run dashboard.py
```
