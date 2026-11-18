# work121 — LLM Response-Pattern Classification & Regression Tracking

A defensive evaluation toolkit for analyzing LLM response robustness across architectures and tracking safety regressions across model versions.

## Modules

- **`response_classifier.py`**:
  - `ResponsePatternClassifier`: Classifies collected single- and multi-turn evaluation transcripts across `indirect_adversarial_suffix`, `persona_drift_escalation`, and `semantic_frame_injection` categories, with a pluggable `SemanticJudge` protocol.
  - `SusceptibilityMapper`: Aggregates per-model failure/erosion rates, Wilson 95% confidence intervals, paired baseline regression rates, and multi-turn drift slopes.
- **`regression_tracker.py`**:
  - `EvalArtifactRecord` & `JsonlRunStore`: Deterministic JSONL serialization capturing the exact prompt template (with SHA-256 fingerprint), rendered evaluation input, response token metadata, semantic-judge drift slope, and outcome—plus optional `redact_raw_content=True` export mode.
  - `RegressionComparator`: Cross-version regression diffing with SHA-256 template/input pinning (`TEMPLATE_OR_INPUT_DRIFT` detection) and continuous drift-slope regression thresholds.

## Running Tests

```bash
pytest
```
