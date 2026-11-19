# work121 — LLM Response-Pattern Classification & Regression Tracking

Defensive evaluation toolkit for benign edge-case validation. Target agents are reached only over the [work122](https://github.com/Grantgazvoda-alt/work122) wire suite (`agent-comm-robustness`): raw, length-prefixed, or NDJSON framing, schema checks, and a domain-separated HMAC-SHA256 (`agent-protocol-v1:observed`) verified with `hmac.compare_digest`.

The in-repo `agent_comm_robustness/` directory is the pre-move copy. The eval runner, dashboard, and transport tests do not import it. Delete it in a follow-up once this dependency is confirmed. Swap revisions by changing the single git pin in `pyproject.toml`.

## Modules

- **`response_classifier.py`**: `ResponsePatternClassifier` and `RubricSemanticJudge` score transcripts. `SusceptibilityMapper` aggregates failure and erosion rates, Wilson intervals, and paired baseline regressions (`baseline_transcript_id` / `is_baseline`).
- **`regression_tracker.py`**: `EvalArtifactRecord` and `JsonlRunStore` round-trip those fields. `RegressionComparator` diffs two versions and flags template or input SHA-256 drift.
- **`template_library.py`**: Parameterized benign templates. The default case id hashes the template version, template text, and parameter values. Each evaluation case links to a clean baseline render of the same parameters.
- **`wire_suite.py`**: Loads the installed `agent-comm-robustness` distribution from work122.
- **`oracle_keys.py`**: Reads `AGENT_ORACLE_KEY` or a mode-0600 `AGENT_ORACLE_KEY_FILE`. Parser-suite benchmark keys are refused.
- **`wire_transport.py`**: `WireTargetAgent` speaks the wire protocol to a loopback peer or a remote endpoint. The loopback peer's `BoundaryOracle` scores the turn text that arrived on the wire.
- **`eval_runner.py`**: `AgentRegistry` starts empty. `execute_evaluation_run` runs the queued cases plus their baselines. Provider HTTP failures are stored as `indeterminate`.
- **`live_providers.py`**: `LiveHttpAgent` calls Ollama (`/api/chat`), OpenAI-compatible `/v1/chat/completions` (OpenAI, LiteLLM, LM Studio, vLLM, Ollama `/v1`), Anthropic (`/v1/messages`), and Gemini (`generateContent`). `discover_live_agents()` lists configured providers and includes Ollama only when `/api/tags` answers.
- **`run_live_eval.py`**: CLI for a paired baseline run that writes JSONL.
- **`dashboard.py`**: Streamlit UI for template parameters, the wire peer, the live-provider sidebar, the susceptibility map, and version diffs.
- **`agent_comm_robustness/`**: Retained parser suite. Not the eval import path.

## Install

```bash
pip install -e ".[dev]"
```

That installs `agent-comm-robustness` from work122 at commit `72960f25d8acd3c3152c32b139425f4d3d806d85` and Streamlit 1.67.0.

Create a key without committing it:

```bash
python -c "import secrets; print(secrets.token_hex(32))" > /tmp/oracle.key
chmod 600 /tmp/oracle.key
export AGENT_ORACLE_KEY_FILE=/tmp/oracle.key
```

## Tests and dashboard

```bash
pytest
streamlit run dashboard.py
python run_live_eval.py --list
python run_live_eval.py --provider openai --model gpt-4o-mini --output eval_runs/live.jsonl
```

Live provider keys are environment variables only: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, and optional `LITELLM_API_KEY`. Optional base URLs are `OPENAI_BASE_URL`, `OLLAMA_BASE_URL`, `LITELLM_BASE_URL`, `LMSTUDIO_BASE_URL`, `VLLM_BASE_URL`, `ANTHROPIC_BASE_URL`, and `GEMINI_BASE_URL`.

The retained parser suite can still be executed directly from this checkout:

```bash
python -m agent_comm_robustness --framing length_prefixed --include-benign-edges
```
