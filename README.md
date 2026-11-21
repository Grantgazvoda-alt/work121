# work121 — LLM Interoperability, OWASP Auditing & Regression Tracking

A unified defensive research and CI evaluation toolkit for analyzing how LLM architectures handle indirect inputs, running live MCP JSON-RPC tool-calling and multimodal probes, auditing OWASP LLM Top 10 (2025) defensive controls, executing 2-hop cross-model relay ablations, and enforcing cross-version safety regression gates.

## Modules

- **`response_classifier.py`**:
  - `ResponsePatternClassifier`: Classifies collected single- and multi-turn evaluation transcripts across `indirect_adversarial_suffix`, `persona_drift_escalation`, and `semantic_frame_injection` categories, with a pluggable `SemanticJudge` protocol.
  - `SusceptibilityMapper`: Aggregates per-model failure/erosion rates, Wilson 95% confidence intervals, paired baseline regression rates, and multi-turn drift slopes.
- **`semantic_judge.py`**:
  - `PolicyAnchorSemanticJudge`: Deterministic, zero-dependency lexical cosine-similarity implementation of `SemanticJudge` comparing assistant turns against safe vs. erosion policy anchors and multi-turn alignment trajectories.
- **`owasp_defensive_auditor.py`**:
  - `OwaspDefensiveCatalog`: Maps evaluation categories to OWASP Top 10 for LLM Applications (`LLM01`, `LLM07`, `LLM08`) with architectural root causes, verification criteria, and automated remediation plans.
  - `DefensiveBoundaryValidator`: Enforces case-insensitive `<untrusted_data>` tag isolation and strict JSON envelope + `intent` allowlist validation for inter-agent communication.
- **`regression_tracker.py`**:
  - `EvalArtifactRecord` & `JsonlRunStore`: Deterministic JSONL serialization capturing the exact prompt template (with SHA-256 fingerprint), rendered evaluation input, response token metadata, semantic-judge drift slope, and outcome—plus optional `redact_raw_content=True` export mode.
  - `RegressionComparator`: Cross-version regression diffing with SHA-256 template/input pinning (`TEMPLATE_OR_INPUT_DRIFT` detection) and continuous drift-slope regression thresholds.
- **`template_library.py`**:
  - `BenignTemplateLibrary` & `ValidationTemplate`: Parameterized library of benign edge-case input-validation templates categorized by technique, testing harmless functional boundaries (instruction-data separation, schema adherence, domain scope, and multi-turn tone stability).
- **`indirect_input_classifier.py`**:
  - `IndirectInputClassifier`, `IndirectInputTemplate`, & `DiagnosticProbe`: 9 cross-architecture academic test templates spanning all 7 `IndirectInputCategory` channels (including MCP tool schemas, chunked RAG streams, and multi-hop relays) and 5 `BoundaryMechanism` paradigms.
  - `ProtocolSerializer` & `InteroperabilityAnalyzer`: Translates diagnostic probes across 5 wire protocols (`OpenAI`, `Anthropic`, `Gemini`, `HuggingFace ChatML`, `Encoder-Decoder Flat`) with `BoundaryFidelity` auditing and an N×N Agent-A → Agent-B handoff compatibility matrix.
  - Supports paired baseline evaluation (`to_paired_test_cases()` for `SusceptibilityMapper.paired_regression_rate`) plus Markdown, JSON, and `booktabs` LaTeX exporters.
- **`eval_runner.py`**:
  - `TargetAgent` protocol, `AgentRegistry`, `SimulatedAgentProfile`, and `execute_evaluation_run` with paired-baseline linkage, full multi-turn transcript persistence, streaming progress callbacks, and custom model adapters.
- **`tool_runtime.py` & `mcp_http_server.py`**:
  - `LocalMcpServer`, `HttpMcpClient`, & Starlette/Uvicorn HTTP JSON-RPC 2.0 server (`/mcp`, `/mcp/clean`, `/tools`, `/health` on port `8775`) with strict `jsonschema` validation and **8 real executable tools**: live Open-Meteo meteorological API (`get_weather`), exact C/F/K math (`convert_temperature`), disk-backed Okapi BM25 retriever over `corpus/*.md` (`search_research_corpus`), `validate_json_schema`, real Pillow PNG chart renderer (`generate_chart_asset`), saved JSONL inspection (`list_eval_runs`, `inspect_eval_run`), and `evaluate_protocol_handoff`.
- **`live_adapters.py`**:
  - `LiveHttpAgent` & `LiveProviderConfig`: Live `TargetAgent` adapters for **Ollama**, **OpenAI / OpenAI-compatible**, **Anthropic**, and **Google Gemini** with automatic HTTP retry backoff, multi-round native MCP tool-calling loops (`run_tool_loop` + automatic tool execution in `run_case`), and automatic real-PNG multimodal routing (`run_multimodal_probe` + `is_multimodal_model`).
  - `EmbeddingAlignmentScorer`: Computes real 768-dim dense cosine alignment per turn via Ollama `/api/embed` (`nomic-embed-text:latest`) for empirical `drift_slope` tracking.
  - `IndirectInvariantJudge`, `LiveLlmSemanticJudge`, & 3-tier `HybridSemanticJudge` (programmatic invariants → live LLM rubric → offline `PolicyAnchorSemanticJudge`).
- **`multi_agent_relay.py`**:
  - `MultiAgentRelayRunner`, `ProtocolRelayGateway`, `GatewayDefensePolicy`, & `RelayScenario`: Executes live 2-hop cross-model relays (`Agent A → Protocol Gateway → Agent B`) with defense ablation toggles (`sanitize_control_tokens`, case-insensitive `escape_xml_delimiters`, `envelope_wrap_json`) and JSON persistence (`to_dict` / `from_dict`).
- **`statistical_analysis.py`**:
  - Pure-Python exact statistical hypothesis testing: two-sided **Fisher's exact test** (`fisher_exact_two_sided`), exact paired **McNemar's test** (`mcnemar_exact_test`, `paired_perturbation_mcnemar`), deterministic 95% **bootstrap mean CIs** (`bootstrap_mean_ci`), pairwise run comparison (`compare_runs_statistically`), and cross-model per-template robustness matrices (`build_template_robustness_matrix`).
- **`report_generator.py`**:
  - `EmpiricalReportGenerator`: Aggregates all persisted JSONL runs (with paired McNemar p-values), 2-hop relay traces, OWASP LLM Top 10 remediation plans, and per-template cross-model breakdowns into publication-ready Markdown (`empirical_report.md`) and `booktabs` LaTeX tables (`empirical_results.tex`).
- **`cli.py`, `run_live_eval.py`, & `hamster_wheel.py`**:
  - `cli.py` (`work121`): Offline/CI regression gate (`run --use-anchor-judge`, `diff --fail-on-regression --fail-on-template-drift`, `audit-owasp`, `stats --fail-on-significant-regression`).
  - `run_live_eval.py` (`work121-live`): Live cross-model evaluation CLI (`--provider`, `--models`, `--all-protocols`, `--paired`, `--mcp-url`, `--embed-model`, `--judge-model`, `--judge-mode`, `--generate-report`).
  - `hamster_wheel.py` (`work121-wheel`): Continuous multi-cycle defensive admission gauntlet (`ModelHamsterWheel`) testing paired indirect inputs, benign templates, wire-protocol translations, 2-hop relay ablations, and OWASP admission verdicts (`ADMIT` / `CONDITIONAL_REVIEW` / `BLOCK`).
- **`agent_comm_robustness/` & `native/`**:
  - Inter-agent wire framing & parser validation suite (`HardenedBoundaryGate`, `PayloadGenerator`, `BenignEdgeGenerator`, `SuiteRunner`) verifying framing resilience (`raw`, `length_prefixed`, `ndjson`), strict UTF-8/JSON, token boundaries, and HMAC-SHA256 `OBSERVED` signatures—paired with a zero-dependency **native Rust binary gate** (`native/src/main.rs`) for microsecond Python-vs-Rust parity checking.
- **`backend_server.py` & `web/` (`work121-server`)**:
  - Unified Starlette/Uvicorn ASGI server (`http://127.0.0.1:8775`) hosting the **10-page interactive Web Workbench (`web/`)**, 44 REST API endpoints, real-time WebSocket job telemetry (`/ws/jobs/{job_id}`), and MCP JSON-RPC 2.0 endpoints (`/mcp`, `/mcp/clean`, `/tools`, `/health`).
- **`dashboard.py`**:
  - Unified 7-tab Streamlit workbench (`Configure & Run`, `Indirect Input Taxonomy`, `Live MCP Tools & Multimodal` [9 real JSON-RPC tools], `Live Multi-Agent Relay Lab`, `Susceptibility & Transcripts` [with exact Fisher/McNemar significance testing], `Version Regression Diff`, `OWASP Audit & Controls`).

## CLI & Server Usage

```bash
# Build Native Rust Wire Gate & Launch Unified 10-Page Web Workbench + MCP Server (port 8775)
cargo build --release --manifest-path native/Cargo.toml
copy native\target\release\work121.exe bin\work121.exe
bin\work121.exe doctor
bin\work121.exe wire-bench --iterations 2000
bin\work121.exe ui --port 8775

# Offline CI Regression Gate, Statistical Hypothesis Testing & OWASP Audit (cli.py)
python cli.py run --agent policy-assistant@v2.0-hardened --agent policy-assistant@v2.1-candidate --use-anchor-judge --out-dir eval_runs
python cli.py diff --baseline eval_runs/<baseline>.jsonl --candidate eval_runs/<candidate>.jsonl --fail-on-regression --fail-on-template-drift
python cli.py stats --baseline eval_runs/<baseline>.jsonl --candidate eval_runs/<candidate>.jsonl --fail-on-significant-regression
python cli.py audit-owasp --run-file eval_runs/<candidate>.jsonl

# Continuous Hamster Wheel Model Admission Gauntlet (hamster_wheel.py)
python hamster_wheel.py --provider ollama --model qwen2.5-coder:7b --cycles 2 --relay-ablation --generate-report --output-dir eval_runs

# Live Cross-Model / 5-Protocol Sweeps (run_live_eval.py)
python run_live_eval.py --provider ollama --models llama3.1:latest qwen2.5-coder:7b --paired --mcp-url http://127.0.0.1:8775/mcp --embed-model nomic-embed-text:latest --output-dir eval_runs
python run_live_eval.py --output-dir eval_runs --generate-report eval_runs/reports
```

## Running Tests & Dashboards

```bash
pytest -v
python backend_server.py --port 8775   # 10-page Web Workbench + REST/WS + MCP server
streamlit run dashboard.py             # 7-tab Streamlit Workbench
```

