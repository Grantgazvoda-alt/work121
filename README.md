# work121 — LLM Interoperability, OWASP Auditing & Regression Tracking

A unified defensive research and CI evaluation toolkit for analyzing how LLM architectures handle indirect inputs, running live MCP JSON-RPC tool-calling and multimodal probes, auditing OWASP LLM Top 10 (2025) defensive controls, executing 2-hop cross-model relay ablations, and enforcing cross-version safety regression gates.

## Modules

- **`response_classifier.py`**:
  - `ResponsePatternClassifier`: Classifies collected single- and multi-turn evaluation transcripts across `indirect_adversarial_suffix`, `persona_drift_escalation`, and `semantic_frame_injection` categories, with a pluggable `SemanticJudge` protocol.
  - `SusceptibilityMapper`: Aggregates per-model failure/erosion rates, Wilson 95% confidence intervals, paired baseline regression rates, and multi-turn drift slopes.
- **`semantic_judge.py`**:
  - `PolicyAnchorSemanticJudge`: Deterministic, zero-dependency lexical cosine-similarity implementation of `SemanticJudge` comparing assistant turns against safe vs. erosion policy anchors and multi-turn alignment trajectories.
- **`owasp_defensive_auditor.py`**:
  - `OwaspDefensiveCatalog`: Maps evaluation categories to OWASP Top 10 for LLM Applications (`LLM01`, `LLM07`, `LLM08`), OWASP Agentic `ASI01:2026`, MITRE ATLAS (`AML.T0051`, `AML.T0054`), and NIST AI 100-2e2025 via `list_technique_references()` (10 technique families including Unicode tag smuggling, Markdown exfil, Policy Puppetry, and H-CoT), architectural root causes, verification criteria, and automated remediation plans.
  - `DefensiveBoundaryValidator`: Enforces case-insensitive `<untrusted_data>` tag isolation, invisible Unicode Tag-block (`U+E0000..U+E007F`) & zero-width stripping, ChatML/`[INST]` delimiter neutralization, Markdown image-exfil defanging, and strict JSON envelope + `intent` allowlist validation.
- **`regression_tracker.py`**:
  - `EvalArtifactRecord` & `JsonlRunStore`: Deterministic JSONL serialization capturing the exact prompt template (with SHA-256 fingerprint), rendered evaluation input, response token metadata, semantic-judge drift slope, and outcome—plus optional `redact_raw_content=True` export mode.
  - `RegressionComparator`: Cross-version regression diffing with SHA-256 template/input pinning (`TEMPLATE_OR_INPUT_DRIFT` detection) and continuous drift-slope regression thresholds.
- **`template_library.py`**:
  - `BenignTemplateLibrary` & `ValidationTemplate`: Parameterized library of benign edge-case and long-horizon input-validation templates categorized by technique, testing instruction-data separation, schema adherence, Policy Puppetry wrappers, threat-intel triage (`threat_intel_ioc_triage_boundary`), and 5–6 turn long-horizon endurance (`long_horizon_crescendo_drift`, `long_horizon_many_shot_context`).
- **`indirect_input_classifier.py`**:
  - `IndirectInputClassifier`, `IndirectInputTemplate`, & `DiagnosticProbe`: 9 cross-architecture academic test templates spanning all 7 `IndirectInputCategory` channels (including MCP tool schemas, chunked RAG streams, and multi-hop relays) and 5 `BoundaryMechanism` paradigms.
  - `ProtocolSerializer` & `InteroperabilityAnalyzer`: Translates diagnostic probes across 5 wire protocols (`OpenAI`, `Anthropic`, `Gemini`, `HuggingFace ChatML`, `Encoder-Decoder Flat`) with `BoundaryFidelity` auditing and an N×N Agent-A → Agent-B handoff compatibility matrix.
  - Supports paired baseline evaluation (`to_paired_test_cases()` for `SusceptibilityMapper.paired_regression_rate`) plus Markdown, JSON, and `booktabs` LaTeX exporters.
- **`eval_runner.py`**:
  - `TargetAgent` protocol, `AgentRegistry`, `SimulatedAgentProfile`, and `execute_evaluation_run` with paired-baseline linkage, full multi-turn transcript persistence, streaming progress callbacks, and custom model adapters.
- **`tool_runtime.py`, `threat_intel.py`, & `mcp_http_server.py`**:
  - `LocalMcpServer`, `HttpMcpClient`, & Starlette/Uvicorn HTTP JSON-RPC 2.0 server (`/mcp`, `/mcp/clean`, `/tools`, `/health` on port `8775`) with strict `jsonschema` validation and **10 real executable tools**: live Open-Meteo meteorological API (`get_weather`), exact C/F/K math (`convert_temperature`), disk-backed Okapi BM25 retriever over `corpus/*.md` (`search_research_corpus`, including MalwareBazaar/vx-underground triage hygiene notes), `validate_json_schema`, real Pillow PNG chart renderer (`generate_chart_asset`), saved JSONL inspection (`list_eval_runs`, `inspect_eval_run`), `evaluate_protocol_handoff`, `statistical_significance_test`, and metadata-only **`lookup_threat_intel`** (`threat_intel.py` supporting abuse.ch **MalwareBazaar** `get_info`/`get_taginfo`/`get_siginfo`/`get_recent` via optional `MALWAREBAZAAR_API_KEY` + curated **vx-underground** defensive advisories with strict download blocking, IOC defanging, and `<untrusted_intel>` isolation).
- **`live_adapters.py` & `live_providers.py`**:
  - `LiveHttpAgent` & `LiveProviderConfig` (`live_adapters.py`) plus `LiveHttpAgent` / `discover_live_agents` / `resolve_spec` (`live_providers.py`): Live adapters for **Google Gemini 4** (`PRIORITY_GEMINI_MODELS`: **`gemini-4-carbon`**, **`gemini-4-neon`**, **`gemini-4-argon`** with `thinkingConfig.thinkingLevel`, `thought` block separation, and `thoughtSignature` preservation), **Ollama** (prioritizing long-context/newer local builds such as `qwen3.8-27b-longctx` and `deepseek-v4.1-flash`), **OpenAI / OpenAI-compatible** (LiteLLM, LM Studio, vLLM, Ollama `/v1`), and **Anthropic**, with retry backoff, native MCP tool-calling loops (`max_rounds=8`), and real-PNG multimodal probes.
  - `EmbeddingAlignmentScorer`: Computes real 768-dim dense cosine alignment per turn via Ollama `/api/embed` (`nomic-embed-text:latest`) for empirical `drift_slope` tracking.
  - `IndirectInvariantJudge`, `LiveLlmSemanticJudge`, & 3-tier `HybridSemanticJudge` (programmatic invariants → live LLM rubric → offline `PolicyAnchorSemanticJudge`).
- **`wire_transport.py`, `wire_suite.py`, `boundary_oracle.py`, & `oracle_keys.py`**:
  - `WireTargetAgent` & `LoopbackWirePeer`: Real TCP loopback `TargetAgent` speaking framed (`raw`, `length_prefixed`, `ndjson`), replay-windowed, nonce-bound, and HMAC-SHA256 `OBSERVED`-signed wire messages loaded via `wire_suite.load_suite()` (with optional upstream `Grantgazvoda-alt/work122` distribution or in-repo fallback).
  - `BoundaryOracle` & `oracle_keys`: Deterministic per-turn boundary pressure policy and env/file HMAC oracle key loader (`.env.example`).
- **`multi_agent_relay.py`**:
  - `MultiAgentRelayRunner`, `ProtocolRelayGateway`, `GatewayDefensePolicy`, & `RelayScenario`: Executes live 2-hop cross-model relays (`Agent A → Protocol Gateway → Agent B`, including `RELAY-04-THREAT-INTEL-SOC-TRIAGE`) with defense ablation toggles (`sanitize_control_tokens`, case-insensitive `escape_xml_delimiters`, `envelope_wrap_json`) and JSON persistence (`to_dict` / `from_dict`).
- **`statistical_analysis.py`**:
  - Pure-Python exact statistical hypothesis testing: two-sided **Fisher's exact test** (`fisher_exact_two_sided`), exact paired **McNemar's test** (`mcnemar_exact_test`, `paired_perturbation_mcnemar`), deterministic 95% **bootstrap mean CIs** (`bootstrap_mean_ci`), pairwise run comparison (`compare_runs_statistically`), and cross-model per-template robustness matrices (`build_template_robustness_matrix`).
- **`report_generator.py`**:
  - `EmpiricalReportGenerator`: Aggregates all persisted JSONL runs (with paired McNemar p-values), 2-hop relay traces, OWASP LLM Top 10 remediation plans, and per-template cross-model breakdowns into publication-ready Markdown (`empirical_report.md`) and `booktabs` LaTeX tables (`empirical_results.tex`).
- **`cli.py`, `run_live_eval.py`, & `hamster_wheel.py`**:
  - `cli.py` (`work121`): Offline/CI regression gate (`run --use-anchor-judge`, `diff --fail-on-regression --fail-on-template-drift`, `audit-owasp`, `stats --fail-on-significant-regression`).
  - `run_live_eval.py` (`work121-live`): Live cross-model evaluation CLI (`--gemini-priority`, `--long-horizon`, `--thinking-level`, `--provider`, `--models`, `--all-protocols`, `--paired`, `--template-id`, `--mcp-url`, `--embed-model`, `--judge-model`, `--judge-mode`, `--generate-report`).
  - `hamster_wheel.py` (`work121-wheel`): Continuous multi-cycle defensive admission gauntlet (`ModelHamsterWheel`) with `--gemini-priority` & `--long-horizon` presets, testing paired indirect inputs, benign templates, wire-protocol translations, 2-hop relay ablations, and OWASP admission verdicts (`ADMIT` / `CONDITIONAL_REVIEW` / `BLOCK`).
- **`agent_comm_robustness/` & `native/`**:
  - Inter-agent wire framing & parser validation suite (`HardenedBoundaryGate`, `PayloadGenerator`, `BenignEdgeGenerator`, `SuiteRunner`) verifying framing resilience (`raw`, `length_prefixed`, `ndjson`), strict UTF-8/JSON, token boundaries, and HMAC-SHA256 `OBSERVED` signatures—paired with a zero-dependency **native Rust binary gate** (`native/src/main.rs`) for microsecond Python-vs-Rust parity checking.
- **`backend_server.py`, `web/`, & `one_tap.ps1` (`work121-server`)**:
  - Unified Starlette/Uvicorn ASGI server (`http://127.0.0.1:8775`) hosting the **10-page interactive Web Workbench (`web/`)**, full `/api/*` REST endpoints with Python + Rust native gate parity, real-time WebSocket job telemetry (`/ws/jobs/{job_id}`), and MCP JSON-RPC 2.0 endpoints (`/mcp`, `/mcp/clean`, `/tools`, `/health`).
  - `one_tap.ps1`: Windows One-Tap launcher starting Ollama, building/verifying `bin/work121.exe`, launching the workbench server, opening Edge/Chrome `--app` mode, and pinning Desktop & Start Menu shortcuts (`work121_art.ico`).
- **`dashboard.py`**:
  - Unified 7-tab Streamlit workbench (`Configure & Run`, `Indirect Input Taxonomy`, `Live MCP Tools & Multimodal` [10 real JSON-RPC tools], `Live Multi-Agent Relay Lab`, `Susceptibility & Transcripts` [with exact Fisher/McNemar significance testing], `Version Regression Diff`, `OWASP Audit & Controls`).

## CLI & Server Usage

```bash
# Install editable package (add [work122] to pin upstream Grantgazvoda-alt/work122 wire suite)
pip install -e ".[ui,dev,work122]"

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
powershell -ExecutionPolicy Bypass -File .\one_tap.ps1   # Windows One-Tap App Window
streamlit run dashboard.py             # 7-tab Streamlit Workbench
```

