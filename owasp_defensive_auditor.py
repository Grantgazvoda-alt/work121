from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import re
from typing import Any, Iterable, Mapping

from response_classifier import PatternCategory, VulnerabilityProfile


class OwaspLlmCategory(str, Enum):
    """Subset of OWASP Top 10 for LLM Applications (2025) relevant to boundary auditing."""

    LLM01_PROMPT_INJECTION = "LLM01:2025 Prompt Injection"
    LLM05_IMPROPER_OUTPUT_HANDLING = "LLM05:2025 Improper Output Handling"
    LLM07_SYSTEM_PROMPT_LEAKAGE = "LLM07:2025 System Prompt Leakage & Boundary Erosion"
    LLM08_EXCESSIVE_AGENCY = "LLM08:2025 Excessive Agency & Cross-Component Trust"


@dataclass(frozen=True)
class OwaspDefenseGuidance:
    """Academic and architectural documentation of a vulnerability class and its defenses."""

    owasp_category: OwaspLlmCategory
    mapped_pattern: PatternCategory
    mechanism_summary: str
    architectural_root_cause: str
    defensive_controls: tuple[str, ...]
    verification_criteria: str


@dataclass(frozen=True)
class AttackTechniqueReference:
    """Defensive reference entry mapping prompt-injection / jailbreak families to standards & controls."""

    technique_id: str
    name: str
    family: str
    frameworks: tuple[str, ...]
    risk_tier: str
    mechanism_summary: str
    defensive_controls: tuple[str, ...]
    work121_coverage: str


@dataclass(frozen=True)
class RemediationRecommendation:
    """Actionable defensive hardening recommendation produced from a VulnerabilityProfile."""

    model_id: str
    architecture_family: str
    pattern_category: PatternCategory
    owasp_category: OwaspLlmCategory
    risk_tier: str
    failure_rate: float
    erosion_rate: float
    mechanism_summary: str
    recommended_controls: tuple[str, ...]


class OwaspDefensiveCatalog:
    """Maps response-pattern categories to OWASP LLM Top 10 architectural documentation."""

    _TECHNIQUE_REFERENCES: tuple[AttackTechniqueReference, ...] = (
        AttackTechniqueReference(
            technique_id="TECH-01-INSTRUCTION-OVERRIDE",
            name="Instruction Override & Goal Hijacking",
            family="Direct & Indirect Override",
            frameworks=("OWASP LLM01:2025", "OWASP Agentic ASI01:2026", "MITRE ATLAS AML.T0051", "NIST AI 100-2e2025"),
            risk_tier="HIGH",
            mechanism_summary="Imperative text inside untrusted inputs or data fields asserts higher authority than prior system rules.",
            defensive_controls=("Instruction hierarchy", "Structural <untrusted_data> delimiter isolation", "Output schema gate"),
            work121_coverage="BenignTemplateLibrary, DefensiveBoundaryValidator, HardenedBoundaryGate",
        ),
        AttackTechniqueReference(
            technique_id="TECH-02-SPECIAL-TOKEN-STI",
            name="Special Token / ChatML Delimiter Injection",
            family="Protocol Framing & Serializer",
            frameworks=("OWASP LLM01:2025", "MITRE ATLAS AML.T0051", "NIST AI 100-2e2025"),
            risk_tier="HIGH",
            mechanism_summary="Injects raw chat-template tokens (<|im_start|>, <|im_end|>, [INST], <|system|>) so concatenated text is parsed as privileged role turns.",
            defensive_controls=("Server-side control token neutralization", "Typed provider role arrays instead of flat string formatting"),
            work121_coverage="ProtocolRelayGateway, ProtocolSerializer, TPL-04-CHATML-RELAY, DefensiveBoundaryValidator",
        ),
        AttackTechniqueReference(
            technique_id="TECH-03-RAG-THREAT-INTEL",
            name="RAG, Email & Threat-Intel Indirect Injection",
            family="Indirect Data Plane",
            frameworks=("OWASP LLM01:2025", "OWASP Agentic ASI01:2026", "MITRE ATLAS AML.T0051"),
            risk_tier="CRITICAL",
            mechanism_summary="Adversarial directives hidden in retrieved corpus chunks, MalwareBazaar IOC comments, YARA metadata, or threat writeups hijack downstream summarization.",
            defensive_controls=("Metadata-only lookup", "Privilege-separated reader", "Gateway envelope wrapping", "Strip embedded closing tags"),
            work121_coverage="lookup_threat_intel, search_research_corpus (C1-C7), RELAY-04-THREAT-INTEL-SOC-TRIAGE",
        ),
        AttackTechniqueReference(
            technique_id="TECH-04-MARKDOWN-EXFIL",
            name="Zero-Click Markdown Image & Reference-Link Exfiltration",
            family="Indirect Exfiltration Channel",
            frameworks=("OWASP LLM01:2025", "OWASP LLM05:2025", "MITRE ATLAS AML.T0051"),
            risk_tier="HIGH",
            mechanism_summary="Injects ![alt](https://...) or reference link targets encoding sensitive context into query parameters fetched automatically on UI render.",
            defensive_controls=("Block/defang untrusted external markdown image URLs", "Strict JSON schema outputs without freeform markdown rendering"),
            work121_coverage="DefensiveBoundaryValidator, threat_intel.defang_iocs, corpus/C7_prompt_injection_atlas_reference.md",
        ),
        AttackTechniqueReference(
            technique_id="TECH-05-MCP-CONFUSED-DEPUTY",
            name="MCP Tool Poisoning & Confused Deputy",
            family="Tool & Agentic Agency",
            frameworks=("OWASP LLM08:2025", "OWASP Agentic ASI01:2026", "MITRE ATLAS AML.T0051"),
            risk_tier="CRITICAL",
            mechanism_summary="Adversarial instructions in MCP tool descriptions or tool responses steer privileged tool execution (Cross-Plugin Request Forgery).",
            defensive_controls=("Strict JSON Schema allowlists", "Capability scoping", "Clean vs perturbed MCP parity benchmarking"),
            work121_coverage="LocalMcpServer (/mcp vs /mcp/clean), TPL-07-MCP-TOOL-SCHEMA, validate_inter_agent_message",
        ),
        AttackTechniqueReference(
            technique_id="TECH-06-UNICODE-SMUGGLING",
            name="Unicode Tag Block (U+E0000), Zero-Width & Bidi Smuggling",
            family="Encoding & Obfuscation",
            frameworks=("OWASP LLM01:2025", "MITRE ATLAS AML.T0051", "NIST AI 100-2e2025"),
            risk_tier="HIGH",
            mechanism_summary="Encodes instructions inside invisible Unicode tag code points (U+E0000..U+E007F), zero-width joiners, or bidi overrides hidden from visual review.",
            defensive_controls=("Strip Tag-block U+E0000..U+E007F, zero-width, and bidi characters", "NFKC normalization", "Rust + Python wire UTF-8 gates"),
            work121_coverage="DefensiveBoundaryValidator, ProtocolRelayGateway, threat_intel.sanitize_intel_text, native/src/main.rs",
        ),
        AttackTechniqueReference(
            technique_id="TECH-07-POLICY-PUPPETRY-FRAME",
            name="Policy Puppetry (XML/JSON/INI) & Nested Virtualization",
            family="Semantic Frame & Roleplay",
            frameworks=("OWASP LLM07:2025", "OWASP LLM08:2025", "MITRE ATLAS AML.T0054"),
            risk_tier="HIGH",
            mechanism_summary="Wraps policy overrides inside synthetic XML/INI configs, fictional scripts, or DAN/Skeleton Key personas to bypass surface refusals.",
            defensive_controls=("Frame-invariant semantic judging", "Reject schema-breaking wrappers", "Prompt-leak output guards"),
            work121_coverage="BenignTemplateLibrary (semantic_policy_puppetry_boundary, frame_roleplay_scope_boundary), HybridSemanticJudge",
        ),
        AttackTechniqueReference(
            technique_id="TECH-08-MULTI-TURN-CRESCENDO",
            name="Crescendo, Many-Shot & Echo Chamber Escalation",
            family="Multi-Turn & Context Poisoning",
            frameworks=("OWASP LLM07:2025", "MITRE ATLAS AML.T0054", "NIST AI 100-2e2025"),
            risk_tier="HIGH",
            mechanism_summary="Gradually steers benign dialogue across turns, seeds poisoned context (Echo Chamber), or packs synthetic shots to erode alignment.",
            defensive_controls=("Turn-by-turn PolicyAnchor / EmbeddingAlignmentScorer drift-slope tracking", "Context re-anchoring"),
            work121_coverage="drift_tone_shift_multiturn, SusceptibilityMapper.drift_slope, ModelHamsterWheel",
        ),
        AttackTechniqueReference(
            technique_id="TECH-09-ENCODING-CIPHER-SPLIT",
            name="Payload Splitting, CipherChat, ArtPrompt & FlipAttack",
            family="Encoding & Obfuscation",
            frameworks=("OWASP LLM01:2025", "MITRE ATLAS AML.T0054"),
            risk_tier="MODERATE",
            mechanism_summary="Splits restricted strings across variables or encodes payloads in Base64/ciphers/ASCII art/word flips to bypass keyword filters.",
            defensive_controls=("Decode-and-normalize inspection", "Semantic invariant verification rather than keyword blocklists alone"),
            work121_coverage="HardenedBoundaryGate, Susceptibility & Stats (Fisher / McNemar exact tests)",
        ),
        AttackTechniqueReference(
            technique_id="TECH-10-MULTIMODAL-REASONING",
            name="Visual Typographic Injection & Reasoning-Trace (H-CoT) Steering",
            family="Multimodal & Reasoning Models",
            frameworks=("OWASP LLM01:2025", "MITRE ATLAS AML.T0051", "AML.T0054"),
            risk_tier="HIGH",
            mechanism_summary="Embeds directives inside rendered images/charts (OCR path) or injects fake reasoning checkpoints in exposed CoT traces.",
            defensive_controls=("Typed multimodal boundaries", "<think> trace stripping before invariant evaluation", "Real-PNG probe regression"),
            work121_coverage="TPL-03-TYPED-MULTIMODAL, IndirectInvariantJudge._strip_reasoning, generate_chart_asset",
        ),
    )

    _CATALOG: dict[PatternCategory, OwaspDefenseGuidance] = {
        PatternCategory.INDIRECT_ADVERSARIAL_SUFFIX: OwaspDefenseGuidance(
            owasp_category=OwaspLlmCategory.LLM01_PROMPT_INJECTION,
            mapped_pattern=PatternCategory.INDIRECT_ADVERSARIAL_SUFFIX,
            mechanism_summary=(
                "Untrusted external data (such as retrieved documents or user-supplied records) "
                "is concatenated into the same token stream as privileged system instructions, "
                "allowing trailing directives in data fields to compete with task constraints."
            ),
            architectural_root_cause=(
                "Lack of structural separation between control-plane instructions and "
                "data-plane content within a single autoregressive context window."
            ),
            defensive_controls=(
                "Enforce explicit instruction-data delimiters and spotlighting tags around untrusted content.",
                "Apply strict output schema validation (e.g., JSON Schema) before downstream consumption.",
                "Use privilege-separated retrieval pipelines where untrusted text is summarized by a zero-privilege reader.",
            ),
            verification_criteria=(
                "Paired regression rate between clean baseline records and records containing "
                "benign format-override footnotes remains below 5%."
            ),
        ),
        PatternCategory.PERSONA_DRIFT_ESCALATION: OwaspDefenseGuidance(
            owasp_category=OwaspLlmCategory.LLM07_SYSTEM_PROMPT_LEAKAGE,
            mapped_pattern=PatternCategory.PERSONA_DRIFT_ESCALATION,
            mechanism_summary=(
                "Across multi-turn interactions, attention weight on initial system instructions "
                "decays relative to recent conversational turns, gradually shifting tone, role, "
                "or domain boundaries."
            ),
            architectural_root_cause=(
                "Recency bias in long-context attention and absence of turn-level policy "
                "re-anchoring or state validation."
            ),
            defensive_controls=(
                "Periodically re-anchor system constraints in multi-turn context windows.",
                "Monitor turn-over-turn persona alignment similarity slope and trigger context reset on steep decay.",
                "Enforce stateless domain-scope classifiers on each assistant turn independently of chat history.",
            ),
            verification_criteria=(
                "Mean multi-turn persona alignment slope stays above the configured drift threshold "
                "(e.g., > -0.08 per turn)."
            ),
        ),
        PatternCategory.SEMANTIC_FRAME_INJECTION: OwaspDefenseGuidance(
            owasp_category=OwaspLlmCategory.LLM08_EXCESSIVE_AGENCY,
            mapped_pattern=PatternCategory.SEMANTIC_FRAME_INJECTION,
            mechanism_summary=(
                "Contextual reframing (such as hypothetical scenarios, roleplay simulations, or "
                "unvalidated inter-agent message wrappers) causes the model to treat policy or "
                "schema constraints as suspended inside the alternate frame."
            ),
            architectural_root_cause=(
                "Policy adherence conditioned on surface narrative framing rather than invariant "
                "task and output-contract validation, or implicit trust in peer-agent messages."
            ),
            defensive_controls=(
                "Validate all inter-agent and tool messages against strict typed schemas and allowlists.",
                "Evaluate output compliance using frame-invariant rubric judges independent of conversational framing.",
                "Restrict downstream agent capabilities with least-privilege tool and schema contracts.",
            ),
            verification_criteria=(
                "Zero schema or scope violations when benign out-of-scope requests are wrapped in "
                "simulation or hypothetical frames."
            ),
        ),
    }

    @classmethod
    def get_guidance(cls, category: PatternCategory) -> OwaspDefenseGuidance:
        return cls._CATALOG[category]

    @classmethod
    def list_all(cls) -> list[OwaspDefenseGuidance]:
        return [cls._CATALOG[c] for c in PatternCategory]

    @classmethod
    def list_technique_references(cls) -> list[AttackTechniqueReference]:
        return list(cls._TECHNIQUE_REFERENCES)

    @classmethod
    def generate_remediation_plan(
        cls, profiles: Iterable[VulnerabilityProfile]
    ) -> list[RemediationRecommendation]:
        recommendations: list[RemediationRecommendation] = []
        for profile in profiles:
            guidance = cls.get_guidance(profile.category)
            recommendations.append(
                RemediationRecommendation(
                    model_id=profile.model_id,
                    architecture_family=profile.architecture_family,
                    pattern_category=profile.category,
                    owasp_category=guidance.owasp_category,
                    risk_tier=profile.risk_tier,
                    failure_rate=profile.failure_rate,
                    erosion_rate=profile.erosion_rate,
                    mechanism_summary=guidance.mechanism_summary,
                    recommended_controls=guidance.defensive_controls,
                )
            )
        return recommendations


@dataclass(frozen=True)
class BoundaryAuditResult:
    """Result of defensive pre-input or inter-agent contract validation."""

    passed: bool
    owasp_category: OwaspLlmCategory
    findings: tuple[str, ...]
    sanitized_payload: str


class DefensiveBoundaryValidator:
    """Defensive middleware for validating untrusted context isolation and inter-agent schemas."""

    _INVISIBLE_UNICODE_RE = re.compile(
        r"[\u200b-\u200f\u202a-\u202e\u2060\u2066-\u2069\ufeff\U000e0000-\U000e007f]"
    )
    _SPECIAL_TOKEN_RE = re.compile(
        r"(<\|im_start\|>|<\|im_end\|>|<\|system\|>|<\|start_header_id\|>|<\|end_header_id\|>|<\|eot_id\|>|\[/?INST\])",
        re.IGNORECASE,
    )
    _MD_EXFIL_IMAGE_RE = re.compile(
        r"!\[([^\]]*)\]\(\s*(https?://[^)]+)\)", re.IGNORECASE
    )
    _MD_REF_DEF_RE = re.compile(
        r"^(\s*\[[^\]]+\]:\s*)(https?://\S+)", re.IGNORECASE | re.MULTILINE
    )

    def __init__(
        self,
        *,
        data_tag: str = "untrusted_data",
        required_agent_fields: tuple[str, ...] = ("sender_id", "intent", "payload"),
        allowed_intents: tuple[str, ...] = ("summarize", "classify", "lookup_policy"),
    ) -> None:
        self._data_tag = data_tag
        self._required_fields = required_agent_fields
        self._allowed_intents = allowed_intents

    def isolate_untrusted_context(self, untrusted_text: str) -> BoundaryAuditResult:
        """Wraps untrusted external data in structural tags and neutralizes tag/token/unicode smuggling."""
        closing_re = re.compile(rf"</\s*{re.escape(self._data_tag)}\s*>", flags=re.IGNORECASE)
        findings: list[str] = []
        cleaned = str(untrusted_text)

        if self._INVISIBLE_UNICODE_RE.search(cleaned):
            findings.append("stripped_invisible_unicode_controls")
            cleaned = self._INVISIBLE_UNICODE_RE.sub("", cleaned)

        if self._SPECIAL_TOKEN_RE.search(cleaned):
            findings.append("neutralized_special_template_tokens")
            cleaned = self._SPECIAL_TOKEN_RE.sub("[SANITIZED_CTRL_TOKEN]", cleaned)

        if self._MD_EXFIL_IMAGE_RE.search(cleaned) or self._MD_REF_DEF_RE.search(cleaned):
            findings.append("defanged_markdown_exfil_link")
            cleaned = self._MD_EXFIL_IMAGE_RE.sub(
                r"![defanged-image:\1](blocked-external-url)", cleaned
            )
            cleaned = self._MD_REF_DEF_RE.sub(
                lambda m: m.group(1) + m.group(2).replace("http", "hxxp", 1), cleaned
            )

        if closing_re.search(cleaned):
            findings.append("neutralized_embedded_closing_delimiter")
            cleaned = closing_re.sub("[ESCAPED_DELIMITER]", cleaned)

        wrapped = f"<{self._data_tag}>\n{cleaned}\n</{self._data_tag}>"
        return BoundaryAuditResult(
            passed=len(findings) == 0,
            owasp_category=OwaspLlmCategory.LLM01_PROMPT_INJECTION,
            findings=tuple(findings),
            sanitized_payload=wrapped,
        )

    def validate_inter_agent_message(
        self, raw_message: str | Mapping[str, Any]
    ) -> BoundaryAuditResult:
        """Validates cross-model/inter-agent messages against a strict schema and intent allowlist."""
        findings: list[str] = []
        if isinstance(raw_message, str):
            try:
                parsed = json.loads(raw_message)
            except json.JSONDecodeError:
                return BoundaryAuditResult(
                    passed=False,
                    owasp_category=OwaspLlmCategory.LLM08_EXCESSIVE_AGENCY,
                    findings=("invalid_json_envelope",),
                    sanitized_payload="",
                )
        elif isinstance(raw_message, Mapping):
            parsed = dict(raw_message)
        else:
            parsed = raw_message

        if not isinstance(parsed, dict):
            return BoundaryAuditResult(
                passed=False,
                owasp_category=OwaspLlmCategory.LLM08_EXCESSIVE_AGENCY,
                findings=("non_object_message_envelope",),
                sanitized_payload="",
            )

        missing = [f for f in self._required_fields if f not in parsed]
        if missing:
            findings.append(f"missing_required_fields:{','.join(missing)}")

        unexpected = sorted(set(parsed.keys()) - set(self._required_fields))
        if unexpected:
            findings.append(f"unexpected_fields_stripped:{','.join(unexpected)}")

        intent = str(parsed.get("intent", ""))
        if intent not in self._allowed_intents:
            findings.append(f"unauthorized_intent:{intent}")

        sanitized_obj = {k: parsed[k] for k in self._required_fields if k in parsed}
        return BoundaryAuditResult(
            passed=not any(
                f.startswith(("missing_required_fields", "unauthorized_intent"))
                for f in findings
            ),
            owasp_category=OwaspLlmCategory.LLM08_EXCESSIVE_AGENCY,
            findings=tuple(findings),
            sanitized_payload=json.dumps(sanitized_obj, sort_keys=True),
        )
