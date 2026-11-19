from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
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
        """Wraps untrusted external data in structural tags and neutralizes tag-closing collisions."""
        closing_tag = f"</{self._data_tag}>"
        findings: list[str] = []
        cleaned = untrusted_text

        if closing_tag.lower() in cleaned.lower():
            findings.append("neutralized_embedded_closing_delimiter")
            cleaned = cleaned.replace(closing_tag, "[ESCAPED_DELIMITER]").replace(
                closing_tag.upper(), "[ESCAPED_DELIMITER]"
            )

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
        else:
            parsed = dict(raw_message)

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
