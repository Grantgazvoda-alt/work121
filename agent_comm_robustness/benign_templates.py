from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from .payloads import (
    PayloadCategory,
    PayloadConfig,
    PayloadTemplate,
    PayloadVariant,
    make_valid_derived,
)
from .protocol import (
    AgentMessage,
    FramingMode,
    Performative,
    Provenance,
    sign_observed_claim,
)


class BenignEdgeCategory(str, Enum):
    UNICODE_LOCALE = "unicode_locale"
    NUMERIC_PRECISION = "numeric_precision"
    LENGTH_DEPTH_BOUNDARY = "length_depth_boundary"
    STRING_ESCAPING = "string_escaping"
    COLLECTION_SHAPE = "collection_shape"
    PROTOCOL_MATRIX = "protocol_matrix"


@dataclass
class BenignTemplateConfig:
    """Parameterized knobs for generating benign edge-case validation inputs."""

    oracle_key: bytes = b"benchmark-oracle-key-v1"
    framing: FramingMode = FramingMode.RAW
    max_wire_bytes: int = 32_768
    max_claim_chars: int = 8_192
    max_nesting_depth: int = 8
    max_id_chars: int = 128
    list_sizes: tuple[int, ...] = (0, 1, 10)
    round_indices: tuple[int, ...] = (0, 1, 255, 2**31 - 1)
    confidence_grid: tuple[float, ...] = (0.0, 1e-6, 0.5, 1.0 - 1e-6, 1.0)
    seed: int = 42
    categories: tuple[BenignEdgeCategory, ...] | None = None

    @classmethod
    def from_payload_config(cls, cfg: PayloadConfig) -> BenignTemplateConfig:
        return cls(
            oracle_key=cfg.oracle_key,
            framing=cfg.framing,
            max_wire_bytes=cfg.max_wire_bytes,
            max_claim_chars=cfg.max_claim_chars,
            max_nesting_depth=cfg.max_nesting_depth,
            seed=cfg.seed,
        )


@dataclass(frozen=True)
class BenignTemplate:
    """A parameterized template expanding into benign valid edge-case PayloadVariants."""

    name: str
    edge_category: BenignEdgeCategory
    factory: Callable[[BenignTemplateConfig, random.Random], list[PayloadVariant]]


def _signed_observed_dict(
    cfg: BenignTemplateConfig,
    *,
    msg_id: str,
    claim: str,
    answer: Any,
    round_idx: int = 1,
    performative: Performative = Performative.PROPOSE,
    constraints: list[str] | None = None,
) -> dict[str, Any]:
    msg = AgentMessage(
        msg_id=msg_id,
        sender="oracle_agent",
        recipient="verifier_b",
        round_idx=round_idx,
        performative=performative,
        claim=claim,
        answer=answer,
        confidence=1.0,
        provenance=Provenance.OBSERVED,
        constraints=list(constraints or []),
        evidence_refs=["oracle:ref-01"],
    )
    msg.oracle_sig = sign_observed_claim(msg, cfg.oracle_key)
    return msg.to_dict()


def _benign_variant(
    case_id: str,
    edge_cat: BenignEdgeCategory,
    description: str,
    payload: dict[str, Any],
    *extra_tags: str,
) -> PayloadVariant:
    return PayloadVariant(
        case_id=case_id,
        category=PayloadCategory.POSITIVE_CONTROL,
        description=description,
        expected_accepted=True,
        expected_error_code=None,
        payload=payload,
        tags=("benign_edge", edge_cat.value, *extra_tags),
    )


# ---------------------------------------------------------------------------
# Built-in Parameterized Benign Edge-Case Templates
# ---------------------------------------------------------------------------

def _unicode_locale_template(
    cfg: BenignTemplateConfig, rng: random.Random
) -> list[PayloadVariant]:
    nfc = unicodedata.normalize("NFC", "café naïve résumé")
    nfd = unicodedata.normalize("NFD", "café naïve résumé")

    samples: list[tuple[str, str, str, str]] = [
        ("nfc_composed", "Unicode NFC composed diacritics", f"Summary: {nfc}", "normalization"),
        ("nfd_decomposed", "Unicode NFD combining marks", f"Summary: {nfd}", "normalization"),
        (
            "emoji_zwj_graphemes",
            "Multi-codepoint ZWJ emoji where len(str) != grapheme count != byte count",
            "Team consensus: 👩‍🔬 + 👨‍💻 → ✅ (family: 👨‍👩‍👧‍👦)",
            "grapheme",
        ),
        (
            "multilingual_scripts",
            "Mixed Latin, CJK, Cyrillic, Greek, Devanagari, and Arabic (no bidi overrides)",
            "Check: 日本語 / 한국어 / Кириллица / Ελληνικά / हिन्दी / العربية",
            "multilingual",
        ),
        (
            "math_scientific_unicode",
            "Unicode scientific & mathematical notation",
            "Bound: ∀ε > 0, ∃δ > 0 s.t. |x − x₀| < δ ⇒ |f(x) − L| ≤ ε (10⁻⁶ μm)",
            "math",
        ),
        (
            "non_ascii_spaces_in_claim",
            "Valid non-breaking space (U+00A0) and thin space (U+2009) in units",
            "Measured speed: 299\u2009792\u2009458\u00a0m/s",
            "whitespace",
        ),
    ]
    return [
        _benign_variant(
            f"benign_unicode_{slug}",
            BenignEdgeCategory.UNICODE_LOCALE,
            desc,
            make_valid_derived(msg_id=f"msg-uni-{idx}", claim=text),
            tag,
        )
        for idx, (slug, desc, text, tag) in enumerate(samples, start=1)
    ]


def _numeric_precision_template(
    cfg: BenignTemplateConfig, rng: random.Random
) -> list[PayloadVariant]:
    variants: list[PayloadVariant] = []

    for idx, conf in enumerate(cfg.confidence_grid):
        variants.append(
            _benign_variant(
                f"benign_num_confidence_{idx}",
                BenignEdgeCategory.NUMERIC_PRECISION,
                f"Confidence boundary value {conf!r}",
                make_valid_derived(msg_id=f"msg-conf-{idx}", confidence=conf),
                "confidence",
            )
        )

    # Integer 0 and 1 are valid JSON numbers inside [0.0, 1.0]
    for int_conf in (0, 1):
        variants.append(
            _benign_variant(
                f"benign_num_confidence_int_{int_conf}",
                BenignEdgeCategory.NUMERIC_PRECISION,
                f"Integer confidence literal {int_conf} inside [0, 1]",
                make_valid_derived(msg_id=f"msg-conf-int-{int_conf}", confidence=int_conf),
                "confidence",
            )
        )

    for ridx in cfg.round_indices:
        variants.append(
            _benign_variant(
                f"benign_num_round_idx_{ridx}",
                BenignEdgeCategory.NUMERIC_PRECISION,
                f"Boundary non-negative round_idx={ridx}",
                make_valid_derived(msg_id=f"msg-rnd-{ridx}", round_idx=ridx),
                "round_idx",
            )
        )

    answer_edges: list[tuple[str, str, Any]] = [
        ("zero_int", "Integer zero answer", 0),
        ("negative_zero_float", "IEEE-754 signed negative zero (-0.0) in signed OBSERVED", -0.0),
        ("tiny_positive_float", "Small scientific notation float (1e-12)", 1e-12),
        ("js_max_safe_int", "IEEE-754 double exact integer ceiling (2^53 - 1)", (1 << 53) - 1),
        ("int64_max", "Signed 64-bit integer max (2^63 - 1)", (1 << 63) - 1),
        ("int64_min", "Signed 64-bit integer min (-2^63)", -(1 << 63)),
    ]
    for slug, desc, val in answer_edges:
        variants.append(
            _benign_variant(
                f"benign_num_answer_{slug}",
                BenignEdgeCategory.NUMERIC_PRECISION,
                desc,
                _signed_observed_dict(
                    cfg,
                    msg_id=f"msg-num-{slug}",
                    claim=f"Observed numeric value {val!r}",
                    answer=val,
                ),
                "numeric_answer",
                "hmac",
            )
        )

    return variants


def _length_depth_boundary_template(
    cfg: BenignTemplateConfig, rng: random.Random
) -> list[PayloadVariant]:
    variants: list[PayloadVariant] = [
        _benign_variant(
            "benign_len_min_1char_fields",
            BenignEdgeCategory.LENGTH_DEPTH_BOUNDARY,
            "Minimum valid 1-character identifiers and 1-character claim",
            make_valid_derived(msg_id="m", sender="a", recipient="b", claim="x"),
            "min_length",
        ),
        _benign_variant(
            "benign_len_exact_max_identifiers",
            BenignEdgeCategory.LENGTH_DEPTH_BOUNDARY,
            f"Identifiers at exact maximum length ({cfg.max_id_chars} chars) with all allowed punctuation",
            make_valid_derived(
                msg_id=("m_0.:-" * ((cfg.max_id_chars // 6) + 1))[: cfg.max_id_chars],
                sender=("s" * cfg.max_id_chars),
                recipient=("r" * cfg.max_id_chars),
            ),
            "exact_max_id",
        ),
    ]

    # Exact max_claim_chars (if it fits comfortably inside max_wire_bytes)
    overhead_reserve = 512
    if cfg.max_claim_chars + overhead_reserve <= cfg.max_wire_bytes:
        variants.append(
            _benign_variant(
                "benign_len_exact_max_claim_chars",
                BenignEdgeCategory.LENGTH_DEPTH_BOUNDARY,
                f"Claim at exact upper length limit ({cfg.max_claim_chars} chars)",
                make_valid_derived(
                    msg_id="msg-max-claim",
                    claim="c" * cfg.max_claim_chars,
                ),
                "exact_max_claim",
            )
        )

    # Exact max_nesting_depth: root dict is level 0; answer lists go down to remaining == 0
    if cfg.max_nesting_depth >= 2:
        nested: Any = "leaf"
        for _ in range(cfg.max_nesting_depth - 1):
            nested = [nested]
        variants.append(
            _benign_variant(
                "benign_depth_exact_max_nesting",
                BenignEdgeCategory.LENGTH_DEPTH_BOUNDARY,
                f"Nested answer at exact max_nesting_depth={cfg.max_nesting_depth}",
                make_valid_derived(msg_id="msg-max-depth", answer=nested),
                "exact_max_depth",
            )
        )

    return variants


def _string_escaping_template(
    cfg: BenignTemplateConfig, rng: random.Random
) -> list[PayloadVariant]:
    cases: list[tuple[str, str, str]] = [
        (
            "multiline_lf_crlf_tabs",
            "Legitimate multiline claim with LF, CRLF, and tab indentation (safe in NDJSON)",
            "Line 1:\n\tstep_a = 10\r\n\tstep_b = 32\nTotal = 42",
        ),
        (
            "quotes_and_backslashes",
            "Escaped quotes, Windows file paths, and regex backslashes inside claim",
            r'Path "C:\Users\research\data\run_01.json" matched regex ^\d+(\.\d+)?$',
        ),
        (
            "embedded_json_literal_text",
            "Serialized JSON snippet quoted as plain text inside claim (tests double-parse bugs)",
            'Upstream tool returned raw snippet: {"status": "ok", "count": 3, "items": [1, 2, 3]}',
        ),
        (
            "markdown_and_latex",
            "Markdown table and LaTeX math symbols inside claim",
            "| metric | value |\n|---|---|\n| F1 | $0.94 \\pm 0.01$ |",
        ),
        (
            "benign_angle_brackets",
            "Ordinary HTML/math angle brackets that are not chat-template special tokens",
            "Inequality check: x < y and y > z, tags like <result>42</result> are plain text",
        ),
    ]
    return [
        _benign_variant(
            f"benign_esc_{slug}",
            BenignEdgeCategory.STRING_ESCAPING,
            desc,
            make_valid_derived(msg_id=f"msg-esc-{idx}", claim=text),
            "escaping",
        )
        for idx, (slug, desc, text) in enumerate(cases, start=1)
    ]


def _collection_shape_template(
    cfg: BenignTemplateConfig, rng: random.Random
) -> list[PayloadVariant]:
    variants: list[PayloadVariant] = []

    for sz in cfg.list_sizes:
        variants.append(
            _benign_variant(
                f"benign_col_list_size_{sz}",
                BenignEdgeCategory.COLLECTION_SHAPE,
                f"Envelope with {sz} constraints and {sz} evidence_refs",
                make_valid_derived(
                    msg_id=f"msg-col-sz-{sz}",
                    constraints=[f"constraint_{i}" for i in range(sz)],
                    evidence_refs=[f"ref_{i}" for i in range(sz)],
                ),
                "list_cardinality",
            )
        )

    answer_shapes: list[tuple[str, str, Any]] = [
        ("null", "Explicit JSON null answer (e.g., abstention)", None),
        ("bool_false", "Boolean False answer", False),
        ("bool_true", "Boolean True answer", True),
        ("empty_string", "Empty string answer", ""),
        ("empty_list", "Empty list answer", []),
        ("empty_object", "Empty JSON object answer", {}),
        (
            "punctuated_dict_keys",
            "Answer object with dotted/spaced/Unicode keys",
            {"metric.v1": 0.95, "velocity (m/s)": 9.81, "Δt": 0.01, "nested": {"ok": True}},
        ),
    ]
    for slug, desc, ans in answer_shapes:
        variants.append(
            _benign_variant(
                f"benign_col_answer_{slug}",
                BenignEdgeCategory.COLLECTION_SHAPE,
                desc,
                _signed_observed_dict(
                    cfg,
                    msg_id=f"msg-shape-{slug}",
                    claim=f"Observed structured answer ({slug})",
                    answer=ans,
                ),
                "answer_shape",
                "hmac",
            )
        )

    return variants


def _protocol_matrix_template(
    cfg: BenignTemplateConfig, rng: random.Random
) -> list[PayloadVariant]:
    """Parameterized cross-product over Performative x Provenance."""
    variants: list[PayloadVariant] = []
    for perf in Performative:
        for prov in Provenance:
            case_id = f"benign_matrix_{perf.value}_{prov.value}"
            if prov == Provenance.OBSERVED:
                payload = _signed_observed_dict(
                    cfg,
                    msg_id=f"msg-mx-{perf.value}-{prov.value}",
                    claim=f"Observed {perf.value} message",
                    answer={"performative": perf.value, "verified": True},
                    performative=perf,
                    constraints=["deterministic_check"],
                )
            else:
                payload = make_valid_derived(
                    msg_id=f"msg-mx-{perf.value}-{prov.value}",
                    performative=perf.value,
                    provenance=prov.value,
                    claim=f"{prov.value.capitalize()} {perf.value} message",
                    oracle_sig=None,
                )
            variants.append(
                _benign_variant(
                    case_id,
                    BenignEdgeCategory.PROTOCOL_MATRIX,
                    f"Valid combination performative={perf.value}, provenance={prov.value}",
                    payload,
                    perf.value,
                    prov.value,
                )
            )
    return variants


BUILTIN_BENIGN_TEMPLATES: tuple[BenignTemplate, ...] = (
    BenignTemplate("unicode_locale", BenignEdgeCategory.UNICODE_LOCALE, _unicode_locale_template),
    BenignTemplate("numeric_precision", BenignEdgeCategory.NUMERIC_PRECISION, _numeric_precision_template),
    BenignTemplate("length_depth_boundary", BenignEdgeCategory.LENGTH_DEPTH_BOUNDARY, _length_depth_boundary_template),
    BenignTemplate("string_escaping", BenignEdgeCategory.STRING_ESCAPING, _string_escaping_template),
    BenignTemplate("collection_shape", BenignEdgeCategory.COLLECTION_SHAPE, _collection_shape_template),
    BenignTemplate("protocol_matrix", BenignEdgeCategory.PROTOCOL_MATRIX, _protocol_matrix_template),
)


class BenignEdgeGenerator:
    """Generates parameterized benign edge-case variants for false-reject & boundary testing."""

    def __init__(self, config: BenignTemplateConfig | None = None) -> None:
        self.config = config or BenignTemplateConfig()
        self._templates: list[BenignTemplate] = list(BUILTIN_BENIGN_TEMPLATES)

    def register_template(self, template: BenignTemplate) -> None:
        self._templates.append(template)

    def generate(self) -> list[PayloadVariant]:
        rng = random.Random(self.config.seed)
        selected = set(self.config.categories) if self.config.categories is not None else None
        results: list[PayloadVariant] = []
        seen_ids: set[str] = set()

        for tmpl in self._templates:
            if selected is not None and tmpl.edge_category not in selected:
                continue
            for variant in tmpl.factory(self.config, rng):
                if variant.case_id in seen_ids:
                    raise ValueError(f"Duplicate benign case_id: {variant.case_id!r}")
                seen_ids.add(variant.case_id)
                results.append(variant)

        return results

    def as_payload_templates(self) -> list[PayloadTemplate]:
        """Adapt benign templates for direct registration inside PayloadGenerator."""
        selected = set(self.config.categories) if self.config.categories is not None else None
        adapted: list[PayloadTemplate] = []
        for tmpl in self._templates:
            if selected is not None and tmpl.edge_category not in selected:
                continue

            def _make_adapter(
                inner: BenignTemplate,
            ) -> Callable[[PayloadConfig, random.Random], list[PayloadVariant]]:
                def _adapter(payload_cfg: PayloadConfig, rng: random.Random) -> list[PayloadVariant]:
                    derived_cfg = BenignTemplateConfig(
                        oracle_key=payload_cfg.oracle_key,
                        framing=payload_cfg.framing,
                        max_wire_bytes=payload_cfg.max_wire_bytes,
                        max_claim_chars=payload_cfg.max_claim_chars,
                        max_nesting_depth=payload_cfg.max_nesting_depth,
                        max_id_chars=self.config.max_id_chars,
                        list_sizes=self.config.list_sizes,
                        round_indices=self.config.round_indices,
                        confidence_grid=self.config.confidence_grid,
                        seed=payload_cfg.seed,
                        categories=self.config.categories,
                    )
                    return inner.factory(derived_cfg, rng)

                return _adapter

            adapted.append(
                PayloadTemplate(
                    name=f"benign_{tmpl.name}",
                    category=PayloadCategory.POSITIVE_CONTROL,
                    factory=_make_adapter(tmpl),
                )
            )
        return adapted
