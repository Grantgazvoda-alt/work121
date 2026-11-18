"""Adversarial robustness test suite for cross-model agent communication parsers."""

from .benign_templates import (
    BUILTIN_BENIGN_TEMPLATES,
    BenignEdgeCategory,
    BenignEdgeGenerator,
    BenignTemplate,
    BenignTemplateConfig,
)
from .encoder import EncodedMessage, MessageEncoder
from .payloads import (
    BUILTIN_TEMPLATES,
    PayloadCategory,
    PayloadConfig,
    PayloadGenerator,
    PayloadTemplate,
    PayloadVariant,
    WireMutation,
    make_valid_derived,
    make_valid_observed,
)
from .protocol import (
    AgentMessage,
    FramingMode,
    HardenedBoundaryGate,
    ParseResult,
    Performative,
    Provenance,
    WireParser,
    sign_observed_claim,
)
from .runner import CaseExecutionRecord, SuiteRunner, SuiteSummary, format_report

__all__ = [
    "AgentMessage",
    "BUILTIN_BENIGN_TEMPLATES",
    "BUILTIN_TEMPLATES",
    "BenignEdgeCategory",
    "BenignEdgeGenerator",
    "BenignTemplate",
    "BenignTemplateConfig",
    "CaseExecutionRecord",
    "EncodedMessage",
    "FramingMode",
    "HardenedBoundaryGate",
    "MessageEncoder",
    "ParseResult",
    "PayloadCategory",
    "PayloadConfig",
    "PayloadGenerator",
    "PayloadTemplate",
    "PayloadVariant",
    "Performative",
    "Provenance",
    "SuiteRunner",
    "SuiteSummary",
    "WireMutation",
    "WireParser",
    "format_report",
    "make_valid_derived",
    "make_valid_observed",
    "sign_observed_claim",
]
