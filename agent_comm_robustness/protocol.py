from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import struct
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Protocol


class Performative(str, Enum):
    PROPOSE = "propose"
    CRITIQUE = "critique"
    VERIFY = "verify"
    REVISE = "revise"
    ABSTAIN = "abstain"


class Provenance(str, Enum):
    OBSERVED = "observed"  # Tool or deterministic oracle output
    DERIVED = "derived"    # Model inference from prior evidence
    ASSUMED = "assumed"    # Unverified hypothesis / heuristic


class FramingMode(str, Enum):
    RAW = "raw"
    LENGTH_PREFIXED = "length_prefixed"  # 4-byte big-endian uint32 length + body
    NDJSON = "ndjson"                    # Single UTF-8 JSON line terminated by \n


REQUIRED_FIELDS = frozenset({
    "msg_id",
    "sender",
    "recipient",
    "round_idx",
    "performative",
    "claim",
    "answer",
    "confidence",
    "provenance",
})
ALLOWED_FIELDS = REQUIRED_FIELDS | frozenset({
    "constraints",
    "evidence_refs",
    "oracle_sig",
})

HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
FORBIDDEN_CTRL_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u202a-\u202e\u2066-\u2069]"
)
SPECIAL_TOKEN_RE = re.compile(
    r"(<\|[^|]{1,32}\|>|\[/?INST\]|</?s>|<<SYS>>|<</SYS>>)"
)
UTF8_BOM = b"\xef\xbb\xbf"


@dataclass
class AgentMessage:
    msg_id: str
    sender: str
    recipient: str
    round_idx: int
    performative: Performative
    claim: str
    answer: Any
    confidence: float
    provenance: Provenance = Provenance.DERIVED
    constraints: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    oracle_sig: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["performative"] = self.performative.value
        payload["provenance"] = self.provenance.value
        return payload

    def to_wire_str(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)


def sign_observed_claim(msg: AgentMessage, secret_key: bytes) -> str:
    """Compute a domain-separated HMAC-SHA256 binding content and envelope context."""
    canonical = json.dumps(
        {
            "domain": "agent-protocol-v1:observed",
            "msg_id": msg.msg_id,
            "round_idx": msg.round_idx,
            "provenance": msg.provenance.value,
            "claim": msg.claim,
            "answer": msg.answer,
            "answer_type": type(msg.answer).__name__,
            "constraints": msg.constraints,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hmac.new(secret_key, canonical, hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class ParseResult:
    accepted: bool
    message: AgentMessage | None = None
    error_code: str | None = None
    error_detail: str | None = None


class WireParser(Protocol):
    """Pluggable interface so any parser or framework gate can be benchmarked."""

    def parse_bytes(self, raw: bytes) -> ParseResult: ...


class _DuplicateKeyError(ValueError):
    pass


class _NonFiniteConstantError(ValueError):
    pass


class HardenedBoundaryGate:
    """Reference parser & cryptographic boundary validator for raw wire byte streams."""

    def __init__(
        self,
        oracle_key: bytes | None = None,
        *,
        framing: FramingMode = FramingMode.RAW,
        require_signed_observed: bool = True,
        max_wire_bytes: int = 32_768,
        max_claim_chars: int = 8_192,
        max_nesting_depth: int = 8,
        reject_special_tokens: bool = True,
    ) -> None:
        self.oracle_key = oracle_key
        self.framing = framing
        self.require_signed_observed = require_signed_observed
        self.max_wire_bytes = max_wire_bytes
        self.max_claim_chars = max_claim_chars
        self.max_nesting_depth = max_nesting_depth
        self.reject_special_tokens = reject_special_tokens

    @staticmethod
    def _reject(code: str, detail: str) -> ParseResult:
        return ParseResult(accepted=False, message=None, error_code=code, error_detail=detail)

    def _unframe(self, raw: bytes) -> tuple[bytes | None, ParseResult | None]:
        if self.framing == FramingMode.RAW:
            return raw, None

        if self.framing == FramingMode.LENGTH_PREFIXED:
            if len(raw) < 4:
                return None, self._reject(
                    "frame_header_truncated",
                    f"expected 4-byte length prefix, got {len(raw)} bytes",
                )
            (declared_len,) = struct.unpack(">I", raw[:4])
            if declared_len == 0:
                return None, self._reject("empty_wire_bytes", "declared frame length is 0")
            if declared_len > self.max_wire_bytes:
                return None, self._reject(
                    "payload_too_large",
                    f"declared frame length {declared_len} exceeds cap {self.max_wire_bytes}",
                )
            body = raw[4:]
            if len(body) != declared_len:
                return None, self._reject(
                    "frame_length_mismatch",
                    f"declared {declared_len} bytes, actual body is {len(body)} bytes",
                )
            return body, None

        if self.framing == FramingMode.NDJSON:
            if not raw.endswith(b"\n"):
                return None, self._reject(
                    "frame_missing_delimiter",
                    "NDJSON frame must terminate with a newline byte",
                )
            body = raw[:-1]
            if b"\n" in body or b"\r" in body:
                return None, self._reject(
                    "frame_embedded_newline",
                    "NDJSON frame contains embedded CR/LF delimiter",
                )
            return body, None

        return None, self._reject("unsupported_framing", f"unknown framing: {self.framing}")

    def parse_bytes(self, raw: bytes) -> ParseResult:
        # 1. Byte-stream presence & framing validation
        if not isinstance(raw, (bytes, bytearray)):
            return self._reject("invalid_wire_type", f"expected bytes, got {type(raw).__name__}")
        raw_bytes = bytes(raw)
        if not raw_bytes:
            return self._reject("empty_wire_bytes", "empty byte stream")

        body, frame_err = self._unframe(raw_bytes)
        if frame_err is not None:
            return frame_err
        assert body is not None

        if not body:
            return self._reject("empty_wire_bytes", "empty frame body")
        if len(body) > self.max_wire_bytes:
            return self._reject(
                "payload_too_large",
                f"body size {len(body)} exceeds max_wire_bytes {self.max_wire_bytes}",
            )
        if body.startswith(UTF8_BOM):
            return self._reject("utf8_bom_forbidden", "UTF-8 BOM prefix is not permitted")

        # 2. Strict UTF-8 decoding
        try:
            text = body.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            return self._reject("invalid_utf8_bytes", str(exc))

        if not text.strip():
            return self._reject("empty_wire_text", "decoded payload is whitespace-only")

        # 3. Strict JSON decoding
        def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for k, v in pairs:
                if k in out:
                    raise _DuplicateKeyError(f"duplicate JSON key: {k!r}")
                out[k] = v
            return out

        def _reject_non_finite(token: str) -> Any:
            raise _NonFiniteConstantError(f"non-finite JSON constant: {token}")

        try:
            data = json.loads(
                text,
                object_pairs_hook=_no_duplicate_keys,
                parse_constant=_reject_non_finite,
            )
        except _DuplicateKeyError as exc:
            return self._reject("duplicate_json_key", str(exc))
        except _NonFiniteConstantError as exc:
            return self._reject("non_finite_json_constant", str(exc))
        except Exception as exc:
            return self._reject("json_parse_error", str(exc))

        if not isinstance(data, dict):
            return self._reject(
                "root_must_be_object",
                f"JSON root must be an object, got {type(data).__name__}",
            )

        # 4. Recursive depth & UTF-16 surrogate validation across the decoded tree
        tree_err = self._validate_tree(data, self.max_nesting_depth)
        if tree_err is not None:
            return tree_err

        # 5. Schema membership checks
        keys = set(data.keys())
        missing = REQUIRED_FIELDS - keys
        if missing:
            return self._reject(
                "missing_required_fields",
                f"missing required fields: {sorted(missing)}",
            )
        extra = keys - ALLOWED_FIELDS
        if extra:
            return self._reject(
                "unexpected_fields",
                f"unexpected fields: {sorted(extra)}",
            )

        # 6. Field types, value domains, and token boundaries
        for id_field in ("msg_id", "sender", "recipient"):
            val = data[id_field]
            if not isinstance(val, str) or not IDENTIFIER_RE.match(val):
                return self._reject("invalid_identifier", f"invalid identifier for {id_field}")

        round_idx = data["round_idx"]
        if isinstance(round_idx, bool) or not isinstance(round_idx, int) or round_idx < 0:
            return self._reject("invalid_round_idx", f"invalid round_idx: {round_idx!r}")

        conf = data["confidence"]
        if (
            isinstance(conf, bool)
            or not isinstance(conf, (int, float))
            or not math.isfinite(conf)
            or not (0.0 <= float(conf) <= 1.0)
        ):
            return self._reject("invalid_confidence", f"invalid confidence: {conf!r}")

        try:
            performative = Performative(data["performative"])
            provenance = Provenance(data["provenance"])
        except Exception as exc:
            return self._reject("invalid_enum", str(exc))

        claim = data["claim"]
        if not isinstance(claim, str) or not claim.strip():
            return self._reject("invalid_or_empty_claim", "claim must be a non-empty string")
        if len(claim) > self.max_claim_chars:
            return self._reject(
                "claim_too_long",
                f"claim length {len(claim)} exceeds max_claim_chars {self.max_claim_chars}",
            )
        if text_err := self._check_text_boundaries(claim, "claim"):
            return text_err

        constraints = data.get("constraints", [])
        if not isinstance(constraints, list) or not all(isinstance(c, str) for c in constraints):
            return self._reject("invalid_constraints_type", "constraints must be list[str]")
        for c in constraints:
            if text_err := self._check_text_boundaries(c, "constraint"):
                return text_err

        evidence_refs = data.get("evidence_refs", [])
        if not isinstance(evidence_refs, list) or not all(isinstance(r, str) for r in evidence_refs):
            return self._reject("invalid_evidence_refs_type", "evidence_refs must be list[str]")
        for ref in evidence_refs:
            if text_err := self._check_text_boundaries(ref, "evidence_ref"):
                return text_err

        oracle_sig = data.get("oracle_sig")
        if oracle_sig is not None and not isinstance(oracle_sig, str):
            return self._reject(
                "invalid_oracle_sig_type",
                f"oracle_sig must be str or null, got {type(oracle_sig).__name__}",
            )

        answer = data["answer"]
        if isinstance(answer, str):
            if text_err := self._check_text_boundaries(answer, "answer"):
                return text_err

        msg = AgentMessage(
            msg_id=data["msg_id"],
            sender=data["sender"],
            recipient=data["recipient"],
            round_idx=round_idx,
            performative=performative,
            claim=claim,
            answer=answer,
            confidence=float(conf),
            provenance=provenance,
            constraints=constraints,
            evidence_refs=evidence_refs,
            oracle_sig=oracle_sig,
        )

        # 7. Cryptographic provenance validation
        if msg.provenance == Provenance.OBSERVED and self.require_signed_observed:
            if not self.oracle_key:
                return self._reject("oracle_key_unconfigured", "oracle key is not configured")
            if not msg.oracle_sig:
                return self._reject(
                    "missing_oracle_signature",
                    "observed provenance requires oracle_sig",
                )
            if not HEX_SHA256_RE.match(msg.oracle_sig):
                return self._reject(
                    "malformed_oracle_signature_format",
                    "oracle_sig must be a 64-char lowercase hex SHA-256 digest",
                )
            expected = sign_observed_claim(msg, self.oracle_key)
            if not hmac.compare_digest(expected, msg.oracle_sig):
                return self._reject(
                    "invalid_oracle_signature",
                    "HMAC signature verification failed",
                )
        elif msg.oracle_sig is not None:
            return self._reject(
                "unexpected_signature_on_non_observed",
                f"oracle_sig is only allowed on observed provenance, got {msg.provenance.value}",
            )

        return ParseResult(accepted=True, message=msg, error_code=None, error_detail=None)

    def _validate_tree(self, obj: Any, remaining: int) -> ParseResult | None:
        if remaining < 0:
            return self._reject(
                "max_nesting_depth_exceeded",
                f"JSON nesting exceeded limit {self.max_nesting_depth}",
            )
        if isinstance(obj, str):
            try:
                obj.encode("utf-8")
            except UnicodeEncodeError as exc:
                return self._reject("invalid_surrogate_in_payload", str(exc))
            return None
        if isinstance(obj, dict):
            for k, v in obj.items():
                if err := self._validate_tree(k, remaining - 1):
                    return err
                if err := self._validate_tree(v, remaining - 1):
                    return err
        elif isinstance(obj, list):
            for item in obj:
                if err := self._validate_tree(item, remaining - 1):
                    return err
        return None

    def _check_text_boundaries(self, text: str, field_name: str) -> ParseResult | None:
        if FORBIDDEN_CTRL_RE.search(text):
            return self._reject(
                "forbidden_control_or_bidi_char",
                f"forbidden control or bidi character in {field_name}",
            )
        if self.reject_special_tokens and SPECIAL_TOKEN_RE.search(text):
            return self._reject(
                "special_token_delimiter",
                f"chat-template special token delimiter in {field_name}",
            )
        return None
