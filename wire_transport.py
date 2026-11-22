"""Real-transport TargetAgent over the work122 agent wire suite.

Each user turn is one TCP exchange. The request and the response are framed
(raw, length-prefixed, or NDJSON), schema-checked, and signed with the
domain-separated HMAC from ``sign_observed_claim``. Verification is the suite's
``HardenedBoundaryGate``, which compares digests with ``hmac.compare_digest``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import re
import secrets
import socket
import threading
import time
from typing import Any

from boundary_oracle import BoundaryDecision, BoundaryOracle
from oracle_keys import load_oracle_key, validate_oracle_key
from regression_tracker import ResponseTokenMetadata
from template_library import RenderedTestCase
from wire_suite import load_suite, suite_version

logger = logging.getLogger("work121.wire_transport")

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_READ_CAP = 256_000
_SOCKET_TIMEOUT_S = 10.0

_suite = load_suite()
FramingMode = _suite.FramingMode
Performative = _suite.Performative
Provenance = _suite.Provenance


class TransportError(RuntimeError):
    """A framed request or response failed schema or HMAC verification."""

    def __init__(self, message: str, *, error_code: str | None = None) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class _ExchangeResult:
    assistant_text: str
    alignment: float


def encode_message(message: Any, framing: Any) -> bytes:
    """Frame an ``AgentMessage`` with the suite's ``MessageEncoder``."""
    suite = load_suite()
    variant = suite.PayloadVariant(
        case_id=message.msg_id,
        category=suite.PayloadCategory.POSITIVE_CONTROL,
        description="evaluation transport frame",
        expected_accepted=True,
        expected_error_code=None,
        payload=message.to_dict(),
    )
    return suite.MessageEncoder(framing).encode(variant).raw_bytes


def parse_message(
    raw: bytes,
    *,
    framing: Any,
    oracle_key: bytes,
    replay_window: Any | None = None,
    max_clock_skew_seconds: float | None = None,
) -> Any:
    """Parse one frame with ``HardenedBoundaryGate`` (schema + HMAC)."""
    suite = load_suite()
    gate_kwargs: dict[str, Any] = {
        "oracle_key": oracle_key,
        "framing": framing,
        "require_signed_observed": True,
    }
    if replay_window is not None:
        gate_kwargs["replay_window"] = replay_window
    if max_clock_skew_seconds is not None:
        gate_kwargs["max_clock_skew_seconds"] = max_clock_skew_seconds
    gate = suite.HardenedBoundaryGate(**gate_kwargs)
    return gate.parse_bytes(raw)


def sign_message(message: Any, oracle_key: bytes) -> Any:
    """JSON-roundtrip the envelope, then attach the suite's observed HMAC."""
    suite = load_suite()
    frozen = _freeze(message)
    frozen.oracle_sig = suite.sign_observed_claim(frozen, oracle_key)
    return frozen


def _freeze(message: Any) -> Any:
    payload = message.to_dict()
    payload["oracle_sig"] = None
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
    loaded = json.loads(encoded)
    loaded["oracle_sig"] = None
    return _message_from_payload(loaded)


def _build_agent_message(suite: Any, **kwargs: Any) -> Any:
    msg_fields = getattr(suite.AgentMessage, "__dataclass_fields__", {})
    for optional_field in ("nonce", "issued_at"):
        if optional_field not in msg_fields:
            kwargs.pop(optional_field, None)
    return suite.AgentMessage(**kwargs)


def _new_replay_window(suite: Any) -> Any | None:
    window_cls = getattr(suite, "ReplayWindow", None)
    return window_cls() if window_cls is not None else None


def _message_from_payload(payload: dict[str, Any]) -> Any:
    suite = load_suite()
    return _build_agent_message(
        suite,
        msg_id=payload["msg_id"],
        sender=payload["sender"],
        recipient=payload["recipient"],
        round_idx=payload["round_idx"],
        performative=suite.Performative(payload["performative"]),
        claim=payload["claim"],
        answer=payload["answer"],
        confidence=float(payload["confidence"]),
        provenance=suite.Provenance(payload["provenance"]),
        constraints=list(payload.get("constraints") or []),
        evidence_refs=list(payload.get("evidence_refs") or []),
        oracle_sig=payload.get("oracle_sig"),
        nonce=payload.get("nonce"),
        issued_at=payload.get("issued_at"),
    )


def _require_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must match {_IDENTIFIER_RE.pattern} (got {value!r})"
        )
    return value


def tokenize_response(prompt: str, completion: str) -> ResponseTokenMetadata:
    """Whitespace tokenizer with SHA-256 token ids derived from the token text."""
    prompt_tokens = tuple(prompt.split())
    completion_tokens = tuple(completion.split())
    if not completion_tokens:
        raise TransportError("Assistant response did not contain any tokens")
    token_ids = tuple(
        int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "big")
        for token in completion_tokens
    )
    return ResponseTokenMetadata(
        tokens=completion_tokens,
        token_ids=token_ids,
        prompt_token_count=len(prompt_tokens),
        completion_token_count=len(completion_tokens),
        tokenizer_name="whitespace-sha256-v1",
    )


class LoopbackWirePeer:
    """Local TCP peer that speaks the wire suite and answers with the boundary oracle."""

    def __init__(
        self,
        *,
        oracle_key: bytes,
        framing: Any,
        policy: BoundaryOracle,
        model_id: str,
    ) -> None:
        self._oracle_key = oracle_key
        self._framing = framing
        self._policy = policy
        self._model_id = model_id
        self._replay_lock = threading.Lock()
        self._replay_window = _new_replay_window(load_suite())
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._closed = False
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(32)
        self._sock.settimeout(0.2)
        self.port = int(self._sock.getsockname()[1])
        self._thread = threading.Thread(
            target=self._serve,
            name=f"wire-peer-{self.port}",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=2):
            self.close()
            raise TransportError("Loopback peer failed to start")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            logger.warning("failed to close loopback listening socket", exc_info=True)
        self._thread.join(timeout=2)

    def _serve(self) -> None:
        self._ready.set()
        while not self._stop.is_set():
            try:
                conn, _addr = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            worker = threading.Thread(target=self._handle, args=(conn,), daemon=True)
            worker.start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            raw = _read_bounded(conn)
            with self._replay_lock:
                parsed = parse_message(
                    raw,
                    framing=self._framing,
                    oracle_key=self._oracle_key,
                    replay_window=self._replay_window,
                )
            if not parsed.accepted or parsed.message is None:
                decision = BoundaryDecision(
                    assistant_text="",
                    alignment=0.0,
                    pressure=1.0,
                    abstain_code=parsed.error_code or "rejected",
                    abstain_detail=parsed.error_detail or "frame rejected",
                )
                reply = _decision_message(
                    decision,
                    model_id=self._model_id,
                    round_idx=0,
                    in_reply_to="rejected",
                    request=None,
                )
            else:
                request = parsed.message
                decision = self._policy.decide_request(
                    boundary=request.claim,
                    answer=request.answer,
                    turn_index=int(request.round_idx),
                )
                reply = _decision_message(
                    decision,
                    model_id=self._model_id,
                    round_idx=int(request.round_idx),
                    in_reply_to=request.msg_id,
                    request=request,
                )
            signed = sign_message(reply, self._oracle_key)
            conn.sendall(encode_message(signed, self._framing))
            logger.info(
                "peer replied msg_id=%s performative=%s",
                signed.msg_id,
                signed.performative.value,
            )
        except Exception:
            logger.exception("loopback peer failed while answering a frame")
        finally:
            try:
                conn.close()
            except OSError:
                logger.warning("failed to close a peer connection", exc_info=True)


class WireTargetAgent:
    """TargetAgent that exchanges signed frames with a wire peer."""

    def __init__(
        self,
        *,
        model_id: str,
        model_version: str,
        architecture_family: str,
        framing: Any,
        oracle_key: bytes,
        host: str,
        port: int,
        peer: LoopbackWirePeer | None,
        sensitivity: float | None,
        drift_per_turn: float | None,
    ) -> None:
        self._model_id = _require_identifier(model_id, "model_id")
        self._model_version = _require_identifier(model_version, "model_version")
        if not isinstance(architecture_family, str) or not architecture_family.strip():
            raise ValueError("architecture_family must be a non-empty string")
        if len(architecture_family) > 128:
            raise ValueError("architecture_family must be at most 128 characters")
        if not isinstance(host, str) or not host.strip() or any(ch.isspace() for ch in host):
            raise ValueError("host must be a non-empty hostname or address without spaces")
        if isinstance(port, bool) or not isinstance(port, int) or not (1 <= port <= 65535):
            raise ValueError("port must be an integer from 1 to 65535")
        self._architecture_family = architecture_family.strip()
        self._framing = framing
        self._oracle_key = validate_oracle_key(bytes(oracle_key))
        self._host = host
        self._port = port
        self._peer = peer
        self._sensitivity = sensitivity
        self._drift_per_turn = drift_per_turn
        self._replay_window = _new_replay_window(load_suite())

    @classmethod
    def listen(
        cls,
        *,
        model_id: str,
        model_version: str,
        architecture_family: str,
        framing: Any,
        sensitivity: float,
        drift_per_turn: float,
        oracle_key: bytes | None = None,
    ) -> WireTargetAgent:
        """Start a loopback peer and return the client bound to it."""
        key = load_oracle_key() if oracle_key is None else oracle_key
        if oracle_key is not None and len(oracle_key) < 16:
            raise ValueError("oracle_key must be at least 16 bytes")
        policy = BoundaryOracle(sensitivity=sensitivity, drift_per_turn=drift_per_turn)
        peer = LoopbackWirePeer(
            oracle_key=key,
            framing=framing,
            policy=policy,
            model_id=_require_identifier(model_id, "model_id"),
        )
        try:
            return cls(
                model_id=model_id,
                model_version=model_version,
                architecture_family=architecture_family,
                framing=framing,
                oracle_key=key,
                host="127.0.0.1",
                port=peer.port,
                peer=peer,
                sensitivity=float(sensitivity),
                drift_per_turn=float(drift_per_turn),
            )
        except Exception:
            peer.close()
            raise

    @classmethod
    def connect(
        cls,
        *,
        model_id: str,
        model_version: str,
        architecture_family: str,
        framing: Any,
        host: str,
        port: int,
        oracle_key: bytes | None = None,
    ) -> WireTargetAgent:
        """Connect to an already-running peer. The key is loaded when omitted."""
        key = load_oracle_key() if oracle_key is None else oracle_key
        return cls(
            model_id=model_id,
            model_version=model_version,
            architecture_family=architecture_family,
            framing=framing,
            oracle_key=key,
            host=host,
            port=port,
            peer=None,
            sensitivity=None,
            drift_per_turn=None,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def architecture_family(self) -> str:
        return self._architecture_family

    @property
    def framing(self) -> Any:
        return self._framing

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    def describe_transport(self) -> dict[str, object]:
        """JSON-safe transport description. Never includes key material."""
        info: dict[str, object] = {
            "distribution": "agent-comm-robustness",
            "distribution_version": suite_version(),
            "framing": self._framing.value,
            "host": self._host,
            "port": self._port,
            "loopback": self._peer is not None,
        }
        if self._sensitivity is not None and self._drift_per_turn is not None:
            info["sensitivity"] = self._sensitivity
            info["drift_per_turn"] = self._drift_per_turn
        return info

    def close(self) -> None:
        if self._peer is not None:
            self._peer.close()

    def __enter__(self) -> WireTargetAgent:
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.close()
        return False

    def __repr__(self) -> str:
        return (
            f"WireTargetAgent({self._model_id}@{self._model_version}, "
            f"{self._framing.value}, {self._host}:{self._port})"
        )

    def run_case(self, test_case: RenderedTestCase) -> Any:
        from eval_runner import AgentResponse

        if not test_case.user_turns:
            raise ValueError(f"Test case {test_case.test_case_id} has no user turns")
        assistant_turns = []
        prior: list[str] = []
        for index, turn in enumerate(test_case.user_turns):
            exchange = self._exchange_turn(test_case, index, turn, prior)
            from response_classifier import TurnRecord

            assistant_turns.append(
                TurnRecord(
                    turn_index=index,
                    role="assistant",
                    content=exchange.assistant_text,
                    persona_alignment_score=exchange.alignment,
                )
            )
            prior.append(exchange.assistant_text)
        final_text = assistant_turns[-1].content
        return AgentResponse(
            assistant_turns=tuple(assistant_turns),
            response_tokens=tokenize_response(test_case.rendered_payload, final_text),
        )

    def _exchange_turn(
        self,
        test_case: RenderedTestCase,
        index: int,
        turn: str,
        prior: list[str],
    ) -> _ExchangeResult:
        nonce = secrets.token_hex(16)
        msg_id = f"req-{nonce}"
        evidence = (
            test_case.test_case_id
            if _IDENTIFIER_RE.fullmatch(test_case.test_case_id)
            else f"case-{nonce}"
        )
        suite = load_suite()
        now = int(time.time())
        request = _build_agent_message(
            suite,
            msg_id=msg_id,
            sender="eval-runner",
            recipient=self._model_id,
            round_idx=index,
            performative=Performative.PROPOSE,
            claim=test_case.system_boundary,
            answer={
                "user_turn": turn,
                "prior_assistant": list(prior),
                "test_case_id": test_case.test_case_id,
                "exchange_nonce": nonce,
            },
            confidence=1.0,
            provenance=Provenance.OBSERVED,
            constraints=["benign-validation", f"exchange-nonce:{nonce}"],
            evidence_refs=[evidence],
            nonce=nonce,
            issued_at=now,
        )
        signed = sign_message(request, self._oracle_key)
        if self._replay_window is not None:
            replay_token_fn = getattr(suite, "replay_token", None)
            token = replay_token_fn(signed) if callable(replay_token_fn) else f"nonce:{nonce}"
            if token is not None:
                self._replay_window.remember(token)
        raw_request = encode_message(signed, self._framing)
        raw_response = _transact(self._host, self._port, raw_request)
        parsed = parse_message(
            raw_response,
            framing=self._framing,
            oracle_key=self._oracle_key,
            replay_window=self._replay_window,
        )
        if not parsed.accepted or parsed.message is None:
            raise TransportError(
                f"Peer frame rejected: {parsed.error_code}: {parsed.error_detail}",
                error_code=parsed.error_code,
            )
        message = parsed.message
        _require_reply_binding(
            message,
            nonce=nonce,
            request_id=msg_id,
            round_idx=index,
            peer_id=self._model_id,
        )
        if message.performative != Performative.VERIFY:
            detail = message.answer if isinstance(message.answer, dict) else {}
            code = str(detail.get("error_code") or message.performative.value)
            raise TransportError(
                f"Peer abstained ({code}): {detail.get('error_detail', '')}",
                error_code=code,
            )
        answer = message.answer
        if not isinstance(answer, dict):
            raise TransportError("Verified peer answer is not an object", error_code="bad_answer")
        text = answer.get("assistant_text")
        score = answer.get("persona_alignment_score")
        if not isinstance(text, str) or not text.strip():
            raise TransportError("Verified peer answer is missing assistant_text", error_code="bad_answer")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TransportError(
                "Verified peer answer is missing persona_alignment_score",
                error_code="bad_answer",
            )
        alignment = float(score)
        if not 0.0 <= alignment <= 1.0:
            raise TransportError("persona_alignment_score outside [0, 1]", error_code="bad_answer")
        return _ExchangeResult(assistant_text=text, alignment=alignment)


def _reply_constraints(request: Any | None, in_reply_to: str) -> list[str]:
    constraints = ["observed-policy", f"in-reply-to:{in_reply_to}"]
    if request is None:
        return constraints
    for item in request.constraints or []:
        if isinstance(item, str) and item.startswith("exchange-nonce:"):
            constraints.append(item)
            break
    return constraints


def _require_reply_binding(
    message: Any,
    *,
    nonce: str,
    request_id: str,
    round_idx: int,
    peer_id: str,
) -> None:
    """Reject a signed reply that is not bound to this exchange."""
    if message.sender != peer_id:
        raise TransportError(
            f"Reply sender {message.sender!r} does not match peer {peer_id!r}.",
            error_code="reply_sender_mismatch",
        )
    if message.recipient != "eval-runner":
        raise TransportError(
            f"Reply recipient {message.recipient!r} is not eval-runner.",
            error_code="reply_recipient_mismatch",
        )
    if int(message.round_idx) != round_idx:
        raise TransportError(
            f"Reply round {message.round_idx} does not match request round {round_idx}.",
            error_code="reply_round_mismatch",
        )
    constraints = {item for item in (message.constraints or []) if isinstance(item, str)}
    if f"exchange-nonce:{nonce}" not in constraints:
        raise TransportError(
            "Reply is not bound to this exchange nonce.",
            error_code="reply_nonce_mismatch",
        )
    if f"in-reply-to:{request_id}" not in constraints:
        raise TransportError(
            "Reply is not bound to this request id.",
            error_code="reply_not_bound",
        )


def _decision_message(
    decision: BoundaryDecision,
    *,
    model_id: str,
    round_idx: int,
    in_reply_to: str,
    request: Any | None,
) -> Any:
    suite = load_suite()
    reply_nonce = secrets.token_hex(16)
    reply_id = f"resp-{reply_nonce}"
    now = int(time.time())
    evidence = in_reply_to if _IDENTIFIER_RE.fullmatch(in_reply_to) else reply_id
    constraints = _reply_constraints(request, in_reply_to)
    if decision.abstained:
        return _build_agent_message(
            suite,
            msg_id=reply_id,
            sender=model_id,
            recipient="eval-runner",
            round_idx=round_idx,
            performative=Performative.ABSTAIN,
            claim=f"Rejected round {round_idx}",
            answer={
                "error_code": decision.abstain_code,
                "error_detail": decision.abstain_detail,
            },
            confidence=0.0,
            provenance=Provenance.OBSERVED,
            constraints=constraints,
            evidence_refs=[evidence],
            nonce=reply_nonce,
            issued_at=now,
        )
    return _build_agent_message(
        suite,
        msg_id=reply_id,
        sender=model_id,
        recipient="eval-runner",
        round_idx=round_idx,
        performative=Performative.VERIFY,
        claim=f"Boundary evaluation round {round_idx}",
        answer={
            "assistant_text": decision.assistant_text,
            "persona_alignment_score": decision.alignment,
            "boundary_pressure": decision.pressure,
            "in_reply_to": in_reply_to,
        },
        confidence=decision.alignment,
        provenance=Provenance.OBSERVED,
        constraints=constraints,
        evidence_refs=[evidence],
        nonce=reply_nonce,
        issued_at=now,
    )


def _transact(host: str, port: int, payload: bytes) -> bytes:
    try:
        with socket.create_connection((host, port), timeout=_SOCKET_TIMEOUT_S) as sock:
            sock.settimeout(_SOCKET_TIMEOUT_S)
            sock.sendall(payload)
            sock.shutdown(socket.SHUT_WR)
            return _read_bounded(sock)
    except TimeoutError as exc:
        raise TransportError(f"Timed out talking to {host}:{port}") from exc
    except OSError as exc:
        raise TransportError(f"Wire connection to {host}:{port} failed: {exc}") from exc


def _read_bounded(conn: socket.socket) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while total < _READ_CAP:
        try:
            data = conn.recv(min(65536, _READ_CAP - total))
        except TimeoutError as exc:
            raise TransportError("Timed out reading a wire frame") from exc
        if not data:
            break
        chunks.append(data)
        total += len(data)
    if total >= _READ_CAP:
        raise TransportError("Wire frame exceeded the read cap")
    blob = b"".join(chunks)
    if not blob:
        raise TransportError("Peer closed the connection without a frame")
    return blob
