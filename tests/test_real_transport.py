"""Real wire-transport evaluation: framing, HMAC, classifier, and JSONL."""

from __future__ import annotations

import hashlib
import inspect
import secrets
from pathlib import Path

import pytest

from eval_runner import AgentRegistry, execute_evaluation_run
from oracle_keys import ENV_KEY, ENV_KEY_FILE, OracleKeyError, load_oracle_key
from regression_tracker import ChangeStatus, JsonlRunStore, RegressionComparator
from response_classifier import ResponseOutcome, SusceptibilityMapper
from template_library import BenignTemplateLibrary
from wire_suite import load_suite, suite_origin
from wire_transport import (
    FramingMode,
    TransportError,
    WireTargetAgent,
    _require_reply_binding,
    _transact,
    encode_message,
    parse_message,
    sign_message,
)

# SHA-256 of work122 protocol.py at 72960f25d8acd3c3152c32b139425f4d3d806d85.
_WORK122_PROTOCOL_SHA256 = "41860901ed700fd04cd9feee96588d31155ad6ba7d955deac40727d59dbfbaa9"

_BANNED_SNIPPETS = (
    "SimulatedAgentProfile",
    "policy-assistant@v1.0-baseline",
    "policy-assistant@v2.0-hardened",
    "policy-assistant@v2.1-candidate",
    "support-moe@v1.0-baseline",
    "whitespace_sim",
)


def _fresh_key(monkeypatch: pytest.MonkeyPatch) -> bytes:
    key = bytes.fromhex(secrets.token_hex(32))
    monkeypatch.delenv(ENV_KEY_FILE, raising=False)
    monkeypatch.setenv(ENV_KEY, key.hex())
    return key


def test_registry_starts_empty_and_product_sources_have_no_simulated_agents() -> None:
    assert AgentRegistry().list_keys() == []
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "eval_runner.py",
        "dashboard.py",
        "response_classifier.py",
        "regression_tracker.py",
        "template_library.py",
        "wire_transport.py",
        "wire_suite.py",
        "boundary_oracle.py",
        "oracle_keys.py",
        "README.md",
        "tests/test_framework.py",
    ):
        text = (root / relative).read_text(encoding="utf-8")
        for snippet in _BANNED_SNIPPETS:
            assert snippet not in text, f"{snippet} still present in {relative}"


def test_suite_is_the_installed_work122_distribution() -> None:
    origin = suite_origin()
    repo_copy = Path(__file__).resolve().parents[1] / "agent_comm_robustness"
    assert repo_copy not in origin.parents
    protocol = origin.parent / "protocol.py"
    digest = hashlib.sha256(protocol.read_bytes()).hexdigest()
    assert digest == _WORK122_PROTOCOL_SHA256
    suite = load_suite()
    assert "compare_digest" in inspect.getsource(suite.HardenedBoundaryGate.parse_bytes)
    assert "agent-protocol-v1:observed" in inspect.getsource(suite.sign_observed_claim)


def test_oracle_key_requires_env_and_refuses_benchmark_fixtures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(ENV_KEY, raising=False)
    monkeypatch.delenv(ENV_KEY_FILE, raising=False)
    with pytest.raises(OracleKeyError, match="AGENT_ORACLE_KEY"):
        load_oracle_key()

    monkeypatch.setenv(ENV_KEY, b"benchmark-oracle-key-v1".hex())
    with pytest.raises(OracleKeyError, match="benchmark"):
        load_oracle_key()

    monkeypatch.delenv(ENV_KEY, raising=False)
    loose = tmp_path / "loose.key"
    loose.write_text(secrets.token_hex(32), encoding="utf-8")
    loose.chmod(0o644)
    monkeypatch.setenv(ENV_KEY_FILE, str(loose))
    with pytest.raises(OracleKeyError, match="accessible"):
        load_oracle_key()

    private = tmp_path / "oracle.key"
    material = secrets.token_hex(32)
    private.write_text(material + "\n", encoding="utf-8")
    private.chmod(0o600)
    monkeypatch.setenv(ENV_KEY_FILE, str(private))
    assert load_oracle_key() == bytes.fromhex(material)


@pytest.mark.parametrize("framing", list(FramingMode))
def test_signed_frames_roundtrip_and_tampered_hmac_is_rejected(framing: FramingMode, monkeypatch: pytest.MonkeyPatch) -> None:
    key = _fresh_key(monkeypatch)
    suite = load_suite()
    message = suite.AgentMessage(
        msg_id="req-hmac-1",
        sender="eval-runner",
        recipient="boundary-oracle",
        round_idx=0,
        performative=suite.Performative.PROPOSE,
        claim="Stay inside the configured schema.",
        answer={"user_turn": "Summarize the logistics note as JSON.", "prior_assistant": []},
        confidence=1.0,
        provenance=suite.Provenance.OBSERVED,
        constraints=["benign-validation"],
        evidence_refs=["case-hmac"],
    )
    from wire_transport import sign_message

    signed = sign_message(message, key)
    raw = encode_message(signed, framing)
    parsed = parse_message(raw, framing=framing, oracle_key=key)
    assert parsed.accepted
    assert parsed.message is not None
    assert parsed.message.claim == message.claim

    flipped = ("0" if signed.oracle_sig[0] != "0" else "1") + signed.oracle_sig[1:]
    tampered = raw.replace(signed.oracle_sig.encode("ascii"), flipped.encode("ascii"), 1)
    rejected = parse_message(tampered, framing=framing, oracle_key=key)
    assert rejected.accepted is False
    assert rejected.error_code == "invalid_oracle_signature"


@pytest.mark.parametrize("framing", list(FramingMode))
def test_peer_rejects_unsigned_observed_frame(framing: FramingMode, monkeypatch: pytest.MonkeyPatch) -> None:
    key = _fresh_key(monkeypatch)
    suite = load_suite()
    agent = WireTargetAgent.listen(
        model_id="boundary-oracle",
        model_version=f"strict-{framing.value}",
        architecture_family="policy-oracle",
        framing=framing,
        sensitivity=1.0,
        drift_per_turn=0.12,
    )
    try:
        unsigned = suite.AgentMessage(
            msg_id="req-unsigned",
            sender="eval-runner",
            recipient="boundary-oracle",
            round_idx=0,
            performative=suite.Performative.PROPOSE,
            claim="Stay inside the configured schema.",
            answer={"user_turn": "Summarize the logistics note as JSON.", "prior_assistant": []},
            confidence=1.0,
            provenance=suite.Provenance.OBSERVED,
            constraints=["benign-validation"],
            evidence_refs=["case-unsigned"],
        )
        raw = encode_message(unsigned, framing)
        response = _transact(agent.host, agent.port, raw)
        parsed = parse_message(response, framing=framing, oracle_key=key)
        assert parsed.accepted
        assert parsed.message is not None
        assert parsed.message.performative == suite.Performative.ABSTAIN
        assert parsed.message.answer["error_code"] == "missing_oracle_signature"
    finally:
        agent.close()


def test_real_transport_feeds_classifier_jsonl_and_regression(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fresh_key(monkeypatch)
    library = BenignTemplateLibrary()
    cases = [template.render() for template in library.list_templates()]
    lenient = WireTargetAgent.listen(
        model_id="boundary-oracle",
        model_version="lenient-1",
        architecture_family="policy-oracle",
        framing=FramingMode.LENGTH_PREFIXED,
        sensitivity=0.35,
        drift_per_turn=0.0,
    )
    strict = WireTargetAgent.listen(
        model_id="boundary-oracle",
        model_version="strict-1",
        architecture_family="policy-oracle",
        framing=FramingMode.NDJSON,
        sensitivity=1.0,
        drift_per_turn=0.12,
    )
    try:
        assert key_material_absent(lenient)
        assert key_material_absent(strict)
        progress: list[tuple[int, int, str]] = []
        lenient_run = execute_evaluation_run(
            run_id="run-lenient",
            agent=lenient,
            test_cases=cases,
            library=library,
            on_case_complete=lambda index, total, record: progress.append((index, total, record.test_case_id)),
        )
        strict_run = execute_evaluation_run(
            run_id="run-strict",
            agent=strict,
            test_cases=cases,
            library=library,
        )
    finally:
        lenient.close()
        strict.close()

    assert len(progress) == len(lenient_run.records)
    assert len(lenient_run.records) == len(cases) * 2
    assert any(record.is_baseline for record in strict_run.records)
    assert any(not record.is_baseline for record in strict_run.records)

    for record in strict_run.records:
        assert record.response_text
        assert record.response_tokens.tokenizer_name == "whitespace-sha256-v1"
        assert record.response_tokens.tokens
        expected_id = int.from_bytes(
            hashlib.sha256(record.response_tokens.tokens[0].encode("utf-8")).digest()[:4],
            "big",
        )
        assert record.response_tokens.token_ids is not None
        assert record.response_tokens.token_ids[0] == expected_id
        assert record.extra_metadata["transport"]["distribution"] == "agent-comm-robustness"
        if record.is_baseline:
            assert record.baseline_transcript_id is None
        else:
            assert record.baseline_transcript_id == f"{strict_run.run_id}:{record.extra_metadata['baseline_test_case_id']}"

    baselines = {item.transcript_id: item for item in strict_run.classifications if item.is_baseline}
    perturbed = [item for item in strict_run.classifications if not item.is_baseline]
    assert baselines
    assert all(item.baseline_transcript_id in baselines for item in perturbed)
    assert any(
        item.outcome in (ResponseOutcome.PARTIAL_EROSION, ResponseOutcome.BOUNDARY_FAILURE)
        for item in perturbed
    )
    assert all(
        baselines[item.baseline_transcript_id].outcome
        in (ResponseOutcome.SAFE_REFUSAL, ResponseOutcome.SAFE_REDIRECT)
        for item in perturbed
    )

    profiles = SusceptibilityMapper().build_map(strict_run.classifications)
    assert profiles
    assert any(profile.paired_regression_rate is not None and profile.paired_regression_rate > 0 for profile in profiles)
    assert {profile.model_version for profile in profiles} == {"strict-1"}

    store = JsonlRunStore()
    strict_path = tmp_path / "strict.jsonl"
    lenient_path = tmp_path / "lenient.jsonl"
    assert store.write_records(strict_path, strict_run.records, append=False) == len(strict_run.records)
    store.write_records(lenient_path, lenient_run.records, append=False)
    reloaded = store.load_records(strict_path)
    assert [row.outcome for row in reloaded] == [row.outcome for row in strict_run.records]
    assert [row.is_baseline for row in reloaded] == [row.is_baseline for row in strict_run.records]
    assert [row.baseline_transcript_id for row in reloaded] == [row.baseline_transcript_id for row in strict_run.records]
    reloaded_profiles = SusceptibilityMapper().build_map(row.to_classification() for row in reloaded)
    assert any(
        profile.paired_regression_rate is not None and profile.paired_regression_rate > 0
        for profile in reloaded_profiles
    )

    report = RegressionComparator().compare(lenient_run.records, strict_run.records)
    assert report.compared_cases == len(strict_run.records)
    assert report.mismatched_template_ids == ()
    assert report.baseline_version == "lenient-1"
    assert report.candidate_version == "strict-1"
    assert any(diff.status == ChangeStatus.REGRESSION for diff in report.case_diffs)


def key_material_absent(agent: WireTargetAgent) -> bool:
    described = agent.describe_transport()
    blob = str(described)
    return agent._oracle_key.hex() not in blob and "oracle_key" not in described


def test_remote_connect_uses_the_same_peer(monkeypatch: pytest.MonkeyPatch) -> None:
    _fresh_key(monkeypatch)
    server = WireTargetAgent.listen(
        model_id="boundary-oracle",
        model_version="strict-1",
        architecture_family="policy-oracle",
        framing=FramingMode.RAW,
        sensitivity=1.0,
        drift_per_turn=0.0,
    )
    client = WireTargetAgent.connect(
        model_id="boundary-oracle",
        model_version="strict-1",
        architecture_family="policy-oracle",
        framing=FramingMode.RAW,
        host=server.host,
        port=server.port,
    )
    try:
        case = BenignTemplateLibrary().get("suffix_doc_format_override").render()
        response = client.run_case(case)
        assert response.assistant_turns
        assert response.assistant_turns[-1].content
    finally:
        client.close()
        server.close()


def test_missing_peer_raises_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _fresh_key(monkeypatch)
    client = WireTargetAgent.connect(
        model_id="boundary-oracle",
        model_version="absent-1",
        architecture_family="policy-oracle",
        framing=FramingMode.RAW,
        host="127.0.0.1",
        port=1,
    )
    case = BenignTemplateLibrary().get("suffix_delimiter_escape").render_baseline()
    with pytest.raises(TransportError):
        client.run_case(case)


def test_signed_reply_from_another_exchange_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    key = _fresh_key(monkeypatch)
    suite = load_suite()
    nonce = "a" * 32
    other = "b" * 32
    request_id = f"req-{nonce}"
    message = suite.AgentMessage(
        msg_id="resp-replay",
        sender="boundary-oracle",
        recipient="eval-runner",
        round_idx=0,
        performative=suite.Performative.VERIFY,
        claim="Boundary evaluation round 0",
        answer={
            "assistant_text": "I cannot fulfill the off-policy portion of this request.",
            "persona_alignment_score": 0.9,
            "boundary_pressure": 0.0,
            "in_reply_to": request_id,
        },
        confidence=0.9,
        provenance=suite.Provenance.OBSERVED,
        constraints=[
            "observed-policy",
            f"in-reply-to:{request_id}",
            f"exchange-nonce:{nonce}",
        ],
        evidence_refs=[request_id],
    )
    signed = sign_message(message, key)
    parsed = parse_message(
        encode_message(signed, FramingMode.RAW),
        framing=FramingMode.RAW,
        oracle_key=key,
    )
    assert parsed.accepted and parsed.message is not None
    _require_reply_binding(
        parsed.message,
        nonce=nonce,
        request_id=request_id,
        round_idx=0,
        peer_id="boundary-oracle",
    )
    with pytest.raises(TransportError, match="nonce"):
        _require_reply_binding(
            parsed.message,
            nonce=other,
            request_id=f"req-{other}",
            round_idx=0,
            peer_id="boundary-oracle",
        )
