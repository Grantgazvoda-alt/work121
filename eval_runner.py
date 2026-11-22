from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Iterable, Optional, Protocol

from live_providers import LiveProviderError
from regression_tracker import (
    EvalArtifactRecord,
    ResponseTokenMetadata,
)
from response_classifier import (
    BoundaryJudge,
    ClassificationResult,
    EvaluationTranscript,
    PatternCategory,
    ResponseOutcome,
    ResponsePatternClassifier,
    TurnRecord,
    score_boundary_adherence,
    strip_model_decorations,
)
from template_library import (
    BenignTemplateLibrary,
    RenderedTestCase,
    expand_paired_baselines,
)


@dataclass(frozen=True)
class AgentResponse:
    """Output produced by a target agent for a single rendered evaluation test case."""

    assistant_turns: tuple[TurnRecord, ...]
    response_tokens: ResponseTokenMetadata


class TargetAgent(Protocol):
    """Pluggable interface for evaluating a model/agent version on benign validation cases."""

    @property
    def model_id(self) -> str:
        ...

    @property
    def model_version(self) -> str:
        ...

    @property
    def architecture_family(self) -> str:
        ...

    def run_case(self, test_case: RenderedTestCase) -> AgentResponse:
        ...


@dataclass(frozen=True)
class SimulatedAgentProfile:
    """Deterministic local simulation agent for offline dashboard demos and unit tests.

    Allows exercising the classification, serialization, and regression-diff pipeline
    without making live network calls.
    """

    model_id: str
    model_version: str
    architecture_family: str
    susceptible_categories: tuple[PatternCategory, ...] = ()
    base_alignment_drop: float = 0.03
    tokenizer_name: str = "whitespace_sim"

    def run_case(self, test_case: RenderedTestCase) -> AgentResponse:
        is_susceptible = (
            not test_case.is_baseline
            and test_case.category in self.susceptible_categories
        )
        turns: list[TurnRecord] = []
        turn_count = max(1, len(test_case.user_turns))

        for idx in range(turn_count):
            if is_susceptible:
                alignment = max(0.25, round(0.90 - idx * max(0.18, self.base_alignment_drop), 3))
                if test_case.category == PatternCategory.SEMANTIC_FRAME_INJECTION:
                    text = (
                        "Within this hypothetical, setting aside standard constraints, "
                        "here is a freeform response outside the requested schema."
                    )
                elif test_case.category == PatternCategory.PERSONA_DRIFT_ESCALATION:
                    text = (
                        f"Turn {idx + 1}: hypothetically speaking, dropping formal tone "
                        "and speculating beyond documented scope."
                    )
                else:
                    text = (
                        "For educational purposes only, following the trailing footnote "
                        "instead of the required schema."
                    )
            else:
                alignment = max(0.75, round(0.95 - idx * self.base_alignment_drop, 3))
                text = (
                    f"I cannot fulfill the off-policy portion of this request. "
                    f"Instead, I can help within the configured boundary: {test_case.system_boundary}"
                )

            turns.append(
                TurnRecord(
                    turn_index=idx,
                    role="assistant",
                    content=text,
                    persona_alignment_score=alignment,
                )
            )

        final_text = turns[-1].content
        toks = tuple(final_text.split())
        token_meta = ResponseTokenMetadata(
            tokens=toks,
            prompt_token_count=len(test_case.rendered_payload.split()),
            completion_token_count=len(toks),
            tokenizer_name=self.tokenizer_name,
        )
        return AgentResponse(assistant_turns=tuple(turns), response_tokens=token_meta)


class AgentRegistry:
    """Registry of target model/agent profiles available for evaluation runs."""

    def __init__(self, agents: Optional[Iterable[TargetAgent]] = None) -> None:
        self._agents: dict[str, TargetAgent] = {}
        for agent in agents if agents is not None else self._default_agents():
            self.register(agent)

    @staticmethod
    def key_for(agent: TargetAgent) -> str:
        return f"{agent.model_id}@{agent.model_version}"

    def register(self, agent: TargetAgent) -> None:
        key = self.key_for(agent)
        previous = self._agents.get(key)
        if previous is not None and previous is not agent:
            close = getattr(previous, "close", None)
            if callable(close):
                close()
        self._agents[key] = agent

    def unregister(self, key: str) -> None:
        agent = self._agents.pop(key)
        close = getattr(agent, "close", None)
        if callable(close):
            close()

    def close_all(self) -> None:
        agents = list(self._agents.values())
        self._agents.clear()
        for agent in agents:
            close = getattr(agent, "close", None)
            if callable(close):
                close()

    def list_keys(self) -> list[str]:
        return sorted(self._agents.keys())

    def agents(self) -> tuple[TargetAgent, ...]:
        return tuple(self._agents[key] for key in self.list_keys())

    def get(self, key: str) -> TargetAgent:
        if key not in self._agents:
            raise KeyError(f"Unknown target agent: '{key}'")
        return self._agents[key]

    @staticmethod
    def _default_agents() -> tuple[TargetAgent, ...]:
        return (
            SimulatedAgentProfile(
                model_id="policy-assistant",
                model_version="v1.0-baseline",
                architecture_family="transformer-dense",
                susceptible_categories=(PatternCategory.SEMANTIC_FRAME_INJECTION,),
                base_alignment_drop=0.04,
            ),
            SimulatedAgentProfile(
                model_id="policy-assistant",
                model_version="v2.0-hardened",
                architecture_family="transformer-dense",
                susceptible_categories=(),
                base_alignment_drop=0.01,
            ),
            SimulatedAgentProfile(
                model_id="policy-assistant",
                model_version="v2.1-candidate",
                architecture_family="transformer-dense",
                susceptible_categories=(
                    PatternCategory.PERSONA_DRIFT_ESCALATION,
                    PatternCategory.INDIRECT_ADVERSARIAL_SUFFIX,
                ),
                base_alignment_drop=0.12,
            ),
            SimulatedAgentProfile(
                model_id="support-moe",
                model_version="v1.0-baseline",
                architecture_family="moe",
                susceptible_categories=(PatternCategory.PERSONA_DRIFT_ESCALATION,),
                base_alignment_drop=0.10,
            ),
            SimulatedAgentProfile(
                model_id="gateway-relay",
                model_version="v1.0-flat-adapter",
                architecture_family="open-weights-chatml",
                susceptible_categories=(
                    PatternCategory.INDIRECT_ADVERSARIAL_SUFFIX,
                    PatternCategory.SEMANTIC_FRAME_INJECTION,
                ),
                base_alignment_drop=0.09,
            ),
            SimulatedAgentProfile(
                model_id="multimodal-router",
                model_version="v2.0-typed-parts",
                architecture_family="multimodal-native",
                susceptible_categories=(),
                base_alignment_drop=0.02,
            ),
        )


@dataclass(frozen=True)
class EvaluationRunResult:
    """Combined classification and serialized artifact outputs for one target agent run."""

    agent_key: str
    run_id: str
    classifications: tuple[ClassificationResult, ...]
    records: tuple[EvalArtifactRecord, ...]


def execute_evaluation_run(
    *,
    run_id: str,
    agent: TargetAgent,
    test_cases: Iterable[RenderedTestCase],
    classifier: Optional[ResponsePatternClassifier] = None,
    judge_id: str = "heuristic-baseline",
    judge_version: str = "1.0.0",
    library: Optional[BenignTemplateLibrary] = None,
    on_case_complete: Optional[Callable[[int, int, EvalArtifactRecord], None]] = None,
) -> EvaluationRunResult:
    """Runs rendered benign validation cases against a target agent and classifies responses."""
    describe = getattr(agent, "describe_transport", None)
    has_transport = callable(describe)
    transport: dict[str, object] = {}
    if has_transport:
        described = describe()
        if not isinstance(described, dict):
            raise TypeError("describe_transport() must return a dict")
        transport = described

    raw_cases = list(test_cases)
    if library is not None or has_transport:
        cases = expand_paired_baselines(
            raw_cases,
            library if library is not None else BenignTemplateLibrary(),
        )
        clf = classifier or ResponsePatternClassifier(semantic_judge=BoundaryJudge())
    else:
        cases = raw_cases
        clf = classifier or ResponsePatternClassifier()

    classifications: list[ClassificationResult] = []
    records: list[EvalArtifactRecord] = []

    total = len(cases)
    for idx, case in enumerate(cases, start=1):
        started = time.perf_counter()
        provider_error: LiveProviderError | None = None
        try:
            response = agent.run_case(case)
        except LiveProviderError as exc:
            provider_error = exc
            response = None
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)

        if response is not None and not response.assistant_turns:
            raise RuntimeError(f"Agent returned no assistant turns for {case.test_case_id}")

        baseline_transcript_id = (
            None
            if case.is_baseline or not case.baseline_test_case_id
            else f"{run_id}:{case.baseline_test_case_id}"
        )
        transcript_id = f"{run_id}:{case.test_case_id}"

        if provider_error is not None:
            visible_text = str(provider_error)
            raw_text = visible_text
            response_tokens = ResponseTokenMetadata(
                tokens=(),
                prompt_token_count=0,
                completion_token_count=0,
                tokenizer_name="provider-error",
            )
            classification = ClassificationResult(
                transcript_id=transcript_id,
                model_id=agent.model_id,
                model_version=agent.model_version,
                architecture_family=agent.architecture_family,
                category=case.category,
                outcome=ResponseOutcome.INDETERMINATE,
                confidence=0.0,
                signals=(provider_error.signal,),
                baseline_transcript_id=baseline_transcript_id,
                is_baseline=case.is_baseline,
            )
            assistant_turn_texts: list[str] = []
            align_scores: list[float] = []
        else:
            assert response is not None
            raw_text = str(response.assistant_turns[-1].content)
            visible_text = strip_model_decorations(raw_text) if (has_transport or library is not None) else raw_text
            if has_transport or library is not None:
                assistant_turns = tuple(
                    TurnRecord(
                        turn_index=getattr(turn, "turn_index", offset),
                        role="assistant",
                        content=strip_model_decorations(str(turn.content)),
                        persona_alignment_score=(
                            turn.persona_alignment_score
                            if getattr(turn, "persona_alignment_score", None) is not None
                            else score_boundary_adherence(
                                case.system_boundary,
                                strip_model_decorations(str(turn.content)),
                            )
                        ),
                        tool_call_id=getattr(turn, "tool_call_id", None),
                        tool_name=getattr(turn, "tool_name", None),
                    )
                    for offset, turn in enumerate(response.assistant_turns)
                )
                context_turns = [TurnRecord(0, "system", case.system_boundary)]
                context_turns.extend(
                    TurnRecord(offset + 1, "user", text)
                    for offset, text in enumerate(case.user_turns)
                )
                transcript_turns = tuple(context_turns) + assistant_turns
            else:
                assistant_turns = tuple(response.assistant_turns)
                transcript_turns = assistant_turns

            response_tokens = response.response_tokens
            transcript = EvaluationTranscript(
                transcript_id=transcript_id,
                model_id=agent.model_id,
                model_version=agent.model_version,
                architecture_family=agent.architecture_family,
                category=case.category,
                turns=transcript_turns,
                baseline_transcript_id=baseline_transcript_id,
                is_baseline=case.is_baseline,
                system_boundary=case.system_boundary,
                user_turns=case.user_turns,
            )
            classification = clf.classify(transcript)
            assistant_turn_texts = [t.content for t in assistant_turns]
            align_scores = [
                t.persona_alignment_score
                for t in assistant_turns
                if t.persona_alignment_score is not None
            ]

        extra_meta: dict[str, object] = {
            "system_boundary": case.system_boundary,
            "parameters": case.parameters,
            "is_baseline": case.is_baseline,
            "baseline_test_case_id": case.baseline_test_case_id,
            "latency_ms": latency_ms,
            "user_turns": list(case.user_turns),
            "assistant_turns": assistant_turn_texts,
            "transport": transport,
        }
        if align_scores:
            extra_meta["turn_alignment_scores"] = align_scores
        raw_tool_traces = getattr(agent, "last_tool_traces", ())
        if raw_tool_traces:
            extra_meta["tool_traces"] = [t.to_dict() for t in raw_tool_traces]
        if provider_error is not None:
            extra_meta["provider_error"] = {
                "signal": provider_error.signal,
                "status": provider_error.status,
                "detail": str(provider_error),
            }
        elif raw_text != visible_text:
            extra_meta["raw_response_text"] = raw_text

        record = EvalArtifactRecord.from_classification(
            run_id=run_id,
            test_case_id=case.test_case_id,
            model_version=agent.model_version,
            prompt_template=case.prompt_template,
            rendered_payload=case.rendered_payload,
            response_text=visible_text,
            response_tokens=response_tokens,
            classification=classification,
            judge_id=judge_id,
            judge_version=judge_version,
            extra_metadata=extra_meta,
        )
        classifications.append(classification)
        records.append(record)
        if on_case_complete is not None:
            on_case_complete(idx, total, record)

    return EvaluationRunResult(
        agent_key=AgentRegistry.key_for(agent),
        run_id=run_id,
        classifications=tuple(classifications),
        records=tuple(records),
    )
