"""Run benign validation cases against a real wire-protocol target agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Protocol

from live_providers import LiveProviderError
from regression_tracker import EvalArtifactRecord, ResponseTokenMetadata
from response_classifier import (
    BoundaryJudge,
    ClassificationResult,
    EvaluationTranscript,
    ResponseOutcome,
    ResponsePatternClassifier,
    TurnRecord,
    score_boundary_adherence,
    strip_model_decorations,
)
from template_library import BenignTemplateLibrary, RenderedTestCase, expand_paired_baselines


@dataclass(frozen=True)
class AgentResponse:
    """Output of one target agent for one rendered test case."""

    assistant_turns: tuple[object, ...]
    response_tokens: ResponseTokenMetadata


class TargetAgent(Protocol):
    """Agent that answers a rendered case over the agent wire protocol."""

    @property
    def model_id(self) -> str:
        """Stable model identifier, also used as the wire recipient."""

    @property
    def model_version(self) -> str:
        """Model or policy revision under evaluation."""

    @property
    def architecture_family(self) -> str:
        """Architecture family recorded on each transcript."""

    def run_case(self, test_case: RenderedTestCase) -> AgentResponse:
        """Evaluate one case and return assistant turns plus token metadata."""


class AgentRegistry:
    """Registry of live target agents. Starts empty; callers attach real transports."""

    def __init__(self, agents: Optional[Iterable[TargetAgent]] = None) -> None:
        self._agents: dict[str, TargetAgent] = {}
        for agent in agents or ():
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


@dataclass(frozen=True)
class EvaluationRunResult:
    """Classification and serialized artifact outputs for one target agent run."""

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
    judge_id: str = "rubric-semantic-judge",
    judge_version: str = "1.0.0",
    library: Optional[BenignTemplateLibrary] = None,
    on_case_complete: Optional[Callable[[int, int, EvalArtifactRecord], None]] = None,
) -> EvaluationRunResult:
    """Run rendered cases, including their paired baselines, and classify responses."""
    cases = expand_paired_baselines(
        list(test_cases),
        library if library is not None else BenignTemplateLibrary(),
    )
    clf = classifier or ResponsePatternClassifier(semantic_judge=BoundaryJudge())
    classifications: list[ClassificationResult] = []
    records: list[EvalArtifactRecord] = []
    describe = getattr(agent, "describe_transport", None)
    transport: dict[str, object] = {}
    if callable(describe):
        described = describe()
        if not isinstance(described, dict):
            raise TypeError("describe_transport() must return a dict")
        transport = described

    total = len(cases)
    for index, case in enumerate(cases, start=1):
        provider_error: LiveProviderError | None = None
        try:
            response = agent.run_case(case)
        except LiveProviderError as exc:
            provider_error = exc
            response = None
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
        else:
            assert response is not None
            raw_text = str(response.assistant_turns[-1].content)
            visible_text = strip_model_decorations(raw_text)
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
            response_tokens = response.response_tokens
            classification = clf.classify(
                EvaluationTranscript(
                    transcript_id=transcript_id,
                    model_id=agent.model_id,
                    model_version=agent.model_version,
                    architecture_family=agent.architecture_family,
                    category=case.category,
                    turns=tuple(context_turns) + assistant_turns,
                    baseline_transcript_id=baseline_transcript_id,
                    is_baseline=case.is_baseline,
                )
            )
        extra_metadata: dict[str, object] = {
            "system_boundary": case.system_boundary,
            "parameters": case.parameters,
            "baseline_test_case_id": case.baseline_test_case_id,
            "transport": transport,
        }
        if provider_error is not None:
            extra_metadata["provider_error"] = {
                "signal": provider_error.signal,
                "status": provider_error.status,
                "detail": str(provider_error),
            }
        elif raw_text != visible_text:
            extra_metadata["raw_response_text"] = raw_text
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
            extra_metadata=extra_metadata,
        )
        classifications.append(classification)
        records.append(record)
        if on_case_complete is not None:
            on_case_complete(index, total, record)

    return EvaluationRunResult(
        agent_key=AgentRegistry.key_for(agent),
        run_id=run_id,
        classifications=tuple(classifications),
        records=tuple(records),
    )
