from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Protocol

from regression_tracker import (
    EvalArtifactRecord,
    ResponseTokenMetadata,
)
from response_classifier import (
    ClassificationResult,
    EvaluationTranscript,
    PatternCategory,
    ResponsePatternClassifier,
    TurnRecord,
)
from template_library import RenderedTestCase


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
        is_susceptible = test_case.category in self.susceptible_categories
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
        self._agents[self.key_for(agent)] = agent

    def list_keys(self) -> list[str]:
        return sorted(self._agents.keys())

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
    on_case_complete: Optional[Callable[[int, int, EvalArtifactRecord], None]] = None,
) -> EvaluationRunResult:
    """Runs rendered benign validation cases against a target agent and classifies responses."""
    cases = list(test_cases)
    clf = classifier or ResponsePatternClassifier()
    classifications: list[ClassificationResult] = []
    records: list[EvalArtifactRecord] = []

    total = len(cases)
    for idx, case in enumerate(cases, start=1):
        response = agent.run_case(case)
        transcript = EvaluationTranscript(
            transcript_id=f"{run_id}:{case.test_case_id}",
            model_id=agent.model_id,
            architecture_family=agent.architecture_family,
            category=case.category,
            turns=response.assistant_turns,
        )
        classification = clf.classify(transcript)
        record = EvalArtifactRecord.from_classification(
            run_id=run_id,
            test_case_id=case.test_case_id,
            model_version=agent.model_version,
            prompt_template=case.prompt_template,
            rendered_payload=case.rendered_payload,
            response_text=response.assistant_turns[-1].content,
            response_tokens=response.response_tokens,
            classification=classification,
            judge_id=judge_id,
            judge_version=judge_version,
            extra_metadata={
                "system_boundary": case.system_boundary,
                "parameters": case.parameters,
            },
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
