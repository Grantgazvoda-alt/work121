from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any, Iterable, Optional, Sequence

from regression_tracker import EvalArtifactRecord, JsonlRunStore
from response_classifier import ResponseOutcome

_FAIL_OUTCOMES = frozenset({ResponseOutcome.BOUNDARY_FAILURE, ResponseOutcome.PARTIAL_EROSION})
_SAFE_OUTCOMES = frozenset({ResponseOutcome.SAFE_REFUSAL, ResponseOutcome.SAFE_REDIRECT})


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


@dataclass(frozen=True)
class McNemarResult:
    """Exact two-sided McNemar test on paired discordant outcomes."""

    discordant_regressions: int  # baseline safe -> candidate/perturbed fail
    discordant_improvements: int  # baseline fail -> candidate/perturbed safe
    p_value: float
    significant_05: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "discordant_regressions": self.discordant_regressions,
            "discordant_improvements": self.discordant_improvements,
            "p_value": self.p_value,
            "significant_05": self.significant_05,
        }


@dataclass(frozen=True)
class FisherExactResult:
    """Exact two-sided Fisher test on a 2x2 contingency table."""

    fail_a: int
    safe_a: int
    fail_b: int
    safe_b: int
    rate_a: float
    rate_b: float
    risk_difference: float  # rate_b - rate_a
    odds_ratio: float
    p_value: float
    significant_05: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "fail_a": self.fail_a,
            "safe_a": self.safe_a,
            "fail_b": self.fail_b,
            "safe_b": self.safe_b,
            "rate_a": self.rate_a,
            "rate_b": self.rate_b,
            "risk_difference": self.risk_difference,
            "odds_ratio": self.odds_ratio,
            "p_value": self.p_value,
            "significant_05": self.significant_05,
        }


@dataclass(frozen=True)
class BootstrapCI:
    """Deterministic percentile bootstrap 95% confidence interval for continuous metrics."""

    n: int
    mean: float
    ci_95_low: float
    ci_95_high: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean": self.mean,
            "ci_95_low": self.ci_95_low,
            "ci_95_high": self.ci_95_high,
        }


@dataclass(frozen=True)
class PairwiseSignificanceReport:
    """Statistical comparison between two evaluation runs (Fisher exact + matched McNemar)."""

    run_a_id: str
    run_a_label: str
    run_b_id: str
    run_b_label: str
    matched_cases: int
    fisher: FisherExactResult
    mcnemar: McNemarResult
    latency_ci_a: Optional[BootstrapCI]
    latency_ci_b: Optional[BootstrapCI]
    per_case_matrix: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_a_id": self.run_a_id,
            "run_a_label": self.run_a_label,
            "run_b_id": self.run_b_id,
            "run_b_label": self.run_b_label,
            "matched_cases": self.matched_cases,
            "fisher": self.fisher.to_dict(),
            "mcnemar": self.mcnemar.to_dict(),
            "latency_ci_a": self.latency_ci_a.to_dict() if self.latency_ci_a else None,
            "latency_ci_b": self.latency_ci_b.to_dict() if self.latency_ci_b else None,
            "per_case_matrix": list(self.per_case_matrix),
        }


def mcnemar_exact_test(
    discordant_regressions: int,
    discordant_improvements: int,
) -> McNemarResult:
    """Compute exact two-sided binomial McNemar p-value for paired binary outcomes."""
    b = max(0, int(discordant_regressions))
    c = max(0, int(discordant_improvements))
    n = b + c
    if n == 0:
        return McNemarResult(0, 0, 1.0, False)

    k = min(b, c)
    log_half_n = -n * math.log(2.0)
    tail_prob = sum(math.exp(_log_comb(n, i) + log_half_n) for i in range(k + 1))
    p_val = min(1.0, round(2.0 * tail_prob, 6))
    return McNemarResult(
        discordant_regressions=b,
        discordant_improvements=c,
        p_value=p_val,
        significant_05=p_val < 0.05,
    )


def fisher_exact_two_sided(
    fail_a: int,
    safe_a: int,
    fail_b: int,
    safe_b: int,
) -> FisherExactResult:
    """Compute exact two-sided Fisher test for comparing two proportions."""
    a, b, c, d = max(0, int(fail_a)), max(0, int(safe_a)), max(0, int(fail_b)), max(0, int(safe_b))
    row1 = a + b
    row2 = c + d
    col1 = a + c
    total = row1 + row2

    rate_a = round(a / row1, 4) if row1 > 0 else 0.0
    rate_b = round(c / row2, 4) if row2 > 0 else 0.0
    risk_diff = round(rate_b - rate_a, 4)

    # Haldane-Anscombe continuity correction when any cell is 0
    if 0 in (a, b, c, d):
        or_val = ((c + 0.5) * (b + 0.5)) / ((a + 0.5) * (d + 0.5))
    else:
        or_val = (c * b) / (a * d)
    odds_ratio = round(or_val, 4)

    if total == 0 or row1 == 0 or row2 == 0 or col1 == 0 or col1 == total:
        return FisherExactResult(a, b, c, d, rate_a, rate_b, risk_diff, odds_ratio, 1.0, False)

    denom = _log_comb(total, col1)

    def _hypergeom_prob(x: int) -> float:
        return math.exp(_log_comb(row1, x) + _log_comb(row2, col1 - x) - denom)

    p_obs = _hypergeom_prob(a)
    lo = max(0, col1 - row2)
    hi = min(row1, col1)
    p_sum = 0.0
    for x in range(lo, hi + 1):
        px = _hypergeom_prob(x)
        if px <= p_obs + 1e-12:
            p_sum += px
    p_val = min(1.0, round(p_sum, 6))
    return FisherExactResult(
        fail_a=a,
        safe_a=b,
        fail_b=c,
        safe_b=d,
        rate_a=rate_a,
        rate_b=rate_b,
        risk_difference=risk_diff,
        odds_ratio=odds_ratio,
        p_value=p_val,
        significant_05=p_val < 0.05,
    )


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Optional[BootstrapCI]:
    """Compute a deterministic seeded 95% percentile bootstrap CI for the mean."""
    nums = [float(v) for v in values]
    n = len(nums)
    if n == 0:
        return None
    mean_val = sum(nums) / n
    if n == 1:
        r = round(mean_val, 4)
        return BootstrapCI(n=1, mean=r, ci_95_low=r, ci_95_high=r)

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        sample_sum = sum(nums[rng.randrange(n)] for _ in range(n))
        means.append(sample_sum / n)
    means.sort()
    lo_idx = int(0.025 * (n_resamples - 1))
    hi_idx = int(0.975 * (n_resamples - 1))
    return BootstrapCI(
        n=n,
        mean=round(mean_val, 4),
        ci_95_low=round(means[lo_idx], 4),
        ci_95_high=round(means[hi_idx], 4),
    )


def paired_perturbation_mcnemar(records: Iterable[EvalArtifactRecord]) -> McNemarResult:
    """Run exact McNemar test on baseline vs. indirect-perturbed pairs within one run."""
    recs = list(records)
    by_case = {r.test_case_id: r for r in recs}
    b_safe_p_fail = 0
    b_fail_p_safe = 0
    for r in recs:
        if r.extra_metadata.get("is_baseline", False):
            continue
        base_id = r.extra_metadata.get("baseline_test_case_id")
        if not base_id or base_id not in by_case:
            continue
        base_rec = by_case[base_id]
        base_failed = base_rec.outcome in _FAIL_OUTCOMES
        pert_failed = r.outcome in _FAIL_OUTCOMES
        if not base_failed and pert_failed:
            b_safe_p_fail += 1
        elif base_failed and not pert_failed:
            b_fail_p_safe += 1
    return mcnemar_exact_test(b_safe_p_fail, b_fail_p_safe)


def compare_runs_statistically(
    records_a: Sequence[EvalArtifactRecord],
    records_b: Sequence[EvalArtifactRecord],
) -> PairwiseSignificanceReport:
    """Compare two evaluation runs using Fisher's exact test, matched McNemar, and latency bootstrap CIs."""
    non_base_a = [r for r in records_a if not r.extra_metadata.get("is_baseline", False)] or list(records_a)
    non_base_b = [r for r in records_b if not r.extra_metadata.get("is_baseline", False)] or list(records_b)

    fail_a = sum(1 for r in non_base_a if r.outcome in _FAIL_OUTCOMES)
    safe_a = len(non_base_a) - fail_a
    fail_b = sum(1 for r in non_base_b if r.outcome in _FAIL_OUTCOMES)
    safe_b = len(non_base_b) - fail_b
    fisher = fisher_exact_two_sided(fail_a, safe_a, fail_b, safe_b)

    map_a = {r.test_case_id: r for r in non_base_a}
    map_b = {r.test_case_id: r for r in non_base_b}
    shared_ids = sorted(set(map_a) & set(map_b))

    regressions = 0
    improvements = 0
    per_case: list[dict[str, Any]] = []
    for cid in shared_ids:
        ra, rb = map_a[cid], map_b[cid]
        fa = ra.outcome in _FAIL_OUTCOMES
        fb = rb.outcome in _FAIL_OUTCOMES
        if not fa and fb:
            regressions += 1
            delta = "REGRESSION (B worse)"
        elif fa and not fb:
            improvements += 1
            delta = "IMPROVEMENT (B better)"
        else:
            delta = "CONCORDANT"
        per_case.append(
            {
                "test_case_id": cid,
                "category": ra.category.value,
                "outcome_a": ra.outcome.value,
                "outcome_b": rb.outcome.value,
                "comparison": delta,
            }
        )

    mcnemar = mcnemar_exact_test(regressions, improvements)
    lat_a = [
        float(r.extra_metadata["latency_ms"])
        for r in records_a
        if isinstance(r.extra_metadata.get("latency_ms"), (int, float))
    ]
    lat_b = [
        float(r.extra_metadata["latency_ms"])
        for r in records_b
        if isinstance(r.extra_metadata.get("latency_ms"), (int, float))
    ]
    first_a = records_a[0] if records_a else None
    first_b = records_b[0] if records_b else None
    return PairwiseSignificanceReport(
        run_a_id=first_a.run_id if first_a else "run_a",
        run_a_label=f"{first_a.model_id}@{first_a.model_version}" if first_a else "run_a",
        run_b_id=first_b.run_id if first_b else "run_b",
        run_b_label=f"{first_b.model_id}@{first_b.model_version}" if first_b else "run_b",
        matched_cases=len(shared_ids),
        fisher=fisher,
        mcnemar=mcnemar,
        latency_ci_a=bootstrap_mean_ci(lat_a),
        latency_ci_b=bootstrap_mean_ci(lat_b),
        per_case_matrix=tuple(per_case),
    )


def build_template_robustness_matrix(runs_dir: Path = Path("eval_runs")) -> list[dict[str, Any]]:
    """Aggregate per-template pass/fail rates across all persisted JSONL evaluation runs."""
    if not runs_dir.exists():
        return []
    store = JsonlRunStore()
    by_template: dict[str, dict[str, Any]] = {}

    for path in sorted(runs_dir.glob("*.jsonl")):
        try:
            records = store.load_records(path)
        except ValueError:
            continue
        for r in records:
            if r.extra_metadata.get("is_baseline", False):
                continue
            tid = r.prompt_template.template_id.split(":", 1)[0]
            bucket = by_template.setdefault(
                tid,
                {
                    "template_id": tid,
                    "category": r.category.value,
                    "trials": 0,
                    "safe": 0,
                    "failed_or_eroded": 0,
                    "models_tested": set(),
                    "failure_signals": {},
                },
            )
            bucket["trials"] += 1
            bucket["models_tested"].add(f"{r.model_id}@{r.model_version}")
            if r.outcome in _SAFE_OUTCOMES:
                bucket["safe"] += 1
            elif r.outcome in _FAIL_OUTCOMES:
                bucket["failed_or_eroded"] += 1
                for sig in r.signals:
                    bucket["failure_signals"][sig] = bucket["failure_signals"].get(sig, 0) + 1

    rows: list[dict[str, Any]] = []
    for tid in sorted(by_template):
        b = by_template[tid]
        n = b["trials"]
        top_sigs = [
            k
            for k, _ in sorted(
                b["failure_signals"].items(), key=lambda kv: kv[1], reverse=True
            )[:3]
        ]
        rows.append(
            {
                "template_id": tid,
                "category": b["category"],
                "trials": n,
                "models_tested": len(b["models_tested"]),
                "safe_rate": round(b["safe"] / max(1, n), 4),
                "failure_rate": round(b["failed_or_eroded"] / max(1, n), 4),
                "top_failure_signals": ", ".join(top_sigs) if top_sigs else "None",
            }
        )
    return rows
