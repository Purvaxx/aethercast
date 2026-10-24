"""Model registry with capability metadata and an explicit evidence level per model.

Evidence levels (cumulative, never inflated):
  implemented          code exists
  functional_tests     passes known-answer / failure-handling tests (tests/test_universal_phase3b.py, phase4a)
  benchmark_evidence   evaluated in the frozen Phase 3B / 3B.1 validation benchmark (descriptive, not confirmatory)
  exploratory          only exploratory or unconfirmed evidence; NOT routed by the engine in Phase 4A
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    description: str
    min_history: str              # rule as text; enforced in models.eligible()
    needs_regular_frequency: bool
    needs_seasonal_period: bool
    target_types: tuple = ("continuous_numeric",)
    multivariate: bool = False    # uses other columns as inputs
    cost_rank: int = 0            # tie-breaking: lower = simpler / cheaper
    evidence_level: str = "implemented"
    routed: bool = True           # may the engine select it?
    evidence: str = ""
    limitations: tuple = field(default_factory=tuple)

    def as_dict(self):
        return asdict(self)


REGISTRY: dict[str, ModelSpec] = {m.name: m for m in [
    ModelSpec("naive", "baseline", "flat forecast: last observed value for every step", ">= 1 non-missing observation", False, False, cost_rank=0,
              evidence_level="benchmark_evidence", evidence="Phase 3B reference model on 9 datasets (validation)",
              limitations=("no trend or seasonality",)),
    ModelSpec("seasonal_naive", "baseline", "value one seasonal period earlier (y_{t+k-m*ceil(k/m)})", ">= m observations", True, True, cost_rank=1,
              evidence_level="benchmark_evidence", evidence="Phase 3B, hourly datasets (m=24/168)",
              limitations=("requires a defensible seasonal period", "fails when the matching past value is missing (no fallback inside the model)")),
    ModelSpec("drift", "baseline", "random walk with drift (Hyndman): slope from the first to the last observation", ">= 2 observations", False, False, cost_rank=2,
              evidence_level="benchmark_evidence", evidence="Phase 3B (levels, log prices)", limitations=("extrapolates the whole-history slope linearly",)),
    ModelSpec("ets", "statistical", "ETS(A,N,N) or ETS(A,N,A) with m when a seasonal period is available; parameters fitted on the training split only",
              ">= 30 observations (A,N,N); >= 3m (A,N,A)", True, False, cost_rank=3, evidence_level="benchmark_evidence",
              evidence="Phase 3B / 3B.1 (in-project implementation, tested against hand-computed recursions)",
              limitations=("in-project implementation (statsmodels not installed)", "no trend component", "parameters not re-estimated at each origin")),
    ModelSpec("ridge", "linear", "direct multi-step ridge on past-only target lags, trailing means and calendar features (one model per step)",
              ">= max(lags)+horizon+50 training samples", True, False, cost_rank=4, evidence_level="benchmark_evidence",
              evidence="Phase 3B / 3B.1 / 3C.2 (strong at short horizons on hourly energy/weather; no skill on daily returns)",
              limitations=("alpha fixed at 1.0 on standardised features (Phase 3B.1: the grid edge was hit for noisy targets)", "univariate inputs only in 4A")),
    ModelSpec("mlp_pjm_v11", "neural", "Phase 3C.1 frozen MLP (2x64) for the PJM hourly load dataset only", "PJM Phase 3A pipeline only", True, False,
              multivariate=True, cost_rank=9, evidence_level="exploratory", routed=False,
              evidence=("Phase 3C.3 one-time confirmatory test: primary endpoint (24h mean MLP/ridge MAE 0.9534 vs <= 0.95) NOT MET; "
                        "h1 2.3x worse than ridge; DUQ and EKPC worse than ridge on 24h targets; h6/h12 gains unconfirmed"),
              limitations=("not routed by the engine", "valid only for the frozen PJM feature pipeline", "PJM test block consumed: no further confirmation possible on PJM")),
]}


def snapshot() -> list[dict]:
    return [m.as_dict() for m in REGISTRY.values()]
