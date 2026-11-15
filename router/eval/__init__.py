"""Evaluation harness: frontier metrics, baselines, bootstrap, calibration, plots."""
from router.eval.baselines import (
    always_cheap,
    always_strong,
    heuristic_scores,
    oracle_scores,
    random_quality_at_cost,
    random_scores,
)
from router.eval.bootstrap import bootstrap_ci, bootstrap_many
from router.eval.calibration import expected_calibration_error, reliability_curve
from router.eval.latency import measure_overhead
from router.eval.metrics import (
    EvalData,
    apgr,
    auc_cost_quality,
    calibrate_threshold,
    cost_at_quality,
    cpt,
    frontier,
    operating_point,
    quality_at_budget,
    route_at_threshold,
)
from router.eval.robustness import perturb_case, perturb_filler, perturb_typos

__all__ = [
    "EvalData", "frontier", "apgr", "auc_cost_quality", "cpt", "cost_at_quality",
    "quality_at_budget", "calibrate_threshold", "route_at_threshold", "operating_point",
    "always_cheap", "always_strong", "oracle_scores", "random_scores",
    "random_quality_at_cost", "heuristic_scores", "bootstrap_ci", "bootstrap_many",
    "expected_calibration_error", "reliability_curve", "perturb_typos", "perturb_case",
    "perturb_filler", "measure_overhead",
]
