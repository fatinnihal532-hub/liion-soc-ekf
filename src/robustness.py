"""Two studies the headline SOC-comparison plot cannot show on its own:

1. Convergence: how fast each estimator recovers from a wrong initial
   guess, holding the model exact. This isolates initial-condition
   recovery from steady-state disturbance rejection - the two are easy
   to conflate if the main run also starts from a bad guess.

2. Model mismatch: how much each estimator degrades when its internal
   model of the battery is wrong (capacity, series resistance, OCV
   table), which is the situation any real battery management system
   is actually in. The exact-model case tests whether the estimator
   is correct; robustness testing checks whether it is useful.
"""
import csv
from pathlib import Path
import numpy as np

from .battery_model import BatteryParams
from .experiment import run_experiment

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _rmse(estimate, reference, mask=None):
    err = (estimate - reference) * 100.0
    if mask is not None:
        err = err[mask]
    return float(np.sqrt(np.mean(err**2)))


def convergence_study(bad_init_offset=0.06, seed=11):
    """Time for each estimator to settle within 0.5 percentage points
    of true SOC, starting from a deliberately wrong initial guess.
    """
    d = run_experiment(
        duration_s=1800.0,
        initial_true_soc=0.82,
        initial_est_soc=0.82 - bad_init_offset,
        seed=seed,
    )
    t = d["t"]
    results = {}
    for name, est in (("Coulomb Counting", d["cc_soc"]), ("Extended Kalman Filter", d["ekf_soc"])):
        err = np.abs((est - d["true_soc"]) * 100.0)
        settled = np.where(err < 0.5)[0]
        # first index after which err stays below 0.5 for the rest of the run
        t_settle = None
        for i in settled:
            if np.all(err[i:] < 0.5):
                t_settle = float(t[i])
                break
        results[name] = {
            "initial_error_pp": float(err[0]),
            "settle_time_s": t_settle,
        }
    return results, d


def mismatch_study(seed=11):
    """RMSE (matched initial condition, so this isolates model error)
    under increasing mismatch between the estimators' assumed battery
    model and the true plant.
    """
    scenarios = [
        ("exact model", None, 0.0),
        ("capacity 5% low", BatteryParams(capacity_ah=38.0), 0.0),
        ("R0 20% high", BatteryParams(r0_ohm=0.090), 0.0),
        ("OCV table +50 mV", None, 0.05),
        ("all three combined",
         BatteryParams(capacity_ah=38.0, r0_ohm=0.090), 0.05),
    ]
    rows = []
    for label, params, bias in scenarios:
        d = run_experiment(
            duration_s=7200.0,
            initial_true_soc=0.82,
            initial_est_soc=0.82,
            estimator_params=params,
            ocv_bias_v=bias,
            seed=seed,
        )
        settle_mask = d["t"] > 600.0  # skip the first 10 minutes of transient
        cc_rmse = _rmse(d["cc_soc"], d["true_soc"], settle_mask)
        ekf_rmse = _rmse(d["ekf_soc"], d["true_soc"], settle_mask)
        rows.append((label, cc_rmse, ekf_rmse))
    return rows


def write_reports():
    RESULTS.mkdir(exist_ok=True)

    conv, _ = convergence_study()
    with open(RESULTS / "convergence.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Estimator", "Initial error (pp)", "Time to settle within 0.5pp (s)"])
        for name, r in conv.items():
            w.writerow([name, f'{r["initial_error_pp"]:.3f}',
                        "never" if r["settle_time_s"] is None else f'{r["settle_time_s"]:.0f}'])

    rows = mismatch_study()
    with open(RESULTS / "robustness.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Scenario", "Coulomb Counting RMSE (%)", "EKF RMSE (%)"])
        for label, cc_rmse, ekf_rmse in rows:
            w.writerow([label, f"{cc_rmse:.4f}", f"{ekf_rmse:.4f}"])

    return conv, rows


if __name__ == "__main__":
    conv, rows = write_reports()
    print("Convergence from a 6-point-of-SOC wrong initial guess:")
    for name, r in conv.items():
        settle = "never" if r["settle_time_s"] is None else f'{r["settle_time_s"]:.0f} s'
        print(f'  {name:24s} settled in {settle}')
    print()
    print("Model-mismatch RMSE, % (after the first 10 minutes):")
    for label, cc_rmse, ekf_rmse in rows:
        print(f'  {label:24s} CC {cc_rmse:7.4f}   EKF {ekf_rmse:7.4f}')
