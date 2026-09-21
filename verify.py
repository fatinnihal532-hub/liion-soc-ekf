"""Battery SOC estimation - model against theory and the estimators against
each other. Ten checks; exits non-zero if any fails.
"""
import sys
import numpy as np

from src.battery_model import TheveninBattery, BatteryParams
from src.experiment import run_experiment
from src.robustness import convergence_study, mismatch_study

PASS, FAIL = 0, 0


def check(name, ok, detail=""):
    global PASS, FAIL
    tag = "pass" if ok else "FAIL"
    print(f"  [{tag}] {name}")
    if detail:
        print(f"         {detail}")
    if ok:
        PASS += 1
    else:
        FAIL += 1


def main():
    print("Li-ion SOC estimation - model against theory")
    print("=" * 60)

    b = TheveninBattery()

    # 1. OCV derivative matches its own closed-form numeric derivative.
    s = np.linspace(0.02, 0.98, 200)
    eps = 1e-6
    numeric = (b.ocv(s + eps) - b.ocv(s - eps)) / (2 * eps)
    analytic = b.docv_dsoc(s)
    err = float(np.max(np.abs(numeric - analytic)))
    check("dOCV/dSOC matches numerical derivative of OCV(SOC)",
          err < 1e-6, f"max error {err:.2e} (tol 1e-6)")

    # 2. OCV is monotonic (a real cell's OCV never falls with rising SOC).
    ocv_curve = b.ocv(np.linspace(0.0, 1.0, 1000))
    check("OCV(SOC) is monotonically non-decreasing",
          bool(np.all(np.diff(ocv_curve) >= -1e-9)))

    # 3. The plateau is flat relative to the knees (this is what makes the
    #    estimation problem realistic rather than easy).
    mid = float(b.docv_dsoc(np.array([0.5]))[0])
    knee = float(b.docv_dsoc(np.array([0.06]))[0])
    check("mid-SOC plateau slope is >3x flatter than the low-SOC knee",
          mid * 3 < knee, f"mid {mid:.2f} V/SOC, knee {knee:.2f} V/SOC")

    # 4. Coulombic efficiency applies only to charging.
    bp = TheveninBattery(BatteryParams(eta_coulomb=0.9))
    gained = bp.step(0.5, 0.0, -10.0, 1.0)[0] - 0.5
    lost = 0.5 - bp.step(0.5, 0.0, 10.0, 1.0)[0]
    ratio = gained / lost
    check("coulombic efficiency (0.9) reduces charge accepted, not charge removed",
          abs(ratio - 0.9) < 1e-6, f"gained/lost = {ratio:.6f}, expected 0.900000")

    # 5. Reproducibility: same seed, same result.
    a = run_experiment(duration_s=300.0, seed=11)
    r = run_experiment(duration_s=300.0, seed=11)
    same = bool(np.allclose(a["ekf_soc"], r["ekf_soc"]))
    check("same seed reproduces identical EKF trajectory", same)

    # 6. EKF's predicted voltage tracks the true (noise-free) voltage,
    #    not just the noisy measurement it was fed.
    d = run_experiment(duration_s=1800.0, seed=11)
    v_rmse = float(np.sqrt(np.mean((d["ekf_voltage_pred"] - d["voltage_true"]) ** 2)))
    check("EKF predicted voltage tracks true terminal voltage",
          v_rmse < 0.02, f"RMSE {v_rmse*1000:.1f} mV (tol 20 mV)")

    # 7. Matched-initial-condition run: with an exact model, both
    #    estimators should be accurate; the EKF should not be worse
    #    than a modest multiple of the Coulomb counter.
    reference = d["true_soc"]
    mask = d["t"] > 600.0
    cc_rmse = float(np.sqrt(np.mean(((d["cc_soc"] - reference) * 100)[mask] ** 2)))
    ekf_rmse = float(np.sqrt(np.mean(((d["ekf_soc"] - reference) * 100)[mask] ** 2)))
    check("both estimators track true SOC to within 0.5 pp RMSE (matched model)",
          cc_rmse < 0.5 and ekf_rmse < 0.5,
          f"CC {cc_rmse:.4f}%, EKF {ekf_rmse:.4f}%")

    # 8. Convergence from a wrong initial guess: CC never corrects it,
    #    the EKF does, quickly.
    conv, _ = convergence_study(bad_init_offset=0.06)
    cc_settle = conv["Coulomb Counting"]["settle_time_s"]
    ekf_settle = conv["Extended Kalman Filter"]["settle_time_s"]
    check("Coulomb counting never recovers from a wrong initial SOC",
          cc_settle is None)
    check("EKF settles within 0.5 pp of true SOC in under 60 s",
          ekf_settle is not None and ekf_settle < 60.0,
          f"settled in {ekf_settle} s")

    # 9/10. Model mismatch: Coulomb counting is exactly invariant to R0/OCV
    #    error (it never uses them); the EKF is not.
    rows = {label: (cc, ekf) for label, cc, ekf in mismatch_study()}
    cc_exact, ekf_exact = rows["exact model"]
    cc_r0, ekf_r0 = rows["R0 20% high"]
    cc_cap, ekf_cap = rows["capacity 5% low"]
    check("Coulomb counting is exactly unaffected by a wrong R0/OCV table",
          abs(cc_r0 - cc_exact) < 1e-6,
          f"CC RMSE exact {cc_exact:.4f}% vs R0-mismatched {cc_r0:.4f}%")
    check("EKF is far more sensitive to R0 error than to capacity error",
          (ekf_r0 - ekf_exact) > 5 * abs(ekf_cap - ekf_exact),
          f"EKF degradation: R0 {ekf_r0-ekf_exact:+.4f} pp vs capacity {ekf_cap-ekf_exact:+.4f} pp")

    print("=" * 60)
    print(f"{PASS} checks passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
