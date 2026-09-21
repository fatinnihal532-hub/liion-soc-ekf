import csv
from pathlib import Path
import numpy as np

from .experiment import run_experiment
from . import robustness
from .plotstyle import render, tidy, footnote
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


def main():
    # Primary comparison: matched initial condition, so this isolates
    # disturbance rejection (current-sensor bias + noise) rather than
    # recovery from a bad initial guess. See results/convergence.csv
    # for the recovery case.
    data = run_experiment(initial_est_soc=0.82)
    reference = data["true_soc"]
    estimators = {
        "Coulomb Counting": data["cc_soc"],
        "Extended Kalman Filter": data["ekf_soc"],
    }

    metrics = {}
    for name, estimate in estimators.items():
        err = (estimate - reference) * 100.0
        metrics[name] = {
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mae": float(np.mean(np.abs(err))),
            "max": float(np.max(np.abs(err))),
        }
    voltage_rmse = float(np.sqrt(np.mean(
        (data["ekf_voltage_pred"] - data["voltage_true"]) ** 2
    )))

    with open(RESULTS / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Estimator", "RMSE (%)", "MAE (%)", "Max absolute error (%)"])
        for name in estimators:
            m = metrics[name]
            writer.writerow([name, f'{m["rmse"]:.4f}', f'{m["mae"]:.4f}', f'{m["max"]:.4f}'])
        writer.writerow(["EKF terminal voltage", f"{voltage_rmse:.4f}", "", ""])

    # Convergence and model-mismatch studies (results/convergence.csv,
    # results/robustness.csv), and a bad-initial-guess run for the
    # convergence figure below.
    conv, mismatch_rows = robustness.write_reports()
    _, conv_run = robustness.convergence_study()

    def fig_soc_comparison(c):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data["t"] / 60.0, reference * 100.0, color=c["ink"],
                lw=2.0, label="True SOC", zorder=3)
        ax.plot(data["t"] / 60.0, data["cc_soc"] * 100.0, color=c["series"][1],
                label="Coulomb counting", zorder=2)
        ax.plot(data["t"] / 60.0, data["ekf_soc"] * 100.0, color=c["series"][0],
                label="EKF", zorder=2, ls="--")
        ax.set_xlabel("time (min)"); ax.set_ylabel("SOC (%)")
        ax.set_title("SOC estimation, matched initial condition")
        tidy(ax, c, legend=True, loc="upper right")
        footnote(fig, "Both estimators start at the plant's true SOC, so the "
                 "gap that opens is disturbance rejection - the 35 mA current-sensor "
                 "bias and measurement noise - not recovery from a bad guess.", c)
        return fig

    def fig_convergence(c):
        t = conv_run["t"] / 60.0
        mask = t <= 15.0
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t[mask], conv_run["true_soc"][mask] * 100.0, color=c["ink"],
                lw=2.0, label="True SOC", zorder=3)
        ax.plot(t[mask], conv_run["cc_soc"][mask] * 100.0, color=c["series"][1],
                label="Coulomb counting", zorder=2)
        ax.plot(t[mask], conv_run["ekf_soc"][mask] * 100.0, color=c["series"][0],
                label="EKF", zorder=2, ls="--")
        ax.set_xlabel("time (min)"); ax.set_ylabel("SOC (%)")
        ax.set_title("Recovering from a wrong initial guess (6 points of SOC)")
        tidy(ax, c, legend=True, loc="lower right")
        ekf_t = conv["Extended Kalman Filter"]["settle_time_s"]
        footnote(fig, f"Coulomb counting carries its starting error forward "
                 f"unchanged. The EKF settles within 0.5 percentage points in "
                 f"{ekf_t:.0f} s because the voltage measurement, not the "
                 f"current integral, tells it where it actually started.", c)
        return fig

    def fig_voltage_fit(c):
        # Full 2-hour trace at 1 Hz is ~7000 noisy points and renders as an
        # unreadable, oversized SVG; a 10-minute window shows the same
        # tracking behaviour (the current profile repeats on this scale)
        # at a fraction of the size.
        window = (data["t"] >= 3600.0) & (data["t"] <= 3900.0)
        tm = (data["t"][window] - 3600.0) / 60.0
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(tm, data["voltage_meas"][window], color=c["muted"],
                lw=0.8, alpha=0.7, label="Measured voltage")
        ax.plot(tm, data["ekf_voltage_pred"][window], color=c["series"][0],
                lw=1.2, label="EKF predicted voltage")
        ax.set_xlabel("time (min, window starting at t = 60 min)")
        ax.set_ylabel("voltage (V)")
        ax.set_title("EKF's predicted terminal voltage against the measurement")
        tidy(ax, c, legend=True)
        footnote(fig, f"A 5-minute window; predicted one step ahead of each "
                 f"measurement from the EKF's own state estimate. RMSE over "
                 f"the full 2-hour run is {voltage_rmse*1000:.1f} mV.", c)
        return fig

    def fig_estimation_error(c):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data["t"] / 60.0, (data["cc_soc"] - reference) * 100.0,
                color=c["series"][1], label="Coulomb counting error")
        ax.plot(data["t"] / 60.0, (data["ekf_soc"] - reference) * 100.0,
                color=c["series"][0], label="EKF error")
        ax.axhline(0.0, color=c["axis"], lw=0.8, ls="--")
        ax.set_xlabel("time (min)"); ax.set_ylabel("SOC error (percentage points)")
        ax.set_title("SOC estimation error, matched initial condition")
        tidy(ax, c, legend=True)
        return fig

    render(fig_soc_comparison, str(RESULTS / "soc_comparison"))
    render(fig_convergence, str(RESULTS / "convergence"))
    render(fig_voltage_fit, str(RESULTS / "voltage_fit"))
    render(fig_estimation_error, str(RESULTS / "estimation_error"))

    print("Battery SOC estimation study complete.")
    for name, m in metrics.items():
        print(f'{name}: RMSE={m["rmse"]:.3f}% | MAE={m["mae"]:.3f}% | Max={m["max"]:.3f}%')
    print(f'EKF terminal voltage RMSE: {voltage_rmse*1000:.1f} mV')
    print()
    print("Convergence from a 6-point wrong initial guess:")
    for name, r in conv.items():
        settle = "never" if r["settle_time_s"] is None else f'{r["settle_time_s"]:.0f} s'
        print(f'  {name}: settled in {settle}')
    print()
    print("Model-mismatch RMSE, % (after the first 10 minutes):")
    for label, cc_rmse, ekf_rmse in mismatch_rows:
        print(f'  {label:24s} CC {cc_rmse:7.4f}   EKF {ekf_rmse:7.4f}')


if __name__ == "__main__":
    main()
