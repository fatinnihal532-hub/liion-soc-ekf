import numpy as np
from .battery_model import TheveninBattery, BatteryParams
from .estimators import CoulombCounter, ExtendedKalmanFilter


def current_profile(t):
    """Synthetic EV-style signed current profile; positive = discharge."""
    base = 7.0 + 2.5 * np.sin(2 * np.pi * t / 480.0)
    accel = 12.0 * (np.sin(2 * np.pi * t / 130.0) > 0.72)
    regen = -8.0 * (np.sin(2 * np.pi * t / 85.0) < -0.80)
    rest = np.where((t % 360.0) < 35.0, -1.5, 0.0)
    return base + accel + regen + rest


def run_experiment(
    duration_s=7200.0,
    dt=1.0,
    seed=11,
    initial_true_soc=0.82,
    initial_est_soc=0.82,
    estimator_params=None,
    ocv_bias_v=0.0,
):
    """Simulate the plant and run both estimators against it.

    `estimator_params` / `ocv_bias_v` let the *estimators'* internal
    model of the battery differ from the plant that actually generates
    the data - the model-mismatch study in `src/robustness.py` uses
    this to show how each estimator degrades when its assumptions are
    wrong, which the exact-model case alone cannot show.

    `initial_est_soc` defaults to matching the plant, so the primary
    comparison is disturbance-rejection (bias + noise), not recovery
    from a wrong initial guess. Pass a different value to also see
    convergence speed from a bad start.
    """
    rng = np.random.default_rng(seed)
    plant = TheveninBattery()
    estimator_battery = TheveninBattery(
        params=estimator_params, ocv_bias_v=ocv_bias_v
    )

    t = np.arange(0.0, duration_s + dt, dt)
    current_true = current_profile(t)

    true_soc = np.empty_like(t)
    true_vrc = np.empty_like(t)
    true_voltage = np.empty_like(t)
    true_soc[0] = initial_true_soc
    true_vrc[0] = 0.0

    for k in range(len(t) - 1):
        true_voltage[k] = plant.voltage(
            true_soc[k], true_vrc[k], current_true[k]
        )
        true_soc[k + 1], true_vrc[k + 1] = plant.step(
            true_soc[k], true_vrc[k], current_true[k], dt
        )

    true_voltage[-1] = plant.voltage(
        true_soc[-1], true_vrc[-1], current_true[-1]
    )

    # Sensor model.
    current_meas = (
        current_true
        + 0.035
        + rng.normal(0.0, 0.05, size=len(t))
    )
    voltage_meas = (
        true_voltage
        + rng.normal(0.0, 0.010, size=len(t))
    )

    cc = CoulombCounter(estimator_battery, initial_est_soc)
    ekf = ExtendedKalmanFilter(estimator_battery, initial_est_soc)

    cc_soc = np.empty_like(t)
    ekf_soc = np.empty_like(t)
    ekf_vrc = np.empty_like(t)
    ekf_voltage_pred = np.empty_like(t)
    innovation = np.empty_like(t)

    for k in range(len(t)):
        cc_soc[k] = cc.update(current_meas[k], dt)
        ekf_soc[k], ekf_vrc[k], innovation[k], ekf_voltage_pred[k] = ekf.update(
            current_meas[k], voltage_meas[k], dt
        )

    return {
        "t": t,
        "current_true": current_true,
        "current_meas": current_meas,
        "voltage_true": true_voltage,
        "voltage_meas": voltage_meas,
        "ekf_voltage_pred": ekf_voltage_pred,
        "true_soc": true_soc,
        "cc_soc": cc_soc,
        "ekf_soc": ekf_soc,
        "ekf_vrc": ekf_vrc,
        "innovation": innovation,
    }
