import numpy as np
from .battery_model import TheveninBattery


class CoulombCounter:
    def __init__(self, battery, initial_soc):
        self.battery = battery
        self.soc = float(np.clip(initial_soc, 0.0, 1.0))

    def update(self, current, dt):
        rate = self.battery.coulomb_rate(current)
        self.soc -= rate * current * dt / self.battery.capacity_c
        self.soc = float(np.clip(self.soc, 0.0, 1.0))
        return self.soc


class ExtendedKalmanFilter:
    """Two-state EKF with x = [SOC, V_Rc]^T."""

    def __init__(self, battery, initial_soc, initial_vrc=0.0,
                 p0=(0.02, 0.05), q=(3e-8, 2e-7), r=0.010):
        self.battery = battery
        self.x = np.array(
            [np.clip(initial_soc, 0.0, 1.0), initial_vrc],
            dtype=float
        )
        self.P = np.diag([p0[0]**2, p0[1]**2])
        self.Q = np.diag([q[0], q[1]])
        self.R = np.array([[r**2]])

    def update(self, current, measured_voltage, dt):
        p = self.battery.p
        soc, vrc = self.x
        alpha = np.exp(-dt / (p.rp_ohm * p.cp_farad))
        rate = self.battery.coulomb_rate(current)

        # Prediction
        soc_pred = soc - rate * current * dt / self.battery.capacity_c
        soc_pred = float(np.clip(soc_pred, 0.0, 1.0))
        vrc_pred = alpha * vrc + p.rp_ohm * (1.0 - alpha) * current

        F = np.array([[1.0, 0.0], [0.0, alpha]])
        x_pred = np.array([soc_pred, vrc_pred])
        P_pred = F @ self.P @ F.T + self.Q

        # Measurement prediction and linearization
        v_pred = self.battery.voltage(soc_pred, vrc_pred, current)
        H = np.array([[self.battery.docv_dsoc(soc_pred), -1.0]])

        innovation_cov = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(innovation_cov)
        residual = float(measured_voltage - v_pred)

        x_new = x_pred + K[:, 0] * residual
        x_new[0] = np.clip(x_new[0], 0.0, 1.0)

        # Joseph-form covariance update
        I = np.eye(2)
        KH = K @ H
        self.P = (
            (I - KH) @ P_pred @ (I - KH).T
            + K @ self.R @ K.T
        )
        self.P = 0.5 * (self.P + self.P.T)
        self.x = x_new
        return float(self.x[0]), float(self.x[1]), residual, float(v_pred)
