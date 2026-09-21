import numpy as np
from dataclasses import dataclass


@dataclass
class BatteryParams:
    capacity_ah: float = 40.0
    eta_coulomb: float = 0.995
    r0_ohm: float = 0.075
    rp_ohm: float = 0.030
    cp_farad: float = 1800.0


class TheveninBattery:
    """First-order Thevenin battery model with SOC-dependent OCV.

    The OCV curve has a flat mid-SOC plateau and two steep knees near
    0% and 100%, the shape that makes SOC estimation from voltage
    alone hard in the range that matters for daily use: a filter earns
    its keep mainly near the knees, not in the flat middle.

    `ocv_bias_v` is not part of the physical model. It exists so an
    *estimator's copy* of this class can be given a wrong OCV table
    (a constant offset) while the plant it is estimating keeps the
    correct one - see `experiment.run_experiment(..., ocv_bias_v=...)`.
    """

    _S0 = 4.2
    _A1, _K1, _S1 = 20.0, 22.0, 0.06
    _A2, _K2, _S2 = 16.0, 22.0, 0.94
    _BASE = 41.5 - (
        (_A1 / _K1) * np.tanh(_K1 * (0.0 - _S1))
        + (_A2 / _K2) * np.tanh(_K2 * (0.0 - _S2))
    )

    def __init__(self, params=None, ocv_bias_v=0.0):
        self.p = params or BatteryParams()
        self.capacity_c = self.p.capacity_ah * 3600.0
        self.ocv_bias_v = ocv_bias_v

    def ocv(self, soc):
        """Representative pack OCV curve for 0 <= SOC <= 1.

        Flat plateau in the middle (~4.2 V per unit SOC), steep knees
        near empty and full (~20-24 V per unit SOC).
        """
        s = np.clip(soc, 0.0, 1.0)
        return (
            self._BASE
            + self._S0 * s
            + (self._A1 / self._K1) * np.tanh(self._K1 * (s - self._S1))
            + (self._A2 / self._K2) * np.tanh(self._K2 * (s - self._S2))
            + self.ocv_bias_v
        )

    def docv_dsoc(self, soc):
        s = np.clip(soc, 0.0, 1.0)
        z1 = self._K1 * (s - self._S1)
        z2 = self._K2 * (s - self._S2)
        return (
            self._S0
            + self._A1 * (1.0 - np.tanh(z1) ** 2)
            + self._A2 * (1.0 - np.tanh(z2) ** 2)
        )

    def voltage(self, soc, v_rc, current):
        """Terminal voltage. Positive current = discharge."""
        return self.ocv(soc) - current * self.p.r0_ohm - v_rc

    def coulomb_rate(self, current):
        """Charge-removal rate per unit current, in the sign convention
        SOC_dot = -rate * current.

        Coulombic efficiency applies only to charging: some of the
        charge pushed in during charging is lost to side reactions, so
        less SOC is gained than the current alone implies. Discharge
        removes charge at its full rate.
        """
        return 1.0 if current >= 0.0 else self.p.eta_coulomb

    def step(self, soc, v_rc, current, dt):
        """Advance the hidden plant state by one timestep."""
        alpha = np.exp(-dt / (self.p.rp_ohm * self.p.cp_farad))
        rate = self.coulomb_rate(current)
        soc_next = np.clip(
            soc - rate * current * dt / self.capacity_c,
            0.0, 1.0
        )
        vrc_next = alpha * v_rc + self.p.rp_ohm * (1.0 - alpha) * current
        return float(soc_next), float(vrc_next)
