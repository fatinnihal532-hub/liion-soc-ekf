import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.battery_model import TheveninBattery, BatteryParams
from src.experiment import run_experiment
from src.robustness import convergence_study, mismatch_study


class TestBatteryModel(unittest.TestCase):
    def test_ocv_increases_with_soc(self):
        b = TheveninBattery()
        soc = np.linspace(0.0, 1.0, 400)
        ocv = np.array([b.ocv(s) for s in soc])
        self.assertGreaterEqual(float(np.min(np.diff(ocv))), -1e-8)

    def test_docv_matches_numeric_derivative(self):
        b = TheveninBattery()
        s = np.linspace(0.02, 0.98, 200)  # away from the SOC clip boundary
        eps = 1e-6
        numeric = (b.ocv(s + eps) - b.ocv(s - eps)) / (2 * eps)
        analytic = b.docv_dsoc(s)
        self.assertLess(float(np.max(np.abs(numeric - analytic))), 1e-6)

    def test_plateau_is_flatter_than_the_knees(self):
        b = TheveninBattery()
        mid = b.docv_dsoc(np.array([0.5]))[0]
        low_knee = b.docv_dsoc(np.array([0.06]))[0]
        high_knee = b.docv_dsoc(np.array([0.94]))[0]
        self.assertLess(mid * 3, low_knee)
        self.assertLess(mid * 3, high_knee)

    def test_voltage_drops_with_positive_current(self):
        b = TheveninBattery()
        v_open = b.voltage(0.7, 0.0, 0.0)
        v_load = b.voltage(0.7, 0.0, 10.0)
        self.assertLess(v_load, v_open)

    def test_soc_bounds(self):
        b = TheveninBattery()
        soc, vrc = 0.05, 0.0
        for _ in range(10000):
            soc, vrc = b.step(soc, vrc, 100.0, 1.0)
        self.assertGreaterEqual(soc, 0.0)
        self.assertLessEqual(soc, 1.0)

    def test_coulombic_efficiency_applies_only_to_charging(self):
        b = TheveninBattery(BatteryParams(eta_coulomb=0.9))
        self.assertEqual(b.coulomb_rate(5.0), 1.0)     # discharge: full rate
        self.assertEqual(b.coulomb_rate(-5.0), 0.9)    # charge: efficiency loss
        soc0 = 0.5
        soc_discharge, _ = b.step(soc0, 0.0, 10.0, 1.0)
        soc_charge, _ = b.step(soc0, 0.0, -10.0, 1.0)
        gained = soc_charge - soc0
        lost = soc0 - soc_discharge
        self.assertAlmostEqual(gained / lost, 0.9, places=6)


class TestExperiment(unittest.TestCase):
    def test_reproducibility(self):
        a = run_experiment(duration_s=300.0, seed=11)
        b = run_experiment(duration_s=300.0, seed=11)
        np.testing.assert_allclose(a["ekf_soc"], b["ekf_soc"])

    def test_ekf_predicted_voltage_tracks_measurement(self):
        d = run_experiment(duration_s=1800.0, seed=11)
        rmse = np.sqrt(np.mean((d["ekf_voltage_pred"] - d["voltage_true"]) ** 2))
        self.assertLess(rmse, 0.02)  # well under the 10 mV sensor noise x2


class TestRobustness(unittest.TestCase):
    def test_ekf_recovers_from_bad_initial_guess_fast(self):
        conv, _ = convergence_study(bad_init_offset=0.06)
        cc_settle = conv["Coulomb Counting"]["settle_time_s"]
        ekf_settle = conv["Extended Kalman Filter"]["settle_time_s"]
        self.assertIsNone(cc_settle)          # never corrects a bad start
        self.assertIsNotNone(ekf_settle)
        self.assertLess(ekf_settle, 60.0)     # settles well within a minute

    def test_ekf_unaffected_by_capacity_error_cc_is_not(self):
        rows = {label: (cc, ekf) for label, cc, ekf in mismatch_study()}
        cc_exact, ekf_exact = rows["exact model"]
        cc_cap, ekf_cap = rows["capacity 5% low"]
        self.assertGreater(cc_cap, cc_exact * 5)     # CC drifts badly
        self.assertLess(abs(ekf_cap - ekf_exact), 0.05)  # EKF barely notices

    def test_ekf_degrades_with_bad_r0_or_ocv_but_cc_is_immune(self):
        rows = {label: (cc, ekf) for label, cc, ekf in mismatch_study()}
        cc_exact, ekf_exact = rows["exact model"]
        cc_r0, ekf_r0 = rows["R0 20% high"]
        # Coulomb counting never looks at voltage, R0, or the OCV table,
        # so it is exactly unaffected by getting them wrong.
        self.assertAlmostEqual(cc_r0, cc_exact, places=6)
        # the EKF's accuracy is bounded by how well R0 is known
        self.assertGreater(ekf_r0, ekf_exact * 5)


if __name__ == "__main__":
    unittest.main()
