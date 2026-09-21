# Methodology

## 1. Battery parameterization

The simulation uses representative pack-level parameters rather than claiming a specific manufacturer cell:

- Nominal capacity: 40 Ah
- Nominal operating-voltage region: approximately 44-49 V (12s-equivalent, LFP-like)
- Series resistance R0: 75 mΩ
- Polarization resistance Rp: 30 mΩ
- Polarization capacitance Cp: 1800 F
- Coulombic efficiency: 0.985, applied only while charging

The OCV(SOC) curve is a closed-form analytical approximation built from a linear base term plus two `tanh` "knee" terms, giving a flat plateau across the middle of the SOC range and steep transitions near 0% and 100% SOC. This mirrors the shape reported for LFP-type cells, where OCV alone carries almost no SOC information away from the knees and a fuel gauge must lean on the accumulated charge count or an internal model to stay accurate mid-range.

## 2. Circuit model

The plant is a first-order Thevenin equivalent circuit: a series resistance R0 in line with the OCV source, and a parallel Rp/Cp pair (not a series pair) modeling polarization relaxation.

```text
      R0        +----- Rp -----+
OCV ---/\/\/----+               +---- V_terminal
                +----- Cp ------+
```

Two states are propagated: SOC and the RC-branch voltage V_Rc. Terminal voltage is

```math
V_{term} = OCV(SOC) - I \cdot R_0 - V_{Rc}
```

with the sign convention that positive current I is discharge.

## 3. Measurement model

The simulated "true" plant runs the exact model above. The measurements available to both estimators are corrupted with:

- 0.05 A current-sensor noise (zero-mean)
- 10 mV voltage-sensor noise (zero-mean)
- 35 mA constant current-sensor bias

The bias is what gives Coulomb counting something to drift on, and gives the voltage-corrected EKF a meaningful correction to make. Away from that bias, plain integration and the EKF should track each other closely, which the results intentionally show rather than hide.

## 4. Estimators

### Coulomb counting

The measured current is integrated directly, scaled by the nominal capacity and by `coulomb_rate(current)`, which returns the coulombic-efficiency factor only while charging (current < 0 in this convention) and 1.0 while discharging. Coulomb counting has no access to the voltage measurement or the OCV table at all, which is the source of both its main weakness (unbounded drift from any current-sensor bias) and its main strength (total immunity to errors in R0, Rp, Cp, or the OCV table, since it never uses them).

### Extended Kalman Filter

State vector: `x = [SOC, V_Rc]`.

Prediction:
- propagate SOC from measured current (same coulomb-rate logic as above),
- propagate V_Rc through the RC branch's discrete-time dynamics.

Correction:
- predict terminal voltage from the internal model's OCV(SOC), R0, and V_Rc,
- form the voltage innovation (measured minus predicted terminal voltage),
- linearize the measurement with `H = [dOCV/dSOC, -1]`,
- update the state and covariance with the Kalman gain, using a Joseph-form covariance update for numerical symmetry and positive-semidefiniteness.

Because the correction step depends on the EKF's own internal copy of the battery model, the estimator's accuracy is only as good as that internal model. The project tests this directly (see Section 6) rather than assuming a perfect internal model, which is the assumption that makes most textbook EKF demonstrations look better than they would in practice.

## 5. Evaluation

The hidden simulated plant SOC is the reference. Both estimators only ever see measured current and measured terminal voltage, never the true SOC or the true model parameters.

Three separate scenarios are evaluated, deliberately kept apart because they answer different questions:

1. **Matched-initial-condition run** - both estimators start from the true SOC, and the estimator's internal model exactly matches the plant. This isolates disturbance rejection (how well each handles the current-sensor bias) and is the fairest apples-to-apples comparison.
2. **Convergence from a bad initial guess** - the estimator starts several percentage points away from the true SOC, with everything else matched. This isolates recovery speed, not disturbance rejection.
3. **Model mismatch** - the EKF's internal battery model is given a capacity error, an R0 error, an OCV bias, or all three, while the plant keeps its true parameters. This tests robustness to the model inaccuracies a real fuel gauge would actually have.

Keeping these separate avoids the most common mistake in this kind of demonstration: mixing a bad initial guess into a "steady-state accuracy" comparison, which makes an estimator look better at rejecting disturbances than it actually is.

## 6. Reproducibility

All experiments use a fixed random seed for the noise sequences. `verify.py` re-derives every reported number from first principles (closed-form OCV slope, analytic-vs-numeric derivative comparison, coulombic-efficiency asymmetry, convergence timing, and the mismatch-sensitivity ordering) and exits non-zero if any of them stop holding, so CI catches a regression before it reaches the README.
