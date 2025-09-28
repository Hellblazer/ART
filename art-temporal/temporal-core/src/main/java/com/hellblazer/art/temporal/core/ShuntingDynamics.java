package com.hellblazer.art.temporal.core;

import java.util.Map;

/**
 * Shunting dynamics implementation from Kazerounian & Grossberg (2014).
 *
 * Equation: dX_i/dt = -A_i * X_i + (B - X_i) * S_i - X_i * Σ(j≠i) I_ij
 *
 * Where:
 * - X_i: Activation of unit i
 * - A_i: Passive decay rate
 * - B: Upper bound
 * - S_i: Excitatory input
 * - I_ij: Lateral inhibition from unit j to unit i
 *
 * Equilibrium: X_eq = (B*E - C*I) / (A + E + I)
 * Stability: Requires eigenvalue < 0
 */
public final class ShuntingDynamics implements DynamicalSystem<ShuntingState, ShuntingParameters> {

    @Override
    public ShuntingState computeDerivative(ShuntingState state, ShuntingParameters params, double time) {
        var activations = state.getActivations();
        var derivatives = new double[activations.length];

        for (int i = 0; i < activations.length; i++) {
            var xi = activations[i];

            // Passive decay: -A * X_i
            var decay = -params.getDecayRate() * xi;

            // Excitatory shunting: (B - X_i) * S_i
            var excitation = (params.getUpperBound() - xi) * state.getExcitatoryInput(i);

            // Inhibitory shunting: -X_i * Σ(j≠i) I_ij
            var inhibition = 0.0;
            for (int j = 0; j < activations.length; j++) {
                if (i != j) {
                    // Simple lateral inhibition with distance-based decay
                    var distance = Math.abs(i - j);
                    var inhibitionStrength = params.getLateralInhibition() * Math.exp(-distance / 5.0);
                    inhibition += inhibitionStrength * activations[j];
                }
            }
            inhibition *= -xi;

            derivatives[i] = decay + excitation + inhibition;
        }

        return new ShuntingState(derivatives, state.getExcitatoryInputs());
    }

    @Override
    public Matrix getJacobian(ShuntingState state, ShuntingParameters params, double time) {
        var n = state.dimension();
        var jacobian = new Matrix(n, n);
        var activations = state.getActivations();

        for (int i = 0; i < n; i++) {
            var xi = activations[i];

            // Diagonal element: ∂f_i/∂x_i
            var diagonal = -params.getDecayRate() - state.getExcitatoryInput(i);

            // Add lateral inhibition contribution
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    var distance = Math.abs(i - j);
                    var inhibitionStrength = params.getLateralInhibition() * Math.exp(-distance / 5.0);
                    diagonal -= inhibitionStrength * activations[j];
                }
            }

            jacobian.set(i, i, diagonal);

            // Off-diagonal elements: ∂f_i/∂x_j (j ≠ i)
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    var distance = Math.abs(i - j);
                    var inhibitionStrength = params.getLateralInhibition() * Math.exp(-distance / 5.0);
                    jacobian.set(i, j, -xi * inhibitionStrength);
                }
            }
        }

        return jacobian;
    }

    @Override
    public ShuntingState computeEquilibrium(ShuntingParameters params, Map<String, Double> inputs) {
        var dimension = inputs.getOrDefault("dimension", 10.0).intValue();
        var equilibriumActivations = new double[dimension];
        var excitatoryInputs = new double[dimension];

        // Extract inputs
        for (int i = 0; i < dimension; i++) {
            excitatoryInputs[i] = inputs.getOrDefault("S_" + i, 0.0);
        }

        // Simplified equilibrium (no lateral inhibition for initial guess)
        for (int i = 0; i < dimension; i++) {
            var si = excitatoryInputs[i];
            if (si > 0) {
                equilibriumActivations[i] = params.getUpperBound() * si /
                                           (params.getDecayRate() + si);
            }
        }

        return new ShuntingState(equilibriumActivations, excitatoryInputs);
    }

    @Override
    public TimeScale getTimeScale() {
        return TimeScale.FAST; // Working memory dynamics at 10-100ms
    }

    @Override
    public void validateParameters(ShuntingParameters parameters) {
        parameters.validate();

        // Additional validation based on paper
        if (parameters.getUpperBound() != 1.0) {
            System.err.println("Warning: Upper bound B should be 1.0 as per paper");
        }

        if (parameters.getDecayRate() < 0.05 || parameters.getDecayRate() > 0.2) {
            System.err.println("Warning: Decay rate A outside typical range [0.05, 0.2]");
        }
    }

    /**
     * Check convergence of shunting dynamics.
     */
    public boolean hasConverged(ShuntingState current, ShuntingState previous, double threshold) {
        if (previous == null) return false;

        var currentActivations = current.getActivations();
        var previousActivations = previous.getActivations();

        double maxChange = 0.0;
        for (int i = 0; i < currentActivations.length; i++) {
            var change = Math.abs(currentActivations[i] - previousActivations[i]);
            maxChange = Math.max(maxChange, change);
        }

        return maxChange < threshold;
    }

    /**
     * Compute primacy gradient index.
     * Returns positive value if early items have higher activation.
     */
    public double computePrimacyGradient(ShuntingState state) {
        var activations = state.getActivations();
        if (activations.length < 2) return 0.0;

        var midpoint = activations.length / 2;
        var earlySum = 0.0;
        var lateSum = 0.0;

        for (int i = 0; i < midpoint; i++) {
            earlySum += activations[i];
        }
        for (int i = midpoint; i < activations.length; i++) {
            lateSum += activations[i];
        }

        var earlyAvg = earlySum / midpoint;
        var lateAvg = lateSum / (activations.length - midpoint);

        return (earlyAvg - lateAvg) / (earlyAvg + lateAvg + 1e-10);
    }
}