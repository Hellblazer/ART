package com.hellblazer.art.temporal.core;

import java.util.Map;

/**
 * Competitive instar learning dynamics from Kazerounian & Grossberg (2014).
 *
 * Equation: dW_ij/dt = L * Y_j * (X_i * Z_i - W_ij)
 *
 * Where:
 * - W_ij: Weight from input i to category j
 * - L: Learning rate (typically 0.1)
 * - Y_j: Category activation (from masking field)
 * - X_i: Input activation (from working memory)
 * - Z_i: Transmitter level (creates primacy gradient)
 *
 * Key features:
 * - Weight normalization: Σw_ij = 1 for each j
 * - Competitive learning: Only active categories learn
 * - Transmitter gating: Early items learn more strongly
 */
public final class InstarLearningDynamics implements DynamicalSystem<InstarLearningState, InstarLearningParameters> {

    // External inputs needed for learning
    private double[] inputActivations;      // X_i from working memory
    private double[] transmitterLevels;     // Z_i from transmitter gates

    public void setInputActivations(double[] activations) {
        this.inputActivations = activations.clone();
    }

    public void setTransmitterLevels(double[] levels) {
        this.transmitterLevels = levels.clone();
    }

    @Override
    public InstarLearningState computeDerivative(InstarLearningState state, InstarLearningParameters params, double time) {
        var weights = state.getWeights();
        var categoryActivations = state.getCategoryActivations();
        var numCategories = params.getNumCategories();
        var numInputs = params.getInputDimension();

        // Validate inputs
        if (inputActivations == null || inputActivations.length != numInputs) {
            throw new IllegalStateException("Input activations not properly set");
        }
        if (transmitterLevels == null || transmitterLevels.length != numInputs) {
            throw new IllegalStateException("Transmitter levels not properly set");
        }

        var derivatives = new double[numCategories][numInputs];

        for (int j = 0; j < numCategories; j++) {
            var yj = categoryActivations[j];

            // Only learn if category is active (winner-take-all)
            if (yj > params.getResetThreshold()) {
                for (int i = 0; i < numInputs; i++) {
                    var xi = inputActivations[i];
                    var zi = transmitterLevels[i];
                    var wij = weights[j][i];

                    // Instar learning rule with transmitter gating
                    var target = xi * zi; // Transmitter gates the learning signal
                    derivatives[j][i] = params.getLearningRate() * yj * (target - wij);
                }
            }
        }

        // Apply weight normalization constraint
        if (params.isNormalizationEnabled()) {
            applyNormalizationConstraint(derivatives, weights);
        }

        // Create new state with weight derivatives
        var newState = new InstarLearningState(weights);

        // Apply derivatives (for visualization/tracking)
        for (int j = 0; j < numCategories; j++) {
            for (int i = 0; i < numInputs; i++) {
                weights[j][i] += derivatives[j][i] * 0.01; // Small step for derivative
            }
        }

        return newState;
    }

    private void applyNormalizationConstraint(double[][] derivatives, double[][] weights) {
        // Ensure weight changes maintain normalization: Σw_ij = 1
        for (int j = 0; j < derivatives.length; j++) {
            var sumDerivative = 0.0;
            var sumWeight = 0.0;

            for (int i = 0; i < derivatives[j].length; i++) {
                sumDerivative += derivatives[j][i];
                sumWeight += weights[j][i];
            }

            // Redistribute to maintain sum = 1
            if (Math.abs(sumDerivative) > 1e-10) {
                var correction = sumDerivative / derivatives[j].length;
                for (int i = 0; i < derivatives[j].length; i++) {
                    derivatives[j][i] -= correction;
                }
            }

            // Renormalize if weights have drifted
            if (Math.abs(sumWeight - 1.0) > 0.01) {
                for (int i = 0; i < derivatives[j].length; i++) {
                    derivatives[j][i] += 0.1 * (1.0 - sumWeight) / derivatives[j].length;
                }
            }
        }
    }

    @Override
    public Matrix getJacobian(InstarLearningState state, InstarLearningParameters params, double time) {
        // Weight dynamics Jacobian is sparse and depends on activations
        var dim = params.getNumCategories() * params.getInputDimension();
        var jacobian = new Matrix(dim, dim);

        var categoryActivations = state.getCategoryActivations();

        for (int idx = 0; idx < dim; idx++) {
            int j = idx / params.getInputDimension();
            int i = idx % params.getInputDimension();

            // Diagonal element: -L * Y_j
            if (categoryActivations[j] > params.getResetThreshold()) {
                jacobian.set(idx, idx, -params.getLearningRate() * categoryActivations[j]);
            }
        }

        return jacobian;
    }

    @Override
    public InstarLearningState computeEquilibrium(InstarLearningParameters params, Map<String, Double> inputs) {
        // At equilibrium: W_ij = X_i * Z_i (for active categories)
        var numCategories = params.getNumCategories();
        var numInputs = params.getInputDimension();
        var equilibrium = new double[numCategories][numInputs];

        // Extract steady-state values
        var xi = new double[numInputs];
        var zi = new double[numInputs];

        for (int i = 0; i < numInputs; i++) {
            xi[i] = inputs.getOrDefault("X_" + i, 0.0);
            zi[i] = inputs.getOrDefault("Z_" + i, 1.0);
        }

        // Initialize weights to normalized values
        for (int j = 0; j < numCategories; j++) {
            var sum = 0.0;
            for (int i = 0; i < numInputs; i++) {
                equilibrium[j][i] = xi[i] * zi[i];
                sum += equilibrium[j][i];
            }

            // Normalize
            if (sum > 0) {
                for (int i = 0; i < numInputs; i++) {
                    equilibrium[j][i] /= sum;
                }
            } else {
                // Default uniform distribution
                for (int i = 0; i < numInputs; i++) {
                    equilibrium[j][i] = 1.0 / numInputs;
                }
            }
        }

        return new InstarLearningState(equilibrium);
    }

    @Override
    public TimeScale getTimeScale() {
        return TimeScale.VERY_SLOW; // Weight adaptation at 1000-10000ms
    }

    @Override
    public void validateParameters(InstarLearningParameters parameters) {
        parameters.validate();

        if (parameters.getLearningRate() < 0.01 || parameters.getLearningRate() > 1.0) {
            throw new IllegalArgumentException(
                "Learning rate should be in range [0.01, 1.0], paper uses 0.1");
        }

        if (parameters.getResetThreshold() < 0.0 || parameters.getResetThreshold() > 1.0) {
            throw new IllegalArgumentException(
                "Reset threshold must be in range [0, 1]");
        }
    }

    /**
     * Calculate the strength of primacy gradient in learning.
     * Measures how much more strongly early items are encoded.
     */
    public double calculatePrimacyStrength(InstarLearningState state) {
        if (transmitterLevels == null || transmitterLevels.length < 2) {
            return 0.0;
        }

        var weights = state.getWeights();
        var numCategories = weights.length;
        var numInputs = weights[0].length;
        var midpoint = numInputs / 2;

        var earlyWeightSum = 0.0;
        var lateWeightSum = 0.0;

        for (int j = 0; j < numCategories; j++) {
            for (int i = 0; i < midpoint; i++) {
                earlyWeightSum += Math.abs(weights[j][i]) * transmitterLevels[i];
            }

            for (int i = midpoint; i < numInputs; i++) {
                lateWeightSum += Math.abs(weights[j][i]) * transmitterLevels[i];
            }
        }

        var earlyAvg = earlyWeightSum / (midpoint * numCategories);
        var lateAvg = lateWeightSum / ((numInputs - midpoint) * numCategories);

        return (earlyAvg - lateAvg) / (earlyAvg + lateAvg + 1e-10);
    }
}