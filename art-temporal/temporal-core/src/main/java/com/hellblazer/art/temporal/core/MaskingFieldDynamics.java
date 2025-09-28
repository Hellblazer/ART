package com.hellblazer.art.temporal.core;

import java.util.Map;

/**
 * Multi-scale masking field dynamics from Kazerounian & Grossberg (2014).
 *
 * Equation: dY_jk/dt = -α * Y_jk + (β - Y_jk) * [f(Y_jk) + I_jk] - Y_jk * Σ g(Y_lm)
 *
 * Where:
 * - Y_jk: Activity of masking field node at position j, scale k
 * - α: Decay rate
 * - β: Upper bound
 * - f(): Self-excitation function
 * - I_jk: External input
 * - g(): Lateral inhibition function
 *
 * Key feature: Asymmetric inhibition between scales creates hierarchical processing.
 */
public final class MaskingFieldDynamics implements DynamicalSystem<MaskingFieldState, MaskingFieldParameters> {

    @Override
    public MaskingFieldState computeDerivative(MaskingFieldState state, MaskingFieldParameters params, double time) {
        var numScales = params.getNumScales();
        var fieldSize = params.getFieldSize();
        var cellActivations = state.getCellActivations();
        var derivatives = new double[numScales][fieldSize];

        for (int scale = 0; scale < numScales; scale++) {
            for (int pos = 0; pos < fieldSize; pos++) {
                var yjk = cellActivations[scale][pos];

                // Passive decay: -α * Y_jk
                var decay = -params.getAlpha() * yjk;

                // Self-excitation: (β - Y_jk) * f(Y_jk)
                var selfExcitation = (params.getBeta() - yjk) *
                                   sigmoid(yjk, 0.2) * params.getSelfExcitation();

                // External input: (β - Y_jk) * I_jk
                var externalInput = (params.getBeta() - yjk) * getInput(state, scale, pos);

                // Lateral inhibition: -Y_jk * Σ g(Y_lm)
                var inhibition = computeLateralInhibition(state, params, scale, pos);

                derivatives[scale][pos] = decay + selfExcitation + externalInput - yjk * inhibition;
            }
        }

        // Create new state with derivatives (would be integrated by numerical method)
        return new MaskingFieldState(derivatives, state.getCellActivations(), state.getPreferredLengths());
    }

    private double getInput(MaskingFieldState state, int scale, int pos) {
        // Input would come from working memory - placeholder for now
        return 0.0;
    }

    private double computeLateralInhibition(MaskingFieldState state, MaskingFieldParameters params,
                                           int currentScale, int currentPos) {
        double totalInhibition = 0.0;
        var cellActivations = state.getCellActivations();

        // Within-scale lateral inhibition
        for (int pos = 0; pos < params.getFieldSize(); pos++) {
            if (pos == currentPos) continue;

            var activity = cellActivations[currentScale][pos];
            var distance = Math.abs(pos - currentPos);
            var spatialFactor = Math.exp(-distance / 5.0); // Spatial decay

            totalInhibition += sigmoid(activity, 0.1) * spatialFactor;
        }

        // Between-scale asymmetric inhibition
        if (params.isMultiScaleEnabled()) {
            for (int scale = 0; scale < params.getNumScales(); scale++) {
                if (scale == currentScale) continue;

                // Sum activity at other scale
                var scaleActivity = 0.0;
                for (int pos = 0; pos < params.getFieldSize(); pos++) {
                    scaleActivity += cellActivations[scale][pos];
                }

                // Apply asymmetric inhibition
                var inhibitionStrength = params.computeInterScaleInhibition(scale, currentScale);
                totalInhibition += scaleActivity * inhibitionStrength;
            }
        }

        return totalInhibition * params.getLateralInhibition();
    }

    private double sigmoid(double x, double threshold) {
        return 1.0 / (1.0 + Math.exp(-(x - threshold) * 10.0));
    }

    @Override
    public Matrix getJacobian(MaskingFieldState state, MaskingFieldParameters params, double time) {
        var dim = state.dimension();
        var jacobian = new Matrix(dim, dim);
        var numScales = params.getNumScales();
        var fieldSize = params.getFieldSize();

        // Convert 2D indices to 1D for matrix
        for (int idx1 = 0; idx1 < dim; idx1++) {
            int scale1 = idx1 / fieldSize;
            int pos1 = idx1 % fieldSize;

            for (int idx2 = 0; idx2 < dim; idx2++) {
                int scale2 = idx2 / fieldSize;
                int pos2 = idx2 % fieldSize;

                double value = 0.0;

                if (idx1 == idx2) {
                    // Diagonal element
                    value = -params.getAlpha() - computeLateralInhibition(state, params, scale1, pos1);
                } else {
                    // Off-diagonal: inhibition coupling
                    var yjk = state.getActivation(scale1, pos1);

                    if (scale1 == scale2) {
                        // Within-scale coupling
                        var distance = Math.abs(pos1 - pos2);
                        var spatialFactor = Math.exp(-distance / 5.0);
                        value = -yjk * params.getLateralInhibition() * spatialFactor;
                    } else {
                        // Between-scale coupling
                        var inhibitionStrength = params.computeInterScaleInhibition(scale2, scale1);
                        value = -yjk * inhibitionStrength;
                    }
                }

                jacobian.set(idx1, idx2, value);
            }
        }

        return jacobian;
    }

    @Override
    public MaskingFieldState computeEquilibrium(MaskingFieldParameters params, Map<String, Double> inputs) {
        // Simplified equilibrium: assume no lateral inhibition initially
        var numScales = params.getNumScales();
        var fieldSize = params.getFieldSize();
        var equilibrium = new double[numScales][fieldSize];
        var preferredLengths = new int[numScales];

        for (int scale = 0; scale < numScales; scale++) {
            preferredLengths[scale] = params.getPreferredLength(scale);

            for (int pos = 0; pos < fieldSize; pos++) {
                var input = inputs.getOrDefault(String.format("I_%d_%d", scale, pos), 0.0);
                if (input > 0) {
                    equilibrium[scale][pos] = params.getBeta() * input / (params.getAlpha() + input);
                }
            }
        }

        return new MaskingFieldState(equilibrium, equilibrium, preferredLengths);
    }

    @Override
    public TimeScale getTimeScale() {
        return TimeScale.MEDIUM; // Masking field operates at 50-500ms
    }

    @Override
    public void validateParameters(MaskingFieldParameters parameters) {
        parameters.validate();

        if (parameters.getAsymmetryFactor() <= 1.0 && parameters.isMultiScaleEnabled()) {
            System.err.println("Warning: Asymmetry factor should be > 1 for proper hierarchical processing");
        }

        if (parameters.getFieldSize() < 10) {
            throw new IllegalArgumentException("Field size too small for meaningful processing");
        }
    }

    /**
     * Identify chunks in the masking field based on activation patterns.
     */
    public ChunkIdentification identifyChunks(MaskingFieldState state, MaskingFieldParameters params) {
        var winner = state.findGlobalWinner();

        if (winner.isValid()) {
            var preferredLength = params.getPreferredLength(winner.scale());
            return new ChunkIdentification(winner.scale(), winner.position(),
                                          preferredLength, winner.activation());
        }

        return ChunkIdentification.EMPTY;
    }

    public record ChunkIdentification(int scale, int position, int length, double strength) {
        public static final ChunkIdentification EMPTY = new ChunkIdentification(-1, -1, 0, 0.0);

        public boolean isValid() {
            return scale >= 0 && position >= 0 && length > 0;
        }
    }
}