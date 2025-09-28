package com.hellblazer.art.laminar.impl;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.laminar.core.PathwayType;
import com.hellblazer.art.laminar.parameters.IPathwayParameters;

/**
 * Bottom-up pathway implementation for feedforward signal propagation.
 * Carries sensory and feature information from lower to higher layers.
 *
 * @author Hal Hildebrand
 */
public class BottomUpPathway extends AbstractPathway {

    private final boolean useNormalization;

    public BottomUpPathway(String sourceLayerId, String targetLayerId,
                          int sourceSize, int targetSize) {
        this(sourceLayerId, targetLayerId, sourceSize, targetSize, true);
    }

    public BottomUpPathway(String sourceLayerId, String targetLayerId,
                          int sourceSize, int targetSize, boolean useNormalization) {
        super(PathwayType.BOTTOM_UP, sourceLayerId, targetLayerId, sourceSize, targetSize);
        this.useNormalization = useNormalization;
    }

    @Override
    protected Pattern processSignal(Pattern signal, IPathwayParameters parameters) {
        // Bottom-up signals often need normalization for stability
        if (useNormalization) {
            signal = normalizeSignal(signal);
        }

        // Apply competitive dynamics if specified
        if (parameters.isCompetitive()) {
            signal = applyCompetition(signal);
        }

        return signal;
    }

    /**
     * Normalize bottom-up signal to prevent unbounded growth.
     */
    private Pattern normalizeSignal(Pattern signal) {
        var l2Norm = signal.l2Norm();
        if (l2Norm > 0) {
            return signal.scale(1.0 / l2Norm);
        }
        return signal;
    }

    /**
     * Apply competitive dynamics to enhance contrast.
     */
    private Pattern applyCompetition(Pattern signal) {
        var result = new double[signal.dimension()];

        // Calculate mean activation
        var mean = 0.0;
        for (int i = 0; i < signal.dimension(); i++) {
            mean += signal.get(i);
        }
        mean /= signal.dimension();

        // Apply contrast enhancement
        for (int i = 0; i < signal.dimension(); i++) {
            var value = signal.get(i);
            if (value > mean) {
                // Enhance above-average activations
                result[i] = value * 1.2;
            } else {
                // Suppress below-average activations
                result[i] = value * 0.8;
            }
        }

        return new DenseVector(result);
    }

    @Override
    public void updateWeights(Pattern source, Pattern target, double learningRate) {
        if (!isAdaptive()) return;

        // ART-style learning: move weights towards input pattern
        for (int i = 0; i < weights.getRows(); i++) {
            if (i < target.dimension() && target.get(i) > 0) {
                // Update weights only for active target neurons
                for (int j = 0; j < weights.getCols(); j++) {
                    var inputValue = j < source.dimension() ? source.get(j) : 0.0;
                    var currentWeight = weights.get(i, j);

                    // Fast learning: w_new = input
                    // Slow learning: w_new = w_old + β(input - w_old)
                    var beta = learningRate;
                    var newWeight = currentWeight + beta * (inputValue - currentWeight);

                    weights.set(i, j, Math.max(0.0, Math.min(1.0, newWeight)));
                }
            }
        }
    }
}