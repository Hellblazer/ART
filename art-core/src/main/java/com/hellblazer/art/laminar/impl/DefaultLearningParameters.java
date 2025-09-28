package com.hellblazer.art.laminar.impl;

import com.hellblazer.art.laminar.parameters.ILearningParameters;

/**
 * Default learning parameters.
 *
 * @author Hal Hildebrand
 */
public record DefaultLearningParameters(
        double learningRate,
        double momentum,
        boolean useAdaptiveLearning,
        double weightDecay
) implements ILearningParameters {

    public static final DefaultLearningParameters DEFAULT = new DefaultLearningParameters(
            0.01, 0.9, false, 0.001
    );

    @Override
    public double getLearningRate() {
        return learningRate;
    }

    @Override
    public double getMomentum() {
        return momentum;
    }

    @Override
    public boolean useAdaptiveLearning() {
        return useAdaptiveLearning;
    }

    @Override
    public double getWeightDecay() {
        return weightDecay;
    }
}