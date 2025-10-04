package com.hellblazer.art.laminar.impl;

import com.hellblazer.art.laminar.parameters.LearningParameters;

/**
 * Default learning parameters.
 */
public record DefaultLearningParameters(
    double getLearningRate,
    double getMomentum,
    boolean isFastLearning,
    double getWeightDecay
) implements LearningParameters {}