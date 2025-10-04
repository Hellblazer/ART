package com.hellblazer.art.laminar.impl;

import com.hellblazer.art.laminar.parameters.PathwayParameters;

/**
 * Default pathway parameters.
 */
public record DefaultPathwayParameters(
    double getGain,
    double getLearningRate,
    boolean isAdaptive
) implements PathwayParameters {}