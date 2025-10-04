package com.hellblazer.art.laminar.impl;

import com.hellblazer.art.laminar.parameters.ResonanceParameters;

/**
 * Default resonance parameters.
 */
public record DefaultResonanceParameters(
    double getInitialVigilance,
    double getVigilanceIncrement,
    double getMaxVigilance,
    boolean isAdaptiveVigilance
) implements ResonanceParameters {}