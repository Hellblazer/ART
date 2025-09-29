package com.hellblazer.art.laminar.impl;

import com.hellblazer.art.laminar.parameters.LaminarShuntingParameters;

/**
 * Default shunting parameters.
 */
public record DefaultShuntingParameters(
    double getDecayRate,
    double getCeiling,
    double getFloor,
    double getSelfExcitation
) implements com.hellblazer.art.laminar.parameters.LaminarShuntingParameters {}