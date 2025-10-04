package com.hellblazer.art.laminar.impl;

import com.hellblazer.art.laminar.parameters.LayerParameters;

/**
 * Default layer parameters.
 */
public record DefaultLayerParameters(
    double getDecayRate,
    double getCeiling,
    double getFloor,
    double getSelfExcitation,
    double getLateralInhibition
) implements LayerParameters {}