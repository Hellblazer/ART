package com.hellblazer.art.laminar.impl;

import com.hellblazer.art.laminar.parameters.IShuntingParameters;

/**
 * Default shunting equation parameters.
 *
 * @author Hal Hildebrand
 */
public record DefaultShuntingParameters(
        double decayRate,
        double upperBound,
        double lowerBound,
        double timeConstant
) implements IShuntingParameters {

    public static final DefaultShuntingParameters DEFAULT = new DefaultShuntingParameters(
            0.1, 1.0, 0.0, 1.0
    );

    @Override
    public double getDecayRate() {
        return decayRate;
    }

    @Override
    public double getUpperBound() {
        return upperBound;
    }

    @Override
    public double getLowerBound() {
        return lowerBound;
    }

    @Override
    public double getTimeConstant() {
        return timeConstant;
    }
}