package com.hellblazer.art.laminar.parameters;

/**
 * Shunting equation parameters.
 *
 * @author Hal Hildebrand
 */
public interface IShuntingParameters {
    double getDecayRate();
    double getUpperBound();
    double getLowerBound();
    double getTimeConstant();
}