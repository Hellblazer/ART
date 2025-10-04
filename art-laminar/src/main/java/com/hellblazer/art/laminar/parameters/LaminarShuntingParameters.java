package com.hellblazer.art.laminar.parameters;

/**
 * Parameters for shunting dynamics.
 *
 * @author Hal Hildebrand
 */
public interface LaminarShuntingParameters {
    double getDecayRate();
    double getCeiling();
    double getFloor();
    double getSelfExcitation();
}