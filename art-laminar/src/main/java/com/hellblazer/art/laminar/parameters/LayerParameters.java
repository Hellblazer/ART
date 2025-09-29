package com.hellblazer.art.laminar.parameters;

/**
 * Parameters for individual layers.
 *
 * @author Hal Hildebrand
 */
public interface LayerParameters {
    double getDecayRate();
    double getCeiling();
    double getFloor();
    double getSelfExcitation();
    double getLateralInhibition();
}