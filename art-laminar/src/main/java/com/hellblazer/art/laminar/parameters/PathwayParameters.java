package com.hellblazer.art.laminar.parameters;

/**
 * Parameters for pathways between layers.
 *
 * @author Hal Hildebrand
 */
public interface PathwayParameters {
    double getGain();
    double getLearningRate();
    boolean isAdaptive();
}