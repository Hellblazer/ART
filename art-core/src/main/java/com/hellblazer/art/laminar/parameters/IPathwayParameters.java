package com.hellblazer.art.laminar.parameters;

/**
 * Parameters for pathway connections.
 *
 * @author Hal Hildebrand
 */
public interface IPathwayParameters {

    // Connection strength
    double getConnectionStrength();
    double getInitialGain();

    // Propagation parameters
    int getPropagationDelay();
    double getSignalAttenuation();

    // Learning parameters
    double getLearningRate();
    boolean isHebbian();
    boolean isCompetitive();

    // Modulation parameters
    double getMaxGain();
    double getMinGain();
    double getGainDecay();
}