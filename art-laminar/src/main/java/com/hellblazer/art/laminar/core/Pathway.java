package com.hellblazer.art.laminar.core;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.PathwayParameters;

/**
 * Interface for pathways connecting layers in a laminar circuit.
 *
 * @author Hal Hildebrand
 */
public interface Pathway {

    String getId();
    String getSourceLayerId();
    String getTargetLayerId();
    PathwayType getType();

    // Signal propagation
    Pattern propagate(Pattern signal, PathwayParameters parameters);

    // Gain control
    double getGain();
    void setGain(double gain);

    // Learning
    boolean isAdaptive();
    void updateWeights(Pattern sourceActivation, Pattern targetActivation, double learningRate);

    // Control
    void reset();
}