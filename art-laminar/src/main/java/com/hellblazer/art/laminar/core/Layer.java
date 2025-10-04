package com.hellblazer.art.laminar.core;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.events.LayerActivationListener;
import com.hellblazer.art.laminar.parameters.LayerParameters;

/**
 * Interface for layers in a laminar cortical circuit.
 *
 * @author Hal Hildebrand
 */
public interface Layer {

    String getId();
    int size();
    LayerType getType();

    // Activation management
    Pattern getActivation();
    void setActivation(Pattern activation);

    // Processing
    Pattern processBottomUp(Pattern input, LayerParameters parameters);
    Pattern processTopDown(Pattern expectation, LayerParameters parameters);
    Pattern processLateral(Pattern lateral, LayerParameters parameters);

    // Weights
    WeightMatrix getWeights();
    void setWeights(WeightMatrix weights);
    void updateWeights(Pattern input, double learningRate);

    // Control
    void reset();
    void addActivationListener(LayerActivationListener listener);
}