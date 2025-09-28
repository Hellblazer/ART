package com.hellblazer.art.laminar.core;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.ILayerParameters;
import com.hellblazer.art.laminar.events.ILayerActivationListener;

/**
 * Interface for laminar circuit layers with shunting dynamics.
 * Implements the Grossberg shunting equations for neural activation.
 *
 * @author Hal Hildebrand
 */
public interface ILayer {

    // === Layer Identity ===

    /**
     * Get unique identifier for this layer.
     *
     * @return Layer ID
     */
    String getId();

    /**
     * Get layer type for routing and processing decisions.
     *
     * @return Layer type
     */
    LayerType getType();

    /**
     * Get the number of neurons in this layer.
     *
     * @return Neuron count
     */
    int size();

    // === Activation Dynamics ===

    /**
     * Update layer activation based on inputs.
     * Implements shunting equations: dx/dt = -Ax + (B-x)E - (x+C)I
     *
     * @param excitation Excitatory input (E)
     * @param inhibition Inhibitory input (I)
     * @param parameters Layer parameters (A, B, C, etc.)
     * @param dt Time step for integration
     */
    void updateActivation(Pattern excitation, Pattern inhibition,
                          ILayerParameters parameters, double dt);

    /**
     * Get current activation pattern.
     *
     * @return Current layer activation
     */
    Pattern getActivation();

    /**
     * Set activation directly (for initialization).
     *
     * @param activation New activation pattern
     */
    void setActivation(Pattern activation);

    /**
     * Reset layer to resting state.
     */
    void reset();

    // === Signal Processing ===

    /**
     * Process bottom-up input signal.
     *
     * @param input Input pattern from lower layer
     * @param parameters Processing parameters
     * @return Transformed signal
     */
    Pattern processBottomUp(Pattern input, ILayerParameters parameters);

    /**
     * Process top-down feedback signal.
     *
     * @param feedback Feedback from higher layer
     * @param parameters Processing parameters
     * @return Transformed signal
     */
    Pattern processTopDown(Pattern feedback, ILayerParameters parameters);

    /**
     * Process horizontal (lateral) connections.
     *
     * @param lateral Input from same-level layers
     * @param parameters Processing parameters
     * @return Transformed signal
     */
    Pattern processLateral(Pattern lateral, ILayerParameters parameters);

    // === Learning ===

    /**
     * Update weights based on current activation and learning signal.
     *
     * @param learningSignal Signal indicating what to learn
     * @param learningRate Learning rate parameter
     */
    void updateWeights(Pattern learningSignal, double learningRate);

    /**
     * Get current weight matrix.
     *
     * @return Weight matrix
     */
    WeightMatrix getWeights();

    /**
     * Check if layer weights are plastic (can learn).
     *
     * @return true if weights can be modified
     */
    boolean isPlastic();

    /**
     * Enable or disable learning for this layer.
     *
     * @param plastic true to enable learning
     */
    void setPlastic(boolean plastic);

    // === Monitoring ===

    /**
     * Get layer statistics for monitoring.
     *
     * @return Layer statistics
     */
    LayerStatistics getStatistics();

    /**
     * Register activation listener.
     *
     * @param listener Activation event listener
     */
    void addActivationListener(ILayerActivationListener listener);
}