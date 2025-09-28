package com.hellblazer.art.laminar.core;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.IPathwayParameters;

/**
 * Interface for connections between layers in laminar circuits.
 * Supports bottom-up, top-down, and horizontal pathways.
 *
 * @author Hal Hildebrand
 */
public interface IPathway {

    // === Pathway Identity ===

    /**
     * Get unique identifier for this pathway.
     *
     * @return Pathway ID
     */
    String getId();

    /**
     * Get pathway type.
     *
     * @return Type of connection
     */
    PathwayType getType();

    /**
     * Get source layer ID.
     *
     * @return Source layer identifier
     */
    String getSourceLayerId();

    /**
     * Get target layer ID.
     *
     * @return Target layer identifier
     */
    String getTargetLayerId();

    // === Signal Propagation ===

    /**
     * Propagate signal through the pathway.
     *
     * @param input Signal from source layer
     * @param parameters Pathway parameters
     * @return Transformed signal for target layer
     */
    Pattern propagate(Pattern input, IPathwayParameters parameters);

    /**
     * Get propagation delay in time steps.
     *
     * @return Delay in time steps
     */
    int getDelay();

    /**
     * Set propagation delay.
     *
     * @param delay Delay in time steps
     */
    void setDelay(int delay);

    // === Connection Weights ===

    /**
     * Get connection weight matrix.
     *
     * @return Weight matrix
     */
    WeightMatrix getWeights();

    /**
     * Update connection weights.
     *
     * @param source Source layer activation
     * @param target Target layer activation
     * @param learningRate Learning rate
     */
    void updateWeights(Pattern source, Pattern target, double learningRate);

    /**
     * Check if pathway weights are adaptive.
     *
     * @return true if weights can change
     */
    boolean isAdaptive();

    /**
     * Enable or disable weight adaptation.
     *
     * @param adaptive true to enable learning
     */
    void setAdaptive(boolean adaptive);

    // === Modulation ===

    /**
     * Apply gain modulation to the pathway.
     *
     * @param gain Multiplicative gain factor
     */
    void applyGain(double gain);

    /**
     * Get current gain value.
     *
     * @return Current gain
     */
    double getGain();

    /**
     * Enable or disable this pathway.
     *
     * @param enabled true to enable signal flow
     */
    void setEnabled(boolean enabled);

    /**
     * Check if pathway is enabled.
     *
     * @return true if signal can flow
     */
    boolean isEnabled();
}