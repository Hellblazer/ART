package com.hellblazer.art.cortical.layers;

import com.hellblazer.art.core.Pattern;

/**
 * Unified layer interface for cortical circuit computation.
 * Represents a single layer in the 6-layer cortical architecture with
 * multi-pathway processing (bottom-up, top-down, lateral).
 *
 * <p>Design Philosophy:
 * <ul>
 *   <li>Unified from art-temporal (LIST PARSE) and art-laminar (6-layer circuits)</li>
 *   <li>Supports both temporal and spatial processing</li>
 *   <li>Multi-pathway integration (feedforward, feedback, lateral)</li>
 *   <li>Biologically-inspired cortical computation</li>
 * </ul>
 *
 * <p>Layer Types:
 * <ul>
 *   <li>Layer 1: Apical dendrites, feedback integration</li>
 *   <li>Layer 2/3: Inter-areal communication, prediction</li>
 *   <li>Layer 4: Thalamic input, feedforward processing</li>
 *   <li>Layer 5: Motor output, deep pyramidal cells</li>
 *   <li>Layer 6: Corticothalamic feedback, gain modulation</li>
 * </ul>
 *
 * @author Migrated from art-temporal + art-laminar to art-cortical (Phase 1)
 */
public interface Layer extends AutoCloseable {

    /**
     * Get the unique identifier for this layer.
     *
     * @return layer ID (e.g., "L4", "L2/3", "L6")
     */
    String getId();

    /**
     * Get the number of units in this layer.
     *
     * @return layer size (number of neurons/columns)
     */
    int size();

    /**
     * Get the type of this layer in the cortical architecture.
     *
     * @return layer type (L1, L2/3, L4, L5, L6)
     */
    LayerType getType();

    /**
     * Get current activation state of this layer.
     *
     * @return current activation pattern (immutable)
     */
    Pattern getActivation();

    /**
     * Set activation state (for initialization or external input).
     *
     * @param activation new activation pattern
     * @throws IllegalArgumentException if activation dimension doesn't match layer size
     */
    void setActivation(Pattern activation);

    /**
     * Process bottom-up (feedforward) input from lower layers.
     * Implements thalamic input processing (Layer 4) or inter-layer feedforward.
     *
     * @param input bottom-up input pattern
     * @param parameters layer-specific processing parameters
     * @return processed activation after bottom-up integration
     * @throws IllegalArgumentException if input dimension is incompatible
     */
    Pattern processBottomUp(Pattern input, LayerParameters parameters);

    /**
     * Process top-down (feedback) input from higher layers.
     * Implements predictive feedback and attentional modulation.
     *
     * @param expectation top-down expectation/prediction pattern
     * @param parameters layer-specific processing parameters
     * @return processed activation after top-down integration
     * @throws IllegalArgumentException if expectation dimension is incompatible
     */
    Pattern processTopDown(Pattern expectation, LayerParameters parameters);

    /**
     * Process lateral (horizontal) connections within the layer.
     * Implements competitive dynamics and contextual modulation.
     *
     * @param lateral lateral input from neighboring columns
     * @param parameters layer-specific processing parameters
     * @return processed activation after lateral integration
     * @throws IllegalArgumentException if lateral dimension is incompatible
     */
    Pattern processLateral(Pattern lateral, LayerParameters parameters);

    /**
     * Update synaptic weights based on learning.
     * Implements Hebbian learning, BCM rule, or other plasticity mechanisms.
     *
     * @param input input pattern that drove this layer
     * @param learningRate learning rate parameter (0.0 to 1.0)
     * @throws IllegalArgumentException if learningRate is out of range
     */
    void updateWeights(Pattern input, double learningRate);

    /**
     * Reset layer to initial state.
     * Clears activations but preserves learned weights unless otherwise specified.
     */
    void reset();

    /**
     * Close and release resources.
     * Default implementation does nothing; override for resource cleanup.
     */
    @Override
    default void close() {
        // Default: no resources to clean up
    }
}
