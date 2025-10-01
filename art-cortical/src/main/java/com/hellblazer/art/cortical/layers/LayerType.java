package com.hellblazer.art.cortical.layers;

/**
 * Types of cortical layers in the 6-layer laminar architecture.
 * Based on mammalian neocortical organization with specialized functional roles.
 *
 * <p>Laminar Organization:
 * <ul>
 *   <li>L1: Apical dendrites, feedback integration, surface contours</li>
 *   <li>L2/3: Inter-areal communication, working memory, category learning</li>
 *   <li>L4: Thalamic input, driving signals, sensory processing</li>
 *   <li>L5: Motor output, deep pyramidal cells, action selection</li>
 *   <li>L6: Corticothalamic feedback, attention, expectation matching</li>
 * </ul>
 *
 * @author Migrated from art-laminar to art-cortical (Phase 3, Milestone 1)
 */
public enum LayerType {
    /**
     * Layer 1: Apical dendrites and feedback integration.
     * - Receives top-down feedback
     * - Processes surface contours via BipoleCell networks
     * - Lateral boundary completion
     * - Time constant: 50-150ms (slow integration)
     */
    LAYER_1("L1 - Apical Dendrites & Feedback"),

    /**
     * Layer 2/3: Inter-areal communication and prediction.
     * - Working memory integration
     * - Category formation via ART dynamics
     * - Predictive coding
     * - Time constant: 50-100ms (medium)
     */
    LAYER_2_3("L2/3 - Inter-areal & Prediction"),

    /**
     * Layer 4: Thalamic driving input.
     * - Primary recipient of thalamic input (LGN, etc.)
     * - Fast feedforward processing
     * - Strong driving signals
     * - Time constant: 10-50ms (fast)
     */
    LAYER_4("L4 - Thalamic Input"),

    /**
     * Layer 5: Motor output and action selection.
     * - Deep pyramidal cells
     * - Decision formation
     * - Subcortical projections
     * - Time constant: 30-100ms (medium-fast)
     */
    LAYER_5("L5 - Motor Output"),

    /**
     * Layer 6: Corticothalamic feedback and attention.
     * - Top-down expectation matching
     * - Attentional modulation
     * - Thalamic gain control
     * - Time constant: 50-150ms (medium-slow)
     */
    LAYER_6("L6 - Corticothalamic Feedback");

    private final String description;

    LayerType(String description) {
        this.description = description;
    }

    /**
     * Get human-readable description of this layer type.
     *
     * @return layer description
     */
    public String getDescription() {
        return description;
    }

    /**
     * Get typical time constant range for this layer type.
     * Returns the midpoint of the biological range.
     *
     * @return time constant in milliseconds
     */
    public double getTypicalTimeConstant() {
        return switch (this) {
            case LAYER_1 -> 100.0;  // 50-150ms range
            case LAYER_2_3 -> 75.0; // 50-100ms range
            case LAYER_4 -> 30.0;   // 10-50ms range (fast)
            case LAYER_5 -> 65.0;   // 30-100ms range
            case LAYER_6 -> 100.0;  // 50-150ms range
        };
    }

    /**
     * Check if this layer primarily receives driving input.
     * Driving input can fire neurons independently.
     *
     * @return true if this is a driving input layer
     */
    public boolean isDrivingLayer() {
        return this == LAYER_4;
    }

    /**
     * Check if this layer primarily processes modulatory input.
     * Modulatory input requires convergence with other signals.
     *
     * @return true if this is a modulatory layer
     */
    public boolean isModulatoryLayer() {
        return this == LAYER_1 || this == LAYER_6;
    }
}
