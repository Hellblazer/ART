package com.hellblazer.art.cortical.learning;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.cortical.layers.WeightMatrix;

/**
 * Base interface for synaptic learning rules.
 *
 * <p>Learning rules define how synaptic weights are updated based on
 * pre-synaptic and post-synaptic activity patterns. This abstraction
 * supports various learning mechanisms including Hebbian, anti-Hebbian,
 * instar/outstar, and more complex plasticity rules.
 *
 * <h2>Biological Motivation</h2>
 * <p>Synaptic plasticity is the biological basis of learning and memory:
 * <ul>
 *   <li><b>Hebbian Plasticity</b>: "Cells that fire together, wire together" (Hebb, 1949)</li>
 *   <li><b>LTP/LTD</b>: Long-term potentiation and depression</li>
 *   <li><b>STDP</b>: Spike-timing-dependent plasticity</li>
 *   <li><b>Neuromodulation</b>: Attention and arousal modulate learning</li>
 * </ul>
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * // Create Hebbian learning rule
 * var learningRule = new HebbianLearning(0.0001, 0.0, 1.0);
 *
 * // Update weights based on activity
 * var updatedWeights = learningRule.update(
 *     preActivation,   // Input pattern
 *     postActivation,  // Layer activation
 *     currentWeights,  // Current synaptic strengths
 *     0.01             // Learning rate
 * );
 *
 * // Apply updated weights to layer
 * layer.setWeights(updatedWeights);
 * }</pre>
 *
 * <h2>Implementation Guidelines</h2>
 * <ul>
 *   <li><b>Stability</b>: Ensure weight updates are bounded and stable</li>
 *   <li><b>Normalization</b>: Maintain weight bounds [minWeight, maxWeight]</li>
 *   <li><b>Decay</b>: Include weight decay to prevent runaway growth</li>
 *   <li><b>Efficiency</b>: Optimize for repeated updates during training</li>
 * </ul>
 *
 * @author Phase 3A: Core Learning Infrastructure
 * @see HebbianLearning
 * @see InstarOutstarLearning
 * @see ResonanceGatedLearning
 */
public interface LearningRule {

    /**
     * Update synaptic weights based on pre- and post-synaptic activity.
     *
     * <p>This method implements the core learning dynamics:
     * <pre>
     * Δw_ij = f(x_i, y_j, w_ij, α)
     * </pre>
     * where:
     * <ul>
     *   <li>x_i: pre-synaptic activation (input)</li>
     *   <li>y_j: post-synaptic activation (output)</li>
     *   <li>w_ij: current weight from neuron i to j</li>
     *   <li>α: learning rate</li>
     * </ul>
     *
     * <p><b>Thread Safety</b>: Implementations should be thread-safe if
     * used in concurrent contexts.
     *
     * <p><b>Side Effects</b>: This method should be pure (no side effects).
     * Weight updates are returned rather than applied directly.
     *
     * @param preActivation Pre-synaptic activation pattern (input)
     * @param postActivation Post-synaptic activation pattern (output)
     * @param currentWeights Current synaptic weight matrix
     * @param learningRate Learning rate [0, 1], typically 0.001-0.1
     * @return Updated weight matrix with same dimensions as currentWeights
     * @throws IllegalArgumentException if patterns or weights have incompatible dimensions
     * @throws IllegalArgumentException if learningRate is outside [0, 1]
     */
    WeightMatrix update(
        Pattern preActivation,
        Pattern postActivation,
        WeightMatrix currentWeights,
        double learningRate
    );

    /**
     * Get a descriptive name for this learning rule.
     *
     * <p>Used for logging and debugging.
     *
     * @return Human-readable name (e.g., "Hebbian", "InstarOutstar")
     */
    default String getName() {
        return getClass().getSimpleName();
    }

    /**
     * Check if this learning rule requires weight normalization.
     *
     * <p>Some learning rules (e.g., Oja's rule) include normalization
     * implicitly, while others (e.g., standard Hebbian) require explicit
     * normalization to prevent weight explosion.
     *
     * @return true if explicit normalization is needed
     */
    default boolean requiresNormalization() {
        return false;
    }

    /**
     * Get the recommended learning rate range for this rule.
     *
     * <p>Different learning rules have different stability regions.
     * This provides guidance for parameter selection.
     *
     * @return recommended [min, max] learning rate range
     */
    default double[] getRecommendedLearningRateRange() {
        return new double[]{0.001, 0.1};
    }
}
