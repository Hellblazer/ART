package com.hellblazer.art.cortical.learning;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.cortical.resonance.ResonanceState;

/**
 * Context information for synaptic learning in cortical layers.
 *
 * <p>Encapsulates all information needed to determine whether learning
 * should occur and how learning rates should be modulated. Integrates
 * multiple factors:
 * <ul>
 *   <li><b>Activity</b>: Pre- and post-synaptic activation patterns</li>
 *   <li><b>Resonance</b>: Consciousness likelihood from Phase 2D</li>
 *   <li><b>Attention</b>: Attentional strength from Layer 1</li>
 *   <li><b>Timing</b>: Timestamp for temporal dynamics</li>
 * </ul>
 *
 * <h2>Learning Gating</h2>
 * <p>The cortical learning framework uses multiple gating signals:
 * <ul>
 *   <li><b>Resonance Gating</b>: Only learn during conscious perception</li>
 *   <li><b>Attention Gating</b>: Focus learning on attended patterns</li>
 *   <li><b>Combined Gating</b>: Both resonance AND attention required</li>
 * </ul>
 *
 * <h2>Learning Rate Modulation</h2>
 * <p>Effective learning rate is modulated by:
 * <pre>
 * α_eff = α_base × m_resonance × m_attention
 * </pre>
 * where:
 * <ul>
 *   <li>α_base: Base learning rate (layer-specific)</li>
 *   <li>m_resonance: Consciousness likelihood [0, 1]</li>
 *   <li>m_attention: Attention strength [0, 1]</li>
 * </ul>
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * // Create learning context from circuit processing
 * var context = new LearningContext(
 *     inputPattern,          // Pre-synaptic activity
 *     layerActivation,       // Post-synaptic activity
 *     resonanceState,        // From Phase 2D (may be null)
 *     attentionStrength,     // From Layer 1 (0-1)
 *     currentTimestamp       // Current time
 * );
 *
 * // Check if learning should occur
 * if (context.shouldLearn(0.7, 0.3)) {
 *     // Get effective learning rate
 *     double baseRate = 0.01;
 *     double effectiveRate = baseRate * context.getLearningRateModulation();
 *
 *     // Apply learning
 *     layer.learn(context, effectiveRate);
 * }
 * }</pre>
 *
 * @param preActivation Pre-synaptic activation pattern (input)
 * @param postActivation Post-synaptic activation pattern (output)
 * @param resonanceState Resonance state from Phase 2D (null if detection disabled)
 * @param attentionStrength Attention strength from Layer 1 [0, 1]
 * @param timestamp Current time in seconds
 *
 * @author Phase 3A: Core Learning Infrastructure
 * @see LearningRule
 * @see ResonanceState
 */
public record LearningContext(
    Pattern preActivation,
    Pattern postActivation,
    ResonanceState resonanceState,
    double attentionStrength,
    double timestamp
) {
    /**
     * Compact constructor with validation.
     */
    public LearningContext {
        if (preActivation == null) {
            throw new IllegalArgumentException("preActivation cannot be null");
        }
        if (postActivation == null) {
            throw new IllegalArgumentException("postActivation cannot be null");
        }
        if (attentionStrength < 0.0 || attentionStrength > 1.0) {
            throw new IllegalArgumentException(
                "attentionStrength must be in [0, 1]: " + attentionStrength
            );
        }
        if (timestamp < 0.0) {
            throw new IllegalArgumentException(
                "timestamp must be non-negative: " + timestamp
            );
        }
    }

    /**
     * Check if learning should occur based on resonance and attention thresholds.
     *
     * <p>Learning occurs when BOTH conditions are met:
     * <ul>
     *   <li>Consciousness likelihood ≥ resonance threshold</li>
     *   <li>Attention strength ≥ attention threshold</li>
     * </ul>
     *
     * <p>If resonance detection is disabled (resonanceState == null),
     * only attention gating is applied.
     *
     * @param resonanceThreshold Minimum consciousness likelihood (typically 0.7)
     * @param attentionThreshold Minimum attention strength (typically 0.3)
     * @return true if learning should occur
     */
    public boolean shouldLearn(double resonanceThreshold, double attentionThreshold) {
        // Check attention threshold
        if (attentionStrength < attentionThreshold) {
            return false;
        }

        // Check resonance threshold (if resonance detection enabled)
        if (resonanceState != null) {
            return resonanceState.consciousnessLikelihood() >= resonanceThreshold;
        }

        // If resonance detection disabled, only check attention
        return true;
    }

    /**
     * Get learning rate modulation factor [0, 1].
     *
     * <p>Combines resonance and attention to modulate the base learning rate:
     * <pre>
     * modulation = consciousness × attention
     * </pre>
     *
     * <p>If resonance detection is disabled, only attention modulation is applied.
     *
     * @return multiplicative modulation factor [0, 1]
     */
    public double getLearningRateModulation() {
        double resonanceMod = resonanceState != null
            ? resonanceState.consciousnessLikelihood()
            : 1.0;  // No modulation if resonance detection disabled

        return resonanceMod * attentionStrength;
    }

    /**
     * Check if resonance detection is enabled.
     *
     * @return true if resonance state is available
     */
    public boolean hasResonanceDetection() {
        return resonanceState != null;
    }

    /**
     * Check if resonance was detected (requires resonance detection enabled).
     *
     * @return true if ART resonance occurred
     */
    public boolean hasResonance() {
        return resonanceState != null && resonanceState.artResonance();
    }

    /**
     * Check if likely conscious perception (requires resonance detection enabled).
     *
     * @param threshold consciousness likelihood threshold (typically 0.7)
     * @return true if likely conscious
     */
    public boolean isLikelyConscious(double threshold) {
        return resonanceState != null && resonanceState.isLikelyConscious(threshold);
    }

    /**
     * Get consciousness likelihood [0, 1] (requires resonance detection enabled).
     *
     * @return consciousness likelihood or 0.0 if detection disabled
     */
    public double getConsciousnessLikelihood() {
        return resonanceState != null ? resonanceState.consciousnessLikelihood() : 0.0;
    }

    /**
     * Create learning context without resonance detection.
     *
     * <p>Useful when resonance detection is disabled but learning is still desired.
     *
     * @param preActivation Pre-synaptic activation
     * @param postActivation Post-synaptic activation
     * @param attentionStrength Attention strength [0, 1]
     * @param timestamp Current time
     * @return learning context with no resonance state
     */
    public static LearningContext withoutResonance(
            Pattern preActivation,
            Pattern postActivation,
            double attentionStrength,
            double timestamp) {
        return new LearningContext(
            preActivation,
            postActivation,
            null,  // No resonance detection
            attentionStrength,
            timestamp
        );
    }

    /**
     * Create learning context with maximum attention and no resonance gating.
     *
     * <p>Useful for testing or when learning should always occur.
     *
     * @param preActivation Pre-synaptic activation
     * @param postActivation Post-synaptic activation
     * @param timestamp Current time
     * @return learning context with full attention, no resonance
     */
    public static LearningContext alwaysLearn(
            Pattern preActivation,
            Pattern postActivation,
            double timestamp) {
        return new LearningContext(
            preActivation,
            postActivation,
            null,  // No resonance detection
            1.0,   // Full attention
            timestamp
        );
    }

    @Override
    public String toString() {
        return "LearningContext[" +
               "resonance=" + (hasResonance() ? "YES" : "NO") +
               ", consciousness=" + String.format("%.3f", getConsciousnessLikelihood()) +
               ", attention=" + String.format("%.3f", attentionStrength) +
               ", modulation=" + String.format("%.3f", getLearningRateModulation()) +
               ", t=" + String.format("%.3f", timestamp) +
               "]";
    }
}
