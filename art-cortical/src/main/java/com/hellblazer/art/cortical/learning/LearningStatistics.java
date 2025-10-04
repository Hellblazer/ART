package com.hellblazer.art.cortical.learning;

import com.hellblazer.art.cortical.resonance.ResonanceState;

/**
 * Tracks learning statistics for cortical layers and circuits.
 *
 * <p>Maintains counters and averages for various learning events:
 * <ul>
 *   <li><b>Learning Events</b>: Total number of weight updates</li>
 *   <li><b>Resonance Gating</b>: Learning during conscious resonance</li>
 *   <li><b>Attention Gating</b>: Learning during high attention</li>
 *   <li><b>Weight Changes</b>: Magnitude of weight updates</li>
 * </ul>
 *
 * <h2>Key Metrics</h2>
 * <ul>
 *   <li><b>Learning Efficiency</b>: Fraction of potential updates that actually occurred</li>
 *   <li><b>Avg Consciousness</b>: Mean consciousness likelihood during learning</li>
 *   <li><b>Avg Attention</b>: Mean attention strength during learning</li>
 *   <li><b>Avg Weight Change</b>: Mean magnitude of weight updates</li>
 * </ul>
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * var stats = new LearningStatistics();
 *
 * // Record learning events during training
 * for (var pattern : trainingSet) {
 *     var result = circuit.processAndLearn(pattern);
 *     if (result.hasResonance()) {
 *         stats.recordLearningEvent(
 *             result.resonanceState(),
 *             attentionStrength,
 *             weightChangeMagnitude
 *         );
 *     }
 * }
 *
 * // Check learning efficiency
 * System.out.println("Learning efficiency: " + stats.getLearningEfficiency());
 * System.out.println("Avg consciousness: " + stats.getAvgConsciousness());
 * System.out.println("Avg weight change: " + stats.getAvgWeightChange());
 * }</pre>
 *
 * @author Phase 3A: Core Learning Infrastructure
 */
public class LearningStatistics {

    private long totalLearningEvents;
    private long resonanceGatedEvents;
    private long attentionGatedEvents;
    private long bothGatedEvents;

    private double sumConsciousness;
    private double sumAttention;
    private double sumWeightChange;

    private double minWeightChange;
    private double maxWeightChange;

    /**
     * Create empty learning statistics.
     */
    public LearningStatistics() {
        reset();
    }

    /**
     * Record a learning event with full context.
     *
     * @param resonanceState Resonance state (may be null)
     * @param attentionStrength Attention strength [0, 1]
     * @param weightChangeMagnitude Magnitude of weight change
     */
    public synchronized void recordLearningEvent(
            ResonanceState resonanceState,
            double attentionStrength,
            double weightChangeMagnitude) {

        totalLearningEvents++;

        // Track gating events
        boolean hasResonance = resonanceState != null && resonanceState.artResonance();
        boolean hasAttention = attentionStrength > 0.3;  // Typical attention threshold

        if (hasResonance) {
            resonanceGatedEvents++;
            sumConsciousness += resonanceState.consciousnessLikelihood();
        }

        if (hasAttention) {
            attentionGatedEvents++;
            sumAttention += attentionStrength;
        }

        if (hasResonance && hasAttention) {
            bothGatedEvents++;
        }

        // Track weight changes
        sumWeightChange += weightChangeMagnitude;
        if (totalLearningEvents == 1) {
            minWeightChange = weightChangeMagnitude;
            maxWeightChange = weightChangeMagnitude;
        } else {
            minWeightChange = Math.min(minWeightChange, weightChangeMagnitude);
            maxWeightChange = Math.max(maxWeightChange, weightChangeMagnitude);
        }
    }

    /**
     * Record a simple learning event (no resonance/attention details).
     *
     * @param weightChangeMagnitude Magnitude of weight change
     */
    public synchronized void recordLearningEvent(double weightChangeMagnitude) {
        recordLearningEvent(null, 0.0, weightChangeMagnitude);
    }

    /**
     * Get total number of learning events.
     */
    public synchronized long getTotalLearningEvents() {
        return totalLearningEvents;
    }

    /**
     * Get number of resonance-gated learning events.
     */
    public synchronized long getResonanceGatedEvents() {
        return resonanceGatedEvents;
    }

    /**
     * Get number of attention-gated learning events.
     */
    public synchronized long getAttentionGatedEvents() {
        return attentionGatedEvents;
    }

    /**
     * Get number of events with both resonance and attention gating.
     */
    public synchronized long getBothGatedEvents() {
        return bothGatedEvents;
    }

    /**
     * Get learning efficiency (fraction of events with resonance gating).
     *
     * @return efficiency [0, 1], or 0.0 if no events
     */
    public synchronized double getLearningEfficiency() {
        return totalLearningEvents > 0
            ? (double) resonanceGatedEvents / totalLearningEvents
            : 0.0;
    }

    /**
     * Get average consciousness likelihood during learning events.
     *
     * @return average [0, 1], or 0.0 if no resonance-gated events
     */
    public synchronized double getAvgConsciousness() {
        return resonanceGatedEvents > 0
            ? sumConsciousness / resonanceGatedEvents
            : 0.0;
    }

    /**
     * Get average attention strength during learning events.
     *
     * @return average [0, 1], or 0.0 if no attention-gated events
     */
    public synchronized double getAvgAttention() {
        return attentionGatedEvents > 0
            ? sumAttention / attentionGatedEvents
            : 0.0;
    }

    /**
     * Get average weight change magnitude.
     *
     * @return average weight change, or 0.0 if no events
     */
    public synchronized double getAvgWeightChange() {
        return totalLearningEvents > 0
            ? sumWeightChange / totalLearningEvents
            : 0.0;
    }

    /**
     * Get minimum weight change magnitude.
     *
     * @return min weight change, or 0.0 if no events
     */
    public synchronized double getMinWeightChange() {
        return totalLearningEvents > 0 ? minWeightChange : 0.0;
    }

    /**
     * Get maximum weight change magnitude.
     *
     * @return max weight change, or 0.0 if no events
     */
    public synchronized double getMaxWeightChange() {
        return totalLearningEvents > 0 ? maxWeightChange : 0.0;
    }

    /**
     * Reset all statistics to zero.
     */
    public synchronized void reset() {
        totalLearningEvents = 0;
        resonanceGatedEvents = 0;
        attentionGatedEvents = 0;
        bothGatedEvents = 0;
        sumConsciousness = 0.0;
        sumAttention = 0.0;
        sumWeightChange = 0.0;
        minWeightChange = 0.0;
        maxWeightChange = 0.0;
    }

    /**
     * Check if any learning has occurred.
     */
    public synchronized boolean hasLearningOccurred() {
        return totalLearningEvents > 0;
    }

    @Override
    public synchronized String toString() {
        return "LearningStatistics[" +
               "total=" + totalLearningEvents +
               ", resonanceGated=" + resonanceGatedEvents +
               ", attentionGated=" + attentionGatedEvents +
               ", bothGated=" + bothGatedEvents +
               ", efficiency=" + String.format("%.3f", getLearningEfficiency()) +
               ", avgConsciousness=" + String.format("%.3f", getAvgConsciousness()) +
               ", avgAttention=" + String.format("%.3f", getAvgAttention()) +
               ", avgWeightChange=" + String.format("%.6f", getAvgWeightChange()) +
               "]";
    }
}
