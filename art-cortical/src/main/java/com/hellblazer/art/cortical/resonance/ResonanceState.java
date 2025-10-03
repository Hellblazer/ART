package com.hellblazer.art.cortical.resonance;

import com.hellblazer.art.cortical.analysis.OscillationMetrics;

/**
 * Enhanced resonance state with oscillatory dynamics and consciousness metrics.
 *
 * <p>Captures both traditional ART resonance (match quality) and oscillatory
 * characteristics (phase synchronization, gamma oscillations) for consciousness
 * research based on Grossberg (2017) CLEARS framework.
 *
 * <h2>Consciousness Likelihood Interpretation</h2>
 * <p>The consciousness likelihood is a heuristic measure based on:
 * <ul>
 *   <li><b>ART Resonance</b> (base): Feature-expectation match quality</li>
 *   <li><b>Phase Synchronization</b> (+0.2): Bottom-up and top-down phase alignment</li>
 *   <li><b>Gamma Oscillations</b> (+0.3): Both pathways in gamma band (30-50 Hz)</li>
 * </ul>
 *
 * <p>High consciousness likelihood (&gt; 0.7) suggests:
 * <ul>
 *   <li>Strong feature-expectation match (ART resonance)</li>
 *   <li>Phase-locked oscillations between bottom-up and top-down</li>
 *   <li>Both pathways oscillating in gamma band</li>
 * </ul>
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * var detector = new EnhancedResonanceDetector(0.7); // vigilance threshold
 * var state = detector.detectResonance(bottomUpFeatures, topDownExpectations, timestamp);
 *
 * if (state.consciousnessLikelihood() > 0.7) {
 *     System.out.println("High consciousness likelihood:");
 *     System.out.println("  ART Resonance: " + state.artResonance());
 *     System.out.println("  Phase Sync: " + state.phaseSynchronized());
 *     System.out.println("  Gamma: " + state.bothInGamma());
 *     System.out.println("  Likelihood: " + state.consciousnessLikelihood());
 * }
 * }</pre>
 *
 * @param artResonance Traditional ART resonance (vigilance threshold met)
 * @param phaseSynchronized Bottom-up and top-down phases synchronized
 * @param bothInGamma Both pathways oscillating in gamma band (30-50 Hz)
 * @param consciousnessLikelihood Heuristic consciousness likelihood [0, 1]
 * @param bottomUpMetrics Oscillation metrics from bottom-up pathway
 * @param topDownMetrics Oscillation metrics from top-down pathway
 * @param matchQuality Feature-expectation match quality [0, 1]
 * @param timestamp Time when resonance was detected (seconds)
 *
 * @author Phase 2D: Enhanced Resonance Detection
 */
public record ResonanceState(
    boolean artResonance,
    boolean phaseSynchronized,
    boolean bothInGamma,
    double consciousnessLikelihood,
    OscillationMetrics bottomUpMetrics,
    OscillationMetrics topDownMetrics,
    double matchQuality,
    double timestamp
) {
    /**
     * Phase synchronization threshold (radians).
     * Within 45 degrees (π/4) is considered synchronized.
     */
    public static final double PHASE_SYNC_THRESHOLD = Math.PI / 4;

    /**
     * Gamma band frequency range (Hz).
     */
    public static final double GAMMA_LOW = 30.0;
    public static final double GAMMA_HIGH = 50.0;

    /**
     * Compact constructor with validation.
     */
    public ResonanceState {
        if (consciousnessLikelihood < 0.0 || consciousnessLikelihood > 1.0) {
            throw new IllegalArgumentException(
                "consciousnessLikelihood must be in [0, 1]: " + consciousnessLikelihood
            );
        }
        if (matchQuality < 0.0 || matchQuality > 1.0) {
            throw new IllegalArgumentException(
                "matchQuality must be in [0, 1]: " + matchQuality
            );
        }
    }

    /**
     * Check if this resonance state indicates likely conscious perception.
     *
     * <p>Requires:
     * <ul>
     *   <li>ART resonance (feature-expectation match)</li>
     *   <li>Phase synchronization between pathways</li>
     *   <li>Both pathways in gamma band</li>
     *   <li>Overall consciousness likelihood &gt; threshold</li>
     * </ul>
     *
     * @param threshold Minimum consciousness likelihood (typically 0.7)
     * @return true if likely conscious perception
     */
    public boolean isLikelyConscious(double threshold) {
        return artResonance
            && phaseSynchronized
            && bothInGamma
            && consciousnessLikelihood >= threshold;
    }

    /**
     * Check if this is a strong resonance state.
     *
     * <p>Strong resonance requires ART resonance and high match quality.
     *
     * @param threshold Minimum match quality (typically 0.8)
     * @return true if strong resonance
     */
    public boolean isStrongResonance(double threshold) {
        return artResonance && matchQuality >= threshold;
    }

    /**
     * Get phase difference between bottom-up and top-down pathways.
     *
     * @return Phase difference in radians [-π, π], or NaN if metrics unavailable
     */
    public double getPhaseDifference() {
        if (bottomUpMetrics == null || topDownMetrics == null) {
            return Double.NaN;
        }
        return bottomUpMetrics.phaseDifferenceWith(topDownMetrics);
    }

    /**
     * Get frequency coherence between pathways.
     *
     * <p>Measures how close the dominant frequencies are.
     *
     * @return Frequency difference in Hz, or NaN if metrics unavailable
     */
    public double getFrequencyCoherence() {
        if (bottomUpMetrics == null || topDownMetrics == null) {
            return Double.NaN;
        }
        return Math.abs(bottomUpMetrics.dominantFrequency() - topDownMetrics.dominantFrequency());
    }

    /**
     * Create resonance state indicating no resonance.
     *
     * @param timestamp Current time
     * @return Resonance state with no resonance detected
     */
    public static ResonanceState none(double timestamp) {
        return new ResonanceState(
            false,  // artResonance
            false,  // phaseSynchronized
            false,  // bothInGamma
            0.0,    // consciousnessLikelihood
            null,   // bottomUpMetrics
            null,   // topDownMetrics
            0.0,    // matchQuality
            timestamp
        );
    }

    @Override
    public String toString() {
        return "ResonanceState[" +
            "ART=" + artResonance +
            ", phaseSync=" + phaseSynchronized +
            ", gamma=" + bothInGamma +
            ", consciousness=" + String.format("%.3f", consciousnessLikelihood) +
            ", match=" + String.format("%.3f", matchQuality) +
            ", t=" + String.format("%.3f", timestamp) +
            "]";
    }
}
