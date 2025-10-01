package com.hellblazer.art.laminar.canonical;

/**
 * Parameters controlling ART matching and reset dynamics.
 *
 * Based on Adaptive Resonance Theory (Carpenter & Grossberg, 1987).
 * Controls the vigilance test and category search behavior.
 *
 * @param vigilance match threshold [0,1] - higher = more specific categories
 * @param resetThreshold error magnitude triggering reset
 * @param maxSearchIterations maximum category search attempts
 * @param enableResonance whether to use resonance dynamics
 *
 * @author Hal Hildebrand
 */
public record MatchingParameters(
    double vigilance,
    double resetThreshold,
    int maxSearchIterations,
    boolean enableResonance
) {
    /**
     * Default parameters based on ART theory.
     *
     * - vigilance: 0.7 (moderately specific categories)
     * - resetThreshold: 0.3 (moderate error tolerance)
     * - maxSearchIterations: 10 (sufficient for typical use)
     * - enableResonance: true (use full ART dynamics)
     */
    public MatchingParameters() {
        this(0.7, 0.3, 10, true);
    }

    /**
     * Validate parameter ranges.
     */
    public MatchingParameters {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("vigilance must be in [0,1]");
        }
        if (resetThreshold < 0.0) {
            throw new IllegalArgumentException("resetThreshold must be non-negative");
        }
        if (maxSearchIterations < 1) {
            throw new IllegalArgumentException("maxSearchIterations must be positive");
        }
    }
}
