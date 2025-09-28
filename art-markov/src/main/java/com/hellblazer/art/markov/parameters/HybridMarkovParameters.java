package com.hellblazer.art.markov.parameters;

import com.hellblazer.art.core.parameters.FuzzyParameters;

/**
 * Immutable parameters for the hybrid ART-Markov system.
 *
 * This combines FuzzyART parameters for state abstraction with
 * Markov chain parameters for transition learning.
 *
 * @param fuzzyParameters Parameters for the underlying FuzzyART state abstraction
 * @param transitionSmoothingFactor Smoothing factor for transition probabilities (0, 1]
 * @param hybridWeight Weight for combining ART and Markov predictions [0, 1]
 * @param convergenceThreshold Threshold for detecting Markov chain convergence
 * @param maxStates Maximum number of states the system can discover
 * @param memoryWindow Window size for computing transition statistics
 */
public record HybridMarkovParameters(
    FuzzyParameters fuzzyParameters,
    double transitionSmoothingFactor,
    double hybridWeight,
    double convergenceThreshold,
    int maxStates,
    int memoryWindow
) {

    /**
     * Constructor with validation.
     */
    public HybridMarkovParameters {
        if (fuzzyParameters == null) {
            throw new IllegalArgumentException("FuzzyParameters cannot be null");
        }
        if (transitionSmoothingFactor <= 0.0 || transitionSmoothingFactor > 1.0) {
            throw new IllegalArgumentException(
                "Transition smoothing factor must be in range (0, 1], got: " + transitionSmoothingFactor
            );
        }
        if (hybridWeight < 0.0 || hybridWeight > 1.0) {
            throw new IllegalArgumentException(
                "Hybrid weight must be in range [0, 1], got: " + hybridWeight
            );
        }
        if (convergenceThreshold <= 0.0 || convergenceThreshold >= 1.0) {
            throw new IllegalArgumentException(
                "Convergence threshold must be in range (0, 1), got: " + convergenceThreshold
            );
        }
        if (maxStates < 2) {
            throw new IllegalArgumentException(
                "Max states must be at least 2, got: " + maxStates
            );
        }
        if (memoryWindow < 2) {
            throw new IllegalArgumentException(
                "Memory window must be at least 2, got: " + memoryWindow
            );
        }
    }

    /**
     * Create default parameters suitable for a simple weather model.
     */
    public static HybridMarkovParameters weatherDefaults() {
        return new HybridMarkovParameters(
            new FuzzyParameters(0.75, 0.01, 0.6), // Moderate vigilance, standard α, β
            0.1,    // Small smoothing for stable transitions
            0.5,    // Equal weight to ART and Markov
            0.001,  // Tight convergence threshold
            4,      // 4 weather states max
            10      // 10-sample memory window
        );
    }

    /**
     * Create parameters for experimentation with different hybrid weights.
     */
    public static HybridMarkovParameters withHybridWeight(double weight) {
        var defaults = weatherDefaults();
        return new HybridMarkovParameters(
            defaults.fuzzyParameters(),
            defaults.transitionSmoothingFactor(),
            weight,
            defaults.convergenceThreshold(),
            defaults.maxStates(),
            defaults.memoryWindow()
        );
    }

    /**
     * Create parameters with different vigilance for state granularity control.
     */
    public static HybridMarkovParameters withVigilance(double vigilance) {
        var defaults = weatherDefaults();
        var newFuzzy = new FuzzyParameters(
            vigilance,
            defaults.fuzzyParameters().alpha(),
            defaults.fuzzyParameters().beta()
        );
        return new HybridMarkovParameters(
            newFuzzy,
            defaults.transitionSmoothingFactor(),
            defaults.hybridWeight(),
            defaults.convergenceThreshold(),
            defaults.maxStates(),
            defaults.memoryWindow()
        );
    }
}