package com.hellblazer.art.temporal.memory;

import java.util.List;

/**
 * Temporal pattern representation from working memory.
 * Contains the sequence of patterns with their weights and primacy gradient.
 */
public record TemporalPattern(
    List<double[]> patterns,
    List<Double> weights,
    double primacyGradient
) {
    /**
     * Get the sequence length.
     */
    public int sequenceLength() {
        return patterns.size();
    }

    /**
     * Check if the pattern is valid.
     */
    public boolean isValid() {
        return patterns != null && weights != null &&
               patterns.size() == weights.size() &&
               !patterns.isEmpty();
    }

    /**
     * Get the combined pattern by weighted average.
     */
    public double[] getCombinedPattern() {
        if (!isValid() || patterns.isEmpty()) {
            return new double[0];
        }

        int dimension = patterns.get(0).length;
        double[] combined = new double[dimension];
        double totalWeight = 0.0;

        for (int i = 0; i < patterns.size(); i++) {
            var pattern = patterns.get(i);
            var weight = weights.get(i);

            for (int j = 0; j < dimension; j++) {
                combined[j] += pattern[j] * weight;
            }
            totalWeight += weight;
        }

        // Normalize by total weight
        if (totalWeight > 0) {
            for (int j = 0; j < dimension; j++) {
                combined[j] /= totalWeight;
            }
        }

        return combined;
    }

    /**
     * Get the strongest pattern (highest weight).
     */
    public double[] getStrongestPattern() {
        if (!isValid() || patterns.isEmpty()) {
            return new double[0];
        }

        int maxIndex = 0;
        double maxWeight = weights.get(0);

        for (int i = 1; i < weights.size(); i++) {
            if (weights.get(i) > maxWeight) {
                maxWeight = weights.get(i);
                maxIndex = i;
            }
        }

        return patterns.get(maxIndex).clone();
    }

    /**
     * Get the average weight.
     */
    public double getAverageWeight() {
        if (!isValid() || weights.isEmpty()) {
            return 0.0;
        }

        double sum = 0.0;
        for (double weight : weights) {
            sum += weight;
        }
        return sum / weights.size();
    }
}