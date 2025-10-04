package com.hellblazer.art.cortical.temporal;

import java.util.List;

/**
 * Temporal pattern representation from working memory.
 * Contains sequence of patterns with their weights and primacy gradient strength.
 *
 * <p>Represents temporal sequences extracted from STORE 2 working memory,
 * where each pattern has an associated activation weight reflecting its
 * retrievability based on primacy gradient and transmitter gating.
 *
 * <p>Based on Kazerounian & Grossberg (2014) STORE 2 model.
 *
 * @param patterns List of pattern vectors in temporal order
 * @param weights Activation weights for each pattern (primacy-weighted)
 * @param primacyGradient Strength of primacy gradient (positive = primacy, negative = recency)
 *
 * @author Migrated from art-temporal/temporal-memory to art-cortical (Phase 2)
 */
public record TemporalPattern(
    List<double[]> patterns,
    List<Double> weights,
    double primacyGradient
) {
    /**
     * Compact constructor with validation.
     */
    public TemporalPattern {
        if (patterns == null || weights == null) {
            throw new IllegalArgumentException("Patterns and weights cannot be null");
        }
        if (patterns.size() != weights.size()) {
            throw new IllegalArgumentException(
                "Patterns and weights must have same length: " +
                patterns.size() + " vs " + weights.size()
            );
        }
    }

    /**
     * Get the sequence length.
     */
    public int sequenceLength() {
        return patterns.size();
    }

    /**
     * Check if the pattern is valid (non-empty with matching dimensions).
     */
    public boolean isValid() {
        return !patterns.isEmpty();
    }

    /**
     * Get the combined pattern by weighted average.
     * Computes pattern = Σ(pattern_i * weight_i) / Σ(weight_i)
     */
    public double[] getCombinedPattern() {
        if (!isValid()) {
            return new double[0];
        }

        int dimension = patterns.get(0).length;
        var combined = new double[dimension];
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
     * Get the strongest pattern (highest activation weight).
     */
    public double[] getStrongestPattern() {
        if (!isValid()) {
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
     * Get the average activation weight.
     */
    public double getAverageWeight() {
        if (!isValid()) {
            return 0.0;
        }

        double sum = 0.0;
        for (double weight : weights) {
            sum += weight;
        }
        return sum / weights.size();
    }
}
