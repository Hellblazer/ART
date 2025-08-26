package com.hellblazer.art.core.results;

import com.hellblazer.art.core.WeightVector;
import java.util.List;
import java.util.Objects;

/**
 * Record representing the result of winner-take-all category selection in ART algorithms.
 * Encapsulates the winning category information including its index, activation value, and weight.
 */
public record CategoryResult(int winnerIndex, double winnerActivation, WeightVector winnerWeight, 
                           double[] allActivations) {
    
    /**
     * Constructor with validation and defensive copying.
     */
    public CategoryResult {
        if (winnerIndex < 0) {
            throw new IllegalArgumentException("Winner index must be non-negative, got: " + winnerIndex);
        }
        if (Double.isNaN(winnerActivation) || Double.isInfinite(winnerActivation)) {
            throw new IllegalArgumentException("Winner activation must be finite, got: " + winnerActivation);
        }
        Objects.requireNonNull(winnerWeight, "Winner weight cannot be null");
        Objects.requireNonNull(allActivations, "All activations array cannot be null");
        
        if (winnerIndex >= allActivations.length) {
            throw new IllegalArgumentException("Winner index " + winnerIndex + 
                " must be < activations array length " + allActivations.length);
        }
        
        // Validate that winnerActivation matches the activation at winnerIndex
        if (Math.abs(winnerActivation - allActivations[winnerIndex]) > 1e-10) {
            throw new IllegalArgumentException("Winner activation " + winnerActivation + 
                " does not match activation at winner index: " + allActivations[winnerIndex]);
        }
        
        // Validate that winner has the highest activation (or tied for highest)
        for (double activation : allActivations) {
            if (activation > winnerActivation + 1e-10) {
                throw new IllegalArgumentException("Winner activation " + winnerActivation + 
                    " is not the maximum; found higher activation: " + activation);
            }
        }
        
        // Copy array to ensure immutability
        allActivations = java.util.Arrays.copyOf(allActivations, allActivations.length);
    }
    
    /**
     * Create a CategoryResult for the winner among a list of categories.
     * @param winnerIndex the index of the winning category
     * @param categories the list of all categories
     * @param activations the activation values for all categories
     * @return new CategoryResult with winner information
     */
    public static CategoryResult of(int winnerIndex, List<? extends WeightVector> categories, 
                                  double[] activations) {
        Objects.requireNonNull(categories, "Categories list cannot be null");
        if (categories.isEmpty()) {
            throw new IllegalArgumentException("Categories list cannot be empty");
        }
        if (winnerIndex < 0 || winnerIndex >= categories.size()) {
            throw new IllegalArgumentException("Winner index " + winnerIndex + 
                " must be in range [0, " + categories.size() + ")");
        }
        
        var winnerWeight = categories.get(winnerIndex);
        var winnerActivation = activations[winnerIndex];
        
        return new CategoryResult(winnerIndex, winnerActivation, winnerWeight, activations);
    }
    
    /**
     * Get the number of categories that were considered.
     * @return the number of categories
     */
    public int categoryCount() {
        return allActivations.length;
    }
    
    /**
     * Get the activation value for a specific category.
     * @param categoryIndex the category index
     * @return the activation value for that category
     * @throws IndexOutOfBoundsException if index is invalid
     */
    public double getActivation(int categoryIndex) {
        if (categoryIndex < 0 || categoryIndex >= allActivations.length) {
            throw new IndexOutOfBoundsException("Category index " + categoryIndex + 
                " out of bounds for " + allActivations.length + " categories");
        }
        return allActivations[categoryIndex];
    }
    
    /**
     * Check if the winner is unique (no ties for highest activation).
     * @return true if winner is unique, false if there are ties
     */
    public boolean isWinnerUnique() {
        int tieCount = 0;
        for (double activation : allActivations) {
            if (Math.abs(activation - winnerActivation) <= 1e-10) {
                tieCount++;
            }
        }
        return tieCount == 1;
    }
    
    /**
     * Get the margin by which the winner exceeded the runner-up.
     * For tied winners, this returns the margin over the best non-tied activation.
     * @return the activation difference between winner and runner-up
     */
    public double getWinnerMargin() {
        double runnerUpActivation = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < allActivations.length; i++) {
            // Skip activations that are tied with the winner (within tolerance)
            if (Math.abs(allActivations[i] - winnerActivation) <= 1e-10) {
                continue;
            }
            if (allActivations[i] > runnerUpActivation) {
                runnerUpActivation = allActivations[i];
            }
        }
        return runnerUpActivation == Double.NEGATIVE_INFINITY ? 
            winnerActivation : winnerActivation - runnerUpActivation;
    }
    
    @Override
    public String toString() {
        return String.format("CategoryResult{winner=%d, activation=%.6f, categories=%d, unique=%s}", 
                           winnerIndex, winnerActivation, allActivations.length, isWinnerUnique());
    }
}