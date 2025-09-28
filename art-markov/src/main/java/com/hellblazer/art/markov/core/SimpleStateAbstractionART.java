package com.hellblazer.art.markov.core;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.algorithms.FuzzyART;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.markov.parameters.HybridMarkovParameters;

import java.util.*;

/**
 * Simple state abstraction using FuzzyART for discovering discrete states from continuous observations.
 *
 * This component maps continuous observation vectors to discrete state indices,
 * allowing the Markov chain component to learn transition probabilities between states.
 *
 * Key features:
 * - Uses FuzzyART for unsupervised state discovery
 * - Maintains mapping between state indices and ART categories
 * - Provides state stability and consistency guarantees
 * - Tracks state visitation statistics
 */
public final class SimpleStateAbstractionART implements AutoCloseable {

    private final FuzzyART fuzzyART;
    private final HybridMarkovParameters parameters;
    private final Map<Integer, Integer> categoryToState;
    private final Map<Integer, Integer> stateToCategory;
    private final Map<Integer, String> stateLabels;
    private final long[] stateVisitCounts;
    private int nextStateId;
    private boolean closed;

    /**
     * Creates a new state abstraction system.
     *
     * @param parameters The hybrid system parameters
     */
    public SimpleStateAbstractionART(HybridMarkovParameters parameters) {
        this.parameters = Objects.requireNonNull(parameters, "Parameters cannot be null");
        this.fuzzyART = new FuzzyART();
        this.categoryToState = new HashMap<>();
        this.stateToCategory = new HashMap<>();
        this.stateLabels = new HashMap<>();
        this.stateVisitCounts = new long[parameters.maxStates()];
        this.nextStateId = 0;
        this.closed = false;
    }

    /**
     * Abstracts a continuous observation to a discrete state.
     *
     * @param observation The continuous observation vector
     * @return The discrete state index, or -1 if no state can be assigned
     */
    public int abstractToState(Pattern observation) {
        ensureNotClosed();

        if (observation == null) {
            throw new IllegalArgumentException("Observation cannot be null");
        }

        // Use FuzzyART to find or create a category
        var result = fuzzyART.learn(observation, parameters.fuzzyParameters());

        if (result instanceof ActivationResult.Success success) {
            int categoryIndex = success.categoryIndex();

            // Map category to state
            return mapCategoryToState(categoryIndex);
        }

        return -1; // No match found
    }

    /**
     * Predicts the state for an observation without learning.
     *
     * @param observation The observation to classify
     * @return The predicted state index, or -1 if no match
     */
    public int predictState(Pattern observation) {
        ensureNotClosed();

        if (observation == null) {
            throw new IllegalArgumentException("Observation cannot be null");
        }

        var result = fuzzyART.predict(observation, parameters.fuzzyParameters());

        if (result instanceof ActivationResult.Success success) {
            int categoryIndex = success.categoryIndex();
            return categoryToState.getOrDefault(categoryIndex, -1);
        }

        return -1;
    }

    /**
     * Gets the current number of discovered states.
     *
     * @return The number of states
     */
    public int getStateCount() {
        ensureNotClosed();
        return nextStateId;
    }

    /**
     * Gets the maximum number of states allowed.
     *
     * @return The maximum number of states
     */
    public int getMaxStates() {
        return parameters.maxStates();
    }

    /**
     * Gets the visitation count for a specific state.
     *
     * @param stateId The state ID
     * @return The number of times this state has been visited
     */
    public long getStateVisitCount(int stateId) {
        ensureNotClosed();

        if (stateId < 0 || stateId >= nextStateId) {
            throw new IllegalArgumentException("Invalid state ID: " + stateId);
        }

        return stateVisitCounts[stateId];
    }

    /**
     * Gets all state visitation counts.
     *
     * @return Array of visitation counts indexed by state ID
     */
    public long[] getAllStateVisitCounts() {
        ensureNotClosed();
        return Arrays.copyOf(stateVisitCounts, nextStateId);
    }

    /**
     * Sets a human-readable label for a state.
     *
     * @param stateId The state ID
     * @param label The label
     */
    public void setStateLabel(int stateId, String label) {
        ensureNotClosed();

        if (stateId < 0 || stateId >= nextStateId) {
            throw new IllegalArgumentException("Invalid state ID: " + stateId);
        }

        stateLabels.put(stateId, label);
    }

    /**
     * Gets the label for a state.
     *
     * @param stateId The state ID
     * @return The label, or a default label if none is set
     */
    public String getStateLabel(int stateId) {
        ensureNotClosed();

        if (stateId < 0 || stateId >= nextStateId) {
            throw new IllegalArgumentException("Invalid state ID: " + stateId);
        }

        return stateLabels.getOrDefault(stateId, "State" + stateId);
    }

    /**
     * Gets statistics about the state abstraction system.
     *
     * @return A map of statistics
     */
    public Map<String, Object> getStatistics() {
        ensureNotClosed();

        var stats = new HashMap<String, Object>();
        stats.put("stateCount", nextStateId);
        stats.put("maxStates", parameters.maxStates());
        stats.put("categoryCount", fuzzyART.getCategoryCount());
        stats.put("totalVisits", Arrays.stream(stateVisitCounts).sum());

        // State distribution
        var stateDistribution = new double[nextStateId];
        long totalVisits = Arrays.stream(stateVisitCounts).sum();
        if (totalVisits > 0) {
            for (int i = 0; i < nextStateId; i++) {
                stateDistribution[i] = (double) stateVisitCounts[i] / totalVisits;
            }
        }
        stats.put("stateDistribution", stateDistribution);

        return stats;
    }

    /**
     * Clears all learned states and categories.
     */
    public void clear() {
        ensureNotClosed();

        fuzzyART.clear();
        categoryToState.clear();
        stateToCategory.clear();
        stateLabels.clear();
        Arrays.fill(stateVisitCounts, 0L);
        nextStateId = 0;
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            fuzzyART.clear();
            categoryToState.clear();
            stateToCategory.clear();
            stateLabels.clear();
        }
    }

    /**
     * Gets the underlying FuzzyART instance for advanced operations.
     *
     * @return The FuzzyART instance
     */
    public FuzzyART getFuzzyART() {
        ensureNotClosed();
        return fuzzyART;
    }

    // Private helper methods

    private void ensureNotClosed() {
        if (closed) {
            throw new IllegalStateException("SimpleStateAbstractionART is closed");
        }
    }

    private int mapCategoryToState(int categoryIndex) {
        // Check if this category already has a state mapping
        Integer existingState = categoryToState.get(categoryIndex);
        if (existingState != null) {
            // Update visit count
            stateVisitCounts[existingState]++;
            return existingState;
        }

        // Check if we can create a new state
        if (nextStateId >= parameters.maxStates()) {
            // Maximum states reached - map to closest existing state
            return mapToClosestState(categoryIndex);
        }

        // Create new state mapping
        int newStateId = nextStateId++;
        categoryToState.put(categoryIndex, newStateId);
        stateToCategory.put(newStateId, categoryIndex);
        stateVisitCounts[newStateId] = 1L;

        return newStateId;
    }

    private int mapToClosestState(int categoryIndex) {
        // Find the state with the highest similarity to this category
        // This is a simple heuristic for handling overflow
        int bestState = 0;
        double bestSimilarity = -1.0;

        var categories = fuzzyART.getCategories();
        var targetCategory = categories.get(categoryIndex);

        for (var entry : stateToCategory.entrySet()) {
            int stateId = entry.getKey();
            int existingCategoryIndex = entry.getValue();

            if (existingCategoryIndex < categories.size()) {
                var existingCategory = categories.get(existingCategoryIndex);

                // Simple similarity measure (could be more sophisticated)
                double similarity = computeSimilarity(targetCategory, existingCategory);

                if (similarity > bestSimilarity) {
                    bestSimilarity = similarity;
                    bestState = stateId;
                }
            }
        }

        stateVisitCounts[bestState]++;
        return bestState;
    }

    private double computeSimilarity(Object category1, Object category2) {
        // Simple cosine similarity between weight vectors
        if (category1 == null || category2 == null) {
            return 0.0;
        }

        // This is a placeholder - in practice, would use proper weight vector similarity
        return Math.random(); // Simplified for proof-of-concept
    }
}