package com.hellblazer.art.markov.core;

import com.hellblazer.art.markov.parameters.HybridMarkovParameters;

import java.util.*;

/**
 * Basic transition learner for building Markov chain transition matrices.
 *
 * This component learns transition probabilities between discrete states and
 * maintains mathematically sound stochastic matrices. It supports:
 * - Online learning of transition probabilities
 * - Laplace smoothing for unseen transitions
 * - Stochastic matrix validation
 * - Steady-state computation
 * - Convergence detection
 */
public final class BasicTransitionLearner {

    private final HybridMarkovParameters parameters;
    private final int maxStates;
    private final double smoothingFactor;

    // Transition counts and probabilities
    private final long[][] transitionCounts;
    private double[][] transitionMatrix;
    private boolean matrixDirty;

    // State sequence tracking for Markov property testing
    private final Deque<Integer> recentStates;
    private final List<Integer> fullStateSequence;

    // Statistics
    private long totalTransitions;
    private double[] steadyStateDistribution;
    private boolean steadyStateValid;

    /**
     * Creates a new transition learner.
     *
     * @param parameters The hybrid system parameters
     */
    public BasicTransitionLearner(HybridMarkovParameters parameters) {
        this.parameters = Objects.requireNonNull(parameters, "Parameters cannot be null");
        this.maxStates = parameters.maxStates();
        this.smoothingFactor = parameters.transitionSmoothingFactor();

        this.transitionCounts = new long[maxStates][maxStates];
        this.transitionMatrix = new double[maxStates][maxStates];
        this.matrixDirty = true;

        this.recentStates = new ArrayDeque<>(parameters.memoryWindow());
        this.fullStateSequence = new ArrayList<>();

        this.totalTransitions = 0L;
        this.steadyStateValid = false;

        // Initialize with uniform smoothing
        initializeWithSmoothing();
    }

    /**
     * Observes a state transition and updates the transition probabilities.
     *
     * @param fromState The source state
     * @param toState The destination state
     */
    public void observeTransition(int fromState, int toState) {
        validateStateIndex(fromState);
        validateStateIndex(toState);

        // Update transition counts
        transitionCounts[fromState][toState]++;
        totalTransitions++;
        matrixDirty = true;
        steadyStateValid = false;

        // Update state sequence tracking
        recentStates.addLast(toState);
        if (recentStates.size() > parameters.memoryWindow()) {
            recentStates.removeFirst();
        }
        fullStateSequence.add(toState);

        // Limit full sequence size for memory management
        if (fullStateSequence.size() > 10000) {
            fullStateSequence.subList(0, 5000).clear();
        }
    }

    /**
     * Observes a sequence of state transitions.
     *
     * @param stateSequence The sequence of states
     */
    public void observeSequence(int[] stateSequence) {
        if (stateSequence == null || stateSequence.length < 2) {
            throw new IllegalArgumentException("State sequence must have at least 2 states");
        }

        for (int i = 1; i < stateSequence.length; i++) {
            observeTransition(stateSequence[i - 1], stateSequence[i]);
        }
    }

    /**
     * Gets the current transition matrix.
     * The matrix is computed lazily and cached until transitions are updated.
     *
     * @return A copy of the current transition matrix
     */
    public double[][] getTransitionMatrix() {
        if (matrixDirty) {
            recomputeTransitionMatrix();
        }

        // Return a deep copy to prevent external modification
        var result = new double[maxStates][];
        for (int i = 0; i < maxStates; i++) {
            result[i] = Arrays.copyOf(transitionMatrix[i], maxStates);
        }

        return result;
    }

    /**
     * Predicts the next state given the current state.
     *
     * @param currentState The current state
     * @return A probability distribution over next states
     */
    public double[] predictNextState(int currentState) {
        validateStateIndex(currentState);

        if (matrixDirty) {
            recomputeTransitionMatrix();
        }

        return Arrays.copyOf(transitionMatrix[currentState], maxStates);
    }

    /**
     * Gets the most likely next state given the current state.
     *
     * @param currentState The current state
     * @return The most likely next state
     */
    public int getMostLikelyNextState(int currentState) {
        var probabilities = predictNextState(currentState);

        int bestState = 0;
        double bestProbability = probabilities[0];

        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > bestProbability) {
                bestProbability = probabilities[i];
                bestState = i;
            }
        }

        return bestState;
    }

    /**
     * Computes the steady-state distribution of the Markov chain.
     *
     * @return The steady-state probability distribution
     */
    public double[] getSteadyStateDistribution() {
        if (!steadyStateValid) {
            recomputeSteadyState();
        }

        return Arrays.copyOf(steadyStateDistribution, maxStates);
    }

    /**
     * Checks if the Markov chain has converged to steady state.
     *
     * @return true if the chain has converged
     */
    public boolean hasConverged() {
        if (matrixDirty) {
            recomputeTransitionMatrix();
        }

        return ValidationLayer.hasConverged(transitionMatrix, parameters.convergenceThreshold());
    }

    /**
     * Tests if the observed transitions satisfy the Markov property.
     *
     * @return true if the Markov property appears to hold
     */
    public boolean satisfiesMarkovProperty() {
        if (fullStateSequence.size() < 10) {
            return false; // Insufficient data
        }

        var sequence = fullStateSequence.stream().mapToInt(Integer::intValue).toArray();
        return ValidationLayer.testMarkovProperty(sequence, 0.1);
    }

    /**
     * Gets statistics about the transition learning.
     *
     * @return A map of statistics
     */
    public Map<String, Object> getStatistics() {
        var stats = new HashMap<String, Object>();
        stats.put("totalTransitions", totalTransitions);
        stats.put("hasConverged", hasConverged());
        stats.put("satisfiesMarkovProperty", satisfiesMarkovProperty());
        stats.put("sequenceLength", fullStateSequence.size());

        // Transition matrix entropy (measure of randomness)
        if (matrixDirty) {
            recomputeTransitionMatrix();
        }
        stats.put("matrixEntropy", computeMatrixEntropy());

        // State persistence (diagonal elements)
        var persistence = new double[maxStates];
        for (int i = 0; i < maxStates; i++) {
            persistence[i] = transitionMatrix[i][i];
        }
        stats.put("statePersistence", persistence);

        return stats;
    }

    /**
     * Clears all learned transitions and resets the system.
     */
    public void clear() {
        // Clear transition counts
        for (int i = 0; i < maxStates; i++) {
            Arrays.fill(transitionCounts[i], 0L);
        }

        recentStates.clear();
        fullStateSequence.clear();
        totalTransitions = 0L;
        matrixDirty = true;
        steadyStateValid = false;

        // Reinitialize with smoothing
        initializeWithSmoothing();
    }

    /**
     * Gets the transition count between two states.
     *
     * @param fromState The source state
     * @param toState The destination state
     * @return The number of observed transitions
     */
    public long getTransitionCount(int fromState, int toState) {
        validateStateIndex(fromState);
        validateStateIndex(toState);
        return transitionCounts[fromState][toState];
    }

    /**
     * Gets the total number of transitions observed.
     *
     * @return The total transition count
     */
    public long getTotalTransitions() {
        return totalTransitions;
    }

    // Private helper methods

    private void validateStateIndex(int state) {
        if (state < 0 || state >= maxStates) {
            throw new IllegalArgumentException(
                "State index " + state + " is out of bounds [0, " + maxStates + ")"
            );
        }
    }

    private void initializeWithSmoothing() {
        // Initialize with small uniform smoothing to ensure valid probabilities
        for (int i = 0; i < maxStates; i++) {
            for (int j = 0; j < maxStates; j++) {
                transitionMatrix[i][j] = smoothingFactor / maxStates;
            }
        }

        // Normalize rows to ensure stochastic property
        for (int i = 0; i < maxStates; i++) {
            transitionMatrix[i] = ValidationLayer.normalizeRow(transitionMatrix[i]);
        }

        matrixDirty = false;
    }

    private void recomputeTransitionMatrix() {
        for (int i = 0; i < maxStates; i++) {
            // Compute row totals for normalization
            long rowTotal = Arrays.stream(transitionCounts[i]).sum();

            if (rowTotal == 0) {
                // No observations for this state - use uniform distribution with smoothing
                Arrays.fill(transitionMatrix[i], smoothingFactor / maxStates);
            } else {
                // Apply Laplace smoothing: (count + α) / (total + α * n)
                double denominator = rowTotal + smoothingFactor;
                for (int j = 0; j < maxStates; j++) {
                    transitionMatrix[i][j] = (transitionCounts[i][j] + smoothingFactor / maxStates) / denominator;
                }
            }

            // Ensure row is properly normalized
            transitionMatrix[i] = ValidationLayer.normalizeRow(transitionMatrix[i]);
        }

        // Validate the resulting matrix
        ValidationLayer.validateStochasticMatrix(transitionMatrix);

        matrixDirty = false;
    }

    private void recomputeSteadyState() {
        if (matrixDirty) {
            recomputeTransitionMatrix();
        }

        steadyStateDistribution = ValidationLayer.computeSteadyState(
            transitionMatrix,
            1000, // max iterations
            1e-8  // tolerance
        );

        steadyStateValid = true;
    }

    private double computeMatrixEntropy() {
        double totalEntropy = 0.0;

        for (int i = 0; i < maxStates; i++) {
            double rowEntropy = 0.0;
            for (int j = 0; j < maxStates; j++) {
                double p = transitionMatrix[i][j];
                if (p > 1e-12) { // Avoid log(0)
                    rowEntropy -= p * Math.log(p);
                }
            }
            totalEntropy += rowEntropy;
        }

        return totalEntropy / maxStates;
    }
}