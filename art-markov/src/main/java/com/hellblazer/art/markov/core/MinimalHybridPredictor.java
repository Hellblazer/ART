package com.hellblazer.art.markov.core;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.markov.parameters.HybridMarkovParameters;

import java.util.*;

/**
 * Minimal hybrid predictor combining ART state abstraction with Markov chain transitions.
 *
 * This is the core component that demonstrates the key innovation: using ART for
 * automatic state discovery from continuous observations, then learning Markov
 * transition dynamics between those discovered states.
 *
 * Key features:
 * - Adaptive state discovery via FuzzyART
 * - Markov transition learning with mathematical validation
 * - Weighted combination of ART and Markov predictions
 * - Performance tracking and convergence monitoring
 * - Mathematical soundness validation
 */
public final class MinimalHybridPredictor implements AutoCloseable {

    private final HybridMarkovParameters parameters;
    private final SimpleStateAbstractionART stateAbstractor;
    private final BasicTransitionLearner transitionLearner;

    // Prediction state
    private int currentState;
    private boolean hasCurrentState;
    private long totalPredictions;
    private long correctPredictions;

    // Performance tracking
    private final Map<String, Double> performanceMetrics;
    private long lastPerformanceUpdate;
    private boolean closed;

    /**
     * Creates a new hybrid predictor.
     *
     * @param parameters The system parameters
     */
    public MinimalHybridPredictor(HybridMarkovParameters parameters) {
        this.parameters = Objects.requireNonNull(parameters, "Parameters cannot be null");
        this.stateAbstractor = new SimpleStateAbstractionART(parameters);
        this.transitionLearner = new BasicTransitionLearner(parameters);

        this.currentState = -1;
        this.hasCurrentState = false;
        this.totalPredictions = 0L;
        this.correctPredictions = 0L;

        this.performanceMetrics = new HashMap<>();
        this.lastPerformanceUpdate = System.nanoTime();
        this.closed = false;

        initializePerformanceMetrics();
    }

    /**
     * Learns from an observation by discovering states and updating transitions.
     *
     * @param observation The continuous observation vector
     * @return The discovered/assigned state for this observation
     */
    public int learn(Pattern observation) {
        ensureNotClosed();

        if (observation == null) {
            throw new IllegalArgumentException("Observation cannot be null");
        }

        // Use ART to discover/assign state
        int newState = stateAbstractor.abstractToState(observation);

        if (newState == -1) {
            // Could not assign a state (should be rare with proper parameters)
            return -1;
        }

        // Learn transition if we have a previous state
        if (hasCurrentState && currentState != -1) {
            transitionLearner.observeTransition(currentState, newState);
        }

        // Update current state
        currentState = newState;
        hasCurrentState = true;

        updatePerformanceMetrics();
        return newState;
    }

    /**
     * Predicts the next state given the current observation.
     * Combines ART-based state classification with Markov transition prediction.
     *
     * @param observation The current observation
     * @return Prediction result containing state probabilities and metadata
     */
    public PredictionResult predict(Pattern observation) {
        ensureNotClosed();

        if (observation == null) {
            throw new IllegalArgumentException("Observation cannot be null");
        }

        long startTime = System.nanoTime();

        // Step 1: Use ART to classify current observation to a state
        int observedState = stateAbstractor.predictState(observation);

        if (observedState == -1) {
            // Cannot classify observation - return uniform prediction
            return createUniformPrediction(startTime);
        }

        // Step 2: Get ART-based prediction (uniform from current state)
        var artPrediction = createUniformDistribution();

        // Step 3: Get Markov-based prediction from transition matrix
        var markovPrediction = transitionLearner.predictNextState(observedState);

        // Step 4: Combine predictions using hybrid weight
        double hybridWeight = parameters.hybridWeight();
        var combinedPrediction = new double[parameters.maxStates()];

        for (int i = 0; i < combinedPrediction.length; i++) {
            combinedPrediction[i] = hybridWeight * artPrediction[i] +
                                   (1.0 - hybridWeight) * markovPrediction[i];
        }

        // Normalize to ensure valid probability distribution
        combinedPrediction = ValidationLayer.normalizeRow(combinedPrediction);

        // Find most likely next state
        int mostLikelyState = findMaxIndex(combinedPrediction);

        long elapsedNanos = System.nanoTime() - startTime;
        totalPredictions++;

        return new PredictionResult(
            observedState,
            mostLikelyState,
            combinedPrediction,
            artPrediction,
            markovPrediction,
            elapsedNanos
        );
    }

    /**
     * Validates a prediction against the actual next observation.
     *
     * @param prediction The prediction result
     * @param actualNextObservation The actual next observation
     * @return true if the prediction was correct
     */
    public boolean validatePrediction(PredictionResult prediction, Pattern actualNextObservation) {
        ensureNotClosed();

        if (prediction == null || actualNextObservation == null) {
            return false;
        }

        // Classify the actual next observation
        int actualNextState = stateAbstractor.predictState(actualNextObservation);

        if (actualNextState == -1) {
            return false; // Cannot validate if we can't classify the actual observation
        }

        boolean correct = (actualNextState == prediction.mostLikelyNextState());
        if (correct) {
            correctPredictions++;
        }

        return correct;
    }

    /**
     * Gets comprehensive system statistics including ART and Markov components.
     *
     * @return Map of statistics
     */
    public Map<String, Object> getStatistics() {
        ensureNotClosed();

        var stats = new HashMap<String, Object>();

        // Overall system stats
        stats.put("totalPredictions", totalPredictions);
        stats.put("correctPredictions", correctPredictions);
        stats.put("accuracy", totalPredictions > 0 ? (double) correctPredictions / totalPredictions : 0.0);
        stats.put("currentState", currentState);
        stats.put("hasCurrentState", hasCurrentState);

        // Component stats
        stats.put("artStats", stateAbstractor.getStatistics());
        stats.put("markovStats", transitionLearner.getStatistics());

        // System properties
        stats.put("hasConverged", transitionLearner.hasConverged());
        stats.put("satisfiesMarkovProperty", transitionLearner.satisfiesMarkovProperty());
        stats.put("steadyState", transitionLearner.getSteadyStateDistribution());

        // Performance metrics
        stats.putAll(performanceMetrics);

        return stats;
    }

    /**
     * Gets the current transition matrix.
     *
     * @return The transition matrix
     */
    public double[][] getTransitionMatrix() {
        ensureNotClosed();
        return transitionLearner.getTransitionMatrix();
    }

    /**
     * Gets the current state count.
     *
     * @return Number of discovered states
     */
    public int getStateCount() {
        ensureNotClosed();
        return stateAbstractor.getStateCount();
    }

    /**
     * Sets a label for a discovered state.
     *
     * @param stateId The state ID
     * @param label The human-readable label
     */
    public void setStateLabel(int stateId, String label) {
        ensureNotClosed();
        stateAbstractor.setStateLabel(stateId, label);
    }

    /**
     * Gets the label for a state.
     *
     * @param stateId The state ID
     * @return The state label
     */
    public String getStateLabel(int stateId) {
        ensureNotClosed();
        return stateAbstractor.getStateLabel(stateId);
    }

    /**
     * Clears all learned data and resets the system.
     */
    public void clear() {
        ensureNotClosed();

        stateAbstractor.clear();
        transitionLearner.clear();

        currentState = -1;
        hasCurrentState = false;
        totalPredictions = 0L;
        correctPredictions = 0L;

        initializePerformanceMetrics();
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            stateAbstractor.close();
            performanceMetrics.clear();
        }
    }

    /**
     * Gets the underlying state abstractor for advanced operations.
     *
     * @return The state abstractor
     */
    public SimpleStateAbstractionART getStateAbstractor() {
        ensureNotClosed();
        return stateAbstractor;
    }

    /**
     * Gets the underlying transition learner for advanced operations.
     *
     * @return The transition learner
     */
    public BasicTransitionLearner getTransitionLearner() {
        ensureNotClosed();
        return transitionLearner;
    }

    // Private helper methods

    private void ensureNotClosed() {
        if (closed) {
            throw new IllegalStateException("MinimalHybridPredictor is closed");
        }
    }

    private void initializePerformanceMetrics() {
        performanceMetrics.clear();
        performanceMetrics.put("averagePredictionTimeNanos", 0.0);
        performanceMetrics.put("memoryUsageBytes", 0.0);
        performanceMetrics.put("throughputPredictionsPerSecond", 0.0);
        lastPerformanceUpdate = System.nanoTime();
    }

    private void updatePerformanceMetrics() {
        long currentTime = System.nanoTime();
        long elapsedNanos = currentTime - lastPerformanceUpdate;

        if (elapsedNanos > 1_000_000_000L) { // Update every second
            // Update throughput
            double throughput = (double) totalPredictions / (elapsedNanos / 1e9);
            performanceMetrics.put("throughputPredictionsPerSecond", throughput);

            // Estimate memory usage (rough approximation)
            long memoryUsage = estimateMemoryUsage();
            performanceMetrics.put("memoryUsageBytes", (double) memoryUsage);

            lastPerformanceUpdate = currentTime;
        }
    }

    private long estimateMemoryUsage() {
        // Rough memory usage estimation
        long usage = 0L;

        // Transition matrix: maxStates^2 * 8 bytes per double
        usage += (long) parameters.maxStates() * parameters.maxStates() * 8L;

        // State abstractor: approximately 1KB per state
        usage += stateAbstractor.getStateCount() * 1024L;

        // Various overhead
        usage += 10240L; // 10KB overhead

        return usage;
    }

    private double[] createUniformDistribution() {
        var distribution = new double[parameters.maxStates()];
        Arrays.fill(distribution, 1.0 / parameters.maxStates());
        return distribution;
    }

    private PredictionResult createUniformPrediction(long startTime) {
        var uniform = createUniformDistribution();
        long elapsedNanos = System.nanoTime() - startTime;

        return new PredictionResult(
            -1,  // observed state unknown
            0,   // default to first state
            uniform,
            uniform,
            uniform,
            elapsedNanos
        );
    }

    private int findMaxIndex(double[] array) {
        int maxIndex = 0;
        double maxValue = array[0];

        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    /**
     * Result of a hybrid prediction containing both ART and Markov components.
     */
    public record PredictionResult(
        int observedState,
        int mostLikelyNextState,
        double[] combinedPrediction,
        double[] artPrediction,
        double[] markovPrediction,
        long predictionTimeNanos
    ) {

        public PredictionResult {
            // Validate inputs
            if (combinedPrediction == null || artPrediction == null || markovPrediction == null) {
                throw new IllegalArgumentException("Prediction arrays cannot be null");
            }

            if (combinedPrediction.length != artPrediction.length ||
                combinedPrediction.length != markovPrediction.length) {
                throw new IllegalArgumentException("Prediction arrays must have the same length");
            }

            // Validate probability distributions
            ValidationLayer.validateProbabilityDistribution(combinedPrediction);
            ValidationLayer.validateProbabilityDistribution(artPrediction);
            ValidationLayer.validateProbabilityDistribution(markovPrediction);
        }

        /**
         * Gets the confidence in the most likely prediction.
         *
         * @return The probability of the most likely next state
         */
        public double getConfidence() {
            return combinedPrediction[mostLikelyNextState];
        }

        /**
         * Gets the entropy of the combined prediction (measure of uncertainty).
         *
         * @return The Shannon entropy of the prediction
         */
        public double getEntropy() {
            double entropy = 0.0;
            for (double p : combinedPrediction) {
                if (p > 1e-12) { // Avoid log(0)
                    entropy -= p * Math.log(p);
                }
            }
            return entropy;
        }
    }
}