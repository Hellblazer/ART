package com.hellblazer.art.hartcq.feedback;

import com.hellblazer.art.hartcq.HARTCQConfig;
import com.hellblazer.art.hartcq.core.MultiChannelProcessor;
import com.hellblazer.art.hartcq.Token;
import com.hellblazer.art.hartcq.templates.Template;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.DoubleAdder;

/**
 * Implements the core feedback loop for HART-CQ processing.
 * Coordinates bi-directional signal flow between bottom-up data processing
 * and top-down expectation generation until convergence is achieved.
 *
 * The feedback loop maintains loop stability and detects convergence conditions
 * while preventing oscillations and ensuring deterministic behavior.
 *
 * @author Claude Code
 */
public class FeedbackLoop implements AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(FeedbackLoop.class);

    private final HARTCQConfig config;
    private final ExpectationManager expectationManager;
    private final ResonanceDetector resonanceDetector;
    private final AdaptationController adaptationController;
    private final MultiChannelProcessor channelProcessor;

    // Loop control parameters
    private final int maxIterations;
    private final double convergenceThreshold;
    private final long maxExecutionTimeMs;
    private final double dampingFactor;

    // Statistics
    private final AtomicInteger totalRuns;
    private final AtomicInteger convergedRuns;
    private final AtomicInteger maxIterationsReached;
    private final AtomicLong totalExecutionTime;
    private final DoubleAdder totalConvergenceTime;

    /**
     * Result of feedback loop execution.
     */
    public static class FeedbackLoopResult {
        private final boolean converged;
        private final int cycles;
        private final double convergenceScore;
        private final double resonanceStrength;
        private final Template matchedTemplate;
        private final float[] processedFeatures;
        private final String prediction;
        private final double confidence;
        private final long executionTimeMs;
        private final String convergenceReason;

        public FeedbackLoopResult(boolean converged, int cycles, double convergenceScore,
                                double resonanceStrength, Template matchedTemplate,
                                float[] processedFeatures, String prediction, double confidence,
                                long executionTimeMs, String convergenceReason) {
            this.converged = converged;
            this.cycles = cycles;
            this.convergenceScore = Math.max(0.0, Math.min(1.0, convergenceScore));
            this.resonanceStrength = Math.max(0.0, Math.min(1.0, resonanceStrength));
            this.matchedTemplate = matchedTemplate;
            this.processedFeatures = processedFeatures != null ? processedFeatures.clone() : null;
            this.prediction = prediction;
            this.confidence = Math.max(0.0, Math.min(1.0, confidence));
            this.executionTimeMs = executionTimeMs;
            this.convergenceReason = convergenceReason;
        }

        public boolean isConverged() { return converged; }
        public int getCycles() { return cycles; }
        public double getConvergenceScore() { return convergenceScore; }
        public double getResonanceStrength() { return resonanceStrength; }
        public Template getMatchedTemplate() { return matchedTemplate; }
        public float[] getProcessedFeatures() {
            return processedFeatures != null ? processedFeatures.clone() : null;
        }
        public String getPrediction() { return prediction; }
        public double getConfidence() { return confidence; }
        public long getExecutionTimeMs() { return executionTimeMs; }
        public String getConvergenceReason() { return convergenceReason; }

        @Override
        public String toString() {
            return "FeedbackLoopResult{converged=%s, cycles=%d, score=%.3f, resonance=%.3f, prediction='%s', confidence=%.3f}"
                .formatted(converged, cycles, convergenceScore, resonanceStrength, prediction, confidence);
        }
    }

    /**
     * Internal state tracking for the feedback loop iteration.
     */
    private static class LoopState {
        private float[] currentFeatures;
        private float[] expectedFeatures;
        private ExpectationManager.ExpectationResult currentExpectation;
        private ResonanceDetector.ResonanceResult currentResonance;
        private AdaptationController.AdaptationResult currentAdaptation;
        private double convergenceScore;
        private boolean isStable;
        private int iterationCount;

        public LoopState() {
            this.convergenceScore = 0.0;
            this.isStable = false;
            this.iterationCount = 0;
        }

        public void updateFeatures(float[] features) {
            this.currentFeatures = features != null ? features.clone() : null;
        }

        public void updateExpectedFeatures(float[] expectedFeatures) {
            this.expectedFeatures = expectedFeatures != null ? expectedFeatures.clone() : null;
        }

        public void updateExpectation(ExpectationManager.ExpectationResult expectation) {
            this.currentExpectation = expectation;
        }

        public void updateResonance(ResonanceDetector.ResonanceResult resonance) {
            this.currentResonance = resonance;
        }

        public void updateAdaptation(AdaptationController.AdaptationResult adaptation) {
            this.currentAdaptation = adaptation;
        }

        public void updateConvergence(double score, boolean stable) {
            this.convergenceScore = score;
            this.isStable = stable;
        }

        public void incrementIteration() {
            this.iterationCount++;
        }

        // Getters
        public float[] getCurrentFeatures() { return currentFeatures; }
        public float[] getExpectedFeatures() { return expectedFeatures; }
        public ExpectationManager.ExpectationResult getCurrentExpectation() { return currentExpectation; }
        public ResonanceDetector.ResonanceResult getCurrentResonance() { return currentResonance; }
        public AdaptationController.AdaptationResult getCurrentAdaptation() { return currentAdaptation; }
        public double getConvergenceScore() { return convergenceScore; }
        public boolean isStable() { return isStable; }
        public int getIterationCount() { return iterationCount; }
    }

    /**
     * Creates a feedback loop with the given components and configuration.
     */
    public FeedbackLoop(HARTCQConfig config, ExpectationManager expectationManager,
                       ResonanceDetector resonanceDetector, AdaptationController adaptationController) {
        this.config = Objects.requireNonNull(config, "Configuration cannot be null");
        this.expectationManager = Objects.requireNonNull(expectationManager, "ExpectationManager cannot be null");
        this.resonanceDetector = Objects.requireNonNull(resonanceDetector, "ResonanceDetector cannot be null");
        this.adaptationController = Objects.requireNonNull(adaptationController, "AdaptationController cannot be null");

        this.channelProcessor = new MultiChannelProcessor();

        // Configure loop parameters
        this.maxIterations = 10; // Reasonable limit for convergence
        this.convergenceThreshold = config.getTemplateConfig().getMatchThreshold();
        this.maxExecutionTimeMs = config.getPerformanceConfig().getProcessingTimeoutMs();
        this.dampingFactor = 0.1; // Prevent oscillations

        // Initialize statistics
        this.totalRuns = new AtomicInteger(0);
        this.convergedRuns = new AtomicInteger(0);
        this.maxIterationsReached = new AtomicInteger(0);
        this.totalExecutionTime = new AtomicLong(0);
        this.totalConvergenceTime = new DoubleAdder();

        logger.info("FeedbackLoop initialized: maxIterations={}, convergenceThreshold={}, maxExecutionTime={}ms",
                   maxIterations, convergenceThreshold, maxExecutionTimeMs);
    }

    /**
     * Runs the feedback loop for the given input tokens.
     *
     * @param inputTokens Input token window
     * @param supervisedLabel Optional supervised label (can be null)
     * @return Feedback loop result
     */
    public FeedbackLoopResult run(Token[] inputTokens, String supervisedLabel) {
        if (inputTokens == null || inputTokens.length == 0) {
            return createFailureResult("No input tokens provided", 0, 0);
        }

        var startTime = System.currentTimeMillis();
        var state = new LoopState();

        try {
            totalRuns.incrementAndGet();

            logger.debug("Starting feedback loop for {} tokens", inputTokens.length);

            // Initial bottom-up processing
            var initialFeatures = processBottomUp(inputTokens);
            state.updateFeatures(initialFeatures);

            // Main feedback loop
            while (!hasConverged(state) && !shouldTerminate(state, startTime)) {
                state.incrementIteration();

                logger.debug("Feedback loop iteration {}", state.getIterationCount());

                // Generate top-down expectations
                var expectationResult = generateTopDownExpectations(inputTokens, state);
                state.updateExpectation(expectationResult);

                // Create expected features from expectations
                var expectedFeatures = createExpectedFeatures(expectationResult, state.getCurrentFeatures());
                state.updateExpectedFeatures(expectedFeatures);

                // Check resonance between bottom-up and top-down
                var resonanceResult = checkResonance(state.getCurrentFeatures(), expectedFeatures,
                                                   expectationResult.getSuggestedTemplate());
                state.updateResonance(resonanceResult);

                // Adapt parameters if needed
                var adaptationResult = adaptParameters(resonanceResult, expectationResult.getSuggestedTemplate(),
                                                     state.getIterationCount());
                state.updateAdaptation(adaptationResult);

                // Update features with damping to prevent oscillations
                if (resonanceResult.isResonant()) {
                    var dampedFeatures = applyDamping(state.getCurrentFeatures(), expectedFeatures);
                    state.updateFeatures(dampedFeatures);
                }

                // Calculate convergence score
                var convergenceScore = calculateConvergenceScore(state);
                var isStable = checkStability(state);
                state.updateConvergence(convergenceScore, isStable);

                logger.debug("Loop iteration {} - convergence: {:.3f}, resonance: {}, stable: {}",
                           state.getIterationCount(), convergenceScore, resonanceResult.isResonant(), isStable);

                // Update expectation manager with observation
                if (supervisedLabel != null && resonanceResult.isResonant()) {
                    expectationManager.updateWithObservation(inputTokens, supervisedLabel,
                                                           expectationResult.getSuggestedTemplate());
                }
            }

            // Create result
            var result = createSuccessResult(state, System.currentTimeMillis() - startTime, supervisedLabel);

            // Update statistics
            if (result.isConverged()) {
                convergedRuns.incrementAndGet();
                totalConvergenceTime.add(result.getExecutionTimeMs());
            }
            if (state.getIterationCount() >= maxIterations) {
                maxIterationsReached.incrementAndGet();
            }

            logger.debug("Feedback loop completed: {}", result);
            return result;

        } catch (Exception e) {
            logger.error("Error in feedback loop execution", e);
            return createFailureResult("Execution error: " + e.getMessage(),
                                     state.getIterationCount(), System.currentTimeMillis() - startTime);
        } finally {
            totalExecutionTime.addAndGet(System.currentTimeMillis() - startTime);
        }
    }

    /**
     * Gets convergence statistics for the feedback loop.
     *
     * @return convergence statistics
     */
    public ConvergenceStats getConvergenceStats() {
        var totalRunsCount = totalRuns.get();
        var convergenceRate = totalRunsCount > 0 ? (double) convergedRuns.get() / totalRunsCount : 0.0;
        var avgExecutionTime = totalRunsCount > 0 ? (double) totalExecutionTime.get() / totalRunsCount : 0.0;
        var avgConvergenceTime = convergedRuns.get() > 0 ? totalConvergenceTime.sum() / convergedRuns.get() : 0.0;

        return new ConvergenceStats(
            totalRunsCount,
            convergedRuns.get(),
            convergenceRate,
            avgExecutionTime,
            avgConvergenceTime,
            maxIterationsReached.get()
        );
    }

    /**
     * Resets the feedback loop statistics and state.
     */
    public void reset() {
        logger.info("Resetting feedback loop");

        totalRuns.set(0);
        convergedRuns.set(0);
        maxIterationsReached.set(0);
        totalExecutionTime.set(0);
        totalConvergenceTime.reset();

        logger.info("Feedback loop reset completed");
    }

    /**
     * Processes input tokens through bottom-up channels.
     */
    private float[] processBottomUp(Token[] tokens) {
        return channelProcessor.processWindow(tokens);
    }

    /**
     * Generates top-down expectations based on current state.
     */
    private ExpectationManager.ExpectationResult generateTopDownExpectations(Token[] tokens, LoopState state) {
        return expectationManager.generateExpectations(tokens);
    }

    /**
     * Creates expected feature vector from expectation result.
     */
    private float[] createExpectedFeatures(ExpectationManager.ExpectationResult expectation, float[] currentFeatures) {
        if (currentFeatures == null) {
            return new float[0];
        }

        var expectedFeatures = new float[currentFeatures.length];

        // If we have a template, use it to guide expectations
        if (expectation.getSuggestedTemplate() != null) {
            var template = expectation.getSuggestedTemplate();
            var confidence = expectation.getOverallConfidence();

            // Create template-influenced features
            for (int i = 0; i < expectedFeatures.length; i++) {
                expectedFeatures[i] = (float) (currentFeatures[i] * (1.0 - confidence) +
                                             template.baseConfidence() * confidence);
            }
        } else {
            // Use prediction-based expectations
            var topPredictionConf = expectation.getTopPredictionConfidence();
            for (int i = 0; i < expectedFeatures.length; i++) {
                expectedFeatures[i] = (float) (currentFeatures[i] * (1.0 - topPredictionConf * 0.5));
            }
        }

        return expectedFeatures;
    }

    /**
     * Checks resonance between bottom-up and top-down features.
     */
    private ResonanceDetector.ResonanceResult checkResonance(float[] bottomUp, float[] topDown, Template template) {
        return resonanceDetector.detectResonance(bottomUp, topDown, template);
    }

    /**
     * Adapts system parameters based on current performance.
     */
    private AdaptationController.AdaptationResult adaptParameters(ResonanceDetector.ResonanceResult resonance,
                                                                Template template, int iteration) {
        var context = new AdaptationController.AdaptationContext(
            resonance.isResonant(),
            resonance.getStrength(),
            template,
            resonance.isResonant() ? 0 : 1, // Simple failure counting
            resonance.isResonant() ? 1 : 0, // Simple success counting
            resonance.isResonant() ? 0 : System.currentTimeMillis() // Time since success
        );

        return adaptationController.adapt(context);
    }

    /**
     * Applies damping to prevent feature oscillations.
     */
    private float[] applyDamping(float[] current, float[] expected) {
        if (current == null || expected == null || current.length != expected.length) {
            return current;
        }

        var damped = new float[current.length];
        for (int i = 0; i < current.length; i++) {
            damped[i] = (float) (current[i] * (1.0 - dampingFactor) + expected[i] * dampingFactor);
        }

        return damped;
    }

    /**
     * Calculates overall convergence score based on current state.
     */
    private double calculateConvergenceScore(LoopState state) {
        if (state.getCurrentResonance() == null) {
            return 0.0;
        }

        var resonanceScore = state.getCurrentResonance().getStrength();
        var confidenceScore = state.getCurrentExpectation() != null ?
            state.getCurrentExpectation().getOverallConfidence() : 0.0;

        return (resonanceScore * 0.7) + (confidenceScore * 0.3);
    }

    /**
     * Checks if the loop state is stable (not oscillating).
     */
    private boolean checkStability(LoopState state) {
        // Simple stability check - in practice, would track feature changes over time
        return state.getCurrentResonance() != null &&
               state.getCurrentResonance().isResonant() &&
               state.getConvergenceScore() > convergenceThreshold;
    }

    /**
     * Checks if the feedback loop has converged.
     */
    private boolean hasConverged(LoopState state) {
        return state.isStable() && state.getConvergenceScore() >= convergenceThreshold;
    }

    /**
     * Checks if the loop should terminate due to limits.
     */
    private boolean shouldTerminate(LoopState state, long startTime) {
        var timeElapsed = System.currentTimeMillis() - startTime;
        return state.getIterationCount() >= maxIterations || timeElapsed >= maxExecutionTimeMs;
    }

    /**
     * Creates a successful feedback loop result.
     */
    private FeedbackLoopResult createSuccessResult(LoopState state, long executionTime, String supervisedLabel) {
        var converged = hasConverged(state);
        var prediction = determinePrediction(state, supervisedLabel);
        var confidence = state.getConvergenceScore();
        var template = state.getCurrentExpectation() != null ?
            state.getCurrentExpectation().getSuggestedTemplate() : null;
        var resonanceStrength = state.getCurrentResonance() != null ?
            state.getCurrentResonance().getStrength() : 0.0;

        var reason = converged ? "Convergence achieved" :
                    state.getIterationCount() >= maxIterations ? "Max iterations reached" : "Timeout reached";

        return new FeedbackLoopResult(
            converged,
            state.getIterationCount(),
            state.getConvergenceScore(),
            resonanceStrength,
            template,
            state.getCurrentFeatures(),
            prediction,
            confidence,
            executionTime,
            reason
        );
    }

    /**
     * Creates a failure result.
     */
    private FeedbackLoopResult createFailureResult(String reason, int cycles, long executionTime) {
        return new FeedbackLoopResult(
            false, cycles, 0.0, 0.0, null, null, "FAILED", 0.0, executionTime, reason
        );
    }

    /**
     * Determines the prediction based on final state.
     */
    private String determinePrediction(LoopState state, String supervisedLabel) {
        if (supervisedLabel != null) {
            return supervisedLabel; // Return supervised label if provided
        }

        if (state.getCurrentExpectation() != null && state.getCurrentExpectation().hasPredictions()) {
            return state.getCurrentExpectation().getTopPrediction();
        }

        if (state.getCurrentResonance() != null && state.getCurrentResonance().isResonant()) {
            return "RESONANCE_ACHIEVED";
        }

        return "NO_PREDICTION";
    }

    /**
     * Closes the feedback loop and releases resources.
     */
    @Override
    public void close() {
        logger.info("Closing feedback loop");
        reset();
        logger.info("Feedback loop closed");
    }

    /**
     * Statistics about feedback loop convergence performance.
     */
    public static class ConvergenceStats {
        private final int totalRuns;
        private final int convergedRuns;
        private final double convergenceRate;
        private final double averageExecutionTimeMs;
        private final double averageConvergenceTimeMs;
        private final int maxIterationsReached;

        public ConvergenceStats(int totalRuns, int convergedRuns, double convergenceRate,
                              double averageExecutionTimeMs, double averageConvergenceTimeMs,
                              int maxIterationsReached) {
            this.totalRuns = totalRuns;
            this.convergedRuns = convergedRuns;
            this.convergenceRate = convergenceRate;
            this.averageExecutionTimeMs = averageExecutionTimeMs;
            this.averageConvergenceTimeMs = averageConvergenceTimeMs;
            this.maxIterationsReached = maxIterationsReached;
        }

        public int getTotalRuns() { return totalRuns; }
        public int getConvergedRuns() { return convergedRuns; }
        public double getConvergenceRate() { return convergenceRate; }
        public double getAverageExecutionTimeMs() { return averageExecutionTimeMs; }
        public double getAverageConvergenceTimeMs() { return averageConvergenceTimeMs; }
        public int getMaxIterationsReached() { return maxIterationsReached; }

        @Override
        public String toString() {
            return "ConvergenceStats{runs=%d, convergenceRate=%.2f%%, avgExecTime=%.2fms, avgConvTime=%.2fms, maxIterReached=%d}"
                .formatted(totalRuns, convergenceRate * 100, averageExecutionTimeMs, averageConvergenceTimeMs, maxIterationsReached);
        }
    }
}