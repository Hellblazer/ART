package com.hellblazer.art.hartcq.feedback;

import com.hellblazer.art.hartcq.HARTCQConfig;
import com.hellblazer.art.hartcq.Token;
import com.hellblazer.art.hartcq.templates.Template;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Main feedback coordination controller for HART-CQ system.
 * Orchestrates top-down expectations with bottom-up data signals to achieve
 * resonance and prevent hallucination through template-based constraints.
 *
 * The feedback controller maintains deterministic behavior when not learning
 * and supports both supervised and unsupervised operational modes.
 *
 * @author Claude Code
 */
public class FeedbackController implements AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(FeedbackController.class);

    private final HARTCQConfig config;
    private final ExpectationManager expectationManager;
    private final ResonanceDetector resonanceDetector;
    private final AdaptationController adaptationController;
    private final FeedbackLoop feedbackLoop;
    private final Executor executor;

    // State management
    private final AtomicBoolean isLearning;
    private final AtomicBoolean isSupervised;
    private final AtomicInteger cycleCounter;
    private final AtomicReference<FeedbackState> currentState;

    /**
     * Represents the current state of the feedback system.
     */
    public enum FeedbackState {
        INITIALIZING,
        WAITING_FOR_INPUT,
        PROCESSING_BOTTOM_UP,
        GENERATING_EXPECTATIONS,
        CHECKING_RESONANCE,
        ADAPTING,
        STABILIZED,
        ERROR
    }

    /**
     * Result of feedback processing cycle.
     */
    public static class FeedbackResult {
        private final boolean resonanceAchieved;
        private final double resonanceStrength;
        private final Template matchedTemplate;
        private final float[] processedFeatures;
        private final String prediction;
        private final double confidence;
        private final int cyclesRequired;

        public FeedbackResult(boolean resonanceAchieved, double resonanceStrength,
                            Template matchedTemplate, float[] processedFeatures,
                            String prediction, double confidence, int cyclesRequired) {
            this.resonanceAchieved = resonanceAchieved;
            this.resonanceStrength = resonanceStrength;
            this.matchedTemplate = matchedTemplate;
            this.processedFeatures = processedFeatures != null ? processedFeatures.clone() : null;
            this.prediction = prediction;
            this.confidence = confidence;
            this.cyclesRequired = cyclesRequired;
        }

        public boolean isResonanceAchieved() { return resonanceAchieved; }
        public double getResonanceStrength() { return resonanceStrength; }
        public Template getMatchedTemplate() { return matchedTemplate; }
        public float[] getProcessedFeatures() {
            return processedFeatures != null ? processedFeatures.clone() : null;
        }
        public String getPrediction() { return prediction; }
        public double getConfidence() { return confidence; }
        public int getCyclesRequired() { return cyclesRequired; }

        @Override
        public String toString() {
            return "FeedbackResult{resonance=%s, strength=%.3f, prediction='%s', confidence=%.3f, cycles=%d}"
                .formatted(resonanceAchieved, resonanceStrength, prediction, confidence, cyclesRequired);
        }
    }

    /**
     * Creates a feedback controller with the given configuration.
     *
     * @param config HART-CQ configuration
     * @param executor Executor for asynchronous processing
     */
    public FeedbackController(HARTCQConfig config, Executor executor) {
        this.config = Objects.requireNonNull(config, "Configuration cannot be null");
        this.executor = Objects.requireNonNull(executor, "Executor cannot be null");

        // Initialize components
        this.expectationManager = new ExpectationManager(config);
        this.resonanceDetector = new ResonanceDetector(config);
        this.adaptationController = new AdaptationController(config);
        this.feedbackLoop = new FeedbackLoop(config, expectationManager, resonanceDetector, adaptationController);

        // Initialize state
        this.isLearning = new AtomicBoolean(false);
        this.isSupervised = new AtomicBoolean(false);
        this.cycleCounter = new AtomicInteger(0);
        this.currentState = new AtomicReference<>(FeedbackState.INITIALIZING);

        logger.info("FeedbackController initialized with vigilance={}",
                   config.getTemplateConfig().getVigilanceParameter());

        // Set state to ready
        this.currentState.set(FeedbackState.WAITING_FOR_INPUT);
    }

    /**
     * Processes input tokens through the feedback system synchronously.
     *
     * @param tokens Input token window
     * @return Feedback processing result
     */
    public FeedbackResult process(Token[] tokens) {
        return processInternal(tokens, null, false).join();
    }

    /**
     * Processes input tokens with supervised learning.
     *
     * @param tokens Input token window
     * @param supervisedLabel Expected category label
     * @return Feedback processing result
     */
    public FeedbackResult processSupervised(Token[] tokens, String supervisedLabel) {
        return processInternal(tokens, supervisedLabel, true).join();
    }

    /**
     * Processes input tokens asynchronously.
     *
     * @param tokens Input token window
     * @return CompletableFuture with feedback result
     */
    public CompletableFuture<FeedbackResult> processAsync(Token[] tokens) {
        return processInternal(tokens, null, false);
    }

    /**
     * Processes input tokens with supervised learning asynchronously.
     *
     * @param tokens Input token window
     * @param supervisedLabel Expected category label
     * @return CompletableFuture with feedback result
     */
    public CompletableFuture<FeedbackResult> processAsyncSupervised(Token[] tokens, String supervisedLabel) {
        return processInternal(tokens, supervisedLabel, true);
    }

    /**
     * Internal processing method that handles both sync and async processing.
     */
    private CompletableFuture<FeedbackResult> processInternal(Token[] tokens, String supervisedLabel, boolean supervised) {
        if (tokens == null || tokens.length == 0) {
            return CompletableFuture.completedFuture(
                createErrorResult("Input tokens cannot be null or empty"));
        }

        // Update operation mode
        this.isSupervised.set(supervised);

        return CompletableFuture.supplyAsync(() -> {
            try {
                var cycle = cycleCounter.incrementAndGet();
                logger.debug("Starting feedback cycle {} for {} tokens", cycle, tokens.length);

                // Update state
                currentState.set(FeedbackState.PROCESSING_BOTTOM_UP);

                // Run feedback loop
                var loopResult = feedbackLoop.run(tokens, supervisedLabel);

                // Create result
                var result = new FeedbackResult(
                    loopResult.isConverged(),
                    loopResult.getResonanceStrength(),
                    loopResult.getMatchedTemplate(),
                    loopResult.getProcessedFeatures(),
                    loopResult.getPrediction(),
                    loopResult.getConfidence(),
                    loopResult.getCycles()
                );

                // Update state based on result
                if (result.isResonanceAchieved()) {
                    currentState.set(FeedbackState.STABILIZED);
                } else {
                    currentState.set(FeedbackState.WAITING_FOR_INPUT);
                }

                logger.debug("Completed feedback cycle {}: {}", cycle, result);
                return result;

            } catch (Exception e) {
                logger.error("Error in feedback processing cycle", e);
                currentState.set(FeedbackState.ERROR);
                return createErrorResult("Processing error: " + e.getMessage());
            }
        }, executor);
    }

    /**
     * Enables or disables learning mode.
     *
     * @param learningEnabled true to enable learning
     */
    public void setLearningEnabled(boolean learningEnabled) {
        boolean wasLearning = this.isLearning.getAndSet(learningEnabled);
        if (wasLearning != learningEnabled) {
            logger.info("Learning mode changed: {} -> {}", wasLearning, learningEnabled);
            adaptationController.setLearningEnabled(learningEnabled);
        }
    }

    /**
     * Checks if learning mode is enabled.
     *
     * @return true if learning is enabled
     */
    public boolean isLearningEnabled() {
        return isLearning.get();
    }

    /**
     * Checks if system is in supervised mode.
     *
     * @return true if supervised mode is active
     */
    public boolean isSupervised() {
        return isSupervised.get();
    }

    /**
     * Gets the current feedback state.
     *
     * @return current state
     */
    public FeedbackState getCurrentState() {
        return currentState.get();
    }

    /**
     * Gets statistics about feedback processing.
     *
     * @return feedback statistics
     */
    public FeedbackStats getStatistics() {
        return new FeedbackStats(
            cycleCounter.get(),
            feedbackLoop.getConvergenceStats(),
            expectationManager.getExpectationStats(),
            resonanceDetector.getResonanceStats(),
            adaptationController.getAdaptationStats()
        );
    }

    /**
     * Resets the feedback controller to initial state.
     */
    public void reset() {
        logger.info("Resetting feedback controller");

        currentState.set(FeedbackState.INITIALIZING);
        cycleCounter.set(0);
        isLearning.set(false);
        isSupervised.set(false);

        // Reset components
        expectationManager.reset();
        resonanceDetector.reset();
        adaptationController.reset();
        feedbackLoop.reset();

        currentState.set(FeedbackState.WAITING_FOR_INPUT);
        logger.info("Feedback controller reset completed");
    }

    /**
     * Creates an error result.
     */
    private FeedbackResult createErrorResult(String message) {
        return new FeedbackResult(false, 0.0, null, null, "ERROR", 0.0, 0);
    }

    /**
     * Closes the feedback controller and releases resources.
     */
    @Override
    public void close() {
        logger.info("Closing feedback controller");
        currentState.set(FeedbackState.INITIALIZING);

        // Close components
        try {
            expectationManager.close();
        } catch (Exception e) {
            logger.warn("Error closing expectation manager", e);
        }

        try {
            adaptationController.close();
        } catch (Exception e) {
            logger.warn("Error closing adaptation controller", e);
        }

        try {
            feedbackLoop.close();
        } catch (Exception e) {
            logger.warn("Error closing feedback loop", e);
        }

        logger.info("Feedback controller closed");
    }

    /**
     * Statistics about feedback processing performance.
     */
    public static class FeedbackStats {
        private final int totalCycles;
        private final FeedbackLoop.ConvergenceStats convergenceStats;
        private final ExpectationManager.ExpectationStats expectationStats;
        private final ResonanceDetector.ResonanceStats resonanceStats;
        private final AdaptationController.AdaptationStats adaptationStats;

        public FeedbackStats(int totalCycles,
                           FeedbackLoop.ConvergenceStats convergenceStats,
                           ExpectationManager.ExpectationStats expectationStats,
                           ResonanceDetector.ResonanceStats resonanceStats,
                           AdaptationController.AdaptationStats adaptationStats) {
            this.totalCycles = totalCycles;
            this.convergenceStats = convergenceStats;
            this.expectationStats = expectationStats;
            this.resonanceStats = resonanceStats;
            this.adaptationStats = adaptationStats;
        }

        public int getTotalCycles() { return totalCycles; }
        public FeedbackLoop.ConvergenceStats getConvergenceStats() { return convergenceStats; }
        public ExpectationManager.ExpectationStats getExpectationStats() { return expectationStats; }
        public ResonanceDetector.ResonanceStats getResonanceStats() { return resonanceStats; }
        public AdaptationController.AdaptationStats getAdaptationStats() { return adaptationStats; }

        @Override
        public String toString() {
            return "FeedbackStats{cycles=%d, convergenceRate=%.2f%%, avgResonance=%.3f}"
                .formatted(totalCycles,
                         convergenceStats.getConvergenceRate() * 100,
                         resonanceStats.getAverageResonanceStrength());
        }
    }
}