package com.hellblazer.art.hartcq.feedback;

import com.hellblazer.art.hartcq.HARTCQConfig;
import com.hellblazer.art.hartcq.templates.Template;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Controls learning and adaptation in the HART-CQ feedback system.
 * Manages vigilance parameter adaptation, learning rate control, and system stability
 * to ensure optimal performance while maintaining deterministic behavior.
 *
 * The adaptation controller implements multiple adaptation strategies including
 * performance-based vigilance adjustment and stability monitoring.
 *
 * @author Claude Code
 */
public class AdaptationController implements AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(AdaptationController.class);

    private final HARTCQConfig config;

    // Learning control parameters
    private final AtomicReference<Double> currentVigilance;
    private final AtomicReference<Double> currentLearningRate;
    private final AtomicBoolean learningEnabled;
    private final AtomicBoolean adaptationEnabled;

    // Adaptation parameters
    private final double minVigilance;
    private final double maxVigilance;
    private final double vigilanceIncrement;
    private final double vigilanceDecrement;
    private final double minLearningRate;
    private final double maxLearningRate;

    // Performance monitoring
    private final AtomicInteger successfulAdaptations;
    private final AtomicInteger failedAdaptations;
    private final AtomicLong lastAdaptationTime;
    private final AtomicReference<AdaptationStrategy> currentStrategy;

    // Stability monitoring
    private final AtomicInteger consecutiveFailures;
    private final AtomicInteger consecutiveSuccesses;
    private final int maxConsecutiveFailures;
    private final long stabilityWindowMs;

    /**
     * Adaptation strategies for different scenarios.
     */
    public enum AdaptationStrategy {
        CONSERVATIVE("Low learning rate, high vigilance for stability"),
        BALANCED("Moderate learning rate and vigilance"),
        AGGRESSIVE("High learning rate, moderate vigilance for fast adaptation"),
        EXPLORATORY("Variable parameters to explore solution space"),
        MAINTENANCE("Minimal adaptation to maintain learned patterns");

        private final String description;

        AdaptationStrategy(String description) {
            this.description = description;
        }

        public String getDescription() { return description; }
    }

    /**
     * Result of adaptation process.
     */
    public static class AdaptationResult {
        private final boolean adapted;
        private final double newVigilance;
        private final double newLearningRate;
        private final AdaptationStrategy strategy;
        private final String reason;
        private final long adaptationTimeMs;

        public AdaptationResult(boolean adapted, double newVigilance, double newLearningRate,
                              AdaptationStrategy strategy, String reason, long adaptationTimeMs) {
            this.adapted = adapted;
            this.newVigilance = newVigilance;
            this.newLearningRate = newLearningRate;
            this.strategy = strategy;
            this.reason = reason;
            this.adaptationTimeMs = adaptationTimeMs;
        }

        public boolean isAdapted() { return adapted; }
        public double getNewVigilance() { return newVigilance; }
        public double getNewLearningRate() { return newLearningRate; }
        public AdaptationStrategy getStrategy() { return strategy; }
        public String getReason() { return reason; }
        public long getAdaptationTimeMs() { return adaptationTimeMs; }

        @Override
        public String toString() {
            return "AdaptationResult{adapted=%s, vigilance=%.3f, learningRate=%.3f, strategy=%s}"
                .formatted(adapted, newVigilance, newLearningRate, strategy);
        }
    }

    /**
     * Context for adaptation decisions.
     */
    public static class AdaptationContext {
        private final boolean resonanceAchieved;
        private final double resonanceStrength;
        private final Template currentTemplate;
        private final int recentFailures;
        private final int recentSuccesses;
        private final long timeSinceLastSuccess;

        public AdaptationContext(boolean resonanceAchieved, double resonanceStrength,
                               Template currentTemplate, int recentFailures, int recentSuccesses,
                               long timeSinceLastSuccess) {
            this.resonanceAchieved = resonanceAchieved;
            this.resonanceStrength = Math.max(0.0, Math.min(1.0, resonanceStrength));
            this.currentTemplate = currentTemplate;
            this.recentFailures = Math.max(0, recentFailures);
            this.recentSuccesses = Math.max(0, recentSuccesses);
            this.timeSinceLastSuccess = timeSinceLastSuccess;
        }

        public boolean isResonanceAchieved() { return resonanceAchieved; }
        public double getResonanceStrength() { return resonanceStrength; }
        public Template getCurrentTemplate() { return currentTemplate; }
        public int getRecentFailures() { return recentFailures; }
        public int getRecentSuccesses() { return recentSuccesses; }
        public long getTimeSinceLastSuccess() { return timeSinceLastSuccess; }

        public double getSuccessRate() {
            var total = recentFailures + recentSuccesses;
            return total > 0 ? (double) recentSuccesses / total : 0.0;
        }

        @Override
        public String toString() {
            return "AdaptationContext{resonance=%s, strength=%.3f, failures=%d, successes=%d, successRate=%.2f%%}"
                .formatted(resonanceAchieved, resonanceStrength, recentFailures, recentSuccesses, getSuccessRate() * 100);
        }
    }

    /**
     * Creates an adaptation controller with the given configuration.
     *
     * @param config HART-CQ configuration
     */
    public AdaptationController(HARTCQConfig config) {
        this.config = Objects.requireNonNull(config, "Configuration cannot be null");

        var templateConfig = config.getTemplateConfig();

        // Initialize parameters from configuration
        this.currentVigilance = new AtomicReference<>(templateConfig.getVigilanceParameter());
        this.currentLearningRate = new AtomicReference<>(templateConfig.getLearningRate());
        this.learningEnabled = new AtomicBoolean(true);
        this.adaptationEnabled = new AtomicBoolean(true);

        // Set adaptation bounds
        this.minVigilance = Math.max(0.1, templateConfig.getVigilanceParameter() * 0.5);
        this.maxVigilance = Math.min(1.0, templateConfig.getVigilanceParameter() * 1.5);
        this.vigilanceIncrement = 0.05;
        this.vigilanceDecrement = 0.02;
        this.minLearningRate = Math.max(0.001, templateConfig.getLearningRate() * 0.1);
        this.maxLearningRate = Math.min(0.5, templateConfig.getLearningRate() * 3.0);

        // Initialize performance monitoring
        this.successfulAdaptations = new AtomicInteger(0);
        this.failedAdaptations = new AtomicInteger(0);
        this.lastAdaptationTime = new AtomicLong(0);
        this.currentStrategy = new AtomicReference<>(AdaptationStrategy.BALANCED);

        // Initialize stability monitoring
        this.consecutiveFailures = new AtomicInteger(0);
        this.consecutiveSuccesses = new AtomicInteger(0);
        this.maxConsecutiveFailures = 5;
        this.stabilityWindowMs = 30000; // 30 seconds

        logger.info("AdaptationController initialized: vigilance=[{}, {}], learningRate=[{}, {}], strategy={}",
                   minVigilance, maxVigilance, minLearningRate, maxLearningRate, currentStrategy.get());
    }

    /**
     * Adapts system parameters based on feedback context.
     *
     * @param context Adaptation context with performance information
     * @return Adaptation result
     */
    public AdaptationResult adapt(AdaptationContext context) {
        if (!adaptationEnabled.get() || !learningEnabled.get()) {
            return createNoAdaptationResult("Adaptation or learning disabled");
        }

        if (context == null) {
            return createNoAdaptationResult("No adaptation context provided");
        }

        var startTime = System.currentTimeMillis();
        try {
            logger.debug("Adapting system parameters based on context: {}", context);

            // Update performance counters
            updatePerformanceCounters(context);

            // Determine adaptation strategy
            var strategy = selectAdaptationStrategy(context);

            // Calculate new parameters
            var newVigilance = calculateNewVigilance(context, strategy);
            var newLearningRate = calculateNewLearningRate(context, strategy);

            // Check if adaptation is needed
            var currentVig = currentVigilance.get();
            var currentLR = currentLearningRate.get();

            var significantVigilanceChange = Math.abs(newVigilance - currentVig) > 0.01;
            var significantLearningRateChange = Math.abs(newLearningRate - currentLR) > 0.001;

            if (!significantVigilanceChange && !significantLearningRateChange && strategy == currentStrategy.get()) {
                return createNoAdaptationResult("No significant parameter changes needed");
            }

            // Apply adaptation
            currentVigilance.set(newVigilance);
            currentLearningRate.set(newLearningRate);
            currentStrategy.set(strategy);
            lastAdaptationTime.set(startTime);

            successfulAdaptations.incrementAndGet();

            var result = new AdaptationResult(
                true,
                newVigilance,
                newLearningRate,
                strategy,
                "Parameters adapted based on performance: " + context.toString(),
                System.currentTimeMillis() - startTime
            );

            logger.info("Adaptation completed: {}", result);
            return result;

        } catch (Exception e) {
            logger.error("Error during adaptation", e);
            failedAdaptations.incrementAndGet();
            return createNoAdaptationResult("Adaptation failed: " + e.getMessage());
        }
    }

    /**
     * Enables or disables learning mode.
     *
     * @param enabled true to enable learning
     */
    public void setLearningEnabled(boolean enabled) {
        boolean wasEnabled = learningEnabled.getAndSet(enabled);
        if (wasEnabled != enabled) {
            logger.info("Learning mode changed: {} -> {}", wasEnabled, enabled);
            if (!enabled) {
                // Reset to conservative strategy when learning is disabled
                currentStrategy.set(AdaptationStrategy.MAINTENANCE);
            }
        }
    }

    /**
     * Enables or disables parameter adaptation.
     *
     * @param enabled true to enable adaptation
     */
    public void setAdaptationEnabled(boolean enabled) {
        boolean wasEnabled = adaptationEnabled.getAndSet(enabled);
        if (wasEnabled != enabled) {
            logger.info("Adaptation mode changed: {} -> {}", wasEnabled, enabled);
        }
    }

    /**
     * Gets current vigilance parameter.
     *
     * @return current vigilance value
     */
    public double getCurrentVigilance() {
        return currentVigilance.get();
    }

    /**
     * Gets current learning rate.
     *
     * @return current learning rate value
     */
    public double getCurrentLearningRate() {
        return currentLearningRate.get();
    }

    /**
     * Gets current adaptation strategy.
     *
     * @return current strategy
     */
    public AdaptationStrategy getCurrentStrategy() {
        return currentStrategy.get();
    }

    /**
     * Checks if learning is enabled.
     *
     * @return true if learning is enabled
     */
    public boolean isLearningEnabled() {
        return learningEnabled.get();
    }

    /**
     * Checks if adaptation is enabled.
     *
     * @return true if adaptation is enabled
     */
    public boolean isAdaptationEnabled() {
        return adaptationEnabled.get();
    }

    /**
     * Checks if system is stable (low failure rate).
     *
     * @return true if system appears stable
     */
    public boolean isStable() {
        return consecutiveFailures.get() < maxConsecutiveFailures / 2;
    }

    /**
     * Gets adaptation statistics.
     *
     * @return adaptation statistics
     */
    public AdaptationStats getAdaptationStats() {
        var totalAdaptations = successfulAdaptations.get() + failedAdaptations.get();
        var successRate = totalAdaptations > 0 ?
            (double) successfulAdaptations.get() / totalAdaptations : 0.0;

        return new AdaptationStats(
            totalAdaptations,
            successfulAdaptations.get(),
            successRate,
            currentVigilance.get(),
            currentLearningRate.get(),
            currentStrategy.get(),
            consecutiveFailures.get(),
            consecutiveSuccesses.get(),
            isStable()
        );
    }

    /**
     * Resets the adaptation controller to initial state.
     */
    public void reset() {
        logger.info("Resetting adaptation controller");

        var templateConfig = config.getTemplateConfig();
        currentVigilance.set(templateConfig.getVigilanceParameter());
        currentLearningRate.set(templateConfig.getLearningRate());
        currentStrategy.set(AdaptationStrategy.BALANCED);

        successfulAdaptations.set(0);
        failedAdaptations.set(0);
        lastAdaptationTime.set(0);

        consecutiveFailures.set(0);
        consecutiveSuccesses.set(0);

        logger.info("Adaptation controller reset completed");
    }

    /**
     * Updates performance counters based on context.
     */
    private void updatePerformanceCounters(AdaptationContext context) {
        if (context.isResonanceAchieved()) {
            consecutiveSuccesses.incrementAndGet();
            consecutiveFailures.set(0);
        } else {
            consecutiveFailures.incrementAndGet();
            consecutiveSuccesses.set(0);
        }
    }

    /**
     * Selects appropriate adaptation strategy based on context.
     */
    private AdaptationStrategy selectAdaptationStrategy(AdaptationContext context) {
        var successRate = context.getSuccessRate();
        var consecutiveFailureCount = consecutiveFailures.get();
        var timeSinceLastSuccess = context.getTimeSinceLastSuccess();

        // Emergency strategy for high failure rate
        if (consecutiveFailureCount >= maxConsecutiveFailures) {
            logger.debug("Switching to CONSERVATIVE strategy due to {} consecutive failures", consecutiveFailureCount);
            return AdaptationStrategy.CONSERVATIVE;
        }

        // Maintenance strategy if learning is disabled
        if (!learningEnabled.get()) {
            return AdaptationStrategy.MAINTENANCE;
        }

        // Exploratory strategy if stuck (low success rate but not total failure)
        if (successRate < 0.3 && timeSinceLastSuccess > stabilityWindowMs) {
            logger.debug("Switching to EXPLORATORY strategy due to low success rate: {}", successRate);
            return AdaptationStrategy.EXPLORATORY;
        }

        // Aggressive strategy for good performance but room for improvement
        if (successRate > 0.7 && context.getResonanceStrength() > 0.8) {
            logger.debug("Switching to AGGRESSIVE strategy due to good performance");
            return AdaptationStrategy.AGGRESSIVE;
        }

        // Conservative strategy for moderate performance
        if (successRate > 0.5) {
            return AdaptationStrategy.CONSERVATIVE;
        }

        // Default to balanced approach
        return AdaptationStrategy.BALANCED;
    }

    /**
     * Calculates new vigilance parameter based on context and strategy.
     */
    private double calculateNewVigilance(AdaptationContext context, AdaptationStrategy strategy) {
        var currentVig = currentVigilance.get();
        var newVigilance = currentVig;

        switch (strategy) {
            case CONSERVATIVE:
                // Increase vigilance for more selective matching
                if (context.getRecentFailures() > context.getRecentSuccesses()) {
                    newVigilance = Math.min(maxVigilance, currentVig + vigilanceIncrement);
                }
                break;

            case AGGRESSIVE:
                // Decrease vigilance for broader matching
                if (context.isResonanceAchieved() && context.getResonanceStrength() > 0.8) {
                    newVigilance = Math.max(minVigilance, currentVig - vigilanceDecrement);
                }
                break;

            case EXPLORATORY:
                // Vary vigilance based on recent performance
                if (context.getSuccessRate() < 0.3) {
                    newVigilance = context.getRecentFailures() > context.getRecentSuccesses() ?
                        Math.min(maxVigilance, currentVig + vigilanceIncrement * 1.5) :
                        Math.max(minVigilance, currentVig - vigilanceDecrement * 1.5);
                }
                break;

            case BALANCED:
                // Small adjustments based on resonance strength
                if (context.isResonanceAchieved()) {
                    if (context.getResonanceStrength() < 0.6) {
                        newVigilance = Math.max(minVigilance, currentVig - vigilanceDecrement * 0.5);
                    }
                } else {
                    newVigilance = Math.min(maxVigilance, currentVig + vigilanceIncrement * 0.5);
                }
                break;

            case MAINTENANCE:
                // No vigilance changes in maintenance mode
                break;
        }

        return Math.max(minVigilance, Math.min(maxVigilance, newVigilance));
    }

    /**
     * Calculates new learning rate based on context and strategy.
     */
    private double calculateNewLearningRate(AdaptationContext context, AdaptationStrategy strategy) {
        var currentLR = currentLearningRate.get();
        var newLearningRate = currentLR;

        switch (strategy) {
            case CONSERVATIVE:
                // Lower learning rate for stability
                newLearningRate = Math.max(minLearningRate, currentLR * 0.95);
                break;

            case AGGRESSIVE:
                // Higher learning rate for fast adaptation
                newLearningRate = Math.min(maxLearningRate, currentLR * 1.05);
                break;

            case EXPLORATORY:
                // Variable learning rate based on performance
                if (context.getSuccessRate() < 0.3) {
                    newLearningRate = Math.min(maxLearningRate, currentLR * 1.1);
                }
                break;

            case BALANCED:
                // Small adjustments based on success rate
                if (context.getSuccessRate() > 0.7) {
                    newLearningRate = Math.min(maxLearningRate, currentLR * 1.02);
                } else if (context.getSuccessRate() < 0.4) {
                    newLearningRate = Math.max(minLearningRate, currentLR * 0.98);
                }
                break;

            case MAINTENANCE:
                // Minimal learning rate in maintenance mode
                newLearningRate = minLearningRate;
                break;
        }

        return Math.max(minLearningRate, Math.min(maxLearningRate, newLearningRate));
    }

    /**
     * Creates a no-adaptation result.
     */
    private AdaptationResult createNoAdaptationResult(String reason) {
        return new AdaptationResult(
            false,
            currentVigilance.get(),
            currentLearningRate.get(),
            currentStrategy.get(),
            reason,
            0
        );
    }

    /**
     * Closes the adaptation controller and releases resources.
     */
    @Override
    public void close() {
        logger.info("Closing adaptation controller");
        setAdaptationEnabled(false);
        setLearningEnabled(false);
        logger.info("Adaptation controller closed");
    }

    /**
     * Statistics about adaptation performance.
     */
    public static class AdaptationStats {
        private final int totalAdaptations;
        private final int successfulAdaptations;
        private final double successRate;
        private final double currentVigilance;
        private final double currentLearningRate;
        private final AdaptationStrategy currentStrategy;
        private final int consecutiveFailures;
        private final int consecutiveSuccesses;
        private final boolean isStable;

        public AdaptationStats(int totalAdaptations, int successfulAdaptations, double successRate,
                             double currentVigilance, double currentLearningRate, AdaptationStrategy currentStrategy,
                             int consecutiveFailures, int consecutiveSuccesses, boolean isStable) {
            this.totalAdaptations = totalAdaptations;
            this.successfulAdaptations = successfulAdaptations;
            this.successRate = successRate;
            this.currentVigilance = currentVigilance;
            this.currentLearningRate = currentLearningRate;
            this.currentStrategy = currentStrategy;
            this.consecutiveFailures = consecutiveFailures;
            this.consecutiveSuccesses = consecutiveSuccesses;
            this.isStable = isStable;
        }

        public int getTotalAdaptations() { return totalAdaptations; }
        public int getSuccessfulAdaptations() { return successfulAdaptations; }
        public double getSuccessRate() { return successRate; }
        public double getCurrentVigilance() { return currentVigilance; }
        public double getCurrentLearningRate() { return currentLearningRate; }
        public AdaptationStrategy getCurrentStrategy() { return currentStrategy; }
        public int getConsecutiveFailures() { return consecutiveFailures; }
        public int getConsecutiveSuccesses() { return consecutiveSuccesses; }
        public boolean isStable() { return isStable; }

        @Override
        public String toString() {
            return "AdaptationStats{adaptations=%d, successRate=%.2f%%, vigilance=%.3f, learningRate=%.4f, strategy=%s, stable=%s}"
                .formatted(totalAdaptations, successRate * 100, currentVigilance, currentLearningRate, currentStrategy, isStable);
        }
    }
}