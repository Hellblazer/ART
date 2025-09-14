package com.hellblazer.art.hartcq.feedback;

import com.hellblazer.art.hartcq.HARTCQConfig;
import com.hellblazer.art.hartcq.templates.Template;
import org.joml.Vector3f;
import org.joml.Vector3fc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.DoubleAdder;

/**
 * Detects resonance conditions in the HART-CQ feedback system.
 * Resonance occurs when top-down expectations match bottom-up processing
 * within acceptable thresholds, indicating stable pattern recognition.
 *
 * The detector uses configurable thresholds and multiple detection methods
 * to ensure robust resonance detection across different input patterns.
 *
 * @author Claude Code
 */
public class ResonanceDetector {
    private static final Logger logger = LoggerFactory.getLogger(ResonanceDetector.class);

    private final HARTCQConfig config;

    // Threshold parameters
    private final double matchThreshold;
    private final double mismatchThreshold;
    private final double vigilanceParameter;
    private final int minStabilityDuration;

    // Detection state
    private volatile double lastResonanceStrength;
    private volatile long lastResonanceTime;
    private volatile boolean isResonant;
    private final AtomicInteger stabilityCounter;

    // Statistics
    private final AtomicInteger totalDetections;
    private final AtomicInteger resonanceDetections;
    private final AtomicLong totalDetectionTime;
    private final DoubleAdder cumulativeResonanceStrength;

    /**
     * Result of resonance detection.
     */
    public static class ResonanceResult {
        private final boolean isResonant;
        private final double strength;
        private final double confidence;
        private final MismatchType mismatchType;
        private final String reason;
        private final long detectionTimeMs;

        public ResonanceResult(boolean isResonant, double strength, double confidence,
                             MismatchType mismatchType, String reason, long detectionTimeMs) {
            this.isResonant = isResonant;
            this.strength = Math.max(0.0, Math.min(1.0, strength));
            this.confidence = Math.max(0.0, Math.min(1.0, confidence));
            this.mismatchType = mismatchType;
            this.reason = reason;
            this.detectionTimeMs = detectionTimeMs;
        }

        public boolean isResonant() { return isResonant; }
        public double getStrength() { return strength; }
        public double getConfidence() { return confidence; }
        public MismatchType getMismatchType() { return mismatchType; }
        public String getReason() { return reason; }
        public long getDetectionTimeMs() { return detectionTimeMs; }

        @Override
        public String toString() {
            return "ResonanceResult{resonant=%s, strength=%.3f, confidence=%.3f, type=%s}"
                .formatted(isResonant, strength, confidence, mismatchType);
        }
    }

    /**
     * Types of mismatch that can occur during resonance detection.
     */
    public enum MismatchType {
        NONE("No mismatch detected"),
        FEATURE_MISMATCH("Feature vectors don't match expectation"),
        TEMPLATE_MISMATCH("Template pattern doesn't match input"),
        CONFIDENCE_MISMATCH("Confidence levels are too low"),
        STABILITY_MISMATCH("Resonance is not stable over time"),
        THRESHOLD_MISMATCH("Values below minimum thresholds");

        private final String description;

        MismatchType(String description) {
            this.description = description;
        }

        public String getDescription() { return description; }
    }

    /**
     * Creates a resonance detector with the given configuration.
     *
     * @param config HART-CQ configuration
     */
    public ResonanceDetector(HARTCQConfig config) {
        this.config = Objects.requireNonNull(config, "Configuration cannot be null");

        // Extract threshold parameters from configuration
        var templateConfig = config.getTemplateConfig();
        this.matchThreshold = templateConfig.getMatchThreshold();
        this.mismatchThreshold = Math.max(0.1, matchThreshold * 0.5); // Mismatch at half match threshold
        this.vigilanceParameter = templateConfig.getVigilanceParameter();
        this.minStabilityDuration = 3; // Require 3 consecutive detections for stability

        // Initialize state
        this.lastResonanceStrength = 0.0;
        this.lastResonanceTime = 0;
        this.isResonant = false;
        this.stabilityCounter = new AtomicInteger(0);

        // Initialize statistics
        this.totalDetections = new AtomicInteger(0);
        this.resonanceDetections = new AtomicInteger(0);
        this.totalDetectionTime = new AtomicLong(0);
        this.cumulativeResonanceStrength = new DoubleAdder();

        logger.info("ResonanceDetector initialized: matchThreshold={}, vigilance={}, minStability={}",
                   matchThreshold, vigilanceParameter, minStabilityDuration);
    }

    /**
     * Detects resonance between bottom-up features and top-down expectations.
     *
     * @param bottomUpFeatures Features from bottom-up processing
     * @param topDownExpectation Expected features from top-down processing
     * @param template Template being evaluated (if any)
     * @return Resonance detection result
     */
    public ResonanceResult detectResonance(float[] bottomUpFeatures, float[] topDownExpectation,
                                         Template template) {
        if (bottomUpFeatures == null || topDownExpectation == null) {
            return createNoResonanceResult(MismatchType.FEATURE_MISMATCH,
                                         "Missing feature vectors", 0);
        }

        var startTime = System.currentTimeMillis();
        try {
            totalDetections.incrementAndGet();

            // Check feature dimension compatibility
            if (bottomUpFeatures.length != topDownExpectation.length) {
                return createNoResonanceResult(MismatchType.FEATURE_MISMATCH,
                    "Feature dimension mismatch: " + bottomUpFeatures.length + " vs " + topDownExpectation.length,
                    System.currentTimeMillis() - startTime);
            }

            // Calculate feature similarity
            var featureSimilarity = calculateFeatureSimilarity(bottomUpFeatures, topDownExpectation);

            // Calculate template match strength (if template provided)
            var templateMatch = template != null ? calculateTemplateMatch(bottomUpFeatures, template) : 0.5;

            // Calculate overall resonance strength
            var resonanceStrength = combineResonanceFactors(featureSimilarity, templateMatch);

            // Determine if resonance threshold is met
            var isResonant = resonanceStrength >= matchThreshold;

            // Check for stability if resonance is detected
            var mismatchType = MismatchType.NONE;
            var reason = "Resonance achieved";
            var confidence = resonanceStrength;

            if (!isResonant) {
                // Determine why resonance wasn't achieved
                if (featureSimilarity < mismatchThreshold) {
                    mismatchType = MismatchType.FEATURE_MISMATCH;
                    reason = "Feature similarity too low: " + String.format("%.3f", featureSimilarity);
                } else if (templateMatch < mismatchThreshold) {
                    mismatchType = MismatchType.TEMPLATE_MISMATCH;
                    reason = "Template match too low: " + String.format("%.3f", templateMatch);
                } else {
                    mismatchType = MismatchType.THRESHOLD_MISMATCH;
                    reason = "Overall strength below threshold: " + String.format("%.3f", resonanceStrength);
                }

                // Reset stability counter on mismatch
                stabilityCounter.set(0);
                confidence = Math.max(0.1, resonanceStrength);

            } else {
                // Check stability for consistent resonance
                var currentStability = stabilityCounter.incrementAndGet();
                if (currentStability < minStabilityDuration) {
                    mismatchType = MismatchType.STABILITY_MISMATCH;
                    reason = "Resonance detected but not yet stable: " + currentStability + "/" + minStabilityDuration;
                    isResonant = false;
                    confidence = resonanceStrength * (currentStability / (double) minStabilityDuration);
                } else {
                    reason = "Stable resonance achieved (stability=" + currentStability + ")";
                    confidence = Math.min(1.0, resonanceStrength + (currentStability * 0.05));
                }
            }

            // Update state
            updateDetectionState(isResonant, resonanceStrength);

            var result = new ResonanceResult(
                isResonant,
                resonanceStrength,
                confidence,
                mismatchType,
                reason,
                System.currentTimeMillis() - startTime
            );

            if (isResonant) {
                resonanceDetections.incrementAndGet();
                logger.debug("Resonance detected: {}", result);
            } else {
                logger.debug("No resonance: {}", result);
            }

            return result;

        } finally {
            totalDetectionTime.addAndGet(System.currentTimeMillis() - startTime);
        }
    }

    /**
     * Detects resonance with confidence threshold checking.
     *
     * @param bottomUpFeatures Features from bottom-up processing
     * @param topDownExpectation Expected features from top-down processing
     * @param template Template being evaluated
     * @param minConfidence Minimum confidence required for resonance
     * @return Resonance detection result
     */
    public ResonanceResult detectResonanceWithConfidence(float[] bottomUpFeatures, float[] topDownExpectation,
                                                       Template template, double minConfidence) {
        var result = detectResonance(bottomUpFeatures, topDownExpectation, template);

        // Additional confidence check
        if (result.isResonant() && result.getConfidence() < minConfidence) {
            return new ResonanceResult(
                false,
                result.getStrength(),
                result.getConfidence(),
                MismatchType.CONFIDENCE_MISMATCH,
                "Confidence below threshold: " + String.format("%.3f < %.3f", result.getConfidence(), minConfidence),
                result.getDetectionTimeMs()
            );
        }

        return result;
    }

    /**
     * Checks if system is currently in resonance state.
     *
     * @return true if system is resonant
     */
    public boolean isInResonance() {
        return isResonant;
    }

    /**
     * Gets the last measured resonance strength.
     *
     * @return resonance strength (0.0 to 1.0)
     */
    public double getLastResonanceStrength() {
        return lastResonanceStrength;
    }

    /**
     * Gets time since last resonance detection.
     *
     * @return milliseconds since last resonance
     */
    public long getTimeSinceLastResonance() {
        return lastResonanceTime > 0 ? System.currentTimeMillis() - lastResonanceTime : -1;
    }

    /**
     * Gets resonance detection statistics.
     *
     * @return resonance statistics
     */
    public ResonanceStats getResonanceStats() {
        var totalDetectionsCount = totalDetections.get();
        var resonanceRate = totalDetectionsCount > 0 ?
            (double) resonanceDetections.get() / totalDetectionsCount : 0.0;
        var avgResonanceStrength = totalDetectionsCount > 0 ?
            cumulativeResonanceStrength.sum() / totalDetectionsCount : 0.0;
        var avgDetectionTime = totalDetectionsCount > 0 ?
            (double) totalDetectionTime.get() / totalDetectionsCount : 0.0;

        return new ResonanceStats(
            totalDetectionsCount,
            resonanceDetections.get(),
            resonanceRate,
            avgResonanceStrength,
            avgDetectionTime,
            stabilityCounter.get()
        );
    }

    /**
     * Resets the resonance detector state.
     */
    public void reset() {
        logger.info("Resetting resonance detector");

        lastResonanceStrength = 0.0;
        lastResonanceTime = 0;
        isResonant = false;
        stabilityCounter.set(0);

        totalDetections.set(0);
        resonanceDetections.set(0);
        totalDetectionTime.set(0);
        cumulativeResonanceStrength.reset();

        logger.info("Resonance detector reset completed");
    }

    /**
     * Calculates similarity between two feature vectors using multiple metrics.
     */
    private double calculateFeatureSimilarity(float[] features1, float[] features2) {
        if (features1.length == 0) {
            return 0.0;
        }

        // Convert to JOML vectors for mathematical operations
        var vec1 = new Vector3f();
        var vec2 = new Vector3f();
        var similarities = new double[Math.min(features1.length / 3, 1)];

        // Calculate similarity for vector segments
        for (int i = 0; i < features1.length && i < features2.length; i += 3) {
            var x1 = features1[i];
            var y1 = i + 1 < features1.length ? features1[i + 1] : 0.0f;
            var z1 = i + 2 < features1.length ? features1[i + 2] : 0.0f;

            var x2 = features2[i];
            var y2 = i + 1 < features2.length ? features2[i + 1] : 0.0f;
            var z2 = i + 2 < features2.length ? features2[i + 2] : 0.0f;

            vec1.set(x1, y1, z1);
            vec2.set(x2, y2, z2);

            // Cosine similarity
            var dotProduct = vec1.dot(vec2);
            var magnitude1 = vec1.length();
            var magnitude2 = vec2.length();

            if (magnitude1 > 0 && magnitude2 > 0) {
                similarities[Math.min(i / 3, similarities.length - 1)] = dotProduct / (magnitude1 * magnitude2);
            }
        }

        // Return average similarity
        return similarities.length > 0 ?
            java.util.Arrays.stream(similarities).average().orElse(0.0) : 0.0;
    }

    /**
     * Calculates template match strength.
     */
    private double calculateTemplateMatch(float[] features, Template template) {
        if (template == null) {
            return 0.0;
        }

        // Use template's base confidence as starting point
        var baseMatch = template.baseConfidence();

        // Calculate feature-based enhancement
        // This is a simplified implementation - in practice, would use more sophisticated matching
        var featureSum = 0.0;
        for (var feature : features) {
            featureSum += Math.abs(feature);
        }
        var featureStrength = Math.min(1.0, featureSum / features.length);

        return baseMatch * 0.7 + featureStrength * 0.3;
    }

    /**
     * Combines multiple resonance factors into overall strength.
     */
    private double combineResonanceFactors(double featureSimilarity, double templateMatch) {
        // Weighted combination of factors
        var featureWeight = 0.6;
        var templateWeight = 0.4;

        var combinedScore = (featureSimilarity * featureWeight) + (templateMatch * templateWeight);

        // Apply vigilance parameter as a scaling factor
        return combinedScore * (1.0 + vigilanceParameter) / 2.0;
    }

    /**
     * Updates detection state after resonance check.
     */
    private void updateDetectionState(boolean resonant, double strength) {
        this.isResonant = resonant;
        this.lastResonanceStrength = strength;
        if (resonant) {
            this.lastResonanceTime = System.currentTimeMillis();
        }

        // Update cumulative statistics
        cumulativeResonanceStrength.add(strength);
    }

    /**
     * Creates a non-resonant result.
     */
    private ResonanceResult createNoResonanceResult(MismatchType mismatchType, String reason, long detectionTime) {
        return new ResonanceResult(false, 0.0, 0.0, mismatchType, reason, detectionTime);
    }

    /**
     * Statistics about resonance detection performance.
     */
    public static class ResonanceStats {
        private final int totalDetections;
        private final int resonanceDetections;
        private final double resonanceRate;
        private final double averageResonanceStrength;
        private final double averageDetectionTimeMs;
        private final int currentStabilityCount;

        public ResonanceStats(int totalDetections, int resonanceDetections, double resonanceRate,
                            double averageResonanceStrength, double averageDetectionTimeMs,
                            int currentStabilityCount) {
            this.totalDetections = totalDetections;
            this.resonanceDetections = resonanceDetections;
            this.resonanceRate = resonanceRate;
            this.averageResonanceStrength = averageResonanceStrength;
            this.averageDetectionTimeMs = averageDetectionTimeMs;
            this.currentStabilityCount = currentStabilityCount;
        }

        public int getTotalDetections() { return totalDetections; }
        public int getResonanceDetections() { return resonanceDetections; }
        public double getResonanceRate() { return resonanceRate; }
        public double getAverageResonanceStrength() { return averageResonanceStrength; }
        public double getAverageDetectionTimeMs() { return averageDetectionTimeMs; }
        public int getCurrentStabilityCount() { return currentStabilityCount; }

        @Override
        public String toString() {
            return "ResonanceStats{detections=%d, resonanceRate=%.2f%%, avgStrength=%.3f, avgTime=%.2fms}"
                .formatted(totalDetections, resonanceRate * 100, averageResonanceStrength, averageDetectionTimeMs);
        }
    }
}