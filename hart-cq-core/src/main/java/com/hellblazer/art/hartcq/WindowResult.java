package com.hellblazer.art.hartcq;

import java.util.List;
import java.util.Objects;

/**
 * Result of window processing in the HART-CQ system.
 * Contains all relevant information about the processing of a sliding window,
 * including extracted features, patterns, confidence metrics, and timing information.
 */
public class WindowResult {
    private final long windowId;
    private final WindowFeatures features;
    private final List<Pattern> patterns;
    private final double confidence;
    private final long processingTimeNanos;
    
    /**
     * Constructs a new WindowResult.
     * 
     * @param windowId unique identifier for this window
     * @param features extracted features from the window
     * @param patterns list of patterns found in the window
     * @param confidence confidence score for the processing result [0.0, 1.0]
     * @param processingTimeNanos time taken to process the window in nanoseconds
     */
    public WindowResult(long windowId, WindowFeatures features, List<Pattern> patterns, 
                       double confidence, long processingTimeNanos) {
        this.windowId = windowId;
        this.features = Objects.requireNonNull(features, "Features cannot be null");
        this.patterns = Objects.requireNonNull(patterns, "Patterns cannot be null");
        this.confidence = Math.max(0.0, Math.min(1.0, confidence)); // Clamp to [0,1]
        this.processingTimeNanos = processingTimeNanos;
    }
    
    /**
     * Gets the unique window identifier.
     * @return window ID
     */
    public long getWindowId() {
        return windowId;
    }
    
    /**
     * Gets the features extracted from this window.
     * @return window features
     */
    public WindowFeatures getFeatures() {
        return features;
    }
    
    /**
     * Gets the patterns found in this window.
     * @return list of patterns
     */
    public List<Pattern> getPatterns() {
        return patterns;
    }
    
    /**
     * Gets the confidence score for this window processing result.
     * @return confidence score between 0.0 and 1.0
     */
    public double getConfidence() {
        return confidence;
    }
    
    /**
     * Gets the processing time for this window in nanoseconds.
     * @return processing time in nanoseconds
     */
    public long getProcessingTimeNanos() {
        return processingTimeNanos;
    }
    
    /**
     * Gets the processing time in milliseconds for convenience.
     * @return processing time in milliseconds
     */
    public double getProcessingTimeMillis() {
        return processingTimeNanos / 1_000_000.0;
    }
    
    /**
     * Checks if this window result indicates successful processing.
     * @return true if confidence is above 0.5 and patterns were found
     */
    public boolean isSuccessful() {
        return confidence > 0.5 && !patterns.isEmpty();
    }
    
    /**
     * Gets the number of patterns found in this window.
     * @return pattern count
     */
    public int getPatternCount() {
        return patterns.size();
    }
    
    /**
     * Calculates a composite quality score based on confidence and feature quality.
     * @return composite quality score [0.0, 1.0]
     */
    public double getQualityScore() {
        return (confidence * 0.7) + (features.getQualityScore() * 0.3);
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        var that = (WindowResult) o;
        return windowId == that.windowId &&
               Double.compare(that.confidence, confidence) == 0 &&
               processingTimeNanos == that.processingTimeNanos &&
               Objects.equals(features, that.features) &&
               Objects.equals(patterns, that.patterns);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(windowId, features, patterns, confidence, processingTimeNanos);
    }
    
    @Override
    public String toString() {
        return String.format("WindowResult[id=%d, patterns=%d, confidence=%.3f, time=%.2fms, quality=%.3f]",
                           windowId, patterns.size(), confidence, getProcessingTimeMillis(), getQualityScore());
    }
    
    /**
     * Simple pattern representation for HART-CQ processing.
     */
    public static class Pattern {
        private final String type;
        private final String value;
        private final double strength;
        private final int position;
        
        public Pattern(String type, String value, double strength, int position) {
            this.type = Objects.requireNonNull(type, "Pattern type cannot be null");
            this.value = Objects.requireNonNull(value, "Pattern value cannot be null");
            this.strength = Math.max(0.0, Math.min(1.0, strength)); // Clamp to [0,1]
            this.position = position;
        }
        
        public String getType() {
            return type;
        }
        
        public String getValue() {
            return value;
        }
        
        public double getStrength() {
            return strength;
        }
        
        public int getPosition() {
            return position;
        }
        
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            var pattern = (Pattern) o;
            return Double.compare(pattern.strength, strength) == 0 &&
                   position == pattern.position &&
                   Objects.equals(type, pattern.type) &&
                   Objects.equals(value, pattern.value);
        }
        
        @Override
        public int hashCode() {
            return Objects.hash(type, value, strength, position);
        }
        
        @Override
        public String toString() {
            return String.format("Pattern[type=%s, value=%s, strength=%.3f, pos=%d]",
                               type, value, strength, position);
        }
    }
}