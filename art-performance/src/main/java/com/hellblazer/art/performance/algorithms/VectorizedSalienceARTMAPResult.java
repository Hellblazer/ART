package com.hellblazer.art.performance.algorithms;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Immutable result of VectorizedSalienceARTMAP prediction with salience-aware metrics.
 * Contains prediction details, activation levels, and salience-specific information.
 */
public record VectorizedSalienceARTMAPResult(
    int predictedCategory,
    double confidence,
    double artAActivation,
    double artBActivation,
    double mapFieldActivation,
    Map<String, Double> salienceMetrics,
    String noMatchReason
) {
    
    /**
     * Constructor with validation and immutability enforcement
     */
    public VectorizedSalienceARTMAPResult {
        if (confidence < 0.0 || confidence > 1.0) {
            throw new IllegalArgumentException("Confidence must be in [0,1], got: " + confidence);
        }
        if (artAActivation < 0.0 || artAActivation > 1.0) {
            throw new IllegalArgumentException("ART-A activation must be in [0,1], got: " + artAActivation);
        }
        if (artBActivation < 0.0 || artBActivation > 1.0) {
            throw new IllegalArgumentException("ART-B activation must be in [0,1], got: " + artBActivation);
        }
        if (mapFieldActivation < 0.0 || mapFieldActivation > 1.0) {
            throw new IllegalArgumentException("Map field activation must be in [0,1], got: " + mapFieldActivation);
        }
        
        // Make defensive copy and wrap in unmodifiable map
        salienceMetrics = Collections.unmodifiableMap(
            new HashMap<>(salienceMetrics != null ? salienceMetrics : Map.of())
        );
    }
    
    /**
     * Create a successful prediction result
     */
    public static VectorizedSalienceARTMAPResult success(
        int predictedCategory,
        double confidence,
        double artAActivation,
        double artBActivation,
        double mapFieldActivation
    ) {
        return new VectorizedSalienceARTMAPResult(
            predictedCategory,
            confidence,
            artAActivation,
            artBActivation,
            mapFieldActivation,
            Map.of(),
            null
        );
    }
    
    /**
     * Create a successful prediction result with salience metrics
     */
    public static VectorizedSalienceARTMAPResult successWithMetrics(
        int predictedCategory,
        double confidence,
        double artAActivation,
        double artBActivation,
        double mapFieldActivation,
        Map<String, Double> salienceMetrics
    ) {
        return new VectorizedSalienceARTMAPResult(
            predictedCategory,
            confidence,
            artAActivation,
            artBActivation,
            mapFieldActivation,
            salienceMetrics,
            null
        );
    }
    
    /**
     * Create a no-match result
     */
    public static VectorizedSalienceARTMAPResult noMatch(String reason) {
        return new VectorizedSalienceARTMAPResult(
            -1,
            0.0,
            0.0,
            0.0,
            0.0,
            Map.of(),
            Objects.requireNonNull(reason, "No match reason cannot be null")
        );
    }
    
    /**
     * Check if prediction was successful
     */
    public boolean isPredictionSuccessful() {
        return predictedCategory >= 0 && noMatchReason == null;
    }
    
    /**
     * Builder for creating results
     */
    public static Builder builder() {
        return new Builder();
    }
    
    public static class Builder {
        private int predictedCategory = -1;
        private double confidence = 0.0;
        private double artAActivation = 0.0;
        private double artBActivation = 0.0;
        private double mapFieldActivation = 0.0;
        private Map<String, Double> salienceMetrics = new HashMap<>();
        private String noMatchReason = null;
        
        public Builder predictedCategory(int predictedCategory) {
            this.predictedCategory = predictedCategory;
            return this;
        }
        
        public Builder confidence(double confidence) {
            this.confidence = confidence;
            return this;
        }
        
        public Builder artAActivation(double artAActivation) {
            this.artAActivation = artAActivation;
            return this;
        }
        
        public Builder artBActivation(double artBActivation) {
            this.artBActivation = artBActivation;
            return this;
        }
        
        public Builder mapFieldActivation(double mapFieldActivation) {
            this.mapFieldActivation = mapFieldActivation;
            return this;
        }
        
        public Builder addSalienceMetric(String name, double value) {
            this.salienceMetrics.put(name, value);
            return this;
        }
        
        public Builder salienceMetrics(Map<String, Double> metrics) {
            this.salienceMetrics = new HashMap<>(metrics);
            return this;
        }
        
        public Builder noMatch(String reason) {
            this.noMatchReason = reason;
            this.predictedCategory = -1;
            this.confidence = 0.0;
            return this;
        }
        
        public VectorizedSalienceARTMAPResult build() {
            return new VectorizedSalienceARTMAPResult(
                predictedCategory,
                confidence,
                artAActivation,
                artBActivation,
                mapFieldActivation,
                salienceMetrics,
                noMatchReason
            );
        }
    }
    
    @Override
    public String toString() {
        if (isPredictionSuccessful()) {
            return String.format(
                "VectorizedSalienceARTMAPResult{category=%d, confidence=%.2f, " +
                "artA=%.2f, artB=%.2f, map=%.2f, metrics=%d}",
                predictedCategory, confidence, artAActivation, artBActivation,
                mapFieldActivation, salienceMetrics.size()
            );
        } else {
            return String.format(
                "VectorizedSalienceARTMAPResult{noMatch, reason='%s'}",
                noMatchReason
            );
        }
    }
}