package com.hellblazer.art.performance.algorithms;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Performance statistics for VectorizedSalienceARTMAP algorithm.
 * Tracks supervised learning metrics, map field operations, and salience adaptation.
 */
public record VectorizedSalienceARTMAPPerformanceStats(
    long totalMapFieldOperations,
    long matchTrackingEvents,
    long resonanceSuccesses,
    long resonanceFailures,
    double averageMapFieldActivation,
    double averageConfidence,
    long crossSalienceAdaptations,
    Map<String, Double> moduleMetrics,
    double learningEfficiency,
    double predictionAccuracy
) {
    
    /**
     * Constructor that ensures immutability of the metrics map
     */
    public VectorizedSalienceARTMAPPerformanceStats {
        // Make defensive copy and wrap in unmodifiable map
        moduleMetrics = Collections.unmodifiableMap(
            new HashMap<>(moduleMetrics != null ? moduleMetrics : Map.of())
        );
    }
    
    /**
     * Create empty statistics instance
     */
    public static VectorizedSalienceARTMAPPerformanceStats empty() {
        return new VectorizedSalienceARTMAPPerformanceStats(
            0L,     // totalMapFieldOperations
            0L,     // matchTrackingEvents
            0L,     // resonanceSuccesses
            0L,     // resonanceFailures
            0.0,    // averageMapFieldActivation
            0.0,    // averageConfidence
            0L,     // crossSalienceAdaptations
            Map.of(), // moduleMetrics (empty)
            0.0,    // learningEfficiency
            0.0     // predictionAccuracy
        );
    }
    
    /**
     * Calculate resonance success rate
     */
    public double getResonanceSuccessRate() {
        long total = resonanceSuccesses + resonanceFailures;
        return total > 0 ? (double) resonanceSuccesses / total : 0.0;
    }
    
    /**
     * Calculate match tracking frequency
     */
    public double getMatchTrackingFrequency() {
        return totalMapFieldOperations > 0 ? 
            (double) matchTrackingEvents / totalMapFieldOperations : 0.0;
    }
    
    /**
     * Get formatted performance summary
     */
    @Override
    public String toString() {
        return String.format(
            "VectorizedSalienceARTMAPPerformanceStats{" +
            "mapFieldOps=%d, matchTracking=%d (%.1f%%), " +
            "resonance=%d/%d (%.1f%% success), " +
            "avgMapActivation=%.3f, avgConfidence=%.3f, " +
            "crossSalienceAdapt=%d, efficiency=%.2f, accuracy=%.2f}",
            totalMapFieldOperations, matchTrackingEvents, getMatchTrackingFrequency() * 100,
            resonanceSuccesses, resonanceSuccesses + resonanceFailures, getResonanceSuccessRate() * 100,
            averageMapFieldActivation, averageConfidence,
            crossSalienceAdaptations, learningEfficiency, predictionAccuracy
        );
    }
}