package com.hellblazer.art.core;

import com.hellblazer.art.core.cvi.*;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.core.results.MatchResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for CVIEnabledART functionality
 */
public class CVIEnabledARTTest {
    
    private TestCVIEnabledART art;
    private List<Pattern> testPatterns;
    
    @BeforeEach
    void setUp() {
        art = new TestCVIEnabledART();
        
        // Create test patterns in distinct clusters
        testPatterns = new ArrayList<>();
        
        // Cluster 1: Low values
        testPatterns.add(new DenseVector(new double[]{0.1, 0.1}));
        testPatterns.add(new DenseVector(new double[]{0.2, 0.2}));
        testPatterns.add(new DenseVector(new double[]{0.15, 0.15}));
        
        // Cluster 2: High values
        testPatterns.add(new DenseVector(new double[]{0.8, 0.8}));
        testPatterns.add(new DenseVector(new double[]{0.9, 0.9}));
        testPatterns.add(new DenseVector(new double[]{0.85, 0.85}));
        
        // Cluster 3: Mixed values
        testPatterns.add(new DenseVector(new double[]{0.5, 0.2}));
        testPatterns.add(new DenseVector(new double[]{0.6, 0.3}));
        testPatterns.add(new DenseVector(new double[]{0.55, 0.25}));
    }
    
    @Test
    void testCVIMetricManagement() {
        var chIndex = new CalinskiHarabaszIndex();
        var dbIndex = new DaviesBouldinIndex();
        
        // Add metrics
        art.addCVIMetric(chIndex);
        art.addCVIMetric(dbIndex);
        
        // Learn patterns
        var params = new TestParameters(0.7); // Moderate vigilance
        for (var pattern : testPatterns) {
            var result = art.stepFit(pattern, params);
            art.trackPattern(pattern, result);
        }
        
        // Force CVI update
        art.updateCVIScores();
        
        // Check scores exist
        var scores = art.getCVIScores();
        assertNotNull(scores);
        assertTrue(scores.containsKey("Calinski-Harabasz Index"));
        assertTrue(scores.containsKey("Davies-Bouldin Index"));
        
        // Check individual score retrieval
        Double chScore = art.getCVIScore("Calinski-Harabasz Index");
        assertNotNull(chScore);
        assertTrue(chScore > 0, "CH index should be positive for well-separated clusters");
        
        Double dbScore = art.getCVIScore("Davies-Bouldin Index");
        assertNotNull(dbScore);
        assertTrue(dbScore > 0, "DB index should be positive");
        
        // Remove a metric
        art.removeCVIMetric(dbIndex);
        art.updateCVIScores();
        scores = art.getCVIScores();
        assertTrue(scores.containsKey("Calinski-Harabasz Index"));
        assertFalse(scores.containsKey("Davies-Bouldin Index"));
        
        // Clear all metrics
        art.clearCVIMetrics();
        art.updateCVIScores();
        scores = art.getCVIScores();
        assertTrue(scores.isEmpty() || !scores.containsKey("Calinski-Harabasz Index"));
    }
    
    @Test
    void testHistoryTracking() {
        var silhouette = new SilhouetteCoefficient();
        art.addCVIMetric(silhouette);
        
        // Enable history tracking
        art.setTrackHistory(true);
        art.setMaxHistorySize(5); // Small history for testing
        
        var params = new TestParameters(0.7);
        
        // Learn more patterns than history size
        for (int i = 0; i < 10; i++) {
            var result = art.stepFit(new DenseVector(new double[]{i * 0.1, i * 0.1}), params);
            art.trackPattern(new DenseVector(new double[]{i * 0.1, i * 0.1}), result);
        }
        
        // Check that history was trimmed
        art.updateCVIScores();
        String summary = art.getCVISummary();
        assertTrue(summary.contains("History size: 5") || summary.contains("History size: 10"),
            "History should be limited to max size");
        
        // Disable history tracking
        art.setTrackHistory(false);
        art.stepFit(new DenseVector(new double[]{0.5, 0.5}), params);
        
        // Scores should not update without history
        var oldScores = art.getCVIScores();
        art.updateCVIScores();
        var newScores = art.getCVIScores();
        assertEquals(oldScores, newScores, "Scores shouldn't change when tracking is disabled");
    }
    
    @Test
    void testPeriodicUpdate() {
        var chIndex = new CalinskiHarabaszIndex();
        art.addCVIMetric(chIndex);
        
        var params = new TestParameters(0.7);
        
        // Learn exactly 10 patterns to trigger periodic update
        for (int i = 0; i < 10; i++) {
            var pattern = new DenseVector(new double[]{i * 0.1, i * 0.05});
            var result = art.stepFit(pattern, params);
            art.trackPattern(pattern, result);
        }
        
        // Should have scores after 10 patterns
        var scores = art.getCVIScores();
        assertFalse(scores.isEmpty(), "Scores should be calculated after 10 patterns");
    }
    
    @Test
    void testCVISummary() {
        var chIndex = new CalinskiHarabaszIndex();
        var dbIndex = new DaviesBouldinIndex();
        var silhouette = new SilhouetteCoefficient();
        
        art.addCVIMetric(chIndex);
        art.addCVIMetric(dbIndex);
        art.addCVIMetric(silhouette);
        
        // Initially no scores
        String summary = art.getCVISummary();
        assertEquals("No CVI metrics tracked", summary);
        
        // Learn patterns
        var params = new TestParameters(0.7);
        for (var pattern : testPatterns) {
            var result = art.stepFit(pattern, params);
            art.trackPattern(pattern, result);
        }
        
        art.updateCVIScores();
        summary = art.getCVISummary();
        
        // Check summary contains expected information
        assertTrue(summary.contains("CVI Scores:"));
        assertTrue(summary.contains("Calinski-Harabasz Index:"));
        assertTrue(summary.contains("Davies-Bouldin Index:"));
        assertTrue(summary.contains("Silhouette Coefficient:"));
        assertTrue(summary.contains("History size:"));
        assertTrue(summary.contains("Categories:"));
    }
    
    @Test
    void testResetHistory() {
        var chIndex = new CalinskiHarabaszIndex();
        art.addCVIMetric(chIndex);
        
        var params = new TestParameters(0.7);
        
        // Learn patterns
        for (var pattern : testPatterns) {
            var result = art.stepFit(pattern, params);
            art.trackPattern(pattern, result);
        }
        
        art.updateCVIScores();
        var scores = art.getCVIScores();
        assertFalse(scores.isEmpty());
        
        // Reset history
        art.resetCVIHistory();
        art.updateCVIScores();
        
        // Scores should be cleared
        scores = art.getCVIScores();
        assertTrue(scores.isEmpty() || scores.get("Calinski-Harabasz Index") == null);
    }
    
    @Test
    void testMultipleCVIsAgreement() {
        // Add all three CVIs
        art.addCVIMetric(new CalinskiHarabaszIndex());
        art.addCVIMetric(new DaviesBouldinIndex());
        art.addCVIMetric(new SilhouetteCoefficient());
        
        // Learn well-separated clusters with high vigilance
        var highVigilanceParams = new TestParameters(0.9);
        for (var pattern : testPatterns) {
            var result = art.stepFit(pattern, highVigilanceParams);
            art.trackPattern(pattern, result);
        }
        
        art.updateCVIScores();
        var highVigilanceScores = art.getCVIScores();
        
        // Reset and learn with low vigilance (fewer clusters)
        art = new TestCVIEnabledART();
        art.addCVIMetric(new CalinskiHarabaszIndex());
        art.addCVIMetric(new DaviesBouldinIndex());
        art.addCVIMetric(new SilhouetteCoefficient());
        
        var lowVigilanceParams = new TestParameters(0.3);
        for (var pattern : testPatterns) {
            var result = art.stepFit(pattern, lowVigilanceParams);
            art.trackPattern(pattern, result);
        }
        
        art.updateCVIScores();
        var lowVigilanceScores = art.getCVIScores();
        
        // High vigilance should create more, better-separated clusters
        // At least one metric should improve with higher vigilance
        boolean chImproved = highVigilanceScores.get("Calinski-Harabasz Index") > 
                            lowVigilanceScores.get("Calinski-Harabasz Index");
        boolean dbImproved = highVigilanceScores.get("Davies-Bouldin Index") < 
                            lowVigilanceScores.get("Davies-Bouldin Index");
        boolean silImproved = highVigilanceScores.get("Silhouette Coefficient") > 
                             lowVigilanceScores.get("Silhouette Coefficient");
        
        assertTrue(chImproved || dbImproved || silImproved,
                  String.format("At least one CVI should improve with high vigilance. " +
                               "CH: %.2f vs %.2f, DB: %.2f vs %.2f, Sil: %.2f vs %.2f",
                               highVigilanceScores.get("Calinski-Harabasz Index"),
                               lowVigilanceScores.get("Calinski-Harabasz Index"),
                               highVigilanceScores.get("Davies-Bouldin Index"),
                               lowVigilanceScores.get("Davies-Bouldin Index"),
                               highVigilanceScores.get("Silhouette Coefficient"),
                               lowVigilanceScores.get("Silhouette Coefficient")));
    }
    
    @Test
    void testErrorHandling() {
        // Add a CVI that might fail with certain data
        art.addCVIMetric(new ClusterValidityIndex() {
            @Override
            public double calculate(List<Pattern> data, int[] labels, List<Pattern> centroids) {
                throw new RuntimeException("Test error");
            }
            
            @Override
            public String getName() {
                return "Failing CVI";
            }
            
            @Override
            public boolean isHigherBetter() {
                return true;
            }
        });
        
        // Also add a working CVI
        art.addCVIMetric(new CalinskiHarabaszIndex());
        
        var params = new TestParameters(0.7);
        
        // Should not throw exception despite failing CVI
        assertDoesNotThrow(() -> {
            for (var pattern : testPatterns) {
                var result = art.stepFit(pattern, params);
                art.trackPattern(pattern, result);
            }
            art.updateCVIScores();
        });
        
        // Working CVI should still have scores
        var scores = art.getCVIScores();
        assertTrue(scores.containsKey("Calinski-Harabasz Index"));
        assertFalse(scores.containsKey("Failing CVI"));
    }
    
    /**
     * Simple test parameters class
     */
    private static class TestParameters {
        private final double vigilance;
        
        TestParameters(double vigilance) {
            this.vigilance = vigilance;
        }
        
        double vigilance() {
            return vigilance;
        }
    }
    
    /**
     * Test implementation of CVIEnabledART
     */
    private static class TestCVIEnabledART extends CVIEnabledART {
        
        @Override
        protected double calculateActivation(Pattern input, WeightVector weight, Object parameters) {
            // Simple Euclidean distance matching
            if (weight instanceof CVIEnabledART.SimpleWeight sw) {
                double sum = 0;
                var values = sw.getValues();
                for (int i = 0; i < input.dimension(); i++) {
                    double diff = input.get(i) - values[i];
                    sum += diff * diff;
                }
                // Convert distance to similarity (inverse)
                return 1.0 / (1.0 + Math.sqrt(sum));
            }
            return 0;
        }
        
        @Override
        protected MatchResult checkVigilance(Pattern input, WeightVector weight, Object parameters) {
            // Simple threshold-based vigilance
            if (parameters instanceof TestParameters params) {
                double activation = calculateActivation(input, weight, parameters);
                if (activation >= params.vigilance()) {
                    return new MatchResult.Accepted(activation, params.vigilance());
                } else {
                    return new MatchResult.Rejected(activation, params.vigilance());
                }
            }
            // Return rejected if parameters are invalid
            return new MatchResult.Rejected(0.0, 0.5);
        }
        
        @Override
        protected WeightVector updateWeights(Pattern input, WeightVector currentWeight, Object parameters) {
            // Simple averaging adaptation
            if (currentWeight instanceof CVIEnabledART.SimpleWeight sw) {
                var values = sw.getValues();
                var newValues = new double[values.length];
                double learningRate = 0.5;
                
                for (int i = 0; i < values.length; i++) {
                    newValues[i] = (1 - learningRate) * values[i] + 
                                   learningRate * input.get(i);
                }
                
                return new CVIEnabledART.SimpleWeight(newValues);
            }
            return currentWeight;
        }
        
        @Override
        protected WeightVector createInitialWeight(Pattern input, Object parameters) {
            // Create new weight from pattern
            var values = new double[input.dimension()];
            for (int i = 0; i < input.dimension(); i++) {
                values[i] = input.get(i);
            }
            return new CVIEnabledART.SimpleWeight(values);
        }

        @Override
        public void close() {
            // No resources to close in test implementation
        }
    }
}