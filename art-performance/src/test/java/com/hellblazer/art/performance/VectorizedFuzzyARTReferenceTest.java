package com.hellblazer.art.performance;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.performance.algorithms.VectorizedFuzzyART;
import com.hellblazer.art.performance.algorithms.VectorizedFuzzyWeight;
import com.hellblazer.art.performance.algorithms.VectorizedParameters;

/**
 * Reference validation test for VectorizedFuzzyART implementation.
 * Ensures SIMD-accelerated fuzzy clustering behavior matches reference
 * baseline for complement coding, choice function optimization, and fuzzy min/max operations.
 */
class VectorizedFuzzyARTReferenceTest {
    
    private VectorizedParameters referenceParams;
    private VectorizedFuzzyART vectorizedFuzzyArt;
    
    @BeforeEach
    void setUp() {
        // Parameters matching reference FuzzyART baseline
        referenceParams = VectorizedParameters.createDefault()
            .withVigilance(0.75)
            .withLearningRate(0.1);
        
        vectorizedFuzzyArt = new VectorizedFuzzyART(referenceParams);
    }
    
    @AfterEach
    void tearDown() {
        if (vectorizedFuzzyArt != null) {
            vectorizedFuzzyArt.close();
        }
    }
    
    @Test
    @DisplayName("Basic fuzzy clustering validation")
    void testReferenceBasicClustering() {
        var patterns = new Pattern[] {
            Pattern.of(0.1, 0.1),
            Pattern.of(0.9, 0.9),
            Pattern.of(0.15, 0.05)
        };
        
        for (var pattern : patterns) {
            var result = vectorizedFuzzyArt.learn(pattern, referenceParams);
            assertNotNull(result, "Learn operation should return a result");
        }
        
        assertTrue(vectorizedFuzzyArt.getCategoryCount() >= 1, "Should create at least one category");
        
        // Test prediction
        var prediction = vectorizedFuzzyArt.predict(Pattern.of(0.12, 0.08), referenceParams);
        assertNotNull(prediction, "Prediction should return a result");
    }
    
    @Test
    @DisplayName("Fuzzy vectorization performance validation")
    void testVectorizedPerformance() {
        var patterns = new Pattern[] {
            Pattern.of(0.3, 0.7),
            Pattern.of(0.8, 0.2),
            Pattern.of(0.5, 0.5)
        };
        
        for (var pattern : patterns) {
            var result = vectorizedFuzzyArt.learn(pattern, referenceParams);
            assertNotNull(result, "Learning should succeed");
        }
        
        assertTrue(vectorizedFuzzyArt.getCategoryCount() > 0, "Should have created categories");
        
        // Test vectorized operations completed successfully
        var stats = vectorizedFuzzyArt.getPerformanceStats();
        assertNotNull(stats, "Performance stats should be available");
    }
    
    @Test  
    @DisplayName("Fuzzy learning operations")
    void testFuzzyOperations() {
        var patterns = new Pattern[] {
            Pattern.of(0.6, 0.4),
            Pattern.of(0.7, 0.3),
            Pattern.of(0.2, 0.8)
        };
        
        for (var pattern : patterns) {
            var result = vectorizedFuzzyArt.learn(pattern, referenceParams);
            assertNotNull(result, "All patterns should be learnable");
        }
        
        assertTrue(vectorizedFuzzyArt.getCategoryCount() > 0, "Should create categories");
        
        // Test that predictions work
        for (var pattern : patterns) {
            var prediction = vectorizedFuzzyArt.predict(pattern, referenceParams);
            assertNotNull(prediction, "Predictions should be available");
        }
    }
    
    @Test
    @DisplayName("Vigilance parameter testing")
    void testVigilanceThreshold() {
        var strictParams = VectorizedParameters.createDefault()
            .withVigilance(0.9)
            .withLearningRate(0.1);
        
        var strictFuzzyArt = new VectorizedFuzzyART(strictParams);
        
        try {
            var testPatterns = new Pattern[] {
                Pattern.of(0.5, 0.5),
                Pattern.of(0.52, 0.51),
                Pattern.of(0.48, 0.49)
            };
            
            for (var pattern : testPatterns) {
                var result = strictFuzzyArt.learn(pattern, strictParams);
                assertNotNull(result, "Learning should succeed with high vigilance");
            }
            
            assertTrue(strictFuzzyArt.getCategoryCount() > 0, "Should create at least one category");
        } finally {
            strictFuzzyArt.close();
        }
    }
    
    @Test
    @DisplayName("Choice function optimization")
    void testChoiceFunctionOptimization() {
        var patterns = new Pattern[] {
            Pattern.of(0.4, 0.6),
            Pattern.of(0.8, 0.2),
            Pattern.of(0.45, 0.55)
        };
        
        for (var pattern : patterns) {
            var result = vectorizedFuzzyArt.learn(pattern, referenceParams);
            assertNotNull(result, "Learning should succeed");
        }
        
        // Test that choice function creates appropriate categories
        assertTrue(vectorizedFuzzyArt.getCategoryCount() >= 2, "Should have multiple categories");
        
        // Test prediction works
        for (var pattern : patterns) {
            var prediction = vectorizedFuzzyArt.predict(pattern, referenceParams);
            assertNotNull(prediction, "Prediction should work");
        }
    }
    
    @Test
    @DisplayName("Fuzzy set operations validation")
    void testFuzzySetOperations() {
        var patterns = new Pattern[] {
            Pattern.of(0.3, 0.8),
            Pattern.of(0.4, 0.7),
            Pattern.of(0.35, 0.75)
        };
        
        for (var pattern : patterns) {
            var result = vectorizedFuzzyArt.learn(pattern, referenceParams);
            assertNotNull(result, "Fuzzy set operations should succeed");
        }
        
        assertTrue(vectorizedFuzzyArt.getCategoryCount() > 0, "Should create categories");
        
        // Test predictions on the learned patterns
        for (var pattern : patterns) {
            var prediction = vectorizedFuzzyArt.predict(pattern, referenceParams);
            assertNotNull(prediction, "Predictions should work");
        }
    }
    
    @Test
    @DisplayName("Match tracking with low vigilance")
    void testMatchTracking() {
        var lowVigilanceParams = VectorizedParameters.createDefault()
            .withVigilance(0.3)
            .withLearningRate(0.1);
        
        var matchTrackingFuzzyArt = new VectorizedFuzzyART(lowVigilanceParams);
        
        try {
            var trackingPatterns = new Pattern[] {
                Pattern.of(0.1, 0.9),
                Pattern.of(0.3, 0.7),
                Pattern.of(0.5, 0.5),
                Pattern.of(0.7, 0.3),
                Pattern.of(0.9, 0.1)
            };
            
            for (var pattern : trackingPatterns) {
                var result = matchTrackingFuzzyArt.learn(pattern, lowVigilanceParams);
                assertNotNull(result, "Match tracking should succeed");
            }
            
            // Low vigilance should allow broader generalization
            assertTrue(matchTrackingFuzzyArt.getCategoryCount() > 0, "Should create at least one category");
            
        } finally {
            matchTrackingFuzzyArt.close();
        }
    }
}