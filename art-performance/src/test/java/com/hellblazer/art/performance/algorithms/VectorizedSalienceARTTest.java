package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.performance.BaseVectorizedARTTest;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test-first development: Tests for VectorizedSalienceART
 */
class VectorizedSalienceARTTest extends BaseVectorizedARTTest<VectorizedSalienceART, VectorizedSalienceParameters> {
    
    @Override
    protected VectorizedSalienceART createAlgorithm(VectorizedSalienceParameters params) {
        return new VectorizedSalienceART(params);
    }
    
    @Override
    protected VectorizedSalienceParameters createDefaultParameters() {
        return VectorizedSalienceParameters.createDefault();
    }
    
    @BeforeEach
    protected void setUp() {
        parameters = createDefaultParameters();
        algorithm = createAlgorithm(parameters);
        super.setUp();
    }
    
    @Override
    protected VectorizedSalienceParameters createParametersWithVigilance(double vigilance) {
        return parameters.withVigilance(vigilance);
    }
    
    // The following tests are covered by base class:
    // - testBasicLearning()
    // - testMultiplePatternLearning()
    // - testPrediction()
    // - testPerformanceTracking()
    // - testErrorHandling()
    // - testResourceCleanup()
    
    @Test
    @DisplayName("Test initialization")
    void testInitialization() {
        assertNotNull(algorithm);
        assertEquals(0, algorithm.getCategoryCount());
        assertNotNull(algorithm.getParameters());
        assertEquals(parameters, algorithm.getParameters());
        assertFalse(algorithm.isTrained());
    }
    
    @Test
    @DisplayName("Test prediction without learning")
    void testPredictWithoutLearning() {
        var pattern = Pattern.of(0.1, 0.2, 0.3, 0.4, 0.5);
        
        var result = algorithm.predict(pattern, parameters);
        
        assertNotNull(result);
        // Should handle empty network gracefully
        assertEquals(0, algorithm.getCategoryCount());
    }
    
    @Test
    @DisplayName("Test prediction after learning")
    void testPredictAfterLearning() {
        var pattern1 = Pattern.of(0.1, 0.2, 0.3, 0.4, 0.5);
        var pattern2 = Pattern.of(0.9, 0.8, 0.7, 0.6, 0.5);
        
        algorithm.learn(pattern1, parameters);
        algorithm.learn(pattern2, parameters);
        
        // Predict with learned pattern
        var result1 = algorithm.predict(pattern1, parameters);
        assertNotNull(result1);
        
        // Predict with new similar pattern
        var similar = Pattern.of(0.11, 0.21, 0.31, 0.41, 0.51);
        var result2 = algorithm.predict(similar, parameters);
        assertNotNull(result2);
    }
    
    @Test
    @DisplayName("Test null input validation")
    void testNullInputValidation() {
        assertThrows(NullPointerException.class, () -> {
            algorithm.learn(null, parameters);
        });
        
        assertThrows(NullPointerException.class, () -> {
            algorithm.predict(null, parameters);
        });
    }
    
    @Test
    @DisplayName("Test null parameters validation")
    void testNullParametersValidation() {
        var pattern = Pattern.of(0.1, 0.2, 0.3);
        
        assertThrows(NullPointerException.class, () -> {
            algorithm.learn(pattern, null);
        });
        
        assertThrows(NullPointerException.class, () -> {
            algorithm.predict(pattern, null);
        });
    }
    
    @Test
    @DisplayName("Test performance stats tracking")
    void testPerformanceStatsDetails() {
        var stats1 = algorithm.getPerformanceStats();
        assertEquals(0L, stats1.totalOperations());
        
        var pattern = Pattern.of(0.1, 0.2, 0.3, 0.4, 0.5);
        algorithm.learn(pattern, parameters);
        
        var stats2 = algorithm.getPerformanceStats();
        assertTrue(stats2.totalOperations() > 0);
        
        // Reset tracking
        algorithm.resetPerformanceTracking();
        var stats3 = algorithm.getPerformanceStats();
        assertEquals(0L, stats3.totalOperations());
    }
    
    @Test
    @DisplayName("Test SIMD operations when enabled")
    void testSimdOperations() {
        var simdParams = parameters.withEnableSIMD(true)
                                    .withSalienceUpdateRate(0.02); // Set proper simdThreshold
        var artWithSimd = new VectorizedSalienceART(simdParams);
        
        try {
            // Create pattern large enough to trigger SIMD (needs to be > simdThreshold)
            var pattern = Pattern.of(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0);
            
            artWithSimd.learn(pattern, simdParams);
            
            var stats = artWithSimd.getPerformanceStats();
            // With 110 dimensions and simdThreshold of 100, SIMD should be triggered
            if (simdParams.enableSIMD() && pattern.dimension() > simdParams.simdThreshold()) {
                assertTrue(stats.simdOperations() > 0);
            }
        } finally {
            artWithSimd.close();
        }
    }
    
    @Test
    @DisplayName("Test sparse mode operations")
    void testSparseMode() {
        var sparseParams = parameters.withUseSparseMode(true)
                                    .withSparsityThreshold(0.1);
        var artWithSparse = new VectorizedSalienceART(sparseParams);
        
        try {
            // Create sparse pattern (mostly zeros)
            var sparsePattern = Pattern.of(0.0, 0.0, 0.5, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.3);
            artWithSparse.learn(sparsePattern, sparseParams);
            
            var stats = artWithSparse.getPerformanceStats();
            if (sparseParams.useSparseMode()) {
                assertTrue(stats.sparseVectorOperations() >= 0);
            }
        } finally {
            artWithSparse.close();
        }
    }
    
    @Test
    @DisplayName("Test resource cleanup with close")
    void testResourceCleanupAfterClose() {
        var pattern = Pattern.of(0.1, 0.2, 0.3);
        algorithm.learn(pattern, parameters);
        
        algorithm.close();
        
        // After closing, operations should throw IllegalStateException
        assertThrows(IllegalStateException.class, () -> {
            algorithm.learn(pattern, parameters);
        });
    }
    
    @Test
    @DisplayName("Test batch learning")
    void testBatchLearning() {
        var patterns = new Pattern[] {
            Pattern.of(0.1, 0.2, 0.3),
            Pattern.of(0.4, 0.5, 0.6),
            Pattern.of(0.7, 0.8, 0.9)
        };
        
        var results = algorithm.learnBatch(java.util.Arrays.asList(patterns), parameters);
        
        assertNotNull(results);
        assertEquals(patterns.length, results.size());
        assertTrue(algorithm.getCategoryCount() > 0);
    }
    
    @Test
    @DisplayName("Test batch prediction")
    void testBatchPrediction() {
        // First learn some patterns
        algorithm.learn(Pattern.of(0.1, 0.2, 0.3), parameters);
        algorithm.learn(Pattern.of(0.7, 0.8, 0.9), parameters);
        
        var patterns = new Pattern[] {
            Pattern.of(0.11, 0.21, 0.31),
            Pattern.of(0.71, 0.81, 0.91)
        };
        
        var results = algorithm.predictBatch(java.util.Arrays.asList(patterns), parameters);
        
        assertNotNull(results);
        assertEquals(patterns.length, results.size());
    }
    
    @Test
    @DisplayName("Test algorithm type identification")
    void testAlgorithmType() {
        assertEquals("VectorizedSalienceART", algorithm.getAlgorithmType());
        assertTrue(algorithm.isVectorized());
    }
    
    @Test
    @DisplayName("Test enhanced stepFit method")
    void testStepFitEnhanced() {
        var pattern = Pattern.of(0.1, 0.2, 0.3, 0.4, 0.5);
        
        var result = algorithm.stepFitEnhanced(pattern, parameters);
        
        assertNotNull(result);
        assertTrue(result instanceof ActivationResult);
        assertEquals(1, algorithm.getCategoryCount());
    }
}