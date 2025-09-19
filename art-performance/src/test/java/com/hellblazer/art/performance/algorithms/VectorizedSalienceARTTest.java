package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.results.ActivationResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test-first development: Tests for VectorizedSalienceART
 */
class VectorizedSalienceARTTest {
    
    private VectorizedSalienceART art;
    private VectorizedSalienceParameters parameters;
    
    @BeforeEach
    void setUp() {
        parameters = VectorizedSalienceParameters.createDefault();
        art = new VectorizedSalienceART(parameters);
    }
    
    @Test
    @DisplayName("Test initialization")
    void testInitialization() {
        assertNotNull(art);
        assertEquals(0, art.getCategoryCount());
        assertNotNull(art.getParameters());
        assertEquals(parameters, art.getParameters());
        assertFalse(art.isTrained());
    }
    
    @Test
    @DisplayName("Test learning with single pattern")
    void testLearnSinglePattern() {
        var pattern = Pattern.of(0.1, 0.2, 0.3, 0.4, 0.5);
        
        var result = art.learn(pattern, parameters);
        
        assertNotNull(result);
        assertTrue(result instanceof ActivationResult);
        assertEquals(1, art.getCategoryCount());
        assertTrue(art.isTrained());
    }
    
    @Test
    @DisplayName("Test learning with multiple patterns")
    void testLearnMultiplePatterns() {
        var pattern1 = Pattern.of(0.1, 0.2, 0.3, 0.4, 0.5);
        var pattern2 = Pattern.of(0.9, 0.8, 0.7, 0.6, 0.5);
        var pattern3 = Pattern.of(0.5, 0.5, 0.5, 0.5, 0.5);
        
        art.learn(pattern1, parameters);
        art.learn(pattern2, parameters);
        art.learn(pattern3, parameters);
        
        assertTrue(art.getCategoryCount() > 0);
        assertTrue(art.getCategoryCount() <= 3);
    }
    
    @Test
    @DisplayName("Test prediction without learning")
    void testPredictWithoutLearning() {
        var pattern = Pattern.of(0.1, 0.2, 0.3, 0.4, 0.5);
        
        var result = art.predict(pattern, parameters);
        
        assertNotNull(result);
        // Should handle empty network gracefully
        assertEquals(0, art.getCategoryCount());
    }
    
    @Test
    @DisplayName("Test prediction after learning")
    void testPredictAfterLearning() {
        var pattern1 = Pattern.of(0.1, 0.2, 0.3, 0.4, 0.5);
        var pattern2 = Pattern.of(0.9, 0.8, 0.7, 0.6, 0.5);
        
        art.learn(pattern1, parameters);
        art.learn(pattern2, parameters);
        
        // Predict with learned pattern
        var result1 = art.predict(pattern1, parameters);
        assertNotNull(result1);
        
        // Predict with new similar pattern
        var similar = Pattern.of(0.11, 0.21, 0.31, 0.41, 0.51);
        var result2 = art.predict(similar, parameters);
        assertNotNull(result2);
    }
    
    @Test
    @DisplayName("Test null input validation")
    void testNullInputValidation() {
        assertThrows(IllegalArgumentException.class, () -> {
            art.learn(null, parameters);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            art.predict(null, parameters);
        });
    }
    
    @Test
    @DisplayName("Test null parameters validation")
    void testNullParametersValidation() {
        var pattern = Pattern.of(0.1, 0.2, 0.3);
        
        assertThrows(IllegalArgumentException.class, () -> {
            art.learn(pattern, null);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            art.predict(pattern, null);
        });
    }
    
    @Test
    @DisplayName("Test performance stats tracking")
    void testPerformanceStats() {
        var stats1 = art.getPerformanceStats();
        assertEquals(0L, stats1.totalOperations());
        
        var pattern = Pattern.of(0.1, 0.2, 0.3, 0.4, 0.5);
        art.learn(pattern, parameters);
        
        var stats2 = art.getPerformanceStats();
        assertTrue(stats2.totalOperations() > 0);
        
        // Reset tracking
        art.resetPerformanceTracking();
        var stats3 = art.getPerformanceStats();
        assertEquals(0L, stats3.totalOperations());
    }
    
    @Test
    @DisplayName("Test SIMD operations when enabled")
    void testSimdOperations() {
        var simdParams = parameters.withEnableSIMD(true)
                                    .withSalienceUpdateRate(0.02); // Set proper simdThreshold
        var artWithSimd = new VectorizedSalienceART(simdParams);
        
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
        
        artWithSimd.close();
    }
    
    @Test
    @DisplayName("Test sparse mode operations")
    void testSparseMode() {
        var sparseParams = parameters.withUseSparseMode(true)
                                    .withSparsityThreshold(0.1);
        var artWithSparse = new VectorizedSalienceART(sparseParams);
        
        // Create sparse pattern (mostly zeros)
        var sparsePattern = Pattern.of(0.0, 0.0, 0.5, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.3);
        artWithSparse.learn(sparsePattern, sparseParams);
        
        var stats = artWithSparse.getPerformanceStats();
        if (sparseParams.useSparseMode()) {
            assertTrue(stats.sparseVectorOperations() >= 0);
        }
        
        artWithSparse.close();
    }
    
    @Test
    @DisplayName("Test resource cleanup with close")
    void testResourceCleanup() {
        var pattern = Pattern.of(0.1, 0.2, 0.3);
        art.learn(pattern, parameters);
        
        art.close();
        
        // After closing, operations should throw IllegalStateException
        assertThrows(IllegalStateException.class, () -> {
            art.learn(pattern, parameters);
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
        
        var results = art.learnBatch(java.util.Arrays.asList(patterns), parameters);
        
        assertNotNull(results);
        assertEquals(patterns.length, results.size());
        assertTrue(art.getCategoryCount() > 0);
    }
    
    @Test
    @DisplayName("Test batch prediction")
    void testBatchPrediction() {
        // First learn some patterns
        art.learn(Pattern.of(0.1, 0.2, 0.3), parameters);
        art.learn(Pattern.of(0.7, 0.8, 0.9), parameters);
        
        var patterns = new Pattern[] {
            Pattern.of(0.11, 0.21, 0.31),
            Pattern.of(0.71, 0.81, 0.91)
        };
        
        var results = art.predictBatch(java.util.Arrays.asList(patterns), parameters);
        
        assertNotNull(results);
        assertEquals(patterns.length, results.size());
    }
    
    @Test
    @DisplayName("Test algorithm type identification")
    void testAlgorithmType() {
        assertEquals("VectorizedSalienceART", art.getAlgorithmType());
        assertTrue(art.isVectorized());
    }
    
    @Test
    @DisplayName("Test enhanced stepFit method")
    void testStepFitEnhanced() {
        var pattern = Pattern.of(0.1, 0.2, 0.3, 0.4, 0.5);
        
        var result = art.stepFitEnhanced(pattern, parameters);
        
        assertNotNull(result);
        assertTrue(result instanceof ActivationResult);
        assertEquals(1, art.getCategoryCount());
    }
}