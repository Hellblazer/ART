package com.hellblazer.art.algorithms;

import com.hellblazer.art.core.*;
import com.hellblazer.art.core.parameters.FuzzyParameters;
import com.hellblazer.art.core.algorithms.FuzzyART;
import com.hellblazer.art.core.results.ActivationResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.ArrayList;

/**
 * Comprehensive test suite for VectorizedFuzzyART implementation.
 * Tests both SIMD and standard computation paths, performance characteristics,
 * and compatibility with FuzzyART semantics.
 */
public class VectorizedFuzzyARTTest {
    
    private VectorizedFuzzyART vectorizedART;
    private VectorizedParameters params;
    private FuzzyART standardART;
    private FuzzyParameters fuzzyParams;
    
    @BeforeEach
    void setUp() {
        // Configure vectorized parameters
        params = new VectorizedParameters(
            0.9,    // vigilanceThreshold
            0.1,    // learningRate
            0.1,    // alpha
            4,      // parallelismLevel
            100,    // parallelThreshold
            1000,   // maxCacheSize
            true,   // enableSIMD
            false,  // enableJOML (not used for FuzzyART)
            0.8     // memoryOptimizationThreshold
        );
        
        vectorizedART = new VectorizedFuzzyART(params);
        
        // Configure standard FuzzyART for comparison
        fuzzyParams = new FuzzyParameters(0.9, 0.1, 0.1);
        standardART = new FuzzyART();
    }
    
    @Test
    @DisplayName("Basic learning and recognition should work correctly")
    void testBasicLearningAndRecognition() {
        // Create simple 2D patterns
        var pattern1 = Pattern.of(0.8, 0.2);
        var pattern2 = Pattern.of(0.3, 0.7);
        var pattern3 = Pattern.of(0.8, 0.3); // Similar to pattern1
        
        // Train on first two patterns
        var result1 = vectorizedART.stepFit(pattern1, params);
        var result2 = vectorizedART.stepFit(pattern2, params);
        
        // Should create two categories
        assertEquals(2, vectorizedART.getCategoryCount());
        assertTrue(result1 instanceof ActivationResult.Success);
        assertTrue(result2 instanceof ActivationResult.Success);
        
        // Test recognition of similar pattern
        var result3 = vectorizedART.stepFit(pattern3, params);
        assertTrue(result3 instanceof ActivationResult.Success);
        
        var successResult3 = (ActivationResult.Success) result3;
        // Should match first category (category 0) due to similarity
        assertEquals(0, successResult3.categoryIndex());
    }
    
    @Test
    @DisplayName("Vigilance parameter should control category creation")
    void testVigilanceControl() {
        var pattern1 = Pattern.of(0.8, 0.2);
        var pattern2 = Pattern.of(0.7, 0.3); // Moderately similar
        
        // High vigilance - should create separate categories
        var highVigilanceParams = new VectorizedParameters(
            0.95, 0.1, 0.1, 4, 100, 1000, true, false, 0.8
        );
        var highVigilanceART = new VectorizedFuzzyART(highVigilanceParams);
        
        highVigilanceART.stepFit(pattern1, highVigilanceParams);
        highVigilanceART.stepFit(pattern2, highVigilanceParams);
        
        assertEquals(2, highVigilanceART.getCategoryCount());
        
        // Low vigilance - should merge into one category
        var lowVigilanceParams = new VectorizedParameters(
            0.5, 0.1, 0.1, 4, 100, 1000, true, false, 0.8
        );
        var lowVigilanceART = new VectorizedFuzzyART(lowVigilanceParams);
        
        lowVigilanceART.stepFit(pattern1, lowVigilanceParams);
        lowVigilanceART.stepFit(pattern2, lowVigilanceParams);
        
        assertEquals(1, lowVigilanceART.getCategoryCount());
        
        highVigilanceART.close();
        lowVigilanceART.close();
    }
    
    @Test
    @DisplayName("SIMD and standard computation should produce equivalent results")
    void testSIMDEquivalence() {
        var patterns = List.of(
            Pattern.of(0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4),
            Pattern.of(0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6),
            Pattern.of(0.8, 0.2, 0.9, 0.1, 0.6, 0.4, 0.7, 0.3),
            Pattern.of(0.2, 0.8, 0.1, 0.9, 0.4, 0.6, 0.3, 0.7)
        );
        
        // Train with SIMD enabled
        var simdParams = new VectorizedParameters(
            0.8, 0.1, 0.1, 4, 100, 1000, true, false, 0.8
        );
        var simdART = new VectorizedFuzzyART(simdParams);
        
        // Train with SIMD disabled
        var noSimdParams = new VectorizedParameters(
            0.8, 0.1, 0.1, 4, 100, 1000, false, false, 0.8
        );
        var noSimdART = new VectorizedFuzzyART(noSimdParams);
        
        // Train both networks
        for (var pattern : patterns) {
            simdART.stepFit(pattern, simdParams);
            noSimdART.stepFit(pattern, noSimdParams);
        }
        
        // Should create same number of categories
        assertEquals(simdART.getCategoryCount(), noSimdART.getCategoryCount());
        
        // Test predictions should be equivalent
        for (var pattern : patterns) {
            var simdResult = simdART.stepFitEnhanced(pattern, simdParams);
            var noSimdResult = noSimdART.stepFitEnhanced(pattern, noSimdParams);
            
            assertTrue(simdResult instanceof ActivationResult.Success);
            assertTrue(noSimdResult instanceof ActivationResult.Success);
            
            var simdSuccess = (ActivationResult.Success) simdResult;
            var noSimdSuccess = (ActivationResult.Success) noSimdResult;
            
            assertEquals(simdSuccess.categoryIndex(), noSimdSuccess.categoryIndex());
            assertEquals(simdSuccess.activationValue(), noSimdSuccess.activationValue(), 1e-6);
        }
        
        simdART.close();
        noSimdART.close();
    }
    
    @Test
    @DisplayName("Complement coding should be applied correctly")
    void testComplementCoding() {
        var input = Pattern.of(0.8, 0.3);
        var complementCoded = VectorizedFuzzyWeight.getComplementCoded(input);
        
        // Should double the dimension
        assertEquals(4, complementCoded.dimension());
        
        // Original values should be preserved
        assertEquals(0.8, complementCoded.get(0), 1e-10);
        assertEquals(0.3, complementCoded.get(1), 1e-10);
        
        // Complement values should be 1 - original
        assertEquals(0.2, complementCoded.get(2), 1e-10);
        assertEquals(0.7, complementCoded.get(3), 1e-10);
    }
    
    @Test
    @DisplayName("Weight updates should follow fuzzy min learning rule")
    void testFuzzyMinLearning() {
        var input = Pattern.of(0.6, 0.4);
        var weight = VectorizedFuzzyWeight.fromInput(Pattern.of(0.8, 0.2), params);
        
        var updatedWeight = weight.updateFuzzy(input, params);
        
        // Complement-coded input: [0.6, 0.4, 0.4, 0.6]
        // Complement-coded weight: [0.8, 0.2, 0.2, 0.8]
        // Fuzzy min: [0.6, 0.2, 0.2, 0.6]
        // New weight = 0.1 * [0.6, 0.2, 0.2, 0.6] + 0.9 * [0.8, 0.2, 0.2, 0.8]
        //            = [0.78, 0.2, 0.2, 0.78]
        
        assertEquals(0.78, updatedWeight.get(0), 1e-6);
        assertEquals(0.2, updatedWeight.get(1), 1e-6);
        assertEquals(0.2, updatedWeight.get(2), 1e-6);
        assertEquals(0.78, updatedWeight.get(3), 1e-6);
    }
    
    @Test
    @DisplayName("Parallel processing should work for large category sets")
    void testParallelProcessing() {
        // Create many patterns to trigger parallel processing
        var patterns = new ArrayList<Pattern>();
        for (int i = 0; i < 150; i++) {
            double x = i / 150.0;
            double y = (i * 37 % 150) / 150.0; // Pseudo-random second dimension
            patterns.add(Pattern.of(x, y));
        }
        
        // Set high vigilance to create more categories and low parallel threshold
        var parallelParams = new VectorizedParameters(
            0.95, 0.1, 0.1, 4, 5, 1000, true, false, 0.8
        );
        var parallelART = new VectorizedFuzzyART(parallelParams);
        
        // Train with many patterns using enhanced stepFit to trigger parallel processing
        for (var pattern : patterns) {
            parallelART.stepFitEnhanced(pattern, parallelParams);
        }
        
        // Should have created many categories (more than parallel threshold)
        assertTrue(parallelART.getCategoryCount() > 5, 
            "Expected categories > 5 but was " + parallelART.getCategoryCount());
        
        // Test additional parallel stepFit
        var testPattern = Pattern.of(0.5, 0.5);
        var result = parallelART.stepFitEnhanced(testPattern, parallelParams);
        assertTrue(result instanceof ActivationResult.Success);
        
        // Check performance stats - should show parallel tasks were executed
        var stats = parallelART.getPerformanceStats();
        assertTrue(stats.totalParallelTasks() > 0, 
            "Expected parallel tasks > 0 but was " + stats.totalParallelTasks());
        
        parallelART.close();
    }
    
    @Test
    @DisplayName("Memory optimization should work correctly")
    void testMemoryOptimization() {
        var patterns = List.of(
            Pattern.of(0.1, 0.2, 0.3, 0.4),
            Pattern.of(0.5, 0.6, 0.7, 0.8),
            Pattern.of(0.9, 0.8, 0.7, 0.6)
        );
        
        // Set small cache size to trigger optimization
        var smallCacheParams = new VectorizedParameters(
            0.8, 0.1, 0.1, 4, 100, 2, true, false, 0.8
        );
        var art = new VectorizedFuzzyART(smallCacheParams);
        
        // Train with patterns
        for (var pattern : patterns) {
            art.stepFit(pattern, smallCacheParams);
        }
        
        // Force memory optimization
        art.optimizeMemory();
        
        // Should still work correctly after optimization
        var result = art.stepFit(patterns.get(0), smallCacheParams);
        assertTrue(result instanceof ActivationResult.Success);
        
        art.close();
    }
    
    @Test
    @DisplayName("Performance statistics should be tracked correctly")
    void testPerformanceTracking() {
        var patterns = List.of(
            Pattern.of(0.8, 0.2),
            Pattern.of(0.3, 0.7),
            Pattern.of(0.6, 0.4)
        );
        
        // Initial stats should be zero
        var initialStats = vectorizedART.getPerformanceStats();
        assertEquals(0, initialStats.totalVectorOperations());
        assertEquals(0, initialStats.totalParallelTasks());
        
        // Train and test
        for (var pattern : patterns) {
            vectorizedART.stepFit(pattern, params);
        }
        
        // Stats should be updated
        var finalStats = vectorizedART.getPerformanceStats();
        assertTrue(finalStats.totalVectorOperations() > 0);
        assertTrue(finalStats.avgComputeTimeMs() >= 0);
        
        // Reset should clear stats
        vectorizedART.resetPerformanceTracking();
        var resetStats = vectorizedART.getPerformanceStats();
        assertEquals(0, resetStats.totalVectorOperations());
        assertEquals(0, resetStats.totalParallelTasks());
    }
    
    @Test
    @DisplayName("Error handling should work correctly")
    void testErrorHandling() {
        // Null parameters should throw exception
        assertThrows(NullPointerException.class, () -> {
            vectorizedART.stepFit(Pattern.of(0.5, 0.5), null);
        });
        
        // Wrong parameter type should throw exception
        assertThrows(IllegalArgumentException.class, () -> {
            vectorizedART.stepFit(Pattern.of(0.5, 0.5), "wrong type");
        });
        
        // Null input should throw exception
        assertThrows(NullPointerException.class, () -> {
            vectorizedART.stepFit(null, params);
        });
    }
    
    @Test
    @DisplayName("VectorizedFuzzyWeight should handle edge cases correctly")
    void testVectorizedFuzzyWeightEdgeCases() {
        // Test noise addition
        var weight = VectorizedFuzzyWeight.fromInput(Pattern.of(0.5, 0.5), params);
        var noisyWeight = weight.addNoise(0.1, params);
        
        // Should maintain valid range [0, 1]
        for (int i = 0; i < noisyWeight.dimension(); i++) {
            assertTrue(noisyWeight.get(i) >= 0.0);
            assertTrue(noisyWeight.get(i) <= 1.0);
        }
        
        // Test vigilance computation
        var input = Pattern.of(0.8, 0.2);
        var vigilance = weight.computeVigilance(input, params);
        assertTrue(vigilance >= 0.0 && vigilance <= 1.0);
    }
    
    @Test
    @DisplayName("Resource cleanup should work correctly")
    void testResourceCleanup() {
        var art = new VectorizedFuzzyART(params);
        
        // Use the ART network
        art.stepFit(Pattern.of(0.5, 0.5), params);
        
        // Close should not throw exception
        assertDoesNotThrow(() -> art.close());
        
        // toString should work even after close
        assertNotNull(art.toString());
    }
}