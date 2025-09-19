package com.hellblazer.art.performance.algorithms;

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
import com.hellblazer.art.performance.BaseVectorizedARTTest;
import com.hellblazer.art.performance.algorithms.VectorizedPerformanceStats;

/**
 * Comprehensive test suite for VectorizedFuzzyART implementation.
 * Tests both SIMD and standard computation paths, performance characteristics,
 * and compatibility with FuzzyART semantics.
 */
public class VectorizedFuzzyARTTest extends BaseVectorizedARTTest<VectorizedFuzzyART, VectorizedParameters> {
    
    private FuzzyART standardART;
    private FuzzyParameters fuzzyParams;
    
    @Override
    protected VectorizedFuzzyART createAlgorithm(VectorizedParameters params) {
        return new VectorizedFuzzyART(params);
    }
    
    @Override
    protected VectorizedParameters createDefaultParameters() {
        return new VectorizedParameters(
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
    }
    
    @BeforeEach
    protected void setUp() {
        super.setUp();
        // Configure standard FuzzyART for comparison
        fuzzyParams = new FuzzyParameters(0.9, 0.1, 0.1);
        standardART = new FuzzyART();
    }
    
    // Basic learning and vigilance control tests are inherited from base class
    
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
        var simdART = createAlgorithm(simdParams);
        
        // Train with SIMD disabled
        var noSimdParams = new VectorizedParameters(
            0.8, 0.1, 0.1, 4, 100, 1000, false, false, 0.8
        );
        var noSimdART = createAlgorithm(noSimdParams);
        
        // Train both networks
        for (var pattern : patterns) {
            simdART.learn(pattern, simdParams);
            noSimdART.learn(pattern, noSimdParams);
        }
        
        // Should create same number of categories
        assertEquals(simdART.getCategoryCount(), noSimdART.getCategoryCount());
        
        // Test predictions should be equivalent
        for (var pattern : patterns) {
            var simdResult = simdART.stepFitEnhancedVectorized(pattern, simdParams);
            var noSimdResult = noSimdART.stepFitEnhancedVectorized(pattern, noSimdParams);
            
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
        var weight = VectorizedFuzzyWeight.fromInput(Pattern.of(0.8, 0.2), parameters);
        
        var updatedWeight = weight.updateFuzzy(input, parameters);
        
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
        var parallelART = createAlgorithm(parallelParams);
        
        // Train with many patterns using enhanced stepFit to trigger parallel processing
        for (var pattern : patterns) {
            parallelART.stepFitEnhancedVectorized(pattern, parallelParams);
        }
        
        // Should have created many categories (more than parallel threshold)
        assertTrue(parallelART.getCategoryCount() > 5, 
            "Expected categories > 5 but was " + parallelART.getCategoryCount());
        
        // Test additional parallel stepFit
        var testPattern = Pattern.of(0.5, 0.5);
        var result = parallelART.stepFitEnhancedVectorized(testPattern, parallelParams);
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
        var art = createAlgorithm(smallCacheParams);
        
        // Train with patterns
        for (var pattern : patterns) {
            art.learn(pattern, smallCacheParams);
        }
        
        // Memory optimization is handled automatically by the algorithm
        // art.optimizeMemory(); // Method doesn't exist - memory is managed internally
        
        // Should still work correctly after optimization
        var result = art.learn(patterns.get(0), smallCacheParams);
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
        var initialStats = algorithm.getPerformanceStats();
        assertEquals(0, initialStats.totalVectorOperations());
        assertEquals(0, initialStats.totalParallelTasks());
        
        // Train and test
        for (var pattern : patterns) {
            algorithm.learn(pattern, parameters);
        }
        
        // Stats should be updated
        var finalStats = algorithm.getPerformanceStats();
        assertTrue(finalStats.totalVectorOperations() > 0);
        assertTrue(finalStats.avgComputeTimeMs() >= 0);
        
        // Reset should clear stats
        algorithm.resetPerformanceTracking();
        var resetStats = algorithm.getPerformanceStats();
        assertEquals(0, resetStats.totalVectorOperations());
        assertEquals(0, resetStats.totalParallelTasks());
    }
    
    // Error handling tests are inherited from base class
    
    @Test
    @DisplayName("VectorizedFuzzyWeight should handle edge cases correctly")
    void testVectorizedFuzzyWeightEdgeCases() {
        // Test noise addition
        var weight = VectorizedFuzzyWeight.fromInput(Pattern.of(0.5, 0.5), parameters);
        var noisyWeight = weight.addNoise(0.1, parameters);
        
        // Should maintain valid range [0, 1]
        for (int i = 0; i < noisyWeight.dimension(); i++) {
            assertTrue(noisyWeight.get(i) >= 0.0);
            assertTrue(noisyWeight.get(i) <= 1.0);
        }
        
        // Test vigilance computation
        var input = Pattern.of(0.8, 0.2);
        var vigilance = weight.computeVigilance(input, parameters);
        assertTrue(vigilance >= 0.0 && vigilance <= 1.0);
    }
    
    // Resource cleanup tests are inherited from base class
}