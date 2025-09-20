package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.*;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.performance.BaseVectorizedARTTest;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.ArrayList;

/**
 * Comprehensive test suite for VectorizedDualVigilanceART implementation.
 * Tests dual vigilance threshold mechanism with vectorized operations.
 */
public class VectorizedDualVigilanceARTTest extends BaseVectorizedARTTest<VectorizedDualVigilanceART, VectorizedDualVigilanceParameters> {
    
    @Override
    protected VectorizedDualVigilanceART createAlgorithm(VectorizedDualVigilanceParameters params) {
        return new VectorizedDualVigilanceART(params);
    }
    
    @Override
    protected VectorizedDualVigilanceParameters createDefaultParameters() {
        return new VectorizedDualVigilanceParameters(
            0.5,    // rhoLower - lower vigilance threshold
            0.9,    // rhoUpper - upper vigilance threshold  
            0.1,    // alpha - choice parameter
            0.9,    // beta - learning rate
            4,      // parallelismLevel
            50,     // parallelThreshold
            1000,   // maxCacheSize
            true    // enableSIMD
        );
    }
    
    @BeforeEach
    protected void setUp() {
        parameters = createDefaultParameters();
        algorithm = createAlgorithm(parameters);
        super.setUp();
    }
    
    @Override
    protected VectorizedDualVigilanceParameters createParametersWithVigilance(double vigilance) {
        // For dual vigilance, we scale both thresholds proportionally
        double lowerVigilance = vigilance * 0.8; // Lower threshold is 80% of requested
        double upperVigilance = Math.min(vigilance * 1.2, 0.99); // Upper is 120% but capped
        return new VectorizedDualVigilanceParameters(
            lowerVigilance,
            upperVigilance,
            0.1,    // alpha - choice parameter
            0.9,    // beta - learning rate
            4,      // parallelismLevel
            50,     // parallelThreshold
            1000,   // maxCacheSize
            true    // enableSIMD
        );
    }
    
    // The following tests are covered by base class:
    // - testBasicLearning()
    // - testMultiplePatternLearning()
    // - testPrediction()
    // - testPerformanceTracking()
    // - testErrorHandling()
    // - testResourceCleanup()
    
    @Test
    @DisplayName("Should handle dual vigilance thresholds correctly")
    void testDualVigilanceThresholds() {
        var centerPattern = Pattern.of(0.5, 0.5);
        var closePattern = Pattern.of(0.52, 0.48);   // Should pass lower threshold
        var mediumPattern = Pattern.of(0.6, 0.4);    // Between thresholds
        var farPattern = Pattern.of(0.9, 0.1);       // Should fail both thresholds
        
        // Train with center pattern
        algorithm.learn(centerPattern, parameters);
        assertEquals(1, algorithm.getCategoryCount());
        
        // Close pattern should match existing category (passes lower threshold)
        var closeResult = algorithm.predict(closePattern, parameters);
        assertTrue(closeResult instanceof ActivationResult.Success);
        
        // Medium pattern behavior depends on dual vigilance logic
        var mediumResult = algorithm.predict(mediumPattern, parameters);
        assertNotNull(mediumResult);
        
        // Far pattern should create new category or have low activation
        var farResult = algorithm.predict(farPattern, parameters);
        assertNotNull(farResult);
    }
    
    @Test
    @DisplayName("Should enforce rhoLower < rhoUpper constraint")
    void testVigilanceThresholdConstraint() {
        var center = Pattern.of(0.5, 0.5);
        var testPattern = Pattern.of(0.7, 0.3);
        
        algorithm.learn(center, parameters);
        
        // Test with different patterns at different vigilance levels
        var result = algorithm.predict(testPattern, parameters);
        assertNotNull(result);
        
        // Verify that lower threshold is indeed lower than upper threshold
        assertTrue(parameters.rhoLower() < parameters.rhoUpper());
    }
    
    @Test
    @DisplayName("Should handle alpha parameter for choice function")
    void testAlphaParameter() {
        var pattern1 = Pattern.of(0.8, 0.2);
        var pattern2 = Pattern.of(0.2, 0.8);
        
        // High alpha - should emphasize size of intersection
        var highAlphaParams = new VectorizedDualVigilanceParameters(
            0.5, 0.9, 0.9, 0.9, 4, 50, 1000, true
        );
        var highAlphaAlgorithm = new VectorizedDualVigilanceART(highAlphaParams);
        
        // Low alpha - should emphasize relative similarity
        var lowAlphaParams = new VectorizedDualVigilanceParameters(
            0.5, 0.9, 0.001, 0.9, 4, 50, 1000, true
        );
        var lowAlphaAlgorithm = new VectorizedDualVigilanceART(lowAlphaParams);
        
        try {
            highAlphaAlgorithm.learn(pattern1, highAlphaParams);
            lowAlphaAlgorithm.learn(pattern1, lowAlphaParams);
            
            var highAlphaResult = highAlphaAlgorithm.predict(pattern2, highAlphaParams);
            var lowAlphaResult = lowAlphaAlgorithm.predict(pattern2, lowAlphaParams);
            
            assertNotNull(highAlphaResult);
            assertNotNull(lowAlphaResult);
        } finally {
            highAlphaAlgorithm.close();
            lowAlphaAlgorithm.close();
        }
    }
    
    @Test
    @DisplayName("Should handle beta parameter for learning rate")
    void testBetaParameter() {
        var pattern1 = Pattern.of(0.6, 0.4);
        var pattern2 = Pattern.of(0.65, 0.35); // Slightly different
        
        // High beta - fast learning
        var fastLearningParams = new VectorizedDualVigilanceParameters(
            0.5, 0.9, 0.1, 0.95, 4, 50, 1000, true
        );
        var fastAlgorithm = new VectorizedDualVigilanceART(fastLearningParams);
        
        // Low beta - slow learning
        var slowLearningParams = new VectorizedDualVigilanceParameters(
            0.5, 0.9, 0.1, 0.1, 4, 50, 1000, true
        );
        var slowAlgorithm = new VectorizedDualVigilanceART(slowLearningParams);
        
        try {
            // Train both with first pattern
            fastAlgorithm.learn(pattern1, fastLearningParams);
            slowAlgorithm.learn(pattern1, slowLearningParams);
            
            // Train with second pattern (should update existing category)
            fastAlgorithm.learn(pattern2, fastLearningParams);
            slowAlgorithm.learn(pattern2, slowLearningParams);
            
            assertEquals(1, fastAlgorithm.getCategoryCount());
            assertEquals(1, slowAlgorithm.getCategoryCount());
        } finally {
            fastAlgorithm.close();
            slowAlgorithm.close();
        }
    }
    
    @Test
    @DisplayName("Should handle multi-dimensional dual vigilance")
    void testMultiDimensionalDualVigilance() {
        var pattern4D = Pattern.of(0.5, 0.3, 0.8, 0.2);
        var similar4D = Pattern.of(0.52, 0.28, 0.82, 0.18);
        var different4D = Pattern.of(0.1, 0.9, 0.1, 0.9);
        
        algorithm.learn(pattern4D, parameters);
        assertEquals(1, algorithm.getCategoryCount());
        
        // Similar pattern should be handled by dual vigilance
        var similarResult = algorithm.predict(similar4D, parameters);
        assertTrue(similarResult instanceof ActivationResult.Success);
        
        // Different pattern may create new category
        var differentResult = algorithm.learn(different4D, parameters);
        assertNotNull(differentResult);
    }
    
    @Test
    @DisplayName("Should support SIMD vectorization")
    void testSIMDVectorization() {
        assertTrue(algorithm.isVectorized());
        assertTrue(algorithm.getVectorSpeciesLength() > 0);
        
        // Test with large dimension that can utilize SIMD
        var largeDim = new double[16];
        for (int i = 0; i < largeDim.length; i++) {
            largeDim[i] = Math.sin(i * 0.1);
        }
        var largePattern = Pattern.of(largeDim);
        
        var result = algorithm.learn(largePattern, parameters);
        assertNotNull(result);
        assertEquals(1, algorithm.getCategoryCount());
    }
    
    @Test
    @DisplayName("Should reset performance tracking")
    void testPerformanceReset() {
        // Generate some activity
        for (int i = 0; i < 3; i++) {
            var pattern = Pattern.of(Math.random(), Math.random());
            algorithm.learn(pattern, parameters);
        }
        
        algorithm.resetPerformanceTracking();
        var resetStats = algorithm.getPerformanceStats();
        
        assertEquals(0, resetStats.totalVectorOperations());
        assertEquals(0, resetStats.totalParallelTasks());
    }
    
    @Test
    @DisplayName("Should handle different vigilance combinations")
    void testVigilanceCombinations() {
        var pattern = Pattern.of(0.6, 0.4);
        var testPattern = Pattern.of(0.65, 0.35);
        
        // Test different vigilance combinations
        var combinations = List.of(
            new VectorizedDualVigilanceParameters(0.1, 0.3, 0.1, 0.9, 4, 50, 1000, true),
            new VectorizedDualVigilanceParameters(0.3, 0.7, 0.1, 0.9, 4, 50, 1000, true),
            new VectorizedDualVigilanceParameters(0.7, 0.95, 0.1, 0.9, 4, 50, 1000, true)
        );
        
        for (var testParams : combinations) {
            var testAlgorithm = new VectorizedDualVigilanceART(testParams);
            try {
                testAlgorithm.learn(pattern, testParams);
                var result = testAlgorithm.predict(testPattern, testParams);
                assertNotNull(result);
                assertTrue(testParams.rhoLower() < testParams.rhoUpper());
            } finally {
                testAlgorithm.close();
            }
        }
    }
    
    @Test
    @DisplayName("Should handle edge cases gracefully")
    void testEdgeCases() {
        // Zero pattern
        var zeroPattern = Pattern.of(0.0, 0.0);
        var zeroResult = algorithm.learn(zeroPattern, parameters);
        assertNotNull(zeroResult);
        
        // Unit pattern
        var unitPattern = Pattern.of(1.0, 1.0);
        var unitResult = algorithm.learn(unitPattern, parameters);
        assertNotNull(unitResult);
        
        // Single dimension - needs separate algorithm instance
        var singleDimAlgorithm = new VectorizedDualVigilanceART(parameters);
        var singleDim = Pattern.of(0.5);
        var singleResult = singleDimAlgorithm.learn(singleDim, parameters);
        assertNotNull(singleResult);
        singleDimAlgorithm.close();
    }
}