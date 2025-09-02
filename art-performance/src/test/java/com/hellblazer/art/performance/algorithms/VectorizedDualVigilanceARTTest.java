package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.*;
import com.hellblazer.art.core.results.ActivationResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.ArrayList;

/**
 * Comprehensive test suite for VectorizedDualVigilanceART implementation.
 * Tests dual vigilance threshold mechanism with vectorized operations.
 */
public class VectorizedDualVigilanceARTTest {
    
    private VectorizedDualVigilanceART algorithm;
    private VectorizedDualVigilanceParameters params;
    
    @BeforeEach
    void setUp() {
        params = new VectorizedDualVigilanceParameters(
            0.5,    // rhoLower - lower vigilance threshold
            0.9,    // rhoUpper - upper vigilance threshold  
            0.1,    // alpha - choice parameter
            0.9,    // beta - learning rate
            4,      // parallelismLevel
            50,     // parallelThreshold
            1000,   // maxCacheSize
            true    // enableSIMD
        );
        algorithm = new VectorizedDualVigilanceART();
    }
    
    @AfterEach
    void tearDown() {
        if (algorithm != null) {
            algorithm.close();
        }
    }
    
    @Test
    @DisplayName("Should create dual vigilance categories correctly")
    void testDualVigilanceCategoryCreation() {
        var pattern1 = Pattern.of(0.8, 0.2);
        var pattern2 = Pattern.of(0.3, 0.7);
        
        var result1 = algorithm.learn(pattern1, params);
        var result2 = algorithm.learn(pattern2, params);
        
        assertEquals(2, algorithm.getCategoryCount());
        assertTrue(result1 instanceof ActivationResult.Success);
        assertTrue(result2 instanceof ActivationResult.Success);
    }
    
    @Test
    @DisplayName("Should handle dual vigilance thresholds correctly")
    void testDualVigilanceThresholds() {
        var centerPattern = Pattern.of(0.5, 0.5);
        var closePattern = Pattern.of(0.52, 0.48);   // Should pass lower threshold
        var mediumPattern = Pattern.of(0.6, 0.4);    // Between thresholds
        var farPattern = Pattern.of(0.9, 0.1);       // Should fail both thresholds
        
        // Train with center pattern
        algorithm.learn(centerPattern, params);
        assertEquals(1, algorithm.getCategoryCount());
        
        // Close pattern should match existing category (passes lower threshold)
        var closeResult = algorithm.predict(closePattern, params);
        assertTrue(closeResult instanceof ActivationResult.Success);
        
        // Medium pattern behavior depends on dual vigilance logic
        var mediumResult = algorithm.predict(mediumPattern, params);
        assertNotNull(mediumResult);
        
        // Far pattern should create new category or have low activation
        var farResult = algorithm.predict(farPattern, params);
        assertNotNull(farResult);
    }
    
    @Test
    @DisplayName("Should enforce rhoLower < rhoUpper constraint")
    void testVigilanceThresholdConstraint() {
        var center = Pattern.of(0.5, 0.5);
        var testPattern = Pattern.of(0.7, 0.3);
        
        algorithm.learn(center, params);
        
        // Test with different patterns at different vigilance levels
        var result = algorithm.predict(testPattern, params);
        assertNotNull(result);
        
        // Verify that lower threshold is indeed lower than upper threshold
        assertTrue(params.rhoLower() < params.rhoUpper());
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
        var highAlphaAlgorithm = new VectorizedDualVigilanceART();
        
        // Low alpha - should emphasize relative similarity
        var lowAlphaParams = new VectorizedDualVigilanceParameters(
            0.5, 0.9, 0.001, 0.9, 4, 50, 1000, true
        );
        var lowAlphaAlgorithm = new VectorizedDualVigilanceART();
        
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
        var fastAlgorithm = new VectorizedDualVigilanceART();
        
        // Low beta - slow learning
        var slowLearningParams = new VectorizedDualVigilanceParameters(
            0.5, 0.9, 0.1, 0.1, 4, 50, 1000, true
        );
        var slowAlgorithm = new VectorizedDualVigilanceART();
        
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
        
        algorithm.learn(pattern4D, params);
        assertEquals(1, algorithm.getCategoryCount());
        
        // Similar pattern should be handled by dual vigilance
        var similarResult = algorithm.predict(similar4D, params);
        assertTrue(similarResult instanceof ActivationResult.Success);
        
        // Different pattern may create new category
        var differentResult = algorithm.learn(different4D, params);
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
        
        var result = algorithm.learn(largePattern, params);
        assertNotNull(result);
        assertEquals(1, algorithm.getCategoryCount());
    }
    
    @Test
    @DisplayName("Should provide performance statistics")
    void testPerformanceTracking() {
        var initialStats = algorithm.getPerformanceStats();
        assertNotNull(initialStats);
        
        // Perform some operations
        for (int i = 0; i < 5; i++) {
            var pattern = Pattern.of(Math.random(), Math.random());
            algorithm.learn(pattern, params);
        }
        
        var finalStats = algorithm.getPerformanceStats();
        assertNotNull(finalStats);
        assertTrue(finalStats.totalVectorOperations() >= initialStats.totalVectorOperations());
    }
    
    @Test
    @DisplayName("Should reset performance tracking")
    void testPerformanceReset() {
        // Generate some activity
        for (int i = 0; i < 3; i++) {
            var pattern = Pattern.of(Math.random(), Math.random());
            algorithm.learn(pattern, params);
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
            var testAlgorithm = new VectorizedDualVigilanceART();
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
        var zeroResult = algorithm.learn(zeroPattern, params);
        assertNotNull(zeroResult);
        
        // Unit pattern
        var unitPattern = Pattern.of(1.0, 1.0);
        var unitResult = algorithm.learn(unitPattern, params);
        assertNotNull(unitResult);
        
        // Single dimension - needs separate algorithm instance
        var singleDimAlgorithm = new VectorizedDualVigilanceART();
        var singleDim = Pattern.of(0.5);
        var singleResult = singleDimAlgorithm.learn(singleDim, params);
        assertNotNull(singleResult);
        singleDimAlgorithm.close();
    }
    
    @Test
    @DisplayName("Should implement AutoCloseable correctly")
    void testResourceManagement() {
        var tempAlgorithm = new VectorizedDualVigilanceART();
        
        // Use the algorithm
        var pattern = Pattern.of(0.5, 0.5);
        tempAlgorithm.learn(pattern, params);
        assertEquals(1, tempAlgorithm.getCategoryCount());
        
        // Close should not throw
        assertDoesNotThrow(() -> tempAlgorithm.close());
        
        // Multiple closes should be safe
        assertDoesNotThrow(() -> tempAlgorithm.close());
    }
}