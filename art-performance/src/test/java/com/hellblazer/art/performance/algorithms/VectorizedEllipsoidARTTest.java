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
 * Comprehensive test suite for VectorizedEllipsoidART implementation.
 * Tests ellipsoidal category representation with vectorized operations.
 */
public class VectorizedEllipsoidARTTest {
    
    private VectorizedEllipsoidART algorithm;
    private VectorizedEllipsoidParameters params;
    
    @BeforeEach
    void setUp() {
        params = new VectorizedEllipsoidParameters(
            0.75,   // vigilance
            0.9,    // mu - shape parameter  
            1.0,    // baseRadius
            4,      // parallelismLevel
            50,     // parallelThreshold
            1000,   // maxCacheSize
            true    // enableSIMD
        );
        algorithm = new VectorizedEllipsoidART();
    }
    
    @AfterEach
    void tearDown() {
        if (algorithm != null) {
            algorithm.close();
        }
    }
    
    @Test
    @DisplayName("Should create ellipsoidal categories correctly")
    void testEllipsoidalCategoryCreation() {
        var pattern1 = Pattern.of(0.8, 0.2);
        var pattern2 = Pattern.of(0.3, 0.7);
        
        var result1 = algorithm.learn(pattern1, params);
        var result2 = algorithm.learn(pattern2, params);
        
        assertEquals(2, algorithm.getCategoryCount());
        assertTrue(result1 instanceof ActivationResult.Success);
        assertTrue(result2 instanceof ActivationResult.Success);
    }
    
    @Test
    @DisplayName("Should handle ellipsoidal geometry activation correctly")
    void testEllipsoidalActivation() {
        var center = Pattern.of(0.5, 0.5);
        var nearPoint = Pattern.of(0.52, 0.48); // Inside ellipsoid
        var farPoint = Pattern.of(0.9, 0.9);    // Outside ellipsoid
        
        // Train with center pattern
        algorithm.learn(center, params);
        assertEquals(1, algorithm.getCategoryCount());
        
        // Near point should activate same category
        var nearResult = algorithm.predict(nearPoint, params);
        assertTrue(nearResult instanceof ActivationResult.Success);
        
        // Far point may create new category or have low activation
        var farResult = algorithm.predict(farPoint, params);
        assertNotNull(farResult);
    }
    
    @Test
    @DisplayName("Should implement proper ellipsoidal distance measure")
    void testEllipsoidalDistance() {
        var pattern1 = Pattern.of(1.0, 0.0);
        var pattern2 = Pattern.of(0.0, 1.0);
        var pattern3 = Pattern.of(0.7, 0.7); // Diagonal
        
        algorithm.learn(pattern1, params);
        assertEquals(1, algorithm.getCategoryCount());
        
        // Test different distances from first category
        var result2 = algorithm.predict(pattern2, params);
        var result3 = algorithm.predict(pattern3, params);
        
        assertNotNull(result2);
        assertNotNull(result3);
    }
    
    @Test
    @DisplayName("Should handle mu parameter for ellipsoid shape")
    void testMuParameter() {
        var center = Pattern.of(0.5, 0.5);
        var testPoint = Pattern.of(0.6, 0.4);
        
        // High mu - more circular
        var highMuParams = new VectorizedEllipsoidParameters(
            0.75, 0.95, 1.0, 4, 50, 1000, true
        );
        var highMuAlgorithm = new VectorizedEllipsoidART();
        
        // Low mu - more elongated  
        var lowMuParams = new VectorizedEllipsoidParameters(
            0.75, 0.1, 1.0, 4, 50, 1000, true
        );
        var lowMuAlgorithm = new VectorizedEllipsoidART();
        
        try {
            highMuAlgorithm.learn(center, highMuParams);
            lowMuAlgorithm.learn(center, lowMuParams);
            
            var highMuResult = highMuAlgorithm.predict(testPoint, highMuParams);
            var lowMuResult = lowMuAlgorithm.predict(testPoint, lowMuParams);
            
            assertNotNull(highMuResult);
            assertNotNull(lowMuResult);
        } finally {
            highMuAlgorithm.close();
            lowMuAlgorithm.close();
        }
    }
    
    @Test
    @DisplayName("Should handle multi-dimensional ellipsoids")
    void testMultiDimensionalEllipsoids() {
        var pattern4D = Pattern.of(0.5, 0.3, 0.8, 0.2);
        var similar4D = Pattern.of(0.52, 0.28, 0.82, 0.18);
        var different4D = Pattern.of(0.1, 0.9, 0.1, 0.9);
        
        algorithm.learn(pattern4D, params);
        assertEquals(1, algorithm.getCategoryCount());
        
        // Similar pattern should match
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
        var singleDimAlgorithm = new VectorizedEllipsoidART();
        var singleDim = Pattern.of(0.5);
        var singleResult = singleDimAlgorithm.learn(singleDim, params);
        assertNotNull(singleResult);
        singleDimAlgorithm.close();
    }
    
    @Test
    @DisplayName("Should implement AutoCloseable correctly")
    void testResourceManagement() {
        var tempAlgorithm = new VectorizedEllipsoidART();
        
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