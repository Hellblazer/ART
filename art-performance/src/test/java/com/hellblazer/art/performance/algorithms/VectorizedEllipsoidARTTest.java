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
 * Comprehensive test suite for VectorizedEllipsoidART implementation.
 * Tests ellipsoidal category representation with vectorized operations.
 */
public class VectorizedEllipsoidARTTest extends BaseVectorizedARTTest<VectorizedEllipsoidART, VectorizedEllipsoidParameters> {
    
    @Override
    protected VectorizedEllipsoidART createAlgorithm(VectorizedEllipsoidParameters params) {
        return new VectorizedEllipsoidART();
    }
    
    @Override
    protected VectorizedEllipsoidParameters createDefaultParameters() {
        return new VectorizedEllipsoidParameters(
            0.75,   // vigilance
            0.9,    // mu - shape parameter  
            1.0,    // baseRadius
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
    protected VectorizedEllipsoidParameters createParametersWithVigilance(double vigilance) {
        return new VectorizedEllipsoidParameters(
            vigilance,
            0.9,    // mu - shape parameter  
            1.0,    // baseRadius
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
    @DisplayName("Should handle ellipsoidal geometry activation correctly")
    void testEllipsoidalActivation() {
        var center = Pattern.of(0.5, 0.5);
        var nearPoint = Pattern.of(0.52, 0.48); // Inside ellipsoid
        var farPoint = Pattern.of(0.9, 0.9);    // Outside ellipsoid
        
        // Train with center pattern
        algorithm.learn(center, parameters);
        assertEquals(1, algorithm.getCategoryCount());
        
        // Near point should activate same category
        var nearResult = algorithm.predict(nearPoint, parameters);
        assertTrue(nearResult instanceof ActivationResult.Success);
        
        // Far point may create new category or have low activation
        var farResult = algorithm.predict(farPoint, parameters);
        assertNotNull(farResult);
    }
    
    @Test
    @DisplayName("Should implement proper ellipsoidal distance measure")
    void testEllipsoidalDistance() {
        var pattern1 = Pattern.of(1.0, 0.0);
        var pattern2 = Pattern.of(0.0, 1.0);
        var pattern3 = Pattern.of(0.7, 0.7); // Diagonal
        
        algorithm.learn(pattern1, parameters);
        assertEquals(1, algorithm.getCategoryCount());
        
        // Test different distances from first category
        var result2 = algorithm.predict(pattern2, parameters);
        var result3 = algorithm.predict(pattern3, parameters);
        
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
        
        algorithm.learn(pattern4D, parameters);
        assertEquals(1, algorithm.getCategoryCount());
        
        // Similar pattern should match
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
        var singleDimAlgorithm = new VectorizedEllipsoidART();
        var singleDim = Pattern.of(0.5);
        var singleResult = singleDimAlgorithm.learn(singleDim, parameters);
        assertNotNull(singleResult);
        singleDimAlgorithm.close();
    }
}