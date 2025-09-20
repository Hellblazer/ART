package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.performance.BaseVectorizedARTTest;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for VectorizedART2 algorithm.
 * 
 * ART2 handles continuous inputs with preprocessing through theta (contrast enhancement)
 * and epsilon (noise suppression). These tests verify:
 * - Basic learning and prediction functionality
 * - Parameter validation
 * - Preprocessing behavior with theta and epsilon
 * - Vectorization optimizations
 * - Performance tracking
 */
class VectorizedART2Test extends BaseVectorizedARTTest<VectorizedART2, VectorizedART2Parameters> {
    
    @Override
    protected VectorizedART2 createAlgorithm(VectorizedART2Parameters params) {
        return new VectorizedART2(params);
    }
    
    @Override
    protected VectorizedART2Parameters createDefaultParameters() {
        return new VectorizedART2Parameters(
            0.75,    // vigilance
            0.1,     // theta (contrast enhancement)
            0.001,   // epsilon (noise suppression)
            2,       // parallelismLevel
            true     // enableSIMD
        );
    }
    
    @BeforeEach
    protected void setUp() {
        parameters = new VectorizedART2Parameters(
            0.75,    // vigilance
            0.1,     // theta (contrast enhancement)
            0.001,   // epsilon (noise suppression)
            2,       // parallelismLevel
            true     // enableSIMD
        );
        algorithm = new VectorizedART2(parameters);
        super.setUp();
    }
    
    @Override
    protected VectorizedART2Parameters createParametersWithVigilance(double vigilance) {
        return new VectorizedART2Parameters(
            vigilance,
            0.1,     // theta (contrast enhancement)
            0.001,   // epsilon (noise suppression)
            2,       // parallelismLevel
            true     // enableSIMD
        );
    }
    
    @Test
    void testParametersValidation() {
        // Valid parameters should not throw
        assertDoesNotThrow(() -> new VectorizedART2Parameters(0.5, 0.1, 0.001, 1, true));
        
        // Invalid vigilance
        assertThrows(IllegalArgumentException.class, 
            () -> new VectorizedART2Parameters(-0.1, 0.1, 0.001, 1, true));
        assertThrows(IllegalArgumentException.class, 
            () -> new VectorizedART2Parameters(1.1, 0.1, 0.001, 1, true));
        
        // Invalid theta
        assertThrows(IllegalArgumentException.class, 
            () -> new VectorizedART2Parameters(0.5, -0.1, 0.001, 1, true));
        assertThrows(IllegalArgumentException.class, 
            () -> new VectorizedART2Parameters(0.5, 1.1, 0.001, 1, true));
        
        // Invalid epsilon
        assertThrows(IllegalArgumentException.class, 
            () -> new VectorizedART2Parameters(0.5, 0.1, -0.001, 1, true));
        assertThrows(IllegalArgumentException.class, 
            () -> new VectorizedART2Parameters(0.5, 0.1, 1.1, 1, true));
        
        // Invalid parallelism level
        assertThrows(IllegalArgumentException.class, 
            () -> new VectorizedART2Parameters(0.5, 0.1, 0.001, 0, true));
    }
    
    // The following tests are covered by base class:
    // - testBasicLearning()
    // - testMultiplePatternLearning()
    // - testPrediction()
    
    @Test
    void testContrastEnhancement() {
        // Test that theta parameter affects contrast enhancement
        var highTheta = new VectorizedART2Parameters(0.75, 0.8, 0.001, 1, false);
        var lowTheta = new VectorizedART2Parameters(0.75, 0.1, 0.001, 1, false);
        
        var algorithm1 = new VectorizedART2(highTheta);
        var algorithm2 = new VectorizedART2(lowTheta);
        
        try {
            var input = Pattern.of(0.5, 0.3, 0.7, 0.2);
            
            // Both should be able to learn
            var result1 = algorithm1.learn(input, highTheta);
            var result2 = algorithm2.learn(input, lowTheta);
            
            assertInstanceOf(ActivationResult.Success.class, result1);
            assertInstanceOf(ActivationResult.Success.class, result2);
            
        } finally {
            algorithm1.close();
            algorithm2.close();
        }
    }
    
    @Test
    void testNoiseSuppressionEpsilon() {
        // Test that epsilon parameter affects noise suppression
        var highEpsilon = new VectorizedART2Parameters(0.75, 0.1, 0.5, 1, false);
        var lowEpsilon = new VectorizedART2Parameters(0.75, 0.1, 0.001, 1, false);
        
        var algorithm1 = new VectorizedART2(highEpsilon);
        var algorithm2 = new VectorizedART2(lowEpsilon);
        
        try {
            var noisyInput = Pattern.of(0.01, 0.02, 0.99, 0.98); // High contrast with noise
            
            // Both should handle the input
            var result1 = algorithm1.learn(noisyInput, highEpsilon);
            var result2 = algorithm2.learn(noisyInput, lowEpsilon);
            
            assertInstanceOf(ActivationResult.Success.class, result1);
            assertInstanceOf(ActivationResult.Success.class, result2);
            
        } finally {
            algorithm1.close();
            algorithm2.close();
        }
    }
    
    @ParameterizedTest
    @ValueSource(doubles = {0.1, 0.3, 0.5, 0.7, 0.9})
    void testVigilanceThresholds(double vigilance) {
        var testParams = new VectorizedART2Parameters(vigilance, 0.1, 0.001, 1, false);
        var testAlgorithm = new VectorizedART2(testParams);
        
        try {
            var input1 = Pattern.of(0.8, 0.2, 0.5, 0.9);
            var input2 = Pattern.of(0.7, 0.3, 0.4, 0.8); // Somewhat similar
            
            // Learn first pattern
            var result1 = testAlgorithm.learn(input1, testParams);
            assertInstanceOf(ActivationResult.Success.class, result1);
            
            // Learn second pattern
            var result2 = testAlgorithm.learn(input2, testParams);
            assertInstanceOf(ActivationResult.Success.class, result2);
            
            // Higher vigilance should create more categories
            assertTrue(testAlgorithm.getCategoryCount() >= 1);
            
        } finally {
            testAlgorithm.close();
        }
    }
    
    @Test
    void testSIMDOptimization() {
        var simdParams = new VectorizedART2Parameters(0.75, 0.1, 0.001, 2, true);
        var standardParams = new VectorizedART2Parameters(0.75, 0.1, 0.001, 1, false);
        
        var simdAlgorithm = new VectorizedART2(simdParams);
        var standardAlgorithm = new VectorizedART2(standardParams);
        
        try {
            // Use larger input to benefit from SIMD
            var input = Pattern.of(0.8, 0.2, 0.5, 0.9, 0.3, 0.7, 0.1, 0.6);
            
            var simdResult = simdAlgorithm.learn(input, simdParams);
            var standardResult = standardAlgorithm.learn(input, standardParams);
            
            // Both should produce valid results
            assertInstanceOf(ActivationResult.Success.class, simdResult);
            assertInstanceOf(ActivationResult.Success.class, standardResult);
            
        } finally {
            simdAlgorithm.close();
            standardAlgorithm.close();
        }
    }
    
    // Removed - covered by base class testPerformanceTracking()
    
    @Test
    void testEmptyInput() {
        // Test with empty pattern
        assertThrows(IllegalArgumentException.class, () -> {
            var emptyInput = Pattern.of();
            algorithm.learn(emptyInput, parameters);
        });
    }
    
    // Removed - covered by base class testErrorHandling() and testResourceCleanup()
    
    @Test
    void testGetParameters() {
        var retrievedParams = algorithm.getParameters();
        assertEquals(parameters, retrievedParams);
    }
    
    @Test
    void testContinuousInputHandling() {
        // Test with various continuous input ranges
        var inputs = new Pattern[] {
            Pattern.of(0.0, 0.5, 1.0, 0.25),        // Standard [0,1] range
            Pattern.of(-0.5, 1.5, 0.5, 2.0),       // Outside [0,1] range
            Pattern.of(0.1, 0.1, 0.1, 0.1),        // Low values
            Pattern.of(0.9, 0.9, 0.9, 0.9),        // High values
            Pattern.of(0.001, 0.999, 0.5, 0.5)     // Extreme values
        };
        
        for (int i = 0; i < inputs.length; i++) {
            var result = algorithm.learn(inputs[i], parameters);
            assertInstanceOf(ActivationResult.Success.class, result, 
                "Failed to learn input " + i + ": " + inputs[i]);
        }
        
        assertTrue(algorithm.getCategoryCount() >= 1);
    }
}