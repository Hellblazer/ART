package com.hellblazer.art.performance.algorithms;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test-first development: Tests for VectorizedSalienceParameters
 */
class VectorizedSalienceParametersTest {
    
    @Test
    @DisplayName("Test default parameters creation")
    void testDefaultParameters() {
        var params = VectorizedSalienceParameters.createDefault();
        
        assertNotNull(params);
        assertEquals(0.75, params.vigilance());
        assertEquals(0.1, params.learningRate());
        assertEquals(0.001, params.alpha());
        assertTrue(params.enableSIMD());
        assertTrue(params.useSparseMode());
        assertEquals(0.01, params.sparsityThreshold());
        assertEquals(0.01, params.salienceUpdateRate());
        assertEquals(VectorizedSalienceParameters.SalienceCalculationType.STATISTICAL, 
                    params.calculationType());
    }
    
    @Test
    @DisplayName("Test parameter validation")
    void testParameterValidation() {
        // Test invalid vigilance
        assertThrows(IllegalArgumentException.class, () ->
            new VectorizedSalienceParameters(
                -0.1, 0.1, 0.001, true, true, 0.01, 0.01,
                VectorizedSalienceParameters.SalienceCalculationType.STATISTICAL,
                true, 0.0, 1.0, 100
            )
        );
        
        assertThrows(IllegalArgumentException.class, () ->
            new VectorizedSalienceParameters(
                1.1, 0.1, 0.001, true, true, 0.01, 0.01,
                VectorizedSalienceParameters.SalienceCalculationType.STATISTICAL,
                true, 0.0, 1.0, 100
            )
        );
        
        // Test invalid learning rate
        assertThrows(IllegalArgumentException.class, () ->
            new VectorizedSalienceParameters(
                0.75, -0.1, 0.001, true, true, 0.01, 0.01,
                VectorizedSalienceParameters.SalienceCalculationType.STATISTICAL,
                true, 0.0, 1.0, 100
            )
        );
        
        // Test invalid alpha
        assertThrows(IllegalArgumentException.class, () ->
            new VectorizedSalienceParameters(
                0.75, 0.1, -0.001, true, true, 0.01, 0.01,
                VectorizedSalienceParameters.SalienceCalculationType.STATISTICAL,
                true, 0.0, 1.0, 100
            )
        );
        
        // Test invalid salience bounds
        assertThrows(IllegalArgumentException.class, () ->
            new VectorizedSalienceParameters(
                0.75, 0.1, 0.001, true, true, 0.01, 0.01,
                VectorizedSalienceParameters.SalienceCalculationType.STATISTICAL,
                true, 0.5, 0.4, 100  // min > max
            )
        );
    }
    
    @Test
    @DisplayName("Test with methods for immutable updates")
    void testWithMethods() {
        var original = VectorizedSalienceParameters.createDefault();
        
        var withVigilance = original.withVigilance(0.9);
        assertEquals(0.9, withVigilance.vigilance());
        assertEquals(original.learningRate(), withVigilance.learningRate());
        
        var withLearning = original.withLearningRate(0.5);
        assertEquals(0.5, withLearning.learningRate());
        assertEquals(original.vigilance(), withLearning.vigilance());
        
        var withSalience = original.withSalienceUpdateRate(0.05);
        assertEquals(0.05, withSalience.salienceUpdateRate());
        
        var withSparse = original.withUseSparseMode(false);
        assertFalse(withSparse.useSparseMode());
        
        var withCalcType = original.withCalculationType(
            VectorizedSalienceParameters.SalienceCalculationType.FREQUENCY
        );
        assertEquals(VectorizedSalienceParameters.SalienceCalculationType.FREQUENCY,
                    withCalcType.calculationType());
    }
    
    @Test
    @DisplayName("Test builder pattern")
    void testBuilder() {
        var params = VectorizedSalienceParameters.builder()
            .vigilance(0.8)
            .learningRate(0.2)
            .alpha(0.002)
            .enableSIMD(false)
            .useSparseMode(true)
            .sparsityThreshold(0.02)
            .salienceUpdateRate(0.03)
            .calculationType(VectorizedSalienceParameters.SalienceCalculationType.INFORMATION_GAIN)
            .adaptiveSalience(true)
            .minimumSalience(0.1)
            .maximumSalience(0.9)
            .simdThreshold(50)
            .build();
        
        assertEquals(0.8, params.vigilance());
        assertEquals(0.2, params.learningRate());
        assertEquals(0.002, params.alpha());
        assertFalse(params.enableSIMD());
        assertTrue(params.useSparseMode());
        assertEquals(0.02, params.sparsityThreshold());
        assertEquals(0.03, params.salienceUpdateRate());
        assertEquals(VectorizedSalienceParameters.SalienceCalculationType.INFORMATION_GAIN,
                    params.calculationType());
        assertTrue(params.adaptiveSalience());
        assertEquals(0.1, params.minimumSalience());
        assertEquals(0.9, params.maximumSalience());
        assertEquals(50, params.simdThreshold());
    }
    
    @Test
    @DisplayName("Test conversion to base parameters")
    void testToBaseParameters() {
        var params = VectorizedSalienceParameters.createDefault();
        var baseParams = params.toBaseParameters();
        
        assertNotNull(baseParams);
        // Verify it creates appropriate base parameters for SalienceAwareART
        assertTrue(baseParams instanceof com.hellblazer.art.core.parameters.FuzzyParameters);
    }
    
    @Test
    @DisplayName("Test high performance preset")
    void testHighPerformancePreset() {
        var params = VectorizedSalienceParameters.createHighPerformance();
        
        assertNotNull(params);
        assertTrue(params.vigilance() > 0.7); // Higher vigilance
        assertTrue(params.learningRate() < 0.1); // Lower learning rate for stability
        assertTrue(params.enableSIMD());
        assertTrue(params.adaptiveSalience());
    }
    
    @Test
    @DisplayName("Test memory optimized preset")
    void testMemoryOptimizedPreset() {
        var params = VectorizedSalienceParameters.createMemoryOptimized();
        
        assertNotNull(params);
        assertTrue(params.useSparseMode());
        assertTrue(params.sparsityThreshold() > 0.01); // More aggressive sparsity
        assertFalse(params.enableSIMD()); // Disable SIMD to save memory
    }
}