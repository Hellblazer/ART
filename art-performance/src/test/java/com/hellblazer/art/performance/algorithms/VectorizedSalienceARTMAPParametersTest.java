package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.artmap.ARTMAPParameters;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test-first development: Tests for VectorizedSalienceARTMAPParameters
 */
class VectorizedSalienceARTMAPParametersTest {
    
    @Test
    @DisplayName("Test default parameters creation")
    void testDefaultParameters() {
        var params = VectorizedSalienceARTMAPParameters.defaults();
        
        assertNotNull(params);
        assertEquals(0.9, params.mapVigilance());
        assertEquals(0.0, params.baselineVigilance());
        assertEquals(0.05, params.vigilanceIncrement());
        assertEquals(0.95, params.maxVigilance());
        assertTrue(params.enableMatchTracking());
        assertFalse(params.enableParallelSearch());
        assertEquals(10, params.maxSearchAttempts());
        assertNotNull(params.artAParams());
        assertNotNull(params.artBParams());
        assertFalse(params.enableCrossSalienceAdaptation());
        assertEquals(0.01, params.salienceTransferRate());
        assertEquals(VectorizedSalienceARTMAPParameters.SalienceMappingStrategy.WEIGHTED_AVERAGE,
                    params.mappingStrategy());
    }
    
    @Test
    @DisplayName("Test parameter validation")
    void testParameterValidation() {
        // Invalid map vigilance
        assertThrows(IllegalArgumentException.class, () ->
            new VectorizedSalienceARTMAPParameters(
                -0.1, 0.0, 0.05, 0.95, true, false, 10,
                VectorizedSalienceParameters.createDefault(),
                VectorizedSalienceParameters.createDefault(),
                false, 0.01,
                VectorizedSalienceARTMAPParameters.SalienceMappingStrategy.WEIGHTED_AVERAGE
            )
        );
        
        // Invalid vigilance increment
        assertThrows(IllegalArgumentException.class, () ->
            new VectorizedSalienceARTMAPParameters(
                0.9, 0.0, -0.1, 0.95, true, false, 10,
                VectorizedSalienceParameters.createDefault(),
                VectorizedSalienceParameters.createDefault(),
                false, 0.01,
                VectorizedSalienceARTMAPParameters.SalienceMappingStrategy.WEIGHTED_AVERAGE
            )
        );
        
        // Max vigilance < map vigilance
        assertThrows(IllegalArgumentException.class, () ->
            new VectorizedSalienceARTMAPParameters(
                0.9, 0.0, 0.05, 0.8, true, false, 10,
                VectorizedSalienceParameters.createDefault(),
                VectorizedSalienceParameters.createDefault(),
                false, 0.01,
                VectorizedSalienceARTMAPParameters.SalienceMappingStrategy.WEIGHTED_AVERAGE
            )
        );
        
        // Null artA parameters
        assertThrows(NullPointerException.class, () ->
            new VectorizedSalienceARTMAPParameters(
                0.9, 0.0, 0.05, 0.95, true, false, 10,
                null,
                VectorizedSalienceParameters.createDefault(),
                false, 0.01,
                VectorizedSalienceARTMAPParameters.SalienceMappingStrategy.WEIGHTED_AVERAGE
            )
        );
    }
    
    @Test
    @DisplayName("Test conversion to ARTMAPParameters")
    void testToARTMAPParameters() {
        var params = VectorizedSalienceARTMAPParameters.defaults();
        var artmapParams = params.toARTMAPParameters();
        
        assertNotNull(artmapParams);
        assertEquals(params.mapVigilance(), artmapParams.mapVigilance());
        assertEquals(params.baselineVigilance(), artmapParams.baselineVigilance());
    }
    
    @Test
    @DisplayName("Test builder pattern")
    void testBuilder() {
        var params = VectorizedSalienceARTMAPParameters.builder()
            .mapVigilance(0.85)
            .baselineVigilance(0.1)
            .vigilanceIncrement(0.1)
            .maxVigilance(0.99)
            .enableMatchTracking(false)
            .enableParallelSearch(true)
            .maxSearchAttempts(20)
            .artAParams(VectorizedSalienceParameters.createHighPerformance())
            .artBParams(VectorizedSalienceParameters.createMemoryOptimized())
            .enableCrossSalienceAdaptation(true)
            .salienceTransferRate(0.05)
            .mappingStrategy(VectorizedSalienceARTMAPParameters.SalienceMappingStrategy.MAX_SALIENCE)
            .build();
        
        assertEquals(0.85, params.mapVigilance());
        assertEquals(0.1, params.baselineVigilance());
        assertEquals(0.1, params.vigilanceIncrement());
        assertEquals(0.99, params.maxVigilance());
        assertFalse(params.enableMatchTracking());
        assertTrue(params.enableParallelSearch());
        assertEquals(20, params.maxSearchAttempts());
        assertTrue(params.enableCrossSalienceAdaptation());
        assertEquals(0.05, params.salienceTransferRate());
        assertEquals(VectorizedSalienceARTMAPParameters.SalienceMappingStrategy.MAX_SALIENCE,
                    params.mappingStrategy());
    }
    
    @Test
    @DisplayName("Test with methods for immutable updates")
    void testWithMethods() {
        var original = VectorizedSalienceARTMAPParameters.defaults();
        
        var withMapVigilance = original.withMapVigilance(0.95);
        assertEquals(0.95, withMapVigilance.mapVigilance());
        assertEquals(original.baselineVigilance(), withMapVigilance.baselineVigilance());
        
        var withBaseline = original.withBaselineVigilance(0.2);
        assertEquals(0.2, withBaseline.baselineVigilance());
        
        var withIncrement = original.withVigilanceIncrement(0.1);
        assertEquals(0.1, withIncrement.vigilanceIncrement());
        
        var withMatchTracking = original.withEnableMatchTracking(false);
        assertFalse(withMatchTracking.enableMatchTracking());
        
        var withSalience = original.withEnableCrossSalienceAdaptation(true);
        assertTrue(withSalience.enableCrossSalienceAdaptation());
    }
    
    @Test
    @DisplayName("Test salience mapping strategies")
    void testSalienceMappingStrategies() {
        var strategies = VectorizedSalienceARTMAPParameters.SalienceMappingStrategy.values();
        assertEquals(3, strategies.length);
        
        // Test each strategy can be set
        for (var strategy : strategies) {
            var params = VectorizedSalienceARTMAPParameters.builder()
                .mappingStrategy(strategy)
                .build();
            assertEquals(strategy, params.mappingStrategy());
        }
    }
}