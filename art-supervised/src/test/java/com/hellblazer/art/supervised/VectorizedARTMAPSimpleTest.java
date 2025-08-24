package com.hellblazer.art.supervised;

import com.hellblazer.art.algorithms.VectorizedParameters;
import com.hellblazer.art.core.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Simple test to validate VectorizedARTMAP basic functionality.
 */
class VectorizedARTMAPSimpleTest {
    
    private VectorizedARTMAP artmap;
    
    @BeforeEach
    void setUp() {
        // Create parameters with corrected API
        var artAParams = VectorizedParameters.createDefault().withVigilance(0.7);
        var artBParams = VectorizedParameters.createDefault().withVigilance(0.8);
        
        var testParams = VectorizedARTMAPParameters.builder()
            .mapVigilance(0.9)
            .baselineVigilance(0.0)
            .vigilanceIncrement(0.05)
            .maxVigilance(0.95)
            .enableMatchTracking(true)
            .maxSearchAttempts(10)
            .artAParams(artAParams)
            .artBParams(artBParams)
            .build();
        
        artmap = new VectorizedARTMAP(testParams);
    }
    
    @Test
    void testBasicTraining() {
        // Create simple vectors
        var input = Pattern.of(1.0, 0.0, 0.0);
        var target = Pattern.of(1.0);
        
        // Test training
        var result = artmap.train(input, target);
        assertNotNull(result);
        assertTrue(result.isSuccess());
    }
    
    @Test
    void testPrediction() {
        // Train first
        var input = Pattern.of(1.0, 0.0, 0.0);
        var target = Pattern.of(1.0);
        artmap.train(input, target);
        
        // Test prediction
        var prediction = artmap.predict(input);
        assertTrue(prediction.isPresent());
    }
    
    @Test
    void testPerformanceMetrics() {
        var metrics = artmap.getPerformanceMetrics();
        assertNotNull(metrics);
        assertEquals(0, metrics.totalTrainingOperations());
    }
}