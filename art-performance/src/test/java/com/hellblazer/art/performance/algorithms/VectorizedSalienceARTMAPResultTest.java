package com.hellblazer.art.performance.algorithms;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test-first development: Tests for VectorizedSalienceARTMAPResult
 */
class VectorizedSalienceARTMAPResultTest {
    
    @Test
    @DisplayName("Test creation with successful prediction")
    void testSuccessfulPrediction() {
        var result = VectorizedSalienceARTMAPResult.success(
            1,  // predictedCategory
            0.95,  // confidence
            0.85,  // artAActivation
            0.90,  // artBActivation
            0.92  // mapFieldActivation
        );
        
        assertNotNull(result);
        assertTrue(result.isPredictionSuccessful());
        assertEquals(1, result.predictedCategory());
        assertEquals(0.95, result.confidence());
        assertEquals(0.85, result.artAActivation());
        assertEquals(0.90, result.artBActivation());
        assertEquals(0.92, result.mapFieldActivation());
        assertNotNull(result.salienceMetrics());
        assertEquals(0, result.salienceMetrics().size());
    }
    
    @Test
    @DisplayName("Test creation with no match")
    void testNoMatch() {
        var result = VectorizedSalienceARTMAPResult.noMatch(
            "No resonance achieved"
        );
        
        assertNotNull(result);
        assertFalse(result.isPredictionSuccessful());
        assertEquals(-1, result.predictedCategory());
        assertEquals(0.0, result.confidence());
        assertEquals(0.0, result.artAActivation());
        assertEquals(0.0, result.artBActivation());
        assertEquals(0.0, result.mapFieldActivation());
        assertEquals("No resonance achieved", result.noMatchReason());
    }
    
    @Test
    @DisplayName("Test with salience metrics")
    void testWithSalienceMetrics() {
        var metrics = Map.of(
            "avgSalience", 0.75,
            "maxSalience", 0.95,
            "minSalience", 0.45
        );
        
        var result = VectorizedSalienceARTMAPResult.successWithMetrics(
            2,  // predictedCategory
            0.88,  // confidence
            0.82,  // artAActivation
            0.86,  // artBActivation
            0.84,  // mapFieldActivation
            metrics
        );
        
        assertNotNull(result);
        assertTrue(result.isPredictionSuccessful());
        assertEquals(2, result.predictedCategory());
        assertEquals(0.88, result.confidence());
        assertEquals(0.82, result.artAActivation());
        assertEquals(0.86, result.artBActivation());
        assertEquals(0.84, result.mapFieldActivation());
        assertNotNull(result.salienceMetrics());
        assertEquals(3, result.salienceMetrics().size());
        assertEquals(0.75, result.salienceMetrics().get("avgSalience"));
        assertEquals(0.95, result.salienceMetrics().get("maxSalience"));
        assertEquals(0.45, result.salienceMetrics().get("minSalience"));
    }
    
    @Test
    @DisplayName("Test builder pattern")
    void testBuilder() {
        var result = VectorizedSalienceARTMAPResult.builder()
            .predictedCategory(3)
            .confidence(0.91)
            .artAActivation(0.87)
            .artBActivation(0.89)
            .mapFieldActivation(0.88)
            .addSalienceMetric("testMetric", 0.55)
            .build();
        
        assertNotNull(result);
        assertTrue(result.isPredictionSuccessful());
        assertEquals(3, result.predictedCategory());
        assertEquals(0.91, result.confidence());
        assertEquals(0.87, result.artAActivation());
        assertEquals(0.89, result.artBActivation());
        assertEquals(0.88, result.mapFieldActivation());
        assertEquals(1, result.salienceMetrics().size());
        assertEquals(0.55, result.salienceMetrics().get("testMetric"));
    }
    
    @Test
    @DisplayName("Test builder with no match")
    void testBuilderNoMatch() {
        var result = VectorizedSalienceARTMAPResult.builder()
            .noMatch("Vigilance threshold exceeded")
            .build();
        
        assertNotNull(result);
        assertFalse(result.isPredictionSuccessful());
        assertEquals(-1, result.predictedCategory());
        assertEquals(0.0, result.confidence());
        assertEquals("Vigilance threshold exceeded", result.noMatchReason());
    }
    
    @Test
    @DisplayName("Test immutability of salience metrics")
    void testSalienceMetricsImmutability() {
        var metrics = Map.of("metric1", 0.5);
        var result = VectorizedSalienceARTMAPResult.successWithMetrics(
            1, 0.9, 0.8, 0.85, 0.87, metrics
        );
        
        // Should not be able to modify the returned map
        assertThrows(UnsupportedOperationException.class, () -> {
            result.salienceMetrics().put("newMetric", 0.6);
        });
    }
    
    @Test
    @DisplayName("Test validation of confidence values")
    void testConfidenceValidation() {
        // Invalid confidence > 1.0
        assertThrows(IllegalArgumentException.class, () -> {
            VectorizedSalienceARTMAPResult.success(1, 1.5, 0.8, 0.85, 0.87);
        });
        
        // Invalid confidence < 0.0
        assertThrows(IllegalArgumentException.class, () -> {
            VectorizedSalienceARTMAPResult.success(1, -0.1, 0.8, 0.85, 0.87);
        });
    }
    
    @Test
    @DisplayName("Test validation of activation values")
    void testActivationValidation() {
        // Invalid artA activation
        assertThrows(IllegalArgumentException.class, () -> {
            VectorizedSalienceARTMAPResult.success(1, 0.9, 1.5, 0.85, 0.87);
        });
        
        // Invalid artB activation
        assertThrows(IllegalArgumentException.class, () -> {
            VectorizedSalienceARTMAPResult.success(1, 0.9, 0.8, -0.1, 0.87);
        });
        
        // Invalid map field activation
        assertThrows(IllegalArgumentException.class, () -> {
            VectorizedSalienceARTMAPResult.success(1, 0.9, 0.8, 0.85, 2.0);
        });
    }
    
    @Test
    @DisplayName("Test toString format")
    void testToString() {
        var result = VectorizedSalienceARTMAPResult.success(
            5, 0.92, 0.88, 0.90, 0.91
        );
        
        var str = result.toString();
        assertNotNull(str);
        assertTrue(str.contains("category=5"));
        assertTrue(str.contains("confidence=0.92"));
    }
}