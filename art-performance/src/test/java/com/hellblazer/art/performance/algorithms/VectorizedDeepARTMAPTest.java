/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 */
package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.*;
import com.hellblazer.art.core.artmap.DeepARTMAPParameters;
import com.hellblazer.art.core.artmap.DeepARTMAPResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.AfterEach;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;

/**
 * Basic test suite for VectorizedDeepARTMAP implementation.
 * Tests core functionality and SIMD optimizations.
 */
public class VectorizedDeepARTMAPTest {
    
    private VectorizedDeepARTMAP vectorizedDeepART;
    private VectorizedDeepARTMAPParameters vectorizedParams;
    
    // Test data for multi-channel processing
    private List<Pattern[]> multiChannelData;
    private int[] supervisedLabels;
    
    @BeforeEach
    void setUp() {
        // Create base DeepARTMAP parameters
        var baseParams = new DeepARTMAPParameters(0.8, 0.1, 1000, true);
        
        // Create vectorized parameters
        vectorizedParams = new VectorizedDeepARTMAPParameters(baseParams, 4, 2, 2, 2, true, 0, 100, 100, 0.8, true);
        
        // Create ART modules for testing (using vectorized versions)
        var modules = List.<BaseART>of(
            new VectorizedFuzzyART(VectorizedParameters.createDefault()),
            new VectorizedFuzzyART(VectorizedParameters.createDefault())
        );
        
        vectorizedDeepART = new VectorizedDeepARTMAP(modules, vectorizedParams);
        
        // Create multi-channel test data
        createTestData();
    }
    
    @AfterEach
    void tearDown() {
        if (vectorizedDeepART != null) {
            vectorizedDeepART.close();
        }
    }
    
    /**
     * Create multi-channel test data for training and testing.
     */
    private void createTestData() {
        int sampleCount = 20;
        
        // Channel 0: FuzzyART patterns
        var channel0 = new Pattern[sampleCount];
        for (int i = 0; i < sampleCount; i++) {
            double x = i / (double) sampleCount;
            channel0[i] = Pattern.of(x, 1.0 - x, x * 0.8, 1.0 - x * 0.8);
        }
        
        // Channel 1: More FuzzyART patterns
        var channel1 = new Pattern[sampleCount];
        for (int i = 0; i < sampleCount; i++) {
            double x = (i * 37 % sampleCount) / (double) sampleCount;
            channel1[i] = Pattern.of(x, x * x, Math.sqrt(x), 1.0 - x);
        }
        
        multiChannelData = List.of(channel0, channel1);
        
        // Create supervised labels (3 classes)
        supervisedLabels = new int[sampleCount];
        for (int i = 0; i < sampleCount; i++) {
            supervisedLabels[i] = i % 3;
        }
    }
    
    @Test
    @DisplayName("Basic construction and initialization should work correctly")
    void testBasicConstruction() {
        assertNotNull(vectorizedDeepART);
        assertFalse(vectorizedDeepART.is_fitted());
    }
    
    @Test
    @DisplayName("Parameter validation should work correctly")
    void testParameterValidation() {
        // Null modules should throw exception
        assertThrows(IllegalArgumentException.class, () -> {
            new VectorizedDeepARTMAP(null, vectorizedParams);
        });
        
        // Empty modules should throw exception
        assertThrows(IllegalArgumentException.class, () -> {
            new VectorizedDeepARTMAP(List.of(), vectorizedParams);
        });
        
        // Null parameters should throw exception
        var modules = List.<BaseART>of(new VectorizedFuzzyART(VectorizedParameters.createDefault()));
        assertThrows(IllegalArgumentException.class, () -> {
            new VectorizedDeepARTMAP(modules, null);
        });
    }
    
    @Test
    @DisplayName("Supervised training should work with parallel processing")
    void testSupervisedTrainingParallel() {
        var result = vectorizedDeepART.fitSupervised(multiChannelData, supervisedLabels);
        
        assertTrue(result instanceof DeepARTMAPResult.Success);
        assertTrue(vectorizedDeepART.is_fitted());
        
        var success = (DeepARTMAPResult.Success) result;
        assertTrue(success.supervisedMode());
        assertEquals(multiChannelData.get(0).length, success.deepLabels().length);
    }
    
    @Test
    @DisplayName("Vectorized prediction should work correctly")
    void testVectorizedPrediction() {
        // Train first
        vectorizedDeepART.fitSupervised(multiChannelData, supervisedLabels);
        
        // Test prediction
        var predictions = vectorizedDeepART.predict(multiChannelData);
        assertEquals(multiChannelData.get(0).length, predictions.length);
        
        // All predictions should be valid category indices
        for (var prediction : predictions) {
            assertTrue(prediction >= 0);
        }
    }
    
    @Test
    @DisplayName("Deep prediction should work correctly")
    void testDeepPrediction() {
        // Train first
        vectorizedDeepART.fitSupervised(multiChannelData, supervisedLabels);
        
        var deepPredictions = vectorizedDeepART.predictDeep(multiChannelData);
        assertEquals(multiChannelData.get(0).length, deepPredictions.length);
        assertEquals(2, deepPredictions[0].length); // 2 layers
    }
    
    @Test
    @DisplayName("SIMD probability calculations should work correctly")
    void testSIMDProbabilityCalculations() {
        // Train first
        vectorizedDeepART.fitSupervised(multiChannelData, supervisedLabels);
        
        var probabilities = vectorizedDeepART.predict_proba(multiChannelData);
        assertEquals(multiChannelData.get(0).length, probabilities.length);
        assertTrue(probabilities[0].length >= 1);
        
        // Check probability constraints
        for (var sampleProbs : probabilities) {
            for (var prob : sampleProbs) {
                assertTrue(prob >= 0.0 && prob <= 1.0);
            }
        }
    }
    
    @Test
    @DisplayName("Performance monitoring should track metrics correctly")
    void testPerformanceMonitoring() {
        // Train and predict
        vectorizedDeepART.fitSupervised(multiChannelData, supervisedLabels);
        vectorizedDeepART.predict(multiChannelData);
        vectorizedDeepART.predict_proba(multiChannelData);
        
        var stats = vectorizedDeepART.getPerformanceStats();
        
        // Check basic metrics
        assertTrue(stats.operationCount() >= 0);
        assertTrue(stats.categoryCount() >= 0);
        
        // Check efficiency calculations
        assertTrue(stats.simdEfficiency() >= 0.0 && stats.simdEfficiency() <= 1.0);
        assertTrue(stats.channelParallelismEfficiency() >= 0.0 && stats.channelParallelismEfficiency() <= 1.0);
        assertTrue(stats.layerParallelismEfficiency() >= 0.0 && stats.layerParallelismEfficiency() <= 1.0);
    }
    
    @Test
    @DisplayName("Error handling should work correctly")
    void testErrorHandling() {
        // Training with null data should throw exception
        assertThrows(IllegalArgumentException.class, () -> {
            vectorizedDeepART.fitSupervised(null, supervisedLabels);
        });
        
        // Training with mismatched label count should throw exception
        var wrongLabels = new int[multiChannelData.get(0).length - 1];
        assertThrows(IllegalArgumentException.class, () -> {
            vectorizedDeepART.fitSupervised(multiChannelData, wrongLabels);
        });
        
        // Prediction before training should throw exception
        assertThrows(IllegalStateException.class, () -> {
            vectorizedDeepART.predict(multiChannelData);
        });
    }
    
    @Test
    @DisplayName("toString should work correctly")
    void testStringRepresentations() {
        vectorizedDeepART.fitSupervised(multiChannelData, supervisedLabels);
        vectorizedDeepART.predict(multiChannelData);
        
        // Test toString
        var toString = vectorizedDeepART.toString();
        assertNotNull(toString);
        assertTrue(toString.contains("VectorizedDeepARTMAP"));
        
        // Test performance stats toString
        var stats = vectorizedDeepART.getPerformanceStats();
        var statsString = stats.toString();
        
        assertNotNull(statsString);
        assertTrue(statsString.contains("VectorizedDeepARTMAPPerformanceStats"));
    }
}