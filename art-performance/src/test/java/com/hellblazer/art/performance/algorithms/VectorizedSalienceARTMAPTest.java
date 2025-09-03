package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.Pattern;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test-first development: Tests for VectorizedSalienceARTMAP
 */
class VectorizedSalienceARTMAPTest {
    
    private VectorizedSalienceARTMAP artmap;
    private VectorizedSalienceARTMAPParameters parameters;
    
    @BeforeEach
    void setUp() {
        parameters = VectorizedSalienceARTMAPParameters.defaults();
        artmap = new VectorizedSalienceARTMAP(parameters);
    }
    
    @Test
    @DisplayName("Test initialization")
    void testInitialization() {
        assertNotNull(artmap);
        assertEquals(0, artmap.getCategoryCount());
        assertNotNull(artmap.getParameters());
        assertEquals(parameters, artmap.getParameters());
        assertFalse(artmap.isTrained());
    }
    
    @Test
    @DisplayName("Test supervised learning with single pattern pair")
    void testLearnSinglePatternPair() {
        var inputPattern = Pattern.of(0.1, 0.2, 0.3, 0.4, 0.5);
        var outputPattern = Pattern.of(1.0, 0.0);  // Binary output
        
        var result = artmap.learn(inputPattern, outputPattern, parameters);
        
        assertNotNull(result);
        assertTrue(result.isPredictionSuccessful());
        assertEquals(1, artmap.getCategoryCount());
        assertTrue(artmap.isTrained());
    }
    
    @Test
    @DisplayName("Test learning multiple pattern pairs")
    void testLearnMultiplePatternPairs() {
        var input1 = Pattern.of(0.1, 0.2, 0.3);
        var output1 = Pattern.of(1.0, 0.0);
        
        var input2 = Pattern.of(0.9, 0.8, 0.7);
        var output2 = Pattern.of(0.0, 1.0);
        
        var input3 = Pattern.of(0.5, 0.5, 0.5);
        var output3 = Pattern.of(1.0, 0.0);
        
        artmap.learn(input1, output1, parameters);
        artmap.learn(input2, output2, parameters);
        artmap.learn(input3, output3, parameters);
        
        assertTrue(artmap.getCategoryCount() > 0);
        assertTrue(artmap.getCategoryCount() <= 3);
    }
    
    @Test
    @DisplayName("Test prediction after learning")
    void testPredictAfterLearning() {
        // Train the network
        var input1 = Pattern.of(0.1, 0.2, 0.3);
        var output1 = Pattern.of(1.0, 0.0);
        artmap.learn(input1, output1, parameters);
        
        var input2 = Pattern.of(0.9, 0.8, 0.7);
        var output2 = Pattern.of(0.0, 1.0);
        artmap.learn(input2, output2, parameters);
        
        // Test prediction on learned pattern
        var result1 = artmap.predict(input1, parameters);
        assertNotNull(result1);
        assertTrue(result1.isPredictionSuccessful());
        
        // Test prediction on similar pattern
        var similar = Pattern.of(0.11, 0.21, 0.31);
        var result2 = artmap.predict(similar, parameters);
        assertNotNull(result2);
    }
    
    @Test
    @DisplayName("Test null input validation")
    void testNullInputValidation() {
        var outputPattern = Pattern.of(1.0, 0.0);
        
        assertThrows(IllegalArgumentException.class, () -> {
            artmap.learn(null, outputPattern, parameters);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            artmap.predict(null, parameters);
        });
    }
    
    @Test
    @DisplayName("Test null output validation")
    void testNullOutputValidation() {
        var inputPattern = Pattern.of(0.1, 0.2, 0.3);
        
        assertThrows(IllegalArgumentException.class, () -> {
            artmap.learn(inputPattern, null, parameters);
        });
    }
    
    @Test
    @DisplayName("Test null parameters validation")
    void testNullParametersValidation() {
        var inputPattern = Pattern.of(0.1, 0.2, 0.3);
        var outputPattern = Pattern.of(1.0, 0.0);
        
        assertThrows(IllegalArgumentException.class, () -> {
            artmap.learn(inputPattern, outputPattern, null);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            artmap.predict(inputPattern, null);
        });
    }
    
    @Test
    @DisplayName("Test performance stats tracking")
    void testPerformanceStats() {
        var stats1 = artmap.getPerformanceStats();
        assertEquals(0L, stats1.totalMapFieldOperations());
        
        var inputPattern = Pattern.of(0.1, 0.2, 0.3);
        var outputPattern = Pattern.of(1.0, 0.0);
        artmap.learn(inputPattern, outputPattern, parameters);
        
        var stats2 = artmap.getPerformanceStats();
        assertTrue(stats2.totalMapFieldOperations() > 0);
        
        artmap.resetPerformanceTracking();
        var stats3 = artmap.getPerformanceStats();
        assertEquals(0L, stats3.totalMapFieldOperations());
    }
    
    @Test
    @DisplayName("Test match tracking")
    void testMatchTracking() {
        var paramsWithTracking = parameters.withEnableMatchTracking(true);
        var artmapWithTracking = new VectorizedSalienceARTMAP(paramsWithTracking);
        
        // Create patterns that should trigger match tracking
        var input1 = Pattern.of(0.2, 0.3, 0.4);
        var output1 = Pattern.of(1.0, 0.0);
        
        var input2 = Pattern.of(0.25, 0.35, 0.45);  // Similar to input1
        var output2 = Pattern.of(0.0, 1.0);  // Different output
        
        artmapWithTracking.learn(input1, output1, paramsWithTracking);
        var result = artmapWithTracking.learn(input2, output2, paramsWithTracking);
        
        assertNotNull(result);
        // Match tracking should have been invoked (categories may or may not be separate)
        assertTrue(artmapWithTracking.getCategoryCount() >= 1);
    }
    
    @Test
    @DisplayName("Test cross-salience adaptation")
    void testCrossSalienceAdaptation() {
        var paramsWithSalience = parameters.withEnableCrossSalienceAdaptation(true)
                                          .withSalienceTransferRate(0.05);
        var artmapWithSalience = new VectorizedSalienceARTMAP(paramsWithSalience);
        
        var input = Pattern.of(0.1, 0.2, 0.3, 0.4, 0.5);
        var output = Pattern.of(1.0, 0.0);
        
        var result = artmapWithSalience.learn(input, output, paramsWithSalience);
        
        assertNotNull(result);
        assertNotNull(result.salienceMetrics());
        // When cross-salience is enabled, metrics should be tracked
        if (paramsWithSalience.enableCrossSalienceAdaptation()) {
            assertFalse(result.salienceMetrics().isEmpty());
        }
    }
    
    @Test
    @DisplayName("Test resource cleanup with close")
    void testResourceCleanup() {
        var inputPattern = Pattern.of(0.1, 0.2, 0.3);
        var outputPattern = Pattern.of(1.0, 0.0);
        artmap.learn(inputPattern, outputPattern, parameters);
        
        artmap.close();
        
        // After closing, operations should throw IllegalStateException
        assertThrows(IllegalStateException.class, () -> {
            artmap.learn(inputPattern, outputPattern, parameters);
        });
    }
    
    @Test
    @DisplayName("Test batch learning")
    void testBatchLearning() {
        var inputPatterns = new Pattern[] {
            Pattern.of(0.1, 0.2, 0.3),
            Pattern.of(0.4, 0.5, 0.6),
            Pattern.of(0.7, 0.8, 0.9)
        };
        
        var outputPatterns = new Pattern[] {
            Pattern.of(1.0, 0.0),
            Pattern.of(0.0, 1.0),
            Pattern.of(1.0, 0.0)
        };
        
        var results = artmap.learnBatch(inputPatterns, outputPatterns, parameters);
        
        assertNotNull(results);
        assertEquals(inputPatterns.length, results.length);
        assertTrue(artmap.getCategoryCount() > 0);
    }
    
    @Test
    @DisplayName("Test batch prediction")
    void testBatchPrediction() {
        // First train some patterns
        artmap.learn(Pattern.of(0.1, 0.2, 0.3), Pattern.of(1.0, 0.0), parameters);
        artmap.learn(Pattern.of(0.7, 0.8, 0.9), Pattern.of(0.0, 1.0), parameters);
        
        var inputPatterns = new Pattern[] {
            Pattern.of(0.11, 0.21, 0.31),
            Pattern.of(0.71, 0.81, 0.91)
        };
        
        var results = artmap.predictBatch(inputPatterns, parameters);
        
        assertNotNull(results);
        assertEquals(inputPatterns.length, results.length);
    }
    
    @Test
    @DisplayName("Test algorithm type identification")
    void testAlgorithmType() {
        assertEquals("VectorizedSalienceARTMAP", artmap.getAlgorithmType());
        assertTrue(artmap.isSupervised());
    }
    
    @Test
    @DisplayName("Test salience mapping strategies")
    void testSalienceMappingStrategies() {
        // Test WEIGHTED_AVERAGE strategy
        var avgParams = parameters.withMapVigilance(0.85);
        var avgArtmap = new VectorizedSalienceARTMAP(avgParams);
        
        var input = Pattern.of(0.3, 0.4, 0.5);
        var output = Pattern.of(1.0, 0.0);
        
        var avgResult = avgArtmap.learn(input, output, avgParams);
        assertNotNull(avgResult);
        
        // Test MAX_SALIENCE strategy
        var maxParams = VectorizedSalienceARTMAPParameters.builder()
            .mappingStrategy(VectorizedSalienceARTMAPParameters.SalienceMappingStrategy.MAX_SALIENCE)
            .build();
        var maxArtmap = new VectorizedSalienceARTMAP(maxParams);
        
        var maxResult = maxArtmap.learn(input, output, maxParams);
        assertNotNull(maxResult);
        
        // Test ADAPTIVE strategy
        var adaptiveParams = VectorizedSalienceARTMAPParameters.builder()
            .mappingStrategy(VectorizedSalienceARTMAPParameters.SalienceMappingStrategy.ADAPTIVE)
            .build();
        var adaptiveArtmap = new VectorizedSalienceARTMAP(adaptiveParams);
        
        var adaptiveResult = adaptiveArtmap.learn(input, output, adaptiveParams);
        assertNotNull(adaptiveResult);
    }
}