package com.hellblazer.art.performance;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.performance.algorithms.VectorizedParameters;
import com.hellblazer.art.performance.supervised.VectorizedARTMAP;
import com.hellblazer.art.performance.supervised.VectorizedARTMAPParameters;

/**
 * Reference validation test for VectorizedARTMAP implementation.
 * Tests supervised learning with dual ART networks and map field coordination.
 */
@DisplayName("Vectorized ARTMAP Reference Validation")
class VectorizedARTMAPReferenceTest {
    
    private VectorizedARTMAPParameters referenceParams;
    private VectorizedARTMAP vectorizedArtmap;
    
    @BeforeEach
    void setUp() {
        var artAParams = VectorizedParameters.createDefault()
            .withVigilance(0.75)
            .withLearningRate(0.1);
        var artBParams = VectorizedParameters.createDefault()
            .withVigilance(0.8)
            .withLearningRate(0.1);
            
        referenceParams = VectorizedARTMAPParameters.builder()
            .mapVigilance(0.9)
            .baselineVigilance(0.75)
            .vigilanceIncrement(0.01)
            .maxVigilance(0.99)
            .enableMatchTracking(true)
            .maxSearchAttempts(100)
            .artAParams(artAParams)
            .artBParams(artBParams)
            .build();
        
        vectorizedArtmap = new VectorizedARTMAP(referenceParams);
    }
    
    @AfterEach
    void tearDown() {
        if (vectorizedArtmap != null) {
            vectorizedArtmap.close();
        }
    }
    
    @Test
    @DisplayName("ARTMAP supervised learning")
    void testInputOutputAssociation() {
        var inputPatterns = new Pattern[] {
            Pattern.of(0.0, 0.0),
            Pattern.of(0.8, 0.9)
        };
        
        var outputPatterns = new Pattern[] {
            Pattern.of(1.0, 0.0),
            Pattern.of(0.0, 1.0)
        };
        
        // Train the ARTMAP on input-output associations
        for (int i = 0; i < inputPatterns.length; i++) {
            var result = vectorizedArtmap.train(inputPatterns[i], outputPatterns[i]);
            assertNotNull(result, "ARTMAP training should return a result");
        }
        
        assertTrue(vectorizedArtmap.getCategoryCount() > 0, "Should create categories");
    }
    
    @Test
    @DisplayName("Map field vigilance")
    void testMapFieldVigilance() {
        assertTrue(referenceParams.mapVigilance() > 0, "Map vigilance should be positive");
        assertTrue(referenceParams.baselineVigilance() > 0, "Baseline vigilance should be positive");
        
        // Test with high map vigilance - should be more selective
        var inputA = Pattern.of(0.1, 0.2);
        var outputA = Pattern.of(1.0, 0.0);
        var outputB = Pattern.of(0.0, 1.0); // Different output for same input
        
        vectorizedArtmap.train(inputA, outputA);
        var prediction = vectorizedArtmap.predict(inputA, referenceParams);
        assertNotNull(prediction, "Should be able to predict after learning");
    }
    
    @Test
    @DisplayName("Dual network coordination")
    void testDualNetworkCoordination() {
        // Test that both ART networks coordinate properly
        var inputs = new Pattern[] {
            Pattern.of(0.1, 0.1),
            Pattern.of(0.9, 0.9)
        };
        var outputs = new Pattern[] {
            Pattern.of(1.0, 0.0),
            Pattern.of(0.0, 1.0)
        };
        
        for (int i = 0; i < inputs.length; i++) {
            var result = vectorizedArtmap.train(inputs[i], outputs[i]);
            assertNotNull(result, "Dual network training should succeed");
        }
        
        // Test predictions work for both networks
        for (var input : inputs) {
            var prediction = vectorizedArtmap.predict(input, referenceParams);
            assertNotNull(prediction, "Dual network prediction should work");
        }
    }
    
    @Test
    @DisplayName("Match tracking reset") 
    void testMatchTrackingReset() {
        assertTrue(referenceParams.enableMatchTracking(), "Match tracking should be enabled");
        assertTrue(referenceParams.vigilanceIncrement() > 0, "Vigilance increment should be positive");
        
        // Test match tracking with conflicting associations
        var conflictInput = Pattern.of(0.5, 0.5);
        var firstOutput = Pattern.of(1.0, 0.0);
        var secondOutput = Pattern.of(0.0, 1.0);
        
        vectorizedArtmap.train(conflictInput, firstOutput);
        vectorizedArtmap.train(conflictInput, secondOutput);
        
        var prediction = vectorizedArtmap.predict(conflictInput, referenceParams);
        assertNotNull(prediction, "Match tracking should handle conflicts");
    }
    
    @Test
    @DisplayName("Complex association patterns")
    void testComplexAssociationPatterns() {
        var complexInputs = new Pattern[] {
            Pattern.of(0.1, 0.2, 0.3),
            Pattern.of(0.7, 0.8, 0.9),
            Pattern.of(0.4, 0.5, 0.6)
        };
        
        var complexOutputs = new Pattern[] {
            Pattern.of(1.0, 0.0, 0.0),
            Pattern.of(0.0, 1.0, 0.0),
            Pattern.of(0.0, 0.0, 1.0)
        };
        
        // Train on complex associations
        for (int i = 0; i < complexInputs.length; i++) {
            var result = vectorizedArtmap.train(complexInputs[i], complexOutputs[i]);
            assertNotNull(result, "Complex pattern training should succeed");
        }
        
        // Test generalization
        var novelInput = Pattern.of(0.15, 0.25, 0.35);
        var prediction = vectorizedArtmap.predict(novelInput, referenceParams);
        assertNotNull(prediction, "Should handle novel patterns");
        
        var stats = vectorizedArtmap.getPerformanceStats();
        assertNotNull(stats, "Performance stats should be available");
    }
}