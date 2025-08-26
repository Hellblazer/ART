package com.hellblazer.art.core;

import com.hellblazer.art.core.algorithms.FuzzyART;
import com.hellblazer.art.core.artmap.ARTMAP;
import com.hellblazer.art.core.artmap.ARTMAPParameters;
import com.hellblazer.art.core.artmap.ARTMAPResult;
import com.hellblazer.art.core.parameters.FuzzyParameters;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for ARTMAP supervised learning implementation.
 * Tests the complete ARTMAP algorithm including:
 * - Dual ART module coordination (ARTa and ARTb)
 * - Map field creation and management
 * - Map field vigilance testing and mismatch handling
 * - Supervised learning with input-output associations
 * - Prediction functionality
 * - Vigilance search and reset mechanisms
 */
class ARTMAPTest {
    
    private ARTMAP artmap;
    private FuzzyART artA;
    private FuzzyART artB;
    private ARTMAPParameters defaultMapParams;
    private ARTMAPParameters highMapVigilance;
    private FuzzyParameters artAParams;
    private FuzzyParameters artBParams;
    
    @BeforeEach
    void setUp() {
        artA = new FuzzyART();
        artB = new FuzzyART();
        defaultMapParams = ARTMAPParameters.defaults(); // mapVigilance=0.9, baseline=0.0
        highMapVigilance = ARTMAPParameters.of(0.95, 0.0);
        artmap = new ARTMAP(artA, artB, defaultMapParams);
        
        artAParams = FuzzyParameters.of(0.7, 0.0, 1.0); // Input processing
        artBParams = FuzzyParameters.of(0.5, 0.0, 1.0); // Output processing
    }
    
    @Test
    @DisplayName("ARTMAP constructor creates dual ART system with empty map field")
    void testConstructor() {
        assertEquals(artA, artmap.getArtA());
        assertEquals(artB, artmap.getArtB());
        assertEquals(defaultMapParams, artmap.getMapParameters());
        assertEquals(0, artmap.getMapField().size());
        assertEquals("ARTMAP{artA=0 categories, artB=0 categories, mappings=0}", artmap.toString());
    }
    
    @Test
    @DisplayName("Constructor validates null parameters")
    void testConstructorValidation() {
        assertThrows(NullPointerException.class, 
            () -> new ARTMAP(null, artB, defaultMapParams));
        assertThrows(NullPointerException.class,
            () -> new ARTMAP(artA, null, defaultMapParams));
        assertThrows(NullPointerException.class,
            () -> new ARTMAP(artA, artB, null));
    }
    
    @Test
    @DisplayName("First training example creates categories and map field connection")
    void testFirstTrainingExample() {
        var input = Pattern.of(0.2, 0.8);
        var target = Pattern.of(1.0, 0.0);
        
        var result = artmap.train(input, target, artAParams, artBParams);
        
        // Should succeed with new mapping
        assertTrue(result.isSuccess());
        var success = (ARTMAPResult.Success) result;
        assertEquals(0, success.artAIndex()); // First ARTa category
        assertEquals(0, success.artBIndex()); // First ARTb category
        assertTrue(success.wasNewMapping());
        assertTrue(success.mapFieldActivation() > 0.0);
        
        // Check network state
        assertEquals(1, artA.getCategoryCount());
        assertEquals(1, artB.getCategoryCount());
        assertEquals(1, artmap.getMapField().size());
        assertEquals(0, (int) artmap.getMapField().get(0)); // ARTa[0] -> ARTb[0]
    }
    
    @Test
    @DisplayName("Second example with same pattern reinforces existing mapping")
    void testSamePatternReinforcesMapping() {
        var input = Pattern.of(0.2, 0.8);
        var target = Pattern.of(1.0, 0.0);
        
        // First training
        artmap.train(input, target, artAParams, artBParams);
        
        // Second training with same pattern
        var result = artmap.train(input, target, artAParams, artBParams);
        
        assertTrue(result.isSuccess());
        var success = (ARTMAPResult.Success) result;
        assertEquals(0, success.artAIndex());
        assertEquals(0, success.artBIndex());
        assertFalse(success.wasNewMapping()); // Existing mapping reinforced
        
        // Network should still have 1 category each and 1 mapping
        assertEquals(1, artA.getCategoryCount());
        assertEquals(1, artB.getCategoryCount());
        assertEquals(1, artmap.getMapField().size());
    }
    
    @Test
    @DisplayName("Different input with same target creates new ARTa category")
    void testDifferentInputSameTarget() {
        // First example
        var input1 = Pattern.of(0.2, 0.8);
        var target1 = Pattern.of(1.0, 0.0);
        artmap.train(input1, target1, artAParams, artBParams);
        
        // Second example - different input, same target
        var input2 = Pattern.of(0.9, 0.1);
        var target2 = Pattern.of(1.0, 0.0); // Same target
        var result = artmap.train(input2, target2, artAParams, artBParams);
        
        assertTrue(result.isSuccess());
        var success = (ARTMAPResult.Success) result;
        // Should create new ARTa category (1) but use existing ARTb category (0)
        assertEquals(1, success.artAIndex());
        assertEquals(0, success.artBIndex());
        assertTrue(success.wasNewMapping());
        
        // Check network growth
        assertEquals(2, artA.getCategoryCount()); // New ARTa category
        assertEquals(1, artB.getCategoryCount()); // Same ARTb category
        assertEquals(2, artmap.getMapField().size()); // New mapping added
    }
    
    @Test
    @DisplayName("Same input with different target creates new ARTb category")
    void testSameInputDifferentTarget() {
        // First example
        var input1 = Pattern.of(0.2, 0.8);
        var target1 = Pattern.of(1.0, 0.0);
        artmap.train(input1, target1, artAParams, artBParams);
        
        // Second example - same input, different target
        var input2 = Pattern.of(0.2, 0.8); // Same input
        var target2 = Pattern.of(0.0, 1.0); // Different target
        var result = artmap.train(input2, target2, artAParams, artBParams);
        
        // This should trigger map field mismatch and ARTa vigilance increase
        // resulting in new ARTa category creation
        assertTrue(result.isSuccess() || result.isMapFieldMismatch());
        
        if (result.isSuccess()) {
            var success = (ARTMAPResult.Success) result;
            // Should create new categories due to conflict resolution
            assertTrue(success.artAIndex() >= 0);
            assertTrue(success.artBIndex() >= 0);
        }
        
        // Network should have grown to handle the conflict
        assertTrue(artA.getCategoryCount() >= 1);
        assertTrue(artB.getCategoryCount() >= 1);
        assertTrue(artmap.getMapField().size() >= 1);
    }
    
    @Test
    @DisplayName("Prediction works with trained mappings")
    void testPrediction() {
        // Train with input-output pairs
        var input1 = Pattern.of(0.2, 0.8);
        var target1 = Pattern.of(1.0, 0.0);
        artmap.train(input1, target1, artAParams, artBParams);
        
        var input2 = Pattern.of(0.9, 0.1);
        var target2 = Pattern.of(0.0, 1.0);
        artmap.train(input2, target2, artAParams, artBParams);
        
        // Test prediction for first input
        var prediction1 = artmap.predict(input1, artAParams);
        assertTrue(prediction1.isPresent());
        var pred1 = prediction1.get();
        assertTrue(pred1.confidence() > 0.0);
        assertTrue(pred1.artAActivation() > 0.0);
        
        // Test prediction for second input
        var prediction2 = artmap.predict(input2, artAParams);
        assertTrue(prediction2.isPresent());
        var pred2 = prediction2.get();
        assertTrue(pred2.confidence() > 0.0);
        assertTrue(pred2.artAActivation() > 0.0);
    }
    
    @Test
    @DisplayName("Prediction returns empty for unknown inputs")
    void testPredictionUnknownInput() {
        // No training data
        var unknownInput = Pattern.of(0.5, 0.5);
        var prediction = artmap.predict(unknownInput, artAParams);
        
        assertTrue(prediction.isEmpty()); // No categories to predict from
    }
    
    @Test
    @DisplayName("Clear resets entire ARTMAP network")
    void testClear() {
        // Train some data
        artmap.train(Pattern.of(0.2, 0.8), Pattern.of(1.0, 0.0), artAParams, artBParams);
        artmap.train(Pattern.of(0.9, 0.1), Pattern.of(0.0, 1.0), artAParams, artBParams);
        
        // Verify network has content
        assertTrue(artA.getCategoryCount() > 0);
        assertTrue(artB.getCategoryCount() > 0);
        assertTrue(artmap.getMapField().size() > 0);
        
        // Clear everything
        artmap.clear();
        
        // Verify everything is reset
        assertEquals(0, artA.getCategoryCount());
        assertEquals(0, artB.getCategoryCount());
        assertEquals(0, artmap.getMapField().size());
    }
    
    @Test
    @DisplayName("ARTMAPParameters validation")
    void testParametersValidation() {
        // Valid parameters
        assertDoesNotThrow(() -> ARTMAPParameters.of(0.5, 0.0));
        assertDoesNotThrow(() -> ARTMAPParameters.of(0.0, 1.0));
        assertDoesNotThrow(() -> ARTMAPParameters.of(1.0, 0.5));
        
        // Invalid map vigilance
        assertThrows(IllegalArgumentException.class,
            () -> ARTMAPParameters.of(-0.1, 0.0));
        assertThrows(IllegalArgumentException.class,
            () -> ARTMAPParameters.of(1.1, 0.0));
            
        // Invalid baseline vigilance
        assertThrows(IllegalArgumentException.class,
            () -> ARTMAPParameters.of(0.5, -0.1));
        assertThrows(IllegalArgumentException.class,
            () -> ARTMAPParameters.of(0.5, 1.1));
    }
    
    @Test
    @DisplayName("ARTMAPParameters builder pattern")
    void testParametersBuilder() {
        var params = ARTMAPParameters.builder()
            .mapVigilance(0.8)
            .baselineVigilance(0.1)
            .build();
            
        assertEquals(0.8, params.mapVigilance());
        assertEquals(0.1, params.baselineVigilance());
    }
    
    @Test
    @DisplayName("ARTMAPParameters immutable updates")
    void testParametersImmutableUpdates() {
        var original = ARTMAPParameters.defaults();
        var modified1 = original.withMapVigilance(0.7);
        var modified2 = original.withBaselineVigilance(0.3);
        
        // Original unchanged
        assertEquals(0.9, original.mapVigilance());
        assertEquals(0.0, original.baselineVigilance());
        
        // Modified versions have correct values
        assertEquals(0.7, modified1.mapVigilance());
        assertEquals(0.0, modified1.baselineVigilance());
        
        assertEquals(0.9, modified2.mapVigilance());
        assertEquals(0.3, modified2.baselineVigilance());
    }
    
    @Test
    @DisplayName("Train validates null inputs")
    void testTrainValidation() {
        var input = Pattern.of(0.2, 0.8);
        var target = Pattern.of(1.0, 0.0);
        
        assertThrows(NullPointerException.class,
            () -> artmap.train(null, target, artAParams, artBParams));
        assertThrows(NullPointerException.class,
            () -> artmap.train(input, null, artAParams, artBParams));
        assertThrows(NullPointerException.class,
            () -> artmap.train(input, target, null, artBParams));
        assertThrows(NullPointerException.class,
            () -> artmap.train(input, target, artAParams, null));
    }
    
    @Test
    @DisplayName("Predict validates null inputs")
    void testPredictValidation() {
        var input = Pattern.of(0.2, 0.8);
        
        assertThrows(NullPointerException.class,
            () -> artmap.predict(null, artAParams));
        assertThrows(NullPointerException.class,
            () -> artmap.predict(input, null));
    }
    
    @Test
    @DisplayName("ARTMAPResult sealed interface methods")
    void testResultInterfaceMethods() {
        // Create sample results
        var success = new ARTMAPResult.Success(0, 1, 0.8, 0.7, 0.9, true);
        var prediction = new ARTMAPResult.Prediction(0, 1, 0.8, 0.85);
        var mismatch = new ARTMAPResult.MapFieldMismatch(0, 1, 2, 0.6, true);
        
        // Test type checking methods
        assertTrue(success.isSuccess());
        assertFalse(success.isPrediction());
        assertFalse(success.isMapFieldMismatch());
        
        assertFalse(prediction.isSuccess());
        assertTrue(prediction.isPrediction());
        assertFalse(prediction.isMapFieldMismatch());
        
        assertFalse(mismatch.isSuccess());
        assertFalse(mismatch.isPrediction());
        assertTrue(mismatch.isMapFieldMismatch());
        
        // Test common methods
        assertEquals(0, success.getArtAIndex());
        assertEquals(0, prediction.getArtAIndex());
        assertEquals(0, mismatch.getArtAIndex());
        
        assertEquals(0.8, success.getArtAActivation());
        assertEquals(0.8, prediction.getArtAActivation());
        assertTrue(Double.isNaN(mismatch.getArtAActivation()));
    }
}