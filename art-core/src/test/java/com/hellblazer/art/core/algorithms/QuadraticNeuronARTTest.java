/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 * 
 * This file is part of Java ART Neural Networks.
 */
package com.hellblazer.art.core.algorithms;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.parameters.QuadraticNeuronARTParameters;
import com.hellblazer.art.core.weights.QuadraticNeuronARTWeight;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.List;
import java.util.ArrayList;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for QuadraticNeuronART.
 * 
 * Based on the Python reference implementation tests.
 * QuadraticNeuronART clusters data in hyper-ellipsoids by utilizing a quadratic
 * neural network for activation and resonance.
 * 
 * @author Hal Hildebrand
 */
public class QuadraticNeuronARTTest {
    
    private QuadraticNeuronART art;
    private QuadraticNeuronARTParameters params;
    
    @BeforeEach
    void setUp() {
        // Initialize with same parameters as Python test
        params = QuadraticNeuronARTParameters.builder()
            .rho(0.7)     // Vigilance parameter
            .sInit(0.5)         // Initial quadratic term
            .learningRateB(0.1) // Learning rate for cluster mean (bias)
            .learningRateW(0.1) // Learning rate for cluster weights
            .learningRateS(0.05) // Learning rate for the quadratic term
            .build();
        
        art = new QuadraticNeuronART();
    }
    
    @Test
    @DisplayName("Initialization should set correct parameters")
    void testInitialization() {
        assertNotNull(params);
        assertEquals(0.7, params.vigilance());
        assertEquals(0.5, params.sInit());
        assertEquals(0.1, params.learningRateB());
        assertEquals(0.1, params.learningRateW());
        assertEquals(0.05, params.learningRateS());
    }
    
    @Test
    @DisplayName("Parameter validation should accept valid parameters")
    void testValidateParams() {
        // Valid parameters should not throw
        assertDoesNotThrow(() -> params.validate());
        
        // Test invalid vigilance
        assertThrows(IllegalArgumentException.class, () -> 
            QuadraticNeuronARTParameters.builder()
                .rho(1.5) // Invalid: > 1.0
                .sInit(0.5)
                .learningRateB(0.1)
                .learningRateW(0.1)
                .learningRateS(0.05)
                .build()
        );
        
        // Test invalid learning rate B
        assertThrows(IllegalArgumentException.class, () ->
            QuadraticNeuronARTParameters.builder()
                .rho(0.7)
                .sInit(0.5)
                .learningRateB(1.1) // Invalid: > 1.0
                .learningRateW(0.1)
                .learningRateS(0.05)
                .build()
        );
        
        // Test negative learning rate W
        assertThrows(IllegalArgumentException.class, () ->
            QuadraticNeuronARTParameters.builder()
                .rho(0.7)
                .sInit(0.5)
                .learningRateB(0.1)
                .learningRateW(-0.1) // Invalid: < 0.0
                .learningRateS(0.05)
                .build()
        );
        
        // Test invalid learning rate S
        assertThrows(IllegalArgumentException.class, () ->
            QuadraticNeuronARTParameters.builder()
                .rho(0.7)
                .sInit(0.5)
                .learningRateB(0.1)
                .learningRateW(0.1)
                .learningRateS(1.5) // Invalid: > 1.0
                .build()
        );
    }
    
    @Test
    @DisplayName("Category choice should calculate activation correctly")
    void testCategoryChoice() {
        // Create a sample pattern (2D)
        var pattern = Pattern.of(0.2, 0.3);
        
        // Create a mock weight: identity matrix, centroid [0.25, 0.35], s=0.5
        // Weight structure: [matrix (4 values), centroid (2 values), s (1 value)]
        var weightData = new double[] {
            1.0, 0.0,  // First row of matrix
            0.0, 1.0,  // Second row of matrix
            0.25, 0.35, // Centroid (bias)
            0.5         // Quadratic term s
        };
        var weight = new QuadraticNeuronARTWeight(weightData, 2);
        
        // Calculate activation
        double activation = art.calculateActivation(pattern, weight, params);
        
        // Should return a value between 0 and 1
        assertTrue(activation >= 0.0 && activation <= 1.0);
        
        // The activation formula: exp(-s^2 * ||z - b||^2)
        // where z = W * i, b is centroid
        // z = [1,0;0,1] * [0.2,0.3] = [0.2,0.3]
        // ||z - b||^2 = ||(0.2,0.3) - (0.25,0.35)||^2 = 0.05^2 + 0.05^2 = 0.005
        // activation = exp(-0.5^2 * 0.005) = exp(-0.00125) ≈ 0.99875
        double expectedActivation = Math.exp(-0.25 * 0.005);
        assertEquals(expectedActivation, activation, 1e-6);
    }
    
    @Test
    @DisplayName("Vigilance check should work correctly")
    void testVigilanceCheck() {
        var pattern = Pattern.of(0.2, 0.3);
        var weightData = new double[] {
            1.0, 0.0,  
            0.0, 1.0,  
            0.25, 0.35, 
            0.5         
        };
        var weight = new QuadraticNeuronARTWeight(weightData, 2);
        
        // Check vigilance with activation value
        var result = art.checkVigilance(pattern, weight, params);
        assertNotNull(result);
        
        // Activation should be high enough to pass vigilance
        double activation = art.calculateActivation(pattern, weight, params);
        if (activation >= params.vigilance()) {
            assertTrue(result.isAccepted());
        } else {
            assertFalse(result.isAccepted());
        }
    }
    
    @Test
    @DisplayName("Update should modify weight components correctly")
    void testUpdate() {
        var pattern = Pattern.of(0.2, 0.3);
        var weightData = new double[] {
            1.0, 0.0,  
            0.0, 1.0,  
            0.25, 0.35, 
            0.5         
        };
        var weight = new QuadraticNeuronARTWeight(weightData, 2);
        
        // Perform update
        var updatedWeight = (QuadraticNeuronARTWeight) art.updateWeights(pattern, weight, params);
        
        assertNotNull(updatedWeight);
        assertEquals(7, updatedWeight.getData().length);
        
        // The s parameter should have changed (typically decreases)
        double originalS = weight.getS();
        double updatedS = updatedWeight.getS();
        assertNotEquals(originalS, updatedS);
        
        // The centroid should have moved toward the pattern
        double[] originalCentroid = weight.getCentroid();
        double[] updatedCentroid = updatedWeight.getCentroid();
        assertNotNull(updatedCentroid);
        assertEquals(2, updatedCentroid.length);
        
        // Centroid should move toward the input pattern
        // Since original centroid is [0.25, 0.35] and input is [0.2, 0.3],
        // the updated centroid should be closer to [0.2, 0.3]
        double distBefore = Math.sqrt(
            Math.pow(originalCentroid[0] - 0.2, 2) + 
            Math.pow(originalCentroid[1] - 0.3, 2)
        );
        double distAfter = Math.sqrt(
            Math.pow(updatedCentroid[0] - 0.2, 2) + 
            Math.pow(updatedCentroid[1] - 0.3, 2)
        );
        assertTrue(distAfter <= distBefore, "Centroid should move toward input");
    }
    
    @Test
    @DisplayName("New weight should create identity matrix with input as centroid")
    void testNewWeight() {
        var pattern = Pattern.of(0.2, 0.3);
        
        var newWeight = (QuadraticNeuronARTWeight) art.createInitialWeight(pattern, params);
        
        assertNotNull(newWeight);
        assertEquals(7, newWeight.getData().length);
        
        // Check that matrix is identity
        double[][] matrix = newWeight.getMatrix();
        assertEquals(2, matrix.length);
        assertEquals(2, matrix[0].length);
        assertEquals(1.0, matrix[0][0], 1e-10);
        assertEquals(0.0, matrix[0][1], 1e-10);
        assertEquals(0.0, matrix[1][0], 1e-10);
        assertEquals(1.0, matrix[1][1], 1e-10);
        
        // Check that centroid matches input
        double[] centroid = newWeight.getCentroid();
        assertEquals(0.2, centroid[0], 1e-10);
        assertEquals(0.3, centroid[1], 1e-10);
        
        // Check that s equals s_init
        assertEquals(params.sInit(), newWeight.getS(), 1e-10);
    }
    
    @Test
    @DisplayName("Get cluster centers should return centroids")
    void testGetClusterCenters() {
        // Train the model with patterns to create weights
        var pattern1 = Pattern.of(0.2, 0.3);
        var pattern2 = Pattern.of(0.4, 0.5);
        
        art.stepFit(pattern1, params);
        art.stepFit(pattern2, params);
        
        var centers = art.getClusterCenters();
        
        assertTrue(centers.size() > 0);
        // Check that centers match the patterns (or close to them)
        for (var center : centers) {
            assertEquals(2, center.length);
        }
    }
    
    @Test
    @DisplayName("Training should create clusters for data")
    void testTraining() {
        // Create sample data
        var data = List.of(
            Pattern.of(0.1, 0.2),
            Pattern.of(0.3, 0.4),
            Pattern.of(0.5, 0.6)
        );
        
        // Train the model
        for (var pattern : data) {
            art.stepFit(pattern, params);
        }
        
        // Should have created at least one cluster
        assertTrue(art.getCategoryCount() > 0);
        
        // Each weight should be properly formed
        for (var weight : art.getWeights()) {
            assertEquals(7, weight.data().length); // 4 (matrix) + 2 (centroid) + 1 (s)
            assertTrue(weight.getS() > 0); // s should be positive
        }
    }
    
    @Test
    @DisplayName("Step fit should assign patterns to categories")
    void testStepFit() {
        // Train with first pattern
        var pattern1 = Pattern.of(0.1, 0.2);
        var result1 = art.stepFit(pattern1, params);
        
        assertNotNull(result1);
        assertTrue(result1 instanceof com.hellblazer.art.core.results.ActivationResult.Success);
        assertEquals(0, ((com.hellblazer.art.core.results.ActivationResult.Success)result1).categoryIndex());
        
        // Test with similar pattern (should match existing category)
        var pattern2 = Pattern.of(0.11, 0.21);
        var result2 = art.stepFit(pattern2, params);
        
        assertNotNull(result2);
        // Should match existing category if within vigilance
        assertTrue(result2 instanceof com.hellblazer.art.core.results.ActivationResult.Success);
    }
    
    @Test
    @DisplayName("Learning should be incremental")
    void testIncrementalLearning() {
        // Train with first batch
        var batch1 = List.of(
            Pattern.of(0.1, 0.1),
            Pattern.of(0.2, 0.2)
        );
        
        for (var pattern : batch1) {
            art.stepFit(pattern, params);
        }
        
        int categoriesAfterBatch1 = art.getCategoryCount();
        assertTrue(categoriesAfterBatch1 > 0);
        
        // Train with second batch
        var batch2 = List.of(
            Pattern.of(0.8, 0.8),
            Pattern.of(0.9, 0.9)
        );
        
        for (var pattern : batch2) {
            art.stepFit(pattern, params);
        }
        
        int categoriesAfterBatch2 = art.getCategoryCount();
        // Should have more categories after second batch (due to distance)
        assertTrue(categoriesAfterBatch2 >= categoriesAfterBatch1);
    }
    
    @Test
    @DisplayName("Vigilance parameter should control category creation")
    void testVigilanceControl() {
        // Test with high vigilance (more categories)
        var highVigilanceParams = QuadraticNeuronARTParameters.builder()
            .rho(0.95)
            .sInit(0.5)
            .learningRateB(0.1)
            .learningRateW(0.1)
            .learningRateS(0.05)
            .build();
        
        var artHighVigilance = new QuadraticNeuronART();
        
        var data = List.of(
            Pattern.of(0.1, 0.2),
            Pattern.of(0.15, 0.25),
            Pattern.of(0.2, 0.3)
        );
        
        for (var pattern : data) {
            artHighVigilance.stepFit(pattern, highVigilanceParams);
        }
        
        // Test with low vigilance (fewer categories)
        var lowVigilanceParams = QuadraticNeuronARTParameters.builder()
            .rho(0.3)
            .sInit(0.5)
            .learningRateB(0.1)
            .learningRateW(0.1)
            .learningRateS(0.05)
            .build();
        
        var artLowVigilance = new QuadraticNeuronART();
        
        for (var pattern : data) {
            artLowVigilance.stepFit(pattern, lowVigilanceParams);
        }
        
        // High vigilance should create more categories
        assertTrue(artHighVigilance.getCategoryCount() >= artLowVigilance.getCategoryCount());
    }
    
    @Test
    @DisplayName("Weight matrix evolution should be tracked correctly")
    void testWeightMatrixEvolution() {
        var pattern1 = Pattern.of(0.3, 0.4);
        var pattern2 = Pattern.of(0.31, 0.41);
        
        // Learn first pattern
        art.stepFit(pattern1, params);
        assertEquals(1, art.getCategoryCount());
        
        var weight = (QuadraticNeuronARTWeight) art.getCategory(0);
        var matrixBefore = weight.getMatrix();
        var centroidBefore = weight.getCentroid();
        var sBefore = weight.getS();
        
        // Learn similar pattern (should update same category)
        art.stepFit(pattern2, params);
        
        // If only one category, weight was updated
        if (art.getCategoryCount() == 1) {
            var updatedWeight = (QuadraticNeuronARTWeight) art.getCategory(0);
            var matrixAfter = updatedWeight.getMatrix();
            var centroidAfter = updatedWeight.getCentroid();
            var sAfter = updatedWeight.getS();
            
            // Something should have changed
            boolean matrixChanged = !java.util.Arrays.deepEquals(matrixBefore, matrixAfter);
            boolean centroidChanged = !java.util.Arrays.equals(centroidBefore, centroidAfter);
            boolean sChanged = sBefore != sAfter;
            
            assertTrue(matrixChanged || centroidChanged || sChanged,
                      "At least one component should have been updated");
        }
    }
}