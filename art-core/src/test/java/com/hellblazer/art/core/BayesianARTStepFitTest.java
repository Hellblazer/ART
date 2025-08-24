/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 */
package com.hellblazer.art.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test actual BayesianART stepFit functionality
 */
public class BayesianARTStepFitTest {
    
    @Test
    public void testSingleStepFit() {
        // Create parameters
        var priorMean = new double[]{0.0, 0.0};
        var priorCovMatrix = new Matrix(2, 2);
        priorCovMatrix.set(0, 0, 1.0);
        priorCovMatrix.set(1, 1, 1.0);
        
        var params = new BayesianParameters(
            0.7, priorMean, priorCovMatrix, 0.1, 1.0, 100
        );
        
        var bayesianART = new BayesianART(params);
        
        // Test first pattern - should create first category
        var pattern1 = Pattern.of(1.0, 2.0);
        var result1 = bayesianART.stepFit(pattern1, params);
        
        assertNotNull(result1);
        assertTrue(result1 instanceof ActivationResult.Success);
        
        var success1 = (ActivationResult.Success) result1;
        assertEquals(0, success1.categoryIndex()); // First category
        assertEquals(1, bayesianART.getCategoryCount());
        
        // Verify the weight is a BayesianWeight
        var weight1 = success1.updatedWeight();
        assertTrue(weight1 instanceof BayesianWeight);
        
        var bayesianWeight1 = (BayesianWeight) weight1;
        assertEquals(1.0, bayesianWeight1.mean().get(0), 1e-6);
        assertEquals(2.0, bayesianWeight1.mean().get(1), 1e-6);
    }
}