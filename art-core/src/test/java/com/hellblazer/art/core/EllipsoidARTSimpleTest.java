/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 * 
 * This file is part of Java ART Neural Networks.
 * 
 * Java ART Neural Networks is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Java ART Neural Networks is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with Java ART Neural Networks. If not, see <https://www.gnu.org/licenses/>.
 */
package com.hellblazer.art.core;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Simple test suite for EllipsoidART to verify test-first development approach.
 * These tests should initially fail with UnsupportedOperationException,
 * then pass as methods are implemented.
 * 
 * @author Hal Hildebrand
 */
class EllipsoidARTSimpleTest {
    
    private EllipsoidART art;
    
    @BeforeEach
    void setUp() {
        var params = new EllipsoidParameters(0.7, 0.1, 2, 0.01, 10.0, 1.5, 100);
        art = new EllipsoidART(params);
    }
    
    @Test
    void testCreation() {
        assertNotNull(art);
        assertEquals(0.7, art.getVigilance(), 1e-6);
        assertEquals(0.1, art.getLearningRate(), 1e-6); 
        assertEquals(2, art.getDimensions());
        assertEquals(0, art.getCategoryCount());
        assertFalse(art.is_fitted());
    }
    
    @Test
    void testParameterValidation() {
        // Test invalid vigilance
        assertThrows(IllegalArgumentException.class, () -> {
            new EllipsoidParameters(-0.1, 0.1, 2, 0.01, 10.0, 1.5, 100);
        });
        
        // Test invalid learning rate
        assertThrows(IllegalArgumentException.class, () -> {
            new EllipsoidParameters(0.7, 1.5, 2, 0.01, 10.0, 1.5, 100);
        });
        
        // Test invalid dimensions
        assertThrows(IllegalArgumentException.class, () -> {
            new EllipsoidParameters(0.7, 0.1, 0, 0.01, 10.0, 1.5, 100);
        });
        
        // Test invalid variance range
        assertThrows(IllegalArgumentException.class, () -> {
            new EllipsoidParameters(0.7, 0.1, 2, 10.0, 5.0, 1.5, 100);
        });
    }
    
    @Test
    void testMahalanobisDistance() {
        var center = new DenseVector(new double[]{0.5, 0.3});
        var covariance = new Matrix(new double[][]{{0.1, 0.02}, {0.02, 0.15}});
        var weight = new EllipsoidWeight(center, covariance, 10);
        var input = new DenseVector(new double[]{0.52, 0.28});
        
        // Now implemented - should return a valid distance
        double distance = art.calculateMahalanobisDistance(input, weight);
        assertTrue(distance >= 0.0, "Mahalanobis distance should be non-negative");
        assertTrue(distance < Double.POSITIVE_INFINITY, "Distance should be finite");
    }
    
    @Test
    void testEllipsoidShapeUpdate() {
        var center = new DenseVector(new double[]{0.0, 0.0});
        var covariance = Matrix.eye(2).multiply(0.1);
        var weight = new EllipsoidWeight(center, covariance, 1);
        var observation = new DenseVector(new double[]{0.3, 0.1});
        var params = new EllipsoidParameters(0.7, 0.5, 2, 0.01, 10.0, 1.5, 100);
        
        // Now implemented - should return an updated weight
        var updatedWeight = art.updateEllipsoidShape(weight, observation, params);
        assertNotNull(updatedWeight);
        assertNotSame(weight, updatedWeight, "Should return a new weight object");
    }
    
    @Test  
    void testVolumeConstraints() {
        var center = new DenseVector(new double[]{0.5, 0.5});
        var largeCov = Matrix.eye(2).multiply(15.0);
        var weight = new EllipsoidWeight(center, largeCov, 5);
        
        // Now implemented - should apply constraints and return constrained weight
        var constrainedWeight = art.applyVolumeConstraints(weight);
        assertNotNull(constrainedWeight);
        // Check that large variances were constrained
        var constrainedCov = constrainedWeight.covariance();
        var params = new EllipsoidParameters(0.7, 0.1, 2, 0.01, 10.0, 1.5, 100);
        for (int i = 0; i < constrainedCov.getRowCount(); i++) {
            assertTrue(constrainedCov.get(i, i) <= params.maxVariance(), "Variance should be constrained to max");
        }
    }
    
    @Test
    void testFitMethod() {
        var trainingData = new double[][]{
            {0.1, 0.2}, {0.15, 0.25}, {0.7, 0.8}
        };
        
        // Now implemented - should fit successfully
        var result = art.fit(trainingData);
        assertNotNull(result);
        assertSame(art, result, "fit should return self for chaining");
        assertTrue(art.is_fitted(), "Model should be fitted after training");
        assertTrue(art.getCategoryCount() > 0, "Should have created at least one category");
    }
    
    @Test
    void testPredictMethod() {
        // First fit some data
        var trainingData = new double[][]{
            {0.1, 0.2}, {0.15, 0.25}, {0.7, 0.8}
        };
        art.fit(trainingData);
        
        var testData = new double[][]{
            {0.12, 0.22}
        };
        
        // Now implemented - should predict successfully
        var predictions = art.predict(testData);
        assertNotNull(predictions);
        assertEquals(1, predictions.length, "Should return one prediction per input");
        assertTrue(predictions[0] >= 0, "Category index should be non-negative");
    }
    
    @Test
    void testScikitClustererInterface() {
        assertTrue(art instanceof ScikitClusterer);
        assertFalse(art.is_fitted());
        
        // get_params() is now implemented
        var params = art.get_params();
        assertNotNull(params);
        
        // cluster_centers() should still throw until the network is trained
        assertThrows(UnsupportedOperationException.class, () -> {
            art.cluster_centers();
        });
    }
}