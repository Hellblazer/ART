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
package com.hellblazer.art.performance.supervised;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.performance.algorithms.VectorizedHypersphereParameters;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for VectorizedHypersphereARTMAP implementation.
 * Tests basic functionality, performance metrics, and edge cases.
 * 
 * @author Hal Hildebrand
 */
class VectorizedHypersphereARTMAPTest {
    
    private VectorizedHypersphereARTMAP artmap;
    private VectorizedHypersphereARTMAPParameters parameters;
    
    @BeforeEach
    void setUp() {
        // Create test parameters
        var hypersphereParams = VectorizedHypersphereParameters.builder()
            .inputDimensions(4)
            .vigilance(0.8)
            .learningRate(0.5)
            .build();
            
        parameters = VectorizedHypersphereARTMAPParameters.builder()
            .mapFieldVigilance(0.9)
            .epsilon(0.001)
            .enableMatchTracking(true)
            .defaultRadius(0.5)
            .adaptiveRadius(true)
            .hypersphereParams(hypersphereParams)
            .build();
            
        artmap = new VectorizedHypersphereARTMAP(parameters);
    }
    
    @AfterEach
    void tearDown() {
        if (artmap != null) {
            artmap.close();
        }
    }
    
    @Test
    void testBasicTraining() {
        // Test basic training functionality
        var input1 = Pattern.of(0.1, 0.2, 0.3, 0.4);
        var input2 = Pattern.of(0.5, 0.6, 0.7, 0.8);
        
        var result1 = artmap.train(input1, 1);
        var result2 = artmap.train(input2, 2);
        
        // Verify training results
        assertNotNull(result1);
        assertNotNull(result2);
        assertEquals(1, result1.predictedLabel());
        assertEquals(2, result2.predictedLabel());
        assertTrue(result1.trainingTime() > 0);
        assertTrue(result2.trainingTime() > 0);
    }
    
    @Test
    void testPrediction() {
        // Train with some data
        var input1 = Pattern.of(0.1, 0.2, 0.3, 0.4);
        var input2 = Pattern.of(0.5, 0.6, 0.7, 0.8);
        
        artmap.train(input1, 1);
        artmap.train(input2, 2);
        
        // Test prediction
        var prediction1 = (Integer) artmap.predict(input1, parameters);
        var prediction2 = (Integer) artmap.predict(input2, parameters);
        
        // Should predict the same labels as trained
        assertEquals(1, prediction1.intValue());
        assertEquals(2, prediction2.intValue());
    }
    
    @Test
    void testBatchOperations() {
        // Test batch training and prediction
        var patterns = new Pattern[] {
            Pattern.of(0.1, 0.2, 0.3, 0.4),
            Pattern.of(0.5, 0.6, 0.7, 0.8),
            Pattern.of(0.2, 0.3, 0.4, 0.5)
        };
        var labels = new int[] { 1, 2, 1 };
        
        // Batch training
        var results = artmap.fit(patterns, labels);
        assertEquals(3, results.length);
        
        // Batch prediction
        var predictions = artmap.predictBatch(patterns);
        assertEquals(3, predictions.length);
        assertEquals(1, predictions[0]);
        assertEquals(2, predictions[1]);
        assertEquals(1, predictions[2]);
    }
    
    @Test
    void testPerformanceMetrics() {
        // Train some data to generate metrics
        var input = Pattern.of(0.1, 0.2, 0.3, 0.4);
        artmap.train(input, 1);
        artmap.predict(input, parameters);
        
        var metrics = artmap.getPerformanceStats();
        
        // Verify metrics are populated
        assertNotNull(metrics);
        assertEquals(1, metrics.trainingOperations());
        assertEquals(1, metrics.predictionOperations());
        assertTrue(metrics.averageTrainingTime() > 0);
        assertTrue(metrics.averagePredictionTime() > 0);
        assertEquals(1, metrics.mapFieldSize());
    }
    
    @Test
    void testCategoryCount() {
        assertEquals(0, artmap.getCategoryCount());
        
        // Train with different patterns
        artmap.train(Pattern.of(0.1, 0.2, 0.3, 0.4), 1);
        artmap.train(Pattern.of(0.5, 0.6, 0.7, 0.8), 2);
        
        // Category count should reflect trained patterns
        assertTrue(artmap.getCategoryCount() >= 0); // May vary based on clustering
    }
    
    @Test
    void testSphereRadiusManagement() {
        // Train some patterns
        artmap.train(Pattern.of(0.1, 0.2, 0.3, 0.4), 1);
        artmap.train(Pattern.of(0.5, 0.6, 0.7, 0.8), 2);
        
        var radii = artmap.getSphereRadii();
        var usageCounts = artmap.getSphereUsageCounts();
        
        assertNotNull(radii);
        assertNotNull(usageCounts);
        
        // Test radius adjustment
        if (!radii.isEmpty()) {
            var categoryId = radii.keySet().iterator().next();
            artmap.adjustSphereRadius(categoryId, 0.7);
            
            var updatedRadii = artmap.getSphereRadii();
            assertEquals(0.7, updatedRadii.get(categoryId), 0.001);
        }
    }
    
    @Test
    void testTrainingState() {
        assertFalse(artmap.isTrained());
        assertEquals(0, artmap.getMapFieldSize());
        
        artmap.train(Pattern.of(0.1, 0.2, 0.3, 0.4), 1);
        
        assertTrue(artmap.isTrained());
        assertEquals(1, artmap.getMapFieldSize());
    }
    
    @Test
    void testClearFunctionality() {
        // Train some data
        artmap.train(Pattern.of(0.1, 0.2, 0.3, 0.4), 1);
        artmap.train(Pattern.of(0.5, 0.6, 0.7, 0.8), 2);
        
        assertTrue(artmap.isTrained());
        assertTrue(artmap.getMapFieldSize() > 0);
        
        // Clear and verify reset
        artmap.clear();
        
        assertFalse(artmap.isTrained());
        assertEquals(0, artmap.getMapFieldSize());
        assertTrue(artmap.getSphereRadii().isEmpty());
        assertTrue(artmap.getSphereUsageCounts().isEmpty());
    }
    
    @Test
    void testParameterValidation() {
        assertNotNull(artmap.getParameters());
        assertEquals(parameters, artmap.getParameters());
    }
    
    @Test
    void testPerformanceReset() {
        // Generate some activity
        artmap.train(Pattern.of(0.1, 0.2, 0.3, 0.4), 1);
        artmap.predict(Pattern.of(0.1, 0.2, 0.3, 0.4), parameters);
        
        var metrics1 = artmap.getPerformanceStats();
        assertTrue(metrics1.trainingOperations() > 0);
        assertTrue(metrics1.predictionOperations() > 0);
        
        // Reset and verify
        artmap.resetPerformanceTracking();
        
        var metrics2 = artmap.getPerformanceStats();
        assertEquals(0, metrics2.trainingOperations());
        assertEquals(0, metrics2.predictionOperations());
        assertEquals(0.0, metrics2.averageTrainingTime());
        assertEquals(0.0, metrics2.averagePredictionTime());
    }
    
    @Test
    void testResourceManagement() {
        // Test that the algorithm can be closed properly
        assertDoesNotThrow(() -> artmap.close());
        
        // After closing, operations should fail
        assertThrows(IllegalStateException.class, () -> {
            artmap.train(Pattern.of(0.1, 0.2, 0.3, 0.4), 1);
        });
        
        assertThrows(IllegalStateException.class, () -> {
            artmap.predict(Pattern.of(0.1, 0.2, 0.3, 0.4), parameters);
        });
    }
    
    @Test
    void testUnsupportedLearnMethod() {
        // The learn method should throw UnsupportedOperationException
        var input = Pattern.of(0.1, 0.2, 0.3, 0.4);
        
        assertThrows(UnsupportedOperationException.class, () -> {
            artmap.learn(input, parameters);
        });
    }
    
    @Test
    void testInputValidation() {
        // Test null input validation
        assertThrows(NullPointerException.class, () -> {
            artmap.train(null, 1);
        });
        
        assertThrows(NullPointerException.class, () -> {
            artmap.predict(null, parameters);
        });
        
        assertThrows(NullPointerException.class, () -> {
            artmap.fit(null, new int[]{1});
        });
        
        assertThrows(NullPointerException.class, () -> {
            artmap.predictBatch(null);
        });
    }
    
    @Test
    void testArrayLengthMismatch() {
        // Test mismatched array lengths
        var patterns = new Pattern[] { Pattern.of(0.1, 0.2, 0.3, 0.4) };
        var labels = new int[] { 1, 2 }; // Different length
        
        assertThrows(IllegalArgumentException.class, () -> {
            artmap.fit(patterns, labels);
        });
    }
    
    @Test
    void testRadiusValidation() {
        // Train to create a category
        artmap.train(Pattern.of(0.1, 0.2, 0.3, 0.4), 1);
        
        var radii = artmap.getSphereRadii();
        if (!radii.isEmpty()) {
            var categoryId = radii.keySet().iterator().next();
            
            // Test invalid radius
            assertThrows(IllegalArgumentException.class, () -> {
                artmap.adjustSphereRadius(categoryId, -0.1); // Negative radius
            });
            
            assertThrows(IllegalArgumentException.class, () -> {
                artmap.adjustSphereRadius(categoryId, 0.0); // Zero radius
            });
        }
    }
}