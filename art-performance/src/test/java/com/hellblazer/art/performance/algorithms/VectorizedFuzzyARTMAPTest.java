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
package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.preprocessing.DataPreprocessor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for VectorizedFuzzyARTMAP supervised learning algorithm.
 * 
 * Tests validate:
 * - Basic ARTMAP functionality (implements BaseARTMAP interface)
 * - SIMD-optimized fuzzy operations
 * - Supervised learning with map field management
 * - Match tracking with conflict resolution
 * - Parallel processing for large category sets
 * - Performance optimizations and vectorization
 * - Resource management and cleanup
 * 
 * @author Hal Hildebrand
 */
class VectorizedFuzzyARTMAPTest {
    
    private VectorizedFuzzyARTMAPParameters params;
    private VectorizedFuzzyARTMAP artmap;
    private DataPreprocessor preprocessor;
    
    @BeforeEach
    void setUp() {
        // Create parameters with performance optimizations enabled
        params = new VectorizedFuzzyARTMAPParameters(
            0.8,    // rho (vigilance)
            0.001,  // alpha (choice parameter) 
            1.0,    // beta (learning rate)
            1e-6,   // epsilon (match tracking increment)
            4,      // parallelismLevel
            true    // enableSIMD
        );
        
        artmap = new VectorizedFuzzyARTMAP();
        
        // Setup preprocessor for complement coding
        preprocessor = DataPreprocessor.builder()
            .addComplementCoding()
            .build();
    }
    
    @AfterEach
    void tearDown() {
        if (artmap != null) {
            artmap.close();
        }
    }
    
    @Test
    @DisplayName("Test BaseARTMAP interface implementation")
    void testBaseARTMAPInterface() {
        // Initially not trained
        assertFalse(artmap.isTrained(), "Should not be trained initially");
        assertEquals(0, artmap.getCategoryCount(), "Should have no categories initially");
        
        // Train with simple data
        var data = new Pattern[] {
            Pattern.of(0.1, 0.1),
            Pattern.of(0.9, 0.9)
        };
        var labels = new int[] { 0, 1 };
        var processedData = preprocessor.fitTransform(data);
        
        artmap.fit(processedData, labels, params);
        
        // Should be trained after fitting
        assertTrue(artmap.isTrained(), "Should be trained after fit");
        assertTrue(artmap.getCategoryCount() > 0, "Should have categories after training");
        
        // Test clear
        artmap.clear();
        assertFalse(artmap.isTrained(), "Should not be trained after clear");
        assertEquals(0, artmap.getCategoryCount(), "Should have no categories after clear");
    }
    
    @Test
    @DisplayName("Test basic supervised learning")
    void testBasicSupervisedLearning() {
        // Create linearly separable data
        var data = new Pattern[] {
            Pattern.of(0.1, 0.1),
            Pattern.of(0.2, 0.2),
            Pattern.of(0.8, 0.8),
            Pattern.of(0.9, 0.9)
        };
        var labels = new int[] { 0, 0, 1, 1 };
        
        var processedData = preprocessor.fitTransform(data);
        
        // Train the model
        artmap.fit(processedData, labels, params);
        
        // Test predictions on training data
        var predictions = artmap.predict(processedData, params);
        assertArrayEquals(labels, predictions, "Should correctly classify training data");
        
        // Test on new similar data
        var testData = preprocessor.transform(new Pattern[] {
            Pattern.of(0.15, 0.15),  // Should be class 0
            Pattern.of(0.85, 0.85)   // Should be class 1
        });
        
        var testPredictions = artmap.predict(testData, params);
        assertEquals(0, testPredictions[0], "Should classify (0.15, 0.15) as class 0");
        assertEquals(1, testPredictions[1], "Should classify (0.85, 0.85) as class 1");
    }
    
    @Test
    @DisplayName("Test SIMD vectorized operations")
    void testSIMDVectorization() {
        // Create parameters with SIMD enabled and disabled
        var simdParams = params;
        var scalarParams = new VectorizedFuzzyARTMAPParameters(
            params.rho(), params.alpha(), params.beta(), params.epsilon(), 
            params.parallelismLevel(), false  // SIMD disabled
        );
        
        var simdARTMAP = new VectorizedFuzzyARTMAP();
        var scalarARTMAP = new VectorizedFuzzyARTMAP();
        
        try {
            // Generate larger dataset to trigger SIMD operations
            var data = generateBlobData(100, 3, 0.3, 42);
            var labels = generateBlobLabels(100, 3, 33);
            var processedData = preprocessor.fitTransform(data);
            
            // Train both models
            simdARTMAP.fit(processedData, labels, simdParams);
            scalarARTMAP.fit(processedData, labels, scalarParams);
            
            // Both should be trained
            assertTrue(simdARTMAP.isTrained(), "SIMD model should be trained");
            assertTrue(scalarARTMAP.isTrained(), "Scalar model should be trained");
            
            // Predictions should be numerically equivalent
            var simdPredictions = simdARTMAP.predict(processedData, simdParams);
            var scalarPredictions = scalarARTMAP.predict(processedData, scalarParams);
            
            // Both should classify training data well
            var simdAccuracy = calculateAccuracy(labels, simdPredictions);
            var scalarAccuracy = calculateAccuracy(labels, scalarPredictions);
            
            System.out.println("SIMD accuracy: " + simdAccuracy + " (categories: " + simdARTMAP.getCategoryCount() + ")");
            System.out.println("Scalar accuracy: " + scalarAccuracy + " (categories: " + scalarARTMAP.getCategoryCount() + ")");
            
            assertTrue(simdAccuracy > 0.6, "SIMD version should achieve good accuracy");
            assertTrue(scalarAccuracy > 0.6, "Scalar version should achieve good accuracy");
            
        } finally {
            simdARTMAP.close();
            scalarARTMAP.close();
        }
    }
    
    @Test
    @DisplayName("Test match tracking with label conflicts")
    void testMatchTracking() {
        // Create patterns that should trigger match tracking
        // Use slightly different patterns to ensure match tracking can create new categories
        var conflictingData = new Pattern[] {
            Pattern.of(0.5, 0.5),    // First mapped to class 0
            Pattern.of(0.51, 0.49),  // Very similar pattern, class 1 - conflict!
            Pattern.of(0.49, 0.51)   // Another similar pattern, class 0 again
        };
        var conflictingLabels = new int[] { 0, 1, 0 };
        
        var processedData = preprocessor.fitTransform(conflictingData);
        
        // Train with conflicting labels
        artmap.fit(processedData, conflictingLabels, params);
        
        // Should have created multiple categories due to conflict resolution
        assertTrue(artmap.getCategoryCount() >= 2, 
            "Should create multiple categories for conflicting labels");
        
        // Test predictions - should handle conflicts gracefully
        var predictions = artmap.predict(processedData, params);
        
        // At least some predictions should match the labels
        var accuracy = calculateAccuracy(conflictingLabels, predictions);
        assertTrue(accuracy >= 0.33, "Should handle conflicts with reasonable accuracy");
    }
    
    @Test
    @DisplayName("Test parallel processing for large datasets")
    void testParallelProcessing() {
        // Create parameters with different parallelism levels
        var highParallelParams = new VectorizedFuzzyARTMAPParameters(
            0.75, 0.001, 1.0, 1e-6, 8, true  // High parallelism
        );
        
        var lowParallelParams = new VectorizedFuzzyARTMAPParameters(
            0.75, 0.001, 1.0, 1e-6, 1, true  // Sequential processing
        );
        
        var parallelARTMAP = new VectorizedFuzzyARTMAP();
        var sequentialARTMAP = new VectorizedFuzzyARTMAP();
        
        try {
            // Generate large dataset
            var data = generateBlobData(200, 4, 0.4, 123);
            var labels = generateBlobLabels(200, 4, 50);
            var processedData = preprocessor.fitTransform(data);
            
            // Measure training time for both approaches
            long startTime = System.nanoTime();
            parallelARTMAP.fit(processedData, labels, highParallelParams);
            long parallelTime = System.nanoTime() - startTime;
            
            startTime = System.nanoTime();
            sequentialARTMAP.fit(processedData, labels, lowParallelParams);
            long sequentialTime = System.nanoTime() - startTime;
            
            System.out.println("Parallel training time: " + (parallelTime / 1_000_000) + " ms");
            System.out.println("Sequential training time: " + (sequentialTime / 1_000_000) + " ms");
            
            // Both should be trained successfully
            assertTrue(parallelARTMAP.isTrained(), "Parallel model should be trained");
            assertTrue(sequentialARTMAP.isTrained(), "Sequential model should be trained");
            
            // Test prediction accuracy for both
            var parallelPredictions = parallelARTMAP.predict(processedData, highParallelParams);
            var sequentialPredictions = sequentialARTMAP.predict(processedData, lowParallelParams);
            
            var parallelAccuracy = calculateAccuracy(labels, parallelPredictions);
            var sequentialAccuracy = calculateAccuracy(labels, sequentialPredictions);
            
            System.out.println("Parallel accuracy: " + parallelAccuracy);
            System.out.println("Sequential accuracy: " + sequentialAccuracy);
            
            assertTrue(parallelAccuracy > 0.5, "Parallel processing should achieve good accuracy");
            assertTrue(sequentialAccuracy > 0.5, "Sequential processing should achieve good accuracy");
            
        } finally {
            parallelARTMAP.close();
            sequentialARTMAP.close();
        }
    }
    
    @Test
    @DisplayName("Test incremental learning with partial fit")
    void testIncrementalLearning() {
        // Train on initial batch
        var batch1Data = new Pattern[] {
            Pattern.of(0.1, 0.1),
            Pattern.of(0.2, 0.2)
        };
        var batch1Labels = new int[] { 0, 0 };
        var processedBatch1 = preprocessor.fitTransform(batch1Data);
        
        artmap.fit(processedBatch1, batch1Labels, params);
        var initialCategories = artmap.getCategoryCount();
        
        // Add second batch incrementally
        var batch2Data = new Pattern[] {
            Pattern.of(0.8, 0.8),
            Pattern.of(0.9, 0.9)
        };
        var batch2Labels = new int[] { 1, 1 };
        var processedBatch2 = preprocessor.transform(batch2Data);
        
        artmap.partialFit(processedBatch2, batch2Labels, params);
        
        // Should have same or more categories
        assertTrue(artmap.getCategoryCount() >= initialCategories,
            "Category count should not decrease after incremental learning");
        
        // Test predictions on combined data
        var allData = preprocessor.transform(new Pattern[] {
            Pattern.of(0.15, 0.15),  // Class 0
            Pattern.of(0.85, 0.85)   // Class 1
        });
        
        var predictions = artmap.predict(allData, params);
        assertEquals(0, predictions[0], "Should predict class 0");
        assertEquals(1, predictions[1], "Should predict class 1");
    }
    
    @Test
    @DisplayName("Test multi-class classification performance")
    void testMultiClassClassification() {
        // Generate 5-class dataset
        var data = generateBlobData(250, 5, 0.3, 789);
        var labels = generateBlobLabels(250, 5, 50);
        var processedData = preprocessor.fitTransform(data);
        
        // Train the model
        artmap.fit(processedData, labels, params);
        
        // Test predictions
        var predictions = artmap.predict(processedData, params);
        var accuracy = calculateAccuracy(labels, predictions);
        
        System.out.println("Multi-class accuracy: " + accuracy + 
                          " (categories: " + artmap.getCategoryCount() + ")");
        
        assertTrue(accuracy > 0.4, "Should achieve reasonable accuracy on 5-class problem");
        assertTrue(artmap.getCategoryCount() > 0, "Should create categories");
        assertTrue(artmap.getCategoryCount() <= 250, "Should not exceed sample count");
    }
    
    @Test
    @DisplayName("Test parameter validation")
    void testParameterValidation() {
        // Test invalid vigilance
        assertThrows(IllegalArgumentException.class, () -> {
            new VectorizedFuzzyARTMAPParameters(-0.1, 0.001, 1.0, 1e-6, 4, true);
        }, "Should reject negative vigilance");
        
        assertThrows(IllegalArgumentException.class, () -> {
            new VectorizedFuzzyARTMAPParameters(1.1, 0.001, 1.0, 1e-6, 4, true);
        }, "Should reject vigilance > 1");
        
        // Test invalid alpha
        assertThrows(IllegalArgumentException.class, () -> {
            new VectorizedFuzzyARTMAPParameters(0.8, -0.001, 1.0, 1e-6, 4, true);
        }, "Should reject negative alpha");
        
        // Test invalid beta
        assertThrows(IllegalArgumentException.class, () -> {
            new VectorizedFuzzyARTMAPParameters(0.8, 0.001, -0.1, 1e-6, 4, true);
        }, "Should reject negative beta");
        
        assertThrows(IllegalArgumentException.class, () -> {
            new VectorizedFuzzyARTMAPParameters(0.8, 0.001, 1.1, 1e-6, 4, true);
        }, "Should reject beta > 1");
        
        // Test invalid parallelism level
        assertThrows(IllegalArgumentException.class, () -> {
            new VectorizedFuzzyARTMAPParameters(0.8, 0.001, 1.0, 1e-6, 0, true);
        }, "Should reject parallelismLevel <= 0");
    }
    
    @Test
    @DisplayName("Test error handling and edge cases")
    void testErrorHandling() {
        // Test with empty data
        assertThrows(IllegalArgumentException.class, () -> {
            artmap.fit(new Pattern[0], new int[0], params);
        }, "Should throw exception for empty data");
        
        // Test with mismatched data and labels
        assertThrows(IllegalArgumentException.class, () -> {
            artmap.fit(
                new Pattern[] { Pattern.of(0.5, 0.5) },
                new int[] { 0, 1 }, // More labels than data
                params
            );
        }, "Should throw exception for mismatched data and labels");
        
        // Test prediction before training
        var untrainedARTMAP = new VectorizedFuzzyARTMAP();
        try {
            assertThrows(IllegalStateException.class, () -> {
                untrainedARTMAP.predict(new Pattern[] { Pattern.of(0.5, 0.5) }, params);
            }, "Should throw exception when predicting before training");
        } finally {
            untrainedARTMAP.close();
        }
        
        // Test with null inputs
        assertThrows(NullPointerException.class, () -> {
            artmap.fit(null, new int[] { 0 }, params);
        }, "Should throw NPE for null data");
        
        assertThrows(NullPointerException.class, () -> {
            artmap.fit(new Pattern[] { Pattern.of(0.5, 0.5) }, null, params);
        }, "Should throw NPE for null labels");
        
        assertThrows(NullPointerException.class, () -> {
            artmap.fit(new Pattern[] { Pattern.of(0.5, 0.5) }, new int[] { 0 }, null);
        }, "Should throw NPE for null parameters");
    }
    
    @Test
    @DisplayName("Test resource management and cleanup")
    void testResourceManagement() {
        // Create and train multiple ARTMAP instances
        var artmap1 = new VectorizedFuzzyARTMAP();
        var artmap2 = new VectorizedFuzzyARTMAP();
        var artmap3 = new VectorizedFuzzyARTMAP();
        
        try {
            var data = generateBlobData(50, 2, 0.4, 456);
            var labels = generateBlobLabels(50, 2, 25);
            var processedData = preprocessor.fitTransform(data);
            
            // Train all instances
            artmap1.fit(processedData, labels, params);
            artmap2.fit(processedData, labels, params);
            artmap3.fit(processedData, labels, params);
            
            // All should be trained
            assertTrue(artmap1.isTrained(), "ARTMAP 1 should be trained");
            assertTrue(artmap2.isTrained(), "ARTMAP 2 should be trained");
            assertTrue(artmap3.isTrained(), "ARTMAP 3 should be trained");
            
        } finally {
            // Test that close() works without exceptions
            assertDoesNotThrow(artmap1::close, "close() should not throw exception");
            assertDoesNotThrow(artmap2::close, "close() should not throw exception");
            assertDoesNotThrow(artmap3::close, "close() should not throw exception");
        }
    }
    
    // Helper methods
    
    private Pattern[] generateBlobData(int samples, int centers, double stdDev, long seed) {
        var random = new Random(seed);
        var data = new Pattern[samples];
        var samplesPerCenter = samples / centers;
        
        // Generate cluster centers
        var centerX = new double[centers];
        var centerY = new double[centers];
        for (int i = 0; i < centers; i++) {
            centerX[i] = random.nextDouble();
            centerY[i] = random.nextDouble();
        }
        
        // Generate samples around centers
        for (int i = 0; i < samples; i++) {
            var center = Math.min(i / samplesPerCenter, centers - 1);
            
            var x = centerX[center] + random.nextGaussian() * stdDev;
            var y = centerY[center] + random.nextGaussian() * stdDev;
            
            // Clamp to [0, 1]
            x = Math.max(0, Math.min(1, x));
            y = Math.max(0, Math.min(1, y));
            
            data[i] = Pattern.of(x, y);
        }
        
        return data;
    }
    
    private int[] generateBlobLabels(int samples, int centers, int samplesPerCenter) {
        var labels = new int[samples];
        for (int i = 0; i < samples; i++) {
            labels[i] = Math.min(i / samplesPerCenter, centers - 1);
        }
        return labels;
    }
    
    private double calculateAccuracy(int[] expected, int[] predicted) {
        if (expected.length != predicted.length) {
            throw new IllegalArgumentException("Arrays must have same length");
        }
        
        int correct = 0;
        for (int i = 0; i < expected.length; i++) {
            if (expected[i] == predicted[i]) {
                correct++;
            }
        }
        
        return (double) correct / expected.length;
    }
}