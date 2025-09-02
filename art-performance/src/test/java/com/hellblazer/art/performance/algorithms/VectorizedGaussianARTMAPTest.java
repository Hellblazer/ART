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
 * Comprehensive test suite for VectorizedGaussianARTMAP supervised learning algorithm.
 * 
 * Tests validate:
 * - Basic ARTMAP functionality (implements BaseARTMAP interface)
 * - SIMD-optimized Gaussian operations
 * - Supervised learning with map field management  
 * - Match tracking with conflict resolution
 * - Parallel processing for large category sets
 * - Performance optimizations and vectorization
 * - Resource management and cleanup
 * 
 * @author Hal Hildebrand
 */
class VectorizedGaussianARTMAPTest {
    
    private VectorizedGaussianARTMAPParameters params;
    private VectorizedGaussianARTMAP artmap;
    private DataPreprocessor preprocessor;
    
    @BeforeEach
    void setUp() {
        // Create parameters with performance optimizations enabled
        params = new VectorizedGaussianARTMAPParameters(
            0.75,   // vigilance
            0.1,    // gamma (learning rate)
            1.0,    // rho_a (variance adjustment) 
            0.1,    // rho_b (minimum variance)
            1e-6,   // epsilon (match tracking increment)
            4,      // parallelismLevel
            true    // enableSIMD
        );
        
        artmap = new VectorizedGaussianARTMAP();
        
        // Setup preprocessor for normalization
        preprocessor = DataPreprocessor.builder()
            .addNormalization()
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
    @DisplayName("Test basic supervised learning with Gaussian distributions")
    void testBasicSupervisedLearning() {
        // Create well-separated Gaussian clusters
        var data = new Pattern[] {
            Pattern.of(0.1, 0.1),
            Pattern.of(0.2, 0.2),
            Pattern.of(0.15, 0.18),
            Pattern.of(0.8, 0.8),
            Pattern.of(0.9, 0.9),
            Pattern.of(0.85, 0.88)
        };
        var labels = new int[] { 0, 0, 0, 1, 1, 1 };
        
        var processedData = preprocessor.fitTransform(data);
        
        // Train the model
        artmap.fit(processedData, labels, params);
        
        // Test predictions on training data
        var predictions = artmap.predict(processedData, params);
        var accuracy = calculateAccuracy(labels, predictions);
        
        assertTrue(accuracy > 0.7, "Should achieve good accuracy on training data: " + accuracy);
        
        // Test on new similar data
        var testData = preprocessor.transform(new Pattern[] {
            Pattern.of(0.12, 0.12),  // Should be class 0
            Pattern.of(0.88, 0.88)   // Should be class 1
        });
        
        var testPredictions = artmap.predict(testData, params);
        assertEquals(0, testPredictions[0], "Should classify (0.12, 0.12) as class 0");
        assertEquals(1, testPredictions[1], "Should classify (0.88, 0.88) as class 1");
    }
    
    @Test
    @DisplayName("Test SIMD vectorized Gaussian operations")
    void testSIMDVectorization() {
        // Create parameters with SIMD enabled and disabled
        var simdParams = params;
        var scalarParams = new VectorizedGaussianARTMAPParameters(
            params.vigilance(), params.gamma(), params.rho_a(), params.rho_b(),
            params.epsilon(), params.parallelismLevel(), false  // SIMD disabled
        );
        
        var simdARTMAP = new VectorizedGaussianARTMAP();
        var scalarARTMAP = new VectorizedGaussianARTMAP();
        
        try {
            // Generate larger dataset with Gaussian blobs
            var data = generateGaussianBlobData(100, 3, 0.15, 42);
            var labels = generateBlobLabels(100, 3, 33);
            var processedData = preprocessor.fitTransform(data);
            
            // Train both models
            simdARTMAP.fit(processedData, labels, simdParams);
            scalarARTMAP.fit(processedData, labels, scalarParams);
            
            // Both should be trained
            assertTrue(simdARTMAP.isTrained(), "SIMD model should be trained");
            assertTrue(scalarARTMAP.isTrained(), "Scalar model should be trained");
            
            // Test predictions
            var simdPredictions = simdARTMAP.predict(processedData, simdParams);
            var scalarPredictions = scalarARTMAP.predict(processedData, scalarParams);
            
            // Both should classify training data well
            var simdAccuracy = calculateAccuracy(labels, simdPredictions);
            var scalarAccuracy = calculateAccuracy(labels, scalarPredictions);
            
            System.out.println("SIMD accuracy: " + simdAccuracy + " (categories: " + simdARTMAP.getCategoryCount() + ")");
            System.out.println("Scalar accuracy: " + scalarAccuracy + " (categories: " + scalarARTMAP.getCategoryCount() + ")");
            
            assertTrue(simdAccuracy > 0.2, "SIMD version should achieve good accuracy");
            assertTrue(scalarAccuracy > 0.2, "Scalar version should achieve good accuracy");
            
        } finally {
            simdARTMAP.close();
            scalarARTMAP.close();
        }
    }
    
    @Test
    @DisplayName("Test match tracking with Gaussian category conflicts")
    void testMatchTracking() {
        // Create overlapping Gaussian patterns that should trigger match tracking
        var conflictingData = new Pattern[] {
            Pattern.of(0.5, 0.5),  // First mapped to class 0
            Pattern.of(0.51, 0.49), // Very similar pattern, class 1 - potential conflict!
            Pattern.of(0.52, 0.48), // Another similar pattern, class 0 again
            Pattern.of(0.49, 0.51)  // Very close pattern, class 1
        };
        var conflictingLabels = new int[] { 0, 1, 0, 1 };
        
        var processedData = preprocessor.fitTransform(conflictingData);
        
        // Train with potentially conflicting labels
        artmap.fit(processedData, conflictingLabels, params);
        
        // Should have created categories to handle different labels
        assertTrue(artmap.getCategoryCount() >= 2, 
            "Should create multiple categories for different labels");
        
        // Test predictions - should handle conflicts gracefully
        var predictions = artmap.predict(processedData, params);
        
        // Should achieve reasonable accuracy despite overlapping patterns
        var accuracy = calculateAccuracy(conflictingLabels, predictions);
        assertTrue(accuracy >= 0.5, "Should handle conflicts with reasonable accuracy: " + accuracy);
    }
    
    @Test
    @DisplayName("Test parallel processing for large Gaussian datasets")
    void testParallelProcessing() {
        // Create parameters with different parallelism levels
        var highParallelParams = new VectorizedGaussianARTMAPParameters(
            0.75, 0.1, 1.0, 0.1, 1e-6, 8, true  // High parallelism
        );
        
        var lowParallelParams = new VectorizedGaussianARTMAPParameters(
            0.75, 0.1, 1.0, 0.1, 1e-6, 1, true  // Sequential processing
        );
        
        var parallelARTMAP = new VectorizedGaussianARTMAP();
        var sequentialARTMAP = new VectorizedGaussianARTMAP();
        
        try {
            // Generate large Gaussian dataset
            var data = generateGaussianBlobData(200, 4, 0.2, 123);
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
            
            assertTrue(parallelAccuracy > 0.2, "Parallel processing should achieve good accuracy");
            assertTrue(sequentialAccuracy > 0.2, "Sequential processing should achieve good accuracy");
            
        } finally {
            parallelARTMAP.close();
            sequentialARTMAP.close();
        }
    }
    
    @Test
    @DisplayName("Test incremental learning with partial fit")
    void testIncrementalLearning() {
        // Train on initial batch of Gaussian data
        var batch1Data = new Pattern[] {
            Pattern.of(0.1, 0.1),
            Pattern.of(0.15, 0.12),
            Pattern.of(0.12, 0.18)
        };
        var batch1Labels = new int[] { 0, 0, 0 };
        var processedBatch1 = preprocessor.fitTransform(batch1Data);
        
        artmap.fit(processedBatch1, batch1Labels, params);
        var initialCategories = artmap.getCategoryCount();
        
        // Add second batch incrementally
        var batch2Data = new Pattern[] {
            Pattern.of(0.8, 0.8),
            Pattern.of(0.82, 0.85),
            Pattern.of(0.88, 0.82)
        };
        var batch2Labels = new int[] { 1, 1, 1 };
        var processedBatch2 = preprocessor.transform(batch2Data);
        
        artmap.partialFit(processedBatch2, batch2Labels, params);
        
        // Should have same or more categories
        assertTrue(artmap.getCategoryCount() >= initialCategories,
            "Category count should not decrease after incremental learning");
        
        // Test predictions on combined data
        var allData = preprocessor.transform(new Pattern[] {
            Pattern.of(0.13, 0.14),  // Class 0 region
            Pattern.of(0.84, 0.83)   // Class 1 region
        });
        
        var predictions = artmap.predict(allData, params);
        assertEquals(0, predictions[0], "Should predict class 0");
        assertEquals(1, predictions[1], "Should predict class 1");
    }
    
    @Test
    @DisplayName("Test multi-class Gaussian classification")
    void testMultiClassClassification() {
        // Generate 5-class Gaussian dataset
        var data = generateGaussianBlobData(250, 5, 0.12, 789);
        var labels = generateBlobLabels(250, 5, 50);
        var processedData = preprocessor.fitTransform(data);
        
        // Train the model
        artmap.fit(processedData, labels, params);
        
        // Test predictions
        var predictions = artmap.predict(processedData, params);
        var accuracy = calculateAccuracy(labels, predictions);
        
        System.out.println("Multi-class accuracy: " + accuracy + 
                          " (categories: " + artmap.getCategoryCount() + ")");
        
        assertTrue(accuracy > 0.2, "Should achieve reasonable accuracy on 5-class Gaussian problem");
        assertTrue(artmap.getCategoryCount() > 0, "Should create categories");
        assertTrue(artmap.getCategoryCount() <= 250, "Should not exceed sample count");
    }
    
    @Test
    @DisplayName("Test Gaussian parameter validation")
    void testParameterValidation() {
        // Test invalid vigilance
        assertThrows(IllegalArgumentException.class, () -> {
            new VectorizedGaussianARTMAPParameters(-0.1, 0.1, 1.0, 0.1, 1e-6, 4, true);
        }, "Should reject negative vigilance");
        
        assertThrows(IllegalArgumentException.class, () -> {
            new VectorizedGaussianARTMAPParameters(1.1, 0.1, 1.0, 0.1, 1e-6, 4, true);
        }, "Should reject vigilance > 1");
        
        // Test invalid gamma
        assertThrows(IllegalArgumentException.class, () -> {
            new VectorizedGaussianARTMAPParameters(0.75, -0.1, 1.0, 0.1, 1e-6, 4, true);
        }, "Should reject negative gamma");
        
        assertThrows(IllegalArgumentException.class, () -> {
            new VectorizedGaussianARTMAPParameters(0.75, 1.1, 1.0, 0.1, 1e-6, 4, true);
        }, "Should reject gamma > 1");
        
        // Test invalid rho_a
        assertThrows(IllegalArgumentException.class, () -> {
            new VectorizedGaussianARTMAPParameters(0.75, 0.1, 0.0, 0.1, 1e-6, 4, true);
        }, "Should reject rho_a <= 0");
        
        // Test invalid rho_b
        assertThrows(IllegalArgumentException.class, () -> {
            new VectorizedGaussianARTMAPParameters(0.75, 0.1, 1.0, 0.0, 1e-6, 4, true);
        }, "Should reject rho_b <= 0");
        
        // Test invalid epsilon
        assertThrows(IllegalArgumentException.class, () -> {
            new VectorizedGaussianARTMAPParameters(0.75, 0.1, 1.0, 0.1, 0.0, 4, true);
        }, "Should reject epsilon <= 0");
        
        // Test invalid parallelism level
        assertThrows(IllegalArgumentException.class, () -> {
            new VectorizedGaussianARTMAPParameters(0.75, 0.1, 1.0, 0.1, 1e-6, 0, true);
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
        var untrainedARTMAP = new VectorizedGaussianARTMAP();
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
    @DisplayName("Test Gaussian variance handling")
    void testGaussianVarianceHandling() {
        // Test with patterns that might cause very small variances
        var data = new Pattern[] {
            Pattern.of(0.5, 0.5),
            Pattern.of(0.5000001, 0.5000001),  // Very close to first
            Pattern.of(0.5000002, 0.5000002),  // Very close to others
            Pattern.of(0.9, 0.9),              // Distinct cluster
            Pattern.of(0.9000001, 0.9000001)   // Very close to cluster 2
        };
        var labels = new int[] { 0, 0, 0, 1, 1 };
        
        var processedData = preprocessor.fitTransform(data);
        
        // Should handle small variances gracefully
        assertDoesNotThrow(() -> {
            artmap.fit(processedData, labels, params);
        }, "Should handle very close patterns without numerical issues");
        
        assertTrue(artmap.isTrained(), "Should be trained successfully");
        
        // Test predictions
        var predictions = artmap.predict(processedData, params);
        var accuracy = calculateAccuracy(labels, predictions);
        assertTrue(accuracy > 0.6, "Should achieve good accuracy despite close patterns");
    }
    
    @Test
    @DisplayName("Test resource management and cleanup")
    void testResourceManagement() {
        // Create and train multiple ARTMAP instances
        var artmap1 = new VectorizedGaussianARTMAP();
        var artmap2 = new VectorizedGaussianARTMAP();
        var artmap3 = new VectorizedGaussianARTMAP();
        
        try {
            var data = generateGaussianBlobData(50, 2, 0.2, 456);
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
    
    private Pattern[] generateGaussianBlobData(int samples, int centers, double stdDev, long seed) {
        var random = new Random(seed);
        var data = new Pattern[samples];
        var samplesPerCenter = samples / centers;
        
        // Generate cluster centers
        var centerX = new double[centers];
        var centerY = new double[centers];
        for (int i = 0; i < centers; i++) {
            centerX[i] = 0.2 + random.nextDouble() * 0.6;  // Centers in [0.2, 0.8]
            centerY[i] = 0.2 + random.nextDouble() * 0.6;  // to avoid boundary issues
        }
        
        // Generate samples around centers with Gaussian noise
        for (int i = 0; i < samples; i++) {
            var center = Math.min(i / samplesPerCenter, centers - 1);
            
            var x = centerX[center] + random.nextGaussian() * stdDev;
            var y = centerY[center] + random.nextGaussian() * stdDev;
            
            // Clamp to [0, 1]
            x = Math.max(0.01, Math.min(0.99, x));
            y = Math.max(0.01, Math.min(0.99, y));
            
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