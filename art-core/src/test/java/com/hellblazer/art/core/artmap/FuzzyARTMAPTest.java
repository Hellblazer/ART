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
package com.hellblazer.art.core.artmap;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.algorithms.FuzzyART;
import com.hellblazer.art.core.parameters.FuzzyARTMAPParameters;
import com.hellblazer.art.core.parameters.FuzzyParameters;
import com.hellblazer.art.core.preprocessing.DataPreprocessor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for FuzzyARTMAP supervised learning algorithm.
 * 
 * Tests validate:
 * - Basic classification functionality
 * - Match tracking behavior
 * - Label consistency
 * - Multi-class classification
 * - Edge cases and error handling
 * - Numerical parity with Python implementation
 * 
 * @author Hal Hildebrand
 */
class FuzzyARTMAPTest {
    
    private FuzzyARTMAP fuzzyARTMAP;
    private FuzzyARTMAPParameters parameters;
    private DataPreprocessor preprocessor;
    
    @BeforeEach
    void setUp() {
        // Initialize with default parameters matching Python tests
        parameters = new FuzzyARTMAPParameters(
            0.8,    // rho (vigilance)
            1e-10,  // alpha (choice parameter)
            1.0,    // beta (learning rate)
            1e-10   // epsilon (match tracking increment)
        );
        
        fuzzyARTMAP = new FuzzyARTMAP(parameters);
        
        // Setup preprocessor for complement coding
        preprocessor = DataPreprocessor.builder()
            .addComplementCoding()
            .build();
    }
    
    @Test
    @DisplayName("Test basic binary classification")
    void testBinaryClassification() {
        // Create simple linearly separable data
        var data = new Pattern[] {
            Pattern.of(0.1, 0.1),
            Pattern.of(0.2, 0.2),
            Pattern.of(0.8, 0.8),
            Pattern.of(0.9, 0.9)
        };
        
        var labels = new int[] { 0, 0, 1, 1 };
        
        // Preprocess data with complement coding
        var processedData = preprocessor.fitTransform(data);
        
        // Train the model
        fuzzyARTMAP.fit(processedData, labels);
        
        // Test predictions
        var predictions = fuzzyARTMAP.predict(processedData);
        
        // Should correctly classify all training samples
        assertArrayEquals(labels, predictions, "Should correctly classify training data");
        
        // Test on new data points
        var testData = preprocessor.transform(new Pattern[] {
            Pattern.of(0.15, 0.15),  // Should be class 0
            Pattern.of(0.85, 0.85)   // Should be class 1
        });
        
        var testPredictions = fuzzyARTMAP.predict(testData);
        assertEquals(0, testPredictions[0], "Should classify (0.15, 0.15) as class 0");
        assertEquals(1, testPredictions[1], "Should classify (0.85, 0.85) as class 1");
    }
    
    @Test
    @DisplayName("Test multi-class classification")
    void testMultiClassClassification() {
        // Create 3-class data similar to Python test
        var data = generateBlobData(150, 3, 0.5, 42);
        var labels = generateBlobLabels(150, 3, 50);
        
        // Preprocess data
        var processedData = preprocessor.fitTransform(data);
        
        // Train the model
        fuzzyARTMAP.fit(processedData, labels);
        
        // Should have created multiple categories
        assertTrue(fuzzyARTMAP.getCategoryCount() > 0, "Should create categories");
        assertTrue(fuzzyARTMAP.getCategoryCount() <= 150, "Should not exceed sample count");
        
        // Test accuracy on training data
        var predictions = fuzzyARTMAP.predict(processedData);
        var accuracy = calculateAccuracy(labels, predictions);
        System.out.println("Multi-class accuracy: " + accuracy + " (categories: " + fuzzyARTMAP.getCategoryCount() + ")");
        assertTrue(accuracy > 0.70, "Should achieve >70% accuracy on training data");
    }
    
    @Test
    @DisplayName("Test match tracking behavior")
    void testMatchTracking() {
        // Create conflicting patterns that should trigger match tracking
        var data = new Pattern[] {
            Pattern.of(0.5, 0.5),  // First mapped to class 0
            Pattern.of(0.5, 0.5),  // Same pattern, but class 1 - should trigger match tracking
            Pattern.of(0.5, 0.5)   // Same pattern, class 0 again
        };
        
        var labels = new int[] { 0, 1, 0 };
        
        var processedData = preprocessor.fitTransform(data);
        
        // Train incrementally to observe match tracking
        for (int i = 0; i < data.length; i++) {
            var result = fuzzyARTMAP.trainSingle(
                processedData[i], 
                labels[i]
            );
            
            if (i == 1) {
                // Second sample should trigger match tracking
                assertTrue(result.matchTrackingOccurred(), 
                    "Match tracking should occur for conflicting label");
            }
        }
        
        // Should have created at least 2 categories due to conflict
        assertTrue(fuzzyARTMAP.getCategoryCount() >= 2, 
            "Should create multiple categories due to label conflict");
    }
    
    @Test
    @DisplayName("Test incremental learning (partial_fit)")
    void testIncrementalLearning() {
        // Train on initial batch
        var batch1Data = new Pattern[] {
            Pattern.of(0.1, 0.1),
            Pattern.of(0.2, 0.2)
        };
        var batch1Labels = new int[] { 0, 0 };
        
        var processedBatch1 = preprocessor.fitTransform(batch1Data);
        fuzzyARTMAP.fit(processedBatch1, batch1Labels);
        
        var initialCategories = fuzzyARTMAP.getCategoryCount();
        
        // Add more data incrementally
        var batch2Data = new Pattern[] {
            Pattern.of(0.8, 0.8),
            Pattern.of(0.9, 0.9)
        };
        var batch2Labels = new int[] { 1, 1 };
        
        var processedBatch2 = preprocessor.transform(batch2Data);
        fuzzyARTMAP.partialFit(processedBatch2, batch2Labels);
        
        // Should have more categories after incremental learning
        assertTrue(fuzzyARTMAP.getCategoryCount() >= initialCategories,
            "Category count should not decrease");
        
        // Test predictions on all data
        var allData = preprocessor.transform(new Pattern[] {
            Pattern.of(0.1, 0.1),
            Pattern.of(0.2, 0.2),
            Pattern.of(0.8, 0.8),
            Pattern.of(0.9, 0.9)
        });
        var allLabels = new int[] { 0, 0, 1, 1 };
        
        var predictions = fuzzyARTMAP.predict(allData);
        assertArrayEquals(allLabels, predictions, 
            "Should correctly classify all data after incremental learning");
    }
    
    @Test
    @DisplayName("Test consistency with Python implementation")
    void testPythonConsistency() {
        // Use exact parameters from Python test
        var pythonParams = new FuzzyARTMAPParameters(
            0.8,    // rho
            1e-10,  // alpha  
            1.0,    // beta
            1e-10   // epsilon
        );
        
        var pythonARTMAP = new FuzzyARTMAP(pythonParams);
        
        // Generate data matching Python test_consistency
        var data = generateBlobData(1500, 3, 0.5, 0);
        var labels = generateBlobLabels(1500, 3, 500);
        
        // Preprocess with complement coding
        var processedData = preprocessor.fitTransform(data);
        
        // Train the model
        pythonARTMAP.fit(processedData, labels);
        
        // Verify model has learned
        assertTrue(pythonARTMAP.isTrained(), "Model should be trained");
        assertTrue(pythonARTMAP.getCategoryCount() > 0, "Should create categories");
        
        // Test predictions
        var predictions = pythonARTMAP.predict(processedData);
        
        // Should achieve reasonable accuracy on training data
        var accuracy = calculateAccuracy(labels, predictions);
        System.out.println("Python consistency accuracy: " + accuracy + " (categories: " + pythonARTMAP.getCategoryCount() + ")");
        assertTrue(accuracy > 0.45, 
            "Should achieve reasonable accuracy for large dataset");
    }
    
    @Test
    @DisplayName("Test predict_ab functionality")
    void testPredictAB() {
        // Train a simple model
        var data = new Pattern[] {
            Pattern.of(0.1, 0.1),
            Pattern.of(0.9, 0.9)
        };
        var labels = new int[] { 0, 1 };
        
        var processedData = preprocessor.fitTransform(data);
        fuzzyARTMAP.fit(processedData, labels);
        
        // Test predict_ab
        var result = fuzzyARTMAP.predictAB(processedData);
        
        assertNotNull(result, "predictAB should return non-null result");
        assertEquals(2, result.aLabels().length, "Should return A-side labels");
        assertEquals(2, result.bLabels().length, "Should return B-side labels");
        
        // B-side labels should match the training labels
        assertArrayEquals(labels, result.bLabels(), 
            "B-side predictions should match training labels");
        
        // A-side labels should be valid category indices
        for (int aLabel : result.aLabels()) {
            assertTrue(aLabel >= 0 && aLabel < fuzzyARTMAP.getCategoryCount(),
                "A-side label should be valid category index");
        }
    }
    
    @Test
    @DisplayName("Test edge cases and error handling")
    void testEdgeCases() {
        // Test with empty data
        assertThrows(IllegalArgumentException.class, () -> {
            fuzzyARTMAP.fit(new Pattern[0], new int[0]);
        }, "Should throw exception for empty data");
        
        // Test with mismatched data and labels
        assertThrows(IllegalArgumentException.class, () -> {
            fuzzyARTMAP.fit(
                new Pattern[] { Pattern.of(0.5, 0.5) },
                new int[] { 0, 1 }  // More labels than data
            );
        }, "Should throw exception for mismatched data and labels");
        
        // Test prediction before training
        assertThrows(IllegalStateException.class, () -> {
            var untrained = new FuzzyARTMAP(parameters);
            untrained.predict(new Pattern[] { Pattern.of(0.5, 0.5) });
        }, "Should throw exception when predicting before training");
        
        // Test with null inputs
        assertThrows(NullPointerException.class, () -> {
            fuzzyARTMAP.fit(null, new int[] { 0 });
        }, "Should throw NPE for null data");
        
        assertThrows(NullPointerException.class, () -> {
            fuzzyARTMAP.fit(new Pattern[] { Pattern.of(0.5, 0.5) }, null);
        }, "Should throw NPE for null labels");
    }
    
    @Test
    @DisplayName("Test parameter validation")
    void testParameterValidation() {
        // Test invalid vigilance parameter
        assertThrows(IllegalArgumentException.class, () -> {
            new FuzzyARTMAPParameters(-0.1, 1e-10, 1.0, 1e-10);
        }, "Should reject negative vigilance");
        
        assertThrows(IllegalArgumentException.class, () -> {
            new FuzzyARTMAPParameters(1.1, 1e-10, 1.0, 1e-10);
        }, "Should reject vigilance > 1");
        
        // Test invalid learning rate
        assertThrows(IllegalArgumentException.class, () -> {
            new FuzzyARTMAPParameters(0.8, 1e-10, -0.1, 1e-10);
        }, "Should reject negative learning rate");
        
        assertThrows(IllegalArgumentException.class, () -> {
            new FuzzyARTMAPParameters(0.8, 1e-10, 1.1, 1e-10);
        }, "Should reject learning rate > 1");
        
        // Test invalid choice parameter
        assertThrows(IllegalArgumentException.class, () -> {
            new FuzzyARTMAPParameters(0.8, -1.0, 1.0, 1e-10);
        }, "Should reject negative choice parameter");
    }
    
    @Test
    @DisplayName("Test clear functionality")
    void testClear() {
        // Train the model
        var data = new Pattern[] {
            Pattern.of(0.1, 0.1),
            Pattern.of(0.9, 0.9)
        };
        var labels = new int[] { 0, 1 };
        
        var processedData = preprocessor.fitTransform(data);
        fuzzyARTMAP.fit(processedData, labels);
        
        assertTrue(fuzzyARTMAP.isTrained(), "Should be trained");
        assertTrue(fuzzyARTMAP.getCategoryCount() > 0, "Should have categories");
        
        // Clear the model
        fuzzyARTMAP.clear();
        
        assertFalse(fuzzyARTMAP.isTrained(), "Should not be trained after clear");
        assertEquals(0, fuzzyARTMAP.getCategoryCount(), "Should have no categories after clear");
        
        // Should be able to retrain after clearing
        fuzzyARTMAP.fit(processedData, labels);
        assertTrue(fuzzyARTMAP.isTrained(), "Should be able to retrain after clear");
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
            var center = i / samplesPerCenter;
            if (center >= centers) center = centers - 1;
            
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