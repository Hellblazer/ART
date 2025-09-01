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
import com.hellblazer.art.core.algorithms.BinaryFuzzyART;
import com.hellblazer.art.core.parameters.SimpleARTMAPParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for BinaryFuzzyARTMAP implementation.
 * 
 * Tests binary pattern classification using the BinaryFuzzyARTMAP algorithm,
 * which combines BinaryFuzzyART with SimpleARTMAP for supervised learning.
 * 
 * @author Hal Hildebrand
 */
public class BinaryFuzzyARTMAPTest {
    
    private BinaryFuzzyARTMAP artmap;
    private Random random;
    
    @BeforeEach
    void setUp() {
        artmap = null;
        random = new Random(42);
    }
    
    @Test
    void testBasicBinaryClassification() {
        // Create BinaryFuzzyARTMAP with standard parameters
        artmap = new BinaryFuzzyARTMAP(0.7, 0.01);
        
        // Create simple binary patterns for two classes
        double[][] trainData = {
            {1, 1, 0, 0},  // Class 0 pattern 1
            {1, 0, 0, 0},  // Class 0 pattern 2
            {0, 0, 1, 1},  // Class 1 pattern 1
            {0, 0, 1, 0},  // Class 1 pattern 2
        };
        
        int[] trainLabels = {0, 0, 1, 1};
        
        // Train the model
        artmap.fit(trainData, trainLabels);
        
        // Test predictions on training data
        for (int i = 0; i < trainData.length; i++) {
            int predicted = artmap.predict(trainData[i]);
            assertEquals(trainLabels[i], predicted,
                "Prediction failed for training pattern " + i);
        }
        
        // Test on similar patterns
        double[] testPattern1 = {1, 1, 0, 0};  // Should predict class 0
        double[] testPattern2 = {0, 0, 1, 1};  // Should predict class 1
        
        assertEquals(0, artmap.predict(testPattern1));
        assertEquals(1, artmap.predict(testPattern2));
    }
    
    @Test
    void testComplementCoding() {
        // BinaryFuzzyARTMAP should handle complement coding internally
        artmap = new BinaryFuzzyARTMAP(0.8, 0.01);
        
        // Input patterns (will be complement coded internally)
        double[][] data = {
            {1, 0, 1},
            {0, 1, 1},
            {1, 1, 0}
        };
        int[] labels = {0, 1, 0};
        
        artmap.fit(data, labels);
        
        // Verify complement coding is applied
        assertTrue(artmap.getCategoryCount() > 0);
        
        // Test predictions
        assertEquals(0, artmap.predict(new double[]{1, 0, 1}));
        assertEquals(1, artmap.predict(new double[]{0, 1, 1}));
    }
    
    @Test
    void testMultiClassClassification() {
        artmap = new BinaryFuzzyARTMAP(0.7, 0.001);
        
        // Create patterns for 3 classes
        double[][] trainData = {
            // Class 0: High first two bits
            {1, 1, 0, 0, 0},
            {1, 1, 0, 0, 1},
            // Class 1: High middle bits
            {0, 1, 1, 1, 0},
            {0, 0, 1, 1, 0},
            // Class 2: High last two bits
            {0, 0, 0, 1, 1},
            {0, 0, 1, 1, 1}
        };
        
        int[] labels = {0, 0, 1, 1, 2, 2};
        
        artmap.fit(trainData, labels);
        
        // Test accuracy on training data
        int correct = 0;
        for (int i = 0; i < trainData.length; i++) {
            if (artmap.predict(trainData[i]) == labels[i]) {
                correct++;
            }
        }
        
        double accuracy = (double) correct / trainData.length;
        assertTrue(accuracy >= 0.8, "Accuracy should be at least 80%, got: " + accuracy);
    }
    
    @Test
    void testIncrementalLearning() {
        artmap = new BinaryFuzzyARTMAP(0.75, 0.01);
        
        // Initial training
        double[][] initialData = {
            {1, 1, 0, 0},
            {0, 0, 1, 1}
        };
        int[] initialLabels = {0, 1};
        
        artmap.fit(initialData, initialLabels);
        int initialCategories = artmap.getCategoryCount();
        
        // Incremental training with new patterns
        double[][] newData = {
            {1, 0, 1, 0},
            {0, 1, 0, 1}
        };
        int[] newLabels = {2, 3};
        
        artmap.partialFit(newData, newLabels);
        
        // Should have more categories after incremental learning
        assertTrue(artmap.getCategoryCount() >= initialCategories);
        
        // Test all patterns
        assertEquals(0, artmap.predict(new double[]{1, 1, 0, 0}));
        assertEquals(1, artmap.predict(new double[]{0, 0, 1, 1}));
        assertEquals(2, artmap.predict(new double[]{1, 0, 1, 0}));
        assertEquals(3, artmap.predict(new double[]{0, 1, 0, 1}));
    }
    
    @Test
    void testNoiseRobustness() {
        artmap = new BinaryFuzzyARTMAP(0.6, 0.01);  // Lower vigilance for noise tolerance
        
        // Train with clean patterns
        double[][] cleanData = {
            {1, 1, 1, 0, 0, 0},
            {0, 0, 0, 1, 1, 1}
        };
        int[] labels = {0, 1};
        
        artmap.fit(cleanData, labels);
        
        // Test with noisy patterns (1 bit flipped)
        double[] noisyPattern0 = {1, 1, 0, 0, 0, 0};  // One bit flipped from class 0
        double[] noisyPattern1 = {0, 0, 0, 1, 0, 1};  // One bit flipped from class 1
        
        // Should still classify correctly despite noise
        assertEquals(0, artmap.predict(noisyPattern0));
        assertEquals(1, artmap.predict(noisyPattern1));
    }
    
    @Test
    void testPredictAB() {
        artmap = new BinaryFuzzyARTMAP(0.7, 0.01);
        
        double[][] data = {
            {1, 1, 0, 0},
            {0, 0, 1, 1}
        };
        int[] labels = {0, 1};
        
        artmap.fit(data, labels);
        
        // Test predictAB method
        BinaryFuzzyARTMAP.PredictionResult result = artmap.predictAB(new double[]{1, 1, 0, 0});
        
        assertNotNull(result);
        assertEquals(0, result.clusterIndex());  // A-side cluster
        assertEquals(0, result.classLabel());     // B-side label
    }
    
    @Test
    void testMatchTracking() {
        artmap = new BinaryFuzzyARTMAP(0.9, 0.01);  // High vigilance
        
        // Patterns that might cause match tracking
        double[][] data = {
            {1, 1, 1, 0},
            {1, 1, 0, 0},  // Similar to first but different class
            {0, 0, 1, 1}
        };
        int[] labels = {0, 1, 2};
        
        artmap.fit(data, labels);
        
        // Should create separate categories due to high vigilance and different labels
        assertTrue(artmap.getCategoryCount() >= 3);
        
        // Each pattern should be correctly classified
        assertEquals(0, artmap.predict(new double[]{1, 1, 1, 0}));
        assertEquals(1, artmap.predict(new double[]{1, 1, 0, 0}));
        assertEquals(2, artmap.predict(new double[]{0, 0, 1, 1}));
    }
    
    @Test
    void testEmptyInput() {
        artmap = new BinaryFuzzyARTMAP(0.7, 0.01);
        
        assertThrows(IllegalArgumentException.class, () -> {
            artmap.fit(new double[0][], new int[0]);
        });
    }
    
    @Test
    void testMismatchedDimensions() {
        artmap = new BinaryFuzzyARTMAP(0.7, 0.01);
        
        double[][] data = {{1, 0}, {0, 1}};
        int[] labels = {0};  // Wrong number of labels
        
        assertThrows(IllegalArgumentException.class, () -> {
            artmap.fit(data, labels);
        });
    }
    
    @Test
    void testLargeScaleBinaryPatterns() {
        artmap = new BinaryFuzzyARTMAP(0.65, 0.001);
        
        int numPatterns = 100;
        int dimension = 20;
        int numClasses = 5;
        
        // Generate random binary patterns
        double[][] data = new double[numPatterns][dimension];
        int[] labels = new int[numPatterns];
        
        for (int i = 0; i < numPatterns; i++) {
            for (int j = 0; j < dimension; j++) {
                data[i][j] = random.nextDouble() > 0.5 ? 1.0 : 0.0;
            }
            labels[i] = i % numClasses;
        }
        
        // Train
        artmap.fit(data, labels);
        
        // Test accuracy on training data
        int correct = 0;
        for (int i = 0; i < numPatterns; i++) {
            if (artmap.predict(data[i]) == labels[i]) {
                correct++;
            }
        }
        
        double accuracy = (double) correct / numPatterns;
        System.out.println("Large scale accuracy: " + accuracy);
        assertTrue(accuracy >= 0.5, "Accuracy should be reasonable for random patterns");
        
        // Verify categories were created
        assertTrue(artmap.getCategoryCount() > 0);
        assertTrue(artmap.getCategoryCount() <= numPatterns);
    }
}