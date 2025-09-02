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
import com.hellblazer.art.core.algorithms.GaussianART;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for GaussianARTMAP implementation.
 * 
 * Tests Gaussian-based supervised learning using the GaussianARTMAP algorithm,
 * which combines GaussianART with SimpleARTMAP for probabilistic classification.
 * 
 * @author Hal Hildebrand
 */
public class GaussianARTMAPTest {
    
    private GaussianARTMAP artmap;
    private Random random;
    
    @BeforeEach
    void setUp() {
        artmap = null;
        random = new Random(42);
    }
    
    @Test
    void testBasicGaussianClassification() {
        // Create GaussianARTMAP with initial sigma values
        double[] sigmaInit = {0.1, 0.1};
        artmap = new GaussianARTMAP(0.7, sigmaInit, 1e-10);
        
        // Create Gaussian-distributed patterns for two classes
        double[][] trainData = {
            // Class 0 - centered around (0.2, 0.2)
            {0.20, 0.20},
            {0.22, 0.18},
            {0.19, 0.21},
            {0.21, 0.19},
            // Class 1 - centered around (0.8, 0.8)
            {0.80, 0.80},
            {0.82, 0.78},
            {0.79, 0.81},
            {0.81, 0.79}
        };
        
        int[] trainLabels = {0, 0, 0, 0, 1, 1, 1, 1};
        
        // Train the model
        artmap.fit(trainData, trainLabels);
        
        // Test predictions on training data
        for (int i = 0; i < trainData.length; i++) {
            int predicted = artmap.predict(trainData[i]);
            assertEquals(trainLabels[i], predicted,
                "Prediction failed for training pattern " + i);
        }
        
        // Test on new patterns near the Gaussian centers
        double[] testPattern1 = {0.23, 0.17};  // Near class 0
        double[] testPattern2 = {0.77, 0.82};  // Near class 1
        
        assertEquals(0, artmap.predict(testPattern1));
        assertEquals(1, artmap.predict(testPattern2));
    }
    
    @Test
    void testMultiDimensionalGaussian() {
        // Test with higher dimensional data
        double[] sigmaInit = {0.1, 0.1, 0.1, 0.1};
        artmap = new GaussianARTMAP(0.65, sigmaInit, 1e-10);
        
        double[][] trainData = {
            // Class 0
            {0.1, 0.1, 0.1, 0.1},
            {0.12, 0.08, 0.11, 0.09},
            // Class 1
            {0.5, 0.5, 0.5, 0.5},
            {0.48, 0.52, 0.49, 0.51},
            // Class 2
            {0.9, 0.9, 0.9, 0.9},
            {0.88, 0.92, 0.91, 0.89}
        };
        
        int[] labels = {0, 0, 1, 1, 2, 2};
        
        artmap.fit(trainData, labels);
        
        // Verify categories were created
        assertTrue(artmap.getCategoryCount() >= 3);
        
        // Test predictions
        assertEquals(0, artmap.predict(new double[]{0.11, 0.09, 0.10, 0.11}));
        assertEquals(1, artmap.predict(new double[]{0.49, 0.51, 0.50, 0.50}));
        assertEquals(2, artmap.predict(new double[]{0.90, 0.91, 0.89, 0.90}));
    }
    
    @Test
    void testVarianceLearning() {
        // Test that variance is learned from data
        double[] sigmaInit = {0.2, 0.2};  // Start with larger variance
        artmap = new GaussianARTMAP(0.6, sigmaInit, 1e-10);
        
        // Create tightly clustered patterns
        double[][] tightCluster = {
            {0.50, 0.50},
            {0.51, 0.49},
            {0.49, 0.51},
            {0.50, 0.50},
            {0.50, 0.51}
        };
        
        int[] labels = {0, 0, 0, 0, 0};
        
        artmap.fit(tightCluster, labels);
        
        // Should recognize patterns very close to the mean
        assertEquals(0, artmap.predict(new double[]{0.505, 0.495}));
        assertEquals(0, artmap.predict(new double[]{0.495, 0.505}));
    }
    
    @Test
    void testIncrementalLearning() {
        double[] sigmaInit = {0.15, 0.15};
        artmap = new GaussianARTMAP(0.7, sigmaInit, 1e-10);
        
        // Initial training
        double[][] initialData = {
            {0.3, 0.3},
            {0.32, 0.28}
        };
        int[] initialLabels = {0, 0};
        
        artmap.fit(initialData, initialLabels);
        int initialCategories = artmap.getCategoryCount();
        
        // Incremental training with new class
        double[][] newData = {
            {0.7, 0.7},
            {0.68, 0.72}
        };
        int[] newLabels = {1, 1};
        
        artmap.partialFit(newData, newLabels);
        
        // Should have more categories after incremental learning
        assertTrue(artmap.getCategoryCount() >= initialCategories);
        
        // Test all patterns
        assertEquals(0, artmap.predict(new double[]{0.31, 0.29}));
        assertEquals(1, artmap.predict(new double[]{0.69, 0.71}));
    }
    
    @Test
    void testPredictWithLikelihood() {
        double[] sigmaInit = {0.1, 0.1};
        artmap = new GaussianARTMAP(0.65, sigmaInit, 1e-10);
        
        double[][] data = {
            {0.25, 0.25},
            {0.75, 0.75}
        };
        int[] labels = {0, 1};
        
        artmap.fit(data, labels);
        
        // Test predictWithLikelihood method
        var result = artmap.predictWithLikelihood(new double[]{0.26, 0.24});
        
        assertNotNull(result);
        assertEquals(0, result.classLabel());  // Should predict class 0
        assertTrue(result.likelihood() > 0);   // Should have positive likelihood
    }
    
    @Test
    void testOverlappingGaussians() {
        // Test handling of overlapping Gaussian distributions
        double[] sigmaInit = {0.2, 0.2};  // Larger variance for overlap
        artmap = new GaussianARTMAP(0.5, sigmaInit, 1e-10);  // Lower vigilance
        
        double[][] data = {
            // Class 0 - centered around (0.4, 0.4)
            {0.40, 0.40},
            {0.38, 0.42},
            {0.42, 0.38},
            // Class 1 - centered around (0.6, 0.6) - overlapping with class 0
            {0.60, 0.60},
            {0.58, 0.62},
            {0.62, 0.58}
        };
        int[] labels = {0, 0, 0, 1, 1, 1};
        
        artmap.fit(data, labels);
        
        // Test boundary region - may go to either class due to overlap
        double[] boundaryPattern = {0.50, 0.50};
        int prediction = artmap.predict(boundaryPattern);
        assertTrue(prediction == 0 || prediction == 1,
            "Boundary pattern should predict one of the overlapping classes");
        
        // Patterns clearly in each class should be correctly classified
        assertEquals(0, artmap.predict(new double[]{0.35, 0.35}));
        assertEquals(1, artmap.predict(new double[]{0.65, 0.65}));
    }
    
    @Test
    void testMultiClassGaussian() {
        double[] sigmaInit = {0.1, 0.1, 0.1};
        artmap = new GaussianARTMAP(0.6, sigmaInit, 1e-10);
        
        // Create patterns for 4 classes
        double[][] trainData = new double[40][3];
        int[] labels = new int[40];
        
        // Generate Gaussian-distributed data for each class
        for (int c = 0; c < 4; c++) {
            double centerX = 0.2 + c * 0.2;
            double centerY = 0.2 + c * 0.2;
            double centerZ = 0.2 + c * 0.2;
            
            for (int i = 0; i < 10; i++) {
                int idx = c * 10 + i;
                trainData[idx][0] = centerX + (random.nextGaussian() * 0.05);
                trainData[idx][1] = centerY + (random.nextGaussian() * 0.05);
                trainData[idx][2] = centerZ + (random.nextGaussian() * 0.05);
                labels[idx] = c;
            }
        }
        
        artmap.fit(trainData, labels);
        
        // Test accuracy on training data
        int correct = 0;
        for (int i = 0; i < trainData.length; i++) {
            if (artmap.predict(trainData[i]) == labels[i]) {
                correct++;
            }
        }
        
        double accuracy = (double) correct / trainData.length;
        assertTrue(accuracy >= 0.7, "Accuracy should be at least 70%, got: " + accuracy);
    }
    
    @Test
    void testAnisotropicVariance() {
        // Test with different variance in each dimension
        double[] sigmaInit = {0.05, 0.2};  // Small variance in X, large in Y
        artmap = new GaussianARTMAP(0.65, sigmaInit, 1e-10);
        
        double[][] data = {
            // Class 0 - spread more in Y direction
            {0.3, 0.2},
            {0.3, 0.3},
            {0.3, 0.4},
            // Class 1 - spread more in Y direction
            {0.7, 0.2},
            {0.7, 0.3},
            {0.7, 0.4}
        };
        int[] labels = {0, 0, 0, 1, 1, 1};
        
        artmap.fit(data, labels);
        
        // Test that X dimension is more discriminative due to smaller initial variance
        assertEquals(0, artmap.predict(new double[]{0.31, 0.35}));  // Close in X to class 0
        assertEquals(1, artmap.predict(new double[]{0.69, 0.35}));  // Close in X to class 1
    }
    
    @Test
    void testEmptyInput() {
        double[] sigmaInit = {0.1, 0.1};
        artmap = new GaussianARTMAP(0.7, sigmaInit, 1e-10);
        
        assertThrows(IllegalArgumentException.class, () -> {
            artmap.fit(new double[0][], new int[0]);
        });
    }
    
    @Test
    void testMismatchedDimensions() {
        double[] sigmaInit = {0.1, 0.1};
        artmap = new GaussianARTMAP(0.7, sigmaInit, 1e-10);
        
        double[][] data = {{0.5, 0.5}, {0.6, 0.6}};
        int[] labels = {0};  // Wrong number of labels
        
        assertThrows(IllegalArgumentException.class, () -> {
            artmap.fit(data, labels);
        });
    }
    
    @Test
    void testInvalidSigma() {
        // Test with invalid sigma values
        assertThrows(IllegalArgumentException.class, () -> {
            double[] sigmaInit = {0.1, -0.1};  // Negative sigma
            new GaussianARTMAP(0.7, sigmaInit, 1e-10);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            double[] sigmaInit = {0.0, 0.1};  // Zero sigma
            new GaussianARTMAP(0.7, sigmaInit, 1e-10);
        });
    }
}