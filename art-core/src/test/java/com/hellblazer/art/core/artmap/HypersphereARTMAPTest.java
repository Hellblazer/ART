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
import com.hellblazer.art.core.algorithms.HypersphereART;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for HypersphereARTMAP implementation.
 * 
 * Tests hypersphere-based supervised learning using the HypersphereARTMAP algorithm,
 * which combines HypersphereART with SimpleARTMAP for spherical cluster classification.
 * 
 * @author Hal Hildebrand
 */
public class HypersphereARTMAPTest {
    
    private HypersphereARTMAP artmap;
    private Random random;
    
    @BeforeEach
    void setUp() {
        artmap = null;
        random = new Random(42);
    }
    
    @Test
    void testBasicHypersphereClassification() {
        // Create HypersphereARTMAP with parameters
        artmap = new HypersphereARTMAP(0.7, 1e-10, 1.0, 0.8);
        
        // Create well-separated clusters for two classes
        double[][] trainData = {
            // Class 0 - cluster at (0.2, 0.2)
            {0.20, 0.20},
            {0.22, 0.18},
            {0.19, 0.21},
            {0.21, 0.19},
            {0.18, 0.22},
            // Class 1 - cluster at (0.8, 0.8)
            {0.80, 0.80},
            {0.82, 0.78},
            {0.79, 0.81},
            {0.81, 0.79},
            {0.78, 0.82}
        };
        
        int[] trainLabels = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
        
        // Train the model
        artmap.fit(trainData, trainLabels);
        
        // Test predictions on training data
        for (int i = 0; i < trainData.length; i++) {
            int predicted = artmap.predict(trainData[i]);
            assertEquals(trainLabels[i], predicted,
                "Prediction failed for training pattern " + i);
        }
        
        // Test on new patterns within clusters
        double[] testPattern1 = {0.21, 0.20};  // Within class 0 cluster
        double[] testPattern2 = {0.80, 0.81};  // Within class 1 cluster
        
        assertEquals(0, artmap.predict(testPattern1));
        assertEquals(1, artmap.predict(testPattern2));
    }
    
    @Test
    void testRadiusBoundedClusters() {
        // Test that r_hat parameter bounds cluster radius
        artmap = new HypersphereARTMAP(0.6, 1e-10, 1.0, 0.3);  // Small r_hat
        
        double[][] data = {
            // Spread out patterns that would normally form one large cluster
            {0.1, 0.1},
            {0.3, 0.3},
            {0.5, 0.5},
            {0.7, 0.7},
            {0.9, 0.9}
        };
        int[] labels = {0, 0, 0, 0, 0};  // All same class
        
        artmap.fit(data, labels);
        
        // Due to radius bound, should create multiple clusters
        assertTrue(artmap.getCategoryCount() > 1,
            "Small radius bound should force multiple clusters");
        
        // All should still predict the same class
        for (double[] pattern : data) {
            assertEquals(0, artmap.predict(pattern));
        }
    }
    
    @Test
    void testMultiDimensionalHypersphere() {
        // Test with higher dimensional data
        artmap = new HypersphereARTMAP(0.65, 1e-10, 1.0, 0.5);
        
        double[][] trainData = {
            // Class 0 - 4D cluster
            {0.1, 0.1, 0.1, 0.1},
            {0.12, 0.08, 0.11, 0.09},
            {0.09, 0.11, 0.10, 0.12},
            // Class 1 - 4D cluster
            {0.5, 0.5, 0.5, 0.5},
            {0.48, 0.52, 0.49, 0.51},
            {0.51, 0.49, 0.52, 0.48},
            // Class 2 - 4D cluster
            {0.9, 0.9, 0.9, 0.9},
            {0.88, 0.92, 0.91, 0.89},
            {0.91, 0.89, 0.88, 0.92}
        };
        
        int[] labels = {0, 0, 0, 1, 1, 1, 2, 2, 2};
        
        artmap.fit(trainData, labels);
        
        // Test predictions
        assertEquals(0, artmap.predict(new double[]{0.11, 0.09, 0.10, 0.11}));
        assertEquals(1, artmap.predict(new double[]{0.49, 0.51, 0.50, 0.50}));
        assertEquals(2, artmap.predict(new double[]{0.90, 0.91, 0.89, 0.90}));
    }
    
    @Test
    void testIncrementalLearning() {
        artmap = new HypersphereARTMAP(0.7, 1e-10, 1.0, 0.6);
        
        // Initial training with one class
        double[][] initialData = {
            {0.3, 0.3},
            {0.32, 0.28},
            {0.28, 0.32}
        };
        int[] initialLabels = {0, 0, 0};
        
        artmap.fit(initialData, initialLabels);
        int initialCategories = artmap.getCategoryCount();
        
        // Incremental training with new class
        double[][] newData = {
            {0.7, 0.7},
            {0.68, 0.72},
            {0.72, 0.68}
        };
        int[] newLabels = {1, 1, 1};
        
        artmap.partialFit(newData, newLabels);
        
        // Should have more categories after incremental learning
        assertTrue(artmap.getCategoryCount() >= initialCategories);
        
        // Test predictions on both old and new data
        assertEquals(0, artmap.predict(new double[]{0.30, 0.31}));
        assertEquals(1, artmap.predict(new double[]{0.70, 0.69}));
    }
    
    @Test
    void testOverlappingClusters() {
        // Test behavior with overlapping spherical clusters
        artmap = new HypersphereARTMAP(0.5, 1e-10, 1.0, 0.5);  // Large r_hat for overlap
        
        double[][] data = {
            // Class 0 - centered around (0.4, 0.4)
            {0.40, 0.40},
            {0.38, 0.42},
            {0.42, 0.38},
            {0.39, 0.41},
            // Class 1 - centered around (0.6, 0.6) - overlapping
            {0.60, 0.60},
            {0.58, 0.62},
            {0.62, 0.58},
            {0.61, 0.59}
        };
        int[] labels = {0, 0, 0, 0, 1, 1, 1, 1};
        
        artmap.fit(data, labels);
        
        // Test boundary region
        double[] boundaryPattern = {0.50, 0.50};
        int prediction = artmap.predict(boundaryPattern);
        assertTrue(prediction == 0 || prediction == 1,
            "Boundary pattern should predict one of the overlapping classes");
        
        // Patterns clearly in each class
        assertEquals(0, artmap.predict(new double[]{0.35, 0.35}));
        assertEquals(1, artmap.predict(new double[]{0.65, 0.65}));
    }
    
    @Test
    void testMultiClassClassification() {
        artmap = new HypersphereARTMAP(0.6, 1e-10, 1.0, 0.4);
        
        // Create patterns for 4 classes
        double[][] trainData = new double[40][3];
        int[] labels = new int[40];
        
        // Generate spherical clusters for each class
        for (int c = 0; c < 4; c++) {
            double centerX = 0.2 + c * 0.2;
            double centerY = 0.2 + c * 0.2;
            double centerZ = 0.2 + c * 0.2;
            
            for (int i = 0; i < 10; i++) {
                int idx = c * 10 + i;
                // Create points within sphere around center
                double theta = random.nextDouble() * 2 * Math.PI;
                double phi = random.nextDouble() * Math.PI;
                double r = random.nextDouble() * 0.1;  // Small radius
                
                trainData[idx][0] = centerX + r * Math.sin(phi) * Math.cos(theta);
                trainData[idx][1] = centerY + r * Math.sin(phi) * Math.sin(theta);
                trainData[idx][2] = centerZ + r * Math.cos(phi);
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
        assertTrue(accuracy >= 0.8, "Accuracy should be at least 80%, got: " + accuracy);
    }
    
    @Test
    void testAdaptiveLearning() {
        // Test that beta parameter affects learning rate
        artmap = new HypersphereARTMAP(0.65, 1e-10, 0.5, 0.5);  // Beta = 0.5 for partial learning
        
        double[][] data = {
            {0.5, 0.5},
            {0.52, 0.48},
            {0.48, 0.52},
            {0.51, 0.49},
            {0.49, 0.51}
        };
        int[] labels = {0, 0, 0, 0, 0};
        
        artmap.fit(data, labels);
        
        // All patterns should still be classified correctly
        for (int i = 0; i < data.length; i++) {
            assertEquals(0, artmap.predict(data[i]));
        }
        
        // Test with pattern near cluster center
        assertEquals(0, artmap.predict(new double[]{0.50, 0.50}));
    }
    
    @Test
    void testMatchTracking() {
        // Test match tracking for label conflicts
        artmap = new HypersphereARTMAP(0.7, 1e-10, 1.0, 0.5);
        
        double[][] data = {
            // Initial pattern for class 0
            {0.5, 0.5},
            // Conflicting pattern (same location, different class)
            {0.5, 0.5},
            // Additional patterns
            {0.3, 0.3},
            {0.7, 0.7}
        };
        int[] labels = {0, 1, 0, 1};
        
        artmap.fit(data, labels);
        
        // Should have created separate categories due to match tracking
        assertTrue(artmap.getCategoryCount() >= 2,
            "Match tracking should create separate categories for conflicting labels");
    }
    
    @Test
    void testEmptyInput() {
        artmap = new HypersphereARTMAP(0.7, 1e-10, 1.0, 0.5);
        
        assertThrows(IllegalArgumentException.class, () -> {
            artmap.fit(new double[0][], new int[0]);
        });
    }
    
    @Test
    void testMismatchedDimensions() {
        artmap = new HypersphereARTMAP(0.7, 1e-10, 1.0, 0.5);
        
        double[][] data = {{0.5, 0.5}, {0.6, 0.6}};
        int[] labels = {0};  // Wrong number of labels
        
        assertThrows(IllegalArgumentException.class, () -> {
            artmap.fit(data, labels);
        });
    }
    
    @Test
    void testInvalidRadius() {
        // Test with invalid r_hat values
        assertThrows(IllegalArgumentException.class, () -> {
            new HypersphereARTMAP(0.7, 1e-10, 1.0, 0.0);  // Zero radius
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            new HypersphereARTMAP(0.7, 1e-10, 1.0, -0.5);  // Negative radius
        });
    }
}