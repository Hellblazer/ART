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
package com.hellblazer.art.core.biclustering;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.BaseART;
import com.hellblazer.art.core.algorithms.FuzzyART;
import com.hellblazer.art.core.parameters.FuzzyParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for BARTMAP biclustering implementation.
 * 
 * Tests the BARTMAP algorithm which performs simultaneous clustering of both
 * rows (samples) and columns (features) to identify biclusters - subsets of
 * rows that exhibit similar patterns across subsets of columns.
 * 
 * @author Hal Hildebrand
 */
public class BARTMAPTest {
    
    private BARTMAP bartmap;
    private Random random;
    
    @BeforeEach
    void setUp() {
        bartmap = null;
        random = new Random(42);
    }
    
    @Test
    void testInitialization() {
        // Create BARTMAP with FuzzyART modules
        var moduleA = new FuzzyART();
        var moduleB = new FuzzyART();
        bartmap = new BARTMAP(moduleA, moduleB, 0.01);
        
        assertEquals(0.01, bartmap.getEta());
        assertNotNull(bartmap.getModuleA());
        assertNotNull(bartmap.getModuleB());
    }
    
    @Test
    void testBasicBiclustering() {
        // Create synthetic data with clear bicluster structure
        double[][] data = createBiclusterData();
        
        var moduleA = new FuzzyART();
        var moduleB = new FuzzyART();
        bartmap = new BARTMAP(moduleA, moduleB, 0.5);
        
        // Fit the model
        bartmap.fit(data);
        
        // Check that biclusters were found
        assertTrue(bartmap.getRowClusterCount() > 0);
        assertTrue(bartmap.getColumnClusterCount() > 0);
        
        // Get the bicluster assignments
        var rowLabels = bartmap.getRowLabels();
        var columnLabels = bartmap.getColumnLabels();
        
        assertNotNull(rowLabels);
        assertNotNull(columnLabels);
        assertEquals(data.length, rowLabels.length);
        assertEquals(data[0].length, columnLabels.length);
    }
    
    @Test
    void testCorrelationBasedClustering() {
        // Create data with correlated features
        int nSamples = 50;
        int nFeatures = 20;
        double[][] data = new double[nSamples][nFeatures];
        
        // Create two groups of correlated features
        for (int i = 0; i < nSamples; i++) {
            double signal1 = random.nextGaussian();
            double signal2 = random.nextGaussian();
            
            // Group 1: features 0-9 correlated with signal1
            for (int j = 0; j < 10; j++) {
                data[i][j] = signal1 + random.nextGaussian() * 0.1;
            }
            
            // Group 2: features 10-19 correlated with signal2
            for (int j = 10; j < 20; j++) {
                data[i][j] = signal2 + random.nextGaussian() * 0.1;
            }
        }
        
        var moduleA = new FuzzyART();
        var moduleB = new FuzzyART();
        bartmap = new BARTMAP(moduleA, moduleB, 0.7);  // High correlation threshold
        
        bartmap.fit(data);
        
        // Should identify the two feature groups
        var columnLabels = bartmap.getColumnLabels();
        
        // Check that features within groups have same labels
        int label0 = columnLabels[0];
        int label10 = columnLabels[10];
        
        // Features 0-9 should have same label
        for (int i = 1; i < 10; i++) {
            assertEquals(label0, columnLabels[i], 
                "Correlated features 0-9 should be in same cluster");
        }
        
        // Features 10-19 should have same label
        for (int i = 11; i < 20; i++) {
            assertEquals(label10, columnLabels[i],
                "Correlated features 10-19 should be in same cluster");
        }
        
        // The two groups should have different labels
        assertNotEquals(label0, label10,
            "Uncorrelated feature groups should be in different clusters");
    }
    
    @Test
    void testGetBiclusters() {
        double[][] data = createBiclusterData();
        
        var moduleA = new FuzzyART();
        var moduleB = new FuzzyART();
        bartmap = new BARTMAP(moduleA, moduleB, 0.5);
        
        bartmap.fit(data);
        
        // Get the bicluster structure
        var biclusters = bartmap.getBiclusters();
        
        assertNotNull(biclusters);
        assertTrue(biclusters.length > 0);
        
        // Each bicluster should have row and column indicators
        for (var bicluster : biclusters) {
            assertNotNull(bicluster.rows());
            assertNotNull(bicluster.columns());
            assertEquals(data.length, bicluster.rows().length);
            assertEquals(data[0].length, bicluster.columns().length);
        }
    }
    
    @Test
    void testMatchCriterion() {
        // Test the Pearson correlation match criterion
        double[][] data = {
            {1.0, 2.0, 3.0, 4.0},
            {1.1, 2.1, 3.1, 4.1},  // Highly correlated with first row
            {4.0, 3.0, 2.0, 1.0},  // Anti-correlated
            {1.2, 2.2, 3.2, 4.2}   // Highly correlated with first row
        };
        
        var moduleA = new FuzzyART();
        var moduleB = new FuzzyART();
        bartmap = new BARTMAP(moduleA, moduleB, 0.9);  // Very high correlation threshold
        
        bartmap.fit(data);
        
        var rowLabels = bartmap.getRowLabels();
        
        // Rows 0, 1, and 3 should be in the same cluster (highly correlated)
        assertEquals(rowLabels[0], rowLabels[1]);
        assertEquals(rowLabels[0], rowLabels[3]);
        
        // Row 2 should be in a different cluster (anti-correlated)
        assertNotEquals(rowLabels[0], rowLabels[2]);
    }
    
    @Test
    void testWithDifferentModules() {
        // Test that BARTMAP works with different ART modules for rows and columns
        double[][] data = createBiclusterData();
        
        // Use different parameters for row and column clustering
        var moduleA = new FuzzyART();  // For rows
        var moduleB = new FuzzyART();  // For columns
        
        bartmap = new BARTMAP(moduleA, moduleB, 0.5);
        
        bartmap.fit(data);
        
        // Should produce valid biclusters
        assertTrue(bartmap.getRowClusterCount() > 0);
        assertTrue(bartmap.getColumnClusterCount() > 0);
    }
    
    @Test
    void testEmptyData() {
        var moduleA = new FuzzyART();
        var moduleB = new FuzzyART();
        bartmap = new BARTMAP(moduleA, moduleB, 0.5);
        
        assertThrows(IllegalArgumentException.class, () -> {
            bartmap.fit(new double[0][0]);
        });
    }
    
    @Test
    void testInvalidEta() {
        var moduleA = new FuzzyART();
        var moduleB = new FuzzyART();
        
        // Eta should be between -1 and 1 (Pearson correlation range)
        assertThrows(IllegalArgumentException.class, () -> {
            new BARTMAP(moduleA, moduleB, 1.5);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            new BARTMAP(moduleA, moduleB, -1.5);
        });
    }
    
    @Test
    void testNullModules() {
        var moduleA = new FuzzyART();
        
        assertThrows(NullPointerException.class, () -> {
            new BARTMAP(null, moduleA, 0.5);
        });
        
        assertThrows(NullPointerException.class, () -> {
            new BARTMAP(moduleA, null, 0.5);
        });
    }
    
    /**
     * Create synthetic data with clear bicluster structure.
     * Creates a matrix with 3 biclusters:
     * - Rows 0-19, Cols 0-9: High values
     * - Rows 20-39, Cols 10-19: High values  
     * - Rows 40-59, Cols 20-29: High values
     * Rest of the matrix has low/noise values
     */
    private double[][] createBiclusterData() {
        int nRows = 60;
        int nCols = 30;
        double[][] data = new double[nRows][nCols];
        
        // Initialize with small random values
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                data[i][j] = random.nextDouble() * 0.1;
            }
        }
        
        // Create bicluster 1: rows 0-19, cols 0-9
        for (int i = 0; i < 20; i++) {
            for (int j = 0; j < 10; j++) {
                data[i][j] = 0.8 + random.nextDouble() * 0.2;
            }
        }
        
        // Create bicluster 2: rows 20-39, cols 10-19
        for (int i = 20; i < 40; i++) {
            for (int j = 10; j < 20; j++) {
                data[i][j] = 0.8 + random.nextDouble() * 0.2;
            }
        }
        
        // Create bicluster 3: rows 40-59, cols 20-29
        for (int i = 40; i < 60; i++) {
            for (int j = 20; j < 30; j++) {
                data[i][j] = 0.8 + random.nextDouble() * 0.2;
            }
        }
        
        return data;
    }
}