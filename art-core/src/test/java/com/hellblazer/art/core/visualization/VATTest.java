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
package com.hellblazer.art.core.visualization;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.Random;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for VAT (Visual Assessment of Clustering Tendency) implementation.
 * 
 * VAT is a visualization algorithm that reorders a dissimilarity matrix to reveal
 * clustering structure through dark diagonal blocks. This Java implementation
 * should be significantly faster than the Python pyclustertend equivalent.
 * 
 * @author Hal Hildebrand
 */
public class VATTest {
    
    private double[][] perfectClusters;      // Data with obvious clusters
    private double[][] randomData;           // Random data with no clusters
    private double[][] ambiguousData;        // Data with subtle clustering
    private double[][] largeDataset;         // Performance testing data
    
    @BeforeEach
    void setUp() {
        var random = new Random(42);
        
        // Create perfect clusters: two tight groups
        perfectClusters = createPerfectClusters();
        
        // Create random data with no clustering tendency
        randomData = createRandomData(random, 50, 2);
        
        // Create ambiguous data with subtle clusters
        ambiguousData = createAmbiguousData(random);
        
        // Create large dataset for performance testing
        largeDataset = createRandomData(random, 500, 10);
    }
    
    /**
     * Test basic VAT functionality with perfect clusters.
     */
    @Test
    void testVATBasicFunctionality() {
        var result = VAT.compute(perfectClusters);
        
        // Basic validation
        assertNotNull(result, "VAT result should not be null");
        assertNotNull(result.getOrderedDissimilarityMatrix(), "ODM should not be null");
        assertEquals(perfectClusters.length, result.getSize(), "Size should match input");
        assertNotNull(result.getReorderingIndices(), "Reordering indices should be available");
        
        // Should detect strong clustering tendency
        assertTrue(result.hasStrongClusteringTendency(), 
            "Should detect strong clustering in perfect clusters");
        
        // Cluster clarity should be high
        assertTrue(result.getClusterClarity() > 0.7, 
            String.format("Cluster clarity should be high: %.3f", result.getClusterClarity()));
        
        // ODM should be symmetric
        var odm = result.getOrderedDissimilarityMatrix();
        for (int i = 0; i < odm.length; i++) {
            for (int j = 0; j < odm[i].length; j++) {
                assertEquals(odm[i][j], odm[j][i], 1e-10, 
                    String.format("ODM should be symmetric at [%d,%d]", i, j));
            }
        }
    }
    
    /**
     * Test VAT performance compared to expected Python performance.
     */
    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void testVATPerformance() {
        int[] sizes = {50, 100, 200, 300};
        
        for (int n : sizes) {
            var data = createRandomData(new Random(42), n, 3);
            
            long startTime = System.nanoTime();
            var result = VAT.compute(data);
            long duration = System.nanoTime() - startTime;
            
            double timeMs = duration / 1_000_000.0;
            double expectedPythonTime = estimatePythonVATTime(n);
            double speedup = expectedPythonTime / timeMs;
            
            System.out.printf("VAT n=%d: %.3f ms (est. Python: %.3f ms, speedup: %.1fx)%n", 
                n, timeMs, expectedPythonTime, speedup);
            
            // Verify reasonable performance bounds
            assertTrue(timeMs < getMaxExpectedTime(n), 
                String.format("Performance too slow for n=%d: %.3f ms", n, timeMs));
            
            // Should be significantly faster than estimated Python time
            assertTrue(speedup > 2.0, 
                String.format("Should be >2x faster than Python for n=%d, got %.1fx", n, speedup));
            
            assertNotNull(result);
            assertEquals(n, result.getSize());
        }
    }
    
    /**
     * Test VAT with perfect clusters should show clear structure.
     */
    @Test
    void testVATWithPerfectClusters() {
        var result = VAT.compute(perfectClusters);
        
        // Should detect clear cluster boundaries
        assertTrue(result.hasStrongClusteringTendency(), 
            "Perfect clusters should show strong clustering tendency");
        
        // Cluster clarity should be very high
        assertTrue(result.getClusterClarity() > 0.8, 
            String.format("Perfect clusters should have high clarity: %.3f", result.getClusterClarity()));
        
        // Should identify correct number of cluster blocks
        int estimatedClusters = result.estimateClusterCount();
        assertEquals(2, estimatedClusters, 
            String.format("Should estimate 2 clusters, got %d", estimatedClusters));
        
        // Dark diagonal blocks should be present
        assertTrue(result.hasDarkDiagonalBlocks(), 
            "Should detect dark diagonal blocks indicating clusters");
    }
    
    /**
     * Test VAT with random data should show no clustering.
     */
    @Test
    void testVATWithRandomData() {
        var result = VAT.compute(randomData);
        
        assertNotNull(result);
        assertFalse(result.hasStrongClusteringTendency(), 
            "Random data should not show strong clustering tendency");
        
        // Cluster clarity should be low
        assertTrue(result.getClusterClarity() < 0.5, 
            String.format("Random data should have low clarity: %.3f", result.getClusterClarity()));
        
        // Should estimate 1 cluster (no structure)
        int estimatedClusters = result.estimateClusterCount();
        assertTrue(estimatedClusters <= 2, 
            String.format("Random data should estimate ≤2 clusters, got %d", estimatedClusters));
    }
    
    /**
     * Test VAT parallel implementation performance.
     */
    @Test
    @Timeout(value = 5, unit = TimeUnit.SECONDS)
    void testVATParallelPerformance() {
        // Test parallel vs sequential implementation
        long sequentialTime = timeVATComputation(() -> VAT.compute(largeDataset));
        long parallelTime = timeVATComputation(() -> VAT.computeParallel(largeDataset));
        
        double speedup = (double) sequentialTime / parallelTime;
        
        System.out.printf("VAT Sequential: %.3f ms%n", sequentialTime / 1_000_000.0);
        System.out.printf("VAT Parallel: %.3f ms%n", parallelTime / 1_000_000.0);
        System.out.printf("Parallel speedup: %.2fx%n", speedup);
        
        // Parallel should be faster or at least not much slower
        assertTrue(speedup > 0.8, 
            String.format("Parallel implementation should not be much slower: %.2fx", speedup));
        
        // Both should produce equivalent results
        var sequentialResult = VAT.compute(largeDataset);
        var parallelResult = VAT.computeParallel(largeDataset);
        
        assertArrayEquals(sequentialResult.getReorderingIndices(), 
                         parallelResult.getReorderingIndices(),
                         "Parallel and sequential should produce same reordering");
    }
    
    /**
     * Test VAT result data integrity and consistency.
     */
    @Test
    void testVATResultIntegrity() {
        var result = VAT.compute(perfectClusters);
        
        // Test reordering indices
        int[] indices = result.getReorderingIndices();
        assertNotNull(indices);
        assertEquals(perfectClusters.length, indices.length);
        
        // Indices should be a valid permutation (0 to n-1)
        boolean[] used = new boolean[indices.length];
        for (int idx : indices) {
            assertTrue(idx >= 0 && idx < indices.length, 
                String.format("Invalid index: %d", idx));
            assertFalse(used[idx], String.format("Duplicate index: %d", idx));
            used[idx] = true;
        }
        
        // All indices should be used
        for (int i = 0; i < used.length; i++) {
            assertTrue(used[i], String.format("Index %d not used", i));
        }
        
        // ODM dimensions should match
        var odm = result.getOrderedDissimilarityMatrix();
        assertEquals(perfectClusters.length, odm.length);
        assertEquals(perfectClusters.length, odm[0].length);
        
        // ODM diagonal should be zero
        for (int i = 0; i < odm.length; i++) {
            assertEquals(0.0, odm[i][i], 1e-10, 
                String.format("ODM diagonal should be zero at [%d,%d]", i, i));
        }
    }
    
    /**
     * Test VAT with edge cases.
     */
    @Test
    void testVATEdgeCases() {
        // Test with minimum data (2 points)
        double[][] twoPoints = {{0.0, 0.0}, {1.0, 1.0}};
        var result = VAT.compute(twoPoints);
        assertNotNull(result);
        assertEquals(2, result.getSize());
        
        // Test with identical points
        double[][] identicalPoints = {{0.5, 0.5}, {0.5, 0.5}, {0.5, 0.5}};
        result = VAT.compute(identicalPoints);
        assertNotNull(result);
        assertEquals(3, result.getSize());
        
        // Should handle identical points gracefully
        assertFalse(result.hasStrongClusteringTendency(), 
            "Identical points should not show clustering tendency");
    }
    
    /**
     * Test VAT error handling.
     */
    @Test
    void testVATErrorHandling() {
        // Test null input
        assertThrows(IllegalArgumentException.class, () -> VAT.compute(null));
        
        // Test empty input
        assertThrows(IllegalArgumentException.class, () -> VAT.compute(new double[0][]));
        
        // Test single point
        assertThrows(IllegalArgumentException.class, 
            () -> VAT.compute(new double[][]{{1.0, 2.0}}));
        
        // Test inconsistent dimensions
        double[][] inconsistent = {{1.0, 2.0}, {3.0}}; // Different lengths
        assertThrows(IllegalArgumentException.class, () -> VAT.compute(inconsistent));
        
        // Test null rows
        double[][] withNull = {{1.0, 2.0}, null, {3.0, 4.0}};
        assertThrows(IllegalArgumentException.class, () -> VAT.compute(withNull));
    }
    
    // Helper methods
    
    private double[][] createPerfectClusters() {
        return new double[][] {
            // Cluster 1: around (0.1, 0.1)
            {0.1, 0.1}, {0.12, 0.08}, {0.08, 0.12}, {0.11, 0.09}, {0.09, 0.11},
            // Cluster 2: around (0.9, 0.9)  
            {0.9, 0.9}, {0.88, 0.92}, {0.92, 0.88}, {0.89, 0.91}, {0.91, 0.89}
        };
    }
    
    private double[][] createRandomData(Random random, int samples, int dimensions) {
        var data = new double[samples][dimensions];
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < dimensions; j++) {
                data[i][j] = random.nextDouble();
            }
        }
        return data;
    }
    
    private double[][] createAmbiguousData(Random random) {
        var data = new double[30][2];
        // Create 3 overlapping clusters
        for (int i = 0; i < 10; i++) {
            data[i] = new double[]{0.3 + random.nextGaussian() * 0.1, 0.3 + random.nextGaussian() * 0.1};
        }
        for (int i = 10; i < 20; i++) {
            data[i] = new double[]{0.5 + random.nextGaussian() * 0.1, 0.7 + random.nextGaussian() * 0.1};
        }
        for (int i = 20; i < 30; i++) {
            data[i] = new double[]{0.7 + random.nextGaussian() * 0.1, 0.3 + random.nextGaussian() * 0.1};
        }
        return data;
    }
    
    private double estimatePythonVATTime(int n) {
        // Conservative estimates based on Python O(n²) complexity
        // These are rough estimates - actual Python times vary significantly
        return Math.max(1.0, 0.001 * n * n); // Minimum 1ms, scales quadratically
    }
    
    private double getMaxExpectedTime(int n) {
        // Conservative maximum expected times for Java implementation
        return Math.max(10.0, 0.0005 * n * n); // Should be faster than Python
    }
    
    private long timeVATComputation(Runnable computation) {
        long startTime = System.nanoTime();
        computation.run();
        return System.nanoTime() - startTime;
    }
}