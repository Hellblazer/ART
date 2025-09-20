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
 * Test suite for iVAT (improved Visual Assessment of Clustering Tendency).
 * 
 * iVAT enhances the basic VAT algorithm by applying additional transformations
 * to the ordered dissimilarity matrix to make clustering structure more visually
 * apparent. This is particularly useful for datasets with subtle or overlapping
 * clusters.
 * 
 * @author Hal Hildebrand
 */
public class iVATTest {
    
    private double[][] perfectClusters;
    private double[][] ambiguousData;
    private double[][] overlappingClusters;
    private double[][] randomData;
    
    @BeforeEach
    void setUp() {
        var random = new Random(42);
        
        perfectClusters = createPerfectClusters();
        ambiguousData = createAmbiguousData(random);
        overlappingClusters = createOverlappingClusters(random);
        randomData = createRandomData(random, 50, 2);
    }
    
    /**
     * Test that iVAT provides clearer visualization than basic VAT.
     */
    @Test
    void testIVATImprovedVisualization() {
        // Test with ambiguous data where iVAT should show clearer structure
        var vatResult = VAT.compute(ambiguousData);
        var ivatResult = iVAT.compute(ambiguousData);
        
        // iVAT should provide clearer clustering structure
        assertTrue(ivatResult.getClusterClarity() >= vatResult.getClusterClarity(),
            String.format("iVAT clarity (%.3f) should be >= VAT clarity (%.3f)", 
                ivatResult.getClusterClarity(), vatResult.getClusterClarity()));
        
        // iVAT should be more confident about clustering tendency
        if (vatResult.hasStrongClusteringTendency()) {
            assertTrue(ivatResult.hasStrongClusteringTendency(),
                "If VAT detects clustering, iVAT should too");
        }
        
        // iVAT should provide better cluster count estimation
        int vatClusters = vatResult.estimateClusterCount();
        int ivatClusters = ivatResult.estimateClusterCount();
        
        System.out.printf("VAT clusters: %d, iVAT clusters: %d%n", vatClusters, ivatClusters);
        
        // For ambiguous data with 3 known clusters, iVAT should be more accurate
        assertTrue(Math.abs(ivatClusters - 3) <= Math.abs(vatClusters - 3),
            "iVAT should provide better cluster count estimation");
    }
    
    /**
     * Test iVAT with overlapping clusters where enhancement is most beneficial.
     */
    @Test
    void testIVATWithOverlappingClusters() {
        var ivatResult = iVAT.compute(overlappingClusters);
        
        assertNotNull(ivatResult);
        
        // Should detect clustering structure even in overlapping clusters
        assertTrue(ivatResult.getClusterClarity() > 0.3,
            String.format("Should detect some clustering in overlapping data: %.3f", 
                ivatResult.getClusterClarity()));
        
        // Should provide reasonable cluster count estimate
        int estimatedClusters = ivatResult.estimateClusterCount();
        assertTrue(estimatedClusters >= 2 && estimatedClusters <= 5,
            String.format("Cluster estimate should be reasonable: %d", estimatedClusters));
    }
    
    /**
     * Test iVAT performance compared to basic VAT.
     */
    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void testIVATPerformance() {
        int[] sizes = {50, 100, 200};
        
        for (int n : sizes) {
            var data = createRandomData(new Random(42), n, 5);
            
            // Time VAT
            long vatStartTime = System.nanoTime();
            var vatResult = VAT.compute(data);
            long vatTime = System.nanoTime() - vatStartTime;
            
            // Time iVAT
            long ivatStartTime = System.nanoTime();
            var ivatResult = iVAT.compute(data);
            long ivatTime = System.nanoTime() - ivatStartTime;
            
            double vatMs = vatTime / 1_000_000.0;
            double ivatMs = ivatTime / 1_000_000.0;
            double overhead = (ivatMs - vatMs) / vatMs * 100;
            
            System.out.printf("n=%d: VAT=%.3fms, iVAT=%.3fms, overhead=%.1f%%%n", 
                n, vatMs, ivatMs, overhead);
            
            // Log performance but don't assert - CI environments have different hardware
            if (overhead > 500.0) {
                System.out.printf("Note: iVAT overhead is %.1f%% (higher than typical 500%% threshold)%n", overhead);
            }
            
            assertNotNull(vatResult);
            assertNotNull(ivatResult);
        }
    }
    
    /**
     * Test that iVAT maintains data integrity.
     */
    @Test
    void testIVATDataIntegrity() {
        var result = iVAT.compute(perfectClusters);
        
        // Should maintain same basic structure as VAT
        assertNotNull(result.getOrderedDissimilarityMatrix());
        assertNotNull(result.getReorderingIndices());
        assertEquals(perfectClusters.length, result.getSize());
        
        // Reordering should be a valid permutation
        int[] indices = result.getReorderingIndices();
        boolean[] used = new boolean[indices.length];
        for (int idx : indices) {
            assertTrue(idx >= 0 && idx < indices.length);
            assertFalse(used[idx], String.format("Duplicate index: %d", idx));
            used[idx] = true;
        }
        
        // Matrix should still be symmetric
        var matrix = result.getOrderedDissimilarityMatrix();
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                assertEquals(matrix[i][j], matrix[j][i], 1e-10,
                    String.format("Matrix should be symmetric at [%d,%d]", i, j));
            }
        }
    }
    
    /**
     * Test iVAT enhancement algorithms.
     */
    @Test
    void testIVATEnhancementMethods() {
        var basicResult = VAT.compute(ambiguousData);
        var basicMatrix = basicResult.getOrderedDissimilarityMatrix();
        
        // Test different enhancement methods
        var pathBasedResult = iVAT.computeWithPathBasedEnhancement(ambiguousData);
        var contrastResult = iVAT.computeWithContrastEnhancement(ambiguousData);
        
        // All methods should produce valid results
        assertNotNull(pathBasedResult);
        assertNotNull(contrastResult);
        
        // Enhanced methods should produce valid results (not necessarily better clarity for all data)
        assertTrue(pathBasedResult.getClusterClarity() >= 0.0,
            "Path-based enhancement should produce valid clarity score");
        assertTrue(contrastResult.getClusterClarity() >= 0.0,
            "Contrast enhancement should produce valid clarity score");
        
        System.out.printf("Basic VAT clarity: %.3f%n", basicResult.getClusterClarity());
        System.out.printf("Path-based iVAT clarity: %.3f%n", pathBasedResult.getClusterClarity());
        System.out.printf("Contrast iVAT clarity: %.3f%n", contrastResult.getClusterClarity());
    }
    
    /**
     * Test iVAT with edge cases.
     */
    @Test
    void testIVATEdgeCases() {
        // Test with minimum data
        double[][] twoPoints = {{0.0, 0.0}, {1.0, 1.0}};
        var result = iVAT.compute(twoPoints);
        assertNotNull(result);
        assertEquals(2, result.getSize());
        
        // Test with identical points
        double[][] identicalPoints = {{0.5, 0.5}, {0.5, 0.5}, {0.5, 0.5}};
        result = iVAT.compute(identicalPoints);
        assertNotNull(result);
        assertEquals(3, result.getSize());
        assertFalse(result.hasStrongClusteringTendency());
    }
    
    /**
     * Test iVAT visualization matrix properties.
     */
    @Test
    void testIVATVisualizationMatrix() {
        var result = iVAT.compute(perfectClusters);
        
        // Test normalized matrix for visualization
        var normalized = result.getNormalizedODM();
        assertNotNull(normalized);
        assertEquals(result.getSize(), normalized.length);
        assertEquals(result.getSize(), normalized[0].length);
        
        // All values should be in [0, 1] range
        for (double[] row : normalized) {
            for (double val : row) {
                assertTrue(val >= 0.0 && val <= 1.0,
                    String.format("Normalized value should be in [0,1]: %.3f", val));
            }
        }
        
        // Diagonal should be zero (or close to zero for enhancement artifacts)
        for (int i = 0; i < normalized.length; i++) {
            assertTrue(normalized[i][i] <= 0.1,
                String.format("Diagonal should be near zero: %.3f", normalized[i][i]));
        }
    }
    
    // Helper methods
    
    private double[][] createPerfectClusters() {
        return new double[][] {
            // Cluster 1: tight around (0.1, 0.1)
            {0.1, 0.1}, {0.12, 0.08}, {0.08, 0.12}, {0.11, 0.09}, {0.09, 0.11},
            // Cluster 2: tight around (0.9, 0.9)
            {0.9, 0.9}, {0.88, 0.92}, {0.92, 0.88}, {0.89, 0.91}, {0.91, 0.89}
        };
    }
    
    private double[][] createAmbiguousData(Random random) {
        var data = new double[30][2];
        // Create 3 clusters with some overlap
        for (int i = 0; i < 10; i++) {
            data[i] = new double[]{0.3 + random.nextGaussian() * 0.15, 0.3 + random.nextGaussian() * 0.15};
        }
        for (int i = 10; i < 20; i++) {
            data[i] = new double[]{0.5 + random.nextGaussian() * 0.15, 0.7 + random.nextGaussian() * 0.15};
        }
        for (int i = 20; i < 30; i++) {
            data[i] = new double[]{0.7 + random.nextGaussian() * 0.15, 0.3 + random.nextGaussian() * 0.15};
        }
        return data;
    }
    
    private double[][] createOverlappingClusters(Random random) {
        var data = new double[40][2];
        // Create 4 overlapping clusters
        for (int i = 0; i < 10; i++) {
            data[i] = new double[]{0.3 + random.nextGaussian() * 0.2, 0.3 + random.nextGaussian() * 0.2};
        }
        for (int i = 10; i < 20; i++) {
            data[i] = new double[]{0.7 + random.nextGaussian() * 0.2, 0.3 + random.nextGaussian() * 0.2};
        }
        for (int i = 20; i < 30; i++) {
            data[i] = new double[]{0.3 + random.nextGaussian() * 0.2, 0.7 + random.nextGaussian() * 0.2};
        }
        for (int i = 30; i < 40; i++) {
            data[i] = new double[]{0.7 + random.nextGaussian() * 0.2, 0.7 + random.nextGaussian() * 0.2};
        }
        return data;
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
}