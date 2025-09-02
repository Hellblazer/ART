package com.hellblazer.art.core.cvi;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.DenseVector;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for Cluster Validity Indices
 */
public class ClusterValidityIndexTest {
    
    private List<Pattern> testData;
    private int[] labels;
    private List<Pattern> centroids;
    
    @BeforeEach
    void setUp() {
        // Create well-separated clusters for testing
        testData = new ArrayList<>();
        
        // Cluster 0: Points around (0.2, 0.2)
        testData.add(new DenseVector(new double[]{0.1, 0.1}));
        testData.add(new DenseVector(new double[]{0.2, 0.2}));
        testData.add(new DenseVector(new double[]{0.3, 0.3}));
        testData.add(new DenseVector(new double[]{0.15, 0.25}));
        
        // Cluster 1: Points around (0.8, 0.8)
        testData.add(new DenseVector(new double[]{0.7, 0.7}));
        testData.add(new DenseVector(new double[]{0.8, 0.8}));
        testData.add(new DenseVector(new double[]{0.9, 0.9}));
        testData.add(new DenseVector(new double[]{0.75, 0.85}));
        
        // Cluster 2: Points around (0.5, 0.2)
        testData.add(new DenseVector(new double[]{0.4, 0.1}));
        testData.add(new DenseVector(new double[]{0.5, 0.2}));
        testData.add(new DenseVector(new double[]{0.6, 0.3}));
        testData.add(new DenseVector(new double[]{0.45, 0.15}));
        
        labels = new int[]{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
        
        // Calculate actual centroids
        centroids = calculateCentroids(testData, labels, 3);
    }
    
    @Test
    void testCalinskiHarabaszIndex() {
        var index = new CalinskiHarabaszIndex();
        
        // Test batch calculation
        double score = index.calculate(testData, labels, centroids);
        
        // CH index should be positive for well-separated clusters
        assertTrue(score > 0, "Calinski-Harabasz index should be positive");
        
        // Test with perfect clustering (each point is its own cluster)
        int[] perfectLabels = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        List<Pattern> perfectCentroids = new ArrayList<>(testData);
        double perfectScore = index.calculate(testData, perfectLabels, perfectCentroids);
        
        // Perfect clustering should have lower score than good clustering
        assertTrue(score > perfectScore, 
            "Well-separated clusters should have higher CH score than single-point clusters");
        
        // Test with single cluster (all same label)
        int[] singleLabels = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        List<Pattern> singleCentroid = calculateCentroids(testData, singleLabels, 1);
        double singleScore = index.calculate(testData, singleLabels, singleCentroid);
        
        // Single cluster should return 0 or NaN (no between-cluster variance)
        assertTrue(Double.isNaN(singleScore) || singleScore == 0, 
            "Single cluster should have undefined or zero CH index");
    }
    
    @Test
    void testCalinskiHarabaszIncrementalUpdate() {
        var index = new CalinskiHarabaszIndex();
        
        // Initialize with first batch
        index.calculate(testData.subList(0, 6), 
                       Arrays.copyOf(labels, 6), 
                       calculateCentroids(testData.subList(0, 6), Arrays.copyOf(labels, 6), 2));
        
        // Add points incrementally
        for (int i = 6; i < testData.size(); i++) {
            boolean updated = index.updateIncremental(testData.get(i), labels[i]);
            assertTrue(updated, "Incremental update should succeed");
        }
        
        // Final incremental score
        double incrementalScore = index.calculate(testData, labels, centroids);
        
        // Reset and calculate batch score
        var batchIndex = new CalinskiHarabaszIndex();
        double batchScore = batchIndex.calculate(testData, labels, centroids);
        
        // Incremental and batch should give similar results
        assertEquals(batchScore, incrementalScore, batchScore * 0.1, 
            "Incremental and batch calculation should be similar");
    }
    
    @Test
    void testDaviesBouldinIndex() {
        var index = new DaviesBouldinIndex();
        
        // Test batch calculation
        double score = index.calculate(testData, labels, centroids);
        
        // DB index should be positive (lower is better)
        assertTrue(score > 0, "Davies-Bouldin index should be positive");
        
        // Test with overlapping clusters (worse clustering)
        List<Pattern> overlappingData = new ArrayList<>();
        // Two overlapping clusters
        for (int i = 0; i < 6; i++) {
            overlappingData.add(new DenseVector(new double[]{0.4 + i * 0.02, 0.4 + i * 0.02}));
        }
        for (int i = 0; i < 6; i++) {
            overlappingData.add(new DenseVector(new double[]{0.45 + i * 0.02, 0.45 + i * 0.02}));
        }
        int[] overlappingLabels = new int[]{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
        List<Pattern> overlappingCentroids = calculateCentroids(overlappingData, overlappingLabels, 2);
        
        double overlappingScore = index.calculate(overlappingData, overlappingLabels, overlappingCentroids);
        
        // Overlapping clusters should have worse (higher) DB index
        assertTrue(overlappingScore > score, 
            "Overlapping clusters should have higher (worse) DB index than well-separated clusters");
    }
    
    @Test
    void testDaviesBouldinEdgeCases() {
        var index = new DaviesBouldinIndex();
        
        // Test with single cluster
        int[] singleLabels = new int[]{0, 0, 0, 0};
        List<Pattern> singleData = testData.subList(0, 4);
        List<Pattern> singleCentroid = calculateCentroids(singleData, singleLabels, 1);
        
        double score = index.calculate(singleData, singleLabels, singleCentroid);
        assertTrue(Double.isNaN(score) || score == 0, 
            "Single cluster should have undefined or zero DB index");
        
        // Test with empty cluster (should handle gracefully)
        int[] labelsWithEmpty = new int[]{0, 0, 1, 1, 3, 3}; // No cluster 2
        List<Pattern> dataSubset = testData.subList(0, 6);
        List<Pattern> centroidsWithEmpty = new ArrayList<>();
        centroidsWithEmpty.add(calculateCentroid(dataSubset.subList(0, 2)));
        centroidsWithEmpty.add(calculateCentroid(dataSubset.subList(2, 4)));
        centroidsWithEmpty.add(null); // Empty cluster
        centroidsWithEmpty.add(calculateCentroid(dataSubset.subList(4, 6)));
        
        // Should handle empty cluster gracefully
        assertDoesNotThrow(() -> index.calculate(dataSubset, labelsWithEmpty, centroidsWithEmpty));
    }
    
    @Test
    void testSilhouetteCoefficient() {
        var index = new SilhouetteCoefficient();
        
        // Test batch calculation
        double score = index.calculate(testData, labels, centroids);
        
        // Silhouette should be between -1 and 1
        assertTrue(score >= -1 && score <= 1, 
            "Silhouette coefficient should be between -1 and 1");
        
        // Well-separated clusters should have positive silhouette
        assertTrue(score > 0, 
            "Well-separated clusters should have positive silhouette coefficient");
        
        // Test with perfect clustering (very high silhouette)
        List<Pattern> perfectData = new ArrayList<>();
        // Two very well separated clusters
        for (int i = 0; i < 4; i++) {
            perfectData.add(new DenseVector(new double[]{0.01 * i, 0.01 * i}));
        }
        for (int i = 0; i < 4; i++) {
            perfectData.add(new DenseVector(new double[]{0.9 + 0.01 * i, 0.9 + 0.01 * i}));
        }
        int[] perfectLabels = new int[]{0, 0, 0, 0, 1, 1, 1, 1};
        List<Pattern> perfectCentroids = calculateCentroids(perfectData, perfectLabels, 2);
        
        double perfectScore = index.calculate(perfectData, perfectLabels, perfectCentroids);
        
        // Very well separated clusters should have high silhouette
        assertTrue(perfectScore > 0.5, 
            "Very well separated clusters should have high silhouette coefficient");
        assertTrue(perfectScore > score, 
            "Better separated clusters should have higher silhouette");
    }
    
    @Test
    void testSilhouetteEdgeCases() {
        var index = new SilhouetteCoefficient();
        
        // Test with single point in cluster
        List<Pattern> singlePointData = new ArrayList<>();
        singlePointData.add(new DenseVector(new double[]{0.1, 0.1}));
        singlePointData.add(new DenseVector(new double[]{0.8, 0.8}));
        singlePointData.add(new DenseVector(new double[]{0.9, 0.9}));
        int[] singlePointLabels = new int[]{0, 1, 1};
        List<Pattern> singlePointCentroids = calculateCentroids(singlePointData, singlePointLabels, 2);
        
        double score = index.calculate(singlePointData, singlePointLabels, singlePointCentroids);
        
        // Should handle single-point clusters gracefully
        assertTrue(score >= -1 && score <= 1, 
            "Silhouette should still be valid with single-point clusters");
        
        // Test with all points in one cluster
        int[] singleClusterLabels = new int[]{0, 0, 0, 0, 0, 0};
        List<Pattern> singleClusterData = testData.subList(0, 6);
        List<Pattern> singleClusterCentroid = calculateCentroids(singleClusterData, singleClusterLabels, 1);
        
        score = index.calculate(singleClusterData, singleClusterLabels, singleClusterCentroid);
        
        // Single cluster should have silhouette of 0 or NaN
        assertTrue(Double.isNaN(score) || Math.abs(score) < 0.001, 
            "Single cluster should have undefined or zero silhouette");
    }
    
    @Test
    void testAllIndicesComparison() {
        var chIndex = new CalinskiHarabaszIndex();
        var dbIndex = new DaviesBouldinIndex();
        var silhouette = new SilhouetteCoefficient();
        
        // Test on well-separated clusters
        double chScore = chIndex.calculate(testData, labels, centroids);
        double dbScore = dbIndex.calculate(testData, labels, centroids);
        double silScore = silhouette.calculate(testData, labels, centroids);
        
        // Create poorly separated clusters
        List<Pattern> poorData = new ArrayList<>();
        for (int i = 0; i < 12; i++) {
            poorData.add(new DenseVector(new double[]{
                0.45 + Math.random() * 0.1, 
                0.45 + Math.random() * 0.1
            }));
        }
        int[] poorLabels = new int[]{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
        List<Pattern> poorCentroids = calculateCentroids(poorData, poorLabels, 3);
        
        double chPoorScore = chIndex.calculate(poorData, poorLabels, poorCentroids);
        double dbPoorScore = dbIndex.calculate(poorData, poorLabels, poorCentroids);
        double silPoorScore = silhouette.calculate(poorData, poorLabels, poorCentroids);
        
        // All indices should agree on relative quality
        assertTrue(chScore > chPoorScore, "CH: Good clustering should score higher");
        assertTrue(dbScore < dbPoorScore, "DB: Good clustering should score lower");
        assertTrue(silScore > silPoorScore, "Silhouette: Good clustering should score higher");
    }
    
    // Helper methods
    
    private List<Pattern> calculateCentroids(List<Pattern> data, int[] labels, int k) {
        List<Pattern> centroids = new ArrayList<>();
        
        for (int cluster = 0; cluster < k; cluster++) {
            List<Pattern> clusterPoints = new ArrayList<>();
            for (int i = 0; i < data.size(); i++) {
                if (labels[i] == cluster) {
                    clusterPoints.add(data.get(i));
                }
            }
            if (!clusterPoints.isEmpty()) {
                centroids.add(calculateCentroid(clusterPoints));
            } else {
                centroids.add(null);
            }
        }
        
        return centroids;
    }
    
    private Pattern calculateCentroid(List<Pattern> points) {
        if (points.isEmpty()) return null;
        
        int dim = points.get(0).dimension();
        double[] centroid = new double[dim];
        
        for (var point : points) {
            for (int i = 0; i < dim; i++) {
                centroid[i] += point.get(i);
            }
        }
        
        for (int i = 0; i < dim; i++) {
            centroid[i] /= points.size();
        }
        
        return new DenseVector(centroid);
    }
    
    // No longer need TestPattern - using DenseVector instead
}