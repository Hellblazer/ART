package com.hellblazer.art.core.cvi;

import com.hellblazer.art.core.Pattern;
import java.util.*;

/**
 * Calinski-Harabasz Index (CH Index) implementation.
 * Also known as the Variance Ratio Criterion.
 * 
 * CH = (SSB / (k - 1)) / (SSW / (n - k))
 * where:
 * - SSB is the between-cluster sum of squares
 * - SSW is the within-cluster sum of squares
 * - k is the number of clusters
 * - n is the number of data points
 * 
 * Higher values indicate better clustering.
 */
public class CalinskiHarabaszIndex implements ClusterValidityIndex {
    
    // Incremental tracking variables
    private boolean incrementalMode = false;
    private final Map<Integer, List<Pattern>> clusterData = new HashMap<>();
    private final Map<Integer, Pattern> clusterCentroids = new HashMap<>();
    private Pattern globalCentroid;
    private int totalPoints = 0;
    
    @Override
    public double calculate(List<Pattern> data, int[] labels, List<Pattern> centroids) {
        if (data.isEmpty() || labels.length != data.size()) {
            throw new IllegalArgumentException("Invalid data or labels");
        }
        
        // Find unique clusters
        var uniqueClusters = new HashSet<Integer>();
        for (int label : labels) {
            uniqueClusters.add(label);
        }
        
        int k = uniqueClusters.size();
        int n = data.size();
        
        if (k <= 1 || k >= n) {
            return 0.0; // Undefined for single cluster or each point as cluster
        }
        
        // Calculate or use provided centroids
        var finalCentroids = centroids != null && centroids.size() == k 
            ? centroids 
            : calculateCentroids(data, labels, k);
        
        // Calculate global centroid
        var global = calculateGlobalCentroid(data);
        
        // Calculate between-cluster sum of squares (SSB)
        double ssb = calculateSSB(data, labels, finalCentroids, global);
        
        // Calculate within-cluster sum of squares (SSW)
        double ssw = calculateSSW(data, labels, finalCentroids);
        
        // Calculate CH index
        if (ssw == 0) {
            return Double.POSITIVE_INFINITY; // Perfect clustering
        }
        
        return (ssb / (k - 1)) / (ssw / (n - k));
    }
    
    @Override
    public String getName() {
        return "Calinski-Harabasz Index";
    }
    
    @Override
    public boolean isHigherBetter() {
        return true;
    }
    
    @Override
    public boolean updateIncremental(Pattern dataPoint, int clusterLabel) {
        incrementalMode = true;
        
        // Add data point to cluster
        clusterData.computeIfAbsent(clusterLabel, k -> new ArrayList<>()).add(dataPoint);
        totalPoints++;
        
        // Update cluster centroid
        updateClusterCentroid(clusterLabel);
        
        // Update global centroid
        updateGlobalCentroid();
        
        return true;
    }
    
    @Override
    public double getIncrementalValue() {
        if (!incrementalMode || clusterData.isEmpty()) {
            throw new IllegalStateException("No incremental data available");
        }
        
        int k = clusterData.size();
        int n = totalPoints;
        
        if (k <= 1 || k >= n) {
            return 0.0;
        }
        
        // Calculate SSB and SSW from incremental data
        double ssb = 0.0;
        double ssw = 0.0;
        
        for (var entry : clusterData.entrySet()) {
            var clusterPoints = entry.getValue();
            var centroid = clusterCentroids.get(entry.getKey());
            
            // Contribution to SSB
            int clusterSize = clusterPoints.size();
            ssb += clusterSize * squaredDistance(centroid, globalCentroid);
            
            // Contribution to SSW
            for (var point : clusterPoints) {
                ssw += squaredDistance(point, centroid);
            }
        }
        
        if (ssw == 0) {
            return Double.POSITIVE_INFINITY;
        }
        
        return (ssb / (k - 1)) / (ssw / (n - k));
    }
    
    @Override
    public void resetIncremental() {
        incrementalMode = false;
        clusterData.clear();
        clusterCentroids.clear();
        globalCentroid = null;
        totalPoints = 0;
    }
    
    private List<Pattern> calculateCentroids(List<Pattern> data, int[] labels, int k) {
        var centroids = new ArrayList<Pattern>(Collections.nCopies(k, null));
        var counts = new int[k];
        var sums = new double[k][];
        
        // Initialize sums arrays
        for (int i = 0; i < data.size(); i++) {
            int label = labels[i];
            if (sums[label] == null) {
                sums[label] = new double[data.get(i).dimension()];
            }
            
            // Add to sum
            for (int d = 0; d < data.get(i).dimension(); d++) {
                sums[label][d] += data.get(i).get(d);
            }
            counts[label]++;
        }
        
        // Calculate centroids
        for (int i = 0; i < k; i++) {
            if (counts[i] > 0) {
                for (int d = 0; d < sums[i].length; d++) {
                    sums[i][d] /= counts[i];
                }
                centroids.set(i, Pattern.of(sums[i]));
            }
        }
        
        return centroids;
    }
    
    private Pattern calculateGlobalCentroid(List<Pattern> data) {
        if (data.isEmpty()) {
            return null;
        }
        
        int dim = data.get(0).dimension();
        var sum = new double[dim];
        
        for (var point : data) {
            for (int d = 0; d < dim; d++) {
                sum[d] += point.get(d);
            }
        }
        
        for (int d = 0; d < dim; d++) {
            sum[d] /= data.size();
        }
        
        return Pattern.of(sum);
    }
    
    private double calculateSSB(List<Pattern> data, int[] labels, List<Pattern> centroids, Pattern global) {
        var clusterSizes = new int[centroids.size()];
        
        for (int label : labels) {
            clusterSizes[label]++;
        }
        
        double ssb = 0.0;
        for (int i = 0; i < centroids.size(); i++) {
            if (centroids.get(i) != null) {
                ssb += clusterSizes[i] * squaredDistance(centroids.get(i), global);
            }
        }
        
        return ssb;
    }
    
    private double calculateSSW(List<Pattern> data, int[] labels, List<Pattern> centroids) {
        double ssw = 0.0;
        
        for (int i = 0; i < data.size(); i++) {
            var centroid = centroids.get(labels[i]);
            if (centroid != null) {
                ssw += squaredDistance(data.get(i), centroid);
            }
        }
        
        return ssw;
    }
    
    private double squaredDistance(Pattern a, Pattern b) {
        if (a == null || b == null) {
            return 0.0;
        }
        
        double sum = 0.0;
        for (int i = 0; i < a.dimension(); i++) {
            double diff = a.get(i) - b.get(i);
            sum += diff * diff;
        }
        return sum;
    }
    
    private void updateClusterCentroid(int clusterLabel) {
        var points = clusterData.get(clusterLabel);
        if (points == null || points.isEmpty()) {
            return;
        }
        
        int dim = points.get(0).dimension();
        var sum = new double[dim];
        
        for (var point : points) {
            for (int d = 0; d < dim; d++) {
                sum[d] += point.get(d);
            }
        }
        
        for (int d = 0; d < dim; d++) {
            sum[d] /= points.size();
        }
        
        clusterCentroids.put(clusterLabel, Pattern.of(sum));
    }
    
    private void updateGlobalCentroid() {
        if (totalPoints == 0) {
            return;
        }
        
        // Get dimension from first available point
        int dim = 0;
        for (var points : clusterData.values()) {
            if (!points.isEmpty()) {
                dim = points.get(0).dimension();
                break;
            }
        }
        
        var sum = new double[dim];
        
        for (var points : clusterData.values()) {
            for (var point : points) {
                for (int d = 0; d < dim; d++) {
                    sum[d] += point.get(d);
                }
            }
        }
        
        for (int d = 0; d < dim; d++) {
            sum[d] /= totalPoints;
        }
        
        globalCentroid = Pattern.of(sum);
    }
}