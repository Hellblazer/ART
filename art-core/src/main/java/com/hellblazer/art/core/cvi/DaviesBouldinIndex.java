package com.hellblazer.art.core.cvi;

import com.hellblazer.art.core.Pattern;
import java.util.*;

/**
 * Davies-Bouldin Index implementation.
 * 
 * DB = (1/k) * Σ max_{i≠j} ((s_i + s_j) / d_{ij})
 * where:
 * - k is the number of clusters
 * - s_i is the average distance from points in cluster i to its centroid
 * - d_{ij} is the distance between centroids of clusters i and j
 * 
 * Lower values indicate better clustering.
 */
public class DaviesBouldinIndex implements ClusterValidityIndex {
    
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
        
        if (k <= 1) {
            return 0.0; // Undefined for single cluster
        }
        
        // Calculate or use provided centroids
        var finalCentroids = centroids != null && !centroids.isEmpty() 
            ? centroids 
            : calculateCentroids(data, labels, k);
        
        // Calculate scatter for each cluster (average distance to centroid)
        var scatter = calculateScatter(data, labels, finalCentroids, k);
        
        // Calculate DB index
        double dbSum = 0.0;
        int validClusters = 0;
        
        for (int i = 0; i < finalCentroids.size(); i++) {
            if (finalCentroids.get(i) == null) {
                continue; // Skip empty clusters
            }
            
            double maxRatio = 0.0;
            
            for (int j = 0; j < finalCentroids.size(); j++) {
                if (i == j || finalCentroids.get(j) == null) {
                    continue;
                }
                
                double centroidDistance = distance(finalCentroids.get(i), finalCentroids.get(j));
                
                if (centroidDistance > 0) {
                    double ratio = (scatter[i] + scatter[j]) / centroidDistance;
                    maxRatio = Math.max(maxRatio, ratio);
                }
            }
            
            dbSum += maxRatio;
            validClusters++;
        }
        
        return validClusters > 0 ? dbSum / validClusters : 0.0;
    }
    
    @Override
    public String getName() {
        return "Davies-Bouldin Index";
    }
    
    @Override
    public boolean isHigherBetter() {
        return false; // Lower is better
    }
    
    private List<Pattern> calculateCentroids(List<Pattern> data, int[] labels, int k) {
        // Find the actual maximum label to handle non-contiguous labels
        int maxLabel = 0;
        for (int label : labels) {
            maxLabel = Math.max(maxLabel, label);
        }
        int arraySize = Math.max(k, maxLabel + 1);
        
        var centroids = new ArrayList<Pattern>(Collections.nCopies(arraySize, null));
        var counts = new int[arraySize];
        var sums = new double[arraySize][];
        
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
        for (int i = 0; i < arraySize; i++) {
            if (counts[i] > 0) {
                for (int d = 0; d < sums[i].length; d++) {
                    sums[i][d] /= counts[i];
                }
                centroids.set(i, Pattern.of(sums[i]));
            }
        }
        
        return centroids;
    }
    
    private double[] calculateScatter(List<Pattern> data, int[] labels, List<Pattern> centroids, int k) {
        // Handle non-contiguous labels
        int maxLabel = 0;
        for (int label : labels) {
            maxLabel = Math.max(maxLabel, label);
        }
        int arraySize = Math.max(k, maxLabel + 1);
        
        var scatter = new double[arraySize];
        var counts = new int[arraySize];
        
        for (int i = 0; i < data.size(); i++) {
            int label = labels[i];
            var centroid = centroids.get(label);
            
            if (centroid != null) {
                scatter[label] += distance(data.get(i), centroid);
                counts[label]++;
            }
        }
        
        // Average the scatter
        for (int i = 0; i < arraySize; i++) {
            if (counts[i] > 0) {
                scatter[i] /= counts[i];
            }
        }
        
        return scatter;
    }
    
    private double distance(Pattern a, Pattern b) {
        if (a == null || b == null) {
            return 0.0;
        }
        
        double sum = 0.0;
        for (int i = 0; i < a.dimension(); i++) {
            double diff = a.get(i) - b.get(i);
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
}