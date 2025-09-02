package com.hellblazer.art.core.cvi;

import com.hellblazer.art.core.Pattern;
import java.util.*;

/**
 * Silhouette Coefficient implementation.
 * 
 * For each point i:
 * - a(i) = average distance to other points in same cluster
 * - b(i) = minimum average distance to points in other clusters
 * - s(i) = (b(i) - a(i)) / max(a(i), b(i))
 * 
 * Overall silhouette = average of all s(i)
 * 
 * Values range from -1 to 1:
 * - Near 1: point is well-matched to its cluster
 * - Near 0: point is on boundary between clusters
 * - Near -1: point may be assigned to wrong cluster
 * 
 * Higher values indicate better clustering.
 */
public class SilhouetteCoefficient implements ClusterValidityIndex {
    
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
        
        // Group data by cluster
        var clusters = groupByCluster(data, labels);
        
        // Calculate silhouette for each point
        double totalSilhouette = 0.0;
        int validPoints = 0;
        
        for (int i = 0; i < data.size(); i++) {
            double silhouette = calculatePointSilhouette(data.get(i), labels[i], clusters);
            if (!Double.isNaN(silhouette)) {
                totalSilhouette += silhouette;
                validPoints++;
            }
        }
        
        return validPoints > 0 ? totalSilhouette / validPoints : 0.0;
    }
    
    @Override
    public String getName() {
        return "Silhouette Coefficient";
    }
    
    @Override
    public boolean isHigherBetter() {
        return true;
    }
    
    private Map<Integer, List<Pattern>> groupByCluster(List<Pattern> data, int[] labels) {
        var clusters = new HashMap<Integer, List<Pattern>>();
        
        for (int i = 0; i < data.size(); i++) {
            clusters.computeIfAbsent(labels[i], k -> new ArrayList<>()).add(data.get(i));
        }
        
        return clusters;
    }
    
    private double calculatePointSilhouette(Pattern point, int clusterLabel, 
                                           Map<Integer, List<Pattern>> clusters) {
        var ownCluster = clusters.get(clusterLabel);
        
        if (ownCluster == null || ownCluster.size() <= 1) {
            return 0.0; // Cannot calculate for single-point cluster
        }
        
        // Calculate a(i) - average distance to points in same cluster
        double a = calculateAverageDistance(point, ownCluster, true);
        
        // Calculate b(i) - minimum average distance to points in other clusters
        double b = Double.POSITIVE_INFINITY;
        
        for (var entry : clusters.entrySet()) {
            if (entry.getKey() != clusterLabel) {
                double avgDist = calculateAverageDistance(point, entry.getValue(), false);
                b = Math.min(b, avgDist);
            }
        }
        
        if (b == Double.POSITIVE_INFINITY) {
            return 0.0; // No other clusters
        }
        
        // Calculate silhouette coefficient
        double maxAB = Math.max(a, b);
        return maxAB > 0 ? (b - a) / maxAB : 0.0;
    }
    
    private double calculateAverageDistance(Pattern point, List<Pattern> cluster, boolean excludeSelf) {
        if (cluster.isEmpty()) {
            return 0.0;
        }
        
        double totalDistance = 0.0;
        int count = 0;
        
        for (var other : cluster) {
            if (excludeSelf && point == other) {
                continue; // Skip self
            }
            
            totalDistance += distance(point, other);
            count++;
        }
        
        return count > 0 ? totalDistance / count : 0.0;
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