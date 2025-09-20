package com.hellblazer.art.core;

import com.hellblazer.art.core.cvi.ClusterValidityIndex;
import com.hellblazer.art.core.results.ActivationResult;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;

/**
 * Extension of BaseART that includes Cluster Validity Index (CVI) tracking.
 * This class adds the ability to monitor clustering quality in real-time
 * using various validity indices.
 */
public abstract class CVIEnabledART extends BaseART {
    
    private final List<ClusterValidityIndex> cviMetrics = new ArrayList<>();
    private final Map<String, Double> latestScores = new ConcurrentHashMap<>();
    private final List<Pattern> dataHistory = new ArrayList<>();
    private final List<Integer> labelHistory = new ArrayList<>();
    private boolean trackHistory = true;
    private int maxHistorySize = 10000; // Limit memory usage
    
    /**
     * Add a CVI metric to track during learning.
     * @param cvi the cluster validity index to track
     */
    public void addCVIMetric(ClusterValidityIndex cvi) {
        cviMetrics.add(cvi);
    }
    
    /**
     * Remove a CVI metric from tracking.
     * @param cvi the cluster validity index to remove
     */
    public void removeCVIMetric(ClusterValidityIndex cvi) {
        cviMetrics.remove(cvi);
        latestScores.remove(cvi.getName());
    }
    
    /**
     * Clear all CVI metrics.
     */
    public void clearCVIMetrics() {
        cviMetrics.clear();
        latestScores.clear();
    }
    
    /**
     * Get the latest CVI scores.
     * @return map of CVI names to their latest scores
     */
    public Map<String, Double> getCVIScores() {
        return new ConcurrentHashMap<>(latestScores);
    }
    
    /**
     * Get a specific CVI score by name.
     * @param name the name of the CVI
     * @return the score, or null if not tracked
     */
    public Double getCVIScore(String name) {
        return latestScores.get(name);
    }
    
    /**
     * Enable or disable history tracking for CVI calculation.
     * @param track whether to track history
     */
    public void setTrackHistory(boolean track) {
        this.trackHistory = track;
        if (!track) {
            dataHistory.clear();
            labelHistory.clear();
        }
    }
    
    /**
     * Set the maximum history size for CVI calculation.
     * @param maxSize the maximum number of patterns to keep in history
     */
    public void setMaxHistorySize(int maxSize) {
        this.maxHistorySize = maxSize;
        trimHistory();
    }
    
    /**
     * Calculate current CVI scores based on the data history.
     * This method should be called periodically or after significant changes.
     */
    public void updateCVIScores() {
        if (!trackHistory || dataHistory.isEmpty() || cviMetrics.isEmpty()) {
            // Only clear scores if we're tracking and have no data
            if (trackHistory && dataHistory.isEmpty()) {
                latestScores.clear();
            }
            return;
        }
        
        // Convert label history to array
        int[] labels = labelHistory.stream().mapToInt(Integer::intValue).toArray();
        
        // Calculate centroids for current categories
        var centroids = calculateCentroids();
        
        // Calculate each CVI
        for (var cvi : cviMetrics) {
            try {
                double score = cvi.calculate(dataHistory, labels, centroids);
                latestScores.put(cvi.getName(), score);
            } catch (Exception e) {
                // Log error but don't fail the learning process
                // Log CVI calculation error - handled gracefully
            }
        }
    }
    
    /**
     * Track patterns after learning.
     * Call this method after using stepFit to track data for CVI calculation.
     * @param input the input pattern that was learned
     * @param result the result from stepFit
     */
    public void trackPattern(Pattern input, ActivationResult result) {
        if (trackHistory && !cviMetrics.isEmpty() && result instanceof ActivationResult.Success success) {
            dataHistory.add(input);
            labelHistory.add(success.categoryIndex());
            trimHistory();
            
            // Update CVIs periodically (e.g., every 10 patterns)
            if (dataHistory.size() % 10 == 0) {
                updateCVIScores();
            }
        }
    }
    
    /**
     * Calculate centroids for all categories.
     * @return list of centroids
     */
    private List<Pattern> calculateCentroids() {
        var centroids = new ArrayList<Pattern>();
        
        for (int i = 0; i < getCategoryCount(); i++) {
            // Get the weight vector for this category
            if (i < categories.size()) {
                var weight = categories.get(i);
                // Convert weight to Pattern
                // This is a simplified approach - subclasses may override
                centroids.add(weightToPattern((WeightVector) weight));
            } else {
                centroids.add(null);
            }
        }
        
        return centroids;
    }
    
    /**
     * Convert a weight vector to a Pattern for CVI calculation.
     * Subclasses should override this to provide appropriate conversion.
     * @param weight the weight vector
     * @return the corresponding pattern
     */
    protected Pattern weightToPattern(WeightVector weight) {
        // Default implementation - try to extract values
        if (weight instanceof SimpleWeight sw) {
            return new DenseVector(sw.getValues());
        }
        // Subclasses should override for other weight types
        throw new UnsupportedOperationException(
            "weightToPattern must be overridden for weight type: " + weight.getClass()
        );
    }
    
    /**
     * Trim history to maintain maximum size.
     */
    private void trimHistory() {
        while (dataHistory.size() > maxHistorySize) {
            dataHistory.remove(0);
            labelHistory.remove(0);
        }
    }
    
    /**
     * Get summary statistics for all tracked CVIs.
     * @return formatted string with CVI statistics
     */
    public String getCVISummary() {
        if (latestScores.isEmpty()) {
            return "No CVI metrics tracked";
        }
        
        var sb = new StringBuilder();
        sb.append("CVI Scores:\n");
        for (var entry : latestScores.entrySet()) {
            sb.append(String.format("  %s: %.4f\n", entry.getKey(), entry.getValue()));
        }
        sb.append(String.format("History size: %d patterns\n", dataHistory.size()));
        sb.append(String.format("Categories: %d\n", getCategoryCount()));
        
        return sb.toString();
    }
    
    /**
     * Reset CVI tracking history.
     */
    public void resetCVIHistory() {
        dataHistory.clear();
        labelHistory.clear();
        latestScores.clear();
    }
    
    /**
     * Simple weight implementation for testing.
     * Production code should use proper weight implementations.
     */
    public static class SimpleWeight implements WeightVector {
        private final double[] values;
        
        public SimpleWeight(double[] values) {
            this.values = values.clone();
        }
        
        public double[] getValues() {
            return values.clone();
        }
        
        @Override
        public double get(int index) {
            return values[index];
        }
        
        @Override
        public int dimension() {
            return values.length;
        }
        
        @Override
        public double l1Norm() {
            double sum = 0;
            for (double v : values) {
                sum += Math.abs(v);
            }
            return sum;
        }
        
        @Override
        public WeightVector update(Pattern input, Object parameters) {
            // Simple averaging update for testing
            var newValues = new double[values.length];
            for (int i = 0; i < values.length; i++) {
                newValues[i] = (values[i] + input.get(i)) / 2.0;
            }
            return new SimpleWeight(newValues);
        }
    }
}