package com.hellblazer.art.core.cvi;

import com.hellblazer.art.core.Pattern;
import java.util.List;

/**
 * Interface for Cluster Validity Indices (CVIs).
 * CVIs are metrics used to evaluate the quality of clustering results.
 */
public interface ClusterValidityIndex {
    
    /**
     * Calculate the validity index for the given clustering.
     * 
     * @param data the original data points
     * @param labels the cluster assignments for each data point
     * @param centroids the cluster centroids (optional, can be computed from data)
     * @return the validity index value
     */
    double calculate(List<Pattern> data, int[] labels, List<Pattern> centroids);
    
    /**
     * Get the name of this validity index.
     * 
     * @return the name of the index
     */
    String getName();
    
    /**
     * Indicates whether higher values are better for this index.
     * 
     * @return true if higher values indicate better clustering, false otherwise
     */
    boolean isHigherBetter();
    
    /**
     * Update the index incrementally with a new data point assignment.
     * Not all indices support incremental updates.
     * 
     * @param dataPoint the new data point
     * @param clusterLabel the cluster assignment
     * @return true if the update was successful, false if incremental updates are not supported
     */
    default boolean updateIncremental(Pattern dataPoint, int clusterLabel) {
        return false; // Default: incremental updates not supported
    }
    
    /**
     * Get the current incremental value of the index.
     * Only valid if incremental updates are supported and have been performed.
     * 
     * @return the current incremental value
     * @throws UnsupportedOperationException if incremental updates are not supported
     */
    default double getIncrementalValue() {
        throw new UnsupportedOperationException("Incremental updates not supported for " + getName());
    }
    
    /**
     * Reset the incremental state of the index.
     */
    default void resetIncremental() {
        // Default: no-op
    }
}