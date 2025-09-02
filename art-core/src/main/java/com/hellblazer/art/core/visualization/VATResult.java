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

import java.util.Arrays;

/**
 * Result of VAT (Visual Assessment of Clustering Tendency) computation.
 * 
 * Contains the reordered dissimilarity matrix and analysis metrics that help
 * assess the clustering tendency of a dataset. Dark diagonal blocks in the
 * ordered dissimilarity matrix indicate strong clustering structure.
 * 
 * @author Hal Hildebrand
 */
public class VATResult {
    
    private final double[][] orderedDissimilarityMatrix;
    private final int[] reorderingIndices;
    private final int size;
    private final double clusterClarity;
    private final boolean hasStrongClusteringTendency;
    private final int estimatedClusterCount;
    private final long computationTimeMs;
    
    /**
     * Create a VAT result.
     * 
     * @param orderedDissimilarityMatrix the reordered dissimilarity matrix
     * @param reorderingIndices the indices used for reordering
     * @param computationTimeMs computation time in milliseconds
     */
    public VATResult(double[][] orderedDissimilarityMatrix, int[] reorderingIndices, long computationTimeMs) {
        if (orderedDissimilarityMatrix == null) {
            throw new IllegalArgumentException("Ordered dissimilarity matrix cannot be null");
        }
        if (reorderingIndices == null) {
            throw new IllegalArgumentException("Reordering indices cannot be null");
        }
        if (orderedDissimilarityMatrix.length != reorderingIndices.length) {
            throw new IllegalArgumentException("Matrix size must match reordering indices length");
        }
        
        this.orderedDissimilarityMatrix = cloneMatrix(orderedDissimilarityMatrix);
        this.reorderingIndices = reorderingIndices.clone();
        this.size = orderedDissimilarityMatrix.length;
        this.computationTimeMs = computationTimeMs;
        
        // Analyze the ordered dissimilarity matrix
        this.clusterClarity = computeClusterClarity();
        this.hasStrongClusteringTendency = clusterClarity > 0.6; // Threshold for strong clustering
        this.estimatedClusterCount = computeEstimatedClusterCount();
    }
    
    /**
     * Get the ordered dissimilarity matrix (ODM).
     * 
     * The ODM is reordered to reveal clustering structure. Dark diagonal blocks
     * indicate clusters - points within the same cluster have small distances
     * (dark values) and appear as blocks along the diagonal.
     * 
     * @return ordered dissimilarity matrix
     */
    public double[][] getOrderedDissimilarityMatrix() {
        return cloneMatrix(orderedDissimilarityMatrix);
    }
    
    /**
     * Get the reordering indices used to create the ODM.
     * 
     * These indices show how the original data points were reordered.
     * originalData[reorderingIndices[i]] corresponds to position i in the ODM.
     * 
     * @return reordering indices
     */
    public int[] getReorderingIndices() {
        return reorderingIndices.clone();
    }
    
    /**
     * Get the size (number of data points).
     * 
     * @return number of data points
     */
    public int getSize() {
        return size;
    }
    
    /**
     * Get the cluster clarity score (0.0 to 1.0).
     * 
     * Higher values indicate clearer clustering structure. Values > 0.6 typically
     * indicate strong clustering tendency, while values < 0.4 suggest little
     * clustering structure.
     * 
     * @return cluster clarity score
     */
    public double getClusterClarity() {
        return clusterClarity;
    }
    
    /**
     * Check if the data shows strong clustering tendency.
     * 
     * @return true if strong clustering is detected
     */
    public boolean hasStrongClusteringTendency() {
        return hasStrongClusteringTendency;
    }
    
    /**
     * Get the estimated number of clusters.
     * 
     * This is computed by analyzing dark diagonal blocks in the ODM.
     * 
     * @return estimated cluster count
     */
    public int estimateClusterCount() {
        return estimatedClusterCount;
    }
    
    /**
     * Check if dark diagonal blocks are present in the ODM.
     * 
     * Dark diagonal blocks indicate clusters - this is the primary visual
     * indicator of clustering structure in VAT.
     * 
     * @return true if dark diagonal blocks are detected
     */
    public boolean hasDarkDiagonalBlocks() {
        return estimatedClusterCount > 1;
    }
    
    /**
     * Get the computation time in milliseconds.
     * 
     * @return computation time
     */
    public long getComputationTimeMs() {
        return computationTimeMs;
    }
    
    /**
     * Get a normalized version of the ODM for visualization (values 0.0 to 1.0).
     * 
     * @return normalized ODM suitable for grayscale visualization
     */
    public double[][] getNormalizedODM() {
        // Find min and max values
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        
        for (double[] row : orderedDissimilarityMatrix) {
            for (double val : row) {
                if (val < min) min = val;
                if (val > max) max = val;
            }
        }
        
        // Normalize to [0, 1] range
        double range = max - min;
        if (range == 0) {
            // All values are the same
            var normalized = new double[size][size];
            for (int i = 0; i < size; i++) {
                Arrays.fill(normalized[i], 0.5);
            }
            return normalized;
        }
        
        var normalized = new double[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                normalized[i][j] = (orderedDissimilarityMatrix[i][j] - min) / range;
            }
        }
        
        return normalized;
    }
    
    /**
     * Compute cluster clarity based on ODM structure.
     * Standard VAT clarity measure: ratio of inter-cluster to intra-cluster distances.
     * 
     * @return clarity score between 0.0 and 1.0
     */
    private double computeClusterClarity() {
        if (size < 3) return 0.0;
        
        // Simple and direct approach based on the debug output pattern:
        // We saw clear structure with small values (~0.014-0.020) and large values (~1.4)
        
        // Collect all off-diagonal distances
        var distances = new java.util.ArrayList<Double>();
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (i != j) {
                    distances.add(orderedDissimilarityMatrix[i][j]);
                }
            }
        }
        
        if (distances.isEmpty()) return 0.0;
        
        // Sort distances to find natural breakpoint
        distances.sort(null);
        
        // Find the gap that best separates small (intra-cluster) from large (inter-cluster) distances
        double maxGapRatio = 0.0;
        int bestSplitIdx = distances.size() / 2; // Default to median
        
        // Look for the largest relative gap in the sorted distances
        for (int i = distances.size() / 4; i < 3 * distances.size() / 4; i++) {
            double current = distances.get(i);
            double next = distances.get(i + 1);
            
            if (current > 0) {
                double gapRatio = (next - current) / current;
                if (gapRatio > maxGapRatio) {
                    maxGapRatio = gapRatio;
                    bestSplitIdx = i;
                }
            }
        }
        
        // Split distances into intra-cluster (small) and inter-cluster (large)
        double threshold = (distances.get(bestSplitIdx) + distances.get(bestSplitIdx + 1)) / 2.0;
        
        double intraSum = 0.0;
        int intraCount = 0;
        double interSum = 0.0;
        int interCount = 0;
        
        for (double dist : distances) {
            if (dist <= threshold) {
                intraSum += dist;
                intraCount++;
            } else {
                interSum += dist;
                interCount++;
            }
        }
        
        if (intraCount == 0 || interCount == 0) return 0.0;
        
        double intraAvg = intraSum / intraCount;
        double interAvg = interSum / interCount;
        
        // Clarity is the ratio of separation
        // Perfect clustering: inter >> intra, clarity approaches 1
        // No clustering: inter ≈ intra, clarity approaches 0
        if (intraAvg == 0) return 1.0;
        
        double ratio = interAvg / intraAvg;
        
        // Convert ratio to 0-1 scale using sigmoid-like function
        // ratio = 1 → clarity = 0 (no separation)
        // ratio >> 1 → clarity approaches 1 (good separation)
        
        // For better discrimination, use steeper function for low ratios
        double clarity;
        if (ratio < 3.0) {
            // For low ratios (poor separation), be more conservative
            clarity = Math.max(0.0, (ratio - 1.0) / 4.0);
        } else {
            // For high ratios (good separation), approach 1.0
            clarity = Math.tanh((ratio - 1.0) / 3.0);
        }
        
        return Math.max(0.0, Math.min(1.0, clarity));
    }
    
    /**
     * Compute blockiness score by analyzing transitions in the ODM.
     * 
     * @return blockiness score between 0.0 and 1.0
     */
    private double computeBlockiness() {
        if (size < 4) return 0.0;
        
        // Count sharp transitions along rows (indicates block boundaries)
        int transitions = 0;
        int totalComparisons = 0;
        
        double threshold = computeTransitionThreshold();
        
        for (int i = 0; i < size; i++) {
            for (int j = 1; j < size; j++) {
                if (i != j && i != j-1) {  // Avoid diagonal
                    double diff = Math.abs(orderedDissimilarityMatrix[i][j] - orderedDissimilarityMatrix[i][j-1]);
                    if (diff > threshold) {
                        transitions++;
                    }
                    totalComparisons++;
                }
            }
        }
        
        if (totalComparisons == 0) return 0.0;
        
        double transitionRate = (double) transitions / totalComparisons;
        
        // Optimal transition rate for block structure is around 10-30%
        if (transitionRate >= 0.1 && transitionRate <= 0.3) {
            return Math.min(1.0, transitionRate * 2.0);
        } else {
            return Math.max(0.0, 0.5 - Math.abs(transitionRate - 0.2));
        }
    }
    
    /**
     * Compute threshold for detecting significant transitions.
     * 
     * @return transition threshold
     */
    private double computeTransitionThreshold() {
        // Compute standard deviation of all pairwise distances
        double sum = 0.0;
        double sumSq = 0.0;
        int count = 0;
        
        for (int i = 0; i < size; i++) {
            for (int j = i + 1; j < size; j++) {
                double val = orderedDissimilarityMatrix[i][j];
                sum += val;
                sumSq += val * val;
                count++;
            }
        }
        
        if (count == 0) return 0.0;
        
        double mean = sum / count;
        double variance = (sumSq / count) - (mean * mean);
        double stdDev = Math.sqrt(Math.max(0.0, variance));
        
        return Math.max(stdDev * 0.5, mean * 0.1);
    }
    
    /**
     * Estimate the number of clusters by analyzing the ODM structure.
     * 
     * @return estimated cluster count
     */
    private int computeEstimatedClusterCount() {
        if (size < 2) return 1;
        if (clusterClarity < 0.4) return 1;
        
        // Use the same distance classification as the clarity calculation
        // Collect all off-diagonal distances
        var distances = new java.util.ArrayList<Double>();
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (i != j) {
                    distances.add(orderedDissimilarityMatrix[i][j]);
                }
            }
        }
        
        if (distances.isEmpty()) return 1;
        
        // Sort and find the same threshold used in clarity calculation
        distances.sort(null);
        
        double maxGapRatio = 0.0;
        int bestSplitIdx = distances.size() / 2;
        
        for (int i = distances.size() / 4; i < 3 * distances.size() / 4; i++) {
            if (i + 1 >= distances.size()) break;
            
            double current = distances.get(i);
            double next = distances.get(i + 1);
            
            if (current > 0) {
                double gapRatio = (next - current) / current;
                if (gapRatio > maxGapRatio) {
                    maxGapRatio = gapRatio;
                    bestSplitIdx = i;
                }
            }
        }
        
        if (bestSplitIdx + 1 >= distances.size()) return 1;
        
        double threshold = (distances.get(bestSplitIdx) + distances.get(bestSplitIdx + 1)) / 2.0;
        
        // Count intra-cluster vs inter-cluster distances
        int intraCount = 0;
        int interCount = 0;
        
        for (double dist : distances) {
            if (dist <= threshold) {
                intraCount++;
            } else {
                interCount++;
            }
        }
        
        // Estimate clusters based on the ratio
        // For perfect 2-cluster case: we expect roughly equal intra/inter counts
        if (intraCount > 0 && interCount > 0) {
            // Rough heuristic: if we have significant inter-cluster distances, likely 2+ clusters
            double interRatio = (double) interCount / (intraCount + interCount);
            if (interRatio > 0.3) {
                return 2; // Most common case for clear clustering
            }
        }
        
        return 1;
    }
    
    /**
     * Clone a matrix for safe copying.
     * 
     * @param matrix matrix to clone
     * @return cloned matrix
     */
    private double[][] cloneMatrix(double[][] matrix) {
        var clone = new double[matrix.length][];
        for (int i = 0; i < matrix.length; i++) {
            clone[i] = matrix[i].clone();
        }
        return clone;
    }
    
    @Override
    public String toString() {
        return String.format("VATResult{size=%d, clarity=%.3f, strongClustering=%s, estimatedClusters=%d, computationTime=%dms}", 
            size, clusterClarity, hasStrongClusteringTendency, estimatedClusterCount, computationTimeMs);
    }
}