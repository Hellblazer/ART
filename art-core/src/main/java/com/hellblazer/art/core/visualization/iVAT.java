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
 * Java implementation of iVAT (improved Visual Assessment of Clustering Tendency).
 * 
 * iVAT enhances the basic VAT algorithm by applying additional transformations
 * to the ordered dissimilarity matrix to make clustering structure more visually
 * apparent. This is particularly beneficial for datasets with subtle, overlapping,
 * or ambiguous cluster structures.
 * 
 * Key improvements over basic VAT:
 * - Path-based distance enhancement
 * - Contrast enhancement for better visualization
 * - Adaptive threshold selection
 * - Multiple enhancement algorithms
 * 
 * @author Hal Hildebrand
 */
public class iVAT {
    
    private static final double DEFAULT_PATH_FACTOR = 1.2;  // More conservative enhancement
    private static final double DEFAULT_CONTRAST_FACTOR = 2.0;  // Stronger contrast enhancement
    
    /**
     * Compute iVAT with default enhancement method.
     * 
     * @param data input data matrix
     * @return enhanced VAT result
     */
    public static VATResult compute(double[][] data) {
        // Use contrast enhancement as default - more reliable than path-based
        return computeWithContrastEnhancement(data);
    }
    
    /**
     * Compute iVAT using path-based enhancement.
     * 
     * Path-based enhancement considers indirect connections between data points
     * through intermediate points, which helps reveal hidden clustering structure.
     * 
     * @param data input data matrix
     * @return enhanced VAT result
     */
    public static VATResult computeWithPathBasedEnhancement(double[][] data) {
        // Start with basic VAT computation
        var basicResult = VAT.compute(data);
        
        long startTime = System.currentTimeMillis();
        
        // Apply path-based enhancement to the ordered dissimilarity matrix
        var enhancedMatrix = applyPathBasedEnhancement(
            basicResult.getOrderedDissimilarityMatrix(), 
            DEFAULT_PATH_FACTOR
        );
        
        long endTime = System.currentTimeMillis();
        
        // Create enhanced result
        return new VATResult(
            enhancedMatrix, 
            basicResult.getReorderingIndices(), 
            endTime - startTime
        );
    }
    
    /**
     * Compute iVAT using contrast enhancement.
     * 
     * Contrast enhancement amplifies the differences between within-cluster
     * and between-cluster distances to make cluster boundaries more apparent.
     * 
     * @param data input data matrix
     * @return enhanced VAT result
     */
    public static VATResult computeWithContrastEnhancement(double[][] data) {
        // Start with basic VAT computation
        var basicResult = VAT.compute(data);
        
        long startTime = System.currentTimeMillis();
        
        // Apply contrast enhancement
        var enhancedMatrix = applyContrastEnhancement(
            basicResult.getOrderedDissimilarityMatrix(),
            DEFAULT_CONTRAST_FACTOR
        );
        
        long endTime = System.currentTimeMillis();
        
        return new VATResult(
            enhancedMatrix,
            basicResult.getReorderingIndices(),
            endTime - startTime
        );
    }
    
    /**
     * Compute iVAT using adaptive enhancement.
     * 
     * Automatically selects the best enhancement method based on data characteristics.
     * 
     * @param data input data matrix
     * @return enhanced VAT result
     */
    public static VATResult computeWithAdaptiveEnhancement(double[][] data) {
        // Analyze data characteristics to choose enhancement method
        var basicResult = VAT.compute(data);
        double clarity = basicResult.getClusterClarity();
        
        // For low clarity data, use path-based enhancement
        // For moderate clarity, use contrast enhancement
        if (clarity < 0.4) {
            return computeWithPathBasedEnhancement(data);
        } else {
            return computeWithContrastEnhancement(data);
        }
    }
    
    /**
     * Apply path-based enhancement to the ordered dissimilarity matrix.
     * 
     * This method considers paths through intermediate points to reveal
     * hidden clustering structure by reducing distances within clusters
     * while maintaining or increasing between-cluster distances.
     * 
     * @param matrix ordered dissimilarity matrix
     * @param pathFactor enhancement strength factor
     * @return enhanced matrix
     */
    private static double[][] applyPathBasedEnhancement(double[][] matrix, double pathFactor) {
        int n = matrix.length;
        var enhanced = new double[n][n];
        
        // Initialize with original matrix
        for (int i = 0; i < n; i++) {
            System.arraycopy(matrix[i], 0, enhanced[i], 0, n);
        }
        
        // Apply Floyd-Warshall-like algorithm to find shorter paths
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i != j) {
                        double pathDistance = (enhanced[i][k] + enhanced[k][j]) / pathFactor;
                        if (pathDistance < enhanced[i][j]) {
                            enhanced[i][j] = pathDistance;
                        }
                    }
                }
            }
        }
        
        // Ensure symmetry is maintained
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double avgDistance = (enhanced[i][j] + enhanced[j][i]) / 2.0;
                enhanced[i][j] = avgDistance;
                enhanced[j][i] = avgDistance;
            }
            enhanced[i][i] = 0.0; // Diagonal should remain zero
        }
        
        return enhanced;
    }
    
    /**
     * Apply contrast enhancement to the ordered dissimilarity matrix.
     * 
     * This method amplifies the contrast between small and large distances
     * to make cluster boundaries more visually apparent.
     * 
     * @param matrix ordered dissimilarity matrix  
     * @param contrastFactor enhancement strength factor
     * @return enhanced matrix
     */
    private static double[][] applyContrastEnhancement(double[][] matrix, double contrastFactor) {
        int n = matrix.length;
        var enhanced = new double[n][n];
        
        // Find statistics for contrast adjustment
        double[] allValues = new double[n * n];
        int idx = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                allValues[idx++] = matrix[i][j];
            }
        }
        Arrays.sort(allValues);
        
        // Use median as reference point
        double median = allValues[allValues.length / 2];
        double maxValue = allValues[allValues.length - 1];
        
        if (maxValue == 0) {
            // All distances are zero - return original matrix
            for (int i = 0; i < n; i++) {
                System.arraycopy(matrix[i], 0, enhanced[i], 0, n);
            }
            return enhanced;
        }
        
        // Apply contrast enhancement
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    enhanced[i][j] = 0.0;
                } else {
                    double normalized = matrix[i][j] / maxValue;
                    double enhanced_val;
                    
                    // Use median as threshold for better separation
                    double threshold = median / maxValue;
                    
                    if (normalized <= threshold) {
                        // Compress small distances (within-cluster)
                        enhanced_val = Math.pow(normalized / threshold, contrastFactor) * threshold * maxValue;
                    } else {
                        // Expand large distances (between-cluster)
                        double range = 1.0 - threshold;
                        double scaledDist = (normalized - threshold) / range;
                        enhanced_val = (threshold + Math.pow(scaledDist, 1.0/contrastFactor) * range) * maxValue;
                    }
                    
                    enhanced[i][j] = enhanced_val;
                }
            }
        }
        
        return enhanced;
    }
    
    /**
     * Apply smoothing to reduce noise in the dissimilarity matrix.
     * 
     * @param matrix input matrix
     * @param windowSize smoothing window size
     * @return smoothed matrix
     */
    private static double[][] applySmoothing(double[][] matrix, int windowSize) {
        int n = matrix.length;
        var smoothed = new double[n][n];
        int halfWindow = windowSize / 2;
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    smoothed[i][j] = 0.0;
                    continue;
                }
                
                double sum = 0.0;
                int count = 0;
                
                // Average over local neighborhood
                for (int di = -halfWindow; di <= halfWindow; di++) {
                    for (int dj = -halfWindow; dj <= halfWindow; dj++) {
                        int ni = i + di;
                        int nj = j + dj;
                        
                        if (ni >= 0 && ni < n && nj >= 0 && nj < n) {
                            sum += matrix[ni][nj];
                            count++;
                        }
                    }
                }
                
                smoothed[i][j] = count > 0 ? sum / count : matrix[i][j];
            }
        }
        
        return smoothed;
    }
    
    /**
     * Compute enhancement quality metric.
     * 
     * @param original original matrix
     * @param enhanced enhanced matrix
     * @return quality improvement score (higher is better)
     */
    private static double computeEnhancementQuality(double[][] original, double[][] enhanced) {
        int n = original.length;
        double originalClarity = computeMatrixClarity(original);
        double enhancedClarity = computeMatrixClarity(enhanced);
        
        return enhancedClarity - originalClarity;
    }
    
    /**
     * Compute matrix clarity score.
     * 
     * @param matrix dissimilarity matrix
     * @return clarity score
     */
    private static double computeMatrixClarity(double[][] matrix) {
        int n = matrix.length;
        if (n < 3) return 0.0;
        
        double diagonalSum = 0.0;
        double offDiagonalSum = 0.0;
        int diagonalCount = 0;
        int offDiagonalCount = 0;
        
        int bandWidth = Math.max(1, n / 10);
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                
                if (Math.abs(i - j) <= bandWidth) {
                    diagonalSum += matrix[i][j];
                    diagonalCount++;
                } else {
                    offDiagonalSum += matrix[i][j];
                    offDiagonalCount++;
                }
            }
        }
        
        if (diagonalCount == 0 || offDiagonalCount == 0) return 0.0;
        
        double avgDiagonal = diagonalSum / diagonalCount;
        double avgOffDiagonal = offDiagonalSum / offDiagonalCount;
        
        if (avgOffDiagonal == 0.0) return 0.0;
        
        double contrast = (avgOffDiagonal - avgDiagonal) / avgOffDiagonal;
        return Math.max(0.0, Math.min(1.0, contrast));
    }
}