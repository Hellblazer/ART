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
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

/**
 * Java implementation of VAT (Visual Assessment of Clustering Tendency).
 * 
 * VAT reorders a dissimilarity matrix to reveal clustering structure through
 * dark diagonal blocks. This implementation is designed to be significantly
 * faster than the Python pyclustertend equivalent by leveraging:
 * 
 * - Compiled Java performance
 * - Efficient distance computations
 * - Parallel processing for large datasets
 * - Optimized matrix operations
 * 
 * The algorithm:
 * 1. Compute pairwise distance matrix
 * 2. Find the pair of objects with maximum distance
 * 3. Iteratively select the next object with minimum distance to selected set
 * 4. Reorder the dissimilarity matrix according to selection order
 * 
 * @author Hal Hildebrand
 */
public class VAT {
    
    private static final int PARALLEL_THRESHOLD = 100; // Use parallel processing above this size
    
    /**
     * Compute VAT for the given dataset.
     * 
     * @param data input data matrix (samples x features)
     * @return VAT result with ordered dissimilarity matrix and analysis
     */
    public static VATResult compute(double[][] data) {
        validateInput(data);
        
        long startTime = System.currentTimeMillis();
        
        // Compute pairwise distance matrix
        double[][] distanceMatrix = computeDistanceMatrix(data);
        
        // Perform VAT reordering
        int[] reorderingIndices = computeVATOrdering(distanceMatrix);
        
        // Create ordered dissimilarity matrix
        double[][] orderedMatrix = reorderMatrix(distanceMatrix, reorderingIndices);
        
        long endTime = System.currentTimeMillis();
        
        return new VATResult(orderedMatrix, reorderingIndices, endTime - startTime);
    }
    
    /**
     * Compute VAT using parallel processing for improved performance on large datasets.
     * 
     * @param data input data matrix (samples x features)
     * @return VAT result with ordered dissimilarity matrix and analysis
     */
    public static VATResult computeParallel(double[][] data) {
        validateInput(data);
        
        if (data.length < PARALLEL_THRESHOLD) {
            return compute(data); // Use sequential version for small datasets
        }
        
        long startTime = System.currentTimeMillis();
        
        // Compute pairwise distance matrix in parallel
        double[][] distanceMatrix = computeDistanceMatrixParallel(data);
        
        // VAT ordering is inherently sequential, but we can optimize distance lookups
        int[] reorderingIndices = computeVATOrderingOptimized(distanceMatrix);
        
        // Create ordered dissimilarity matrix
        double[][] orderedMatrix = reorderMatrix(distanceMatrix, reorderingIndices);
        
        long endTime = System.currentTimeMillis();
        
        return new VATResult(orderedMatrix, reorderingIndices, endTime - startTime);
    }
    
    /**
     * Validate input data.
     * 
     * @param data input data to validate
     */
    private static void validateInput(double[][] data) {
        if (data == null) {
            throw new IllegalArgumentException("Data cannot be null");
        }
        if (data.length == 0) {
            throw new IllegalArgumentException("Data cannot be empty");
        }
        if (data.length == 1) {
            throw new IllegalArgumentException("VAT requires at least 2 data points");
        }
        
        int expectedDimensions = data[0] != null ? data[0].length : 0;
        if (expectedDimensions == 0) {
            throw new IllegalArgumentException("Data must have at least one dimension");
        }
        
        for (int i = 0; i < data.length; i++) {
            if (data[i] == null) {
                throw new IllegalArgumentException(String.format("Data row %d cannot be null", i));
            }
            if (data[i].length != expectedDimensions) {
                throw new IllegalArgumentException(
                    String.format("Inconsistent dimensions: row %d has %d dimensions, expected %d", 
                        i, data[i].length, expectedDimensions));
            }
        }
    }
    
    /**
     * Compute pairwise Euclidean distance matrix.
     * 
     * @param data input data matrix
     * @return symmetric distance matrix
     */
    private static double[][] computeDistanceMatrix(double[][] data) {
        int n = data.length;
        var distMatrix = new double[n][n];
        
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double dist = euclideanDistance(data[i], data[j]);
                distMatrix[i][j] = dist;
                distMatrix[j][i] = dist; // Symmetric
            }
            // Diagonal is already 0.0
        }
        
        return distMatrix;
    }
    
    /**
     * Compute pairwise distance matrix using parallel processing.
     * 
     * @param data input data matrix
     * @return symmetric distance matrix
     */
    private static double[][] computeDistanceMatrixParallel(double[][] data) {
        int n = data.length;
        var distMatrix = new double[n][n];
        
        // Parallel computation of upper triangle
        IntStream.range(0, n).parallel().forEach(i -> {
            for (int j = i + 1; j < n; j++) {
                double dist = euclideanDistance(data[i], data[j]);
                distMatrix[i][j] = dist;
                distMatrix[j][i] = dist; // Symmetric
            }
        });
        
        return distMatrix;
    }
    
    /**
     * Compute Euclidean distance between two points.
     * 
     * @param point1 first point
     * @param point2 second point
     * @return Euclidean distance
     */
    private static double euclideanDistance(double[] point1, double[] point2) {
        double sumSquares = 0.0;
        for (int i = 0; i < point1.length; i++) {
            double diff = point1[i] - point2[i];
            sumSquares += diff * diff;
        }
        return Math.sqrt(sumSquares);
    }
    
    /**
     * Compute VAT ordering using the classic VAT algorithm.
     * 
     * @param distanceMatrix pairwise distance matrix
     * @return reordering indices
     */
    private static int[] computeVATOrdering(double[][] distanceMatrix) {
        int n = distanceMatrix.length;
        var reordering = new int[n];
        var selected = new boolean[n];
        
        // Step 1: Find the pair with maximum distance
        double maxDist = 0.0;
        int maxI = 0, maxJ = 1;
        
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (distanceMatrix[i][j] > maxDist) {
                    maxDist = distanceMatrix[i][j];
                    maxI = i;
                    maxJ = j;
                }
            }
        }
        
        // Start with the maximum distance pair
        reordering[0] = maxI;
        reordering[1] = maxJ;
        selected[maxI] = true;
        selected[maxJ] = true;
        
        // Step 2: Iteratively add remaining points
        for (int k = 2; k < n; k++) {
            int nextPoint = -1;
            double minDistance = Double.MAX_VALUE;
            
            // Find unselected point with minimum distance to any selected point
            for (int i = 0; i < n; i++) {
                if (selected[i]) continue;
                
                double distToSelected = Double.MAX_VALUE;
                for (int j = 0; j < k; j++) {
                    int selectedPoint = reordering[j];
                    distToSelected = Math.min(distToSelected, distanceMatrix[i][selectedPoint]);
                }
                
                if (distToSelected < minDistance) {
                    minDistance = distToSelected;
                    nextPoint = i;
                }
            }
            
            reordering[k] = nextPoint;
            selected[nextPoint] = true;
        }
        
        return reordering;
    }
    
    /**
     * Compute VAT ordering with optimized distance lookups for large datasets.
     * 
     * @param distanceMatrix pairwise distance matrix
     * @return reordering indices
     */
    private static int[] computeVATOrderingOptimized(double[][] distanceMatrix) {
        int n = distanceMatrix.length;
        var reordering = new int[n];
        var selected = new boolean[n];
        var minDistToSelected = new double[n];
        
        // Initialize with maximum values
        Arrays.fill(minDistToSelected, Double.MAX_VALUE);
        
        // Step 1: Find the pair with maximum distance
        double maxDist = 0.0;
        int maxI = 0, maxJ = 1;
        
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (distanceMatrix[i][j] > maxDist) {
                    maxDist = distanceMatrix[i][j];
                    maxI = i;
                    maxJ = j;
                }
            }
        }
        
        // Start with the maximum distance pair
        reordering[0] = maxI;
        reordering[1] = maxJ;
        selected[maxI] = true;
        selected[maxJ] = true;
        
        // Update minimum distances to selected set
        for (int i = 0; i < n; i++) {
            if (!selected[i]) {
                minDistToSelected[i] = Math.min(distanceMatrix[i][maxI], distanceMatrix[i][maxJ]);
            }
        }
        
        // Step 2: Iteratively add remaining points with optimized distance tracking
        for (int k = 2; k < n; k++) {
            int nextPoint = -1;
            double minDistance = Double.MAX_VALUE;
            
            // Find unselected point with minimum distance to selected set
            for (int i = 0; i < n; i++) {
                if (!selected[i] && minDistToSelected[i] < minDistance) {
                    minDistance = minDistToSelected[i];
                    nextPoint = i;
                }
            }
            
            reordering[k] = nextPoint;
            selected[nextPoint] = true;
            
            // Update minimum distances for remaining unselected points
            for (int i = 0; i < n; i++) {
                if (!selected[i]) {
                    minDistToSelected[i] = Math.min(minDistToSelected[i], distanceMatrix[i][nextPoint]);
                }
            }
        }
        
        return reordering;
    }
    
    /**
     * Reorder the distance matrix according to VAT ordering.
     * 
     * @param originalMatrix original distance matrix
     * @param reordering reordering indices
     * @return reordered matrix
     */
    private static double[][] reorderMatrix(double[][] originalMatrix, int[] reordering) {
        int n = originalMatrix.length;
        var reorderedMatrix = new double[n][n];
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                reorderedMatrix[i][j] = originalMatrix[reordering[i]][reordering[j]];
            }
        }
        
        return reorderedMatrix;
    }
}