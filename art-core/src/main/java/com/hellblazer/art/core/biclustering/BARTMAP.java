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
package com.hellblazer.art.core.biclustering;

import com.hellblazer.art.core.BaseART;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.parameters.FuzzyParameters;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * BARTMAP for biclustering analysis.
 * 
 * This class implements BARTMAP (Biclustering Adaptive Resonance Theory MAP) as described in:
 * Xu, R., & Wunsch II, D. C. (2011). BARTMAP: A viable structure for biclustering.
 * Neural Networks, 24, 709–716. doi:10.1016/j.neunet.2011.03.020.
 * 
 * BARTMAP performs simultaneous clustering of both rows (samples) and columns (features)
 * to identify biclusters - subsets of rows that exhibit similar patterns across subsets
 * of columns. It uses two ART modules: one for clustering rows and another for clustering
 * columns, with a Pearson correlation criterion to ensure coherence within biclusters.
 * 
 * Key features:
 * - Simultaneous row and column clustering
 * - Pearson correlation-based match criterion
 * - Identifies coherent biclusters in data matrices
 * - Useful for gene expression analysis, recommendation systems, etc.
 * 
 * @author Hal Hildebrand
 */
public class BARTMAP {
    
    /**
     * Represents a single bicluster with row and column membership indicators.
     */
    public record Bicluster(boolean[] rows, boolean[] columns) {
        public Bicluster {
            Objects.requireNonNull(rows, "Rows cannot be null");
            Objects.requireNonNull(columns, "Columns cannot be null");
        }
    }
    
    private final BaseART moduleA;  // For clustering rows (samples)
    private final BaseART moduleB;  // For clustering columns (features)
    private final double eta;       // Minimum Pearson correlation threshold
    
    private double[][] data;        // The data matrix being clustered
    private int[] rowLabels;        // Cluster labels for rows
    private int[] columnLabels;     // Cluster labels for columns
    
    /**
     * Create a new BARTMAP biclustering model.
     * 
     * @param moduleA the ART module for clustering rows (samples)
     * @param moduleB the ART module for clustering columns (features)
     * @param eta the minimum Pearson correlation required for row clustering (-1.0 to 1.0)
     */
    public BARTMAP(BaseART moduleA, BaseART moduleB, double eta) {
        Objects.requireNonNull(moduleA, "Module A cannot be null");
        Objects.requireNonNull(moduleB, "Module B cannot be null");
        
        if (eta < -1.0 || eta > 1.0) {
            throw new IllegalArgumentException(
                "Eta must be between -1.0 and 1.0 (Pearson correlation range)"
            );
        }
        
        this.moduleA = moduleA;
        this.moduleB = moduleB;
        this.eta = eta;
    }
    
    /**
     * Fit the model to the data.
     * 
     * @param X the data matrix to bicluster
     * @return this instance for method chaining
     */
    public BARTMAP fit(double[][] X) {
        return fit(X, 1);
    }
    
    /**
     * Fit the model to the data with multiple iterations.
     * 
     * @param X the data matrix to bicluster
     * @param maxIter the number of iterations to fit the model
     * @return this instance for method chaining
     */
    public BARTMAP fit(double[][] X, int maxIter) {
        validateData(X);
        
        this.data = X;
        int nRows = X.length;
        int nCols = X[0].length;
        
        // Prepare data for column clustering (transpose)
        double[][] X_T = transpose(X);
        
        // First, cluster the columns independently
        this.columnLabels = new int[nCols];
        for (int iter = 0; iter < maxIter; iter++) {
            for (int j = 0; j < nCols; j++) {
                var pattern = Pattern.of(X_T[j]);
                var params = new FuzzyParameters(0.5, 0.01, 1.0);
                
                var result = moduleB.stepFit(pattern, params);
                if (result instanceof com.hellblazer.art.core.results.ActivationResult.Success success) {
                    columnLabels[j] = success.categoryIndex();
                } else {
                    columnLabels[j] = moduleB.getCategoryCount() - 1;
                }
            }
        }
        
        // Then, cluster the rows considering column clusters and correlation
        this.rowLabels = new int[nRows];
        for (int iter = 0; iter < maxIter; iter++) {
            for (int i = 0; i < nRows; i++) {
                var pattern = Pattern.of(X[i]);
                var params = new FuzzyParameters(0.5, 0.01, 1.0);
                
                // Custom match reset function that considers correlation
                final int rowIdx = i;
                var result = moduleA.stepFit(pattern, params, 
                    (p, w, cat, par, cache) -> matchResetFunc(rowIdx, cat),
                    com.hellblazer.art.core.MatchTrackingMode.MT_PLUS,
                    1e-10);
                
                if (result instanceof com.hellblazer.art.core.results.ActivationResult.Success success) {
                    rowLabels[i] = success.categoryIndex();
                } else {
                    rowLabels[i] = moduleA.getCategoryCount() - 1;
                }
            }
        }
        
        return this;
    }
    
    /**
     * Custom match reset function that enforces correlation criterion.
     * 
     * @param rowIndex the index of the row being clustered
     * @param clusterA the candidate row cluster
     * @return true if the match is permitted, false otherwise
     */
    private boolean matchResetFunc(int rowIndex, int clusterA) {
        // Check if the row meets the correlation criterion with at least one column cluster
        for (int clusterB = 0; clusterB < moduleB.getCategoryCount(); clusterB++) {
            if (matchCriterionBin(rowIndex, clusterB)) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * Check if a row meets the correlation criterion for a column cluster.
     * 
     * @param rowIndex the row index
     * @param columnCluster the column cluster to check
     * @return true if the average correlation exceeds eta
     */
    private boolean matchCriterionBin(int rowIndex, int columnCluster) {
        double avgCorr = averagePearsonCorr(rowIndex, columnCluster);
        return avgCorr >= eta;
    }
    
    /**
     * Calculate average Pearson correlation for a row across a column cluster.
     * 
     * @param rowIndex the row index
     * @param columnCluster the column cluster
     * @return the average Pearson correlation
     */
    private double averagePearsonCorr(int rowIndex, int columnCluster) {
        // Get indices of columns in this cluster
        var columnIndices = new ArrayList<Integer>();
        for (int j = 0; j < columnLabels.length; j++) {
            if (columnLabels[j] == columnCluster) {
                columnIndices.add(j);
            }
        }
        
        if (columnIndices.isEmpty()) {
            return 0.0;
        }
        
        // Get rows already assigned to clusters
        var assignedRows = new ArrayList<Integer>();
        for (int i = 0; i < rowIndex; i++) {
            if (rowLabels[i] >= 0) {
                assignedRows.add(i);
            }
        }
        
        if (assignedRows.isEmpty()) {
            return 1.0;  // First row always matches
        }
        
        // Calculate average correlation with assigned rows
        double sumCorr = 0.0;
        int count = 0;
        
        for (int otherRow : assignedRows) {
            double corr = pearsonCorrelation(
                getSubvector(data[rowIndex], columnIndices),
                getSubvector(data[otherRow], columnIndices)
            );
            sumCorr += corr;
            count++;
        }
        
        return count > 0 ? sumCorr / count : 0.0;
    }
    
    /**
     * Calculate Pearson correlation between two vectors.
     * 
     * @param a first vector
     * @param b second vector
     * @return Pearson correlation coefficient
     */
    private double pearsonCorrelation(double[] a, double[] b) {
        if (a.length != b.length || a.length == 0) {
            return 0.0;
        }
        
        double meanA = mean(a);
        double meanB = mean(b);
        
        double numerator = 0.0;
        double denomA = 0.0;
        double denomB = 0.0;
        
        for (int i = 0; i < a.length; i++) {
            double diffA = a[i] - meanA;
            double diffB = b[i] - meanB;
            numerator += diffA * diffB;
            denomA += diffA * diffA;
            denomB += diffB * diffB;
        }
        
        if (denomA == 0.0 || denomB == 0.0) {
            return 0.0;
        }
        
        return numerator / (Math.sqrt(denomA) * Math.sqrt(denomB));
    }
    
    /**
     * Get a subvector based on indices.
     * 
     * @param vector the original vector
     * @param indices the indices to extract
     * @return the subvector
     */
    private double[] getSubvector(double[] vector, List<Integer> indices) {
        var result = new double[indices.size()];
        for (int i = 0; i < indices.size(); i++) {
            result[i] = vector[indices.get(i)];
        }
        return result;
    }
    
    /**
     * Calculate the mean of a vector.
     * 
     * @param vector the input vector
     * @return the mean value
     */
    private double mean(double[] vector) {
        if (vector.length == 0) {
            return 0.0;
        }
        double sum = 0.0;
        for (double v : vector) {
            sum += v;
        }
        return sum / vector.length;
    }
    
    /**
     * Transpose a matrix.
     * 
     * @param matrix the input matrix
     * @return the transposed matrix
     */
    private double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        var result = new double[cols][rows];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        
        return result;
    }
    
    /**
     * Get the biclusters identified by the model.
     * 
     * @return array of biclusters
     */
    public Bicluster[] getBiclusters() {
        if (rowLabels == null || columnLabels == null) {
            return new Bicluster[0];
        }
        
        int nRowClusters = moduleA.getCategoryCount();
        int nColClusters = moduleB.getCategoryCount();
        
        var biclusters = new ArrayList<Bicluster>();
        
        // Create a bicluster for each combination of row and column clusters
        for (int rowCluster = 0; rowCluster < nRowClusters; rowCluster++) {
            for (int colCluster = 0; colCluster < nColClusters; colCluster++) {
                var rows = new boolean[rowLabels.length];
                var cols = new boolean[columnLabels.length];
                
                // Mark rows in this row cluster
                for (int i = 0; i < rowLabels.length; i++) {
                    rows[i] = (rowLabels[i] == rowCluster);
                }
                
                // Mark columns in this column cluster
                for (int j = 0; j < columnLabels.length; j++) {
                    cols[j] = (columnLabels[j] == colCluster);
                }
                
                biclusters.add(new Bicluster(rows, cols));
            }
        }
        
        return biclusters.toArray(new Bicluster[0]);
    }
    
    /**
     * Get the cluster labels for rows.
     * 
     * @return array of row cluster labels
     */
    public int[] getRowLabels() {
        return rowLabels == null ? new int[0] : rowLabels.clone();
    }
    
    /**
     * Get the cluster labels for columns.
     * 
     * @return array of column cluster labels
     */
    public int[] getColumnLabels() {
        return columnLabels == null ? new int[0] : columnLabels.clone();
    }
    
    /**
     * Get the number of row clusters.
     * 
     * @return number of row clusters
     */
    public int getRowClusterCount() {
        return moduleA.getCategoryCount();
    }
    
    /**
     * Get the number of column clusters.
     * 
     * @return number of column clusters
     */
    public int getColumnClusterCount() {
        return moduleB.getCategoryCount();
    }
    
    /**
     * Get the eta parameter (minimum correlation threshold).
     * 
     * @return eta value
     */
    public double getEta() {
        return eta;
    }
    
    /**
     * Get the module A (row clustering module).
     * 
     * @return module A
     */
    public BaseART getModuleA() {
        return moduleA;
    }
    
    /**
     * Get the module B (column clustering module).
     * 
     * @return module B
     */
    public BaseART getModuleB() {
        return moduleB;
    }
    
    /**
     * Validate input data.
     * 
     * @param X the data matrix
     */
    private void validateData(double[][] X) {
        Objects.requireNonNull(X, "Data cannot be null");
        
        if (X.length == 0) {
            throw new IllegalArgumentException("Data cannot be empty");
        }
        
        if (X[0].length == 0) {
            throw new IllegalArgumentException("Data must have at least one column");
        }
        
        // Check for consistent dimensions
        int cols = X[0].length;
        for (int i = 1; i < X.length; i++) {
            if (X[i].length != cols) {
                throw new IllegalArgumentException(
                    "All rows must have the same number of columns"
                );
            }
        }
    }
}