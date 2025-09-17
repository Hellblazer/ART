package com.hellblazer.art.performance.algorithms;

import java.util.Objects;

/**
 * Parameters for VectorizedBARTMAP (Biclustering Adaptive Resonance Theory MAP).
 * 
 * BARTMAP performs simultaneous clustering of rows and columns to identify
 * biclusters - subsets of rows that exhibit similar patterns across subsets
 * of columns. Uses Pearson correlation to ensure coherence within biclusters.
 */
public class VectorizedBARTMAPParameters {
    
    // Core parameters
    private final double rowVigilance;
    private final double columnVigilance;
    private final double eta;  // Minimum Pearson correlation threshold
    private final double alpha;
    private final double learningRate;
    
    // Biclustering parameters
    private final int minBiclusterRows;
    private final int minBiclusterCols;
    private final double coherenceThreshold;
    private final boolean allowOverlapping;
    
    // Iteration control
    private final int maxIterations;
    private final double convergenceThreshold;
    
    // Performance parameters
    private final boolean useParallelProcessing;
    private final int batchSize;
    
    // Base parameters for vectorization
    private final VectorizedParameters baseParameters;
    
    public VectorizedBARTMAPParameters(
            double rowVigilance,
            double columnVigilance,
            double eta,
            double alpha,
            double learningRate,
            int minBiclusterRows,
            int minBiclusterCols,
            double coherenceThreshold,
            boolean allowOverlapping,
            int maxIterations,
            double convergenceThreshold,
            boolean useParallelProcessing,
            int batchSize,
            VectorizedParameters baseParameters) {
        
        validateParameters(rowVigilance, columnVigilance, eta, alpha, learningRate,
                         minBiclusterRows, minBiclusterCols, coherenceThreshold,
                         maxIterations, convergenceThreshold, batchSize);
        
        this.rowVigilance = rowVigilance;
        this.columnVigilance = columnVigilance;
        this.eta = eta;
        this.alpha = alpha;
        this.learningRate = learningRate;
        this.minBiclusterRows = minBiclusterRows;
        this.minBiclusterCols = minBiclusterCols;
        this.coherenceThreshold = coherenceThreshold;
        this.allowOverlapping = allowOverlapping;
        this.maxIterations = maxIterations;
        this.convergenceThreshold = convergenceThreshold;
        this.useParallelProcessing = useParallelProcessing;
        this.batchSize = batchSize;
        this.baseParameters = Objects.requireNonNull(baseParameters, "Base parameters cannot be null");
    }
    
    /**
     * Create default BARTMAP parameters.
     */
    public VectorizedBARTMAPParameters() {
        this(0.5, 0.5, 0.7, 0.001, 0.5, 2, 2, 0.8, false,
             100, 0.001, true, 32, VectorizedParameters.createDefault());
    }
    
    /**
     * Create BARTMAP parameters with specified correlation threshold.
     */
    public static VectorizedBARTMAPParameters withCorrelation(double eta) {
        return new VectorizedBARTMAPParameters(
            0.5, 0.5, eta, 0.001, 0.5, 2, 2, 0.8, false,
            100, 0.001, true, 32, VectorizedParameters.createDefault()
        );
    }
    
    private static void validateParameters(double rowVigilance, double columnVigilance,
                                          double eta, double alpha, double learningRate,
                                          int minBiclusterRows, int minBiclusterCols,
                                          double coherenceThreshold, int maxIterations,
                                          double convergenceThreshold, int batchSize) {
        if (rowVigilance < 0.0 || rowVigilance > 1.0) {
            throw new IllegalArgumentException("Row vigilance must be in [0, 1], got: " + rowVigilance);
        }
        if (columnVigilance < 0.0 || columnVigilance > 1.0) {
            throw new IllegalArgumentException("Column vigilance must be in [0, 1], got: " + columnVigilance);
        }
        if (eta < -1.0 || eta > 1.0) {
            throw new IllegalArgumentException("Eta (correlation threshold) must be in [-1, 1], got: " + eta);
        }
        if (alpha < 0.0) {
            throw new IllegalArgumentException("Alpha must be non-negative, got: " + alpha);
        }
        if (learningRate < 0.0 || learningRate > 1.0) {
            throw new IllegalArgumentException("Learning rate must be in [0, 1], got: " + learningRate);
        }
        if (minBiclusterRows < 1) {
            throw new IllegalArgumentException("Min bicluster rows must be >= 1, got: " + minBiclusterRows);
        }
        if (minBiclusterCols < 1) {
            throw new IllegalArgumentException("Min bicluster columns must be >= 1, got: " + minBiclusterCols);
        }
        if (coherenceThreshold < 0.0 || coherenceThreshold > 1.0) {
            throw new IllegalArgumentException("Coherence threshold must be in [0, 1], got: " + coherenceThreshold);
        }
        if (maxIterations < 1) {
            throw new IllegalArgumentException("Max iterations must be >= 1, got: " + maxIterations);
        }
        if (convergenceThreshold < 0.0 || convergenceThreshold > 1.0) {
            throw new IllegalArgumentException("Convergence threshold must be in [0, 1], got: " + convergenceThreshold);
        }
        if (batchSize < 1) {
            throw new IllegalArgumentException("Batch size must be >= 1, got: " + batchSize);
        }
    }
    
    // Getters
    
    public double getRowVigilance() {
        return rowVigilance;
    }
    
    public double getColumnVigilance() {
        return columnVigilance;
    }
    
    public double getEta() {
        return eta;
    }
    
    public double getAlpha() {
        return alpha;
    }
    
    public double getLearningRate() {
        return learningRate;
    }
    
    public int getMinBiclusterRows() {
        return minBiclusterRows;
    }
    
    public int getMinBiclusterCols() {
        return minBiclusterCols;
    }
    
    public double getCoherenceThreshold() {
        return coherenceThreshold;
    }
    
    public boolean isAllowOverlapping() {
        return allowOverlapping;
    }
    
    public int getMaxIterations() {
        return maxIterations;
    }
    
    public double getConvergenceThreshold() {
        return convergenceThreshold;
    }
    
    public boolean isUseParallelProcessing() {
        return useParallelProcessing;
    }
    
    public int getBatchSize() {
        return batchSize;
    }
    
    public VectorizedParameters getBaseParameters() {
        return baseParameters;
    }
    
    /**
     * Calculate Pearson correlation between two vectors.
     */
    public double calculatePearsonCorrelation(double[] x, double[] y) {
        if (x.length != y.length) {
            throw new IllegalArgumentException("Vectors must have same length");
        }
        
        int n = x.length;
        double sumX = 0.0, sumY = 0.0, sumXY = 0.0;
        double sumX2 = 0.0, sumY2 = 0.0;
        
        for (int i = 0; i < n; i++) {
            sumX += x[i];
            sumY += y[i];
            sumXY += x[i] * y[i];
            sumX2 += x[i] * x[i];
            sumY2 += y[i] * y[i];
        }
        
        double numerator = n * sumXY - sumX * sumY;
        double denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
        
        if (denominator == 0) {
            return 0.0;
        }
        
        return numerator / denominator;
    }
    
    /**
     * Check if a bicluster meets minimum size requirements.
     */
    public boolean isValidBicluster(int rowCount, int colCount) {
        return rowCount >= minBiclusterRows && colCount >= minBiclusterCols;
    }
    
    @Override
    public String toString() {
        return String.format("VectorizedBARTMAPParameters{rowVig=%.3f, colVig=%.3f, eta=%.3f, " +
                           "alpha=%.4f, lr=%.3f, minRows=%d, minCols=%d, coherence=%.3f}",
                           rowVigilance, columnVigilance, eta, alpha, learningRate,
                           minBiclusterRows, minBiclusterCols, coherenceThreshold);
    }
}