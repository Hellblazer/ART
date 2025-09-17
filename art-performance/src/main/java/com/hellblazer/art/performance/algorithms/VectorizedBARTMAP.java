package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.algorithms.FuzzyART;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.biclustering.BARTMAP;
import com.hellblazer.art.core.parameters.FuzzyParameters;
import com.hellblazer.art.performance.VectorizedARTAlgorithm;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Vectorized BARTMAP (Biclustering Adaptive Resonance Theory MAP) implementation.
 * Performs simultaneous row and column clustering for biclustering analysis.
 */
public class VectorizedBARTMAP implements VectorizedARTAlgorithm<VectorizedBARTMAP.PerformanceMetrics, VectorizedBARTMAPParameters> {

    private final BARTMAP bartmap;
    private final VectorizedBARTMAPParameters defaultParams;
    private final ReentrantReadWriteLock lock;
    
    // Cached data matrix for biclustering
    private double[][] dataMatrix;
    private int nRows;
    private int nCols;
    
    // Performance tracking
    private final AtomicLong simdOperations;
    private final AtomicLong totalOperations;
    private final AtomicLong correlationCalculations;
    private final AtomicLong biclustersFound;
    private final AtomicLong rowClusteringOps;
    private final AtomicLong columnClusteringOps;
    private long startTime;
    
    /**
     * Performance metrics for vectorized BARTMAP.
     */
    public record PerformanceMetrics(
        long simdOperations,
        long totalOperations,
        long correlationCalculations,
        long biclustersFound,
        long rowClusteringOps,
        long columnClusteringOps,
        long elapsedTimeNanos,
        double throughputOpsPerSec,
        double simdUtilization,
        int numBiclusters,
        double avgBiclusterCoherence
    ) {
        public static PerformanceMetrics empty() {
            return new PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0, 0.0);
        }
    }
    
    /**
     * Create a new VectorizedBARTMAP with default parameters.
     */
    public VectorizedBARTMAP() {
        this(new VectorizedBARTMAPParameters());
    }
    
    /**
     * Create a new VectorizedBARTMAP with specified parameters.
     */
    public VectorizedBARTMAP(VectorizedBARTMAPParameters parameters) {
        // Create ART modules for row and column clustering
        var rowModule = new FuzzyART();
        var colModule = new FuzzyART();
        
        this.bartmap = new BARTMAP(rowModule, colModule, parameters.getEta());
        this.defaultParams = parameters;
        this.lock = new ReentrantReadWriteLock();
        
        // Initialize performance counters
        this.simdOperations = new AtomicLong();
        this.totalOperations = new AtomicLong();
        this.correlationCalculations = new AtomicLong();
        this.biclustersFound = new AtomicLong();
        this.rowClusteringOps = new AtomicLong();
        this.columnClusteringOps = new AtomicLong();
        this.startTime = System.nanoTime();
    }
    
    /**
     * Fit the BARTMAP model to a data matrix.
     * 
     * @param X the data matrix (rows are samples, columns are features)
     * @return the fitted BARTMAP model
     */
    public BARTMAP fitMatrix(double[][] X) {
        return fitMatrix(X, defaultParams);
    }
    
    /**
     * Fit the BARTMAP model to a data matrix with specific parameters.
     * 
     * @param X the data matrix
     * @param params the parameters to use
     * @return the fitted BARTMAP model
     */
    public BARTMAP fitMatrix(double[][] X, VectorizedBARTMAPParameters params) {
        lock.writeLock().lock();
        try {
            totalOperations.incrementAndGet();
            
            // Store the data matrix
            this.dataMatrix = X;
            this.nRows = X.length;
            this.nCols = X[0].length;
            
            // Track SIMD operations for matrix processing
            long simdOps = estimateMatrixSimdOperations(nRows, nCols, params.getMaxIterations());
            simdOperations.addAndGet(simdOps);
            
            // Perform biclustering with multiple iterations
            bartmap.fit(X, params.getMaxIterations());
            
            // Track clustering operations
            rowClusteringOps.addAndGet(nRows * params.getMaxIterations());
            columnClusteringOps.addAndGet(nCols * params.getMaxIterations());
            
            // Extract and count biclusters
            var biclusters = bartmap.getBiclusters();
            var filteredBiclusters = filterBiclusters(biclusters, params.getMinBiclusterRows(), params.getMinBiclusterCols());
            biclustersFound.addAndGet(filteredBiclusters.size());
            
            return bartmap;
            
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    public Object learn(double[] input) {
        // For single pattern learning, treat as a single row
        double[][] matrix = {input};
        fitMatrix(matrix, defaultParams);
        return bartmap.getRowLabels()[0];
    }
    
    @Override
    public Object learn(Pattern input, VectorizedBARTMAPParameters params) {
        // Convert pattern to matrix row and fit
        double[] data = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            data[i] = input.get(i);
        }
        double[][] matrix = {data};
        fitMatrix(matrix, params);
        return bartmap.getRowLabels()[0];
    }
    
    public int predict(double[] input) {
        var result = predict(Pattern.of(input), defaultParams);
        return result instanceof Integer ? (Integer) result : -1;
    }
    
    @Override
    public Object predict(Pattern input, VectorizedBARTMAPParameters params) {
        lock.readLock().lock();
        try {
            totalOperations.incrementAndGet();
            
            // For prediction, find the best matching row cluster
            if (dataMatrix == null || nRows == 0) {
                return -1;  // Model not fitted yet
            }
            
            // Convert pattern to array
            double[] data = new double[input.dimension()];
            for (int i = 0; i < input.dimension(); i++) {
                data[i] = input.get(i);
            }
            
            // Find best matching bicluster by correlation
            int bestCluster = -1;
            double bestCorrelation = -1.0;
            
            var rowLabels = bartmap.getRowLabels();
            for (int i = 0; i < nRows; i++) {
                double correlation = params.calculatePearsonCorrelation(dataMatrix[i], data);
                correlationCalculations.incrementAndGet();
                
                if (correlation > bestCorrelation && correlation >= params.getEta()) {
                    bestCorrelation = correlation;
                    bestCluster = rowLabels[i];
                }
            }
            
            // Track SIMD operations
            simdOperations.addAndGet(input.dimension() * nRows);
            
            return bestCluster;
            
        } finally {
            lock.readLock().unlock();
        }
    }
    
    @Override
    public int getCategoryCount() {
        lock.readLock().lock();
        try {
            // Return the number of unique biclusters
            var biclusters = bartmap.getBiclusters();
            var filteredBiclusters = filterBiclusters(biclusters, 
                defaultParams.getMinBiclusterRows(),
                defaultParams.getMinBiclusterCols());
            return filteredBiclusters.size();
        } finally {
            lock.readLock().unlock();
        }
    }
    
    @Override
    public VectorizedBARTMAPParameters getParameters() {
        return defaultParams;
    }
    
    @Override
    public PerformanceMetrics getPerformanceStats() {
        lock.readLock().lock();
        try {
            long elapsed = System.nanoTime() - startTime;
            long totalOps = totalOperations.get();
            long simdOps = simdOperations.get();
            
            double throughput = totalOps > 0 ? 
                (double) totalOps / (elapsed / 1_000_000_000.0) : 0.0;
            double simdUtil = totalOps > 0 ? 
                (double) simdOps / (totalOps * 100) : 0.0;
            
            // Get bicluster statistics
            var biclusters = bartmap.getBiclusters();
            var filteredBiclusters = filterBiclusters(biclusters,
                defaultParams.getMinBiclusterRows(),
                defaultParams.getMinBiclusterCols());
            int numBiclusters = filteredBiclusters.size();
            
            // Calculate average bicluster coherence
            double avgCoherence = calculateAverageCoherence(filteredBiclusters);
            
            return new PerformanceMetrics(
                simdOps,
                totalOps,
                correlationCalculations.get(),
                biclustersFound.get(),
                rowClusteringOps.get(),
                columnClusteringOps.get(),
                elapsed,
                throughput,
                simdUtil,
                numBiclusters,
                avgCoherence
            );
        } finally {
            lock.readLock().unlock();
        }
    }
    
    @Override
    public void resetPerformanceTracking() {
        simdOperations.set(0);
        totalOperations.set(0);
        correlationCalculations.set(0);
        biclustersFound.set(0);
        rowClusteringOps.set(0);
        columnClusteringOps.set(0);
        startTime = System.nanoTime();
    }
    
    /**
     * Estimate SIMD operations for matrix processing.
     */
    private long estimateMatrixSimdOperations(int rows, int cols, int iterations) {
        // Row clustering operations
        long rowOps = (long) rows * cols * iterations * 3;
        
        // Column clustering operations  
        long colOps = (long) cols * rows * iterations * 3;
        
        // Correlation calculations
        long corrOps = (long) rows * cols * iterations * 2;
        
        return rowOps + colOps + corrOps;
    }
    
    /**
     * Filter biclusters by minimum size requirements.
     */
    private List<BARTMAP.Bicluster> filterBiclusters(BARTMAP.Bicluster[] biclusters, int minRows, int minCols) {
        List<BARTMAP.Bicluster> filtered = new ArrayList<>();
        for (var bicluster : biclusters) {
            int rowCount = 0;
            int colCount = 0;
            
            // Count rows
            for (boolean row : bicluster.rows()) {
                if (row) rowCount++;
            }
            
            // Count columns
            for (boolean col : bicluster.columns()) {
                if (col) colCount++;
            }
            
            // Add if meets minimum requirements
            if (rowCount >= minRows && colCount >= minCols) {
                filtered.add(bicluster);
            }
        }
        return filtered;
    }
    
    /**
     * Calculate average coherence of biclusters.
     */
    private double calculateAverageCoherence(List<BARTMAP.Bicluster> biclusters) {
        if (biclusters.isEmpty() || dataMatrix == null) {
            return 0.0;
        }
        
        double totalCoherence = 0.0;
        int validBiclusters = 0;
        
        for (var bicluster : biclusters) {
            double coherence = calculateBiclusterCoherence(bicluster);
            if (!Double.isNaN(coherence)) {
                totalCoherence += coherence;
                validBiclusters++;
            }
        }
        
        return validBiclusters > 0 ? totalCoherence / validBiclusters : 0.0;
    }
    
    /**
     * Calculate coherence of a single bicluster using Pearson correlation.
     */
    private double calculateBiclusterCoherence(BARTMAP.Bicluster bicluster) {
        var rows = bicluster.rows();
        var cols = bicluster.columns();
        
        // Get row indices
        List<Integer> rowIndices = new ArrayList<>();
        for (int i = 0; i < rows.length; i++) {
            if (rows[i]) {
                rowIndices.add(i);
            }
        }
        
        // Get column indices
        List<Integer> colIndices = new ArrayList<>();
        for (int j = 0; j < cols.length; j++) {
            if (cols[j]) {
                colIndices.add(j);
            }
        }
        
        if (rowIndices.size() < 2 || colIndices.isEmpty()) {
            return 0.0;
        }
        
        // Calculate average pairwise correlation between rows
        double totalCorrelation = 0.0;
        int pairCount = 0;
        
        for (int i = 0; i < rowIndices.size() - 1; i++) {
            for (int j = i + 1; j < rowIndices.size(); j++) {
                double[] row1 = extractSubvector(dataMatrix[rowIndices.get(i)], colIndices);
                double[] row2 = extractSubvector(dataMatrix[rowIndices.get(j)], colIndices);
                
                double corr = defaultParams.calculatePearsonCorrelation(row1, row2);
                correlationCalculations.incrementAndGet();
                
                if (!Double.isNaN(corr)) {
                    totalCorrelation += Math.abs(corr);
                    pairCount++;
                }
            }
        }
        
        return pairCount > 0 ? totalCorrelation / pairCount : 0.0;
    }
    
    /**
     * Extract subvector from array given column indices.
     */
    private double[] extractSubvector(double[] vector, List<Integer> indices) {
        double[] subvector = new double[indices.size()];
        for (int i = 0; i < indices.size(); i++) {
            subvector[i] = vector[indices.get(i)];
        }
        return subvector;
    }
    
    /**
     * Get the underlying BARTMAP model.
     */
    public BARTMAP getBARTMAP() {
        return bartmap;
    }
    
    /**
     * Get discovered biclusters.
     */
    public List<BARTMAP.Bicluster> getBiclusters() {
        lock.readLock().lock();
        try {
            var biclusters = bartmap.getBiclusters();
            return filterBiclusters(biclusters,
                defaultParams.getMinBiclusterRows(),
                defaultParams.getMinBiclusterCols());
        } finally {
            lock.readLock().unlock();
        }
    }
    
    @Override
    public void close() {
        // No resources to release
    }
    
    @Override
    public String toString() {
        var stats = getPerformanceStats();
        return String.format("VectorizedBARTMAP{biclusters=%d, coherence=%.3f, " +
                           "correlations=%d, rowOps=%d, colOps=%d}",
                           stats.numBiclusters(), stats.avgBiclusterCoherence(),
                           stats.correlationCalculations(),
                           stats.rowClusteringOps(), stats.columnClusteringOps());
    }
}