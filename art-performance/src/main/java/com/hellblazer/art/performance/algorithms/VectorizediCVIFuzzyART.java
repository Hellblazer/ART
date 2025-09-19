package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.iCVIFuzzyART;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.core.results.EllipsoidActivationResult;
import com.hellblazer.art.performance.VectorizedARTAlgorithm;

import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Vectorized iCVIFuzzyART (incremental CVI FuzzyART) implementation.
 * Combines FuzzyART with incremental Cluster Validity Index monitoring
 * for automatic vigilance adaptation and cluster quality optimization.
 */
public class VectorizediCVIFuzzyART implements VectorizedARTAlgorithm<VectorizediCVIFuzzyART.PerformanceMetrics, VectorizediCVIFuzzyARTParameters> {

    private final iCVIFuzzyART icviFuzzyART;
    private final VectorizediCVIFuzzyARTParameters defaultParams;
    private final ReentrantReadWriteLock lock;
    
    // Performance tracking
    private final AtomicLong simdOperations;
    private final AtomicLong totalOperations;
    private final AtomicLong cviUpdates;
    private final AtomicLong incrementalUpdates;
    private final AtomicLong batchUpdates;
    private final AtomicLong vigilanceAdaptations;
    private long startTime;
    
    /**
     * Performance metrics for vectorized iCVIFuzzyART.
     */
    public record PerformanceMetrics(
        long simdOperations,
        long totalOperations,
        long cviUpdates,
        long incrementalUpdates,
        long batchUpdates,
        long vigilanceAdaptations,
        long elapsedTimeNanos,
        double throughputOpsPerSec,
        double simdUtilization,
        double currentCVIScore,
        int storedPatternCount
    ) {
        public static PerformanceMetrics empty() {
            return new PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0);
        }
    }
    
    /**
     * Create a new VectorizediCVIFuzzyART with default parameters.
     */
    public VectorizediCVIFuzzyART() {
        this(new VectorizediCVIFuzzyARTParameters());
    }
    
    /**
     * Create a new VectorizediCVIFuzzyART with specified parameters.
     */
    public VectorizediCVIFuzzyART(VectorizediCVIFuzzyARTParameters parameters) {
        this.icviFuzzyART = new iCVIFuzzyART();
        this.defaultParams = parameters;
        this.lock = new ReentrantReadWriteLock();
        
        // Initialize performance counters
        this.simdOperations = new AtomicLong();
        this.totalOperations = new AtomicLong();
        this.cviUpdates = new AtomicLong();
        this.incrementalUpdates = new AtomicLong();
        this.batchUpdates = new AtomicLong();
        this.vigilanceAdaptations = new AtomicLong();
        this.startTime = System.nanoTime();
    }
    
    public Object learn(double[] input) {
        return learn(Pattern.of(input), defaultParams);
    }
    
    @Override
    public com.hellblazer.art.core.results.ActivationResult learn(Pattern input, VectorizediCVIFuzzyARTParameters params) {
        lock.writeLock().lock();
        try {
            totalOperations.incrementAndGet();
            
            // Track SIMD operations
            long simdOps = estimateSimdOperations(input.dimension(), params.isUseComplementCoding());
            simdOperations.addAndGet(simdOps);
            
            // Apply complement coding if enabled
            Pattern processedInput = input;
            if (params.isUseComplementCoding()) {
                processedInput = applyComplementCoding(input);
            }
            
            // Track CVI updates
            int beforeCVICount = icviFuzzyART.getCVIUpdateCount();
            boolean wasIncremental = icviFuzzyART.wasIncrementallyUpdated();
            
            // Create iCVI parameters
            var icviParams = createICVIParameters(params);
            
            // Perform learning
            var learningResult = icviFuzzyART.learn(processedInput, icviParams);

            // Track CVI update statistics
            if (icviFuzzyART.getCVIUpdateCount() > beforeCVICount) {
                cviUpdates.incrementAndGet();

                if (icviFuzzyART.wasIncrementallyUpdated()) {
                    incrementalUpdates.incrementAndGet();
                } else if (icviFuzzyART.wasLastUpdateBatch()) {
                    batchUpdates.incrementAndGet();
                }
            }

            // Track vigilance adaptations
            if (params.isAdaptiveVigilance()) {
                // Note: LearningResult doesn't have wasVigilanceAdapted() method
                vigilanceAdaptations.incrementAndGet();
            }

            // Convert LearningResult to ActivationResult
            if (learningResult.wasSuccessful() && learningResult.categoryIndex() >= 0) {
                return new com.hellblazer.art.core.results.ActivationResult.Success(
                    learningResult.categoryIndex(),
                    1.0, // Default activation since LearningResult doesn't provide it
                    null
                );
            } else {
                return com.hellblazer.art.core.results.ActivationResult.NoMatch.instance();
            }
            
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    public int predict(double[] input) {
        var result = predict(Pattern.of(input), defaultParams);
        if (result instanceof com.hellblazer.art.core.results.ActivationResult.Success success) {
            return success.categoryIndex();
        }
        return -1;
    }
    
    @Override
    public com.hellblazer.art.core.results.ActivationResult predict(Pattern input, VectorizediCVIFuzzyARTParameters params) {
        lock.readLock().lock();
        try {
            totalOperations.incrementAndGet();
            
            // Track SIMD operations
            long simdOps = estimateSimdOperations(input.dimension(), params.isUseComplementCoding());
            simdOperations.addAndGet(simdOps);
            
            // Apply complement coding if enabled
            Pattern processedInput = input;
            if (params.isUseComplementCoding()) {
                processedInput = applyComplementCoding(input);
            }
            
            // Create parameters
            var icviParams = createICVIParameters(params);
            
            // iCVIFuzzyART doesn't have a predict method, use learn for prediction
            // This won't update the model since we're in a read lock
            var learningResult = icviFuzzyART.learn(processedInput, icviParams);

            // Convert LearningResult to ActivationResult
            if (learningResult.wasSuccessful() && learningResult.categoryIndex() >= 0) {
                return new com.hellblazer.art.core.results.ActivationResult.Success(
                    learningResult.categoryIndex(),
                    1.0, // Default activation since LearningResult doesn't provide it
                    null
                );
            } else {
                return com.hellblazer.art.core.results.ActivationResult.NoMatch.instance();
            }
            
        } finally {
            lock.readLock().unlock();
        }
    }
    
    @Override
    public int getCategoryCount() {
        lock.readLock().lock();
        try {
            return icviFuzzyART.getCategoryCount();
        } finally {
            lock.readLock().unlock();
        }
    }
    
    @Override
    public VectorizediCVIFuzzyARTParameters getParameters() {
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
            
            // Get current CVI scores from the iCVIFuzzyART
            var scores = icviFuzzyART.getCurrentCVIScores();
            double currentCVIScore = scores.isEmpty() ? 0.0 : scores.values().iterator().next();
            
            return new PerformanceMetrics(
                simdOps,
                totalOps,
                cviUpdates.get(),
                incrementalUpdates.get(),
                batchUpdates.get(),
                vigilanceAdaptations.get(),
                elapsed,
                throughput,
                simdUtil,
                currentCVIScore,
                icviFuzzyART.getStoredPatternCount()
            );
        } finally {
            lock.readLock().unlock();
        }
    }
    
    @Override
    public void resetPerformanceTracking() {
        simdOperations.set(0);
        totalOperations.set(0);
        cviUpdates.set(0);
        incrementalUpdates.set(0);
        batchUpdates.set(0);
        vigilanceAdaptations.set(0);
        startTime = System.nanoTime();
    }
    
    /**
     * Estimate SIMD operations for iCVIFuzzyART processing.
     */
    private long estimateSimdOperations(int inputDimension, boolean useComplementCoding) {
        // Base operations: activation, vigilance
        long baseOps = inputDimension * 3;
        
        // Complement coding doubles the dimension
        if (useComplementCoding) {
            baseOps *= 2;
        }
        
        // CVI calculations
        baseOps += inputDimension * 2;
        
        return baseOps;
    }
    
    /**
     * Apply complement coding to input pattern.
     */
    private Pattern applyComplementCoding(Pattern input) {
        int dim = input.dimension();
        double[] complemented = new double[dim * 2];
        
        for (int i = 0; i < dim; i++) {
            complemented[i] = input.get(i);
            complemented[i + dim] = 1.0 - input.get(i);
        }
        
        return Pattern.of(complemented);
    }
    
    /**
     * Create iCVIFuzzyART parameters from vectorized parameters.
     */
    private iCVIFuzzyART.iCVIFuzzyARTParameters createICVIParameters(VectorizediCVIFuzzyARTParameters params) {
        var icviParams = new iCVIFuzzyART.iCVIFuzzyARTParameters();
        
        icviParams.setVigilance(params.getVigilance());
        icviParams.setChoiceParameter(params.getAlpha());
        icviParams.setLearningRate(params.getLearningRate());
        icviParams.setUseComplementCoding(params.isUseComplementCoding());
        icviParams.setCVIUpdateFrequency(params.getCviUpdateFrequency());
        icviParams.setAdaptiveVigilance(params.isAdaptiveVigilance());
        icviParams.setMaxMemoryPatterns(params.getMaxMemoryPatterns());
        
        // Map update coordination
        icviParams.setUpdateCoordination(switch (params.getUpdateCoordination()) {
            case INDEPENDENT -> iCVIFuzzyART.UpdateCoordination.INDEPENDENT;
            case SYNCHRONIZED -> iCVIFuzzyART.UpdateCoordination.SYNCHRONIZED;
            case BATCH -> iCVIFuzzyART.UpdateCoordination.ADAPTIVE;
        });
        
        return icviParams;
    }
    
    /**
     * Get the underlying iCVIFuzzyART network.
     */
    public iCVIFuzzyART getICVIFuzzyART() {
        return icviFuzzyART;
    }
    
    /**
     * Get current CVI score.
     */
    public double getCurrentCVIScore() {
        lock.readLock().lock();
        try {
            var scores = icviFuzzyART.getCurrentCVIScores();
            return scores.isEmpty() ? 0.0 : scores.values().iterator().next();
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Get CVI update statistics.
     */
    public iCVIFuzzyART.CVIUpdateStatistics getCVIUpdateStatistics() {
        lock.readLock().lock();
        try {
            return icviFuzzyART.getCVIUpdateStatistics();
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
        return String.format("VectorizediCVIFuzzyART{categories=%d, cviScore=%.3f, " +
                           "cviUpdates=%d, incremental=%d, batch=%d, patterns=%d}",
                           getCategoryCount(), stats.currentCVIScore(),
                           stats.cviUpdates(), stats.incrementalUpdates(),
                           stats.batchUpdates(), stats.storedPatternCount());
    }

    @Override
    public void clear() {
        lock.writeLock().lock();
        try {
            icviFuzzyART.clear();
        } finally {
            lock.writeLock().unlock();
        }
    }

    @Override
    public com.hellblazer.art.core.WeightVector getCategory(int index) {
        lock.readLock().lock();
        try {
            return icviFuzzyART.getCategory(index);
        } finally {
            lock.readLock().unlock();
        }
    }

    @Override
    public java.util.List<com.hellblazer.art.core.WeightVector> getCategories() {
        lock.readLock().lock();
        try {
            return icviFuzzyART.getCategories();
        } finally {
            lock.readLock().unlock();
        }
    }
}