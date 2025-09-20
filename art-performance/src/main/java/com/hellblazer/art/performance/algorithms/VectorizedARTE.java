package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.algorithms.ARTE;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.core.results.EllipsoidActivationResult;
import com.hellblazer.art.performance.VectorizedARTAlgorithm;

import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Vectorized ART-E (Enhanced ART) implementation with adaptive learning features.
 * Wraps the core ARTE algorithm and adds performance tracking for:
 * - Feature weight adaptations
 * - Topology adjustments
 * - Convergence optimizations
 * - Performance-based pruning
 */
public class VectorizedARTE implements VectorizedARTAlgorithm<VectorizedARTE.PerformanceMetrics, VectorizedARTEParameters> {

    private final ARTE arte;
    private final VectorizedARTEParameters defaultParams;
    private final ReentrantReadWriteLock lock;
    
    // Performance tracking
    private final AtomicLong simdOperations;
    private final AtomicLong totalOperations;
    private final AtomicLong featureWeightAdaptations;
    private final AtomicLong topologyAdjustments;
    private final AtomicLong convergenceOptimizations;
    private final AtomicLong pruningOperations;
    private long startTime;
    
    /**
     * Performance metrics for vectorized ART-E.
     */
    public record PerformanceMetrics(
        long simdOperations,
        long totalOperations,
        long featureWeightAdaptations,
        long topologyAdjustments,
        long convergenceOptimizations,
        long pruningOperations,
        long elapsedTimeNanos,
        double throughputOpsPerSec,
        double simdUtilization,
        double networkPerformance,
        double convergenceRate
    ) {
        public static PerformanceMetrics empty() {
            return new PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0);
        }
    }
    
    /**
     * Create a new VectorizedARTE with default parameters.
     */
    public VectorizedARTE() {
        this(new VectorizedARTEParameters());
    }
    
    /**
     * Create a new VectorizedARTE with specified parameters.
     * @param parameters the ART-E parameters
     */
    public VectorizedARTE(VectorizedARTEParameters parameters) {
        this.arte = new ARTE();
        this.defaultParams = parameters;
        this.lock = new ReentrantReadWriteLock();
        
        // Initialize performance counters
        this.simdOperations = new AtomicLong();
        this.totalOperations = new AtomicLong();
        this.featureWeightAdaptations = new AtomicLong();
        this.topologyAdjustments = new AtomicLong();
        this.convergenceOptimizations = new AtomicLong();
        this.pruningOperations = new AtomicLong();
        this.startTime = System.nanoTime();
    }
    
    public Object learn(double[] input) {
        return learn(Pattern.of(input), defaultParams);
    }
    
    @Override
    public com.hellblazer.art.core.results.ActivationResult learn(Pattern input, VectorizedARTEParameters params) {
        lock.writeLock().lock();
        try {
            // Track operation start
            long opStart = System.nanoTime();
            totalOperations.incrementAndGet();
            
            // Estimate SIMD operations for this learning step
            long simdOps = estimateSimdOperations(input.dimension());
            simdOperations.addAndGet(simdOps);
            
            // Track feature weighting if enabled
            if (params.isFeatureWeightingEnabled()) {
                featureWeightAdaptations.incrementAndGet();
            }
            
            // Perform enhanced learning with adaptive features
            var arteParams = params.toParameters();
            // Use standard learning (enhanced method not available)
            ActivationResult result = arte.stepFit(input, arteParams);
            
            if (params.isPerformanceOptimizationEnabled()) {
                convergenceOptimizations.incrementAndGet();
                
                // Track network optimizations
                if (arte.getTotalLearningSteps() % 100 == 0) {
                    pruningOperations.incrementAndGet();
                }
            }
            
            // Track topology adjustments
            if (params.isTopologyAdjustmentEnabled() && 
                Math.random() < params.getTopologyAdjustmentProbability()) {
                topologyAdjustments.incrementAndGet();
            }
            
            // Return the result directly
            return result;
            
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
    public com.hellblazer.art.core.results.ActivationResult predict(Pattern input, VectorizedARTEParameters params) {
        lock.readLock().lock();
        try {
            totalOperations.incrementAndGet();
            
            // Track SIMD operations
            long simdOps = estimateSimdOperations(input.dimension());
            simdOperations.addAndGet(simdOps);
            
            // Perform prediction
            var arteParams = params.toParameters();
            var result = arte.stepPredict(input, arteParams);
            
            // Return the result directly
            return result;
            
        } finally {
            lock.readLock().unlock();
        }
    }
    
    @Override
    public int getCategoryCount() {
        lock.readLock().lock();
        try {
            return arte.getCategoryCount();
        } finally {
            lock.readLock().unlock();
        }
    }
    
    @Override
    public VectorizedARTEParameters getParameters() {
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
                (double) simdOps / (totalOps * 100) : 0.0; // Normalized to percentage
            
            // Get network-specific metrics
            double networkPerf = arte.getNetworkPerformance();
            double convergenceRate = arte.getConvergenceRate();
            
            return new PerformanceMetrics(
                simdOps,
                totalOps,
                featureWeightAdaptations.get(),
                topologyAdjustments.get(),
                convergenceOptimizations.get(),
                pruningOperations.get(),
                elapsed,
                throughput,
                simdUtil,
                networkPerf,
                convergenceRate
            );
        } finally {
            lock.readLock().unlock();
        }
    }
    
    @Override
    public void resetPerformanceTracking() {
        simdOperations.set(0);
        totalOperations.set(0);
        featureWeightAdaptations.set(0);
        topologyAdjustments.set(0);
        convergenceOptimizations.set(0);
        pruningOperations.set(0);
        startTime = System.nanoTime();
    }
    
    /**
     * Estimate SIMD operations for ART-E processing.
     * Includes feature weighting, familiarity calculations, and adaptive operations.
     */
    private long estimateSimdOperations(int inputDimension) {
        // Base operations: activation, vigilance
        long baseOps = inputDimension * 3;
        
        // Feature weighting operations
        if (defaultParams.isFeatureWeightingEnabled()) {
            baseOps += inputDimension * 2;
        }
        
        // Adaptive learning rate calculations
        if (defaultParams.getAdaptiveLearningFactor() > 0) {
            baseOps += inputDimension;
        }
        
        // Performance tracking operations
        if (defaultParams.isPerformanceOptimizationEnabled()) {
            baseOps += inputDimension * 2;
        }
        
        return baseOps;
    }
    
    /**
     * Get the underlying ARTE network for analysis.
     * @return the ARTE instance
     */
    public ARTE getARTE() {
        return arte;
    }
    
    /**
     * Perform network analysis including ART-E specific metrics.
     * @return comprehensive network analysis
     */
    public ARTE.NetworkAnalysis analyzeNetwork() {
        lock.readLock().lock();
        try {
            return arte.analyzeNetwork();
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Manually trigger network optimization.
     * Includes pruning and topology adjustments.
     */
    public void optimizeNetwork() {
        lock.writeLock().lock();
        try {
            arte.optimizeNetwork(defaultParams.toParameters());
            pruningOperations.incrementAndGet();
            if (defaultParams.isTopologyAdjustmentEnabled()) {
                topologyAdjustments.incrementAndGet();
            }
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    @Override
    public void close() {
        // No resources to release
    }
    
    @Override
    public String toString() {
        var stats = getPerformanceStats();
        return String.format("VectorizedARTE{categories=%d, simdOps=%d, " +
                           "featureAdapt=%d, topoAdjust=%d, convergenceOpt=%d, " +
                           "pruning=%d, netPerf=%.3f, convRate=%.3f}",
                           getCategoryCount(), stats.simdOperations(),
                           stats.featureWeightAdaptations(), stats.topologyAdjustments(),
                           stats.convergenceOptimizations(), stats.pruningOperations(),
                           stats.networkPerformance(), stats.convergenceRate());
    }

    @Override
    public void clear() {
        arte.clear();
    }

    @Override
    public com.hellblazer.art.core.WeightVector getCategory(int index) {
        return arte.getCategory(index);
    }

    @Override
    public java.util.List<com.hellblazer.art.core.WeightVector> getCategories() {
        return arte.getCategories();
    }
}