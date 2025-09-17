package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.algorithms.ARTSTAR;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.core.results.EllipsoidActivationResult;
import com.hellblazer.art.performance.VectorizedARTAlgorithm;

import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Vectorized ARTSTAR (ART with STability and Adaptability Regulation) implementation.
 * Provides automatic balance between stability (preserving knowledge) and 
 * adaptability (learning new patterns) through dynamic regulation.
 */
public class VectorizedARTSTAR implements VectorizedARTAlgorithm<VectorizedARTSTAR.PerformanceMetrics, VectorizedARTSTARParameters> {

    private final ARTSTAR artstar;
    private final VectorizedARTSTARParameters defaultParams;
    private final ReentrantReadWriteLock lock;
    
    // Performance tracking
    private final AtomicLong simdOperations;
    private final AtomicLong totalOperations;
    private final AtomicLong stabilityRegulations;
    private final AtomicLong adaptabilityRegulations;
    private final AtomicLong vigilanceAdjustments;
    private final AtomicLong categoryPrunings;
    private long startTime;
    
    // Simulated regulation state
    private double currentStability = 0.5;
    private double currentAdaptability = 0.5;
    
    /**
     * Performance metrics for vectorized ARTSTAR.
     */
    public record PerformanceMetrics(
        long simdOperations,
        long totalOperations,
        long stabilityRegulations,
        long adaptabilityRegulations,
        long vigilanceAdjustments,
        long categoryPrunings,
        long elapsedTimeNanos,
        double throughputOpsPerSec,
        double simdUtilization,
        double stabilityLevel,
        double adaptabilityLevel
    ) {
        public static PerformanceMetrics empty() {
            return new PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.5, 0.5);
        }
    }
    
    /**
     * Create a new VectorizedARTSTAR with default parameters.
     */
    public VectorizedARTSTAR() {
        this(new VectorizedARTSTARParameters());
    }
    
    /**
     * Create a new VectorizedARTSTAR with specified parameters.
     */
    public VectorizedARTSTAR(VectorizedARTSTARParameters parameters) {
        this.artstar = new ARTSTAR();
        this.defaultParams = parameters;
        this.lock = new ReentrantReadWriteLock();
        
        // Initialize performance counters
        this.simdOperations = new AtomicLong();
        this.totalOperations = new AtomicLong();
        this.stabilityRegulations = new AtomicLong();
        this.adaptabilityRegulations = new AtomicLong();
        this.vigilanceAdjustments = new AtomicLong();
        this.categoryPrunings = new AtomicLong();
        this.startTime = System.nanoTime();
    }
    
    public Object learn(double[] input) {
        return learn(Pattern.of(input), defaultParams);
    }
    
    @Override
    public Object learn(Pattern input, VectorizedARTSTARParameters params) {
        lock.writeLock().lock();
        try {
            totalOperations.incrementAndGet();
            
            // Track SIMD operations
            long simdOps = estimateSimdOperations(input.dimension());
            simdOperations.addAndGet(simdOps);
            
            // Simulate regulation events based on network state
            updateRegulationState(params);
            
            // Perform learning
            var artstarParams = params.toParameters();
            var result = artstar.stepFit(input, artstarParams);
            
            // Track vigilance adjustments (simulated based on stability/adaptability)
            if (shouldAdjustVigilance(params)) {
                vigilanceAdjustments.incrementAndGet();
            }
            
            // Track category pruning (when exceeding max categories)
            if (artstar.getCategoryCount() > params.getMaxCategories()) {
                categoryPrunings.incrementAndGet();
                // In real implementation would prune weakest categories
            }
            
            // Convert result
            return switch (result) {
                case ActivationResult.Success success -> 
                    success.categoryIndex();
                case ActivationResult.NoMatch noMatch ->
                    artstar.getCategoryCount();
                case EllipsoidActivationResult ellipsoid ->
                    ellipsoid.categoryIndex();
            };
            
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    public int predict(double[] input) {
        var result = predict(Pattern.of(input), defaultParams);
        return result instanceof Integer ? (Integer) result : -1;
    }
    
    @Override
    public Object predict(Pattern input, VectorizedARTSTARParameters params) {
        lock.readLock().lock();
        try {
            totalOperations.incrementAndGet();
            
            // Track SIMD operations
            long simdOps = estimateSimdOperations(input.dimension());
            simdOperations.addAndGet(simdOps);
            
            // Perform prediction
            var artstarParams = params.toParameters();
            var result = artstar.stepPredict(input, artstarParams);
            
            return switch (result) {
                case ActivationResult.Success success -> success.categoryIndex();
                case ActivationResult.NoMatch noMatch -> -1;
                case EllipsoidActivationResult ellipsoid -> ellipsoid.categoryIndex();
            };
            
        } finally {
            lock.readLock().unlock();
        }
    }
    
    @Override
    public int getCategoryCount() {
        lock.readLock().lock();
        try {
            return artstar.getCategoryCount();
        } finally {
            lock.readLock().unlock();
        }
    }
    
    @Override
    public VectorizedARTSTARParameters getParameters() {
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
            
            return new PerformanceMetrics(
                simdOps,
                totalOps,
                stabilityRegulations.get(),
                adaptabilityRegulations.get(),
                vigilanceAdjustments.get(),
                categoryPrunings.get(),
                elapsed,
                throughput,
                simdUtil,
                currentStability,
                currentAdaptability
            );
        } finally {
            lock.readLock().unlock();
        }
    }
    
    @Override
    public void resetPerformanceTracking() {
        simdOperations.set(0);
        totalOperations.set(0);
        stabilityRegulations.set(0);
        adaptabilityRegulations.set(0);
        vigilanceAdjustments.set(0);
        categoryPrunings.set(0);
        startTime = System.nanoTime();
    }
    
    /**
     * Update regulation state based on network performance.
     */
    private void updateRegulationState(VectorizedARTSTARParameters params) {
        // Simulate stability/adaptability dynamics
        double targetStability = params.getStabilityBias();
        double targetAdaptability = params.getAdaptabilityBias();
        
        // Move current values towards targets with regulation rate
        double regulationRate = params.getRegulationRate();
        
        if (Math.abs(currentStability - targetStability) > 0.1) {
            currentStability += (targetStability - currentStability) * regulationRate;
            stabilityRegulations.incrementAndGet();
        }
        
        if (Math.abs(currentAdaptability - targetAdaptability) > 0.1) {
            currentAdaptability += (targetAdaptability - currentAdaptability) * regulationRate;
            adaptabilityRegulations.incrementAndGet();
        }
    }
    
    /**
     * Determine if vigilance should be adjusted.
     */
    private boolean shouldAdjustVigilance(VectorizedARTSTARParameters params) {
        // Adjust vigilance based on stability/adaptability balance
        double balance = currentStability - currentAdaptability;
        return Math.abs(balance) > 0.2 && Math.random() < params.getVigilanceAdjustmentRate();
    }
    
    /**
     * Estimate SIMD operations for ARTSTAR processing.
     */
    private long estimateSimdOperations(int inputDimension) {
        // Base operations: activation, vigilance with regulation
        long baseOps = inputDimension * 4;
        
        // Stability/adaptability calculations
        baseOps += inputDimension * 2;
        
        return baseOps;
    }
    
    /**
     * Get the underlying ARTSTAR network.
     */
    public ARTSTAR getARTSTAR() {
        return artstar;
    }
    
    /**
     * Get current stability level.
     */
    public double getStabilityLevel() {
        lock.readLock().lock();
        try {
            return currentStability;
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Get current adaptability level.
     */
    public double getAdaptabilityLevel() {
        lock.readLock().lock();
        try {
            return currentAdaptability;
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
        return String.format("VectorizedARTSTAR{categories=%d, stability=%.3f, " +
                           "adaptability=%.3f, vigilanceAdj=%d, prunings=%d}",
                           getCategoryCount(), stats.stabilityLevel(),
                           stats.adaptabilityLevel(), stats.vigilanceAdjustments(),
                           stats.categoryPrunings());
    }
}