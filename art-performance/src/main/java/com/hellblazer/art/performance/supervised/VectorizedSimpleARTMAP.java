package com.hellblazer.art.performance.supervised;

import com.hellblazer.art.performance.algorithms.VectorizedART;
import com.hellblazer.art.performance.algorithms.VectorizedPerformanceStats;
import com.hellblazer.art.performance.VectorizedARTAlgorithm;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.artmap.SimpleARTMAP;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.core.MatchResetFunction;
import com.hellblazer.art.core.MatchTrackingMode;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * High-performance vectorized SimpleARTMAP implementation for supervised classification.
 * 
 * SimpleARTMAP is a simplified version of ARTMAP that uses:
 * - A single ART module for clustering input patterns
 * - A map field that maintains many-to-one mappings from clusters to labels
 * - Match tracking to handle label conflicts by adjusting vigilance
 * 
 * This vectorized implementation provides:
 * - SIMD optimization for pattern processing
 * - Performance optimization with parallel processing capabilities
 * - Comprehensive performance metrics and result tracking
 * - Thread-safe operations with proper resource management
 * - Type-safe parameter handling
 */
public class VectorizedSimpleARTMAP implements VectorizedARTAlgorithm<VectorizedSimpleARTMAP.PerformanceMetrics, VectorizedSimpleARTMAPParameters>, AutoCloseable {
    
    
    // Core components
    private final SimpleARTMAP baseSimpleARTMAP;
    private final VectorizedART vectorizedArtA;
    private final VectorizedSimpleARTMAPParameters vectorizedParams;
    
    // Enhanced map field with statistics
    private final Map<Integer, Integer> enhancedMapField = new ConcurrentHashMap<>();
    private final Map<Integer, Double> mapFieldStrengths = new ConcurrentHashMap<>();
    private final Map<Integer, Long> mapFieldUsageCounts = new ConcurrentHashMap<>();
    
    // Performance tracking
    private final AtomicLong trainingOperations = new AtomicLong(0);
    private final AtomicLong predictionOperations = new AtomicLong(0);
    private final AtomicLong matchTrackingSearches = new AtomicLong(0);
    private final AtomicLong mapFieldMismatches = new AtomicLong(0);
    private volatile double totalTrainingTime = 0.0;
    private volatile double totalPredictionTime = 0.0;
    private volatile double totalSearchDepth = 0.0;
    
    // Thread safety
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
    
    // Resource management
    private volatile boolean closed = false;
    
    /**
     * Performance metrics for VectorizedSimpleARTMAP.
     */
    public record PerformanceMetrics(
        long trainingOperations,
        long predictionOperations,
        long matchTrackingSearches,
        long mapFieldMismatches,
        double averageTrainingTime,
        double averagePredictionTime,
        double averageSearchDepth,
        int categoriesCreated,
        int mapFieldSize,
        VectorizedPerformanceStats artAStats
    ) {}
    
    /**
     * Create a new VectorizedSimpleARTMAP with specified parameters.
     * 
     * @param parameters the VectorizedSimpleARTMAP-specific parameters
     */
    public VectorizedSimpleARTMAP(VectorizedSimpleARTMAPParameters parameters) {
        this.vectorizedParams = Objects.requireNonNull(parameters, "Parameters cannot be null");
        this.vectorizedArtA = new VectorizedART(parameters.artAParams());
        this.baseSimpleARTMAP = new SimpleARTMAP(vectorizedArtA, parameters.toSimpleARTMAPParameters());
        
        // Initialized VectorizedSimpleARTMAP with parameters
    }
    
    // VectorizedARTAlgorithm interface implementation
    
    @Override
    public Object learn(Pattern input, VectorizedSimpleARTMAPParameters parameters) {
        throw new UnsupportedOperationException("SimpleARTMAP requires both input and label. Use train(Pattern, int) instead.");
    }
    
    @Override
    public Object predict(Pattern input, VectorizedSimpleARTMAPParameters parameters) {
        ensureNotClosed();
        Objects.requireNonNull(input, "input cannot be null");
        Objects.requireNonNull(parameters, "parameters cannot be null");
        
        var startTime = System.nanoTime();
        try {
            predictionOperations.incrementAndGet();
            
            // Use the base SimpleARTMAP for prediction
            int result = baseSimpleARTMAP.predict(input, parameters.artAParams());
            
            return result;
        } finally {
            var elapsedTime = (System.nanoTime() - startTime) / 1_000_000.0; // ms
            lock.writeLock().lock();
            try {
                totalPredictionTime += elapsedTime;
            } finally {
                lock.writeLock().unlock();
            }
        }
    }
    
    @Override
    public VectorizedSimpleARTMAPParameters getParameters() {
        return vectorizedParams;
    }
    
    @Override
    public int getCategoryCount() {
        return vectorizedArtA.getCategoryCount();
    }
    
    @Override
    public PerformanceMetrics getPerformanceStats() {
        lock.readLock().lock();
        try {
            var artAStats = vectorizedArtA.getPerformanceStats();
            
            return new PerformanceMetrics(
                trainingOperations.get(),
                predictionOperations.get(),
                matchTrackingSearches.get(),
                mapFieldMismatches.get(),
                trainingOperations.get() > 0 ? totalTrainingTime / trainingOperations.get() : 0.0,
                predictionOperations.get() > 0 ? totalPredictionTime / predictionOperations.get() : 0.0,
                matchTrackingSearches.get() > 0 ? totalSearchDepth / matchTrackingSearches.get() : 0.0,
                vectorizedArtA.getCategoryCount(),
                enhancedMapField.size(),
                artAStats
            );
        } finally {
            lock.readLock().unlock();
        }
    }
    
    @Override
    public void resetPerformanceTracking() {
        lock.writeLock().lock();
        try {
            trainingOperations.set(0);
            predictionOperations.set(0);
            matchTrackingSearches.set(0);
            mapFieldMismatches.set(0);
            totalTrainingTime = 0.0;
            totalPredictionTime = 0.0;
            totalSearchDepth = 0.0;
            vectorizedArtA.resetPerformanceTracking();
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    @Override
    public void close() {
        if (!closed) {
            lock.writeLock().lock();
            try {
                if (!closed) {
                    // Closing VectorizedSimpleARTMAP
                    vectorizedArtA.close();
                    enhancedMapField.clear();
                    mapFieldStrengths.clear();
                    mapFieldUsageCounts.clear();
                    closed = true;
                }
            } finally {
                lock.writeLock().unlock();
            }
        }
    }
    
    private void ensureNotClosed() {
        if (closed) {
            throw new IllegalStateException("VectorizedSimpleARTMAP has been closed");
        }
    }
    
    // === SimpleARTMAP-specific methods ===
    
    /**
     * Training result for VectorizedSimpleARTMAP.
     */
    public record TrainResult(
        int categoryA,
        int predictedLabel,
        boolean matchTrackingOccurred,
        double adjustedVigilance,
        double trainingTime
    ) {}
    
    /**
     * Train on a single input pattern with its label.
     * 
     * @param input the input pattern
     * @param label the class label
     * @return the training result with performance metrics
     */
    public TrainResult train(Pattern input, int label) {
        ensureNotClosed();
        Objects.requireNonNull(input, "input cannot be null");
        
        var startTime = System.nanoTime();
        try {
            trainingOperations.incrementAndGet();
            
            // Use the base SimpleARTMAP for training
            var result = baseSimpleARTMAP.train(input, label, vectorizedParams.artAParams());
            
            // Update enhanced map field statistics
            lock.writeLock().lock();
            try {
                enhancedMapField.put(result.categoryA(), label);
                mapFieldUsageCounts.merge(result.categoryA(), 1L, Long::sum);
                
                if (result.matchTrackingOccurred()) {
                    matchTrackingSearches.incrementAndGet();
                    totalSearchDepth += 1.0; // Basic search depth tracking
                }
            } finally {
                lock.writeLock().unlock();
            }
            
            var elapsedTime = (System.nanoTime() - startTime) / 1_000_000.0; // ms
            return new TrainResult(
                result.categoryA(),
                result.predictedLabel(),
                result.matchTrackingOccurred(),
                result.adjustedVigilance(),
                elapsedTime
            );
        } finally {
            var elapsedTime = (System.nanoTime() - startTime) / 1_000_000.0; // ms
            lock.writeLock().lock();
            try {
                totalTrainingTime += elapsedTime;
            } finally {
                lock.writeLock().unlock();
            }
        }
    }
    
    /**
     * Train on multiple input patterns with their labels.
     * 
     * @param data array of input patterns
     * @param labels array of class labels
     * @return array of training results
     */
    public TrainResult[] fit(Pattern[] data, int[] labels) {
        ensureNotClosed();
        Objects.requireNonNull(data, "data cannot be null");
        Objects.requireNonNull(labels, "labels cannot be null");
        
        if (data.length != labels.length) {
            throw new IllegalArgumentException("data and labels must have same length");
        }
        
        var results = new TrainResult[data.length];
        for (int i = 0; i < data.length; i++) {
            results[i] = train(data[i], labels[i]);
        }
        return results;
    }
    
    /**
     * Predict class labels for multiple input patterns.
     * 
     * @param data array of input patterns
     * @return array of predicted class labels
     */
    public int[] predictBatch(Pattern[] data) {
        ensureNotClosed();
        Objects.requireNonNull(data, "data cannot be null");
        
        var predictions = new int[data.length];
        for (int i = 0; i < data.length; i++) {
            predictions[i] = (Integer) predict(data[i], vectorizedParams);
        }
        return predictions;
    }
    
    /**
     * Get the size of the map field.
     * 
     * @return number of cluster-to-label mappings
     */
    public int getMapFieldSize() {
        return baseSimpleARTMAP.getMapFieldSize();
    }
    
    /**
     * Check if the algorithm has been trained.
     * 
     * @return true if trained on at least one pattern
     */
    public boolean isTrained() {
        return baseSimpleARTMAP.isTrained();
    }
    
    /**
     * Clear all learned patterns and reset the algorithm.
     */
    public void clear() {
        ensureNotClosed();
        lock.writeLock().lock();
        try {
            baseSimpleARTMAP.clear();
            enhancedMapField.clear();
            mapFieldStrengths.clear();
            mapFieldUsageCounts.clear();
            resetPerformanceTracking();
        } finally {
            lock.writeLock().unlock();
        }
    }
}