package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.*;
import com.hellblazer.art.core.hierarchical.SMART;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.core.results.MatchResult;
import com.hellblazer.art.performance.VectorizedARTAlgorithm;

import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Vectorized SMART (Self-Monitoring Adaptive Resonance Theory) implementation.
 * Performs hierarchical clustering with multiple layers of different vigilance values.
 */
public class VectorizedSMART extends BaseART<VectorizedSMARTParameters> implements VectorizedARTAlgorithm<VectorizedSMART.PerformanceMetrics, VectorizedSMARTParameters> {

    private final SMART smart;
    private final VectorizedSMARTParameters defaultParams;
    private final ReentrantReadWriteLock lock;
    
    // Layer-specific tracking
    private final int[] categoriesPerLayer;
    private final Map<Integer, Set<Integer>> layerCategoryMapping;
    
    // Performance tracking
    private final AtomicLong simdOperations;
    private final AtomicLong totalOperations;
    private final AtomicLong hierarchicalMappings;
    private final AtomicLong categoryPathUpdates;
    private final AtomicLong interLayerConnections;
    private final AtomicLong[] layerOperations;
    private long startTime;
    
    /**
     * Performance metrics for vectorized SMART.
     */
    public record PerformanceMetrics(
        long simdOperations,
        long totalOperations,
        long hierarchicalMappings,
        long categoryPathUpdates,
        long interLayerConnections,
        long[] layerOperations,
        long elapsedTimeNanos,
        double throughputOpsPerSec,
        double simdUtilization,
        int totalCategories,
        double avgCategoriesPerLayer
    ) {
        public static PerformanceMetrics empty() {
            return new PerformanceMetrics(0, 0, 0, 0, 0, new long[0], 0, 0.0, 0.0, 0, 0.0);
        }
    }
    
    /**
     * Create a new VectorizedSMART with default parameters.
     */
    public VectorizedSMART() {
        this(new VectorizedSMARTParameters());
    }
    
    /**
     * Create a new VectorizedSMART with specified parameters.
     */
    public VectorizedSMART(VectorizedSMARTParameters parameters) {
        // Create SMART based on module type
        this.smart = createSMART(parameters);
        this.defaultParams = parameters;
        this.lock = new ReentrantReadWriteLock();
        
        // Initialize layer tracking
        int numLayers = parameters.getNumLayers();
        this.categoriesPerLayer = new int[numLayers];
        this.layerCategoryMapping = new HashMap<>();
        
        // Initialize performance counters
        this.simdOperations = new AtomicLong();
        this.totalOperations = new AtomicLong();
        this.hierarchicalMappings = new AtomicLong();
        this.categoryPathUpdates = new AtomicLong();
        this.interLayerConnections = new AtomicLong();
        
        this.layerOperations = new AtomicLong[numLayers];
        for (int i = 0; i < numLayers; i++) {
            layerOperations[i] = new AtomicLong();
        }
        
        this.startTime = System.nanoTime();
    }
    
    /**
     * Create SMART instance based on module type.
     */
    private SMART createSMART(VectorizedSMARTParameters params) {
        var vigilanceValues = params.getVigilanceValues();
        var alpha = params.getAlpha();
        var beta = params.getLearningRate();
        
        return switch (params.getModuleType()) {
            case FUZZY_ART -> SMART.createWithFuzzyART(vigilanceValues, alpha, beta);
            case GAUSSIAN_ART -> SMART.createWithGaussianART(
                vigilanceValues, params.getInputDimension(), 1.0
            );
            case BAYESIAN_ART -> SMART.createWithBayesianART(
                vigilanceValues, params.getInputDimension(), 1.0, 1.0, 
                params.getMaxCategoriesPerLayer()
            );
            default -> SMART.createWithFuzzyART(vigilanceValues, alpha, beta);
        };
    }
    
    public Object learn(double[] input) {
        return learn(Pattern.of(input), defaultParams);
    }
    
    public ActivationResult learnWithParams(Pattern input, VectorizedSMARTParameters params) {
        lock.writeLock().lock();
        try {
            totalOperations.incrementAndGet();
            
            // Track SIMD operations across layers
            long simdOps = estimateLayerSimdOperations(input.dimension(), params.getNumLayers());
            simdOperations.addAndGet(simdOps);
            
            // Perform hierarchical learning
            var patterns = List.of(input);
            var labels = smart.fit(patterns, 1);
            
            // Track layer operations
            for (int layer = 0; layer < params.getNumLayers(); layer++) {
                layerOperations[layer].incrementAndGet();
            }
            
            // Update layer mapping based on the single label returned
            if (labels != null && labels.length > 0) {
                int categoryIndex = labels[0];
                
                // For now, treat as layer 0 category
                layerCategoryMapping.computeIfAbsent(0, k -> new HashSet<>())
                    .add(categoryIndex);
                categoriesPerLayer[0] = Math.max(categoriesPerLayer[0], categoryIndex + 1);
            }
            
            // Track hierarchical mappings (these methods don't exist in SMART yet)
            // For now, just track basic operations
            if (params.isTrackCategoryPaths()) {
                categoryPathUpdates.incrementAndGet();
            }
            
            // Track inter-layer connections if enabled
            if (params.isAllowInterLayerConnections()) {
                hierarchicalMappings.incrementAndGet();
                interLayerConnections.incrementAndGet();
            }
            
            // Return activation result
            if (labels != null && labels.length > 0) {
                int categoryIndex = labels[0];
                if (categoryIndex < getCategoryCount()) {
                    return new ActivationResult.Success(categoryIndex, 1.0, getCategory(categoryIndex));
                }
            }
            return ActivationResult.NoMatch.instance();
            
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    public int predict(double[] input) {
        var result = predict(Pattern.of(input), defaultParams);
        if (result instanceof ActivationResult.Success success) {
            return success.categoryIndex();
        }
        return -1;
    }
    
    public ActivationResult predictWithParams(Pattern input, VectorizedSMARTParameters params) {
        lock.readLock().lock();
        try {
            totalOperations.incrementAndGet();
            
            // Track SIMD operations
            long simdOps = estimateLayerSimdOperations(input.dimension(), params.getNumLayers());
            simdOperations.addAndGet(simdOps);
            
            // Perform prediction
            var patterns = List.of(input);
            var labels = smart.predict(patterns);
            
            // Track layer operations for prediction
            for (int layer = 0; layer < params.getNumLayers(); layer++) {
                layerOperations[layer].incrementAndGet();
            }
            
            // Return activation result
            if (labels != null && labels.length > 0) {
                int categoryIndex = labels[0];
                if (categoryIndex < getCategoryCount()) {
                    return new ActivationResult.Success(categoryIndex, 1.0, getCategory(categoryIndex));
                }
            }
            return ActivationResult.NoMatch.instance();
            
        } finally {
            lock.readLock().unlock();
        }
    }
    
    public int getTotalCategoryCount() {
        lock.readLock().lock();
        try {
            // Return total categories across all layers
            int total = 0;
            for (int count : categoriesPerLayer) {
                total += count;
            }
            return total;
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Get category count for a specific layer.
     */
    public int getCategoryCount(int layer) {
        lock.readLock().lock();
        try {
            if (layer >= 0 && layer < categoriesPerLayer.length) {
                return categoriesPerLayer[layer];
            }
            return 0;
        } finally {
            lock.readLock().unlock();
        }
    }
    
    @Override
    public VectorizedSMARTParameters getParameters() {
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
            
            // Calculate total categories and average per layer
            int totalCategories = 0;
            for (int count : categoriesPerLayer) {
                totalCategories += count;
            }
            double avgCategoriesPerLayer = categoriesPerLayer.length > 0 ?
                (double) totalCategories / categoriesPerLayer.length : 0.0;
            
            // Copy layer operations
            long[] layerOpsCopy = new long[layerOperations.length];
            for (int i = 0; i < layerOperations.length; i++) {
                layerOpsCopy[i] = layerOperations[i].get();
            }
            
            return new PerformanceMetrics(
                simdOps,
                totalOps,
                hierarchicalMappings.get(),
                categoryPathUpdates.get(),
                interLayerConnections.get(),
                layerOpsCopy,
                elapsed,
                throughput,
                simdUtil,
                totalCategories,
                avgCategoriesPerLayer
            );
        } finally {
            lock.readLock().unlock();
        }
    }
    
    @Override
    public void resetPerformanceTracking() {
        simdOperations.set(0);
        totalOperations.set(0);
        hierarchicalMappings.set(0);
        categoryPathUpdates.set(0);
        interLayerConnections.set(0);
        for (var counter : layerOperations) {
            counter.set(0);
        }
        startTime = System.nanoTime();
    }
    
    /**
     * Estimate SIMD operations across hierarchical layers.
     */
    private long estimateLayerSimdOperations(int inputDimension, int numLayers) {
        // Base operations per layer: activation, match, learning
        long baseOpsPerLayer = inputDimension * 4;
        
        // Parallel processing benefit
        if (defaultParams.isParallelLayerProcessing()) {
            // Assume some overlap in parallel processing
            return baseOpsPerLayer * numLayers / 2;
        } else {
            // Sequential processing
            return baseOpsPerLayer * numLayers;
        }
    }
    
    /**
     * Convert Pattern to double array.
     */
    private double[] patternToArray(Pattern pattern) {
        double[] array = new double[pattern.dimension()];
        for (int i = 0; i < pattern.dimension(); i++) {
            array[i] = pattern.get(i);
        }
        return array;
    }
    
    /**
     * Get the underlying SMART model.
     */
    public SMART getSMART() {
        return smart;
    }
    
    /**
     * Get hierarchical mappings between layers.
     * Note: These methods are not yet implemented in SMART.
     */
    public Map<Object, int[]> getHierarchicalMappings() {
        lock.readLock().lock();
        try {
            // Return empty map for now - SMART doesn't have this method yet
            return new HashMap<>();
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Get category paths through the hierarchy.
     * Note: These methods are not yet implemented in SMART.
     */
    public List<int[]> getCategoryPaths() {
        lock.readLock().lock();
        try {
            // Return empty list for now - SMART doesn't have this method yet
            return new ArrayList<>();
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Get layer-specific category mapping.
     */
    public Map<Integer, Set<Integer>> getLayerCategoryMapping() {
        lock.readLock().lock();
        try {
            return new HashMap<>(layerCategoryMapping);
        } finally {
            lock.readLock().unlock();
        }
    }
    
    // Implement BaseART abstract methods
    
    @Override
    protected double calculateActivation(Pattern input, WeightVector weight, VectorizedSMARTParameters parameters) {
        // SMART uses its internal implementation
        // This is a placeholder for the abstract method requirement
        return 1.0;
    }
    
    @Override
    protected MatchResult checkVigilance(Pattern input, WeightVector weight, VectorizedSMARTParameters parameters) {
        // SMART handles vigilance internally across layers
        // This is a placeholder for the abstract method requirement
        return new MatchResult.Accepted(1.0, parameters.getVigilanceValues()[0]);
    }
    
    @Override
    protected WeightVector updateWeights(Pattern input, WeightVector currentWeight, VectorizedSMARTParameters parameters) {
        // SMART handles weight updates internally
        // This is a placeholder for the abstract method requirement
        return currentWeight;
    }
    
    @Override
    protected WeightVector createInitialWeight(Pattern input, VectorizedSMARTParameters parameters) {
        // SMART creates weights internally
        // Create a placeholder weight vector
        double[] data = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            data[i] = input.get(i);
        }
        return new WeightVector() {
            @Override
            public double get(int index) {
                return data[index];
            }
            
            @Override
            public int dimension() {
                return data.length;
            }
            
            @Override
            public double l1Norm() {
                double sum = 0.0;
                for (double d : data) {
                    sum += Math.abs(d);
                }
                return sum;
            }
            
            @Override
            public WeightVector update(Pattern pattern, Object parameters) {
                // Placeholder implementation
                return this;
            }
        };
    }
    
    @Override
    public void close() {
        // No resources to release
    }
    
    @Override
    public String toString() {
        var stats = getPerformanceStats();
        return String.format("VectorizedSMART{layers=%d, totalCategories=%d, " +
                           "avgPerLayer=%.1f, mappings=%d, connections=%d}",
                           defaultParams.getNumLayers(), stats.totalCategories(),
                           stats.avgCategoriesPerLayer(), stats.hierarchicalMappings(),
                           stats.interLayerConnections());
    }
}