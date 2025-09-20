package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.*;
import com.hellblazer.art.core.results.MatchResult;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.performance.AbstractVectorizedART;
import com.hellblazer.art.performance.VectorizedARTAlgorithm;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.RecursiveTask;

/**
 * High-performance vectorized ART2 implementation using Java Vector API.
 * 
 * ART2 is designed for continuous analog input patterns with preprocessing:
 * - Contrast enhancement using theta parameter
 * - Noise suppression using epsilon parameter  
 * - Normalized weight vectors (unit length)
 * - Dot product activation function
 * - Distance-based vigilance criterion
 * - Convex combination learning
 * 
 * Features:
 * - SIMD-optimized vector operations (dot products, distances, updates)
 * - ART2-specific preprocessing with contrast enhancement and noise suppression
 * - Parallel processing for large category sets
 * - Cache-optimized data structures
 * - Performance monitoring and metrics
 * 
 * This implementation maintains full compatibility with ART2 semantics
 * while providing significant performance improvements through vectorization.
 */
public class VectorizedART2 extends AbstractVectorizedART<VectorizedPerformanceStats, VectorizedART2Parameters> {
    
    private static final Logger log = LoggerFactory.getLogger(VectorizedART2.class);
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    
    private final Map<Integer, float[]> inputCache = new ConcurrentHashMap<>();
    public VectorizedART2(VectorizedART2Parameters defaultParams) {
        super(defaultParams);
        log.info("Initialized VectorizedART2 with {} parallel threads, vector species: {}, theta={}, epsilon={}",
                 defaultParams.parallelismLevel(), SPECIES.toString(), defaultParams.theta(), defaultParams.epsilon());
    }
    
    /**
     * Convert WeightVector to VectorizedART2Weight for compatibility with BaseART.
     */
    private VectorizedART2Weight convertToVectorizedART2Weight(WeightVector weight) {
        if (weight instanceof VectorizedART2Weight art2Weight) {
            return art2Weight;
        }
        
        // Create VectorizedART2Weight from any WeightVector
        var weights = new double[weight.dimension()];
        for (int i = 0; i < weight.dimension(); i++) {
            weights[i] = weight.get(i);
        }
        
        return new VectorizedART2Weight(weights, System.currentTimeMillis(), 0);
    }
    
    @Override
    protected double calculateActivation(Pattern input, WeightVector weight, VectorizedART2Parameters parameters) {
        // BaseART already checks for null inputs
        var art2Params = parameters;
        
        // Convert WeightVector to VectorizedART2Weight
        VectorizedART2Weight art2Weight = convertToVectorizedART2Weight(weight);
        
        trackVectorOperation();
        return art2Weight.computeActivation(input, art2Params);
    }
    
    @Override
    protected MatchResult checkVigilance(Pattern input, WeightVector weight, VectorizedART2Parameters parameters) {
        // BaseART already checks for null inputs
        var art2Params = parameters;
        
        // Convert WeightVector to VectorizedART2Weight
        VectorizedART2Weight art2Weight = convertToVectorizedART2Weight(weight);
        
        trackVectorOperation();
        trackMatchOperation();
        double similarity = art2Weight.computeVigilance(input, art2Params);
        return similarity >= art2Params.vigilance() ? 
               new MatchResult.Accepted(similarity, art2Params.vigilance()) : 
               new MatchResult.Rejected(similarity, art2Params.vigilance());
    }
    
    @Override
    protected WeightVector updateWeights(Pattern input, WeightVector currentWeight, VectorizedART2Parameters parameters) {
        // BaseART already checks for null inputs
        var art2Params = parameters;
        
        // Convert and update
        VectorizedART2Weight art2Weight = convertToVectorizedART2Weight(currentWeight);
        trackVectorOperation();
        return art2Weight.updateART2(input, art2Params);
    }
    
    @Override
    protected WeightVector createInitialWeight(Pattern input, VectorizedART2Parameters parameters) {
        // BaseART already checks for null inputs
        var art2Params = parameters;
        
        trackVectorOperation();
        return VectorizedART2Weight.fromInput(input, art2Params);
    }
    
    // AbstractVectorizedART implementation

    protected void validateParameters(VectorizedART2Parameters params) {
        Objects.requireNonNull(params, "Parameters cannot be null");
        // Additional ART2-specific validation could go here
    }

    @Override
    protected VectorizedPerformanceStats createPerformanceStats(
            long vectorOps, long parallelTasks, long activations,
            long matches, long learnings, double avgTime) {
        return new VectorizedPerformanceStats(
            vectorOps,
            parallelTasks,
            avgTime,
            getComputePool().getActiveThreadCount(),
            inputCache.size(),
            getCategoryCount(),
            activations,
            matches,
            learnings
        );
    }
    
    // Not @Override - parent doesn't have this method
    protected Object performVectorizedLearning(Pattern input, VectorizedART2Parameters params) {
        return stepFitEnhanced(input, params);
    }
    
    // Not @Override - parent doesn't have this method
    protected Object performVectorizedPrediction(Pattern input, VectorizedART2Parameters params) {
        // Handle null gracefully by using defaults
        if (input == null) {
            // Match dimension of existing categories if any exist
            if (getCategoryCount() > 0) {
                var firstCategory = getCategories().get(0);
                int dim = firstCategory.dimension();
                double[] values = new double[dim];
                Arrays.fill(values, 0.5);
                input = Pattern.of(values);
            } else {
                // Use default 4-dimensional pattern for first category
                input = Pattern.of(0.5, 0.5, 0.5, 0.5);
            }
        }
        if (params == null) {
            params = VectorizedART2Parameters.createDefault();
        }
        return stepFit(input, params);
    }
    
    protected void clearAlgorithmState() {
        inputCache.clear();
        // Reset tracking is handled by parent class
    }
    
    protected void closeAlgorithmResources() {
        inputCache.clear();
    }

    /**
     * Enhanced stepFit with performance optimizations and parallel processing.
     */
    public ActivationResult stepFitEnhancedVectorized(Pattern input, VectorizedART2Parameters params) {
        // Handle null gracefully by using defaults
        if (input == null) {
            // Match dimension of existing categories if any exist
            if (getCategoryCount() > 0) {
                var firstCategory = getCategories().get(0);
                int dim = firstCategory.dimension();
                double[] values = new double[dim];
                Arrays.fill(values, 0.5);
                input = Pattern.of(values);
            } else {
                // Use default 4-dimensional pattern for first category
                input = Pattern.of(0.5, 0.5, 0.5, 0.5);
            }
        }
        if (params == null) {
            params = VectorizedART2Parameters.createDefault();
        }
        
        long startTime = System.nanoTime();
        
        try {
            // Use parallel processing for large category sets
            if (getCategoryCount() > 50) { // Threshold for parallel processing
                return parallelStepFit(input, params);
            } else {
                return stepFit(input, params);
            }
        } finally {
            updatePerformanceMetrics(startTime);
        }
    }
    
    /**
     * High-performance parallel step fit using ForkJoinPool.
     */
    private ActivationResult parallelStepFit(Pattern input, VectorizedART2Parameters params) {
        if (getCategoryCount() == 0) {
            return stepFit(input, params);
        }
        
        var task = new ParallelActivationTask(input, params, 0, getCategoryCount());
        var result = getComputePool().invoke(task);
        trackParallelTask();
        return result;
    }
    
    /**
     * Parallel activation computation task for ART2.
     */
    private class ParallelActivationTask extends RecursiveTask<ActivationResult> {
        private final Pattern input;
        private final VectorizedART2Parameters params;
        private final int startIndex;
        private final int endIndex;
        private static final int THRESHOLD = 100;
        
        ParallelActivationTask(Pattern input, VectorizedART2Parameters params, int startIndex, int endIndex) {
            this.input = input;
            this.params = params;
            this.startIndex = startIndex;
            this.endIndex = endIndex;
        }
        
        @Override
        protected ActivationResult compute() {
            if (endIndex - startIndex <= THRESHOLD) {
                return computeSequentialRange();
            }
            
            int mid = (startIndex + endIndex) / 2;
            var leftTask = new ParallelActivationTask(input, params, startIndex, mid);
            var rightTask = new ParallelActivationTask(input, params, mid, endIndex);
            
            leftTask.fork();
            var rightResult = rightTask.compute();
            var leftResult = leftTask.join();
            
            return chooseBestResult(leftResult, rightResult);
        }
        
        private ActivationResult computeSequentialRange() {
            double maxActivation = -1.0;
            int bestCategory = -1;
            WeightVector bestWeight = null;
            
            for (int i = startIndex; i < endIndex; i++) {
                var weight = getCategory(i);
                double activation = calculateActivation(input, weight, params);
                if (activation > maxActivation) {
                    var vigilanceResult = checkVigilance(input, weight, params);
                    if (vigilanceResult.isAccepted()) {
                        maxActivation = activation;
                        bestCategory = i;
                        bestWeight = weight;
                    }
                }
            }
            
            if (bestCategory >= 0) {
                var updatedWeight = updateWeights(input, bestWeight, params);
                return new ActivationResult.Success(bestCategory, maxActivation, updatedWeight);
            } else {
                // Create new category
                var newWeight = createInitialWeight(input, params);
                return new ActivationResult.Success(getCategoryCount(), 1.0, newWeight);
            }
        }
        
        private ActivationResult chooseBestResult(ActivationResult left, ActivationResult right) {
            if (left instanceof ActivationResult.Success leftSuccess &&
                right instanceof ActivationResult.Success rightSuccess) {
                return leftSuccess.activationValue() >= rightSuccess.activationValue() ? left : right;
            } else if (left instanceof ActivationResult.Success) {
                return left;
            } else if (right instanceof ActivationResult.Success) {
                return right;
            } else {
                return left;
            }
        }
    }
    
    /**
     * Update performance metrics.
     */
    private void updatePerformanceMetrics(long startTime) {
        long elapsed = System.nanoTime() - startTime;
        double elapsedMs = elapsed / 1_000_000.0;
        updateComputeTime(elapsedMs);
    }
    
    /**
     * Optimize memory usage by trimming caches.
     */
    public void optimizeMemory() {
        if (inputCache.size() > 1000) { // Default cache limit
            inputCache.clear();
            log.info("Input cache cleared to optimize memory usage");
        }
    }
    
    // VectorizedARTAlgorithm interface implementation
    // These are all provided by AbstractVectorizedART as final methods
    
    /**
     * ART2-specific preprocessing method for external use.
     */
    public Pattern preprocessInput(Pattern input) {
        return VectorizedART2Weight.preprocessART2Input(input, getParameters());
    }
    
    /**
     * Normalize input to unit length (ART2 requirement).
     */
    public Pattern normalizeInput(Pattern input) {
        return VectorizedART2Weight.normalizeToUnitLength(input);
    }
    
    /**
     * Get contrast enhancement effectiveness for current parameters.
     */
    public double getContrastEnhancement() {
        return getParameters().theta();
    }
    
    /**
     * Get noise suppression level for current parameters.
     */
    public double getNoiseSupression() {
        return getParameters().epsilon();
    }
    
    // learnBatch is provided as final method by parent class
    
    // predictBatch is provided as final method by parent class
    
}