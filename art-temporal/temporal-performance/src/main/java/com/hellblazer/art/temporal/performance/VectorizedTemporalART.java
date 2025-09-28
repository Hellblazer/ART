package com.hellblazer.art.temporal.performance;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.performance.VectorizedARTAlgorithm;
import com.hellblazer.art.temporal.integration.*;
import com.hellblazer.art.temporal.memory.TemporalPattern;
import jdk.incubator.vector.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.stream.IntStream;

/**
 * High-performance vectorized TemporalART implementation.
 * Combines all vectorized components for maximum throughput.
 * Implements the standard VectorizedARTAlgorithm interface.
 */
public class VectorizedTemporalART implements VectorizedARTAlgorithm<VectorizedTemporalART.PerformanceStats, TemporalARTParameters> {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private final TemporalARTParameters parameters;
    private final VectorizedWorkingMemory workingMemory;
    private final VectorizedMaskingField maskingField;

    private final int maxCategories;
    private final int inputDimension;
    private final int vectorLength;

    // Category storage with lock-free structures
    private final ConcurrentHashMap<Integer, TemporalCategory> categories;
    private final AtomicInteger categoryCount;

    // Vectorized arrays for fast similarity computation
    private double[] categoryWeights;
    private double[][] categoryPrototypes;
    private double[][] rawPatterns; // Store original patterns for prediction
    private double[] activations;
    private double[] matches;

    // Performance metrics
    private final AtomicLong processTime;
    private final AtomicLong learnTime;
    private final AtomicLong predictTime;
    private final AtomicInteger patternsProcessed;

    // Thread pool for parallel processing
    private final ForkJoinPool computePool;

    // Current simulation time
    private double currentTime;

    public VectorizedTemporalART(TemporalARTParameters parameters) {
        this.parameters = parameters;
        this.workingMemory = new VectorizedWorkingMemory(parameters.getMemoryParameters());
        this.maskingField = new VectorizedMaskingField(
            parameters.getMaskingParameters(),
            workingMemory
        );

        this.maxCategories = parameters.getMaxCategories();
        this.inputDimension = parameters.getInputDimension();
        this.vectorLength = SPECIES.length();

        this.categories = new ConcurrentHashMap<>(maxCategories);
        this.categoryCount = new AtomicInteger(0);

        this.categoryWeights = new double[maxCategories];
        this.categoryPrototypes = new double[maxCategories][inputDimension];
        this.rawPatterns = new double[maxCategories][inputDimension]; // Store original patterns
        this.activations = new double[maxCategories];
        this.matches = new double[maxCategories];

        this.processTime = new AtomicLong(0);
        this.learnTime = new AtomicLong(0);
        this.predictTime = new AtomicLong(0);
        this.patternsProcessed = new AtomicInteger(0);

        this.computePool = new ForkJoinPool(
            parameters.getParallelismLevel(),
            ForkJoinPool.defaultForkJoinWorkerThreadFactory,
            null,
            true
        );

        this.currentTime = 0.0;
    }

    // === VectorizedARTAlgorithm Interface Implementation ===

    /**
     * Learn from input pattern (ARTAlgorithm interface).
     */
    @Override
    public ActivationResult learn(Pattern input, TemporalARTParameters params) {
        var inputData = extractPatternData(input);
        int category = processInput(inputData, currentTime);
        if (category >= 0) {
            // Return the updated category prototype as weight
            var weight = new TemporalWeight(categoryPrototypes[category]);
            double activation = computeActivation(categoryPrototypes[category], inputData);
            return new ActivationResult.Success(category, activation, weight);
        } else {
            return ActivationResult.NoMatch.instance();
        }
    }

    /**
     * Predict category for input pattern (ARTAlgorithm interface).
     */
    @Override
    public ActivationResult predict(Pattern input, TemporalARTParameters params) {
        var inputData = extractPatternData(input);
        int category = predictCategory(inputData);
        if (category >= 0) {
            // Return the category prototype as weight
            var weight = new TemporalWeight(categoryPrototypes[category]);
            double activation = computeActivation(categoryPrototypes[category], inputData);
            return new ActivationResult.Success(category, activation, weight);
        } else {
            return ActivationResult.NoMatch.instance();
        }
    }

    /**
     * Get category count (ARTAlgorithm interface).
     */
    @Override
    public int getCategoryCount() {
        return categoryCount.get();
    }

    /**
     * Get all categories (ARTAlgorithm interface).
     */
    @Override
    public List<WeightVector> getCategories() {
        var result = new ArrayList<WeightVector>();
        int count = categoryCount.get();
        for (int i = 0; i < count; i++) {
            result.add(new TemporalWeight(categoryPrototypes[i]));
        }
        return Collections.unmodifiableList(result);
    }

    /**
     * Get specific category (ARTAlgorithm interface).
     */
    @Override
    public WeightVector getCategory(int index) {
        if (index < 0 || index >= categoryCount.get()) {
            throw new IndexOutOfBoundsException("Category index " + index + " out of bounds for " + categoryCount.get() + " categories");
        }
        return new TemporalWeight(categoryPrototypes[index]);
    }

    /**
     * Get current parameters (VectorizedARTAlgorithm interface).
     */
    @Override
    public TemporalARTParameters getParameters() {
        return parameters;
    }

    /**
     * Get performance statistics (VectorizedARTAlgorithm interface).
     */
    @Override
    public VectorizedTemporalART.PerformanceStats getPerformanceStats() {
        return getStats();
    }

    /**
     * Reset performance tracking (VectorizedARTAlgorithm interface).
     */
    @Override
    public void resetPerformanceTracking() {
        processTime.set(0);
        learnTime.set(0);
        predictTime.set(0);
        patternsProcessed.set(0);
    }

    /**
     * Clear all categories and reset state (ARTAlgorithm interface).
     */
    @Override
    public void clear() {
        reset();
    }

    /**
     * Close and release resources (VectorizedARTAlgorithm interface).
     */
    @Override
    public void close() {
        shutdown();
    }

    /**
     * Get SIMD vector species length (VectorizedARTAlgorithm interface).
     */
    @Override
    public int getVectorSpeciesLength() {
        return SPECIES.length();
    }

    // === Internal Implementation Methods ===

    /**
     * Extract pattern data from Pattern interface.
     */
    private double[] extractPatternData(Pattern pattern) {
        if (pattern instanceof com.hellblazer.art.core.DenseVector dv) {
            return dv.data();
        }
        // Fallback for other Pattern implementations
        var data = new double[pattern.dimension()];
        for (int i = 0; i < pattern.dimension(); i++) {
            data[i] = pattern.get(i);
        }
        return data;
    }

    /**
     * Process temporal input with vectorized operations.
     */
    public int processInput(double[] input, double timestamp) {
        long startTime = System.nanoTime();

        // Store in working memory
        workingMemory.storeItem(input, timestamp);

        // Evolve dynamics (vectorized)
        workingMemory.evolveDynamics(parameters.getIntegrationTimeStep());

        // Get temporal pattern
        var temporalPattern = workingMemory.getTemporalPattern();

        // Process through masking field
        maskingField.processTemporalPattern(temporalPattern);

        // Find best matching category using raw pattern (parallel + vectorized)
        int winner = findBestCategoryParallelRaw(input);

        if (winner >= 0) {
            // Update category
            updateCategoryVectorized(winner, temporalPattern, input);
        } else if (categoryCount.get() < maxCategories) {
            // Create new category
            winner = createNewCategory(temporalPattern, input);
        }

        long elapsed = System.nanoTime() - startTime;
        processTime.addAndGet(elapsed);
        patternsProcessed.incrementAndGet();

        return winner;
    }

    /**
     * Find best matching category using parallel + vectorized search.
     */
    private int findBestCategoryParallel(TemporalPattern pattern) {
        int count = categoryCount.get();
        if (count == 0) return -1;

        // Prepare input for comparison
        var averagePattern = computeAveragePatternVectorized(pattern);

        // Parallel search with vectorization
        try {
            return computePool.submit(() ->
                IntStream.range(0, count)
                    .parallel()
                    .mapToObj(i -> {
                        double similarity = computeSimilarityVectorized(
                            categoryPrototypes[i],
                            averagePattern
                        );
                        double match = similarity * categoryWeights[i];
                        return new CategoryMatch(i, similarity, match);
                    })
                    .filter(m -> m.similarity >= parameters.getVigilance())
                    .max(Comparator.comparingDouble(m -> m.match))
                    .map(m -> m.index)
                    .orElse(-1)
            ).get();
        } catch (Exception e) {
            return -1;
        }
    }

    /**
     * Find best matching category using raw patterns (parallel + vectorized search).
     */
    private int findBestCategoryParallelRaw(double[] input) {
        int count = categoryCount.get();
        if (count == 0) return -1;

        // Parallel search with raw pattern matching
        try {
            return computePool.submit(() ->
                IntStream.range(0, count)
                    .parallel()
                    .mapToObj(i -> {
                        // Extract only relevant dimensions for comparison
                        var storedPattern = java.util.Arrays.copyOf(rawPatterns[i], input.length);
                        double similarity = computeSimilarityVectorized(
                            storedPattern,
                            input
                        );
                        double match = similarity * categoryWeights[i];
                        // Raw pattern matching for category assignment
                        return new CategoryMatch(i, similarity, match);
                    })
                    .filter(m -> m.similarity >= parameters.getVigilance())
                    .max(Comparator.comparingDouble(m -> m.match))
                    .map(m -> m.index)
                    .orElse(-1)
            ).get();
        } catch (Exception e) {
            return -1;
        }
    }

    /**
     * Compute average pattern with vectorization.
     */
    private double[] computeAveragePatternVectorized(TemporalPattern pattern) {
        var patterns = pattern.patterns();
        var weights = pattern.weights();

        if (patterns.isEmpty()) {
            return new double[inputDimension];
        }


        double[] result = new double[inputDimension];
        double totalWeight = 0.0;

        // Vectorized weighted average
        for (int p = 0; p < patterns.size(); p++) {
            var pat = patterns.get(p);
            double weight = weights.get(p);
            totalWeight += weight;

            // Use the actual pattern length for bounds checking
            var patternLength = Math.min(pat.length, inputDimension);
            int i = 0;
            int bound = SPECIES.loopBound(patternLength);

            // Only use vectorized operations if the pattern is large enough
            for (; i < bound; i += vectorLength) {
                var vPat = DoubleVector.fromArray(SPECIES, pat, i);
                var vResult = DoubleVector.fromArray(SPECIES, result, i);
                vResult = vResult.add(vPat.mul(weight));
                vResult.intoArray(result, i);
            }

            // Handle remaining elements with scalar operations
            for (; i < patternLength; i++) {
                result[i] += pat[i] * weight;
            }
        }

        // Normalize
        if (totalWeight > 0) {
            var vNorm = DoubleVector.broadcast(SPECIES, 1.0 / totalWeight);

            int i = 0;
            int bound = SPECIES.loopBound(inputDimension);

            for (; i < bound; i += vectorLength) {
                var vResult = DoubleVector.fromArray(SPECIES, result, i);
                vResult = vResult.mul(vNorm);
                vResult.intoArray(result, i);
            }

            for (; i < inputDimension; i++) {
                result[i] /= totalWeight;
            }
        }

        return result;
    }

    /**
     * Compute activation value for the result.
     */
    private double computeActivation(double[] prototype, double[] input) {
        return computeSimilarityVectorized(prototype, input);
    }

    /**
     * Vectorized cosine similarity computation.
     */
    private double computeSimilarityVectorized(double[] a, double[] b) {
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        // Use the minimum length to avoid bounds issues
        var comparisonLength = Math.min(Math.min(a.length, b.length), inputDimension);
        int i = 0;
        int bound = SPECIES.loopBound(comparisonLength);

        // Only use vectorized operations if arrays are large enough
        for (; i < bound; i += vectorLength) {
            var vA = DoubleVector.fromArray(SPECIES, a, i);
            var vB = DoubleVector.fromArray(SPECIES, b, i);

            dotProduct += vA.mul(vB).reduceLanes(VectorOperators.ADD);
            normA += vA.mul(vA).reduceLanes(VectorOperators.ADD);
            normB += vB.mul(vB).reduceLanes(VectorOperators.ADD);
        }

        // Handle remaining elements with scalar operations
        for (; i < comparisonLength; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        if (normA > 0 && normB > 0) {
            return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
        }
        return 0.0;
    }

    /**
     * Update category with vectorized learning.
     */
    private void updateCategoryVectorized(int index, TemporalPattern pattern, double[] rawInput) {
        long startTime = System.nanoTime();

        var category = categories.get(index);
        var avgPattern = computeAveragePatternVectorized(pattern);
        double learningRate = parameters.getLearningRate();

        // Vectorized prototype update
        var prototype = categoryPrototypes[index];

        // Use the minimum length to avoid bounds issues
        var updateLength = Math.min(Math.min(prototype.length, avgPattern.length), inputDimension);
        int i = 0;
        int bound = SPECIES.loopBound(updateLength);

        // Only use vectorized operations if arrays are large enough
        for (; i < bound; i += vectorLength) {
            var vProto = DoubleVector.fromArray(SPECIES, prototype, i);
            var vPattern = DoubleVector.fromArray(SPECIES, avgPattern, i);

            // w_new = (1-lr)*w_old + lr*pattern
            vProto = vProto.mul(1.0 - learningRate)
                          .add(vPattern.mul(learningRate));
            vProto.intoArray(prototype, i);
        }

        // Handle remaining elements with scalar operations
        for (; i < updateLength; i++) {
            prototype[i] = (1.0 - learningRate) * prototype[i] +
                          learningRate * avgPattern[i];
        }

        // Update weight
        categoryWeights[index] = Math.min(1.0,
            categoryWeights[index] + parameters.getWeightIncrease());

        // Update category metadata
        category.incrementAccessCount();
        category.updateLastAccess();

        // Store temporal pattern
        category.addTemporalPattern(pattern);

        long elapsed = System.nanoTime() - startTime;
        learnTime.addAndGet(elapsed);
    }

    /**
     * Create new category.
     */
    private int createNewCategory(TemporalPattern pattern, double[] rawInput) {
        int index = categoryCount.getAndIncrement();
        if (index >= maxCategories) {
            categoryCount.decrementAndGet();
            return -1;
        }

        var avgPattern = computeAveragePatternVectorized(pattern);
        // Only copy the minimum of avgPattern length and inputDimension to avoid bounds issues
        var copyLength = Math.min(avgPattern.length, inputDimension);
        System.arraycopy(avgPattern, 0, categoryPrototypes[index], 0, copyLength);

        // Store raw pattern for prediction matching
        var rawCopyLength = Math.min(rawInput.length, inputDimension);
        System.arraycopy(rawInput, 0, rawPatterns[index], 0, rawCopyLength);

        categoryWeights[index] = parameters.getInitialWeight();

        // Store both raw pattern and temporal prototype for different use cases

        var category = new TemporalCategory(
            avgPattern,
            1, // sequence length
            1, // temporal span
            currentTime
        );
        categories.put(index, category);

        return index;
    }

    /**
     * Predict category for input pattern (vectorized).
     * Uses raw pattern matching for individual pattern prediction.
     */
    public int predictCategory(double[] input) {
        long startTime = System.nanoTime();

        int count = categoryCount.get();
        if (count == 0) return -1;

        // Use raw pattern matching instead of temporal processing
        int winner = -1;
        double maxActivation = 0.0;

        // Compare input against stored raw patterns for accurate matching

        for (int i = 0; i < count; i++) {
            // Extract only the relevant dimensions for comparison
            var storedPattern = java.util.Arrays.copyOf(rawPatterns[i], input.length);
            double similarity = computeSimilarityVectorized(
                storedPattern,
                input
            );
            if (similarity >= parameters.getVigilance() &&
                similarity > maxActivation) {
                maxActivation = similarity;
                winner = i;
            }
        }

        long elapsed = System.nanoTime() - startTime;
        predictTime.addAndGet(elapsed);

        return winner;
    }

    /**
     * Apply temporal processing to input for prediction without affecting working memory state.
     */
    private double[] applyTemporalProcessingForPrediction(double[] input) {
        // DEBUG: Print input
        System.out.println("DEBUG: Input pattern: " + java.util.Arrays.toString(input));

        // Create temporary working memory for temporal processing
        var tempMemory = new VectorizedWorkingMemory(parameters.getMemoryParameters());

        // Apply same temporal processing as training
        tempMemory.storeItem(input, currentTime);
        tempMemory.evolveDynamics(parameters.getIntegrationTimeStep());
        var temporalPattern = tempMemory.getTemporalPattern();

        // DEBUG: Print temporal pattern info
        System.out.println("DEBUG: Temporal pattern size: " + temporalPattern.patterns().size());
        System.out.println("DEBUG: Temporal pattern weights: " + temporalPattern.weights());

        // Get processed pattern (same as training path)
        var processedInput = computeAveragePatternVectorized(temporalPattern);

        // DEBUG: Print processed input
        System.out.println("DEBUG: Processed input: " + java.util.Arrays.toString(processedInput));

        // tempMemory is automatically garbage collected
        return processedInput;
    }

    /**
     * Batch process multiple inputs in parallel.
     */
    public CompletableFuture<int[]> processBatch(List<double[]> inputs,
                                                  List<Double> timestamps) {
        return CompletableFuture.supplyAsync(() -> {
            int[] results = new int[inputs.size()];

            // Process in parallel chunks
            int chunkSize = Math.max(1, inputs.size() / computePool.getParallelism());
            List<CompletableFuture<Void>> futures = new ArrayList<>();

            for (int start = 0; start < inputs.size(); start += chunkSize) {
                int end = Math.min(start + chunkSize, inputs.size());
                int finalStart = start;

                futures.add(CompletableFuture.runAsync(() -> {
                    for (int i = finalStart; i < end; i++) {
                        results[i] = processInput(inputs.get(i), timestamps.get(i));
                    }
                }, computePool));
            }

            CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();
            return results;
        }, computePool);
    }

    /**
     * Get performance statistics.
     */
    public PerformanceStats getStats() {
        int patterns = patternsProcessed.get();
        if (patterns == 0) {
            return new PerformanceStats(0, 0, 0, 0, 0);
        }

        return new PerformanceStats(
            processTime.get() / (double) patterns / 1_000_000.0,  // ms per pattern
            learnTime.get() / 1_000_000_000.0,  // total seconds
            predictTime.get() / 1_000_000_000.0,  // total seconds
            patterns,
            categoryCount.get()
        );
    }

    public void reset() {
        categories.clear();
        categoryCount.set(0);

        Arrays.fill(categoryWeights, 0.0);
        for (var proto : categoryPrototypes) {
            Arrays.fill(proto, 0.0);
        }
        for (var raw : rawPatterns) {
            Arrays.fill(raw, 0.0);
        }
        Arrays.fill(activations, 0.0);
        Arrays.fill(matches, 0.0);

        workingMemory.reset();
        maskingField.reset();

        processTime.set(0);
        learnTime.set(0);
        predictTime.set(0);
        patternsProcessed.set(0);
    }

    public void shutdown() {
        computePool.shutdown();
        try {
            if (!computePool.awaitTermination(5, TimeUnit.SECONDS)) {
                computePool.shutdownNow();
            }
        } catch (InterruptedException e) {
            computePool.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }


    // Inner classes
    private record CategoryMatch(int index, double similarity, double match) {}

    public record PerformanceStats(
        double avgProcessTimeMs,
        double totalLearnTimeSeconds,
        double totalPredictTimeSeconds,
        int patternsProcessed,
        int categoryCount
    ) {}

    // Getters
    public TemporalARTState getState() {
        return new TemporalARTState(
            workingMemory.getState(),
            maskingField.getState(),
            new ArrayList<>(categories.values()),
            currentTime,
            true // learning enabled
        );
    }
}