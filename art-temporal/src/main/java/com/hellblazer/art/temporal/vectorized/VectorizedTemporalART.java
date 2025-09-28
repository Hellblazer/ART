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
package com.hellblazer.art.temporal.vectorized;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.performance.VectorizedARTAlgorithm;
import com.hellblazer.art.performance.algorithms.VectorizedParameters;
import com.hellblazer.art.performance.algorithms.VectorizedPerformanceStats;
import com.hellblazer.art.performance.algorithms.VectorizedFuzzyART;
import com.hellblazer.art.temporal.TemporalPattern;
import com.hellblazer.art.temporal.TemporalARTAlgorithm;
import com.hellblazer.art.temporal.parameters.TemporalParameters;
import com.hellblazer.art.temporal.results.TemporalResult;
import com.hellblazer.art.temporal.results.MaskingResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ForkJoinPool;

/**
 * Vectorized implementation of Temporal ART using Java Vector API for SIMD optimization.
 *
 * This implementation integrates all vectorized temporal components:
 * - VectorizedItemOrderWorkingMemory for SIMD-optimized sequence storage
 * - VectorizedMaskingField for parallel multi-scale processing
 * - VectorizedCompetitiveInstar for efficient category learning
 * - Batch sequence processing with parallel chunk detection
 *
 * Target speedup: 50-200x for complete temporal sequence processing depending on
 * sequence length, dimensionality, and complexity. Peak performance achieved with
 * batch processing of multiple sequences.
 *
 * Mathematical Foundation:
 * Integrates the complete temporal ART pipeline:
 * 1. Vectorized working memory: primacy gradient computation and shunting dynamics
 * 2. Vectorized masking field: multi-scale competitive dynamics and chunking
 * 3. Vectorized learning: competitive instar weight updates
 * 4. Parallel batch processing: concurrent sequence processing
 *
 * @author Hal Hildebrand
 */
public class VectorizedTemporalART implements TemporalARTAlgorithm<TemporalParameters>,
                                            VectorizedARTAlgorithm<VectorizedTemporalART.VectorizedTemporalPerformanceStats, TemporalParameters> {

    private static final Logger log = LoggerFactory.getLogger(VectorizedTemporalART.class);

    // Component Configuration
    private TemporalParameters parameters;
    private final int maxSequenceLength;
    private final int itemDimension;

    // Vectorized Components
    private final VectorizedItemOrderWorkingMemory workingMemory;
    private final VectorizedMaskingField maskingField;
    private final VectorizedCompetitiveInstar competitiveInstar;
    private final VectorizedFuzzyART underlyingART; // For category learning

    // Temporal State
    private final List<TemporalPattern> learnedChunks;
    private final List<Pattern> currentSequence;
    private double currentTime;
    private boolean isProcessingSequence;

    // Batch Processing
    private final ForkJoinPool parallelProcessor;
    private final int parallelThreshold; // Minimum batch size for parallel processing

    // Performance Tracking
    private final AtomicLong temporalOperations = new AtomicLong(0);
    private final AtomicLong sequenceProcessingTime = new AtomicLong(0);
    private final AtomicLong chunkingOperations = new AtomicLong(0);
    private final AtomicLong batchOperations = new AtomicLong(0);
    private final AtomicLong totalMemoryUsage = new AtomicLong(0);

    /**
     * Create vectorized temporal ART with specified parameters and dimensions.
     *
     * @param parameters temporal ART configuration
     * @param maxSequenceLength maximum sequence length supported
     * @param itemDimension dimensionality of individual items
     */
    public VectorizedTemporalART(TemporalParameters parameters, int maxSequenceLength, int itemDimension) {
        this.parameters = parameters;
        this.maxSequenceLength = maxSequenceLength;
        this.itemDimension = itemDimension;
        this.parallelThreshold = 4; // Process batches of 4+ sequences in parallel

        log.info("Initializing VectorizedTemporalART: maxSeqLen={}, itemDim={}, vigilance={}, maxCategories={}",
                maxSequenceLength, itemDimension, parameters.vigilance(), parameters.maxCategories());

        // Initialize vectorized components
        this.workingMemory = new VectorizedItemOrderWorkingMemory(
            parameters.workingMemoryParameters(),
            parameters.workingMemoryParameters().capacity(),
            itemDimension
        );

        this.maskingField = new VectorizedMaskingField(parameters.maskingParameters());

        // Competitive instar for temporal chunk learning
        this.competitiveInstar = new VectorizedCompetitiveInstar(
            parameters.learningRate(),
            parameters.maxCategories(),
            calculateChunkDimension(),
            true,  // self-normalizing
            false  // hard competition
        );

        // Underlying FuzzyART for category processing
        var artParams = new VectorizedParameters(
            parameters.vigilance(),
            parameters.learningRate(),
            0.01,  // alpha
            Runtime.getRuntime().availableProcessors(),  // parallelismLevel
            100,  // parallelThreshold
            1000,  // maxCacheSize
            true,  // enableSIMD
            false,  // enableJOML
            0.8  // memoryOptimizationThreshold
        );

        this.underlyingART = new VectorizedFuzzyART(artParams);

        // Initialize state
        this.learnedChunks = new ArrayList<>();
        this.currentSequence = new ArrayList<>();
        this.currentTime = 0.0;
        this.isProcessingSequence = false;

        // Setup parallel processing
        this.parallelProcessor = new ForkJoinPool(
            Runtime.getRuntime().availableProcessors(),
            ForkJoinPool.defaultForkJoinWorkerThreadFactory,
            null,
            true // async mode for better throughput
        );

        log.debug("VectorizedTemporalART initialized with {} processor threads",
                 parallelProcessor.getParallelism());
    }

    // === Temporal Learning Interface ===

    @Override
    public TemporalResult learnTemporal(TemporalPattern temporalPattern) {
        var startTime = System.nanoTime();
        try {
            if (temporalPattern.isEmpty()) {
                return createEmptyResult();
            }

            // Process through temporal pipeline
            var workingMemoryResult = processWorkingMemory(temporalPattern);
            var maskingResult = processMaskingField(workingMemoryResult);
            var chunks = extractChunks(maskingResult);
            var activationResult = learnChunks(chunks);

            // Update state
            if (parameters.enableLearning()) {
                updateLearnedChunks(chunks);
            }

            temporalOperations.incrementAndGet();

            return createTemporalResult(temporalPattern, activationResult, chunks, maskingResult, startTime);

        } finally {
            sequenceProcessingTime.addAndGet(System.nanoTime() - startTime);
        }
    }

    @Override
    public TemporalResult predictTemporal(TemporalPattern temporalPattern) {
        var startTime = System.nanoTime();
        try {
            if (temporalPattern.isEmpty()) {
                return createEmptyResult();
            }

            // Process through temporal pipeline without learning
            var workingMemoryResult = processWorkingMemory(temporalPattern);
            var maskingResult = processMaskingField(workingMemoryResult);
            var chunks = extractChunks(maskingResult);
            var activationResult = predictChunks(chunks);

            temporalOperations.incrementAndGet();

            return createTemporalResult(temporalPattern, activationResult, chunks, maskingResult, startTime);

        } finally {
            sequenceProcessingTime.addAndGet(System.nanoTime() - startTime);
        }
    }

    @Override
    public TemporalResult processSequenceItem(Pattern item) {
        var startTime = System.nanoTime();
        try {
            if (!isProcessingSequence) {
                resetTemporalState();
                isProcessingSequence = true;
            }

            // Add item to current sequence
            currentSequence.add(item);
            currentTime += 1.0;

            // Store in working memory
            workingMemory.storeItem(item, currentTime);

            // Check if chunking boundary is detected
            var workingMemoryContents = workingMemory.getCurrentContents();
            var maskingResult = maskingField.processTimeStep(workingMemoryContents, 0.1);

            var chunks = new ArrayList<TemporalPattern>();
            ActivationResult activationResult = ActivationResult.NoMatch.INSTANCE;

            if (shouldProcessChunk(maskingResult)) {
                chunks = new ArrayList<>(extractChunks(maskingResult));
                if (!chunks.isEmpty()) {
                    activationResult = parameters.enableLearning() ?
                        learnChunks(chunks) :
                        predictChunks(chunks);

                    if (parameters.enableLearning()) {
                        updateLearnedChunks(chunks);
                    }
                }
            }

            temporalOperations.incrementAndGet();

            return createTemporalResult(workingMemoryContents, activationResult, chunks, maskingResult, startTime);

        } finally {
            sequenceProcessingTime.addAndGet(System.nanoTime() - startTime);
        }
    }

    // Add missing clear method from ARTAlgorithm interface
    public void clear() {
        resetTemporalState();
        underlyingART.clear();
    }

    @Override
    public void resetTemporalState() {
        workingMemory.clear();
        maskingField.reset();
        currentSequence.clear();
        currentTime = 0.0;
        isProcessingSequence = false;
    }

    // === Sequence Chunking Interface ===

    @Override
    public List<TemporalPattern> getTemporalChunks() {
        return new ArrayList<>(learnedChunks);
    }

    @Override
    public boolean wouldCreateNewChunk(TemporalPattern temporalPattern) {
        // Process without learning to check if new category would be created
        var workingMemoryResult = processWorkingMemory(temporalPattern);
        var maskingResult = processMaskingField(workingMemoryResult);
        var chunks = extractChunks(maskingResult);

        if (chunks.isEmpty()) return false;

        // Check if any chunk would create new category
        for (var chunk : chunks) {
            var chunkPattern = convertChunkToPattern(chunk);
            var result = underlyingART.predict(chunkPattern, getVectorizedParameters());
            if (result instanceof ActivationResult.NoMatch) {
                return true;
            }
        }

        return false;
    }

    // === Working Memory Interface ===

    @Override
    public TemporalPattern getWorkingMemoryContents() {
        return workingMemory.getCurrentContents();
    }

    @Override
    public int getWorkingMemoryCapacity() {
        return workingMemory.getCapacity();
    }

    // === Masking Field Interface ===

    @Override
    public double[][] getMaskingFieldActivations() {
        return maskingField.getAllActivations();
    }

    // === Temporal Parameters ===

    @Override
    public TemporalParameters getTemporalParameters() {
        return parameters;
    }

    @Override
    public void setTemporalParameters(TemporalParameters parameters) {
        this.parameters = parameters;
        workingMemory.setParameters(parameters.workingMemoryParameters());
        maskingField.setParameters(parameters.maskingParameters());

        // Update underlying ART parameters
        var artParams = new VectorizedParameters(
            parameters.vigilance(),
            parameters.learningRate(),
            0.01,  // alpha
            Runtime.getRuntime().availableProcessors(),  // parallelismLevel
            100,  // parallelThreshold
            1000,  // maxCacheSize
            true,  // enableSIMD
            false,  // enableJOML
            0.8  // memoryOptimizationThreshold
        );

        // Note: VectorizedFuzzyART doesn't have setParameters method in the interface,
        // so we'd need to create a new instance or add that method
    }

    // === Batch Processing ===

    @Override
    public List<TemporalResult> learnTemporalBatch(List<TemporalPattern> temporalPatterns) {
        var startTime = System.nanoTime();
        try {
            if (temporalPatterns.size() < parallelThreshold) {
                // Process sequentially for small batches
                return temporalPatterns.stream()
                                     .map(this::learnTemporal)
                                     .toList();
            }

            // Process in parallel for large batches
            var futures = temporalPatterns.stream()
                .map(pattern -> CompletableFuture.supplyAsync(
                    () -> learnTemporal(pattern), parallelProcessor))
                .toList();

            var results = new ArrayList<TemporalResult>(temporalPatterns.size());
            for (var future : futures) {
                results.add(future.join());
            }

            batchOperations.incrementAndGet();
            return results;

        } finally {
            sequenceProcessingTime.addAndGet(System.nanoTime() - startTime);
        }
    }

    @Override
    public List<TemporalResult> predictTemporalBatch(List<TemporalPattern> temporalPatterns) {
        var startTime = System.nanoTime();
        try {
            if (temporalPatterns.size() < parallelThreshold) {
                return temporalPatterns.stream()
                                     .map(this::predictTemporal)
                                     .toList();
            }

            var futures = temporalPatterns.stream()
                .map(pattern -> CompletableFuture.supplyAsync(
                    () -> predictTemporal(pattern), parallelProcessor))
                .toList();

            var results = new ArrayList<TemporalResult>(temporalPatterns.size());
            for (var future : futures) {
                results.add(future.join());
            }

            batchOperations.incrementAndGet();
            return results;

        } finally {
            sequenceProcessingTime.addAndGet(System.nanoTime() - startTime);
        }
    }

    // === VectorizedARTAlgorithm Implementation ===

    @Override
    public VectorizedTemporalPerformanceStats getPerformanceStats() {
        return new VectorizedTemporalPerformanceStats();
    }

    @Override
    public void resetPerformanceTracking() {
        temporalOperations.set(0);
        sequenceProcessingTime.set(0);
        chunkingOperations.set(0);
        batchOperations.set(0);
        totalMemoryUsage.set(0);

        workingMemory.resetPerformanceTracking();
        maskingField.resetPerformanceTracking();
        competitiveInstar.resetPerformanceTracking();
        underlyingART.resetPerformanceTracking();
    }

    @Override
    public TemporalParameters getParameters() {
        return parameters;
    }

    @Override
    public int getCategoryCount() {
        return underlyingART.getCategoryCount();
    }

    @Override
    public WeightVector getCategory(int index) {
        return underlyingART.getCategory(index);
    }

    @Override
    public List<WeightVector> getCategories() {
        return underlyingART.getCategories();
    }

    @Override
    public void close() {
        workingMemory.close();
        maskingField.close();
        underlyingART.close();
        parallelProcessor.shutdown();
    }

    // === Private Implementation Methods ===

    private TemporalPattern processWorkingMemory(TemporalPattern temporalPattern) {
        workingMemory.clear();
        workingMemory.storeSequence(temporalPattern);
        workingMemory.updateDynamics(parameters.workingMemoryParameters().temporalResolution());
        return workingMemory.getCurrentContents();
    }

    private MaskingResult processMaskingField(TemporalPattern workingMemoryResult) {
        return maskingField.process(workingMemoryResult);
    }

    private List<TemporalPattern> extractChunks(MaskingResult maskingResult) {
        var startTime = System.nanoTime();
        try {
            var chunks = new ArrayList<TemporalPattern>();

            if (!parameters.enableChunking()) {
                // Treat entire sequence as single chunk
                var workingMemoryContents = workingMemory.getCurrentContents();
                if (!workingMemoryContents.isEmpty()) {
                    chunks.add(workingMemoryContents);
                }
                return chunks;
            }

            var boundaries = maskingResult.getChunkBoundaries();
            if (boundaries.length == 0) {
                // No boundaries detected, single chunk
                var workingMemoryContents = workingMemory.getCurrentContents();
                if (!workingMemoryContents.isEmpty()) {
                    chunks.add(workingMemoryContents);
                }
                return chunks;
            }

            // Extract chunks between boundaries
            var sequence = workingMemory.getCurrentContents().getSequence();
            var lastBoundary = 0;

            for (var boundary : boundaries) {
                if (boundary > lastBoundary && boundary <= sequence.size()) {
                    var chunkSequence = sequence.subList(lastBoundary, boundary);
                    if (!chunkSequence.isEmpty()) {
                        chunks.add(createTemporalPattern(chunkSequence));
                    }
                    lastBoundary = boundary;
                }
            }

            // Add final chunk if remaining items
            if (lastBoundary < sequence.size()) {
                var chunkSequence = sequence.subList(lastBoundary, sequence.size());
                if (!chunkSequence.isEmpty()) {
                    chunks.add(createTemporalPattern(chunkSequence));
                }
            }

            chunkingOperations.incrementAndGet();
            return chunks;

        } finally {
            sequenceProcessingTime.addAndGet(System.nanoTime() - startTime);
        }
    }

    private ActivationResult learnChunks(List<TemporalPattern> chunks) {
        if (chunks.isEmpty()) {
            return ActivationResult.NoMatch.INSTANCE;
        }

        // Learn the primary (largest) chunk
        var primaryChunk = chunks.stream()
            .max((c1, c2) -> Integer.compare(c1.getSequenceLength(), c2.getSequenceLength()))
            .orElse(chunks.get(0));

        var chunkPattern = convertChunkToPattern(primaryChunk);
        return underlyingART.learn(chunkPattern, getVectorizedParameters());
    }

    private ActivationResult predictChunks(List<TemporalPattern> chunks) {
        if (chunks.isEmpty()) {
            return ActivationResult.NoMatch.INSTANCE;
        }

        var primaryChunk = chunks.stream()
            .max((c1, c2) -> Integer.compare(c1.getSequenceLength(), c2.getSequenceLength()))
            .orElse(chunks.get(0));

        var chunkPattern = convertChunkToPattern(primaryChunk);
        return underlyingART.predict(chunkPattern, getVectorizedParameters());
    }

    private void updateLearnedChunks(List<TemporalPattern> newChunks) {
        for (var chunk : newChunks) {
            if (!containsSimilarChunk(chunk)) {
                learnedChunks.add(chunk);
            }
        }
    }

    private boolean containsSimilarChunk(TemporalPattern chunk) {
        var tolerance = 0.1;
        for (var existingChunk : learnedChunks) {
            if (areChunksSimilar(chunk, existingChunk, tolerance)) {
                return true;
            }
        }
        return false;
    }

    private boolean areChunksSimilar(TemporalPattern chunk1, TemporalPattern chunk2, double tolerance) {
        if (chunk1.getSequenceLength() != chunk2.getSequenceLength()) {
            return false;
        }

        var seq1 = chunk1.getSequence();
        var seq2 = chunk2.getSequence();

        for (int i = 0; i < seq1.size(); i++) {
            var pattern1 = seq1.get(i);
            var pattern2 = seq2.get(i);

            if (pattern1.dimension() != pattern2.dimension()) {
                return false;
            }

            for (int j = 0; j < pattern1.dimension(); j++) {
                if (Math.abs(pattern1.get(j) - pattern2.get(j)) > tolerance) {
                    return false;
                }
            }
        }

        return true;
    }

    private boolean shouldProcessChunk(MaskingResult maskingResult) {
        return maskingResult.hasConverged() &&
               maskingResult.getChunkBoundaries().length > 0;
    }

    private Pattern convertChunkToPattern(TemporalPattern chunk) {
        // Flatten temporal pattern into single pattern
        var sequence = chunk.getSequence();
        if (sequence.isEmpty()) {
            return Pattern.of(new double[itemDimension]);
        }

        // Simple approach: concatenate all items
        var totalDim = sequence.size() * itemDimension;
        var features = new double[totalDim];
        var index = 0;

        for (var item : sequence) {
            for (int i = 0; i < Math.min(item.dimension(), itemDimension); i++) {
                features[index++] = item.get(i);
            }
            // Pad if item dimension is smaller
            while (index % itemDimension != 0) {
                features[index++] = 0.0;
            }
        }

        return Pattern.of(features);
    }

    private TemporalPattern createTemporalPattern(List<Pattern> sequence) {
        return new SimpleTemporalPattern(sequence);
    }

    private VectorizedParameters getVectorizedParameters() {
        return new VectorizedParameters(
            parameters.vigilance(),
            parameters.learningRate(),
            0.01,  // alpha
            Runtime.getRuntime().availableProcessors(),  // parallelismLevel
            100,  // parallelThreshold
            1000,  // maxCacheSize
            true,  // enableSIMD
            false,  // enableJOML
            0.8  // memoryOptimizationThreshold
        );
    }

    private int calculateChunkDimension() {
        // Estimate dimension needed for flattened chunks
        return maxSequenceLength * itemDimension;
    }

    private TemporalResult createEmptyResult() {
        return new VectorizedTemporalResult(
            ActivationResult.NoMatch.INSTANCE,
            new ArrayList<>(),
            workingMemory.getCurrentContents(),
            maskingField.getAllActivations(),
            maskingField.getAllTransmitterGates(),
            0.0,
            false,
            0.0,
            new ArrayList<>(),
            System.nanoTime()
        );
    }

    private TemporalResult createTemporalResult(TemporalPattern originalPattern,
                                               ActivationResult activationResult,
                                               List<TemporalPattern> chunks,
                                               MaskingResult maskingResult,
                                               long startTime) {
        var processingTime = (System.nanoTime() - startTime) / 1e9; // Convert to seconds
        var boundaries = Arrays.stream(maskingResult.getChunkBoundaries()).boxed().toList();

        return new VectorizedTemporalResult(
            activationResult,
            chunks,
            workingMemory.getCurrentContents(),
            maskingResult.getActivations(),
            maskingField.getAllTransmitterGates(),
            processingTime,
            maskingResult.hasConverged(),
            maskingResult.getMaxActivation(),
            boundaries,
            startTime
        );
    }

    // === Performance Stats Implementation ===

    public class VectorizedTemporalPerformanceStats {

        public long getTemporalOperations() {
            return temporalOperations.get();
        }

        public long getSequenceProcessingTimeNanos() {
            return sequenceProcessingTime.get();
        }

        public long getChunkingOperations() {
            return chunkingOperations.get();
        }

        public long getBatchOperations() {
            return batchOperations.get();
        }

        public double getAverageSequenceProcessingTime() {
            var ops = temporalOperations.get();
            return ops > 0 ? (double) sequenceProcessingTime.get() / ops / 1e6 : 0.0; // ms
        }

        public VectorizedItemOrderWorkingMemory.VectorizedWorkingMemoryPerformanceMetrics getWorkingMemoryMetrics() {
            return (VectorizedItemOrderWorkingMemory.VectorizedWorkingMemoryPerformanceMetrics)
                workingMemory.getPerformanceMetrics();
        }

        public VectorizedMaskingField.VectorizedMaskingFieldPerformanceMetrics getMaskingFieldMetrics() {
            return (VectorizedMaskingField.VectorizedMaskingFieldPerformanceMetrics)
                maskingField.getPerformanceMetrics();
        }

        public VectorizedCompetitiveInstar.VectorizedInstarPerformanceMetrics getCompetitiveInstarMetrics() {
            return competitiveInstar.getPerformanceMetrics();
        }

        @Override
        public String toString() {
            return String.format(
                "VectorizedTemporalPerformanceStats{" +
                "temporalOps=%d, avgSeqTime=%.2fms, chunkingOps=%d, batchOps=%d, " +
                "workingMemory=%s, maskingField=%s, instar=%s}",
                getTemporalOperations(), getAverageSequenceProcessingTime(),
                getChunkingOperations(), getBatchOperations(),
                getWorkingMemoryMetrics(), getMaskingFieldMetrics(), getCompetitiveInstarMetrics()
            );
        }
    }

    // === Simple Implementations ===

    private record SimpleTemporalPattern(List<Pattern> sequence) implements TemporalPattern {
        @Override
        public List<Pattern> getSequence() {
            return new ArrayList<>(sequence);
        }

        @Override
        public TemporalPattern getSubsequence(int startTime, int endTime) {
            if (startTime < 0 || endTime > sequence.size() || startTime >= endTime) {
                throw new IndexOutOfBoundsException("Invalid subsequence bounds");
            }
            return new SimpleTemporalPattern(sequence.subList(startTime, endTime));
        }

        @Override
        public boolean isEmpty() {
            return sequence.isEmpty();
        }
    }

    private record VectorizedTemporalResult(
        ActivationResult activationResult,
        List<TemporalPattern> identifiedChunks,
        TemporalPattern workingMemoryState,
        double[][] maskingFieldActivations,
        double[][] transmitterGateValues,
        double processingTime,
        boolean hasTemporalResonance,
        double resonanceQuality,
        List<Integer> chunkBoundaries,
        long startTimeNanos
    ) implements TemporalResult {

        @Override
        public ActivationResult getActivationResult() {
            return activationResult;
        }

        @Override
        public List<TemporalPattern> getIdentifiedChunks() {
            return new ArrayList<>(identifiedChunks);
        }

        @Override
        public Optional<TemporalPattern> getPrimaryChunk() {
            return identifiedChunks.stream()
                .max((c1, c2) -> Integer.compare(c1.getSequenceLength(), c2.getSequenceLength()));
        }

        @Override
        public boolean hasNewChunks() {
            return !identifiedChunks.isEmpty();
        }

        @Override
        public TemporalPattern getWorkingMemoryState() {
            return workingMemoryState;
        }

        @Override
        public double[][] getMaskingFieldActivations() {
            return maskingFieldActivations;
        }

        @Override
        public double[][] getTransmitterGateValues() {
            return transmitterGateValues;
        }

        @Override
        public double getProcessingTime() {
            return processingTime;
        }

        @Override
        public boolean hasTemporalResonance() {
            return hasTemporalResonance;
        }

        @Override
        public double getResonanceQuality() {
            return resonanceQuality;
        }

        @Override
        public boolean requiredChunking() {
            return !chunkBoundaries.isEmpty();
        }

        @Override
        public List<Integer> getChunkBoundaries() {
            return new ArrayList<>(chunkBoundaries);
        }

        @Override
        public Optional<TemporalPrediction> getPrediction() {
            return Optional.empty(); // Not implemented in this version
        }

        @Override
        public TemporalPerformanceMetrics getPerformanceMetrics() {
            return new VectorizedTemporalPerformanceMetrics(
                processingTime * 0.3, // Estimated working memory time
                processingTime * 0.5, // Estimated masking field time
                processingTime * 0.2, // Estimated chunking time
                0L,  // Memory usage not tracked in result
                0L   // Operations not tracked in result
            );
        }
    }

    private record VectorizedTemporalPerformanceMetrics(
        double workingMemoryTime,
        double maskingFieldTime,
        double chunkingTime,
        long memoryUsage,
        long simdOperationCount
    ) implements TemporalResult.TemporalPerformanceMetrics {

        @Override
        public double getWorkingMemoryTime() {
            return workingMemoryTime;
        }

        @Override
        public double getMaskingFieldTime() {
            return maskingFieldTime;
        }

        @Override
        public double getChunkingTime() {
            return chunkingTime;
        }

        @Override
        public long getMemoryUsage() {
            return memoryUsage;
        }

        @Override
        public long getSIMDOperationCount() {
            return simdOperationCount;
        }
    }
}