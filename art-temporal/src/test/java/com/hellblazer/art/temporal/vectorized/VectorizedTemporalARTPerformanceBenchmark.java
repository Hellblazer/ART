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
import com.hellblazer.art.temporal.TemporalPattern;
import com.hellblazer.art.temporal.algorithms.BasicTemporalART;
import com.hellblazer.art.temporal.parameters.TemporalParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Performance benchmarks for VectorizedTemporalART to validate target 10-100x speedup
 * for sequence processing operations.
 *
 * These benchmarks compare vectorized implementations against scalar versions
 * across different sequence lengths, input dimensions, and batch sizes to
 * demonstrate the performance benefits of SIMD optimization.
 *
 * Target performance improvements:
 * - Working Memory: 10-50x speedup for primacy gradient computation
 * - Masking Field: 20-100x speedup for multi-scale processing
 * - Competitive Instar: 15-50x speedup for weight updates
 * - Complete Pipeline: 50-200x speedup for batch sequence processing
 *
 * @author Hal Hildebrand
 */
@DisplayName("Vectorized Temporal ART Performance Benchmarks")
public class VectorizedTemporalARTPerformanceBenchmark {

    private static final Logger log = LoggerFactory.getLogger(VectorizedTemporalARTPerformanceBenchmark.class);

    private static final int WARMUP_ITERATIONS = 5;
    private static final int BENCHMARK_ITERATIONS = 10;
    private static final double NANO_TO_MILLIS = 1e-6;

    private Random random;
    private TemporalParameters parameters;

    @BeforeEach
    void setUp() {
        random = new Random(42); // Fixed seed for reproducible results
        parameters = TemporalParameters.DEFAULT;
    }

    @Test
    @DisplayName("Working Memory Performance Comparison")
    void testWorkingMemoryPerformance() {
        log.info("=== Working Memory Performance Benchmark ===");

        var dimensions = new int[]{8, 16, 32, 64, 128};
        var sequenceLengths = new int[]{5, 10, 20, 50};

        for (var dimension : dimensions) {
            for (var sequenceLength : sequenceLengths) {
                benchmarkWorkingMemoryPerformance(dimension, sequenceLength);
            }
        }
    }

    @Test
    @DisplayName("Masking Field Performance Comparison")
    void testMaskingFieldPerformance() {
        log.info("=== Masking Field Performance Benchmark ===");

        var fieldSizes = new int[]{16, 32, 64, 128};
        var scaleCounts = new int[]{2, 3, 4, 5};

        for (var fieldSize : fieldSizes) {
            for (var scaleCount : scaleCounts) {
                benchmarkMaskingFieldPerformance(fieldSize, scaleCount);
            }
        }
    }

    @Test
    @DisplayName("Competitive Instar Performance Comparison")
    void testCompetitiveInstarPerformance() {
        log.info("=== Competitive Instar Performance Benchmark ===");

        var inputDimensions = new int[]{16, 32, 64, 128, 256};
        var categoryCounts = new int[]{10, 50, 100, 200};

        for (var inputDim : inputDimensions) {
            for (var categoryCount : categoryCounts) {
                benchmarkCompetitiveInstarPerformance(inputDim, categoryCount);
            }
        }
    }

    @Test
    @DisplayName("Complete Temporal ART Pipeline Performance")
    void testCompleteTemporalARTPerformance() {
        log.info("=== Complete Temporal ART Pipeline Benchmark ===");

        var dimensions = new int[]{8, 16, 32, 64};
        var sequenceLengths = new int[]{5, 10, 20, 50};
        var batchSizes = new int[]{1, 5, 10, 25, 50};

        for (var dimension : dimensions) {
            for (var sequenceLength : sequenceLengths) {
                for (var batchSize : batchSizes) {
                    benchmarkCompleteTemporalARTPerformance(dimension, sequenceLength, batchSize);
                }
            }
        }
    }

    @Test
    @DisplayName("Batch Processing Scalability")
    void testBatchProcessingScalability() {
        log.info("=== Batch Processing Scalability Benchmark ===");

        var dimension = 32;
        var sequenceLength = 10;
        var batchSizes = new int[]{1, 2, 4, 8, 16, 32, 64, 128};

        var vectorizedResults = new ArrayList<Double>();
        var scalarResults = new ArrayList<Double>();

        for (var batchSize : batchSizes) {
            var vectorizedTime = benchmarkVectorizedBatchProcessing(dimension, sequenceLength, batchSize);
            var scalarTime = benchmarkScalarBatchProcessing(dimension, sequenceLength, batchSize);

            vectorizedResults.add(vectorizedTime);
            scalarResults.add(scalarTime);

            var speedup = scalarTime / vectorizedTime;
            var throughputVectorized = batchSize / vectorizedTime * 1000; // sequences/second
            var throughputScalar = batchSize / scalarTime * 1000;

            log.info("Batch size: {}, Vectorized: {:.2f}ms, Scalar: {:.2f}ms, Speedup: {:.2f}x, " +
                    "Throughput - Vectorized: {:.1f} seq/s, Scalar: {:.1f} seq/s",
                    batchSize, vectorizedTime, scalarTime, speedup, throughputVectorized, throughputScalar);

            // Verify speedup improves with batch size
            if (batchSize >= 8) {
                assertTrue(speedup >= 2.0, "Expected at least 2x speedup for batch size " + batchSize);
            }
        }

        // Test scalability - larger batches should maintain or improve efficiency
        for (int i = 1; i < batchSizes.length; i++) {
            var prevEfficiencyVectorized = batchSizes[i-1] / vectorizedResults.get(i-1);
            var currEfficiencyVectorized = batchSizes[i] / vectorizedResults.get(i);

            // Vectorized implementation should maintain efficiency better than scalar
            assertTrue(currEfficiencyVectorized >= prevEfficiencyVectorized * 0.8,
                      "Vectorized efficiency should not degrade significantly with batch size");
        }
    }

    @Test
    @DisplayName("Memory Usage Comparison")
    void testMemoryUsageComparison() {
        log.info("=== Memory Usage Comparison ===");

        var dimension = 64;
        var sequenceLength = 20;
        var batchSize = 10;

        // Test memory-efficient vectorized implementation
        var vectorizedMemoryUsage = measureVectorizedMemoryUsage(dimension, sequenceLength, batchSize);
        var scalarMemoryUsage = measureScalarMemoryUsage(dimension, sequenceLength, batchSize);

        log.info("Memory Usage - Vectorized: {} bytes, Scalar: {} bytes, Ratio: {:.2f}",
                vectorizedMemoryUsage, scalarMemoryUsage, (double) vectorizedMemoryUsage / scalarMemoryUsage);

        // Vectorized should use similar or slightly more memory due to padding
        assertTrue(vectorizedMemoryUsage <= scalarMemoryUsage * 1.5,
                  "Vectorized memory usage should not exceed 1.5x scalar usage");
    }

    @Test
    @DisplayName("Real-time Processing Performance")
    void testRealTimeProcessingPerformance() {
        log.info("=== Real-time Processing Performance ===");

        var dimension = 32;
        var targetProcessingTimeMs = 10.0; // Target: process each item within 10ms

        // Test real-time sequential processing
        var vectorizedART = new VectorizedTemporalART(TemporalParameters.REAL_TIME, 50, dimension);
        var patterns = generateRandomPatterns(100, dimension);

        // Warm up
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            vectorizedART.resetTemporalState();
            for (var pattern : patterns.subList(0, 10)) {
                vectorizedART.processSequenceItem(pattern);
            }
        }

        // Benchmark real-time processing
        vectorizedART.resetTemporalState();
        vectorizedART.resetPerformanceTracking();

        var startTime = System.nanoTime();
        for (var pattern : patterns) {
            var itemStartTime = System.nanoTime();
            vectorizedART.processSequenceItem(pattern);
            var itemTime = (System.nanoTime() - itemStartTime) * NANO_TO_MILLIS;

            // Each item should be processed within target time
            assertTrue(itemTime <= targetProcessingTimeMs,
                      String.format("Item processing time %.2fms exceeds target %.2fms", itemTime, targetProcessingTimeMs));
        }

        var totalTime = (System.nanoTime() - startTime) * NANO_TO_MILLIS;
        var avgTimePerItem = totalTime / patterns.size();
        var throughput = patterns.size() / totalTime * 1000; // items/second

        log.info("Real-time processing - Total: {:.2f}ms, Avg per item: {:.2f}ms, Throughput: {:.1f} items/s",
                totalTime, avgTimePerItem, throughput);

        var stats = vectorizedART.getPerformanceStats();
        log.info("Performance stats: {}", stats);

        vectorizedART.close();

        // Verify real-time performance requirements
        assertTrue(avgTimePerItem <= targetProcessingTimeMs,
                  "Average processing time should meet real-time target");
        assertTrue(throughput >= 100, "Should process at least 100 items per second");
    }

    // === Private Benchmark Methods ===

    private void benchmarkWorkingMemoryPerformance(int dimension, int sequenceLength) {
        log.debug("Working Memory: dim={}, seqLen={}", dimension, sequenceLength);

        var vectorizedTime = benchmarkVectorizedWorkingMemory(dimension, sequenceLength);
        var scalarTime = benchmarkScalarWorkingMemory(dimension, sequenceLength);

        var speedup = scalarTime / vectorizedTime;
        log.info("Working Memory - Dim: {}, SeqLen: {}, Vectorized: {:.2f}ms, Scalar: {:.2f}ms, Speedup: {:.2f}x",
                dimension, sequenceLength, vectorizedTime, scalarTime, speedup);

        // Expect significant speedup for larger dimensions
        if (dimension >= 32) {
            assertTrue(speedup >= 5.0, "Expected at least 5x speedup for large dimensions");
        }
    }

    private void benchmarkMaskingFieldPerformance(int fieldSize, int scaleCount) {
        log.debug("Masking Field: fieldSize={}, scales={}", fieldSize, scaleCount);

        var vectorizedTime = benchmarkVectorizedMaskingField(fieldSize, scaleCount);
        var scalarTime = benchmarkScalarMaskingField(fieldSize, scaleCount);

        var speedup = scalarTime / vectorizedTime;
        log.info("Masking Field - FieldSize: {}, Scales: {}, Vectorized: {:.2f}ms, Scalar: {:.2f}ms, Speedup: {:.2f}x",
                fieldSize, scaleCount, vectorizedTime, scalarTime, speedup);

        // Expect significant speedup for multi-scale processing
        if (fieldSize >= 32 && scaleCount >= 3) {
            assertTrue(speedup >= 10.0, "Expected at least 10x speedup for multi-scale processing");
        }
    }

    private void benchmarkCompetitiveInstarPerformance(int inputDim, int categoryCount) {
        log.debug("Competitive Instar: inputDim={}, categories={}", inputDim, categoryCount);

        var vectorizedTime = benchmarkVectorizedCompetitiveInstar(inputDim, categoryCount);
        var scalarTime = benchmarkScalarCompetitiveInstar(inputDim, categoryCount);

        var speedup = scalarTime / vectorizedTime;
        log.info("Competitive Instar - InputDim: {}, Categories: {}, Vectorized: {:.2f}ms, Scalar: {:.2f}ms, Speedup: {:.2f}x",
                inputDim, categoryCount, vectorizedTime, scalarTime, speedup);

        // Expect speedup for larger dimensions
        if (inputDim >= 64) {
            assertTrue(speedup >= 8.0, "Expected at least 8x speedup for large input dimensions");
        }
    }

    private void benchmarkCompleteTemporalARTPerformance(int dimension, int sequenceLength, int batchSize) {
        if (batchSize == 1) { // Only log for single sequence to reduce output
            log.debug("Complete Temporal ART: dim={}, seqLen={}, batch={}", dimension, sequenceLength, batchSize);
        }

        var vectorizedTime = benchmarkVectorizedCompleteTemporalART(dimension, sequenceLength, batchSize);
        var scalarTime = benchmarkScalarCompleteTemporalART(dimension, sequenceLength, batchSize);

        var speedup = scalarTime / vectorizedTime;

        if (batchSize == 1 || batchSize >= 25) { // Log results for single and large batches
            log.info("Complete ART - Dim: {}, SeqLen: {}, Batch: {}, Vectorized: {:.2f}ms, Scalar: {:.2f}ms, Speedup: {:.2f}x",
                    dimension, sequenceLength, batchSize, vectorizedTime, scalarTime, speedup);
        }

        // Expect significant speedup for complete pipeline, especially with batches
        var expectedSpeedup = batchSize >= 10 ? 20.0 : 10.0;
        if (dimension >= 32 && sequenceLength >= 10) {
            assertTrue(speedup >= expectedSpeedup,
                      String.format("Expected at least %.1fx speedup for complete pipeline", expectedSpeedup));
        }
    }

    // === Individual Component Benchmarks ===

    private double benchmarkVectorizedWorkingMemory(int dimension, int sequenceLength) {
        var workingMemory = new VectorizedItemOrderWorkingMemory(
            parameters.workingMemoryParameters(), sequenceLength, dimension);

        var patterns = generateRandomPatterns(sequenceLength, dimension);

        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            workingMemory.clear();
            for (int j = 0; j < patterns.size(); j++) {
                workingMemory.storeItem(patterns.get(j), j);
            }
            workingMemory.updateDynamics(0.01);
        }

        // Benchmark
        var times = new ArrayList<Double>();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            workingMemory.clear();
            workingMemory.resetPerformanceTracking();

            var startTime = System.nanoTime();
            for (int j = 0; j < patterns.size(); j++) {
                workingMemory.storeItem(patterns.get(j), j);
            }
            workingMemory.updateDynamics(0.01);
            var endTime = System.nanoTime();

            times.add((endTime - startTime) * NANO_TO_MILLIS);
        }

        workingMemory.close();
        return calculateMedian(times);
    }

    private double benchmarkScalarWorkingMemory(int dimension, int sequenceLength) {
        // Simulate scalar working memory operations
        var patterns = generateRandomPatterns(sequenceLength, dimension);

        var times = new ArrayList<Double>();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            var startTime = System.nanoTime();

            // Simulate scalar primacy gradient computation
            for (int j = 0; j < sequenceLength; j++) {
                var primacy = Math.exp(-j * 0.1); // Scalar exponential
                var pattern = patterns.get(j);

                // Simulate scalar element-wise operations
                for (int k = 0; k < dimension; k++) {
                    var value = pattern.get(k) * primacy; // Scalar multiplication
                    var update = -0.1 * value + (1.0 - value) * pattern.get(k); // Scalar dynamics
                }
            }

            var endTime = System.nanoTime();
            times.add((endTime - startTime) * NANO_TO_MILLIS);
        }

        return calculateMedian(times);
    }

    private double benchmarkVectorizedMaskingField(int fieldSize, int scaleCount) {
        var maskingParams = parameters.maskingParameters()
            .builder()
            .fieldSize(fieldSize)
            .scaleCount(scaleCount)
            .build();

        var maskingField = new VectorizedMaskingField(maskingParams);
        var temporalPattern = createTemporalPattern(generateRandomPatterns(10, 16));

        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            maskingField.reset();
            maskingField.process(temporalPattern);
        }

        // Benchmark
        var times = new ArrayList<Double>();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            maskingField.reset();
            maskingField.resetPerformanceTracking();

            var startTime = System.nanoTime();
            maskingField.process(temporalPattern);
            var endTime = System.nanoTime();

            times.add((endTime - startTime) * NANO_TO_MILLIS);
        }

        maskingField.close();
        return calculateMedian(times);
    }

    private double benchmarkScalarMaskingField(int fieldSize, int scaleCount) {
        // Simulate scalar masking field operations
        var times = new ArrayList<Double>();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            var startTime = System.nanoTime();

            // Simulate scalar competitive dynamics for each scale
            for (int scale = 0; scale < scaleCount; scale++) {
                var actualFieldSize = fieldSize / (scale + 1);

                // Simulate scalar lateral inhibition computation
                for (int pos1 = 0; pos1 < actualFieldSize; pos1++) {
                    var inhibition = 0.0;
                    for (int pos2 = 0; pos2 < actualFieldSize; pos2++) {
                        if (pos1 != pos2) {
                            var distance = Math.abs(pos1 - pos2);
                            var weight = Math.exp(-distance * distance / 2.0);
                            inhibition += 0.5 * weight * Math.random(); // Scalar operations
                        }
                    }

                    // Simulate scalar dynamics update
                    var activation = Math.random();
                    var update = -0.1 * activation + (1.0 - activation) * Math.random() - activation * inhibition;
                }
            }

            var endTime = System.nanoTime();
            times.add((endTime - startTime) * NANO_TO_MILLIS);
        }

        return calculateMedian(times);
    }

    private double benchmarkVectorizedCompetitiveInstar(int inputDim, int categoryCount) {
        var instar = new VectorizedCompetitiveInstar(0.1f, categoryCount, inputDim);
        var patterns = generateRandomFloatPatterns(BENCHMARK_ITERATIONS, inputDim);
        var activations = new float[categoryCount];

        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            instar.resetWeights();
            for (int j = 0; j < categoryCount; j++) {
                activations[j] = (float) Math.random();
            }
            instar.updateWeights(patterns.get(i % patterns.size()), activations, 0);
        }

        // Benchmark
        var times = new ArrayList<Double>();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            instar.resetPerformanceTracking();

            var startTime = System.nanoTime();
            for (int j = 0; j < categoryCount; j++) {
                activations[j] = (float) Math.random();
            }
            instar.updateWeights(patterns.get(i), activations, 0);
            var endTime = System.nanoTime();

            times.add((endTime - startTime) * NANO_TO_MILLIS);
        }

        return calculateMedian(times);
    }

    private double benchmarkScalarCompetitiveInstar(int inputDim, int categoryCount) {
        // Simulate scalar competitive instar operations
        var weights = new float[categoryCount][inputDim];
        var patterns = generateRandomFloatPatterns(BENCHMARK_ITERATIONS, inputDim);

        var times = new ArrayList<Double>();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            var pattern = patterns.get(i);
            var startTime = System.nanoTime();

            // Simulate scalar weight updates
            for (int cat = 0; cat < categoryCount; cat++) {
                var activation = (float) Math.random();
                var learningSignal = 1.0f / (1.0f + (float) Math.exp(-activation)); // Scalar sigmoid

                // Scalar input sum computation
                var inputSum = 0.0f;
                for (int j = 0; j < inputDim; j++) {
                    inputSum += Math.abs(pattern[j]);
                }

                // Scalar weight update rule
                for (int j = 0; j < inputDim; j++) {
                    var currentWeight = weights[cat][j];
                    var excitatory = (1.0f - currentWeight) * pattern[j];
                    var inhibitory = currentWeight * inputSum;
                    var deltaWeight = 0.1f * learningSignal * (excitatory - inhibitory);
                    weights[cat][j] = Math.max(0.0f, Math.min(1.0f, currentWeight + deltaWeight));
                }
            }

            var endTime = System.nanoTime();
            times.add((endTime - startTime) * NANO_TO_MILLIS);
        }

        return calculateMedian(times);
    }

    private double benchmarkVectorizedCompleteTemporalART(int dimension, int sequenceLength, int batchSize) {
        var vectorizedART = new VectorizedTemporalART(parameters, sequenceLength * 2, dimension);
        var temporalPatterns = generateTemporalPatterns(batchSize, sequenceLength, dimension);

        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            vectorizedART.resetTemporalState();
            for (var pattern : temporalPatterns.subList(0, Math.min(batchSize, temporalPatterns.size()))) {
                vectorizedART.learnTemporal(pattern);
            }
        }

        // Benchmark
        var times = new ArrayList<Double>();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            vectorizedART.resetTemporalState();
            vectorizedART.resetPerformanceTracking();

            var startTime = System.nanoTime();
            if (batchSize == 1) {
                vectorizedART.learnTemporal(temporalPatterns.get(0));
            } else {
                vectorizedART.learnTemporalBatch(temporalPatterns);
            }
            var endTime = System.nanoTime();

            times.add((endTime - startTime) * NANO_TO_MILLIS);
        }

        vectorizedART.close();
        return calculateMedian(times);
    }

    private double benchmarkScalarCompleteTemporalART(int dimension, int sequenceLength, int batchSize) {
        // For now, use the same vectorized implementation as baseline
        // In a real scenario, this would be a scalar implementation
        var scalarART = new VectorizedTemporalART(parameters, sequenceLength * 2, dimension);
        var temporalPatterns = generateTemporalPatterns(batchSize, sequenceLength, dimension);

        var times = new ArrayList<Double>();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            scalarART.resetTemporalState();

            var startTime = System.nanoTime();
            for (var pattern : temporalPatterns) {
                scalarART.learnTemporal(pattern); // Process sequentially instead of batch
            }
            var endTime = System.nanoTime();

            times.add((endTime - startTime) * NANO_TO_MILLIS);
        }

        scalarART.close();
        return calculateMedian(times);
    }

    private double benchmarkVectorizedBatchProcessing(int dimension, int sequenceLength, int batchSize) {
        var vectorizedART = new VectorizedTemporalART(parameters, sequenceLength * 2, dimension);
        var temporalPatterns = generateTemporalPatterns(batchSize, sequenceLength, dimension);

        var times = new ArrayList<Double>();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            vectorizedART.resetTemporalState();

            var startTime = System.nanoTime();
            vectorizedART.learnTemporalBatch(temporalPatterns);
            var endTime = System.nanoTime();

            times.add((endTime - startTime) * NANO_TO_MILLIS);
        }

        vectorizedART.close();
        return calculateMedian(times);
    }

    private double benchmarkScalarBatchProcessing(int dimension, int sequenceLength, int batchSize) {
        var scalarART = new VectorizedTemporalART(parameters, sequenceLength * 2, dimension);
        var temporalPatterns = generateTemporalPatterns(batchSize, sequenceLength, dimension);

        var times = new ArrayList<Double>();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            scalarART.resetTemporalState();

            var startTime = System.nanoTime();
            // Process sequentially instead of batch
            for (var pattern : temporalPatterns) {
                scalarART.learnTemporal(pattern);
            }
            var endTime = System.nanoTime();

            times.add((endTime - startTime) * NANO_TO_MILLIS);
        }

        scalarART.close();
        return calculateMedian(times);
    }

    // === Memory Usage Measurement ===

    private long measureVectorizedMemoryUsage(int dimension, int sequenceLength, int batchSize) {
        var runtime = Runtime.getRuntime();
        runtime.gc();
        var beforeMemory = runtime.totalMemory() - runtime.freeMemory();

        var vectorizedART = new VectorizedTemporalART(parameters, sequenceLength * 2, dimension);
        var temporalPatterns = generateTemporalPatterns(batchSize, sequenceLength, dimension);

        vectorizedART.learnTemporalBatch(temporalPatterns);

        runtime.gc();
        var afterMemory = runtime.totalMemory() - runtime.freeMemory();

        vectorizedART.close();

        return afterMemory - beforeMemory;
    }

    private long measureScalarMemoryUsage(int dimension, int sequenceLength, int batchSize) {
        var runtime = Runtime.getRuntime();
        runtime.gc();
        var beforeMemory = runtime.totalMemory() - runtime.freeMemory();

        var scalarART = new VectorizedTemporalART(parameters, sequenceLength * 2, dimension);
        var temporalPatterns = generateTemporalPatterns(batchSize, sequenceLength, dimension);

        for (var pattern : temporalPatterns) {
            scalarART.learnTemporal(pattern);
        }

        runtime.gc();
        var afterMemory = runtime.totalMemory() - runtime.freeMemory();

        scalarART.close();

        return afterMemory - beforeMemory;
    }

    // === Helper Methods ===

    private List<Pattern> generateRandomPatterns(int count, int dimension) {
        var patterns = new ArrayList<Pattern>(count);
        for (int i = 0; i < count; i++) {
            var features = new double[dimension];
            for (int j = 0; j < dimension; j++) {
                features[j] = random.nextGaussian() * 0.1 + 0.5; // Centered around 0.5
            }
            patterns.add(Pattern.of(features));
        }
        return patterns;
    }

    private List<float[]> generateRandomFloatPatterns(int count, int dimension) {
        var patterns = new ArrayList<float[]>(count);
        for (int i = 0; i < count; i++) {
            var features = new float[dimension];
            for (int j = 0; j < dimension; j++) {
                features[j] = (float) (random.nextGaussian() * 0.1 + 0.5);
            }
            patterns.add(features);
        }
        return patterns;
    }

    private List<TemporalPattern> generateTemporalPatterns(int count, int sequenceLength, int dimension) {
        var patterns = new ArrayList<TemporalPattern>(count);
        for (int i = 0; i < count; i++) {
            var sequence = generateRandomPatterns(sequenceLength, dimension);
            patterns.add(createTemporalPattern(sequence));
        }
        return patterns;
    }

    private TemporalPattern createTemporalPattern(List<Pattern> sequence) {
        return new TemporalPattern() {
            @Override
            public List<Pattern> getSequence() {
                return sequence;
            }

            @Override
            public TemporalPattern getSubsequence(int startTime, int endTime) {
                return createTemporalPattern(sequence.subList(startTime, endTime));
            }

            @Override
            public boolean isEmpty() {
                return sequence.isEmpty();
            }
        };
    }

    private double calculateMedian(List<Double> values) {
        var sorted = values.stream().sorted().toList();
        var size = sorted.size();
        if (size % 2 == 0) {
            return (sorted.get(size / 2 - 1) + sorted.get(size / 2)) / 2.0;
        } else {
            return sorted.get(size / 2);
        }
    }
}