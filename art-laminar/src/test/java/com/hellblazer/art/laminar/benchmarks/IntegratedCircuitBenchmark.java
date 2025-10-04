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
package com.hellblazer.art.laminar.benchmarks;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.canonical.CircuitParameters;
import com.hellblazer.art.laminar.canonical.FullLaminarCircuitImpl;
import com.hellblazer.art.laminar.integration.ARTCircuitParameters;
import com.hellblazer.art.laminar.integration.ARTLaminarCircuit;
import com.hellblazer.art.laminar.integration.VectorizedARTLaminarCircuit;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * Comprehensive JMH benchmark comparing all three laminar circuit implementations:
 * 1. FullLaminarCircuitImpl (baseline - manual template management)
 * 2. ARTLaminarCircuit (FuzzyART integration)
 * 3. VectorizedARTLaminarCircuit (SIMD-optimized ART)
 *
 * Measures:
 * - Single pattern processing throughput
 * - Batch processing efficiency
 * - Category formation performance
 * - Memory allocation patterns
 * - Relative speedups
 *
 * Expected Performance Hierarchy (100D patterns):
 * 1. VectorizedARTLaminarCircuit: ~5-10ms (5-10x faster)
 * 2. ARTLaminarCircuit: ~40-80ms (similar or slightly faster than baseline)
 * 3. FullLaminarCircuitImpl: ~50-100ms (baseline)
 *
 * Run with:
 *   mvn clean package
 *   java -jar target/benchmarks.jar IntegratedCircuitBenchmark
 */
@BenchmarkMode({Mode.Throughput, Mode.AverageTime})
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Fork(value = 1, jvmArgs = {
    "--add-modules=jdk.incubator.vector",
    "--enable-preview",
    "-Xmx4g",
    "-XX:+UseG1GC"
})
@Warmup(iterations = 5, time = 2, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 10, time = 3, timeUnit = TimeUnit.SECONDS)
public class IntegratedCircuitBenchmark {

    @Param({"64", "128", "256"})
    private int inputSize;

    private FullLaminarCircuitImpl manualCircuit;
    private ARTLaminarCircuit artCircuit;
    private VectorizedARTLaminarCircuit vectorizedCircuit;
    private Pattern[] patterns;
    private Random random;

    @Setup(Level.Trial)
    public void setup() {
        random = new Random(42);

        // Manual circuit (baseline)
        var manualParams = CircuitParameters.builder()
            .inputSize(inputSize)
            .categorySize(inputSize / 2)
            .vigilance(0.75)
            .learningRate(0.5)
            .timeStep(0.01)
            .topDownGain(0.8)
            .expectationThreshold(0.1)
            .maxSearchIterations(100)
            .resetThreshold(0.5)
            .build();
        manualCircuit = new FullLaminarCircuitImpl(manualParams);

        // ART circuits (equivalent parameters)
        var artParams = ARTCircuitParameters.builder(inputSize)
            .maxCategories(inputSize / 2)
            .vigilance(0.75)
            .learningRate(0.5)
            .choiceParameter(0.001)
            .topDownGain(0.8)
            .timeStep(0.01)
            .expectationThreshold(0.1)
            .maxSearchIterations(100)
            .build();
        artCircuit = new ARTLaminarCircuit(artParams);
        vectorizedCircuit = new VectorizedARTLaminarCircuit(artParams);

        // Generate test patterns with realistic structure
        patterns = new Pattern[200];
        for (int i = 0; i < 200; i++) {
            var data = new double[inputSize];
            // Create 20 clusters with 10 patterns each
            var cluster = i / 10;
            var baseValue = (cluster * 0.05) % 1.0;
            for (int j = 0; j < inputSize; j++) {
                var noise = random.nextGaussian() * 0.1;
                data[j] = Math.max(0.0, Math.min(1.0, baseValue + noise));
            }
            patterns[i] = new DenseVector(data);
        }

        // Pre-train all circuits with first 20 patterns
        for (int i = 0; i < 20; i++) {
            manualCircuit.process(patterns[i]);
            artCircuit.process(patterns[i]);
            vectorizedCircuit.process(patterns[i]);
        }
    }

    // ========== Single Pattern Processing ==========

    @Benchmark
    public void baseline_singlePattern(Blackhole bh) {
        var pattern = patterns[random.nextInt(patterns.length)];
        var result = manualCircuit.process(pattern);
        bh.consume(result);
    }

    @Benchmark
    public void art_singlePattern(Blackhole bh) {
        var pattern = patterns[random.nextInt(patterns.length)];
        var result = artCircuit.process(pattern);
        bh.consume(result);
    }

    @Benchmark
    public void vectorized_singlePattern(Blackhole bh) {
        var pattern = patterns[random.nextInt(patterns.length)];
        var result = vectorizedCircuit.process(pattern);
        bh.consume(result);
    }

    // ========== Batch Processing ==========

    @Benchmark
    public void baseline_batch10(Blackhole bh) {
        for (int i = 0; i < 10; i++) {
            var result = manualCircuit.process(patterns[i]);
            bh.consume(result);
        }
    }

    @Benchmark
    public void art_batch10(Blackhole bh) {
        for (int i = 0; i < 10; i++) {
            var result = artCircuit.process(patterns[i]);
            bh.consume(result);
        }
    }

    @Benchmark
    public void vectorized_batch10(Blackhole bh) {
        for (int i = 0; i < 10; i++) {
            var result = vectorizedCircuit.process(patterns[i]);
            bh.consume(result);
        }
    }

    // ========== Sequential Learning ==========

    @Benchmark
    public void baseline_sequentialLearning(Blackhole bh) {
        for (int i = 0; i < 5; i++) {
            var pattern = patterns[20 + i]; // Use unseen patterns
            var result = manualCircuit.process(pattern);
            bh.consume(result);
        }
    }

    @Benchmark
    public void art_sequentialLearning(Blackhole bh) {
        for (int i = 0; i < 5; i++) {
            var pattern = patterns[20 + i];
            var result = artCircuit.process(pattern);
            bh.consume(result);
        }
    }

    @Benchmark
    public void vectorized_sequentialLearning(Blackhole bh) {
        for (int i = 0; i < 5; i++) {
            var pattern = patterns[20 + i];
            var result = vectorizedCircuit.process(pattern);
            bh.consume(result);
        }
    }

    @TearDown(Level.Iteration)
    public void logStats() {
        System.out.printf("[%dD] Categories - Manual: %d, ART: %d, Vectorized: %d%n",
            inputSize,
            manualCircuit.getPredictionGenerator().getCommittedCount(),
            artCircuit.getCategoryCount(),
            vectorizedCircuit.getCategoryCount());

        var vecStats = vectorizedCircuit.getPerformanceStats();
        if (vecStats != null) {
            System.out.printf("[%dD] Vectorized - Vector ops: %d, Parallel: %d, Efficiency: %.1f%%%n",
                inputSize,
                vecStats.totalVectorOperations(),
                vecStats.totalParallelTasks(),
                vecStats.getParallelEfficiency() * 100);
        }
    }

    @TearDown(Level.Iteration)
    public void resetCircuits() {
        // Reset to pre-trained state
        manualCircuit.reset();
        artCircuit.reset();
        vectorizedCircuit.reset();

        // Re-train with first 20 patterns
        for (int i = 0; i < 20; i++) {
            manualCircuit.process(patterns[i]);
            artCircuit.process(patterns[i]);
            vectorizedCircuit.process(patterns[i]);
        }

        vectorizedCircuit.resetPerformanceTracking();
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        try {
            if (artCircuit != null) {
                artCircuit.close();
            }
            if (vectorizedCircuit != null) {
                vectorizedCircuit.close();
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to close circuits", e);
        }
    }

    /**
     * Main method to run benchmarks programmatically.
     */
    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
            .include(IntegratedCircuitBenchmark.class.getSimpleName())
            .build();

        new Runner(opt).run();
    }
}