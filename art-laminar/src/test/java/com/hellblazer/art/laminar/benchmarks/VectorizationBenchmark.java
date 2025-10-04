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
 * JMH benchmark measuring SIMD vectorization speedup between standard
 * ARTLaminarCircuit and VectorizedARTLaminarCircuit.
 *
 * Measures:
 * - SIMD speedup at different dimensions
 * - Vectorization efficiency
 * - Performance statistics accuracy
 * - Memory throughput improvements
 *
 * Expected Results:
 * - Vectorized faster for larger dimensions (256D: 3-5x speedup)
 * - Vectorized similar or slower for small dimensions (50D: 0.8-1.2x)
 * - Speedup increases with dimension size
 * - More patterns = better amortization of SIMD overhead
 *
 * Run with:
 *   mvn clean package
 *   java -jar target/benchmarks.jar VectorizationBenchmark
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Fork(value = 1, jvmArgs = {
    "--add-modules=jdk.incubator.vector",
    "--enable-preview",
    "-Xmx4g"
})
@Warmup(iterations = 5, time = 2, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 10, time = 3, timeUnit = TimeUnit.SECONDS)
public class VectorizationBenchmark {

    @Param({"50", "100", "256", "512"})
    private int inputSize;

    @Param({"100", "1000"})
    private int patternCount;

    private ARTLaminarCircuit standardCircuit;
    private VectorizedARTLaminarCircuit vectorizedCircuit;
    private Pattern[] patterns;
    private Random random;

    @Setup(Level.Trial)
    public void setup() {
        random = new Random(42);

        var params = ARTCircuitParameters.builder(inputSize)
            .maxCategories(inputSize / 2)
            .vigilance(0.75)
            .learningRate(0.5)
            .choiceParameter(0.001)
            .topDownGain(0.8)
            .timeStep(0.01)
            .expectationThreshold(0.1)
            .maxSearchIterations(100)
            .build();

        standardCircuit = new ARTLaminarCircuit(params);
        vectorizedCircuit = new VectorizedARTLaminarCircuit(params);

        // Generate diverse test patterns for better performance measurement
        patterns = new Pattern[patternCount];
        for (int i = 0; i < patternCount; i++) {
            var data = new double[inputSize];
            for (int j = 0; j < inputSize; j++) {
                data[j] = Math.max(0.0, Math.min(1.0, random.nextDouble()));
            }
            patterns[i] = new DenseVector(data);
        }

        // Warm up with some learning
        for (int i = 0; i < Math.min(20, patternCount); i++) {
            standardCircuit.process(patterns[i]);
            vectorizedCircuit.process(patterns[i]);
        }
    }

    @Benchmark
    public void standardProcessing(Blackhole bh) {
        var pattern = patterns[random.nextInt(patterns.length)];
        var result = standardCircuit.process(pattern);
        bh.consume(result);
    }

    @Benchmark
    public void vectorizedProcessing(Blackhole bh) {
        var pattern = patterns[random.nextInt(patterns.length)];
        var result = vectorizedCircuit.process(pattern);
        bh.consume(result);
    }

    @Benchmark
    public void standardBatch(Blackhole bh) {
        var batchSize = Math.min(50, patterns.length);
        for (int i = 0; i < batchSize; i++) {
            var result = standardCircuit.process(patterns[i]);
            bh.consume(result);
        }
    }

    @Benchmark
    public void vectorizedBatch(Blackhole bh) {
        var batchSize = Math.min(50, patterns.length);
        for (int i = 0; i < batchSize; i++) {
            var result = vectorizedCircuit.process(patterns[i]);
            bh.consume(result);
        }
    }

    @TearDown(Level.Iteration)
    public void logPerformanceStats() {
        var stats = vectorizedCircuit.getPerformanceStats();
        if (stats != null) {
            System.out.printf("[%dD, %d patterns] Vector ops: %d, Parallel tasks: %d, Avg: %.2fms%n",
                inputSize,
                patternCount,
                stats.totalVectorOperations(),
                stats.totalParallelTasks(),
                stats.avgComputeTimeMs());
        }
    }

    @TearDown(Level.Iteration)
    public void resetPerformanceTracking() {
        vectorizedCircuit.resetPerformanceTracking();
        standardCircuit.reset();
        vectorizedCircuit.reset();
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        try {
            if (standardCircuit != null) {
                standardCircuit.close();
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
            .include(VectorizationBenchmark.class.getSimpleName())
            .build();

        new Runner(opt).run();
    }
}