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
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * JMH benchmark comparing FullLaminarCircuitImpl (manual template management)
 * vs ARTLaminarCircuit (FuzzyART-based category learning).
 *
 * Measures:
 * - Processing overhead of ART integration
 * - Category formation efficiency
 * - Template learning performance
 * - Pattern recognition speed
 *
 * Expected Results:
 * - ART circuit slightly slower initially (complement coding overhead)
 * - ART circuit faster after many patterns (better category organization)
 * - 10-20% overhead for ART integration
 *
 * Run with:
 *   mvn clean package
 *   java -jar target/benchmarks.jar ARTIntegrationBenchmark
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Fork(value = 1, jvmArgs = {
    "--add-modules=jdk.incubator.vector",
    "--enable-preview",
    "-Xmx2g"
})
@Warmup(iterations = 3, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 2, timeUnit = TimeUnit.SECONDS)
public class ARTIntegrationBenchmark {

    @Param({"50", "100", "200"})
    private int inputSize;

    private FullLaminarCircuitImpl manualCircuit;
    private ARTLaminarCircuit artCircuit;
    private Pattern[] patterns;
    private Random random;

    @Setup(Level.Trial)
    public void setup() {
        random = new Random(42);

        // Manual circuit with baseline parameters
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

        // ART circuit with equivalent parameters
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

        // Generate test patterns with varied similarity
        patterns = new Pattern[100];
        for (int i = 0; i < 100; i++) {
            var data = new double[inputSize];
            for (int j = 0; j < inputSize; j++) {
                // Create patterns with clustering structure
                var cluster = i / 10; // 10 patterns per cluster
                var clusterCenter = cluster * 0.1;
                data[j] = Math.max(0.0, Math.min(1.0,
                    clusterCenter + (random.nextGaussian() * 0.15)));
            }
            patterns[i] = new DenseVector(data);
        }
    }

    @Benchmark
    public void manualCircuitProcessing() {
        var pattern = patterns[random.nextInt(patterns.length)];
        manualCircuit.process(pattern);
    }

    @Benchmark
    public void artCircuitProcessing() {
        var pattern = patterns[random.nextInt(patterns.length)];
        artCircuit.process(pattern);
    }

    @Benchmark
    public void manualCircuitBatch() {
        for (int i = 0; i < 10; i++) {
            manualCircuit.process(patterns[i]);
        }
    }

    @Benchmark
    public void artCircuitBatch() {
        for (int i = 0; i < 10; i++) {
            artCircuit.process(patterns[i]);
        }
    }

    @TearDown(Level.Iteration)
    public void resetCircuits() {
        manualCircuit.reset();
        artCircuit.reset();
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        try {
            if (artCircuit != null) {
                artCircuit.close();
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to close ART circuit", e);
        }
    }

    /**
     * Main method to run benchmarks programmatically.
     */
    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
            .include(ARTIntegrationBenchmark.class.getSimpleName())
            .build();

        new Runner(opt).run();
    }
}