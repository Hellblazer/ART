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
import com.hellblazer.art.laminar.builders.LaminarCircuitBuilder;
import com.hellblazer.art.laminar.core.LaminarCircuit;
import com.hellblazer.art.laminar.impl.DefaultLaminarParameters;
import com.hellblazer.art.laminar.impl.DefaultLearningParameters;
import com.hellblazer.art.laminar.impl.LaminarCircuitImpl;
import com.hellblazer.art.laminar.performance.VectorizedLaminarCircuit;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * JMH benchmarks comparing standard vs vectorized laminar circuit implementations.
 *
 * Measures performance across different:
 * - Input dimensions (small, medium, large)
 * - Pattern densities (sparse, medium, dense)
 * - Operation types (learning, prediction, batch processing)
 *
 * Run with: mvn test -Dtest=LaminarCircuitBenchmark
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Fork(value = 1, jvmArgs = {
    "--add-modules=jdk.incubator.vector",
    "--enable-preview",
    "-Xmx2g"
})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 2)
public class LaminarCircuitBenchmark {

    @Param({"100", "500", "1000"})
    private int inputDimension;

    @Param({"0.1", "0.5", "0.9"})
    private double patternDensity;

    private LaminarCircuit<DefaultLaminarParameters> standardCircuit;
    private VectorizedLaminarCircuit<DefaultLaminarParameters> vectorizedCircuit;
    private DefaultLaminarParameters parameters;
    private Pattern[] testPatterns;
    private Random random;

    @Setup(Level.Trial)
    public void setup() {
        random = new Random(42);

        // Create parameters
        parameters = DefaultLaminarParameters.builder()
            .withLearningParameters(new DefaultLearningParameters(0.5, 0.0, false, 0.0))
            .withVigilance(0.8)
            .build();

        // Create standard circuit
        standardCircuit = new LaminarCircuitBuilder<DefaultLaminarParameters>()
            .withParameters(parameters)
            .withInputLayer(inputDimension, false)
            .withFeatureLayer(inputDimension)
            .withCategoryLayer(100)
            .withStandardConnections()
            .build();

        // Create vectorized circuit
        vectorizedCircuit = new VectorizedLaminarCircuit<>(parameters);
        new LaminarCircuitBuilder<DefaultLaminarParameters>()
            .withParameters(parameters)
            .withInputLayer(inputDimension, false)
            .withFeatureLayer(inputDimension)
            .withCategoryLayer(100)
            .withStandardConnections()
            .build(); // Configure the vectorized circuit

        // Generate test patterns
        testPatterns = generatePatterns(100, inputDimension, patternDensity);

        // Pre-train both circuits with same data
        for (int i = 0; i < 10; i++) {
            standardCircuit.learn(testPatterns[i], parameters);
            vectorizedCircuit.learn(testPatterns[i], parameters);
        }
    }

    @Benchmark
    public void standardLearn() {
        var pattern = testPatterns[random.nextInt(testPatterns.length)];
        standardCircuit.learn(pattern, parameters);
    }

    @Benchmark
    public void vectorizedLearn() {
        var pattern = testPatterns[random.nextInt(testPatterns.length)];
        vectorizedCircuit.learn(pattern, parameters);
    }

    @Benchmark
    public void standardPredict() {
        var pattern = testPatterns[random.nextInt(testPatterns.length)];
        standardCircuit.predict(pattern, parameters);
    }

    @Benchmark
    public void vectorizedPredict() {
        var pattern = testPatterns[random.nextInt(testPatterns.length)];
        vectorizedCircuit.predict(pattern, parameters);
    }

    @Benchmark
    public void standardBatchProcess() {
        for (int i = 0; i < 10; i++) {
            standardCircuit.processCycle(testPatterns[i], parameters);
        }
    }

    @Benchmark
    public void vectorizedBatchProcess() {
        for (int i = 0; i < 10; i++) {
            vectorizedCircuit.processCycle(testPatterns[i], parameters);
        }
    }

    private Pattern[] generatePatterns(int count, int dimension, double density) {
        var patterns = new Pattern[count];
        for (int i = 0; i < count; i++) {
            var values = new double[dimension];
            for (int j = 0; j < dimension; j++) {
                if (random.nextDouble() < density) {
                    values[j] = random.nextDouble();
                }
            }
            patterns[i] = new DenseVector(values);
        }
        return patterns;
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        if (vectorizedCircuit != null) {
            vectorizedCircuit.close();
        }
    }

    /**
     * Main method to run benchmarks programmatically.
     */
    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
            .include(LaminarCircuitBenchmark.class.getSimpleName())
            .build();

        new Runner(opt).run();
    }
}