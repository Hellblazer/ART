/*
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 */

package com.hellblazer.art.performance.benchmarks;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.algorithms.HypersphereART;
import com.hellblazer.art.core.parameters.HypersphereParameters;
import com.hellblazer.art.performance.algorithms.VectorizedHypersphereART;
import com.hellblazer.art.performance.algorithms.VectorizedHypersphereParameters;

/**
 * JMH benchmarks comparing SIMD-optimized VectorizedHypersphereART 
 * with scalar HypersphereART implementation
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Fork(1)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 2)
public class VectorizedHypersphereARTBenchmark {

    // Test data with different dimensions
    private List<Pattern> smallPatterns;    // 4D
    private List<Pattern> mediumPatterns;   // 16D  
    private List<Pattern> largePatterns;    // 64D
    
    private HypersphereParameters scalarParams;
    private VectorizedHypersphereParameters simdParams;
    
    @Setup
    public void setUp() {
        // Configure parameters
        scalarParams = HypersphereParameters.of(0.8, 0.5, false);
        simdParams = VectorizedHypersphereParameters.highPerformance(16);
        
        // Generate test patterns
        smallPatterns = generateRandomPatterns(100, 4);
        mediumPatterns = generateRandomPatterns(100, 16);
        largePatterns = generateRandomPatterns(100, 64);
    }

    @Benchmark
    @Group("learning")
    public void scalarLearning16D(Blackhole bh) {
        var network = new HypersphereART();
        for (var pattern : mediumPatterns) {
            bh.consume(network.stepFit(pattern, scalarParams));
        }
    }

    @Benchmark
    @Group("learning")
    public void simdLearning16D(Blackhole bh) {
        var network = new VectorizedHypersphereART(simdParams);
        for (var pattern : mediumPatterns) {
            bh.consume(network.learn(pattern));
        }
    }

    @Benchmark
    @Group("learning")
    public void scalarLearning64D(Blackhole bh) {
        var network = new HypersphereART();
        var params64 = HypersphereParameters.of(0.8, 0.5, false);
        for (var pattern : largePatterns) {
            bh.consume(network.stepFit(pattern, params64));
        }
    }

    @Benchmark
    @Group("learning")
    public void simdLearning64D(Blackhole bh) {
        var simdParams64 = VectorizedHypersphereParameters.highPerformance(64);
        var network = new VectorizedHypersphereART(simdParams64);
        for (var pattern : largePatterns) {
            bh.consume(network.learn(pattern));
        }
    }

    @Benchmark
    @Group("mixed_workload")
    public void scalarMixedWorkload(Blackhole bh) {
        var network = new HypersphereART();
        
        // Mixed learning operations
        for (int i = 0; i < 50; i++) {
            var pattern = mediumPatterns.get(i % mediumPatterns.size());
            bh.consume(network.stepFit(pattern, scalarParams));
        }
    }

    @Benchmark
    @Group("mixed_workload")
    public void simdMixedWorkload(Blackhole bh) {
        var network = new VectorizedHypersphereART(simdParams);
        
        // Mixed learning operations
        for (int i = 0; i < 50; i++) {
            var pattern = mediumPatterns.get(i % mediumPatterns.size());
            bh.consume(network.learn(pattern));
        }
    }

    @Benchmark
    @Group("vigilance_sweep")
    public void scalarVigilanceSweep(Blackhole bh) {
        double[] vigilanceLevels = {0.1, 0.3, 0.5, 0.7, 0.9};
        
        for (var vigilance : vigilanceLevels) {
            var params = HypersphereParameters.of(vigilance, 0.5, false);
            var network = new HypersphereART();
            
            // Learn a subset of patterns
            var testPatterns = mediumPatterns.subList(0, 20);
            for (var pattern : testPatterns) {
                bh.consume(network.stepFit(pattern, params));
            }
        }
    }

    @Benchmark
    @Group("vigilance_sweep")
    public void simdVigilanceSweep(Blackhole bh) {
        double[] vigilanceLevels = {0.1, 0.3, 0.5, 0.7, 0.9};
        
        for (var vigilance : vigilanceLevels) {
            var params = VectorizedHypersphereParameters.builder()
                .vigilance(vigilance)
                .learningRate(0.5)
                .inputDimensions(16)
                .maxCategories(50)
                .enableSIMD(true)
                .build();
            var network = new VectorizedHypersphereART(params);
            
            // Learn a subset of patterns
            var testPatterns = mediumPatterns.subList(0, 20);
            for (var pattern : testPatterns) {
                bh.consume(network.learn(pattern));
            }
        }
    }

    private List<Pattern> generateRandomPatterns(int count, int dimensions) {
        var random = ThreadLocalRandom.current();
        return IntStream.range(0, count)
            .mapToObj(i -> {
                var values = new double[dimensions];
                for (int j = 0; j < dimensions; j++) {
                    values[j] = random.nextGaussian() * 2.0;
                }
                return (Pattern) new DenseVector(values);
            })
            .toList();
    }

    public static void main(String[] args) throws RunnerException {
        var opt = new OptionsBuilder()
            .include(VectorizedHypersphereARTBenchmark.class.getSimpleName())
            .forks(1)
            .warmupIterations(3)
            .measurementIterations(5)
            .build();

        new Runner(opt).run();
    }
}