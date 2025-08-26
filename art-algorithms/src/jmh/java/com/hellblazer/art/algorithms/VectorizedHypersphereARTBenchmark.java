/*
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 */

package com.hellblazer.art.algorithms;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.algorithms.HypersphereART;
import com.hellblazer.art.core.parameters.HypersphereParameters;

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

    // Scalar reference implementation
    private HypersphereART scalarNetwork;
    
    // SIMD optimized implementation
    private VectorizedHypersphereART simdNetwork;
    
    // SIMD with scalar fallback
    private VectorizedHypersphereART hybridNetwork;
    
    // Test data with different dimensions
    private List<Pattern> smallPatterns;    // 4D
    private List<Pattern> mediumPatterns;   // 16D  
    private List<Pattern> largePatterns;    // 64D
    
    @Setup
    public void setUp() {
        // Configure scalar network
        var scalarParams = HypersphereParameters.builder()
            .vigilance(0.8)
            .learningRate(0.5)
            .inputDimensions(16)
            .maxCategories(100)
            .build();
        scalarNetwork = new HypersphereART(scalarParams);
        
        // Configure SIMD network
        var simdParams = VectorizedHypersphereParameters.highPerformance(16);
        simdNetwork = new VectorizedHypersphereART(simdParams);
        
        // Configure hybrid network (SIMD with scalar fallback)
        var hybridParams = VectorizedHypersphereParameters.builder()
            .vigilance(0.8)
            .learningRate(0.5)
            .inputDimensions(16)
            .maxCategories(100)
            .enableSIMD(true)
            .simdThreshold(4) // Use SIMD for dimensions >= 4
            .build();
        hybridNetwork = new VectorizedHypersphereART(hybridParams);
        
        // Generate test patterns
        smallPatterns = generateRandomPatterns(1000, 4);
        mediumPatterns = generateRandomPatterns(1000, 16);
        largePatterns = generateRandomPatterns(1000, 64);
    }

    @Benchmark
    @Group("learning")
    public void scalarLearning4D(Blackhole bh) {
        var network = new HypersphereART(createScalarParams(4));
        for (var pattern : smallPatterns) {
            bh.consume(network.learn(pattern));
        }
    }

    @Benchmark
    @Group("learning")
    public void simdLearning4D(Blackhole bh) {
        var network = new VectorizedHypersphereART(createSIMDParams(4));
        for (var pattern : smallPatterns) {
            bh.consume(network.learn(pattern));
        }
    }

    @Benchmark
    @Group("learning")
    public void scalarLearning16D(Blackhole bh) {
        var network = new HypersphereART(createScalarParams(16));
        for (var pattern : mediumPatterns) {
            bh.consume(network.learn(pattern));
        }
    }

    @Benchmark
    @Group("learning")
    public void simdLearning16D(Blackhole bh) {
        var network = new VectorizedHypersphereART(createSIMDParams(16));
        for (var pattern : mediumPatterns) {
            bh.consume(network.learn(pattern));
        }
    }

    @Benchmark
    @Group("learning")
    public void scalarLearning64D(Blackhole bh) {
        var scalarParams64 = HypersphereParameters.builder()
            .vigilance(0.8)
            .learningRate(0.5)
            .inputDimensions(64)
            .maxCategories(100)
            .build();
        var network = new HypersphereART(scalarParams64);
        for (var pattern : largePatterns) {
            bh.consume(network.learn(pattern));
        }
    }

    @Benchmark
    @Group("learning")
    public void simdLearning64D(Blackhole bh) {
        var network = new VectorizedHypersphereART(createSIMDParams(64));
        for (var pattern : largePatterns) {
            bh.consume(network.learn(pattern));
        }
    }

    @Benchmark
    @Group("classification")
    public void scalarClassification16D(Blackhole bh) {
        // Pre-train with some patterns
        var trainPatterns = mediumPatterns.subList(0, 100);
        trainPatterns.forEach(scalarNetwork::learn);
        
        // Benchmark classification
        var testPatterns = mediumPatterns.subList(100, 200);
        for (var pattern : testPatterns) {
            bh.consume(scalarNetwork.classify(pattern));
        }
    }

    @Benchmark
    @Group("classification")
    public void simdClassification16D(Blackhole bh) {
        // Pre-train with some patterns
        var trainPatterns = mediumPatterns.subList(0, 100);
        trainPatterns.forEach(simdNetwork::learn);
        
        // Benchmark classification
        var testPatterns = mediumPatterns.subList(100, 200);
        for (var pattern : testPatterns) {
            bh.consume(simdNetwork.classify(pattern));
        }
    }

    @Benchmark
    @Group("distance")
    public void scalarDistanceCalculation(Blackhole bh) {
        // Pre-train to create categories with weights
        var trainPatterns = mediumPatterns.subList(0, 50);
        var network = new HypersphereART(createScalarParams(16));
        trainPatterns.forEach(network::learn);
        
        // Benchmark distance calculations through classification
        var testPatterns = mediumPatterns.subList(50, 150);
        for (var pattern : testPatterns) {
            bh.consume(network.classify(pattern));
        }
    }

    @Benchmark
    @Group("distance")
    public void simdDistanceCalculation(Blackhole bh) {
        // Pre-train to create categories with weights
        var trainPatterns = mediumPatterns.subList(0, 50);
        var network = new VectorizedHypersphereART(createSIMDParams(16));
        trainPatterns.forEach(network::learn);
        
        // Benchmark distance calculations through classification
        var testPatterns = mediumPatterns.subList(50, 150);
        for (var pattern : testPatterns) {
            bh.consume(network.classify(pattern));
        }
    }

    @Benchmark
    @Group("mixed_workload")
    public void scalarMixedWorkload(Blackhole bh) {
        var network = new HypersphereART(createScalarParams(16));
        
        // Mixed learning and classification
        for (int i = 0; i < 200; i++) {
            var pattern = mediumPatterns.get(i);
            if (i % 3 == 0) {
                bh.consume(network.learn(pattern));
            } else {
                bh.consume(network.classify(pattern));
            }
        }
    }

    @Benchmark
    @Group("mixed_workload")
    public void simdMixedWorkload(Blackhole bh) {
        var network = new VectorizedHypersphereART(createSIMDParams(16));
        
        // Mixed learning and classification
        for (int i = 0; i < 200; i++) {
            var pattern = mediumPatterns.get(i);
            if (i % 3 == 0) {
                bh.consume(network.learn(pattern));
            } else {
                bh.consume(network.classify(pattern));
            }
        }
    }

    @Benchmark
    @Group("vigilance_sweep")
    public void scalarVigilanceSweep(Blackhole bh) {
        double[] vigilanceLevels = {0.1, 0.3, 0.5, 0.7, 0.9};
        
        for (var vigilance : vigilanceLevels) {
            var params = HypersphereParameters.builder()
                .vigilance(vigilance)
                .learningRate(0.5)
                .inputDimensions(16)
                .maxCategories(50)
                .build();
            var network = new HypersphereART(params);
            
            // Learn a subset of patterns
            var testPatterns = mediumPatterns.subList(0, 100);
            for (var pattern : testPatterns) {
                bh.consume(network.learn(pattern));
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
            var testPatterns = mediumPatterns.subList(0, 100);
            for (var pattern : testPatterns) {
                bh.consume(network.learn(pattern));
            }
        }
    }

    @Benchmark
    @Group("memory_pressure")
    public void scalarMemoryPressure(Blackhole bh) {
        var network = new HypersphereART(createScalarParams(16));
        
        // Create many categories to test memory usage
        var diversePatterns = generateDiversePatterns(500, 16);
        for (var pattern : diversePatterns) {
            bh.consume(network.learn(pattern));
        }
        
        // Then test classification performance with many categories
        var testPatterns = mediumPatterns.subList(0, 100);
        for (var pattern : testPatterns) {
            bh.consume(network.classify(pattern));
        }
    }

    @Benchmark
    @Group("memory_pressure")
    public void simdMemoryPressure(Blackhole bh) {
        var network = new VectorizedHypersphereART(createSIMDParams(16));
        
        // Create many categories to test memory usage
        var diversePatterns = generateDiversePatterns(500, 16);
        for (var pattern : diversePatterns) {
            bh.consume(network.learn(pattern));
        }
        
        // Then test classification performance with many categories
        var testPatterns = mediumPatterns.subList(0, 100);
        for (var pattern : testPatterns) {
            bh.consume(network.classify(pattern));
        }
    }

    // Helper methods

    private HypersphereParameters createScalarParams(int dimensions) {
        return HypersphereParameters.builder()
            .vigilance(0.8)
            .learningRate(0.5)
            .inputDimensions(dimensions)
            .maxCategories(100)
            .build();
    }

    private VectorizedHypersphereParameters createSIMDParams(int dimensions) {
        return VectorizedHypersphereParameters.builder()
            .vigilance(0.8)
            .learningRate(0.5)
            .inputDimensions(dimensions)
            .maxCategories(100)
            .enableSIMD(true)
            .build();
    }

    private List<Pattern> generateRandomPatterns(int count, int dimensions) {
        var random = ThreadLocalRandom.current();
        return IntStream.range(0, count)
            .mapToObj(i -> {
                var values = new double[dimensions];
                for (int j = 0; j < dimensions; j++) {
                    values[j] = random.nextGaussian() * 2.0;
                }
                return new DenseVector(values);
            })
            .toList();
    }

    private List<Pattern> generateDiversePatterns(int count, int dimensions) {
        var random = ThreadLocalRandom.current();
        return IntStream.range(0, count)
            .mapToObj(i -> {
                var values = new double[dimensions];
                // Create more diverse patterns to force category creation
                var baseValue = random.nextDouble() * 20.0 - 10.0;
                for (int j = 0; j < dimensions; j++) {
                    values[j] = baseValue + random.nextGaussian() * 0.1;
                }
                return new DenseVector(values);
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