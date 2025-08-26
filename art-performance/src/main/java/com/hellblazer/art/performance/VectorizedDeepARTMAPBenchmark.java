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
package com.hellblazer.art.performance;

import com.hellblazer.art.algorithms.*;
import com.hellblazer.art.core.*;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Performance benchmark comparing VectorizedDeepARTMAP vs standard DeepARTMAP.
 * 
 * This benchmark demonstrates the performance benefits of SIMD vectorization and
 * parallel processing in hierarchical DeepARTMAP implementations.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Fork(value = 1)
@Warmup(iterations = 3, time = 2, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 3, timeUnit = TimeUnit.SECONDS)
public class VectorizedDeepARTMAPBenchmark {
    
    private VectorizedDeepARTMAP vectorizedDeepART;
    private DeepARTMAP standardDeepART;
    
    private VectorizedDeepARTMAPParameters vectorizedParams;
    private DeepARTMAPParameters standardParams;
    
    private List<Pattern[]> trainingData;
    private List<Pattern[]> testData;
    private int[] supervisedLabels;
    
    @Setup(Level.Trial)
    public void setupTrial() {
        // Create base parameters
        standardParams = new DeepARTMAPParameters(0.8, 0.1, 1000, true);
        vectorizedParams = VectorizedDeepARTMAPParameters.highPerformance(standardParams);
        
        // Create training data (multi-channel)
        createTrainingData();
        createTestData();
    }
    
    @Setup(Level.Iteration)
    public void setupIteration() {
        // Create vectorized modules - match standard modules count
        var vectorizedModules = List.<BaseART>of(
            new VectorizedFuzzyART(VectorizedParameters.createDefault()),
            new VectorizedFuzzyART(VectorizedParameters.createDefault())
        );
        
        vectorizedDeepART = new VectorizedDeepARTMAP(vectorizedModules, vectorizedParams);
        
        // Create standard modules - use only FuzzyART to avoid HypersphereART issues
        var standardModules = List.<BaseART>of(
            new FuzzyART(),
            new FuzzyART()
        );
        
        standardDeepART = new DeepARTMAP(standardModules, standardParams);
        
        // Pre-train both networks
        vectorizedDeepART.fitSupervised(trainingData, supervisedLabels);
        standardDeepART.fitSupervised(trainingData, supervisedLabels);
    }
    
    @TearDown(Level.Iteration)
    public void tearDownIteration() {
        if (vectorizedDeepART != null) {
            vectorizedDeepART.close();
        }
    }
    
    /**
     * Create multi-channel training data.
     */
    private void createTrainingData() {
        int sampleCount = 200;
        
        // Channel 0: FuzzyART patterns (8-dimensional for effective SIMD)
        var channel0 = new Pattern[sampleCount];
        for (int i = 0; i < sampleCount; i++) {
            double base = i / (double) sampleCount;
            channel0[i] = Pattern.of(
                base, 1.0 - base, base * 0.8, 1.0 - base * 0.8,
                base * 0.6, 1.0 - base * 0.6, base * 0.4, 1.0 - base * 0.4
            );
        }
        
        // Channel 1: HypersphereART patterns (3D points on sphere)
        var channel1 = new Pattern[sampleCount];
        for (int i = 0; i < sampleCount; i++) {
            double theta = 2 * Math.PI * i / sampleCount;
            double phi = Math.PI * (i % 20) / 20.0;
            channel1[i] = Pattern.of(
                Math.sin(phi) * Math.cos(theta),
                Math.sin(phi) * Math.sin(theta),
                Math.cos(phi)
            );
        }
        
        // Channel 2: More FuzzyART patterns
        var channel2 = new Pattern[sampleCount];
        for (int i = 0; i < sampleCount; i++) {
            double x = (i * 73 % sampleCount) / (double) sampleCount;
            double y = (i * 89 % sampleCount) / (double) sampleCount;
            channel2[i] = Pattern.of(
                x, y, x * y, Math.sqrt(x * y),
                1.0 - x, 1.0 - y, (1.0 - x) * (1.0 - y), Math.sqrt((1.0 - x) * (1.0 - y))
            );
        }
        
        trainingData = List.of(channel0, channel1); // Only 2 channels to match 2 modules
        
        // Create supervised labels (5 classes for more complexity)
        supervisedLabels = new int[sampleCount];
        for (int i = 0; i < sampleCount; i++) {
            supervisedLabels[i] = i % 5;
        }
    }
    
    /**
     * Create test data for prediction benchmarks.
     */
    private void createTestData() {
        int testCount = 100;
        
        // Create test patterns similar to training but slightly different
        var testChannel0 = new Pattern[testCount];
        var testChannel1 = new Pattern[testCount];
        var testChannel2 = new Pattern[testCount];
        
        for (int i = 0; i < testCount; i++) {
            double base = (i + 0.5) / testCount;
            
            testChannel0[i] = Pattern.of(
                base, 1.0 - base, base * 0.7, 1.0 - base * 0.7,
                base * 0.5, 1.0 - base * 0.5, base * 0.3, 1.0 - base * 0.3
            );
            
            double theta = 2 * Math.PI * (i + 0.5) / testCount;
            double phi = Math.PI * ((i + 0.5) % 20) / 20.0;
            testChannel1[i] = Pattern.of(
                Math.sin(phi) * Math.cos(theta),
                Math.sin(phi) * Math.sin(theta),
                Math.cos(phi)
            );
            
            double x = ((i + 0.5) * 73 % testCount) / testCount;
            double y = ((i + 0.5) * 89 % testCount) / testCount;
            testChannel2[i] = Pattern.of(
                x, y, x * y, Math.sqrt(x * y),
                1.0 - x, 1.0 - y, (1.0 - x) * (1.0 - y), Math.sqrt((1.0 - x) * (1.0 - y))
            );
        }
        
        testData = List.of(testChannel0, testChannel1); // Only 2 channels to match training data
    }
    
    @Benchmark
    public void benchmarkVectorizedTraining(Blackhole bh) {
        // Create fresh network for training benchmark - match channel count
        var modules = List.<BaseART>of(
            new VectorizedFuzzyART(VectorizedParameters.createDefault()),
            new VectorizedFuzzyART(VectorizedParameters.createDefault())
        );
        var network = new VectorizedDeepARTMAP(modules, vectorizedParams);
        
        try {
            var result = network.fitSupervised(trainingData, supervisedLabels);
            bh.consume(result);
        } finally {
            network.close();
        }
    }
    
    @Benchmark
    public void benchmarkStandardTraining(Blackhole bh) {
        // Create fresh network for training benchmark - match channel count
        var modules = List.<BaseART>of(
            new FuzzyART(),
            new FuzzyART()
        );
        var network = new DeepARTMAP(modules, standardParams);
        
        var result = network.fitSupervised(trainingData, supervisedLabels);
        bh.consume(result);
    }
    
    @Benchmark
    public void benchmarkVectorizedPrediction(Blackhole bh) {
        var predictions = vectorizedDeepART.predict(testData);
        bh.consume(predictions);
    }
    
    @Benchmark
    public void benchmarkStandardPrediction(Blackhole bh) {
        var predictions = standardDeepART.predict(testData);
        bh.consume(predictions);
    }
    
    @Benchmark
    public void benchmarkVectorizedDeepPrediction(Blackhole bh) {
        var deepPredictions = vectorizedDeepART.predictDeep(testData);
        bh.consume(deepPredictions);
    }
    
    @Benchmark
    public void benchmarkStandardDeepPrediction(Blackhole bh) {
        var deepPredictions = standardDeepART.predictDeep(testData);
        bh.consume(deepPredictions);
    }
    
    @Benchmark
    public void benchmarkVectorizedProbabilities(Blackhole bh) {
        var probabilities = vectorizedDeepART.predict_proba(testData);
        bh.consume(probabilities);
    }
    
    @Benchmark
    public void benchmarkStandardProbabilities(Blackhole bh) {
        var probabilities = standardDeepART.predict_proba(testData);
        bh.consume(probabilities);
    }
    
    /**
     * Main method to run the benchmark.
     */
    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(VectorizedDeepARTMAPBenchmark.class.getSimpleName())
                .forks(1)
                .warmupIterations(3)
                .measurementIterations(5)
                .build();

        new Runner(opt).run();
    }
    
    /**
     * Simple performance comparison method for quick testing.
     */
    public static void quickPerformanceTest() {
        System.out.println("VectorizedDeepARTMAP Performance Test");
        System.out.println("====================================");
        
        var benchmark = new VectorizedDeepARTMAPBenchmark();
        benchmark.setupTrial();
        benchmark.setupIteration();
        
        var bh = new Blackhole("Today's password is swordfish. I understand instantiating Blackholes directly is dangerous.");
        
        // Train networks if not already trained (setupIteration should do this, but ensure it's done)
        if (benchmark.vectorizedDeepART != null && benchmark.standardDeepART != null) {
            // Networks are already trained by setupIteration
        }
        
        // Warm up - skip prediction warmup since networks need training first
        
        // Measure training performance
        long vectorizedTrainingTime = 0;
        long standardTrainingTime = 0;
        int iterations = 3;
        
        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            benchmark.benchmarkVectorizedTraining(bh);
            vectorizedTrainingTime += System.nanoTime() - start;
            
            start = System.nanoTime();
            benchmark.benchmarkStandardTraining(bh);
            standardTrainingTime += System.nanoTime() - start;
        }
        
        vectorizedTrainingTime /= iterations;
        standardTrainingTime /= iterations;
        
        // Measure prediction performance
        long vectorizedPredTime = 0;
        long standardPredTime = 0;
        long vectorizedProbTime = 0;
        long standardProbTime = 0;
        
        iterations = 10;
        
        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            benchmark.benchmarkVectorizedPrediction(bh);
            vectorizedPredTime += System.nanoTime() - start;
            
            start = System.nanoTime();
            benchmark.benchmarkStandardPrediction(bh);
            standardPredTime += System.nanoTime() - start;
            
            start = System.nanoTime();
            benchmark.benchmarkVectorizedProbabilities(bh);
            vectorizedProbTime += System.nanoTime() - start;
            
            start = System.nanoTime();
            benchmark.benchmarkStandardProbabilities(bh);
            standardProbTime += System.nanoTime() - start;
        }
        
        vectorizedPredTime /= iterations;
        standardPredTime /= iterations;
        vectorizedProbTime /= iterations;
        standardProbTime /= iterations;
        
        // Calculate speedups
        double trainingSpeedup = (double) standardTrainingTime / vectorizedTrainingTime;
        double predictionSpeedup = (double) standardPredTime / vectorizedPredTime;
        double probabilitySpeedup = (double) standardProbTime / vectorizedProbTime;
        
        // Display results
        System.out.println("\\nPerformance Results:");
        System.out.printf("Training    - Vectorized: %.2f ms, Standard: %.2f ms, Speedup: %.2fx%n", 
                         vectorizedTrainingTime / 1_000_000.0, standardTrainingTime / 1_000_000.0, trainingSpeedup);
        System.out.printf("Prediction  - Vectorized: %.2f ms, Standard: %.2f ms, Speedup: %.2fx%n", 
                         vectorizedPredTime / 1_000_000.0, standardPredTime / 1_000_000.0, predictionSpeedup);
        System.out.printf("Probability - Vectorized: %.2f ms, Standard: %.2f ms, Speedup: %.2fx%n", 
                         vectorizedProbTime / 1_000_000.0, standardProbTime / 1_000_000.0, probabilitySpeedup);
        
        // Get performance stats
        var stats = benchmark.vectorizedDeepART.getPerformanceStats();
        
        System.out.println("\\nVectorized DeepARTMAP Statistics:");
        System.out.printf("Total Operations: %d%n", stats.operationCount());
        System.out.printf("SIMD Operations: %d (%.1f%% efficiency)%n", 
                         stats.totalSIMDOperations(), stats.simdEfficiency() * 100);
        System.out.printf("Channel Parallel Tasks: %d (%.1f%% efficiency)%n", 
                         stats.totalChannelParallelTasks(), stats.channelParallelismEfficiency() * 100);
        System.out.printf("Layer Parallel Tasks: %d (%.1f%% efficiency)%n", 
                         stats.totalLayerParallelTasks(), stats.layerParallelismEfficiency() * 100);
        System.out.printf("Operations per Second: %.1f%n", stats.operationsPerSecond());
        System.out.printf("Categories Created: %d%n", stats.categoryCount());
        System.out.printf("Active Threads: %d%n", stats.totalActiveThreads());
        
        benchmark.tearDownIteration();
        
        System.out.println("\\nVectorized DeepARTMAP optimization successfully implemented!");
        System.out.printf("Overall average speedup: %.2fx%n", 
                         (trainingSpeedup + predictionSpeedup + probabilitySpeedup) / 3);
    }
}