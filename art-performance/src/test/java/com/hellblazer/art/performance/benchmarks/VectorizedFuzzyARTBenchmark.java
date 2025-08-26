package com.hellblazer.art.performance.benchmarks;

import com.hellblazer.art.performance.algorithms.VectorizedFuzzyART;
import com.hellblazer.art.performance.algorithms.VectorizedParameters;
import com.hellblazer.art.core.*;
import com.hellblazer.art.core.algorithms.FuzzyART;
import com.hellblazer.art.core.parameters.FuzzyParameters;
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
 * Performance benchmark comparing SIMD vs standard computation in VectorizedFuzzyART.
 * 
 * This benchmark demonstrates the performance benefits of SIMD vectorization
 * in FuzzyART implementations using the Java Vector API.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Fork(value = 1)
@Warmup(iterations = 3, time = 2, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 3, timeUnit = TimeUnit.SECONDS)
public class VectorizedFuzzyARTBenchmark {
    
    private VectorizedFuzzyART simdART;
    private VectorizedFuzzyART standardART;
    private FuzzyART baselineFuzzyART;
    
    private VectorizedParameters simdParams;
    private VectorizedParameters standardParams;
    private FuzzyParameters baselineParams;
    
    private List<Pattern> trainingPatterns;
    private List<Pattern> testPatterns;
    
    @Setup(Level.Trial)
    public void setupTrial() {
        // SIMD-enabled parameters
        simdParams = new VectorizedParameters(
            0.8,    // vigilanceThreshold
            0.1,    // learningRate  
            0.001,  // alpha
            4,      // parallelismLevel
            50,     // parallelThreshold
            1000,   // maxCacheSize
            true,   // enableSIMD
            false,  // enableJOML
            0.8     // memoryOptimizationThreshold
        );
        
        // Standard (non-SIMD) parameters
        standardParams = new VectorizedParameters(
            0.8,    // vigilanceThreshold
            0.1,    // learningRate
            0.001,  // alpha
            4,      // parallelismLevel
            50,     // parallelThreshold
            1000,   // maxCacheSize
            false,  // enableSIMD - DISABLED
            false,  // enableJOML
            0.8     // memoryOptimizationThreshold
        );
        
        // Baseline FuzzyART parameters
        baselineParams = new FuzzyParameters(0.8, 0.001, 0.1);
        
        // Create training patterns (8-dimensional for effective SIMD)
        trainingPatterns = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            double base = i / 100.0;
            var pattern = Pattern.of(
                base, 1.0 - base, 
                base * 0.8, 1.0 - base * 0.8,
                base * 0.6, 1.0 - base * 0.6,
                base * 0.4, 1.0 - base * 0.4
            );
            trainingPatterns.add(pattern);
        }
        
        // Create test patterns for prediction
        testPatterns = new ArrayList<>();
        for (int i = 0; i < 50; i++) {
            double base = (i + 0.5) / 50.0;
            var pattern = Pattern.of(
                base, 1.0 - base,
                base * 0.9, 1.0 - base * 0.9,
                base * 0.7, 1.0 - base * 0.7,
                base * 0.5, 1.0 - base * 0.5
            );
            testPatterns.add(pattern);
        }
    }
    
    @Setup(Level.Iteration)
    public void setupIteration() {
        // Create fresh ART instances for each iteration
        simdART = new VectorizedFuzzyART(simdParams);
        standardART = new VectorizedFuzzyART(standardParams);
        baselineFuzzyART = new FuzzyART();
        
        // Pre-train all networks with the same patterns for fair comparison
        for (var pattern : trainingPatterns) {
            simdART.stepFit(pattern, simdParams);
            standardART.stepFit(pattern, standardParams);
            baselineFuzzyART.stepFit(pattern, baselineParams);
        }
    }
    
    @TearDown(Level.Iteration)
    public void tearDownIteration() {
        simdART.close();
        standardART.close();
        // baselineFuzzyART doesn't need explicit cleanup
    }
    
    @Benchmark
    public void benchmarkSIMDPrediction(Blackhole bh) {
        for (var pattern : testPatterns) {
            var result = simdART.stepFitEnhanced(pattern, simdParams);
            bh.consume(result);
        }
    }
    
    @Benchmark
    public void benchmarkStandardPrediction(Blackhole bh) {
        for (var pattern : testPatterns) {
            var result = standardART.stepFitEnhanced(pattern, standardParams);
            bh.consume(result);
        }
    }
    
    @Benchmark
    public void benchmarkBaselineFuzzyART(Blackhole bh) {
        for (var pattern : testPatterns) {
            var result = baselineFuzzyART.stepFit(pattern, baselineParams);
            bh.consume(result);
        }
    }
    
    @Benchmark
    public void benchmarkSIMDTraining(Blackhole bh) {
        var art = new VectorizedFuzzyART(simdParams);
        for (var pattern : trainingPatterns) {
            var result = art.stepFit(pattern, simdParams);
            bh.consume(result);
        }
        art.close();
    }
    
    @Benchmark
    public void benchmarkStandardTraining(Blackhole bh) {
        var art = new VectorizedFuzzyART(standardParams);
        for (var pattern : trainingPatterns) {
            var result = art.stepFit(pattern, standardParams);
            bh.consume(result);
        }
        art.close();
    }
    
    @Benchmark
    public void benchmarkBaselineTraining(Blackhole bh) {
        var art = new FuzzyART();
        for (var pattern : trainingPatterns) {
            var result = art.stepFit(pattern, baselineParams);
            bh.consume(result);
        }
    }
    
    /**
     * Main method to run the benchmark.
     */
    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(VectorizedFuzzyARTBenchmark.class.getSimpleName())
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
        System.out.println("VectorizedFuzzyART Performance Test");
        System.out.println("===================================");
        
        var benchmark = new VectorizedFuzzyARTBenchmark();
        benchmark.setupTrial();
        benchmark.setupIteration();
        
        var bh = new Blackhole("Today's password is swordfish. I understand instantiating Blackholes directly is dangerous.");
        
        // Warm up
        for (int i = 0; i < 10; i++) {
            benchmark.benchmarkSIMDPrediction(bh);
            benchmark.benchmarkStandardPrediction(bh);
        }
        
        // Measure SIMD performance
        long simdStart = System.nanoTime();
        for (int i = 0; i < 100; i++) {
            benchmark.benchmarkSIMDPrediction(bh);
        }
        long simdTime = System.nanoTime() - simdStart;
        
        // Measure standard performance
        long standardStart = System.nanoTime();
        for (int i = 0; i < 100; i++) {
            benchmark.benchmarkStandardPrediction(bh);
        }
        long standardTime = System.nanoTime() - standardStart;
        
        double speedup = (double) standardTime / simdTime;
        
        System.out.printf("SIMD Time: %.2f ms%n", simdTime / 1_000_000.0);
        System.out.printf("Standard Time: %.2f ms%n", standardTime / 1_000_000.0);
        System.out.printf("Speedup: %.2fx%n", speedup);
        
        // Get performance stats
        var simdStats = benchmark.simdART.getPerformanceStats();
        var standardStats = benchmark.standardART.getPerformanceStats();
        
        System.out.println("\nPerformance Statistics:");
        System.out.printf("SIMD Vector Operations: %d%n", simdStats.totalVectorOperations());
        System.out.printf("Standard Vector Operations: %d%n", standardStats.totalVectorOperations());
        System.out.printf("SIMD Categories: %d%n", simdStats.categoryCount());
        System.out.printf("Standard Categories: %d%n", standardStats.categoryCount());
        
        benchmark.tearDownIteration();
        
        System.out.println("\nSIMD optimization successfully implemented!");
    }
}