package com.hellblazer.art.performance;

import com.hellblazer.art.algorithms.*;
import com.hellblazer.art.core.*;
import com.hellblazer.art.supervised.*;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.List;
import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * JMH benchmarks for VectorizedARTMAP performance analysis.
 * Measures training and prediction performance across different configurations,
 * dataset sizes, and parallelism settings.
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 2, timeUnit = TimeUnit.SECONDS)
@Fork(1)
public class VectorizedARTMAPBenchmark {
    
    @Param({"100", "1000", "5000"})
    private int datasetSize;
    
    @Param({"3", "5", "10"})
    private int numClasses;
    
    @Param({"true", "false"})
    private boolean enableSIMD;
    
    @Param({"true", "false"})
    private boolean enableJOML;
    
    @Param({"true", "false"})
    private boolean enableParallelSearch;
    
    private VectorizedARTMAP artmap;
    private VectorizedARTMAP artmapOptimized;
    private VectorizedARTMAP artmapBaseline;
    private List<TrainingSample> trainingData;
    private List<Pattern> predictionInputs;
    private Random random;
    
    @Setup(Level.Trial)
    public void setupTrial() {
        random = new Random(42); // Fixed seed for reproducible benchmarks
        
        // Generate training data
        trainingData = generateTrainingData(datasetSize, numClasses);
        
        // Generate prediction inputs (subset of training data)
        predictionInputs = new ArrayList<>();
        for (int i = 0; i < Math.min(100, datasetSize); i++) {
            predictionInputs.add(trainingData.get(i).input());
        }
    }
    
    @Setup(Level.Iteration)
    public void setupIteration() {
        // Create ARTMAP instances with different configurations
        artmap = createARTMAP(enableSIMD, enableJOML, enableParallelSearch);
        artmapOptimized = createOptimizedARTMAP();
        artmapBaseline = createBaselineARTMAP();
    }
    
    @TearDown(Level.Iteration)
    public void tearDownIteration() {
        closeARTMAP(artmap);
        closeARTMAP(artmapOptimized);
        closeARTMAP(artmapBaseline);
    }
    
    // ================== Training Benchmarks ==================
    
    @Benchmark
    public void trainingSingle(Blackhole bh) {
        var sample = trainingData.get(random.nextInt(trainingData.size()));
        var result = artmap.train(sample.input(), sample.target());
        bh.consume(result);
    }
    
    @Benchmark
    public void trainingBatch(Blackhole bh) {
        for (var sample : trainingData) {
            var result = artmap.train(sample.input(), sample.target());
            bh.consume(result);
        }
    }
    
    @Benchmark
    public void trainingOptimized(Blackhole bh) {
        for (var sample : trainingData) {
            var result = artmapOptimized.train(sample.input(), sample.target());
            bh.consume(result);
        }
    }
    
    @Benchmark
    public void trainingBaseline(Blackhole bh) {
        for (var sample : trainingData) {
            var result = artmapBaseline.train(sample.input(), sample.target());
            bh.consume(result);
        }
    }
    
    // ================== Prediction Benchmarks ==================
    
    @Benchmark
    public void predictionSingle(Blackhole bh) {
        // Pre-train the model
        var trainedArtmap = createARTMAP(enableSIMD, enableJOML, enableParallelSearch);
        for (int i = 0; i < Math.min(1000, trainingData.size()); i++) {
            var sample = trainingData.get(i);
            trainedArtmap.train(sample.input(), sample.target());
        }
        
        var input = predictionInputs.get(random.nextInt(predictionInputs.size()));
        var prediction = trainedArtmap.predict(input);
        bh.consume(prediction);
        
        closeARTMAP(trainedArtmap);
    }
    
    @Benchmark
    public void predictionBatch(Blackhole bh) {
        // Pre-train the model
        var trainedArtmap = createARTMAP(enableSIMD, enableJOML, enableParallelSearch);
        for (int i = 0; i < Math.min(1000, trainingData.size()); i++) {
            var sample = trainingData.get(i);
            trainedArtmap.train(sample.input(), sample.target());
        }
        
        for (var input : predictionInputs) {
            var prediction = trainedArtmap.predict(input);
            bh.consume(prediction);
        }
        
        closeARTMAP(trainedArtmap);
    }
    
    // ================== Pattern Operations Benchmarks ==================
    
    @Benchmark
    public void vectorOperations(Blackhole bh) {
        var vector1 = Pattern.of(random.nextDouble(), random.nextDouble(), random.nextDouble());
        var vector2 = Pattern.of(random.nextDouble(), random.nextDouble(), random.nextDouble());
        
        // Test various pattern operations
        var l1Norm = vector1.l1Norm();
        var l2Norm = vector1.l2Norm();
        var scaled = vector1.scale(0.5);
        
        bh.consume(l1Norm);
        bh.consume(l2Norm);
        bh.consume(scaled);
    }
    
    @Benchmark
    public void vectorCreation(Blackhole bh) {
        var values = new double[3];
        for (int i = 0; i < values.length; i++) {
            values[i] = random.nextDouble();
        }
        
        var vector = Pattern.of(values);
        bh.consume(vector);
    }
    
    // ================== Match Tracking Benchmarks ==================
    
    @Benchmark
    public void matchTrackingScenario(Blackhole bh) {
        var artmapWithTracking = createARTMAP(enableSIMD, enableJOML, true); // Force match tracking
        
        // Create scenario that will trigger match tracking
        var input = Pattern.of(0.5, 0.5, 0.5);
        var target1 = Pattern.of(1.0);
        var target2 = Pattern.of(0.0);
        
        var result1 = artmapWithTracking.train(input, target1);
        var result2 = artmapWithTracking.train(input, target2); // Different target, same input
        
        bh.consume(result1);
        bh.consume(result2);
        
        closeARTMAP(artmapWithTracking);
    }
    
    // ================== Memory Usage Benchmarks ==================
    
    @Benchmark
    public void memoryEfficiencyTest(Blackhole bh) {
        var memoryTestArtmap = createARTMAP(enableSIMD, enableJOML, enableParallelSearch);
        
        // Train with many samples to test memory efficiency
        for (int i = 0; i < Math.min(5000, datasetSize); i++) {
            var sample = trainingData.get(i % trainingData.size());
            var result = memoryTestArtmap.train(sample.input(), sample.target());
            bh.consume(result);
        }
        
        // Test memory state
        var categoryCount = memoryTestArtmap.getArtA().getCategoryCount();
        var mapFieldSize = memoryTestArtmap.getMapField().size();
        
        bh.consume(categoryCount);
        bh.consume(mapFieldSize);
        
        closeARTMAP(memoryTestArtmap);
    }
    
    // ================== Concurrent Performance Benchmarks ==================
    
    @Benchmark
    @Threads(4)
    public void concurrentTraining(Blackhole bh) {
        var sample = trainingData.get(random.nextInt(trainingData.size()));
        var result = artmap.train(sample.input(), sample.target());
        bh.consume(result);
    }
    
    @Benchmark
    @Threads(4)
    public void concurrentPrediction(Blackhole bh) {
        // Pre-train the model once
        for (int i = 0; i < Math.min(100, trainingData.size()); i++) {
            var sample = trainingData.get(i);
            artmap.train(sample.input(), sample.target());
        }
        
        var input = predictionInputs.get(random.nextInt(predictionInputs.size()));
        var prediction = artmap.predict(input);
        bh.consume(prediction);
    }
    
    // ================== Optimization Comparison Benchmarks ==================
    
    @Benchmark
    public void simdVsNoSimd(Blackhole bh) {
        var simdArtmap = createARTMAP(true, enableJOML, enableParallelSearch);
        var noSimdArtmap = createARTMAP(false, enableJOML, enableParallelSearch);
        
        var sample = trainingData.get(random.nextInt(trainingData.size()));
        
        var simdResult = simdArtmap.train(sample.input(), sample.target());
        var noSimdResult = noSimdArtmap.train(sample.input(), sample.target());
        
        bh.consume(simdResult);
        bh.consume(noSimdResult);
        
        closeARTMAP(simdArtmap);
        closeARTMAP(noSimdArtmap);
    }
    
    @Benchmark
    public void jomlVsNoJoml(Blackhole bh) {
        var jomlArtmap = createARTMAP(enableSIMD, true, enableParallelSearch);
        var noJomlArtmap = createARTMAP(enableSIMD, false, enableParallelSearch);
        
        var sample = trainingData.get(random.nextInt(trainingData.size()));
        
        var jomlResult = jomlArtmap.train(sample.input(), sample.target());
        var noJomlResult = noJomlArtmap.train(sample.input(), sample.target());
        
        bh.consume(jomlResult);
        bh.consume(noJomlResult);
        
        closeARTMAP(jomlArtmap);
        closeARTMAP(noJomlArtmap);
    }
    
    // ================== Helper Methods ==================
    
    private List<TrainingSample> generateTrainingData(int size, int classes) {
        var data = new ArrayList<TrainingSample>();
        
        for (int i = 0; i < size; i++) {
            var classId = random.nextInt(classes);
            
            // Generate class-specific patterns with some noise
            var input = Pattern.of(
                classId * 0.3 + random.nextGaussian() * 0.1,
                Math.sin(classId * Math.PI / classes) + random.nextGaussian() * 0.1,
                Math.cos(classId * Math.PI / classes) + random.nextGaussian() * 0.1
            );
            
            var target = Pattern.of(classId);
            data.add(new TrainingSample(input, target));
        }
        
        return data;
    }
    
    private VectorizedARTMAP createARTMAP(boolean simd, boolean joml, boolean parallel) {
        var artAParams = VectorizedParameters.createDefault()
            .withVigilance(0.7)
            .withCacheSettings(1000, simd, joml);
        
        var artBParams = VectorizedParameters.createDefault()
            .withVigilance(0.8)
            .withCacheSettings(1000, simd, joml);
        
        var artmapParams = VectorizedARTMAPParameters.builder()
            .mapVigilance(0.9)
            .baselineVigilance(0.0)
            .vigilanceIncrement(0.05)
            .maxVigilance(0.95)
            .enableMatchTracking(true)
            .enableParallelSearch(parallel)
            .maxSearchAttempts(10)
            .artAParams(artAParams)
            .artBParams(artBParams)
            .build();
        
        return new VectorizedARTMAP(artmapParams);
    }
    
    private VectorizedARTMAP createOptimizedARTMAP() {
        // Fully optimized configuration
        return createARTMAP(true, true, true);
    }
    
    private VectorizedARTMAP createBaselineARTMAP() {
        // Minimal optimization configuration
        return createARTMAP(false, false, false);
    }
    
    private void closeARTMAP(VectorizedARTMAP artmap) {
        if (artmap instanceof AutoCloseable) {
            try {
                ((AutoCloseable) artmap).close();
            } catch (Exception e) {
                // Log error but don't fail benchmark
            }
        }
    }
    
    /**
     * Record class for training samples
     */
    private record TrainingSample(Pattern input, Pattern target) {}
}