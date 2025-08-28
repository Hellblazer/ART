package com.hellblazer.art.performance.benchmarks;

import com.hellblazer.art.core.algorithms.TopoART;
import com.hellblazer.art.performance.algorithms.VectorizedTopoART;
import com.hellblazer.art.core.parameters.TopoARTParameters;
import com.hellblazer.art.core.utils.MathOperations;
import com.hellblazer.art.performance.algorithms.VectorizedMathOperations;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * JMH performance benchmarks comparing standard and vectorized TopoART implementations.
 * 
 * Benchmarks measure:
 * - Mathematical operations (complement coding, activation, vigilance)
 * - Single pattern learning performance
 * - Batch learning throughput
 * - Memory usage patterns
 * - Clustering performance
 * 
 * Run with: mvn exec:java -Dexec.mainClass="com.hellblazer.art.core.benchmarks.TopoARTBenchmark"
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@State(Scope.Benchmark)
@Fork(1)
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 3)
public class TopoARTBenchmark {
    
    // Test parameters
    private static final int INPUT_DIMENSION = 32;
    private static final int COMPLEMENT_DIMENSION = INPUT_DIMENSION * 2;
    private static final int NUM_PATTERNS = 1000;
    private static final int BATCH_SIZE = 100;
    
    // Network parameters
    private TopoARTParameters parameters;
    
    // Test data
    private double[][] testPatterns;
    private double[] singlePattern;
    private double[] complementSinglePattern;
    private double[] weights1;
    private double[] weights2;
    
    // Network instances
    private TopoART standardTopoART;
    private VectorizedTopoART vectorizedTopoART;
    
    // Random number generator
    private Random random;
    
    @Setup
    public void setup() {
        random = new Random(42); // Fixed seed for reproducibility
        
        // Create network parameters
        parameters = TopoARTParameters.builder()
            .inputDimension(INPUT_DIMENSION)
            .vigilanceA(0.75)
            .learningRateSecond(0.5)
            .alpha(0.001)
            .phi(5)
            .tau(100)
            .build();
        
        // Initialize networks
        standardTopoART = new TopoART(parameters);
        vectorizedTopoART = new VectorizedTopoART(parameters);
        
        // Generate test patterns
        generateTestData();
        
        System.out.printf("TopoART Benchmark Setup:\n");
        System.out.printf("  Input Dimension: %d\n", INPUT_DIMENSION);
        System.out.printf("  Test Patterns: %d\n", NUM_PATTERNS);
        System.out.printf("  Vectorization Available: %s\n", VectorizedTopoART.isVectorizedSupported());
        if (VectorizedTopoART.isVectorizedSupported()) {
            System.out.printf("  Vector Info: %s\n", VectorizedTopoART.getVectorInfo());
        }
        System.out.println();
    }
    
    private void generateTestData() {
        // Generate realistic test patterns with some structure
        testPatterns = new double[NUM_PATTERNS][];
        
        for (int i = 0; i < NUM_PATTERNS; i++) {
            var pattern = new double[INPUT_DIMENSION];
            
            // Generate patterns with some clustering structure
            int cluster = i % 5; // 5 clusters
            double baseValue = cluster / 5.0;
            
            for (int j = 0; j < INPUT_DIMENSION; j++) {
                // Add some noise around the base value
                pattern[j] = Math.max(0.0, Math.min(1.0, 
                    baseValue + 0.3 * (random.nextGaussian() * 0.2)));
            }
            
            testPatterns[i] = pattern;
        }
        
        // Single pattern for focused benchmarks
        singlePattern = testPatterns[0];
        complementSinglePattern = MathOperations.complementCode(singlePattern);
        
        // Weight vectors for math operation benchmarks
        weights1 = new double[COMPLEMENT_DIMENSION];
        weights2 = new double[COMPLEMENT_DIMENSION];
        
        for (int i = 0; i < COMPLEMENT_DIMENSION; i++) {
            weights1[i] = random.nextDouble();
            weights2[i] = random.nextDouble();
        }
    }
    
    // =========================================================================================
    // Mathematical Operations Benchmarks
    // =========================================================================================
    
    @Benchmark
    public double[] standardComplementCoding(Blackhole blackhole) {
        return MathOperations.complementCode(singlePattern);
    }
    
    @Benchmark
    public double[] vectorizedComplementCoding(Blackhole blackhole) {
        return VectorizedMathOperations.complementCode(singlePattern);
    }
    
    @Benchmark
    public double[] standardComponentWiseMin(Blackhole blackhole) {
        return MathOperations.componentWiseMin(complementSinglePattern, weights1);
    }
    
    @Benchmark
    public double[] vectorizedComponentWiseMin(Blackhole blackhole) {
        return VectorizedMathOperations.componentWiseMin(complementSinglePattern, weights1);
    }
    
    @Benchmark
    public double standardActivation(Blackhole blackhole) {
        return MathOperations.activation(complementSinglePattern, weights1, parameters.alpha());
    }
    
    @Benchmark
    public double vectorizedActivation(Blackhole blackhole) {
        return VectorizedMathOperations.activation(complementSinglePattern, weights1, parameters.alpha());
    }
    
    @Benchmark
    public boolean standardVigilance(Blackhole blackhole) {
        return MathOperations.matchFunction(complementSinglePattern, weights1, parameters.vigilanceA());
    }
    
    @Benchmark
    public boolean vectorizedVigilance(Blackhole blackhole) {
        return VectorizedMathOperations.matchFunction(complementSinglePattern, weights1, parameters.vigilanceA());
    }
    
    // =========================================================================================
    // Learning Performance Benchmarks  
    // =========================================================================================
    
    @Benchmark
    public void standardSinglePatternLearning(Blackhole blackhole) {
        standardTopoART = new TopoART(parameters);
        standardTopoART.learn(singlePattern);
        blackhole.consume(standardTopoART);
    }
    
    @Benchmark
    public void vectorizedSinglePatternLearning(Blackhole blackhole) {
        vectorizedTopoART = new VectorizedTopoART(parameters);
        var result = vectorizedTopoART.learn(singlePattern);
        blackhole.consume(result);
    }
    
    @Benchmark
    @OperationsPerInvocation(BATCH_SIZE)
    public void standardBatchLearning(Blackhole blackhole) {
        standardTopoART = new TopoART(parameters);
        for (int i = 0; i < BATCH_SIZE; i++) {
            standardTopoART.learn(testPatterns[i % testPatterns.length]);
            blackhole.consume(i);
        }
    }
    
    @Benchmark
    @OperationsPerInvocation(BATCH_SIZE)
    public void vectorizedBatchLearning(Blackhole blackhole) {
        vectorizedTopoART = new VectorizedTopoART(parameters);
        for (int i = 0; i < BATCH_SIZE; i++) {
            var result = vectorizedTopoART.learn(testPatterns[i % testPatterns.length]);
            blackhole.consume(result);
        }
    }
    
    /**
     * Main method to run benchmarks standalone.
     */
    public static void main(String[] args) throws Exception {
        Options opt = new OptionsBuilder()
            .include(TopoARTBenchmark.class.getSimpleName())
            .forks(1)
            .warmupIterations(3)
            .measurementIterations(5)
            .shouldFailOnError(true)
            .jvmArgs("-Xmx4g", "--enable-preview", "--add-modules", "jdk.incubator.vector")
            .build();
        
        System.out.println("Starting TopoART Performance Benchmarks...");
        System.out.println("This will take several minutes to complete.");
        System.out.println();
        
        new Runner(opt).run();
        
        System.out.println();
        System.out.println("Benchmark completed. Results show operations per second (higher is better).");
        System.out.println("Compare 'standard' vs 'vectorized' implementations for performance gains.");
    }
}