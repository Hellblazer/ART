package com.hellblazer.art.temporal.performance;

import com.hellblazer.art.temporal.dynamics.*;
import com.hellblazer.art.temporal.memory.*;
import com.hellblazer.art.temporal.masking.*;
import com.hellblazer.art.temporal.core.ActivationState;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * JMH benchmarks comparing standard vs vectorized implementations.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Fork(1)
@Warmup(iterations = 3)
@Measurement(iterations = 5)
public class PerformanceBenchmark {

    @Param({"10", "50", "100", "500"})
    private int dimension;

    @Param({"100", "1000"})
    private int iterations;

    private ShuntingDynamicsImpl standardShunting;
    private VectorizedShuntingDynamics vectorizedShunting;

    private WorkingMemory standardMemory;
    private VectorizedWorkingMemory vectorizedMemory;

    private MaskingField standardMasking;
    private VectorizedMaskingField vectorizedMasking;

    private double[] testInput;
    private List<double[]> testPatterns;
    private ActivationState testState;

    @Setup
    public void setup() {
        // Initialize shunting dynamics
        var shuntingParams = ShuntingParameters.competitiveDefaults(dimension);
        standardShunting = new ShuntingDynamicsImpl(shuntingParams, dimension);
        vectorizedShunting = new VectorizedShuntingDynamics(shuntingParams, dimension);

        // Initialize working memory
        var memoryParams = WorkingMemoryParameters.paperDefaults();
        standardMemory = new WorkingMemory(memoryParams);
        vectorizedMemory = new VectorizedWorkingMemory(memoryParams);

        // Initialize masking field
        var maskingParams = MaskingFieldParameters.listLearningDefaults();
        standardMasking = new MaskingField(maskingParams, standardMemory);
        vectorizedMasking = new VectorizedMaskingField(maskingParams, vectorizedMemory);

        // Create test data
        Random rand = new Random(42);
        testInput = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            testInput[i] = rand.nextDouble();
        }

        testPatterns = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            double[] pattern = new double[dimension];
            for (int j = 0; j < dimension; j++) {
                pattern[j] = rand.nextDouble();
            }
            testPatterns.add(pattern);
        }

        testState = new ActivationState(testInput);
    }

    // ===== SHUNTING DYNAMICS BENCHMARKS =====

    @Benchmark
    public ActivationState standardShuntingEvolution() {
        standardShunting.setExcitatoryInput(testInput);
        ActivationState state = testState;
        for (int i = 0; i < iterations; i++) {
            state = standardShunting.evolve(state, 0.01);
        }
        return state;
    }

    @Benchmark
    public ActivationState vectorizedShuntingEvolution() {
        vectorizedShunting.setExcitatoryInput(testInput);
        ActivationState state = testState;
        for (int i = 0; i < iterations; i++) {
            state = vectorizedShunting.evolve(state, 0.01);
        }
        return state;
    }

    @Benchmark
    public double standardShuntingEnergy() {
        standardShunting.setState(testState);
        return standardShunting.computeEnergy();
    }

    @Benchmark
    public double vectorizedShuntingEnergy() {
        vectorizedShunting.setState(testState);
        return vectorizedShunting.computeEnergyVectorized();
    }

    @Benchmark
    public boolean standardShuntingConvergence() {
        standardShunting.setState(testState);
        return standardShunting.hasConverged(0.001);
    }

    @Benchmark
    public boolean vectorizedShuntingConvergence() {
        vectorizedShunting.setState(testState);
        return vectorizedShunting.hasConvergedVectorized(0.001);
    }

    // ===== WORKING MEMORY BENCHMARKS =====

    @Benchmark
    public void standardMemoryStore() {
        standardMemory.reset();
        for (int i = 0; i < testPatterns.size(); i++) {
            standardMemory.storeItem(testPatterns.get(i), i * 0.1);
        }
    }

    @Benchmark
    public void vectorizedMemoryStore() {
        vectorizedMemory.reset();
        for (int i = 0; i < testPatterns.size(); i++) {
            vectorizedMemory.storeItem(testPatterns.get(i), i * 0.1);
        }
    }

    @Benchmark
    public void standardMemoryEvolution() {
        // evolveDynamics is private, store items to trigger evolution
        for (int i = 0; i < iterations; i++) {
            double[] pattern = new double[dimension];
            pattern[i % dimension] = 1.0;
            standardMemory.storeItem(pattern, i * 0.001);
        }
    }

    @Benchmark
    public void vectorizedMemoryEvolution() {
        // evolveDynamics is private, store items to trigger evolution
        for (int i = 0; i < iterations; i++) {
            double[] pattern = new double[dimension];
            pattern[i % dimension] = 1.0;
            vectorizedMemory.storeItem(pattern, i * 0.001);
        }
    }

    @Benchmark
    public WorkingMemory.TemporalPattern standardMemoryRetrieval() {
        return standardMemory.getTemporalPattern();
    }

    @Benchmark
    public TemporalPattern vectorizedMemoryRetrieval() {
        return vectorizedMemory.getTemporalPattern();
    }

    // ===== MASKING FIELD BENCHMARKS =====

    @Benchmark
    public void standardMaskingProcess() {
        var wmPattern = standardMemory.getTemporalPattern();
        // Convert to standalone TemporalPattern
        var patterns = new ArrayList<double[]>();
        patterns.add(wmPattern.getCombinedPattern());
        var weights = new ArrayList<Double>();
        weights.add(1.0);
        var pattern = new TemporalPattern(patterns, weights, 0.3);
        standardMasking.processTemporalPattern(pattern);
    }

    @Benchmark
    public void vectorizedMaskingProcess() {
        var pattern = vectorizedMemory.getTemporalPattern();
        // VectorizedWorkingMemory already returns standalone TemporalPattern
        vectorizedMasking.processTemporalPattern(pattern);
    }

    // ===== MULTI-SCALE BENCHMARKS =====

    @Benchmark
    public void standardMultiScale() {
        var params = MultiScaleParameters.defaults(dimension);
        var dynamics = new MultiScaleDynamics(params);

        for (int i = 0; i < iterations; i++) {
            dynamics.update(testInput, 0.001);
        }
    }

    @Benchmark
    @Threads(4)  // Test with multiple threads
    public void vectorizedMultiScaleParallel() {
        // Simulate parallel processing of multiple sequences
        var params = MultiScaleParameters.defaults(dimension);
        var dynamics = new MultiScaleDynamics(params);

        for (int i = 0; i < iterations; i++) {
            dynamics.update(testInput, 0.001);
        }
    }

    // ===== MAIN METHOD FOR STANDALONE EXECUTION =====

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
            .include(PerformanceBenchmark.class.getSimpleName())
            .forks(1)
            .build();

        new Runner(opt).run();
    }
}