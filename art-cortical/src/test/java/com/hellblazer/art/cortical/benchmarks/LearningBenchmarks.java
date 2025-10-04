package com.hellblazer.art.cortical.benchmarks;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.cortical.layers.WeightMatrix;
import com.hellblazer.art.cortical.learning.*;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * JMH Benchmarks for Learning Rule Performance - Phase 4A.
 *
 * <h2>Benchmark Objectives</h2>
 * <ul>
 *   <li>Measure weight update latency for each learning rule</li>
 *   <li>Identify opportunities for vectorization</li>
 *   <li>Establish learning throughput baselines</li>
 *   <li>Measure memory allocation during learning</li>
 * </ul>
 *
 * <h2>Benchmark Categories</h2>
 * <ol>
 *   <li><b>Hebbian Learning</b>: Classic Hebbian plasticity</li>
 *   <li><b>BCM Learning</b>: Sliding threshold adaptation</li>
 *   <li><b>InstarOutstar Learning</b>: ART-style learning</li>
 *   <li><b>Resonance-Gated Learning</b>: Consciousness-modulated</li>
 * </ol>
 *
 * <h2>Running Benchmarks</h2>
 * <pre>
 * # Run all learning benchmarks
 * mvn test -Dtest=LearningBenchmarks
 *
 * # With allocation profiler
 * mvn test -Dtest=LearningBenchmarks -Djmh.prof=gc
 * </pre>
 *
 * @author Phase 4A: Benchmarking Infrastructure
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 2, timeUnit = TimeUnit.SECONDS)
@Fork(value = 1, jvmArgs = {"-Xmx2G"})
public class LearningBenchmarks {

    @Param({"32", "64", "128", "256"})
    private int preSize;

    @Param({"32", "64", "128"})
    private int postSize;

    private Pattern preActivation;
    private Pattern postActivation;
    private WeightMatrix weights;

    private HebbianLearning hebbianLearning;
    private BCMLearning bcmLearning;
    private InstarOutstarLearning instarLearning;
    private InstarOutstarLearning outstarLearning;
    private InstarOutstarLearning bidirectionalLearning;

    private static final double LEARNING_RATE = 0.1;

    @Setup(Level.Trial)
    public void setup() {
        var random = new Random(42);

        // Create random activation patterns
        var preValues = new double[preSize];
        var postValues = new double[postSize];
        for (int i = 0; i < preSize; i++) {
            preValues[i] = random.nextDouble();
        }
        for (int j = 0; j < postSize; j++) {
            postValues[j] = random.nextDouble();
        }

        preActivation = new DenseVector(preValues);
        postActivation = new DenseVector(postValues);

        // Create weight matrix
        weights = new WeightMatrix(postSize, preSize);
        for (int j = 0; j < postSize; j++) {
            for (int i = 0; i < preSize; i++) {
                weights.set(j, i, random.nextDouble());
            }
        }

        // Create learning rules
        hebbianLearning = new HebbianLearning(0.0001, 0.0, 1.0);
        bcmLearning = BCMLearning.createBalanced();
        instarLearning = InstarOutstarLearning.createInstar();
        outstarLearning = InstarOutstarLearning.createOutstar();
        bidirectionalLearning = InstarOutstarLearning.createBidirectional();
    }

    /**
     * Benchmark Hebbian learning weight update.
     *
     * <p>Hebbian rule: Δw = α × x ⊗ y - β × w
     *
     * <p>Expected: ~100-500 ns/update (depends on matrix size)
     */
    @Benchmark
    public void benchmarkHebbianLearning(Blackhole bh) {
        var updated = hebbianLearning.update(
            preActivation,
            postActivation,
            weights,
            LEARNING_RATE
        );
        bh.consume(updated);
    }

    /**
     * Benchmark BCM learning weight update.
     *
     * <p>BCM rule: Δw = α × φ(y, θ) × x where φ(y, θ) = y × (y - θ)
     *
     * <p>Expected: ~200-1000 ns/update (includes threshold adaptation)
     */
    @Benchmark
    public void benchmarkBCMLearning(Blackhole bh) {
        var updated = bcmLearning.update(
            preActivation,
            postActivation,
            weights,
            LEARNING_RATE
        );
        bh.consume(updated);
    }

    /**
     * Benchmark Instar learning weight update.
     *
     * <p>Instar rule: Δw = α × y × (x - w) (bottom-up recognition)
     *
     * <p>Expected: ~150-600 ns/update
     */
    @Benchmark
    public void benchmarkInstarLearning(Blackhole bh) {
        var updated = instarLearning.update(
            preActivation,
            postActivation,
            weights,
            LEARNING_RATE
        );
        bh.consume(updated);
    }

    /**
     * Benchmark Outstar learning weight update.
     *
     * <p>Outstar rule: Δw = α × y × (x - w) (top-down prediction)
     *
     * <p>Expected: ~150-600 ns/update
     */
    @Benchmark
    public void benchmarkOutstarLearning(Blackhole bh) {
        var updated = outstarLearning.update(
            preActivation,
            postActivation,
            weights,
            LEARNING_RATE
        );
        bh.consume(updated);
    }

    /**
     * Benchmark Bidirectional learning weight update.
     *
     * <p>Both instar and outstar combined.
     *
     * <p>Expected: ~300-1200 ns/update (2x instar cost)
     */
    @Benchmark
    public void benchmarkBidirectionalLearning(Blackhole bh) {
        var updated = bidirectionalLearning.update(
            preActivation,
            postActivation,
            weights,
            LEARNING_RATE
        );
        bh.consume(updated);
    }

    /**
     * Benchmark weight matrix allocation.
     *
     * <p>Isolates the allocation cost to understand memory overhead.
     *
     * <p>Expected: ~50-200 ns/allocation
     */
    @Benchmark
    public void benchmarkWeightMatrixAllocation(Blackhole bh) {
        var newWeights = new WeightMatrix(postSize, preSize);
        bh.consume(newWeights);
    }

    /**
     * Benchmark weight matrix copy.
     *
     * <p>Measures the cost of copying weight values.
     *
     * <p>Expected: ~100-400 ns/copy
     */
    @Benchmark
    public void benchmarkWeightMatrixCopy(Blackhole bh) {
        var newWeights = new WeightMatrix(postSize, preSize);
        for (int j = 0; j < postSize; j++) {
            for (int i = 0; i < preSize; i++) {
                newWeights.set(j, i, weights.get(j, i));
            }
        }
        bh.consume(newWeights);
    }

    /**
     * Main method for running benchmarks directly.
     */
    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}
