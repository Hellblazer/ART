package com.hellblazer.art.cortical.benchmarks;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.cortical.dynamics.ShuntingDynamics;
import com.hellblazer.art.cortical.dynamics.ShuntingParameters;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * JMH Benchmarks for Neural Dynamics - Phase 4A.
 *
 * <h2>Benchmark Objectives</h2>
 * <ul>
 *   <li>Measure shunting dynamics convergence time</li>
 *   <li>Identify opportunities for parallelization</li>
 *   <li>Establish dynamics computation baselines</li>
 *   <li>Measure Lyapunov energy convergence</li>
 * </ul>
 *
 * <h2>Benchmark Categories</h2>
 * <ol>
 *   <li><b>Single Iteration</b>: One dynamics update step</li>
 *   <li><b>Full Convergence</b>: Iterate until Lyapunov stable</li>
 *   <li><b>Fast Dynamics</b>: Layer 4 parameters (10ms)</li>
 *   <li><b>Slow Dynamics</b>: Layer 1 parameters (1000ms)</li>
 * </ol>
 *
 * <h2>Running Benchmarks</h2>
 * <pre>
 * # Run all dynamics benchmarks
 * mvn test -Dtest=DynamicsBenchmarks
 *
 * # With profiler
 * mvn test -Dtest=DynamicsBenchmarks -Djmh.prof=stack
 * </pre>
 *
 * @author Phase 4A: Benchmarking Infrastructure
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 2, timeUnit = TimeUnit.SECONDS)
@Fork(value = 1, jvmArgs = {"-Xmx2G"})
public class DynamicsBenchmarks {

    @Param({"64", "128", "256", "512"})
    private int dimension;

    private double[] inputArray;
    private ShuntingParameters fastParams;   // Layer 4 (10ms)
    private ShuntingParameters mediumParams; // Layer 2/3 (30ms)
    private ShuntingParameters slowParams;   // Layer 1 (200ms)

    private ShuntingDynamics fastDynamics;
    private ShuntingDynamics mediumDynamics;
    private ShuntingDynamics slowDynamics;

    @Setup(Level.Trial)
    public void setup() {
        var random = new Random(42);

        // Create random input array
        inputArray = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            inputArray[i] = random.nextDouble();
        }

        // Create parameters with different time constants
        fastParams = ShuntingParameters.builder(dimension)
            .ceiling(1.0)
            .floor(0.0)
            .selfExcitation(0.3)
            .inhibitoryStrength(0.1)
            .timeStep(0.001)  // 1ms timestep
            .build();

        mediumParams = ShuntingParameters.builder(dimension)
            .ceiling(1.0)
            .floor(0.0)
            .selfExcitation(0.2)
            .inhibitoryStrength(0.15)
            .timeStep(0.001)
            .build();

        slowParams = ShuntingParameters.builder(dimension)
            .ceiling(1.0)
            .floor(0.0)
            .selfExcitation(0.1)
            .inhibitoryStrength(0.2)
            .timeStep(0.001)
            .build();

        // Create dynamics instances
        fastDynamics = new ShuntingDynamics(fastParams);
        mediumDynamics = new ShuntingDynamics(mediumParams);
        slowDynamics = new ShuntingDynamics(slowParams);
    }

    /**
     * Benchmark single iteration of shunting dynamics (fast).
     *
     * <p>One step of the dynamics equation:
     * dx/dt = -Ax + (B - x)E - (x + C)I
     *
     * <p>Expected: ~10-50 µs/iteration (depends on dimension)
     */
    @Benchmark
    public void benchmarkSingleIteration(Blackhole bh) {
        fastDynamics.setExcitatoryInput(inputArray);
        var output = fastDynamics.update(0.001);
        bh.consume(output);
    }

    /**
     * Benchmark full convergence with fast dynamics (Layer 4).
     *
     * <p>Iterates until Lyapunov energy stabilizes (typically 5-10 iterations).
     *
     * <p>Expected: ~50-500 µs/convergence
     */
    @Benchmark
    public void benchmarkFastConvergence(Blackhole bh) {
        fastDynamics.reset();
        fastDynamics.setExcitatoryInput(inputArray);

        // Run until convergence (max 100 iterations)
        var output = inputArray.clone();
        for (int i = 0; i < 100 && !fastDynamics.hasConverged(); i++) {
            output = fastDynamics.update(0.001);
        }
        bh.consume(output);
    }

    /**
     * Benchmark full convergence with medium dynamics (Layer 2/3).
     *
     * <p>Expected: ~100-1000 µs/convergence
     */
    @Benchmark
    public void benchmarkMediumConvergence(Blackhole bh) {
        mediumDynamics.reset();
        mediumDynamics.setExcitatoryInput(inputArray);

        var output = inputArray.clone();
        for (int i = 0; i < 100 && !mediumDynamics.hasConverged(); i++) {
            output = mediumDynamics.update(0.001);
        }
        bh.consume(output);
    }

    /**
     * Benchmark full convergence with slow dynamics (Layer 1).
     *
     * <p>Expected: ~200-2000 µs/convergence
     */
    @Benchmark
    public void benchmarkSlowConvergence(Blackhole bh) {
        slowDynamics.reset();
        slowDynamics.setExcitatoryInput(inputArray);

        var output = inputArray.clone();
        for (int i = 0; i < 100 && !slowDynamics.hasConverged(); i++) {
            output = slowDynamics.update(0.001);
        }
        bh.consume(output);
    }

    /**
     * Benchmark Lyapunov energy computation.
     *
     * <p>Measures convergence detection overhead.
     *
     * <p>Expected: ~5-20 µs/computation
     */
    @Benchmark
    public void benchmarkLyapunovEnergy(Blackhole bh) {
        var energy = fastDynamics.computeEnergy();
        bh.consume(energy);
    }

    /**
     * Benchmark dynamics reset operation.
     *
     * <p>Measures the cost of resetting dynamics state.
     */
    @Benchmark
    public void benchmarkDynamicsReset(Blackhole bh) {
        fastDynamics.reset();
        bh.consume(fastDynamics);
    }

    /**
     * Main method for running benchmarks directly.
     */
    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}
