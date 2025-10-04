package com.hellblazer.art.cortical.benchmarks;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.cortical.layers.Layer1;
import com.hellblazer.art.cortical.layers.Layer23;
import com.hellblazer.art.cortical.layers.Layer4;
import com.hellblazer.art.cortical.layers.Layer6;
import com.hellblazer.art.cortical.layers.LayerParameters;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * JMH Benchmarks for Individual Layer Performance - Phase 4A.
 *
 * <h2>Benchmark Objectives</h2>
 * <ul>
 *   <li>Establish baseline performance for each cortical layer</li>
 *   <li>Identify hotspots for optimization</li>
 *   <li>Measure layer-specific processing latency</li>
 *   <li>Track memory allocation patterns</li>
 * </ul>
 *
 * <h2>Benchmark Categories</h2>
 * <ol>
 *   <li><b>Layer 4 Forward</b>: Thalamic driving input (fastest layer)</li>
 *   <li><b>Layer 2/3 Forward</b>: Horizontal grouping and prediction</li>
 *   <li><b>Layer 1 Forward</b>: Sustained attention (slowest layer)</li>
 *   <li><b>Layer 6 Feedback</b>: Top-down expectations</li>
 * </ol>
 *
 * <h2>Running Benchmarks</h2>
 * <pre>
 * # Run all layer benchmarks
 * mvn test -Dtest=LayerBenchmarks
 *
 * # Run specific benchmark
 * mvn test -Dtest=LayerBenchmarks#benchmarkLayer4Forward
 *
 * # With GC profiler
 * mvn test -Dtest=LayerBenchmarks -Djmh.prof=gc
 * </pre>
 *
 * @author Phase 4A: Benchmarking Infrastructure
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 2, timeUnit = TimeUnit.SECONDS)
@Fork(value = 1, jvmArgs = {"--add-modules=jdk.incubator.vector", "-Xmx2G"})
public class LayerBenchmarks {

    @Param({"64", "128", "256", "512"})
    private int dimension;

    private Pattern input;
    private Layer4 layer4;
    private Layer23 layer23;
    private Layer1 layer1;
    private Layer6 layer6;

    private LayerParameters layer4Params;
    private LayerParameters layer23Params;
    private LayerParameters layer1Params;
    private LayerParameters layer6Params;

    @Setup(Level.Trial)
    public void setup() {
        var random = new Random(42);

        // Create random input pattern
        var values = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            values[i] = random.nextDouble();
        }
        input = new DenseVector(values);

        // Create layers with constructors
        layer4 = new Layer4("L4", dimension);
        layer23 = new Layer23("L2/3", dimension);
        layer1 = new Layer1("L1", dimension);
        layer6 = new Layer6("L6", dimension);

        // Create default parameters via process methods
        // Parameters are created internally when processBottomUp/TopDown is called with null
        layer4Params = null;  // Will use defaults
        layer23Params = null;  // Will use defaults
        layer1Params = null;  // Will use defaults
        layer6Params = null;  // Will use defaults
    }

    @TearDown(Level.Trial)
    public void tearDown() throws Exception {
        if (layer4 != null) layer4.close();
        if (layer23 != null) layer23.close();
        if (layer1 != null) layer1.close();
        if (layer6 != null) layer6.close();
    }

    /**
     * Benchmark Layer 4 bottom-up processing.
     *
     * <p>Layer 4 is the fastest layer (10-50ms time constant) and processes
     * every input pattern. This is the primary throughput bottleneck.
     *
     * <p>Expected: ~10-50 µs/pattern (depends on dimension)
     */
    @Benchmark
    public void benchmarkLayer4Forward(Blackhole bh) {
        var output = layer4.processBottomUp(input, layer4Params);
        bh.consume(output);
    }

    /**
     * Benchmark Layer 2/3 bottom-up processing.
     *
     * <p>Layer 2/3 handles horizontal grouping and complex cell responses
     * (30-150ms time constant).
     *
     * <p>Expected: ~20-100 µs/pattern
     */
    @Benchmark
    public void benchmarkLayer23Forward(Blackhole bh) {
        var output = layer23.processBottomUp(input, layer23Params);
        bh.consume(output);
    }

    /**
     * Benchmark Layer 1 bottom-up processing.
     *
     * <p>Layer 1 handles sustained attention with apical dendrites
     * (200-1000ms time constant). Slowest layer.
     *
     * <p>Expected: ~50-200 µs/pattern
     */
    @Benchmark
    public void benchmarkLayer1Forward(Blackhole bh) {
        var output = layer1.processBottomUp(input, layer1Params);
        bh.consume(output);
    }

    /**
     * Benchmark Layer 6 top-down processing.
     *
     * <p>Layer 6 provides corticothalamic feedback and expectations
     * (100-500ms time constant).
     *
     * <p>Expected: ~30-150 µs/pattern
     */
    @Benchmark
    public void benchmarkLayer6TopDown(Blackhole bh) {
        var output = layer6.processTopDown(input, layer6Params);
        bh.consume(output);
    }

    /**
     * Benchmark Layer 4 reset operation.
     *
     * <p>Measures the cost of resetting layer state.
     */
    @Benchmark
    public void benchmarkLayer4Reset(Blackhole bh) {
        layer4.reset();
        bh.consume(layer4);
    }

    /**
     * Main method for running benchmarks directly.
     */
    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}
