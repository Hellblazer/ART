package com.hellblazer.art.cortical.benchmarks;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.cortical.layers.CorticalCircuit;
import com.hellblazer.art.cortical.layers.LayerParameters;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.lang.reflect.Method;
import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * JMH Benchmarks for Full Cortical Circuit - Phase 4A.
 *
 * <h2>Benchmark Objectives</h2>
 * <ul>
 *   <li>Measure end-to-end circuit latency</li>
 *   <li>Identify circuit-level bottlenecks</li>
 *   <li>Establish throughput baselines</li>
 *   <li>Measure temporal processor overhead</li>
 * </ul>
 *
 * <h2>Benchmark Categories</h2>
 * <ol>
 *   <li><b>Full Circuit</b>: Complete 6-layer processing</li>
 *   <li><b>Bottom-Up Only</b>: Layer 4 → 2/3 → 1</li>
 *   <li><b>Top-Down Only</b>: Layer 6 → 2/3</li>
 *   <li><b>With Temporal</b>: Circuit + LIST PARSE chunking</li>
 * </ol>
 *
 * <h2>Running Benchmarks</h2>
 * <pre>
 * # Run all circuit benchmarks
 * mvn test -Dtest=CircuitBenchmarks
 *
 * # With detailed profiling
 * mvn test -Dtest=CircuitBenchmarks -Djmh.prof=stack
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
public class CircuitBenchmarks {

    @Param({"64", "128", "256"})
    private int dimension;

    private Pattern input;
    private CorticalCircuit circuit;

    @Setup(Level.Trial)
    public void setup() throws Exception {
        var random = new Random(42);

        // Create random input pattern
        var values = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            values[i] = random.nextDouble();
        }
        input = new DenseVector(values);

        // Use reflection to access package-private parameter builders
        var layer1ParamsClass = Class.forName("com.hellblazer.art.cortical.layers.Layer1Parameters");
        var layer23ParamsClass = Class.forName("com.hellblazer.art.cortical.layers.Layer23Parameters");
        var layer4ParamsClass = Class.forName("com.hellblazer.art.cortical.layers.Layer4Parameters");
        var layer5ParamsClass = Class.forName("com.hellblazer.art.cortical.layers.Layer5Parameters");
        var layer6ParamsClass = Class.forName("com.hellblazer.art.cortical.layers.Layer6Parameters");

        var layer1Params = layer1ParamsClass.cast(createLayerParameters("Layer1Parameters", dimension));
        var layer23Params = layer23ParamsClass.cast(createLayerParameters("Layer23Parameters", dimension));
        var layer4Params = layer4ParamsClass.cast(createLayerParameters("Layer4Parameters", dimension));
        var layer5Params = layer5ParamsClass.cast(createLayerParameters("Layer5Parameters", dimension));
        var layer6Params = layer6ParamsClass.cast(createLayerParameters("Layer6Parameters", dimension));

        // Create minimal temporal processor (stub for benchmarking)
        var temporalProcessor = createMinimalTemporalProcessor(dimension);

        // Create circuit with constructor using reflection
        var constructor = CorticalCircuit.class.getDeclaredConstructor(
            int.class,
            layer1ParamsClass,
            layer23ParamsClass,
            layer4ParamsClass,
            layer5ParamsClass,
            layer6ParamsClass,
            com.hellblazer.art.cortical.temporal.TemporalProcessor.class
        );
        circuit = constructor.newInstance(
            dimension,
            layer1Params,
            layer23Params,
            layer4Params,
            layer5Params,
            layer6Params,
            temporalProcessor
        );
    }

    /**
     * Create layer parameters using reflection to access package-private builders.
     */
    @SuppressWarnings("unchecked")
    private <T> T createLayerParameters(String className, int dimension) throws Exception {
        var clazz = Class.forName("com.hellblazer.art.cortical.layers." + className);
        var builderMethod = clazz.getMethod("builder");
        var builder = builderMethod.invoke(null);

        // Set size if needed (for Layer23Parameters)
        if (className.equals("Layer23Parameters")) {
            var sizeMethod = builder.getClass().getMethod("size", int.class);
            builder = sizeMethod.invoke(builder, dimension);
        }

        var buildMethod = builder.getClass().getMethod("build");
        return (T) buildMethod.invoke(builder);
    }

    /**
     * Create minimal temporal processor for benchmarking.
     */
    private com.hellblazer.art.cortical.temporal.TemporalProcessor createMinimalTemporalProcessor(int size) throws Exception {
        // Create parameter classes using reflection
        var wmParamsClass = Class.forName("com.hellblazer.art.cortical.temporal.WorkingMemoryParameters");
        var mfParamsClass = Class.forName("com.hellblazer.art.cortical.temporal.MaskingFieldParameters");

        // Create WorkingMemoryParameters with defaults
        var wmBuilderMethod = wmParamsClass.getMethod("builder");
        var wmBuilder = wmBuilderMethod.invoke(null);
        var wmSizeMethod = wmBuilder.getClass().getMethod("itemDimension", int.class);
        wmBuilder = wmSizeMethod.invoke(wmBuilder, size);
        var wmBuildMethod = wmBuilder.getClass().getMethod("build");
        var wmParams = wmBuildMethod.invoke(wmBuilder);

        // Create MaskingFieldParameters with defaults
        var mfBuilderMethod = mfParamsClass.getMethod("builder");
        var mfBuilder = mfBuilderMethod.invoke(null);
        var mfSizeMethod = mfBuilder.getClass().getMethod("dimension", int.class);
        mfBuilder = mfSizeMethod.invoke(mfBuilder, size);
        var mfBuildMethod = mfBuilder.getClass().getMethod("build");
        var mfParams = mfBuildMethod.invoke(mfBuilder);

        // Create TemporalProcessor
        var constructor = com.hellblazer.art.cortical.temporal.TemporalProcessor.class
            .getDeclaredConstructor(wmParamsClass, mfParamsClass);
        return constructor.newInstance(wmParams, mfParams);
    }

    @TearDown(Level.Trial)
    public void tearDown() throws Exception {
        if (circuit != null) {
            circuit.close();
        }
    }

    /**
     * Benchmark full 6-layer cortical circuit processing.
     *
     * <p>This represents the complete cortical processing pipeline:
     * <ol>
     *   <li>Layer 4: Thalamic input</li>
     *   <li>Layer 2/3: Horizontal grouping</li>
     *   <li>Layer 1: Sustained attention</li>
     *   <li>Layer 6: Top-down feedback</li>
     *   <li>Layer 5: Motor output</li>
     * </ol>
     *
     * <p>Expected: ~100-500 µs/pattern (sum of all layers)
     */
    @Benchmark
    public void benchmarkFullCircuit(Blackhole bh) {
        var result = circuit.process(input);
        bh.consume(result);
    }

    /**
     * Benchmark bottom-up pathway only (Layer 4 → 2/3 → 1).
     *
     * <p>Expected: ~80-350 µs/pattern
     */
    @Benchmark
    public void benchmarkBottomUpOnly(Blackhole bh) {
        // Process bottom-up pathway with null parameters (uses defaults)
        var l4Out = circuit.getLayer4().processBottomUp(input, null);
        var l23Out = circuit.getLayer23().processBottomUp(l4Out, null);
        var l1Out = circuit.getLayer1().processBottomUp(l23Out, null);

        bh.consume(l1Out);
    }

    /**
     * Benchmark top-down pathway only (Layer 6 → 2/3 → 4).
     *
     * <p>Expected: ~50-250 µs/pattern
     */
    @Benchmark
    public void benchmarkTopDownOnly(Blackhole bh) {
        // Process top-down pathway with null parameters (uses defaults)
        var l6Out = circuit.getLayer6().processTopDown(input, null);
        var l23Out = circuit.getLayer23().processTopDown(l6Out, null);

        bh.consume(l23Out);
    }

    /**
     * Benchmark circuit with detailed output.
     *
     * <p>Measures overhead of collecting detailed activation information.
     *
     * <p>Expected: ~110-550 µs/pattern (5-10% overhead)
     */
    @Benchmark
    public void benchmarkDetailedOutput(Blackhole bh) {
        var result = circuit.processDetailed(input);
        bh.consume(result);
    }

    /**
     * Main method for running benchmarks directly.
     */
    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}
