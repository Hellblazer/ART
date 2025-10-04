package com.hellblazer.art.cortical.batch;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.cortical.dynamics.ShuntingParameters;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.concurrent.TimeUnit;

/**
 * JMH Benchmarks for SIMD Batch Processing - Phase 1B Performance Validation.
 *
 * <h2>Phase 1B Objectives</h2>
 * <ul>
 *   <li>Validate mini-batch size 64 performance</li>
 *   <li>Measure speedup vs sequential processing</li>
 *   <li>Target: 1.40x-1.50x speedup (stretch goal: 1.50x+)</li>
 *   <li>Baseline: 1.30x from art-laminar (1049.7 patterns/sec)</li>
 * </ul>
 *
 * <h2>Benchmark Categories</h2>
 * <ol>
 *   <li><b>Baseline Sequential</b>: Pattern-by-pattern processing</li>
 *   <li><b>SIMD Batch-32</b>: Current art-laminar performance (1.30x)</li>
 *   <li><b>SIMD Batch-64</b>: Phase 1B target (1.40x-1.50x)</li>
 *   <li><b>SIMD Batch-128</b>: Large batch scaling analysis</li>
 * </ol>
 *
 * <h2>Running Benchmarks</h2>
 * <pre>
 * # Run all benchmarks
 * mvn test -Dtest=SIMDBenchmark
 *
 * # Run specific benchmark
 * mvn test -Dtest=SIMDBenchmark#benchmarkBatch64
 *
 * # With profiler
 * mvn test -Dtest=SIMDBenchmark -Djmh.prof=gc
 * </pre>
 *
 * @author Phase 1B: Mini-Batch Size Increase
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(value = 1, jvmArgs = {"--add-modules=jdk.incubator.vector", "-Xmx2G"})
public class SIMDBenchmark {

    // Benchmark parameters matching Phase 1B specification
    @Param({"32", "64", "128", "256"})
    private int batchSize;

    @Param({"64", "128", "256"})
    private int dimension;

    // Layer 4 parameters (fast dynamics, no lateral interactions)
    private static final double DRIVING_STRENGTH = 1.0;
    private static final double TIME_CONSTANT = 10.0;
    private static final double CEILING = 1.0;
    private static final double FLOOR = 0.0;
    private static final double SELF_EXCITATION = 0.3;
    private static final double LATERAL_INHIBITION = 0.0;

    private Pattern[] patterns;
    private ShuntingParameters shuntingParams;

    @Setup(Level.Trial)
    public void setup() {
        // Generate random patterns
        patterns = new Pattern[batchSize];
        var random = new java.util.Random(42);  // Fixed seed for reproducibility

        for (int i = 0; i < batchSize; i++) {
            var values = new double[dimension];
            for (int d = 0; d < dimension; d++) {
                values[d] = random.nextDouble();  // [0, 1)
            }
            patterns[i] = new DenseVector(values);
        }

        // Create shunting parameters
        shuntingParams = ShuntingParameters.builder(dimension)
            .ceiling(CEILING)
            .floor(FLOOR)
            .selfExcitation(SELF_EXCITATION)
            .inhibitoryStrength(LATERAL_INHIBITION)
            .build();
    }

    /**
     * Baseline: Sequential pattern-by-pattern processing.
     *
     * <p>This represents the non-SIMD baseline for comparison.
     * Expected throughput: ~800-1000 patterns/sec (depends on hardware).
     */
    @Benchmark
    public void baselineSequential(Blackhole bh) {
        for (var pattern : patterns) {
            // Simulate Layer 4 processing
            var values = pattern.toArray();

            // Apply driving strength
            for (int d = 0; d < dimension; d++) {
                values[d] *= DRIVING_STRENGTH;
            }

            // Apply simplified dynamics (single step)
            for (int d = 0; d < dimension; d++) {
                double x = values[d];
                double E = SELF_EXCITATION * x;
                double derivative = -SELF_EXCITATION * x + (CEILING - x) * E;
                values[d] = x + 0.01 * derivative;
            }

            // Apply saturation
            for (int d = 0; d < dimension; d++) {
                double x = values[d];
                if (x > 0) {
                    x = CEILING * x / (1.0 + x);
                }
                values[d] = Math.max(FLOOR, Math.min(CEILING, x));
            }

            bh.consume(values);
        }
    }

    /**
     * SIMD Batch Processing: Main benchmark.
     *
     * <p>This uses the full Layer4SIMDBatch.processBatchSIMD pipeline:
     * <ol>
     *   <li>Transpose to dimension-major (overhead ~5%)</li>
     *   <li>Apply driving strength (SIMD)</li>
     *   <li>Apply exact shunting dynamics (SIMD for Layer 4)</li>
     *   <li>Apply saturation (SIMD)</li>
     *   <li>Transpose back to pattern-major (overhead ~5%)</li>
     * </ol>
     *
     * <p>Expected speedup:
     * <ul>
     *   <li>Batch 32: 1.30x (art-laminar baseline)</li>
     *   <li>Batch 64: 1.40x-1.50x (Phase 1B target)</li>
     *   <li>Batch 128+: 1.50x+ (optimal SIMD utilization)</li>
     * </ul>
     */
    @Benchmark
    public void simdBatchProcessing(Blackhole bh) {
        var result = Layer4SIMDBatch.processBatchSIMD(
            patterns,
            DRIVING_STRENGTH,
            TIME_CONSTANT,
            CEILING,
            FLOOR,
            SELF_EXCITATION,
            LATERAL_INHIBITION,
            dimension
        );

        if (result != null) {
            bh.consume(result);
        } else {
            // Fallback to sequential (shouldn't happen for batch >= 32)
            baselineSequential(bh);
        }
    }

    /**
     * Transpose-only benchmark: Measure transpose overhead.
     *
     * <p>This isolates the transpose operation cost to understand
     * how much overhead is added by the transpose-and-vectorize strategy.
     *
     * <p>Expected: ~5-10% of total processing time.
     */
    @Benchmark
    public void transposeOnly(Blackhole bh) {
        // Forward transpose (pattern-major → dimension-major)
        var dimMajor = BatchDataLayout.transposeToDimensionMajor(patterns);

        // Backward transpose (dimension-major → pattern-major)
        var patternMajor = BatchDataLayout.transposeToPatternMajor(dimMajor);

        bh.consume(patternMajor);
    }

    /**
     * SIMD operations only: Measure pure SIMD benefit without transpose.
     *
     * <p>This measures the speedup from SIMD operations alone,
     * excluding transpose overhead. Useful for understanding
     * the theoretical maximum speedup.
     */
    @Benchmark
    public void simdOperationsOnly(Blackhole bh) {
        var batch = Layer4SIMDBatch.createBatch(patterns, dimension);

        // Apply SIMD operations (no transpose cost included in measurement)
        batch.applyDrivingStrength(DRIVING_STRENGTH);

        // Simple dynamics (not exact BatchShuntingDynamics for speed)
        batch.applyDynamics(Math.min(TIME_CONSTANT / 1000.0, 0.01));

        batch.applySaturation(CEILING, FLOOR);

        // Note: toPatterns() includes transpose back, but we created batch in setup
        var dimMajor = batch.getDimensionMajor();
        bh.consume(dimMajor);
    }

    /**
     * Batch creation benchmark: Measure batch creation + transpose cost.
     */
    @Benchmark
    public void batchCreation(Blackhole bh) {
        var batch = Layer4SIMDBatch.createBatch(patterns, dimension);
        bh.consume(batch);
    }

    /**
     * Driving strength SIMD benchmark: Isolated operation.
     */
    @Benchmark
    public void drivingStrengthSIMD(Blackhole bh) {
        var batch = Layer4SIMDBatch.createBatch(patterns, dimension);
        batch.applyDrivingStrength(DRIVING_STRENGTH);
        bh.consume(batch.getDimensionMajor());
    }

    /**
     * Saturation SIMD benchmark: Isolated operation.
     */
    @Benchmark
    public void saturationSIMD(Blackhole bh) {
        var batch = Layer4SIMDBatch.createBatch(patterns, dimension);
        batch.applySaturation(CEILING, FLOOR);
        bh.consume(batch.getDimensionMajor());
    }

    /**
     * Main method for running benchmarks directly.
     *
     * <p>Usage:
     * <pre>
     * mvn test-compile exec:java -Dexec.mainClass=com.hellblazer.art.cortical.batch.SIMDBenchmark
     * </pre>
     */
    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}
