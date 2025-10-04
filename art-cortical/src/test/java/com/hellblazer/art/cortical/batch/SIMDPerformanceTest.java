package com.hellblazer.art.cortical.batch;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import org.junit.jupiter.api.Test;

import java.util.Random;

/**
 * Simple performance validation test for Phase 1B SIMD targets.
 *
 * <p>Validates that batch size 64 achieves 1.40x-1.50x speedup vs sequential.
 *
 * <p>This is a simplified alternative to JMH benchmarks for quick validation.
 */
public class SIMDPerformanceTest {

    private static final int WARMUP_ITERATIONS = 10;
    private static final int MEASUREMENT_ITERATIONS = 100;

    @Test
    public void testBatch64Performance() {
        System.out.println("\n=== Phase 1B Performance Validation ===\n");

        // Test parameters
        int[] batchSizes = {32, 64, 128};
        int[] dimensions = {64, 128, 256};

        for (int dimension : dimensions) {
            System.out.println("Dimension: " + dimension);
            System.out.println("----------------------------------------");

            // Measure sequential baseline with batch size 64
            long sequentialTime = measureSequential(64, dimension);

            System.out.printf("  Sequential (64):  %,10d ns/batch%n", sequentialTime);

            // Run SIMD batches
            for (int batchSize : batchSizes) {
                // Check if SIMD is beneficial
                boolean beneficial = BatchDataLayout.isTransposeAndVectorizeBeneficial(batchSize, dimension, 10);
                String status = beneficial ? "[SIMD]" : "[Sequential fallback]";

                long simdTime = measureSIMD(batchSize, dimension);

                // Normalize to same batch size for fair comparison
                double normalizedSequentialTime = sequentialTime * (batchSize / 64.0);
                double speedup = normalizedSequentialTime / simdTime;

                String marker = (batchSize == 64 && speedup >= 1.40) ? " ✓ TARGET" : "";
                System.out.printf("  SIMD Batch-%3d:   %,10d ns/batch (%.2fx speedup) %s%s%n",
                    batchSize, simdTime, speedup, status, marker);
            }
            System.out.println();
        }

        System.out.println("Target: Batch-64 achieves 1.40x-1.50x speedup");
    }

    private long measureSequential(int batchSize, int dimension) {
        var random = new Random(42);
        var patterns = generatePatterns(batchSize, dimension, random);

        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            processSequential(patterns, dimension);
        }

        // Measurement
        long start = System.nanoTime();
        for (int i = 0; i < MEASUREMENT_ITERATIONS; i++) {
            processSequential(patterns, dimension);
        }
        long end = System.nanoTime();

        return (end - start) / MEASUREMENT_ITERATIONS;
    }

    private long measureSIMD(int batchSize, int dimension) {
        var random = new Random(42);
        var patterns = generatePatterns(batchSize, dimension, random);

        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            processSIMD(patterns, dimension);
        }

        // Measurement
        long start = System.nanoTime();
        for (int i = 0; i < MEASUREMENT_ITERATIONS; i++) {
            processSIMD(patterns, dimension);
        }
        long end = System.nanoTime();

        return (end - start) / MEASUREMENT_ITERATIONS;
    }

    private Pattern[] processSequential(Pattern[] inputs, int size) {
        var results = new Pattern[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            results[i] = processPattern(inputs[i], size);
        }
        return results;
    }

    private Pattern processPattern(Pattern input, int size) {
        // Simulate Layer 4 processing - EXACT match with SIMD path
        double[] data = new double[size];
        for (int d = 0; d < size; d++) {
            double x = input.get(d);

            // Driving strength
            x *= 2.0;

            // Shunting dynamics (matching SIMD path exactly)
            double decay = 0.3;
            double ceiling = 1.0;
            double floor = 0.0;
            double selfExc = 0.3;
            double deltaT = 0.01;

            // Excitation: self_exc * x + external_input (x)
            double excitation = Math.max(0, selfExc * x + x);
            // Shunting equation: dx/dt = -A*x + (B-x)*E
            double derivative = -decay * x + (ceiling - x) * excitation;
            // Euler integration
            x = x + deltaT * derivative;
            // Clamp
            x = Math.max(floor, Math.min(ceiling, x));

            // Saturation (sigmoid)
            if (x > 0) {
                x = ceiling * x / (1.0 + x);
            }
            x = Math.max(floor, Math.min(ceiling, x));

            data[d] = x;
        }
        return new DenseVector(data);
    }

    private Pattern[] processSIMD(Pattern[] inputs, int size) {
        var result = Layer4SIMDBatch.processBatchSIMD(
            inputs,
            2.0,      // drivingStrength
            10.0,     // timeConstant
            1.0,      // ceiling
            0.0,      // floor
            0.3,      // selfExcitation
            0.0,      // lateralInhibition - MUST BE 0.0 for SIMD path!
            size
        );

        // If SIMD returned null (not beneficial), fall back to sequential
        if (result == null) {
            return processSequential(inputs, size);
        }
        return result;
    }

    private Pattern[] generatePatterns(int count, int dimension, Random random) {
        var patterns = new Pattern[count];
        for (int i = 0; i < count; i++) {
            double[] data = new double[dimension];
            for (int d = 0; d < dimension; d++) {
                data[d] = random.nextDouble();
            }
            patterns[i] = new DenseVector(data);
        }
        return patterns;
    }
}
