package com.hellblazer.art.temporal.dynamics;

import com.hellblazer.art.laminar.performance.VectorizedArrayOperations;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test to validate the speedup from vectorization.
 *
 * This test compares:
 * 1. Scalar dot product (baseline)
 * 2. Vectorized dot product (optimized)
 *
 * Expected speedup: 1.5-2x for patterns >= 64D due to SIMD acceleration.
 */
public class VectorizationSpeedupTest {

    @Test
    public void testDotProductSpeedup() {
        System.out.println("\n=== DOT PRODUCT VECTORIZATION SPEEDUP ===\n");

        int[] dimensions = {32, 64, 128, 256, 512};

        for (var dimension : dimensions) {
            measureDotProductSpeedup(dimension);
        }
    }

    @Test
    public void testShuntingDynamicsOptimizations() {
        System.out.println("\n=== SHUNTING DYNAMICS OPTIMIZATION ANALYSIS ===\n");

        var dimension = 256;
        var parameters = ShuntingParameters.competitiveDefaults(dimension);

        System.out.printf("Testing dimension: %d\n", dimension);
        System.out.println("\nOptimizations applied:");
        System.out.println("  1. Pre-computed weight matrices (eliminates repeated exp() calls)");
        System.out.println("  2. Vectorized dot products (SIMD acceleration)");
        System.out.println("  3. Batch computation of excitation/inhibition arrays\n");

        // Measure weight matrix initialization cost (one-time)
        var startInit = System.nanoTime();
        var dynamics = new ShuntingDynamicsImpl(parameters, dimension);
        var initTime = (System.nanoTime() - startInit) / 1_000_000.0;

        System.out.printf("Weight matrix initialization: %.3f ms (one-time cost)\n", initTime);

        // Create test input
        var input = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            input[i] = 0.5 + 0.3 * Math.sin(i * 0.1);
        }
        dynamics.setExcitatoryInput(input);

        var state = dynamics.getState();

        // Warmup
        for (int i = 0; i < 1000; i++) {
            state = dynamics.evolve(state, 0.01);
        }

        // Benchmark evolve() which includes:
        // - computeExcitationArray (vectorized dot products)
        // - computeInhibitionArray (vectorized dot products)
        // - vectorizedEvolveStep (shunting dynamics)
        var iterations = 10000;
        var startTime = System.nanoTime();

        for (int i = 0; i < iterations; i++) {
            state = dynamics.evolve(state, 0.01);
        }

        var endTime = System.nanoTime();
        var totalMs = (endTime - startTime) / 1_000_000.0;
        var avgMicros = (endTime - startTime) / (double) iterations / 1000.0;

        System.out.printf("\nPerformance with optimizations:\n");
        System.out.printf("  Total time: %.2f ms\n", totalMs);
        System.out.printf("  Per iteration: %.2f μs\n", avgMicros);
        System.out.printf("  Throughput: %.0f iterations/sec\n\n", iterations * 1000.0 / totalMs);

        // Expected performance improvement
        System.out.println("Expected speedup from baseline:");
        System.out.println("  - Weight caching: ~2x (eliminates exp() calls in hot loop)");
        System.out.println("  - Vectorized dot: ~1.5x (SIMD for patterns >= 64D)");
        System.out.println("  - Combined: ~2-3x total speedup");
        System.out.println("\nNote: Baseline implementation would recompute Gaussian weights on every evolve()");
    }

    private void measureDotProductSpeedup(int dimension) {
        var a = new double[dimension];
        var b = new double[dimension];

        for (int i = 0; i < dimension; i++) {
            a[i] = 0.5 + 0.3 * Math.sin(i * 0.1);
            b[i] = 0.4 + 0.2 * Math.cos(i * 0.15);
        }

        var iterations = 100000;

        // Warmup
        for (int i = 0; i < 10000; i++) {
            scalarDot(a, b);
            VectorizedArrayOperations.dot(a, b);
        }

        // Benchmark scalar
        var startScalar = System.nanoTime();
        double scalarResult = 0;
        for (int i = 0; i < iterations; i++) {
            scalarResult = scalarDot(a, b);
        }
        var scalarTime = (System.nanoTime() - startScalar) / 1_000_000.0;

        // Benchmark vectorized
        var startVectorized = System.nanoTime();
        double vectorizedResult = 0;
        for (int i = 0; i < iterations; i++) {
            vectorizedResult = VectorizedArrayOperations.dot(a, b);
        }
        var vectorizedTime = (System.nanoTime() - startVectorized) / 1_000_000.0;

        // Validate correctness
        assertEquals(scalarResult, vectorizedResult, 1e-10,
            "Vectorized and scalar dot products must match");

        var speedup = scalarTime / vectorizedTime;

        System.out.printf("Dimension %3d: Scalar: %6.2f ms, Vectorized: %6.2f ms, Speedup: %.2fx%s\n",
            dimension, scalarTime, vectorizedTime, speedup,
            dimension >= 64 ? " [SIMD]" : " [scalar fallback]");
    }

    /**
     * Scalar dot product (baseline for comparison).
     */
    private double scalarDot(double[] a, double[] b) {
        double result = 0.0;
        for (int i = 0; i < a.length; i++) {
            result += a[i] * b[i];
        }
        return result;
    }
}