package com.hellblazer.art.temporal.dynamics;

import com.hellblazer.art.temporal.core.ActivationState;
import org.junit.jupiter.api.Test;

/**
 * Performance test for vectorized ShuntingDynamicsImpl.
 *
 * Expected speedup: 1.5-2x due to:
 * - Vectorized dot products in excitation/inhibition computation
 * - Pre-computed weight matrices (eliminates repeated Gaussian calculations)
 * - SIMD acceleration for patterns >= 64D
 */
public class ShuntingDynamicsPerformanceTest {

    @Test
    public void testVectorizedPerformance() {
        // Test with different dimensions to show SIMD threshold effect
        int[] dimensions = {32, 64, 128, 256};

        System.out.println("\n=== SHUNTING DYNAMICS VECTORIZATION PERFORMANCE ===\n");

        for (var dimension : dimensions) {
            measurePerformance(dimension);
        }
    }

    private void measurePerformance(int dimension) {
        var parameters = ShuntingParameters.competitiveDefaults(dimension);
        var dynamics = new ShuntingDynamicsImpl(parameters, dimension);

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

        // Benchmark
        var iterations = 10000;
        var startTime = System.nanoTime();

        for (int i = 0; i < iterations; i++) {
            state = dynamics.evolve(state, 0.01);
        }

        var endTime = System.nanoTime();
        var totalMs = (endTime - startTime) / 1_000_000.0;
        var avgMicros = (endTime - startTime) / (double) iterations / 1000.0;

        System.out.printf("Dimension %3d: %8.2f ms total, %6.2f μs/evolve%s\n",
            dimension, totalMs, avgMicros,
            dimension >= 64 ? " [SIMD active]" : " [scalar]");
    }
}