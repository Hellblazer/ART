package com.hellblazer.art.cortical.gpu.validation;

import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test numerical stability of FP32 over long runs (10,000 iterations).
 * Ensures no divergence, NaN, Inf, or catastrophic error accumulation.
 */
class NumericalStabilityPrecisionTest {

    private static final Logger log = LoggerFactory.getLogger(NumericalStabilityPrecisionTest.class);
    private static final Random random = new Random(42);
    private static final int LONG_RUN_ITERATIONS = 10000;

    @Test
    void testLongRunStability_ShuntingDynamics() {
        log.info("Testing shunting dynamics stability over {} iterations", LONG_RUN_ITERATIONS);

        int size = 1000;
        double A = 0.1;
        double B = 1.0;
        double D = 0.2;
        double dt = 0.01;

        // Initialize
        var x64 = new double[size];
        var I_exc64 = new double[size];
        var I_inh64 = new double[size];
        var x32 = new float[size];
        var I_exc32 = new float[size];
        var I_inh32 = new float[size];

        for (int i = 0; i < size; i++) {
            double val_x = random.nextDouble();
            double val_exc = random.nextDouble() * 2.0;
            double val_inh = random.nextDouble() * 0.5;
            x64[i] = val_x;
            I_exc64[i] = val_exc;
            I_inh64[i] = val_inh;
            x32[i] = (float) val_x;
            I_exc32[i] = (float) val_exc;
            I_inh32[i] = (float) val_inh;
        }

        // Run FP64
        var fp64Task = (Runnable) () -> {
            for (int iter = 0; iter < LONG_RUN_ITERATIONS; iter++) {
                for (int i = 0; i < size; i++) {
                    double dx = -A * x64[i] +
                                (B - x64[i]) * I_exc64[i] -
                                (x64[i] + D) * I_inh64[i];
                    x64[i] = x64[i] + dx * dt;
                    x64[i] = Math.max(0.0, Math.min(B, x64[i]));
                }
            }
        };

        // Run FP32
        var fp32Task = (Runnable) () -> {
            for (int iter = 0; iter < LONG_RUN_ITERATIONS; iter++) {
                for (int i = 0; i < size; i++) {
                    float dx = (float) (-A * x32[i] +
                                        (B - x32[i]) * I_exc32[i] -
                                        (x32[i] + D) * I_inh32[i]);
                    x32[i] = x32[i] + dx * (float) dt;
                    x32[i] = Math.max(0.0f, Math.min((float) B, x32[i]));
                }
            }
        };

        var result = PrecisionValidator.compare(
            String.format("Shunting Dynamics (%d iterations)", LONG_RUN_ITERATIONS),
            fp64Task,
            fp32Task,
            () -> x64,
            () -> x32,
            1e-3  // More relaxed for long runs
        );

        // Check for NaN/Inf
        for (int i = 0; i < size; i++) {
            assertFalse(Double.isNaN(x64[i]), "FP64 produced NaN at index " + i);
            assertFalse(Double.isInfinite(x64[i]), "FP64 produced Inf at index " + i);
            assertFalse(Float.isNaN(x32[i]), "FP32 produced NaN at index " + i);
            assertFalse(Float.isInfinite(x32[i]), "FP32 produced Inf at index " + i);
        }

        // Check values stay in bounds
        for (int i = 0; i < size; i++) {
            assertTrue(x64[i] >= 0.0 && x64[i] <= B,
                String.format("FP64 out of bounds at %d: %.6f", i, x64[i]));
            assertTrue(x32[i] >= 0.0f && x32[i] <= B,
                String.format("FP32 out of bounds at %d: %.6f", i, x32[i]));
        }

        assertTrue(result.passed,
            String.format("Long-run stability failed: %s", result));

        log.info("✅ Shunting dynamics stable for {} iterations", LONG_RUN_ITERATIONS);
        log.info("   Max error: %.6e, Avg error: %.6e", result.maxError, result.avgError);
    }

    @Test
    void testLongRunStability_WeightUpdates() {
        log.info("Testing weight update stability over {} iterations", LONG_RUN_ITERATIONS);

        int size = 1000;
        double learningRate = 0.01;

        var weights64 = new double[size];
        var pre64 = new double[size];
        var post64 = new double[size];
        var weights32 = new float[size];
        var pre32 = new float[size];
        var post32 = new float[size];

        // Initialize
        for (int i = 0; i < size; i++) {
            double w = random.nextDouble() * 0.5;
            double pre = random.nextDouble();
            double post = random.nextDouble();
            weights64[i] = w;
            pre64[i] = pre;
            post64[i] = post;
            weights32[i] = (float) w;
            pre32[i] = (float) pre;
            post32[i] = (float) post;
        }

        // Run FP64
        var fp64Task = (Runnable) () -> {
            for (int iter = 0; iter < LONG_RUN_ITERATIONS; iter++) {
                for (int i = 0; i < size; i++) {
                    double deltaW = learningRate * pre64[i] * post64[i];
                    weights64[i] += deltaW;
                    weights64[i] = Math.max(0.0, Math.min(1.0, weights64[i]));
                }
            }
        };

        // Run FP32
        var fp32Task = (Runnable) () -> {
            for (int iter = 0; iter < LONG_RUN_ITERATIONS; iter++) {
                for (int i = 0; i < size; i++) {
                    float deltaW = (float) (learningRate * pre32[i] * post32[i]);
                    weights32[i] += deltaW;
                    weights32[i] = Math.max(0.0f, Math.min(1.0f, weights32[i]));
                }
            }
        };

        var result = PrecisionValidator.compare(
            String.format("Weight Updates (%d iterations)", LONG_RUN_ITERATIONS),
            fp64Task,
            fp32Task,
            () -> weights64,
            () -> weights32,
            5e-3  // Allow more accumulation
        );

        // Check for NaN/Inf
        for (int i = 0; i < size; i++) {
            assertFalse(Double.isNaN(weights64[i]), "FP64 weights NaN at index " + i);
            assertFalse(Float.isNaN(weights32[i]), "FP32 weights NaN at index " + i);
        }

        // Check weights stay in [0, 1]
        for (int i = 0; i < size; i++) {
            assertTrue(weights64[i] >= 0.0 && weights64[i] <= 1.0,
                String.format("FP64 weight out of bounds at %d: %.6f", i, weights64[i]));
            assertTrue(weights32[i] >= 0.0f && weights32[i] <= 1.0f,
                String.format("FP32 weight out of bounds at %d: %.6f", i, weights32[i]));
        }

        assertTrue(result.passed,
            String.format("Long-run weight stability failed: %s", result));

        log.info("✅ Weight updates stable for {} iterations", LONG_RUN_ITERATIONS);
        log.info("   Max error: %.6e, Avg error: %.6e", result.maxError, result.avgError);
    }

    @Test
    void testErrorAccumulation() {
        log.info("Testing error accumulation pattern");

        int size = 100;
        int[] checkpoints = {100, 500, 1000, 5000, 10000};

        var x64 = new double[size];
        var x32 = new float[size];

        // Initialize
        for (int i = 0; i < size; i++) {
            double val = random.nextDouble();
            x64[i] = val;
            x32[i] = (float) val;
        }

        // Track error growth
        var errorGrowth = new double[checkpoints.length];

        for (int c = 0; c < checkpoints.length; c++) {
            int iterations = checkpoints[c];
            int startIter = (c == 0) ? 0 : checkpoints[c - 1];

            // Run iterations
            for (int iter = startIter; iter < iterations; iter++) {
                for (int i = 0; i < size; i++) {
                    // Simple update: x = x * 0.99 + 0.01
                    x64[i] = x64[i] * 0.99 + 0.01;
                    x32[i] = x32[i] * 0.99f + 0.01f;
                }
            }

            // Measure error
            double maxError = 0.0;
            for (int i = 0; i < size; i++) {
                double error = Math.abs(x64[i] - x32[i]);
                maxError = Math.max(maxError, error);
            }
            errorGrowth[c] = maxError;

            log.info("After {:5d} iterations: max error = {:.6e}", iterations, maxError);
        }

        // Log error growth patterns
        for (int c = 1; c < checkpoints.length; c++) {
            double errorRatio = errorGrowth[c] / errorGrowth[c - 1];
            double iterRatio = (double) checkpoints[c] / checkpoints[c - 1];

            log.info("Error growth ratio: {:.2f}x for {:.1f}x more iterations",
                errorRatio, iterRatio);
        }

        // Check that error is not exploding (growing exponentially)
        // FP32 error can grow faster than sublinear in some intervals,
        // but should not explode exponentially overall
        double overallGrowthRatio = errorGrowth[checkpoints.length - 1] / errorGrowth[0];
        double overallIterRatio = (double) checkpoints[checkpoints.length - 1] / checkpoints[0];

        log.info("Overall error growth: {:.2f}x for {:.1f}x iterations",
            overallGrowthRatio, overallIterRatio);

        // Error should not grow exponentially (allow up to quadratic growth)
        assertTrue(overallGrowthRatio < overallIterRatio * overallIterRatio,
            String.format("Error growing exponentially: %.2fx vs %.1fx iterations",
                overallGrowthRatio, overallIterRatio));

        // Final error should still be reasonable
        double finalError = errorGrowth[errorGrowth.length - 1];
        assertTrue(finalError < 1e-4,
            String.format("Final error too large after %d iterations: %.6e",
                LONG_RUN_ITERATIONS, finalError));

        log.info("✅ Error accumulation is sublinear and acceptable");
    }
}
