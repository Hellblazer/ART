package com.hellblazer.art.cortical.gpu.validation;

import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test precision of basic mathematical operations: FP32 vs FP64.
 * This establishes baseline precision expectations.
 */
class BasicOperationsPrecisionTest {

    private static final Logger log = LoggerFactory.getLogger(BasicOperationsPrecisionTest.class);
    // FP32 tolerance: machine epsilon for float is ~1.19e-7
    // Realistic tolerance for single operations is ~1e-6 to 1e-5
    private static final double FP32_TOLERANCE = 5e-6;  // Allow for rounding in FP32
    private static final int TEST_SIZE = 10000;
    private static final Random random = new Random(42);

    @Test
    void testAddition_FP32vsFP64() {
        log.info("Testing addition precision");

        // Generate test data
        var a64 = new double[TEST_SIZE];
        var b64 = new double[TEST_SIZE];
        var a32 = new float[TEST_SIZE];
        var b32 = new float[TEST_SIZE];

        for (int i = 0; i < TEST_SIZE; i++) {
            double val_a = random.nextDouble() * 10.0;
            double val_b = random.nextDouble() * 10.0;
            a64[i] = val_a;
            b64[i] = val_b;
            a32[i] = (float) val_a;
            b32[i] = (float) val_b;
        }

        // Compute FP64
        var result64 = new double[TEST_SIZE];
        var fp64Task = (Runnable) () -> {
            for (int i = 0; i < TEST_SIZE; i++) {
                result64[i] = a64[i] + b64[i];
            }
        };

        // Compute FP32
        var result32 = new float[TEST_SIZE];
        var fp32Task = (Runnable) () -> {
            for (int i = 0; i < TEST_SIZE; i++) {
                result32[i] = a32[i] + b32[i];
            }
        };

        // Validate
        var result = PrecisionValidator.compare(
            "Addition",
            fp64Task,
            fp32Task,
            () -> result64,
            () -> result32,
            FP32_TOLERANCE
        );

        assertTrue(result.passed,
            "Addition should pass FP32 precision: " + result);
        assertTrue(result.maxError < 5e-6,
            "Max error should be < 5e-6 for FP32: " + result.maxError);
    }

    @Test
    void testMultiplication_FP32vsFP64() {
        log.info("Testing multiplication precision");

        var a64 = new double[TEST_SIZE];
        var b64 = new double[TEST_SIZE];
        var a32 = new float[TEST_SIZE];
        var b32 = new float[TEST_SIZE];

        for (int i = 0; i < TEST_SIZE; i++) {
            double val_a = random.nextDouble() * 5.0;
            double val_b = random.nextDouble() * 5.0;
            a64[i] = val_a;
            b64[i] = val_b;
            a32[i] = (float) val_a;
            b32[i] = (float) val_b;
        }

        var result64 = new double[TEST_SIZE];
        var fp64Task = (Runnable) () -> {
            for (int i = 0; i < TEST_SIZE; i++) {
                result64[i] = a64[i] * b64[i];
            }
        };

        var result32 = new float[TEST_SIZE];
        var fp32Task = (Runnable) () -> {
            for (int i = 0; i < TEST_SIZE; i++) {
                result32[i] = a32[i] * b32[i];
            }
        };

        var result = PrecisionValidator.compare(
            "Multiplication",
            fp64Task,
            fp32Task,
            () -> result64,
            () -> result32,
            FP32_TOLERANCE
        );

        assertTrue(result.passed,
            "Multiplication should pass FP32 precision: " + result);
    }

    @Test
    void testShuntingDynamics_FP32vsFP64() {
        log.info("Testing shunting dynamics equation precision");

        // Shunting equation: dx/dt = -Ax + (B-x)I_exc - (x+D)I_inh
        double A = 0.1;
        double B = 1.0;
        double D = 0.2;
        double dt = 0.01;

        var x64 = new double[TEST_SIZE];
        var I_exc64 = new double[TEST_SIZE];
        var I_inh64 = new double[TEST_SIZE];
        var x32 = new float[TEST_SIZE];
        var I_exc32 = new float[TEST_SIZE];
        var I_inh32 = new float[TEST_SIZE];

        // Initialize
        for (int i = 0; i < TEST_SIZE; i++) {
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

        // Run 100 iterations of shunting dynamics
        var fp64Task = (Runnable) () -> {
            for (int iter = 0; iter < 100; iter++) {
                for (int i = 0; i < TEST_SIZE; i++) {
                    double dx = -A * x64[i] +
                                (B - x64[i]) * I_exc64[i] -
                                (x64[i] + D) * I_inh64[i];
                    x64[i] = x64[i] + dx * dt;
                    x64[i] = Math.max(0.0, Math.min(B, x64[i]));
                }
            }
        };

        var fp32Task = (Runnable) () -> {
            for (int iter = 0; iter < 100; iter++) {
                for (int i = 0; i < TEST_SIZE; i++) {
                    float dx = (float) (-A * x32[i] +
                                        (B - x32[i]) * I_exc32[i] -
                                        (x32[i] + D) * I_inh32[i]);
                    x32[i] = x32[i] + dx * (float) dt;
                    x32[i] = Math.max(0.0f, Math.min((float) B, x32[i]));
                }
            }
        };

        var result = PrecisionValidator.compare(
            "Shunting Dynamics (100 iterations)",
            fp64Task,
            fp32Task,
            () -> x64,
            () -> x32,
            1e-4  // Slightly relaxed due to accumulation
        );

        assertTrue(result.passed,
            "Shunting dynamics should pass with FP32: " + result);
        assertTrue(result.maxError < 1e-3,
            "Max error after 100 iterations should be < 1e-3: " + result.maxError);

        log.info("Worst errors in shunting dynamics:");
        for (int i = 0; i < Math.min(5, result.worstErrors.size()); i++) {
            log.info("  {}", result.worstErrors.get(i));
        }
    }

    @Test
    void testHebbianWeightUpdate_FP32vsFP64() {
        log.info("Testing Hebbian weight update precision");

        double learningRate = 0.01;
        int iterations = 1000;

        var weights64 = new double[TEST_SIZE];
        var pre64 = new double[TEST_SIZE];
        var post64 = new double[TEST_SIZE];
        var weights32 = new float[TEST_SIZE];
        var pre32 = new float[TEST_SIZE];
        var post32 = new float[TEST_SIZE];

        // Initialize
        for (int i = 0; i < TEST_SIZE; i++) {
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

        // Run Hebbian learning: Δw = η * pre * post
        var fp64Task = (Runnable) () -> {
            for (int iter = 0; iter < iterations; iter++) {
                for (int i = 0; i < TEST_SIZE; i++) {
                    double deltaW = learningRate * pre64[i] * post64[i];
                    weights64[i] += deltaW;
                    weights64[i] = Math.max(0.0, Math.min(1.0, weights64[i]));
                }
            }
        };

        var fp32Task = (Runnable) () -> {
            for (int iter = 0; iter < iterations; iter++) {
                for (int i = 0; i < TEST_SIZE; i++) {
                    float deltaW = (float) (learningRate * pre32[i] * post32[i]);
                    weights32[i] += deltaW;
                    weights32[i] = Math.max(0.0f, Math.min(1.0f, weights32[i]));
                }
            }
        };

        var result = PrecisionValidator.compare(
            "Hebbian Learning (1000 iterations)",
            fp64Task,
            fp32Task,
            () -> weights64,
            () -> weights32,
            1e-4  // Relaxed due to accumulation
        );

        assertTrue(result.passed,
            "Hebbian learning should converge similarly with FP32: " + result);
        assertTrue(result.maxError < 1e-3,
            "Weight difference after 1000 updates should be < 1e-3: " + result.maxError);

        log.info("Hebbian learning precision: max error = %.6e, avg error = %.6e",
            result.maxError, result.avgError);
    }

    @Test
    void testComplementCoding_FP32vsFP64() {
        log.info("Testing complement coding precision");

        var input64 = new double[TEST_SIZE];
        var input32 = new float[TEST_SIZE];

        for (int i = 0; i < TEST_SIZE; i++) {
            double val = random.nextDouble();
            input64[i] = val;
            input32[i] = (float) val;
        }

        // Complement coding: [x, 1-x]
        var coded64 = new double[TEST_SIZE * 2];
        var fp64Task = (Runnable) () -> {
            for (int i = 0; i < TEST_SIZE; i++) {
                coded64[i] = input64[i];
                coded64[TEST_SIZE + i] = 1.0 - input64[i];
            }
        };

        var coded32 = new float[TEST_SIZE * 2];
        var fp32Task = (Runnable) () -> {
            for (int i = 0; i < TEST_SIZE; i++) {
                coded32[i] = input32[i];
                coded32[TEST_SIZE + i] = 1.0f - input32[i];
            }
        };

        var result = PrecisionValidator.compare(
            "Complement Coding",
            fp64Task,
            fp32Task,
            () -> coded64,
            () -> coded32,
            FP32_TOLERANCE
        );

        assertTrue(result.passed,
            "Complement coding should pass FP32 precision: " + result);
    }
}
