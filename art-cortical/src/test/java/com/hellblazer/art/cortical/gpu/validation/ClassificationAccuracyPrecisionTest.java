package com.hellblazer.art.cortical.gpu.validation;

import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test that FP32 precision doesn't degrade classification accuracy.
 * Simulates simplified ART category matching and activation.
 */
class ClassificationAccuracyPrecisionTest {

    private static final Logger log = LoggerFactory.getLogger(ClassificationAccuracyPrecisionTest.class);
    private static final Random random = new Random(42);

    @Test
    void testCategoryActivation_FP32vsFP64() {
        log.info("Testing category activation precision");

        int numPatterns = 1000;
        int numCategories = 50;
        int patternSize = 128;

        // Generate random patterns and category weights
        var patterns = generatePatterns(numPatterns, patternSize);
        var categoryWeights = generatePatterns(numCategories, patternSize);

        // Compute category activations with FP64
        var activations64 = new double[numPatterns * numCategories];
        var fp64Task = (Runnable) () -> {
            for (int p = 0; p < numPatterns; p++) {
                for (int c = 0; c < numCategories; c++) {
                    activations64[p * numCategories + c] =
                        computeActivation64(patterns[p], categoryWeights[c]);
                }
            }
        };

        // Compute category activations with FP32
        var patterns32 = convertToFloat(patterns);
        var weights32 = convertToFloat(categoryWeights);
        var activations32 = new float[numPatterns * numCategories];
        var fp32Task = (Runnable) () -> {
            for (int p = 0; p < numPatterns; p++) {
                for (int c = 0; c < numCategories; c++) {
                    activations32[p * numCategories + c] =
                        computeActivation32(patterns32[p], weights32[c]);
                }
            }
        };

        var result = PrecisionValidator.compare(
            "Category Activation",
            fp64Task,
            fp32Task,
            () -> activations64,
            () -> activations32,
            1e-5  // Relaxed for accumulated operations
        );

        assertTrue(result.passed,
            "Category activation should pass with FP32: " + result);
        log.info("Category activation precision validated");
    }

    @Test
    void testCategorySelection_FP32vsFP64() {
        log.info("Testing category selection (classification) agreement");

        int numPatterns = 1000;
        int numCategories = 50;
        int patternSize = 128;

        var patterns = generatePatterns(numPatterns, patternSize);
        var categoryWeights = generatePatterns(numCategories, patternSize);

        // Find best matching category for each pattern (FP64)
        var selected64 = new int[numPatterns];
        for (int p = 0; p < numPatterns; p++) {
            double maxActivation = Double.NEGATIVE_INFINITY;
            int bestCategory = -1;

            for (int c = 0; c < numCategories; c++) {
                double activation = computeActivation64(patterns[p], categoryWeights[c]);
                if (activation > maxActivation) {
                    maxActivation = activation;
                    bestCategory = c;
                }
            }
            selected64[p] = bestCategory;
        }

        // Find best matching category for each pattern (FP32)
        var patterns32 = convertToFloat(patterns);
        var weights32 = convertToFloat(categoryWeights);
        var selected32 = new int[numPatterns];
        for (int p = 0; p < numPatterns; p++) {
            float maxActivation = Float.NEGATIVE_INFINITY;
            int bestCategory = -1;

            for (int c = 0; c < numCategories; c++) {
                float activation = computeActivation32(patterns32[p], weights32[c]);
                if (activation > maxActivation) {
                    maxActivation = activation;
                    bestCategory = c;
                }
            }
            selected32[p] = bestCategory;
        }

        // Compute classification agreement
        int agreements = 0;
        for (int p = 0; p < numPatterns; p++) {
            if (selected64[p] == selected32[p]) {
                agreements++;
            }
        }

        double accuracy = (double) agreements / numPatterns;
        log.info("Classification agreement: {}/{} ({:.2f}%)",
            agreements, numPatterns, accuracy * 100);

        assertTrue(accuracy >= 0.999,
            String.format("Classification agreement should be ≥99.9%%, got %.2f%%", accuracy * 100));

        log.info("✅ Classification accuracy: {:.2f}% agreement (target: ≥99.9%)", accuracy * 100);
    }

    @Test
    void testResonanceCriterion_FP32vsFP64() {
        log.info("Testing resonance criterion precision");

        int numTests = 10000;
        int patternSize = 128;
        double vigilance = 0.7;

        // Generate test patterns and category weights
        var patterns = generatePatterns(numTests, patternSize);
        var weights = generatePatterns(numTests, patternSize);

        // Test resonance criterion: |x ∧ w| / |x| ≥ ρ
        var resonance64 = new boolean[numTests];
        for (int i = 0; i < numTests; i++) {
            double match = computeMatch64(patterns[i], weights[i]);
            double norm = computeNorm64(patterns[i]);
            resonance64[i] = (match / norm) >= vigilance;
        }

        var patterns32 = convertToFloat(patterns);
        var weights32 = convertToFloat(weights);
        var resonance32 = new boolean[numTests];
        for (int i = 0; i < numTests; i++) {
            float match = computeMatch32(patterns32[i], weights32[i]);
            float norm = computeNorm32(patterns32[i]);
            resonance32[i] = (match / norm) >= vigilance;
        }

        // Count agreements
        int agreements = 0;
        for (int i = 0; i < numTests; i++) {
            if (resonance64[i] == resonance32[i]) {
                agreements++;
            }
        }

        double agreement = (double) agreements / numTests;
        log.info("Resonance criterion agreement: {}/{} ({:.2f}%)",
            agreements, numTests, agreement * 100);

        assertTrue(agreement >= 0.999,
            String.format("Resonance agreement should be ≥99.9%%, got %.2f%%", agreement * 100));

        log.info("✅ Resonance criterion: {:.2f}% agreement", agreement * 100);
    }

    // Helper methods

    private double[][] generatePatterns(int count, int size) {
        var patterns = new double[count][size];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < size; j++) {
                patterns[i][j] = random.nextDouble();
            }
        }
        return patterns;
    }

    private float[][] convertToFloat(double[][] patterns) {
        var result = new float[patterns.length][patterns[0].length];
        for (int i = 0; i < patterns.length; i++) {
            for (int j = 0; j < patterns[i].length; j++) {
                result[i][j] = (float) patterns[i][j];
            }
        }
        return result;
    }

    // ART category activation: T_j = |x ∧ w_j| / (α + |w_j|)
    private double computeActivation64(double[] pattern, double[] weight) {
        double alpha = 0.001;  // Choice parameter
        double match = computeMatch64(pattern, weight);
        double weightNorm = computeNorm64(weight);
        return match / (alpha + weightNorm);
    }

    private float computeActivation32(float[] pattern, float[] weight) {
        float alpha = 0.001f;
        float match = computeMatch32(pattern, weight);
        float weightNorm = computeNorm32(weight);
        return match / (alpha + weightNorm);
    }

    // Fuzzy AND (min): |x ∧ w|
    private double computeMatch64(double[] x, double[] w) {
        double sum = 0.0;
        for (int i = 0; i < x.length; i++) {
            sum += Math.min(x[i], w[i]);
        }
        return sum;
    }

    private float computeMatch32(float[] x, float[] w) {
        float sum = 0.0f;
        for (int i = 0; i < x.length; i++) {
            sum += Math.min(x[i], w[i]);
        }
        return sum;
    }

    // L1 norm
    private double computeNorm64(double[] x) {
        double sum = 0.0;
        for (double v : x) {
            sum += v;
        }
        return sum;
    }

    private float computeNorm32(float[] x) {
        float sum = 0.0f;
        for (float v : x) {
            sum += v;
        }
        return sum;
    }
}
