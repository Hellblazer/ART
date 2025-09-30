package com.hellblazer.art.laminar.canonical;

import com.hellblazer.art.core.Pattern;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for PredictionGenerator - generates top-down expectations from category activations.
 *
 * Implements test-first RED-GREEN-REFACTOR approach for Day 1 of Phase 2 Week 3.
 * These tests verify the prediction generation mechanism that creates top-down
 * expectations from Layer 5 category activations.
 *
 * @author Claude Code
 */
class PredictionGeneratorTest {

    private static final double EPSILON = 0.01;
    private PredictionGenerator generator;
    private int featureDimension;

    @BeforeEach
    void setUp() {
        featureDimension = 10;
        generator = new PredictionGenerator(featureDimension, new PredictionParameters());
    }

    /**
     * Test 1: Single category generates expected pattern.
     *
     * Given: A single active category with learned template
     * When: Generate expectation from single category activation
     * Then: Expectation matches category template scaled by topDownGain
     */
    @Test
    void testSingleCategoryPrediction() {
        // Learn a specific template for category 0
        var template = Pattern.of(0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0);
        generator.updateTemplate(0, template, 1.0);  // Learning rate 1.0 = immediate learning

        // Activate only category 0 with full strength
        var categoryActivation = Pattern.of(1.0, 0.0, 0.0, 0.0, 0.0);
        var expectation = generator.generateExpectation(categoryActivation);

        // Expectation should be template scaled by default topDownGain (0.5)
        assertEquals(featureDimension, expectation.dimension());
        for (int i = 0; i < featureDimension; i++) {
            var expected = template.get(i) * 0.5;  // default topDownGain = 0.5
            assertEquals(expected, expectation.get(i), EPSILON,
                "Feature " + i + " expectation mismatch");
        }
    }

    /**
     * Test 2: Multiple categories blend predictions.
     *
     * Given: Two active categories with different templates
     * When: Generate expectation from both categories
     * Then: Expectation is weighted average of templates
     */
    @Test
    void testMultipleCategoryBlending() {
        // Learn two distinct templates
        var template1 = Pattern.of(1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0);
        var template2 = Pattern.of(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

        generator.updateTemplate(0, template1, 1.0);
        generator.updateTemplate(1, template2, 1.0);

        // Activate both categories with equal strength
        var categoryActivation = Pattern.of(0.5, 0.5, 0.0, 0.0, 0.0);
        var expectation = generator.generateExpectation(categoryActivation);

        // Expectation should be weighted average: (T1*0.5 + T2*0.5) / (0.5+0.5) * gain
        // Which simplifies to: (T1 + T2) / 2 * 0.5
        assertEquals(featureDimension, expectation.dimension());
        for (int i = 0; i < featureDimension; i++) {
            var expected = ((template1.get(i) + template2.get(i)) / 2.0) * 0.5;
            assertEquals(expected, expectation.get(i), EPSILON,
                "Feature " + i + " blending mismatch");
        }
    }

    /**
     * Test 3: Attention modulates prediction strength.
     *
     * Given: Category with template and varying activation strengths
     * When: Generate expectations with different category activation levels
     * Then: Prediction strength varies with activation (but still normalized)
     */
    @Test
    void testAttentionModulation() {
        // Learn template for category 0
        var template = Pattern.of(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8);
        generator.updateTemplate(0, template, 1.0);

        // Test with weak activation
        var weakActivation = Pattern.of(0.2, 0.0, 0.0, 0.0, 0.0);
        var weakExpectation = generator.generateExpectation(weakActivation);

        // Test with strong activation
        var strongActivation = Pattern.of(1.0, 0.0, 0.0, 0.0, 0.0);
        var strongExpectation = generator.generateExpectation(strongActivation);

        // Both should produce same expectation pattern (normalized by total activation)
        // because single category activation normalizes out
        assertEquals(featureDimension, weakExpectation.dimension());
        assertEquals(featureDimension, strongExpectation.dimension());

        for (int i = 0; i < featureDimension; i++) {
            assertEquals(strongExpectation.get(i), weakExpectation.get(i), EPSILON,
                "Normalized expectations should be equal for single category");
        }
    }

    /**
     * Test 4: Zero activations produce no prediction.
     *
     * Given: All category activations are zero
     * When: Generate expectation
     * Then: Expectation is all zeros (no active prediction)
     */
    @Test
    void testZeroCategoryHandling() {
        // Learn templates for several categories
        generator.updateTemplate(0, Pattern.of(0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0), 1.0);
        generator.updateTemplate(1, Pattern.of(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), 1.0);

        // All categories inactive
        var zeroActivation = Pattern.of(0.0, 0.0, 0.0, 0.0, 0.0);
        var expectation = generator.generateExpectation(zeroActivation);

        // Should produce zero expectation
        assertEquals(featureDimension, expectation.dimension());
        for (int i = 0; i < featureDimension; i++) {
            assertEquals(0.0, expectation.get(i), EPSILON,
                "Zero activation should produce zero expectation");
        }
    }

    /**
     * Test 5: Predictions normalized to [0,1].
     *
     * Given: Templates with values that could exceed [0,1] when blended
     * When: Generate expectation
     * Then: All prediction values are in valid [0,1] range
     */
    @Test
    void testPredictionNormalization() {
        // Learn templates with high values
        var template1 = Pattern.of(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        var template2 = Pattern.of(0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9);

        generator.updateTemplate(0, template1, 1.0);
        generator.updateTemplate(1, template2, 1.0);

        // Strong activation of multiple categories
        var categoryActivation = Pattern.of(0.8, 0.6, 0.0, 0.0, 0.0);
        var expectation = generator.generateExpectation(categoryActivation);

        // All values should be in [0,1] range
        assertEquals(featureDimension, expectation.dimension());
        for (int i = 0; i < featureDimension; i++) {
            assertTrue(expectation.get(i) >= 0.0,
                "Feature " + i + " below 0.0: " + expectation.get(i));
            assertTrue(expectation.get(i) <= 1.0,
                "Feature " + i + " above 1.0: " + expectation.get(i));
        }
    }

    /**
     * Test 6: Template learning converges with appropriate time constants.
     *
     * Given: Initial uncommitted template (all ones)
     * When: Repeatedly update template toward target pattern
     * Then: Template gradually converges to target (incremental learning)
     */
    @Test
    void testTemporalDynamics() {
        // Target pattern to learn
        var target = Pattern.of(0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.3, 0.2, 0.1, 0.0);

        // Initial uncommitted template should be all ones
        var initialTemplate = generator.getCategoryTemplate(0);
        for (int i = 0; i < featureDimension; i++) {
            assertEquals(1.0, initialTemplate.get(i), EPSILON,
                "Uncommitted template should be all ones");
        }

        // Simulate learning with slow learning rate (temporal dynamics)
        var learningRate = 0.1;  // Slow learning
        for (int iteration = 0; iteration < 50; iteration++) {
            generator.updateTemplate(0, target, learningRate);
        }

        // Template should have partially converged toward target
        var learnedTemplate = generator.getCategoryTemplate(0);

        // After 50 iterations with rate 0.1, should be much closer to target than initial
        // Using formula: T_new = T_old + rate * (target - T_old)
        // After n iterations: T_n ≈ target + (1-rate)^n * (T_0 - target)
        // After 50: (1-0.1)^50 ≈ 0.005, so should be ~99.5% to target
        for (int i = 0; i < featureDimension; i++) {
            var convergence = Math.abs(learnedTemplate.get(i) - target.get(i));
            assertTrue(convergence < 0.1,
                "Feature " + i + " should converge: convergence=" + convergence);
        }

        // Continue learning - should converge even closer
        for (int iteration = 0; iteration < 50; iteration++) {
            generator.updateTemplate(0, target, learningRate);
        }

        var fullyLearnedTemplate = generator.getCategoryTemplate(0);
        for (int i = 0; i < featureDimension; i++) {
            assertEquals(target.get(i), fullyLearnedTemplate.get(i), 0.01,
                "Feature " + i + " should be fully learned");
        }
    }
}