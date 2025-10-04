package com.hellblazer.art.cortical.learning;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.cortical.layers.WeightMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Hebbian learning rule.
 *
 * <p>Phase 3A: Core Learning Infrastructure
 *
 * @author Phase 3A Tests
 */
public class HebbianLearningTest {

    private static final double TOLERANCE = 1e-10;

    /**
     * Test basic Hebbian learning with simple activations.
     */
    @Test
    public void testBasicHebbianUpdate() {
        var learning = new HebbianLearning(0.0001, 0.0, 1.0);

        // Simple 2x2 scenario
        var preActivation = new DenseVector(new double[]{1.0, 0.0});
        var postActivation = new DenseVector(new double[]{1.0, 0.0});

        var weights = new WeightMatrix(2, 2);
        weights.set(0, 0, 0.5);  // Initial weight

        // Apply learning
        var newWeights = learning.update(preActivation, postActivation, weights, 0.1);

        // Hebbian: Δw[0][0] = 0.1 * 1.0 * 1.0 = 0.1
        // Decay: -0.0001 * 0.1 * 0.5 = -0.000005
        // New weight: 0.5 + 0.1 - 0.000005 ≈ 0.599995
        assertTrue(newWeights.get(0, 0) > 0.5, "Weight should increase due to correlated activity");
        assertTrue(newWeights.get(0, 0) < 0.61, "Weight should not increase too much");

        // Other weights should remain near zero (no co-activation)
        assertTrue(Math.abs(newWeights.get(0, 1)) < 0.001, "No co-activation");
        assertTrue(Math.abs(newWeights.get(1, 0)) < 0.001, "No co-activation");
        assertTrue(Math.abs(newWeights.get(1, 1)) < 0.001, "No co-activation");
    }

    /**
     * Test that weight decay prevents runaway growth.
     */
    @Test
    public void testWeightDecay() {
        var learning = new HebbianLearning(0.01, 0.0, 1.0);  // High decay

        var preActivation = new DenseVector(new double[]{0.0});
        var postActivation = new DenseVector(new double[]{0.0});

        var weights = new WeightMatrix(1, 1);
        weights.set(0, 0, 0.8);  // High initial weight

        // Apply learning with no activity (only decay should occur)
        var newWeights = learning.update(preActivation, postActivation, weights, 0.1);

        // Decay: -0.01 * 0.1 * 0.8 = -0.0008
        // New weight: 0.8 - 0.0008 = 0.7992
        assertTrue(newWeights.get(0, 0) < 0.8, "Weight should decay");
        assertEquals(0.7992, newWeights.get(0, 0), 0.0001, "Decay should reduce weight");
    }

    /**
     * Test weight bounds (clipping).
     */
    @Test
    public void testWeightBounds() {
        var learning = new HebbianLearning(0.0, 0.0, 1.0);  // No decay for clarity

        // Test upper bound
        var preActivation = new DenseVector(new double[]{1.0});
        var postActivation = new DenseVector(new double[]{1.0});

        var weights = new WeightMatrix(1, 1);
        weights.set(0, 0, 0.95);  // Near upper bound

        // Large learning rate should clip to 1.0
        var newWeights = learning.update(preActivation, postActivation, weights, 1.0);
        assertEquals(1.0, newWeights.get(0, 0), TOLERANCE, "Weight should clip to max");

        // Test lower bound
        weights = new WeightMatrix(1, 1);
        weights.set(0, 0, 0.0);  // At lower bound

        preActivation = new DenseVector(new double[]{0.0});
        postActivation = new DenseVector(new double[]{0.0});

        newWeights = learning.update(preActivation, postActivation, weights, 0.1);
        assertEquals(0.0, newWeights.get(0, 0), TOLERANCE, "Weight should stay at min");
    }

    /**
     * Test zero learning rate (no change).
     */
    @Test
    public void testZeroLearningRate() {
        var learning = new HebbianLearning();

        var preActivation = new DenseVector(new double[]{1.0, 1.0});
        var postActivation = new DenseVector(new double[]{1.0, 1.0});

        var weights = new WeightMatrix(2, 2);
        weights.set(0, 0, 0.5);
        weights.set(0, 1, 0.3);
        weights.set(1, 0, 0.7);
        weights.set(1, 1, 0.2);

        // Zero learning rate = no change
        var newWeights = learning.update(preActivation, postActivation, weights, 0.0);

        assertEquals(0.5, newWeights.get(0, 0), TOLERANCE);
        assertEquals(0.3, newWeights.get(0, 1), TOLERANCE);
        assertEquals(0.7, newWeights.get(1, 0), TOLERANCE);
        assertEquals(0.2, newWeights.get(1, 1), TOLERANCE);
    }

    /**
     * Test that Hebbian learning strengthens correlated connections.
     */
    @Test
    public void testHebbianStrengthening() {
        var learning = new HebbianLearning(0.0, 0.0, 1.0);  // No decay

        var weights = new WeightMatrix(3, 3);

        // Train with correlated patterns
        for (int i = 0; i < 100; i++) {
            var pattern = new DenseVector(new double[]{1.0, 0.5, 0.0});
            var output = new DenseVector(new double[]{1.0, 0.5, 0.0});

            weights = learning.update(pattern, output, weights, 0.01);
        }

        // Weights should reflect correlations
        // Strong: (0,0), (1,1) - both active
        // Medium: (0,1), (1,0) - partially active
        // Weak: all involving index 2 - not active
        assertTrue(weights.get(0, 0) > 0.5, "Strongly correlated connection");
        assertTrue(weights.get(1, 1) > 0.2, "Moderately correlated connection");
        assertTrue(weights.get(2, 2) < 0.01, "Uncorrelated connection");
    }

    /**
     * Test dimension validation.
     */
    @Test
    public void testDimensionValidation() {
        var learning = new HebbianLearning();

        var preActivation = new DenseVector(new double[]{1.0, 1.0});
        var postActivation = new DenseVector(new double[]{1.0, 1.0, 1.0});  // Wrong size

        var weights = new WeightMatrix(2, 2);

        assertThrows(IllegalArgumentException.class, () ->
            learning.update(preActivation, postActivation, weights, 0.1)
        );
    }

    /**
     * Test learning rate validation.
     */
    @Test
    public void testLearningRateValidation() {
        var learning = new HebbianLearning();

        var preActivation = new DenseVector(new double[]{1.0});
        var postActivation = new DenseVector(new double[]{1.0});
        var weights = new WeightMatrix(1, 1);

        assertThrows(IllegalArgumentException.class, () ->
            learning.update(preActivation, postActivation, weights, -0.1)
        );

        assertThrows(IllegalArgumentException.class, () ->
            learning.update(preActivation, postActivation, weights, 1.1)
        );
    }

    /**
     * Test parameter validation in constructor.
     */
    @Test
    public void testConstructorValidation() {
        assertThrows(IllegalArgumentException.class, () ->
            new HebbianLearning(-0.1, 0.0, 1.0)  // Negative decay
        );

        assertThrows(IllegalArgumentException.class, () ->
            new HebbianLearning(1.1, 0.0, 1.0)  // Decay > 1
        );

        assertThrows(IllegalArgumentException.class, () ->
            new HebbianLearning(0.01, 1.0, 0.0)  // Min > max
        );
    }

    /**
     * Test default constructor.
     */
    @Test
    public void testDefaultConstructor() {
        var learning = new HebbianLearning();

        assertEquals(0.0001, learning.getDecayRate(), TOLERANCE);
        assertEquals(0.0, learning.getMinWeight(), TOLERANCE);
        assertEquals(1.0, learning.getMaxWeight(), TOLERANCE);
    }

    /**
     * Test getName().
     */
    @Test
    public void testGetName() {
        var learning = new HebbianLearning();
        assertEquals("Hebbian", learning.getName());
    }

    /**
     * Test requiresNormalization().
     */
    @Test
    public void testRequiresNormalization() {
        var learning = new HebbianLearning();
        assertFalse(learning.requiresNormalization(), "Hebbian with decay doesn't need normalization");
    }

    /**
     * Test getRecommendedLearningRateRange().
     */
    @Test
    public void testRecommendedLearningRateRange() {
        var learning = new HebbianLearning();
        var range = learning.getRecommendedLearningRateRange();

        assertEquals(2, range.length);
        assertTrue(range[0] > 0.0, "Min learning rate should be positive");
        assertTrue(range[1] > range[0], "Max should be greater than min");
        assertTrue(range[1] <= 1.0, "Max learning rate should be <= 1.0");
    }

    /**
     * Test toString().
     */
    @Test
    public void testToString() {
        var learning = new HebbianLearning(0.01, 0.0, 1.0);
        var str = learning.toString();

        assertTrue(str.contains("Hebbian"));
        assertTrue(str.contains("0.01"));
        assertTrue(str.contains("0.0"));
        assertTrue(str.contains("1.0"));
    }

    /**
     * Test realistic training scenario with multiple patterns.
     */
    @Test
    public void testRealisticTrainingScenario() {
        var learning = new HebbianLearning(0.0001, 0.0, 1.0);

        var weights = new WeightMatrix(4, 4);

        // Training patterns (simple identity mapping)
        var patterns = new DenseVector[]{
            new DenseVector(new double[]{1.0, 0.0, 0.0, 0.0}),
            new DenseVector(new double[]{0.0, 1.0, 0.0, 0.0}),
            new DenseVector(new double[]{0.0, 0.0, 1.0, 0.0}),
            new DenseVector(new double[]{0.0, 0.0, 0.0, 1.0})
        };

        // Train for multiple epochs
        for (int epoch = 0; epoch < 50; epoch++) {
            for (var pattern : patterns) {
                weights = learning.update(pattern, pattern, weights, 0.01);
            }
        }

        // Diagonal weights should be strong (auto-associative)
        for (int i = 0; i < 4; i++) {
            assertTrue(weights.get(i, i) > 0.3, "Diagonal weight " + i + " should be strong");
        }

        // Off-diagonal weights should remain weak
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (i != j) {
                    assertTrue(weights.get(i, j) < 0.1, "Off-diagonal weight [" + i + "][" + j + "] should be weak");
                }
            }
        }
    }
}
