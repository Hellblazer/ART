package com.hellblazer.art.cortical.learning;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.cortical.layers.WeightMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for BCM (Bienenstock-Cooper-Munro) learning.
 *
 * <p>Verifies:
 * <ul>
 *   <li>Sliding threshold adaptation</li>
 *   <li>LTP (Long-Term Potentiation) when y > θ</li>
 *   <li>LTD (Long-Term Depression) when y < θ</li>
 *   <li>Selectivity development</li>
 *   <li>Homeostatic stability</li>
 * </ul>
 *
 * @author Phase 3D: Advanced Learning Tests
 */
class BCMLearningTest {

    @Test
    void testConstructorValidation() {
        // Invalid threshold decay rate
        assertThrows(IllegalArgumentException.class, () ->
            new BCMLearning(-0.1, 0.0001, 0.0, 1.0)
        );
        assertThrows(IllegalArgumentException.class, () ->
            new BCMLearning(1.5, 0.0001, 0.0, 1.0)
        );

        // Invalid weight decay rate
        assertThrows(IllegalArgumentException.class, () ->
            new BCMLearning(0.5, -0.1, 0.0, 1.0)
        );
        assertThrows(IllegalArgumentException.class, () ->
            new BCMLearning(0.5, 1.5, 0.0, 1.0)
        );

        // Invalid weight bounds
        assertThrows(IllegalArgumentException.class, () ->
            new BCMLearning(0.5, 0.0001, -0.1, 1.0)
        );
        assertThrows(IllegalArgumentException.class, () ->
            new BCMLearning(0.5, 0.0001, 0.8, 0.5)
        );

        // Valid parameters
        assertDoesNotThrow(() ->
            new BCMLearning(0.5, 0.0001, 0.0, 1.0)
        );
    }

    @Test
    void testFactoryMethods() {
        var competitive = BCMLearning.createCompetitive();
        assertEquals(0.8, competitive.getThresholdDecayRate());
        assertEquals(0.0001, competitive.getWeightDecayRate());

        var balanced = BCMLearning.createBalanced();
        assertEquals(0.5, balanced.getThresholdDecayRate());
        assertEquals(0.0005, balanced.getWeightDecayRate());

        var homeostatic = BCMLearning.createHomeostatic();
        assertEquals(0.1, homeostatic.getThresholdDecayRate());
        assertEquals(0.0001, homeostatic.getWeightDecayRate());
    }

    @Test
    void testThresholdInitialization() {
        var bcm = BCMLearning.createBalanced();

        // Thresholds should be null before first use
        assertNull(bcm.getModificationThresholds());

        // Apply learning once to initialize
        var pattern = new DenseVector(new double[]{0.5, 0.5});
        var activation = new DenseVector(new double[]{0.3, 0.7});
        var weights = new WeightMatrix(2, 2);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                weights.set(i, j, 0.5);
            }
        }

        bcm.update(pattern, activation, weights, 0.1);

        // Thresholds should now be initialized
        var thresholds = bcm.getModificationThresholds();
        assertNotNull(thresholds);
        assertEquals(2, thresholds.length);

        // All thresholds should be positive
        for (double threshold : thresholds) {
            assertTrue(threshold > 0.0);
        }
    }

    @Test
    void testThresholdAdaptation() {
        var bcm = new BCMLearning(0.5, 0.0, 0.0, 1.0);  // Fast threshold

        var pattern = new DenseVector(new double[]{1.0, 1.0});
        var lowActivation = new DenseVector(new double[]{0.2, 0.2});
        var highActivation = new DenseVector(new double[]{0.8, 0.8});

        var weights = new WeightMatrix(2, 2);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                weights.set(i, j, 0.5);
            }
        }

        // Train with low activation
        for (int i = 0; i < 10; i++) {
            weights = bcm.update(pattern, lowActivation, weights, 0.1);
        }
        var thresholdsLow = bcm.getModificationThresholds();

        // Train with high activation
        for (int i = 0; i < 10; i++) {
            weights = bcm.update(pattern, highActivation, weights, 0.1);
        }
        var thresholdsHigh = bcm.getModificationThresholds();

        // Thresholds should increase with higher activity
        assertTrue(thresholdsHigh[0] > thresholdsLow[0]);
        assertTrue(thresholdsHigh[1] > thresholdsLow[1]);
    }

    @Test
    void testLTPandLTD() {
        // Use slower threshold for this test to demonstrate clear LTP/LTD
        var bcm = new BCMLearning(0.1, 0.0005, 0.0, 1.0);

        var pattern = new DenseVector(new double[]{1.0, 0.0});
        var weights = new WeightMatrix(1, 2);
        weights.set(0, 0, 0.5);
        weights.set(0, 1, 0.5);

        // First, establish a threshold
        var activation = new DenseVector(new double[]{0.5});
        for (int i = 0; i < 5; i++) {
            weights = bcm.update(pattern, activation, weights, 0.1);
        }

        double initialWeight0 = weights.get(0, 0);
        double initialWeight1 = weights.get(0, 1);

        // Now test LTP with high activation (y > θ)
        var highActivation = new DenseVector(new double[]{0.9});
        for (int i = 0; i < 10; i++) {
            weights = bcm.update(pattern, highActivation, weights, 0.1);
        }

        // Weight for active input (1.0) should increase (LTP)
        assertTrue(weights.get(0, 0) > initialWeight0);

        // Weight for inactive input (0.0) might decrease or stay similar
        // (depends on threshold dynamics)

        // Now test LTD with low activation (y < θ)
        initialWeight0 = weights.get(0, 0);
        var lowActivation = new DenseVector(new double[]{0.1});
        for (int i = 0; i < 10; i++) {
            weights = bcm.update(pattern, lowActivation, weights, 0.1);
        }

        // With very low activation, weights should decrease (LTD)
        assertTrue(weights.get(0, 0) < initialWeight0);
    }

    @Test
    void testSelectivityDevelopment() {
        var bcm = BCMLearning.createCompetitive();  // Fast threshold for quick selectivity

        // Two different patterns
        var pattern1 = new DenseVector(new double[]{1.0, 0.0, 0.0});
        var pattern2 = new DenseVector(new double[]{0.0, 1.0, 0.0});

        var weights = new WeightMatrix(2, 3);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                weights.set(i, j, 0.5);
            }
        }

        // Alternately present both patterns, with different neurons responding
        for (int iter = 0; iter < 20; iter++) {
            // Neuron 0 responds strongly to pattern1
            var activation1 = new DenseVector(new double[]{0.8, 0.2});
            weights = bcm.update(pattern1, activation1, weights, 0.2);

            // Neuron 1 responds strongly to pattern2
            var activation2 = new DenseVector(new double[]{0.2, 0.8});
            weights = bcm.update(pattern2, activation2, weights, 0.2);
        }

        // Neuron 0 should be selective to pattern1 (high weight for feature 0)
        assertTrue(weights.get(0, 0) > weights.get(0, 1));
        assertTrue(weights.get(0, 0) > weights.get(0, 2));

        // Neuron 1 should be selective to pattern2 (high weight for feature 1)
        assertTrue(weights.get(1, 1) > weights.get(1, 0));
        assertTrue(weights.get(1, 1) > weights.get(1, 2));
    }

    @Test
    void testWeightDecay() {
        var bcm = new BCMLearning(0.1, 0.2, 0.0, 1.0);  // High weight decay

        var zeroPattern = new DenseVector(new double[]{0.0, 0.0});
        var zeroActivation = new DenseVector(new double[]{0.0});

        var weights = new WeightMatrix(1, 2);
        weights.set(0, 0, 0.8);
        weights.set(0, 1, 0.6);

        // Apply learning with zero input/activation
        for (int i = 0; i < 10; i++) {
            weights = bcm.update(zeroPattern, zeroActivation, weights, 0.5);
        }

        // Weights should decay toward zero
        assertTrue(weights.get(0, 0) < 0.8);
        assertTrue(weights.get(0, 1) < 0.6);
    }

    @Test
    void testWeightBounds() {
        var bcm = BCMLearning.createBalanced();

        var pattern = new DenseVector(new double[]{1.0, 1.0});
        var activation = new DenseVector(new double[]{0.9});

        var weights = new WeightMatrix(1, 2);
        weights.set(0, 0, 0.95);
        weights.set(0, 1, 0.95);

        // Apply aggressive learning that would exceed bounds
        for (int i = 0; i < 20; i++) {
            weights = bcm.update(pattern, activation, weights, 0.9);
        }

        // Weights should be clipped to [0, 1]
        assertTrue(weights.get(0, 0) <= 1.0);
        assertTrue(weights.get(0, 1) <= 1.0);
        assertTrue(weights.get(0, 0) >= 0.0);
        assertTrue(weights.get(0, 1) >= 0.0);
    }

    @Test
    void testResetThresholds() {
        var bcm = BCMLearning.createBalanced();

        var pattern = new DenseVector(new double[]{1.0, 0.5});
        var activation = new DenseVector(new double[]{0.7, 0.3});
        var weights = new WeightMatrix(2, 2);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                weights.set(i, j, 0.5);
            }
        }

        // Train to establish thresholds
        for (int i = 0; i < 10; i++) {
            weights = bcm.update(pattern, activation, weights, 0.1);
        }

        var thresholdsBefore = bcm.getModificationThresholds();
        assertTrue(thresholdsBefore[0] > 0.1);  // Should have adapted

        // Reset thresholds
        bcm.resetThresholds();

        var thresholdsAfter = bcm.getModificationThresholds();
        assertEquals(0.1, thresholdsAfter[0], 0.01);  // Back to initial value
        assertEquals(0.1, thresholdsAfter[1], 0.01);
    }

    @Test
    void testDimensionValidation() {
        var bcm = BCMLearning.createBalanced();

        var pattern = new DenseVector(new double[]{1.0, 0.0});
        var activation = new DenseVector(new double[]{1.0});
        var wrongWeights = new WeightMatrix(1, 3);  // Wrong size

        assertThrows(IllegalArgumentException.class, () ->
            bcm.update(pattern, activation, wrongWeights, 0.1)
        );
    }

    @Test
    void testLearningRateValidation() {
        var bcm = BCMLearning.createBalanced();

        var pattern = new DenseVector(new double[]{1.0, 0.0});
        var activation = new DenseVector(new double[]{1.0});
        var weights = new WeightMatrix(1, 2);

        assertThrows(IllegalArgumentException.class, () ->
            bcm.update(pattern, activation, weights, -0.1)
        );
        assertThrows(IllegalArgumentException.class, () ->
            bcm.update(pattern, activation, weights, 1.5)
        );

        assertDoesNotThrow(() ->
            bcm.update(pattern, activation, weights, 0.0)
        );
        assertDoesNotThrow(() ->
            bcm.update(pattern, activation, weights, 1.0)
        );
    }

    @Test
    void testGetName() {
        assertEquals("BCM", BCMLearning.createBalanced().getName());
    }

    @Test
    void testRequiresNormalization() {
        assertFalse(BCMLearning.createBalanced().requiresNormalization());
    }

    @Test
    void testRecommendedLearningRates() {
        var range = BCMLearning.createBalanced().getRecommendedLearningRateRange();
        assertEquals(0.01, range[0], 0.001);
        assertEquals(0.5, range[1], 0.001);
    }

    @Test
    void testToString() {
        var bcm = new BCMLearning(0.5, 0.001, 0.0, 1.0);
        var str = bcm.toString();
        assertTrue(str.contains("BCMLearning"));
        assertTrue(str.contains("0.5"));
        assertTrue(str.contains("0.001"));
    }

    @Test
    void testHomeostaticStability() {
        var bcm = BCMLearning.createHomeostatic();  // Very slow threshold

        var pattern = new DenseVector(new double[]{1.0});
        var weights = new WeightMatrix(1, 1);
        weights.set(0, 0, 0.5);

        // Train with consistent activation
        double targetActivation = 0.6;
        for (int i = 0; i < 100; i++) {
            var activation = new DenseVector(new double[]{targetActivation});
            weights = bcm.update(pattern, activation, weights, 0.05);
        }

        // Threshold should converge near y²
        var thresholds = bcm.getModificationThresholds();
        double expectedThreshold = targetActivation * targetActivation;

        // With homeostatic learning, threshold should approach y²
        assertTrue(Math.abs(thresholds[0] - expectedThreshold) < 0.2);
    }
}
