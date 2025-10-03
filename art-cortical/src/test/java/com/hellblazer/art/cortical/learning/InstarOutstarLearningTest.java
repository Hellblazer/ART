package com.hellblazer.art.cortical.learning;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.cortical.layers.WeightMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for InstarOutstarLearning (ART-style learning).
 *
 * <p>Verifies:
 * <ul>
 *   <li>Instar learning (bottom-up recognition)</li>
 *   <li>Outstar learning (top-down prediction)</li>
 *   <li>Bidirectional learning (both modes)</li>
 *   <li>Weight convergence to patterns</li>
 *   <li>Decay and bounds</li>
 * </ul>
 *
 * @author Phase 3D: Advanced Learning Tests
 */
class InstarOutstarLearningTest {

    @Test
    void testConstructorValidation() {
        // Null mode
        assertThrows(IllegalArgumentException.class, () ->
            new InstarOutstarLearning(null, 0.0, 0.0, 1.0)
        );

        // Invalid decay rate
        assertThrows(IllegalArgumentException.class, () ->
            new InstarOutstarLearning(InstarOutstarLearning.LearningMode.INSTAR, -0.1, 0.0, 1.0)
        );
        assertThrows(IllegalArgumentException.class, () ->
            new InstarOutstarLearning(InstarOutstarLearning.LearningMode.INSTAR, 1.5, 0.0, 1.0)
        );

        // Invalid weight bounds
        assertThrows(IllegalArgumentException.class, () ->
            new InstarOutstarLearning(InstarOutstarLearning.LearningMode.INSTAR, 0.0, -0.1, 1.0)
        );
        assertThrows(IllegalArgumentException.class, () ->
            new InstarOutstarLearning(InstarOutstarLearning.LearningMode.INSTAR, 0.0, 0.5, 0.3)
        );

        // Valid parameters
        assertDoesNotThrow(() ->
            new InstarOutstarLearning(InstarOutstarLearning.LearningMode.INSTAR, 0.0, 0.0, 1.0)
        );
    }

    @Test
    void testFactoryMethods() {
        var instar = InstarOutstarLearning.createInstar();
        assertEquals(InstarOutstarLearning.LearningMode.INSTAR, instar.getMode());
        assertEquals(0.0, instar.getDecayRate());

        var outstar = InstarOutstarLearning.createOutstar();
        assertEquals(InstarOutstarLearning.LearningMode.OUTSTAR, outstar.getMode());
        assertEquals(0.0001, outstar.getDecayRate());

        var bidir = InstarOutstarLearning.createBidirectional();
        assertEquals(InstarOutstarLearning.LearningMode.BOTH, bidir.getMode());
        assertEquals(0.0001, bidir.getDecayRate());
    }

    @Test
    void testInstarLearningConvergence() {
        var learning = InstarOutstarLearning.createInstar();

        // Target pattern to learn
        var targetPattern = new DenseVector(new double[]{1.0, 0.0, 1.0, 0.0});
        var categoryActivation = new DenseVector(new double[]{1.0});  // Single active category

        // Initial random weights
        var weights = new WeightMatrix(1, 4);
        weights.set(0, 0, 0.3);
        weights.set(0, 1, 0.7);
        weights.set(0, 2, 0.5);
        weights.set(0, 3, 0.2);

        // Learn the pattern (instar: weights → pattern)
        for (int iter = 0; iter < 50; iter++) {
            weights = learning.update(targetPattern, categoryActivation, weights, 0.5);
        }

        // Weights should converge to target pattern
        assertEquals(1.0, weights.get(0, 0), 0.1);  // Should be close to 1.0
        assertEquals(0.0, weights.get(0, 1), 0.1);  // Should be close to 0.0
        assertEquals(1.0, weights.get(0, 2), 0.1);  // Should be close to 1.0
        assertEquals(0.0, weights.get(0, 3), 0.1);  // Should be close to 0.0
    }

    @Test
    void testOutstarLearningConvergence() {
        var learning = InstarOutstarLearning.createOutstar();

        // Pattern to learn
        var inputPattern = new DenseVector(new double[]{1.0, 0.5, 0.0, 0.5});
        var categoryActivation = new DenseVector(new double[]{1.0, 0.0});  // First category active

        // Initial weights
        var weights = new WeightMatrix(2, 4);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {
                weights.set(i, j, 0.5);
            }
        }

        // Learn the pattern (outstar: weights → pattern for active category)
        for (int iter = 0; iter < 50; iter++) {
            weights = learning.update(inputPattern, categoryActivation, weights, 0.3);
        }

        // First category should learn the pattern
        assertEquals(1.0, weights.get(0, 0), 0.2);
        assertEquals(0.5, weights.get(0, 1), 0.2);
        assertEquals(0.0, weights.get(0, 2), 0.2);
        assertEquals(0.5, weights.get(0, 3), 0.2);

        // Second category should remain unchanged (not active)
        assertEquals(0.5, weights.get(1, 0), 0.1);
        assertEquals(0.5, weights.get(1, 1), 0.1);
    }

    @Test
    void testBidirectionalLearning() {
        var learning = InstarOutstarLearning.createBidirectional();

        var pattern = new DenseVector(new double[]{1.0, 0.0, 1.0});
        var activation = new DenseVector(new double[]{1.0, 0.0});

        var weights = new WeightMatrix(2, 3);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                weights.set(i, j, 0.5);
            }
        }

        // Learn with both instar and outstar
        for (int iter = 0; iter < 30; iter++) {
            weights = learning.update(pattern, activation, weights, 0.4);
        }

        // First neuron should learn the pattern (it's active)
        assertTrue(weights.get(0, 0) > 0.7);  // Should move toward 1.0
        assertTrue(weights.get(0, 1) < 0.3);  // Should move toward 0.0
        assertTrue(weights.get(0, 2) > 0.7);  // Should move toward 1.0

        // Second neuron should remain relatively unchanged (not active)
        assertTrue(Math.abs(weights.get(1, 0) - 0.5) < 0.2);
    }

    @Test
    void testWeightDecay() {
        var learning = new InstarOutstarLearning(
            InstarOutstarLearning.LearningMode.INSTAR,
            0.1,  // High decay rate
            0.0, 1.0
        );

        // No input pattern (all zeros)
        var zeroPattern = new DenseVector(new double[]{0.0, 0.0, 0.0});
        var activation = new DenseVector(new double[]{1.0});

        // Start with non-zero weights
        var weights = new WeightMatrix(1, 3);
        weights.set(0, 0, 0.8);
        weights.set(0, 1, 0.6);
        weights.set(0, 2, 0.4);

        // Apply learning multiple times with zero input
        for (int iter = 0; iter < 20; iter++) {
            weights = learning.update(zeroPattern, activation, weights, 0.5);
        }

        // Weights should decay toward zero
        assertTrue(weights.get(0, 0) < 0.4);  // Decayed from 0.8
        assertTrue(weights.get(0, 1) < 0.3);  // Decayed from 0.6
        assertTrue(weights.get(0, 2) < 0.2);  // Decayed from 0.4
    }

    @Test
    void testWeightBounds() {
        var learning = InstarOutstarLearning.createInstar();

        // Pattern with values that would push weights out of bounds
        var largePattern = new DenseVector(new double[]{1.0, 1.0, 1.0});
        var activation = new DenseVector(new double[]{1.0});

        // Start near max bound
        var weights = new WeightMatrix(1, 3);
        weights.set(0, 0, 0.95);
        weights.set(0, 1, 0.95);
        weights.set(0, 2, 0.95);

        // Apply aggressive learning
        for (int iter = 0; iter < 10; iter++) {
            weights = learning.update(largePattern, activation, weights, 0.9);
        }

        // Weights should be clipped to [0, 1]
        assertTrue(weights.get(0, 0) <= 1.0);
        assertTrue(weights.get(0, 1) <= 1.0);
        assertTrue(weights.get(0, 2) <= 1.0);
        assertTrue(weights.get(0, 0) >= 0.0);
    }

    @Test
    void testMultipleCategoryLearning() {
        var learning = InstarOutstarLearning.createInstar();

        // Two different patterns for two categories
        var pattern1 = new DenseVector(new double[]{1.0, 0.0, 0.0});
        var pattern2 = new DenseVector(new double[]{0.0, 1.0, 0.0});

        var weights = new WeightMatrix(2, 3);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                weights.set(i, j, 0.5);
            }
        }

        // Train first category with pattern1
        for (int iter = 0; iter < 20; iter++) {
            var activation1 = new DenseVector(new double[]{1.0, 0.0});
            weights = learning.update(pattern1, activation1, weights, 0.5);
        }

        // Train second category with pattern2
        for (int iter = 0; iter < 20; iter++) {
            var activation2 = new DenseVector(new double[]{0.0, 1.0});
            weights = learning.update(pattern2, activation2, weights, 0.5);
        }

        // Category 1 should learn pattern1
        assertTrue(weights.get(0, 0) > 0.7);
        assertTrue(weights.get(0, 1) < 0.3);

        // Category 2 should learn pattern2
        assertTrue(weights.get(1, 0) < 0.3);
        assertTrue(weights.get(1, 1) > 0.7);
    }

    @Test
    void testDimensionValidation() {
        var learning = InstarOutstarLearning.createInstar();

        var pattern = new DenseVector(new double[]{1.0, 0.0});
        var activation = new DenseVector(new double[]{1.0});
        var wrongWeights = new WeightMatrix(1, 3);  // Wrong size

        assertThrows(IllegalArgumentException.class, () ->
            learning.update(pattern, activation, wrongWeights, 0.5)
        );
    }

    @Test
    void testLearningRateValidation() {
        var learning = InstarOutstarLearning.createInstar();

        var pattern = new DenseVector(new double[]{1.0, 0.0});
        var activation = new DenseVector(new double[]{1.0});
        var weights = new WeightMatrix(1, 2);

        // Invalid learning rates
        assertThrows(IllegalArgumentException.class, () ->
            learning.update(pattern, activation, weights, -0.1)
        );
        assertThrows(IllegalArgumentException.class, () ->
            learning.update(pattern, activation, weights, 1.5)
        );

        // Valid learning rates
        assertDoesNotThrow(() ->
            learning.update(pattern, activation, weights, 0.0)
        );
        assertDoesNotThrow(() ->
            learning.update(pattern, activation, weights, 1.0)
        );
    }

    @Test
    void testGetName() {
        assertEquals("InstarOutstar[INSTAR]",
            InstarOutstarLearning.createInstar().getName());
        assertEquals("InstarOutstar[OUTSTAR]",
            InstarOutstarLearning.createOutstar().getName());
        assertEquals("InstarOutstar[BOTH]",
            InstarOutstarLearning.createBidirectional().getName());
    }

    @Test
    void testRequiresNormalization() {
        assertFalse(InstarOutstarLearning.createInstar().requiresNormalization());
        assertFalse(InstarOutstarLearning.createOutstar().requiresNormalization());
        assertFalse(InstarOutstarLearning.createBidirectional().requiresNormalization());
    }

    @Test
    void testRecommendedLearningRates() {
        var instarRange = InstarOutstarLearning.createInstar().getRecommendedLearningRateRange();
        assertEquals(0.1, instarRange[0], 0.01);
        assertEquals(0.9, instarRange[1], 0.01);

        var outstarRange = InstarOutstarLearning.createOutstar().getRecommendedLearningRateRange();
        assertEquals(0.05, outstarRange[0], 0.01);
        assertEquals(0.5, outstarRange[1], 0.01);
    }

    @Test
    void testToString() {
        var learning = new InstarOutstarLearning(
            InstarOutstarLearning.LearningMode.INSTAR,
            0.001, 0.0, 1.0
        );
        var str = learning.toString();
        assertTrue(str.contains("InstarOutstarLearning"));
        assertTrue(str.contains("INSTAR"));
        assertTrue(str.contains("0.001"));
    }
}
