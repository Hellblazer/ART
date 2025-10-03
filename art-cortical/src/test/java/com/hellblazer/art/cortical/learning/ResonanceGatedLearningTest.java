package com.hellblazer.art.cortical.learning;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.cortical.layers.WeightMatrix;
import com.hellblazer.art.cortical.resonance.ResonanceState;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for resonance-gated learning wrapper.
 *
 * <p>Verifies that learning is correctly gated by consciousness likelihood
 * and modulated by resonance state from Phase 2D.
 *
 * @author Phase 3B: Resonance-Gated Learning Tests
 */
class ResonanceGatedLearningTest {

    private HebbianLearning baseLearning;
    private ResonanceGatedLearning gatedLearning;

    @BeforeEach
    void setUp() {
        baseLearning = new HebbianLearning(0.0, 0.0, 1.0);
        gatedLearning = new ResonanceGatedLearning(baseLearning, 0.7);
    }

    @Test
    void testConstructorValidation() {
        // Null learning rule
        assertThrows(IllegalArgumentException.class, () ->
            new ResonanceGatedLearning(null, 0.7)
        );

        // Invalid threshold (negative)
        assertThrows(IllegalArgumentException.class, () ->
            new ResonanceGatedLearning(baseLearning, -0.1)
        );

        // Invalid threshold (> 1.0)
        assertThrows(IllegalArgumentException.class, () ->
            new ResonanceGatedLearning(baseLearning, 1.5)
        );

        // Valid thresholds
        assertDoesNotThrow(() -> new ResonanceGatedLearning(baseLearning, 0.0));
        assertDoesNotThrow(() -> new ResonanceGatedLearning(baseLearning, 1.0));
        assertDoesNotThrow(() -> new ResonanceGatedLearning(baseLearning, 0.5));
    }

    @Test
    void testGatingBelowThreshold() {
        // Create learning context with low consciousness
        var resonanceState = new ResonanceState(
            false,           // artResonance
            false,           // phaseSynchronized
            false,           // bothInGamma
            0.5,             // consciousnessLikelihood (below 0.7 threshold)
            null,            // bottomUpMetrics
            null,            // topDownMetrics
            0.5,             // matchQuality
            0.0              // timestamp
        );

        var context = new LearningContext(
            new DenseVector(new double[]{1.0, 0.0}),  // pre
            new DenseVector(new double[]{0.5, 0.5}),  // post
            resonanceState,
            1.0,  // attention (high)
            0.0   // timestamp
        );

        var currentWeights = new WeightMatrix(2, 2);
        currentWeights.set(0, 0, 0.5);
        currentWeights.set(1, 1, 0.5);

        // Learning should be blocked (consciousness = 0.5 < 0.7)
        var newWeights = gatedLearning.updateWithContext(context, currentWeights, 0.1);

        // Weights should be unchanged
        assertSame(currentWeights, newWeights);
        assertEquals(0.5, newWeights.get(0, 0), 1e-10);
        assertEquals(0.5, newWeights.get(1, 1), 1e-10);
    }

    @Test
    void testGatingAboveThreshold() {
        // Create learning context with high consciousness
        var resonanceState = new ResonanceState(
            true,            // artResonance
            true,            // phaseSynchronized
            true,            // bothInGamma
            0.9,             // consciousnessLikelihood (above 0.7 threshold)
            null,            // bottomUpMetrics
            null,            // topDownMetrics
            0.9,             // matchQuality
            0.0              // timestamp
        );

        var context = new LearningContext(
            new DenseVector(new double[]{1.0, 0.0}),  // pre
            new DenseVector(new double[]{1.0, 0.0}),  // post
            resonanceState,
            1.0,  // attention
            0.0   // timestamp
        );

        var currentWeights = new WeightMatrix(2, 2);
        currentWeights.set(0, 0, 0.5);

        // Learning should occur (consciousness = 0.9 > 0.7)
        var newWeights = gatedLearning.updateWithContext(context, currentWeights, 0.1);

        // Weights should change
        assertNotSame(currentWeights, newWeights);
        // Weight should increase due to Hebbian rule (pre=1.0, post=1.0)
        assertTrue(newWeights.get(0, 0) > 0.5);
    }

    @Test
    void testConsciousnessModulation() {
        // Test that learning rate is scaled by consciousness likelihood
        var lowConsciousness = new ResonanceState(
            true, true, true, 0.7, null, null, 0.7, 0.0  // consciousness = 0.7
        );
        var highConsciousness = new ResonanceState(
            true, true, true, 0.95, null, null, 0.95, 0.0  // consciousness = 0.95
        );

        var preActivation = new DenseVector(new double[]{1.0, 0.0});
        var postActivation = new DenseVector(new double[]{1.0, 0.0});

        // Low consciousness context
        var contextLow = new LearningContext(
            preActivation, postActivation, lowConsciousness, 1.0, 0.0
        );

        var weightsLow = new WeightMatrix(2, 2);
        weightsLow.set(0, 0, 0.5);
        var newWeightsLow = gatedLearning.updateWithContext(contextLow, weightsLow, 0.1);

        // High consciousness context
        var contextHigh = new LearningContext(
            preActivation, postActivation, highConsciousness, 1.0, 0.0
        );

        var weightsHigh = new WeightMatrix(2, 2);
        weightsHigh.set(0, 0, 0.5);
        var newWeightsHigh = gatedLearning.updateWithContext(contextHigh, weightsHigh, 0.1);

        // Higher consciousness should result in larger weight change
        double changeLow = Math.abs(newWeightsLow.get(0, 0) - 0.5);
        double changeHigh = Math.abs(newWeightsHigh.get(0, 0) - 0.5);
        assertTrue(changeHigh > changeLow);
    }

    @Test
    void testNoResonanceDetection() {
        // When resonance detection is disabled, fall back to base learning
        var context = new LearningContext(
            new DenseVector(new double[]{1.0, 0.0}),
            new DenseVector(new double[]{1.0, 0.0}),
            null,  // No resonance state
            1.0,
            0.0
        );

        var currentWeights = new WeightMatrix(2, 2);
        currentWeights.set(0, 0, 0.5);

        // Should apply base learning (no gating)
        var newWeights = gatedLearning.updateWithContext(context, currentWeights, 0.1);

        assertNotSame(currentWeights, newWeights);
        assertTrue(newWeights.get(0, 0) > 0.5);  // Learning occurred
    }

    @Test
    void testThresholdAtExactValue() {
        // Test behavior at exact threshold value
        var resonanceState = new ResonanceState(
            true, true, true, 0.7, null, null, 0.7, 0.0  // consciousness exactly 0.7
        );

        var context = new LearningContext(
            new DenseVector(new double[]{1.0, 0.0}),
            new DenseVector(new double[]{1.0, 0.0}),
            resonanceState,
            1.0,
            0.0
        );

        var currentWeights = new WeightMatrix(2, 2);
        currentWeights.set(0, 0, 0.5);

        // At exact threshold, learning should occur
        var newWeights = gatedLearning.updateWithContext(context, currentWeights, 0.1);
        assertNotSame(currentWeights, newWeights);
        assertTrue(newWeights.get(0, 0) > 0.5);
    }

    @Test
    void testFallbackUpdate() {
        // Test the fallback update() method (without context)
        Pattern preActivation = new DenseVector(new double[]{1.0, 0.0});
        Pattern postActivation = new DenseVector(new double[]{1.0, 0.0});
        var currentWeights = new WeightMatrix(2, 2);
        currentWeights.set(0, 0, 0.5);

        // Should delegate to base learning
        var newWeights = gatedLearning.update(
            preActivation, postActivation, currentWeights, 0.1
        );

        assertNotSame(currentWeights, newWeights);
        assertTrue(newWeights.get(0, 0) > 0.5);
    }

    @Test
    void testGetters() {
        assertEquals(baseLearning, gatedLearning.getBaseLearning());
        assertEquals(0.7, gatedLearning.getResonanceThreshold(), 1e-10);
    }

    @Test
    void testName() {
        assertTrue(gatedLearning.getName().contains("ResonanceGated"));
        assertTrue(gatedLearning.getName().contains("Hebbian"));
    }

    @Test
    void testRequiresNormalization() {
        // Should delegate to base learning
        assertEquals(
            baseLearning.requiresNormalization(),
            gatedLearning.requiresNormalization()
        );
    }

    @Test
    void testLearningRateRange() {
        var baseRange = baseLearning.getRecommendedLearningRateRange();
        var gatedRange = gatedLearning.getRecommendedLearningRateRange();

        // Gated learning should recommend higher rates (50% increase)
        assertEquals(baseRange[0] * 1.5, gatedRange[0], 1e-10);
        assertEquals(baseRange[1] * 1.5, gatedRange[1], 1e-10);
    }

    @Test
    void testToString() {
        var str = gatedLearning.toString();
        assertTrue(str.contains("ResonanceGatedLearning"));
        assertTrue(str.contains("Hebbian"));
        assertTrue(str.contains("0.7"));
    }

    @Test
    void testDifferentThresholds() {
        // Test with different threshold values
        var lowThreshold = new ResonanceGatedLearning(baseLearning, 0.3);
        var highThreshold = new ResonanceGatedLearning(baseLearning, 0.9);

        var resonanceState = new ResonanceState(
            true, true, false, 0.6, null, null, 0.6, 0.0
        );

        var context = new LearningContext(
            new DenseVector(new double[]{1.0, 0.0}),
            new DenseVector(new double[]{1.0, 0.0}),
            resonanceState,
            1.0,
            0.0
        );

        var weights = new WeightMatrix(2, 2);
        weights.set(0, 0, 0.5);

        // Low threshold (0.3) should allow learning
        var newWeightsLow = lowThreshold.updateWithContext(context, weights, 0.1);
        assertTrue(newWeightsLow.get(0, 0) > 0.5);

        // High threshold (0.9) should block learning
        var weightsHigh = new WeightMatrix(2, 2);
        weightsHigh.set(0, 0, 0.5);
        var newWeightsHigh = highThreshold.updateWithContext(context, weightsHigh, 0.1);
        assertEquals(0.5, newWeightsHigh.get(0, 0), 1e-10);
    }

    @Test
    void testMultipleUpdates() {
        // Test sequence of updates with varying consciousness
        var weights = new WeightMatrix(2, 2);
        weights.set(0, 0, 0.5);

        var preActivation = new DenseVector(new double[]{1.0, 0.0});
        var postActivation = new DenseVector(new double[]{1.0, 0.0});

        // Update 1: Below threshold
        var resonance1 = new ResonanceState(true, false, false, 0.5, null, null, 0.5, 0.0);
        var context1 = new LearningContext(preActivation, postActivation, resonance1, 1.0, 0.0);
        weights = gatedLearning.updateWithContext(context1, weights, 0.1);
        double weight1 = weights.get(0, 0);
        assertEquals(0.5, weight1, 1e-10);  // No change

        // Update 2: Above threshold
        var resonance2 = new ResonanceState(true, true, true, 0.8, null, null, 0.8, 0.0);
        var context2 = new LearningContext(preActivation, postActivation, resonance2, 1.0, 0.0);
        weights = gatedLearning.updateWithContext(context2, weights, 0.1);
        double weight2 = weights.get(0, 0);
        assertTrue(weight2 > weight1);  // Learning occurred

        // Update 3: Below threshold again
        var context3 = new LearningContext(preActivation, postActivation, resonance1, 1.0, 0.0);
        weights = gatedLearning.updateWithContext(context3, weights, 0.1);
        double weight3 = weights.get(0, 0);
        assertEquals(weight2, weight3, 1e-10);  // No change
    }
}
