package com.hellblazer.art.cortical.learning;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.cortical.resonance.ResonanceState;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for LearningContext.
 *
 * <p>Phase 3A: Core Learning Infrastructure
 *
 * @author Phase 3A Tests
 */
public class LearningContextTest {

    /**
     * Test basic context creation.
     */
    @Test
    public void testBasicContextCreation() {
        var pre = new DenseVector(new double[]{1.0, 0.5});
        var post = new DenseVector(new double[]{0.8, 0.6});

        var context = new LearningContext(pre, post, null, 0.7, 1.0);

        assertSame(pre, context.preActivation());
        assertSame(post, context.postActivation());
        assertNull(context.resonanceState());
        assertEquals(0.7, context.attentionStrength());
        assertEquals(1.0, context.timestamp());
    }

    /**
     * Test validation of attention strength.
     */
    @Test
    public void testAttentionStrengthValidation() {
        var pre = new DenseVector(new double[]{1.0});
        var post = new DenseVector(new double[]{1.0});

        assertThrows(IllegalArgumentException.class, () ->
            new LearningContext(pre, post, null, -0.1, 1.0)  // Negative attention
        );

        assertThrows(IllegalArgumentException.class, () ->
            new LearningContext(pre, post, null, 1.1, 1.0)  // Attention > 1
        );
    }

    /**
     * Test validation of timestamp.
     */
    @Test
    public void testTimestampValidation() {
        var pre = new DenseVector(new double[]{1.0});
        var post = new DenseVector(new double[]{1.0});

        assertThrows(IllegalArgumentException.class, () ->
            new LearningContext(pre, post, null, 0.5, -1.0)  // Negative timestamp
        );
    }

    /**
     * Test validation of patterns.
     */
    @Test
    public void testPatternValidation() {
        var post = new DenseVector(new double[]{1.0});

        assertThrows(IllegalArgumentException.class, () ->
            new LearningContext(null, post, null, 0.5, 1.0)  // Null pre
        );

        var pre = new DenseVector(new double[]{1.0});

        assertThrows(IllegalArgumentException.class, () ->
            new LearningContext(pre, null, null, 0.5, 1.0)  // Null post
        );
    }

    /**
     * Test shouldLearn() with resonance and attention.
     */
    @Test
    public void testShouldLearnWithResonance() {
        var pre = new DenseVector(new double[]{1.0});
        var post = new DenseVector(new double[]{1.0});

        // High consciousness + high attention = learn
        var resonanceState = new ResonanceState(
            true, true, true,
            0.9,  // High consciousness
            null, null,
            0.9, 1.0
        );

        var context = new LearningContext(pre, post, resonanceState, 0.8, 1.0);
        assertTrue(context.shouldLearn(0.7, 0.3), "Should learn with high consciousness and attention");

        // Low consciousness + high attention = no learn
        resonanceState = new ResonanceState(
            false, false, false,
            0.5,  // Low consciousness
            null, null,
            0.5, 1.0
        );

        context = new LearningContext(pre, post, resonanceState, 0.8, 1.0);
        assertFalse(context.shouldLearn(0.7, 0.3), "Should not learn with low consciousness");

        // High consciousness + low attention = no learn
        resonanceState = new ResonanceState(
            true, true, true,
            0.9,  // High consciousness
            null, null,
            0.9, 1.0
        );

        context = new LearningContext(pre, post, resonanceState, 0.2, 1.0);
        assertFalse(context.shouldLearn(0.7, 0.3), "Should not learn with low attention");
    }

    /**
     * Test shouldLearn() without resonance detection.
     */
    @Test
    public void testShouldLearnWithoutResonance() {
        var pre = new DenseVector(new double[]{1.0});
        var post = new DenseVector(new double[]{1.0});

        // No resonance detection, only attention matters
        var context = new LearningContext(pre, post, null, 0.8, 1.0);
        assertTrue(context.shouldLearn(0.7, 0.3), "Should learn with high attention (no resonance gating)");

        context = new LearningContext(pre, post, null, 0.2, 1.0);
        assertFalse(context.shouldLearn(0.7, 0.3), "Should not learn with low attention");
    }

    /**
     * Test getLearningRateModulation() with resonance.
     */
    @Test
    public void testLearningRateModulationWithResonance() {
        var pre = new DenseVector(new double[]{1.0});
        var post = new DenseVector(new double[]{1.0});

        var resonanceState = new ResonanceState(
            true, true, true,
            0.8,  // Consciousness
            null, null,
            0.8, 1.0
        );

        var context = new LearningContext(pre, post, resonanceState, 0.5, 1.0);  // Attention

        // Modulation = consciousness × attention = 0.8 × 0.5 = 0.4
        assertEquals(0.4, context.getLearningRateModulation(), 0.001);
    }

    /**
     * Test getLearningRateModulation() without resonance.
     */
    @Test
    public void testLearningRateModulationWithoutResonance() {
        var pre = new DenseVector(new double[]{1.0});
        var post = new DenseVector(new double[]{1.0});

        var context = new LearningContext(pre, post, null, 0.5, 1.0);

        // Modulation = 1.0 × attention = 1.0 × 0.5 = 0.5
        assertEquals(0.5, context.getLearningRateModulation(), 0.001);
    }

    /**
     * Test hasResonanceDetection().
     */
    @Test
    public void testHasResonanceDetection() {
        var pre = new DenseVector(new double[]{1.0});
        var post = new DenseVector(new double[]{1.0});

        var resonanceState = ResonanceState.none(1.0);
        var context = new LearningContext(pre, post, resonanceState, 0.5, 1.0);
        assertTrue(context.hasResonanceDetection());

        context = new LearningContext(pre, post, null, 0.5, 1.0);
        assertFalse(context.hasResonanceDetection());
    }

    /**
     * Test hasResonance().
     */
    @Test
    public void testHasResonance() {
        var pre = new DenseVector(new double[]{1.0});
        var post = new DenseVector(new double[]{1.0});

        var resonanceState = new ResonanceState(
            true, false, false,
            0.8, null, null, 0.8, 1.0
        );

        var context = new LearningContext(pre, post, resonanceState, 0.5, 1.0);
        assertTrue(context.hasResonance());

        resonanceState = new ResonanceState(
            false, false, false,
            0.3, null, null, 0.3, 1.0
        );

        context = new LearningContext(pre, post, resonanceState, 0.5, 1.0);
        assertFalse(context.hasResonance());

        context = new LearningContext(pre, post, null, 0.5, 1.0);
        assertFalse(context.hasResonance());
    }

    /**
     * Test isLikelyConscious().
     */
    @Test
    public void testIsLikelyConscious() {
        var pre = new DenseVector(new double[]{1.0});
        var post = new DenseVector(new double[]{1.0});

        var resonanceState = new ResonanceState(
            true, true, true,
            0.9, null, null, 0.9, 1.0
        );

        var context = new LearningContext(pre, post, resonanceState, 0.5, 1.0);
        assertTrue(context.isLikelyConscious(0.7));
        assertFalse(context.isLikelyConscious(0.95));

        context = new LearningContext(pre, post, null, 0.5, 1.0);
        assertFalse(context.isLikelyConscious(0.7));
    }

    /**
     * Test getConsciousnessLikelihood().
     */
    @Test
    public void testGetConsciousnessLikelihood() {
        var pre = new DenseVector(new double[]{1.0});
        var post = new DenseVector(new double[]{1.0});

        var resonanceState = new ResonanceState(
            true, true, true,
            0.85, null, null, 0.85, 1.0
        );

        var context = new LearningContext(pre, post, resonanceState, 0.5, 1.0);
        assertEquals(0.85, context.getConsciousnessLikelihood(), 0.001);

        context = new LearningContext(pre, post, null, 0.5, 1.0);
        assertEquals(0.0, context.getConsciousnessLikelihood(), 0.001);
    }

    /**
     * Test withoutResonance() factory method.
     */
    @Test
    public void testWithoutResonanceFactory() {
        var pre = new DenseVector(new double[]{1.0});
        var post = new DenseVector(new double[]{1.0});

        var context = LearningContext.withoutResonance(pre, post, 0.7, 2.0);

        assertSame(pre, context.preActivation());
        assertSame(post, context.postActivation());
        assertNull(context.resonanceState());
        assertEquals(0.7, context.attentionStrength());
        assertEquals(2.0, context.timestamp());
        assertFalse(context.hasResonanceDetection());
    }

    /**
     * Test alwaysLearn() factory method.
     */
    @Test
    public void testAlwaysLearnFactory() {
        var pre = new DenseVector(new double[]{1.0});
        var post = new DenseVector(new double[]{1.0});

        var context = LearningContext.alwaysLearn(pre, post, 3.0);

        assertSame(pre, context.preActivation());
        assertSame(post, context.postActivation());
        assertNull(context.resonanceState());
        assertEquals(1.0, context.attentionStrength());  // Full attention
        assertEquals(3.0, context.timestamp());
        assertTrue(context.shouldLearn(0.0, 0.0));  // Always learns
    }

    /**
     * Test toString().
     */
    @Test
    public void testToString() {
        var pre = new DenseVector(new double[]{1.0});
        var post = new DenseVector(new double[]{1.0});

        var resonanceState = new ResonanceState(
            true, true, true,
            0.9, null, null, 0.9, 1.0
        );

        var context = new LearningContext(pre, post, resonanceState, 0.8, 5.5);
        var str = context.toString();

        assertTrue(str.contains("LearningContext"));
        assertTrue(str.contains("resonance=YES"));
        assertTrue(str.contains("0.9"));  // Consciousness
        assertTrue(str.contains("0.8"));  // Attention
        assertTrue(str.contains("5.5"));  // Timestamp
    }
}
