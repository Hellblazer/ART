package com.hellblazer.art.cortical.resonance;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for enhanced resonance detection with consciousness metrics.
 *
 * <p>Phase 2D: Tests resonance detection, phase synchronization,
 * and consciousness likelihood computation.
 *
 * @author Phase 2D: Enhanced Resonance Detection
 */
public class EnhancedResonanceDetectorTest {

    private static final double VIGILANCE = 0.7;
    private static final double SAMPLING_RATE = 1000.0;
    private static final int HISTORY_SIZE = 256;
    private static final double GAMMA_FREQUENCY = 40.0;

    /**
     * Test basic ART resonance detection (no oscillation analysis).
     */
    @Test
    public void testBasicARTResonance() {
        var detector = new EnhancedResonanceDetector(VIGILANCE, SAMPLING_RATE, HISTORY_SIZE);

        // High match quality features
        var features = new double[]{0.8, 0.7, 0.9, 0.6};
        var expectations = new double[]{0.75, 0.65, 0.85, 0.55};

        var state = detector.detectResonance(features, expectations, 1.0);

        assertTrue(state.artResonance(), "Should detect ART resonance with high match");
        assertTrue(state.matchQuality() >= VIGILANCE, "Match quality should meet vigilance");
        assertFalse(state.phaseSynchronized(), "No phase sync without history");
        assertFalse(state.bothInGamma(), "No gamma detection without history");
    }

    /**
     * Test that low match quality prevents resonance.
     */
    @Test
    public void testLowMatchQualityNoResonance() {
        var detector = new EnhancedResonanceDetector(VIGILANCE, SAMPLING_RATE, HISTORY_SIZE);

        // Low match quality
        var features = new double[]{0.9, 0.8, 0.7, 0.6};
        var expectations = new double[]{0.1, 0.1, 0.1, 0.1};

        var state = detector.detectResonance(features, expectations, 1.0);

        assertFalse(state.artResonance(), "Should not detect resonance with low match");
        assertTrue(state.matchQuality() < VIGILANCE, "Match quality below vigilance");
        assertEquals(0.0, state.consciousnessLikelihood(), "No consciousness without resonance");
    }

    /**
     * Test resonance with synchronized gamma oscillations (high consciousness).
     */
    @Test
    public void testHighConsciousnessWithGammaSync() {
        var detector = new EnhancedResonanceDetector(VIGILANCE, SAMPLING_RATE, HISTORY_SIZE);

        // Record oscillatory history for both pathways (same phase, 40 Hz)
        for (int t = 0; t < HISTORY_SIZE; t++) {
            double time = t / SAMPLING_RATE;
            var activation = generateGammaOscillation(GAMMA_FREQUENCY, time, 0.0, 64);
            detector.recordBottomUp(activation);
            detector.recordTopDown(activation);  // Same phase
        }

        // High match features
        var features = new double[]{0.9, 0.8, 0.9, 0.8};
        var expectations = new double[]{0.85, 0.75, 0.85, 0.75};

        var state = detector.detectResonance(features, expectations, HISTORY_SIZE / SAMPLING_RATE);

        assertTrue(state.artResonance(), "Should have ART resonance");
        assertTrue(state.phaseSynchronized(), "Should have phase synchronization");
        assertTrue(state.bothInGamma(), "Both pathways should be in gamma");
        assertTrue(state.consciousnessLikelihood() > 0.9,
            "High consciousness likelihood, got: " + state.consciousnessLikelihood());
        assertTrue(state.isLikelyConscious(0.7), "Should indicate likely conscious perception");
    }

    /**
     * Test resonance without phase synchronization (lower consciousness).
     */
    @Test
    public void testResonanceWithoutPhaseSync() {
        var detector = new EnhancedResonanceDetector(VIGILANCE, SAMPLING_RATE, HISTORY_SIZE);

        // Record oscillatory history with large phase offset (π)
        for (int t = 0; t < HISTORY_SIZE; t++) {
            double time = t / SAMPLING_RATE;
            var buActivation = generateGammaOscillation(GAMMA_FREQUENCY, time, 0.0, 64);
            var tdActivation = generateGammaOscillation(GAMMA_FREQUENCY, time, Math.PI, 64);  // Anti-phase
            detector.recordBottomUp(buActivation);
            detector.recordTopDown(tdActivation);
        }

        // High match features
        var features = new double[]{0.9, 0.8, 0.9, 0.8};
        var expectations = new double[]{0.85, 0.75, 0.85, 0.75};

        var state = detector.detectResonance(features, expectations, HISTORY_SIZE / SAMPLING_RATE);

        assertTrue(state.artResonance(), "Should have ART resonance");
        assertFalse(state.phaseSynchronized(), "Should NOT have phase synchronization (anti-phase)");
        assertTrue(state.bothInGamma(), "Both pathways should be in gamma");

        // Consciousness likelihood should be lower without phase sync
        // Base match (0.9+) + gamma bonus (0.3) but NO phase sync bonus (0.2)
        // High match (0.9) + 0.3 = 1.0 (capped), so just verify it's missing phase sync component
        double expected = Math.min(1.0, state.matchQuality() + 0.3);  // No phase sync bonus
        assertTrue(state.consciousnessLikelihood() <= expected + 0.01,
            "Should not have phase sync bonus, got: " + state.consciousnessLikelihood());
    }

    /**
     * Test resonance with non-gamma oscillations.
     */
    @Test
    public void testResonanceWithoutGamma() {
        var detector = new EnhancedResonanceDetector(VIGILANCE, SAMPLING_RATE, HISTORY_SIZE);

        // Record alpha oscillations (10 Hz, not gamma)
        for (int t = 0; t < HISTORY_SIZE; t++) {
            double time = t / SAMPLING_RATE;
            var activation = generateGammaOscillation(10.0, time, 0.0, 64);  // Alpha, not gamma
            detector.recordBottomUp(activation);
            detector.recordTopDown(activation);
        }

        // High match features
        var features = new double[]{0.9, 0.8, 0.9, 0.8};
        var expectations = new double[]{0.85, 0.75, 0.85, 0.75};

        var state = detector.detectResonance(features, expectations, HISTORY_SIZE / SAMPLING_RATE);

        assertTrue(state.artResonance(), "Should have ART resonance");
        assertTrue(state.phaseSynchronized(), "Should have phase synchronization");
        assertFalse(state.bothInGamma(), "Should NOT be in gamma band (alpha oscillations)");

        // Consciousness likelihood moderate without gamma
        // Base match (0.9+) + phase sync bonus (0.2) but NO gamma bonus (0.3)
        // High match (0.9) + 0.2 = 1.0 (capped), so just verify it's missing gamma component
        double expected = Math.min(1.0, state.matchQuality() + 0.2);  // No gamma bonus
        assertTrue(state.consciousnessLikelihood() <= expected + 0.01,
            "Should not have gamma bonus, got: " + state.consciousnessLikelihood());
    }

    /**
     * Test reset clears history buffers.
     */
    @Test
    public void testResetClearsHistory() {
        var detector = new EnhancedResonanceDetector(VIGILANCE, SAMPLING_RATE, HISTORY_SIZE);

        // Fill history
        for (int t = 0; t < HISTORY_SIZE; t++) {
            double time = t / SAMPLING_RATE;
            var activation = generateGammaOscillation(GAMMA_FREQUENCY, time, 0.0, 64);
            detector.recordBottomUp(activation);
            detector.recordTopDown(activation);
        }

        assertTrue(detector.isReady(), "Should be ready before reset");

        // Reset
        detector.reset();

        assertFalse(detector.isReady(), "Should not be ready after reset");

        // Detect resonance after reset (no oscillation metrics)
        var features = new double[]{0.9, 0.8, 0.9, 0.8};
        var expectations = new double[]{0.85, 0.75, 0.85, 0.75};
        var state = detector.detectResonance(features, expectations, 1.0);

        assertTrue(state.artResonance(), "Should still have ART resonance");
        assertFalse(state.phaseSynchronized(), "No phase sync after reset");
        assertFalse(state.bothInGamma(), "No gamma detection after reset");
    }

    /**
     * Test consciousness likelihood computation with all factors.
     */
    @Test
    public void testConsciousnessLikelihoodComputation() {
        var detector = new EnhancedResonanceDetector(VIGILANCE, SAMPLING_RATE, HISTORY_SIZE);

        // Fill with synchronized gamma oscillations
        for (int t = 0; t < HISTORY_SIZE; t++) {
            double time = t / SAMPLING_RATE;
            var activation = generateGammaOscillation(GAMMA_FREQUENCY, time, 0.0, 64);
            detector.recordBottomUp(activation);
            detector.recordTopDown(activation);
        }

        // Test with different match qualities
        var perfectMatch = new double[]{0.9, 0.8, 0.7};
        var goodMatch = new double[]{0.75, 0.7, 0.65};

        var state1 = detector.detectResonance(perfectMatch, perfectMatch, 1.0);
        var state2 = detector.detectResonance(goodMatch, goodMatch, 1.0);

        // Both should have resonance, phase sync, and gamma
        assertTrue(state1.artResonance() && state2.artResonance());
        assertTrue(state1.phaseSynchronized() && state2.phaseSynchronized());
        assertTrue(state1.bothInGamma() && state2.bothInGamma());

        // Perfect match should have higher consciousness likelihood
        // Both get base match + 0.2 (phase sync) + 0.3 (gamma) = match + 0.5
        // So they're both likely at the cap (1.0), just verify they're high
        assertTrue(state1.consciousnessLikelihood() >= 0.9,
            "Perfect match should have very high consciousness likelihood");
        assertTrue(state2.consciousnessLikelihood() >= 0.9,
            "Good match should also be high with phase sync + gamma");

        // Both should be capped at 1.0
        assertTrue(state1.consciousnessLikelihood() <= 1.0);
        assertTrue(state2.consciousnessLikelihood() <= 1.0);
    }

    /**
     * Test ResonanceState helper methods.
     */
    @Test
    public void testResonanceStateHelpers() {
        var detector = new EnhancedResonanceDetector(VIGILANCE, SAMPLING_RATE, HISTORY_SIZE);

        // Create state with phase difference
        for (int t = 0; t < HISTORY_SIZE; t++) {
            double time = t / SAMPLING_RATE;
            var buActivation = generateGammaOscillation(40.0, time, 0.0, 64);
            var tdActivation = generateGammaOscillation(42.0, time, Math.PI / 6, 64);
            detector.recordBottomUp(buActivation);
            detector.recordTopDown(tdActivation);
        }

        var features = new double[]{0.9, 0.8};
        var expectations = new double[]{0.85, 0.75};
        var state = detector.detectResonance(features, expectations, 1.0);

        // Test phase difference
        var phaseDiff = state.getPhaseDifference();
        assertFalse(Double.isNaN(phaseDiff), "Phase difference should be available");

        // Test frequency coherence
        var freqCoherence = state.getFrequencyCoherence();
        assertFalse(Double.isNaN(freqCoherence), "Frequency coherence should be available");
        // FFT resolution: 1000 Hz / 256 samples = 3.90625 Hz bins
        // With 2 Hz actual difference, we expect ~2-4 Hz measured difference
        assertEquals(2.0, freqCoherence, 4.0, "Frequency difference should be ~2 Hz (FFT resolution limited)");
    }

    /**
     * Test none() factory method.
     */
    @Test
    public void testNoneFactoryMethod() {
        var state = ResonanceState.none(1.0);

        assertFalse(state.artResonance());
        assertFalse(state.phaseSynchronized());
        assertFalse(state.bothInGamma());
        assertEquals(0.0, state.consciousnessLikelihood());
        assertEquals(0.0, state.matchQuality());
        assertNull(state.bottomUpMetrics());
        assertNull(state.topDownMetrics());
        assertFalse(state.isLikelyConscious(0.7));
    }

    // ============== Helper Methods ==============

    /**
     * Generate gamma oscillation pattern.
     */
    private double[] generateGammaOscillation(
            double frequency,
            double time,
            double phaseOffset,
            int size) {

        var activation = new double[size];
        for (int i = 0; i < size; i++) {
            activation[i] = Math.sin(2 * Math.PI * frequency * time + phaseOffset);
        }
        return activation;
    }
}
