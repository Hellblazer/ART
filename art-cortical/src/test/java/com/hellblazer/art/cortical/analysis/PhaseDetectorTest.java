package com.hellblazer.art.cortical.analysis;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for phase detection and synchronization analysis.
 *
 * <p>Phase 2A: Mathematical Foundation - Tests phase computation for
 * oscillatory signals and synchronization detection.
 *
 * <p>Based on ART Cortical Enhancement Plan Section 3.2 (Phase 2A).
 *
 * @author Phase 2: Oscillatory Dynamics Integration
 */
public class PhaseDetectorTest {

    private static final double PHASE_EPSILON = 0.1;  // 0.1 radian tolerance (~5.7 degrees)

    /**
     * Test phase detection for in-phase signals (0° phase difference).
     */
    @Test
    public void testInPhaseSignals() {
        var samplingRate = 1000.0;
        var duration = 1.0;
        var frequency = 40.0;

        var signal1 = generateSineWave(frequency, samplingRate, duration, 0.0);
        var signal2 = generateSineWave(frequency, samplingRate, duration, 0.0);

        var detector = new PhaseDetector();
        var phaseDiff = detector.computePhaseDifference(signal1, signal2);

        assertEquals(0.0, phaseDiff, PHASE_EPSILON,
            "In-phase signals should have 0 phase difference");
    }

    /**
     * Test phase detection for signals with π/4 (45°) phase shift.
     */
    @Test
    public void testQuarterPiPhaseShift() {
        var samplingRate = 1000.0;
        var duration = 1.0;
        var frequency = 40.0;

        var signal1 = generateSineWave(frequency, samplingRate, duration, 0.0);
        var signal2 = generateSineWave(frequency, samplingRate, duration, Math.PI / 4);

        var detector = new PhaseDetector();
        var phaseDiff = detector.computePhaseDifference(signal1, signal2);

        assertEquals(Math.PI / 4, phaseDiff, PHASE_EPSILON,
            "π/4 phase shift should be detected");
    }

    /**
     * Test phase detection for anti-phase signals (π or 180° phase difference).
     */
    @Test
    public void testAntiPhaseSignals() {
        var samplingRate = 1000.0;
        var duration = 1.0;
        var frequency = 40.0;

        var signal1 = generateSineWave(frequency, samplingRate, duration, 0.0);
        var signal2 = generateSineWave(frequency, samplingRate, duration, Math.PI);

        var detector = new PhaseDetector();
        var phaseDiff = detector.computePhaseDifference(signal1, signal2);

        assertEquals(Math.PI, phaseDiff, PHASE_EPSILON,
            "Anti-phase signals should have π phase difference");
    }

    /**
     * Test phase synchronization detection (phase-locked).
     */
    @Test
    public void testPhaseSynchronizationDetection() {
        var samplingRate = 1000.0;
        var duration = 1.0;
        var frequency = 40.0;

        var detector = new PhaseDetector();

        // Test 1: Phase-locked signals (small phase difference)
        var signal1 = generateSineWave(frequency, samplingRate, duration, 0.0);
        var signal2 = generateSineWave(frequency, samplingRate, duration, 0.1);  // Small shift

        var synced = detector.isPhaseSynchronized(signal1, signal2, Math.PI / 4);
        assertTrue(synced, "Signals with 0.1 rad difference should be phase-synchronized");

        // Test 2: Non-synchronized signals (large phase difference)
        var signal3 = generateSineWave(frequency, samplingRate, duration, Math.PI / 2);

        var notSynced = detector.isPhaseSynchronized(signal1, signal3, Math.PI / 4);
        assertFalse(notSynced, "Signals with π/2 difference should not be phase-synchronized");
    }

    /**
     * Test phase computation with Hilbert transform.
     */
    @Test
    public void testHilbertTransformPhase() {
        var samplingRate = 1000.0;
        var duration = 1.0;
        var frequency = 40.0;

        var signal = generateSineWave(frequency, samplingRate, duration, 0.0);

        var detector = new PhaseDetector();
        var instantPhase = detector.computeInstantaneousPhase(signal);

        assertNotNull(instantPhase, "Instantaneous phase should be computed");
        assertEquals(signal.length, instantPhase.length,
            "Phase array should match signal length");

        // Check that phase evolves linearly for pure sine wave
        // Phase should increase by ~2π every period
        var period = 1.0 / frequency;  // seconds
        var samplesPerPeriod = (int) (samplingRate * period);

        // Check phase progression over one period
        // For wrapped phase [-π, π], compute unwrapped cumulative phase change
        double cumulativePhase = 0.0;
        for (int i = 1; i <= samplesPerPeriod; i++) {
            double diff = instantPhase[i] - instantPhase[i-1];
            // Unwrap phase differences
            if (diff < -Math.PI) diff += 2 * Math.PI;
            else if (diff > Math.PI) diff -= 2 * Math.PI;
            cumulativePhase += diff;
        }

        // The cumulative phase change over one period should be ~2π
        assertEquals(2 * Math.PI, Math.abs(cumulativePhase), 0.5,
            "Phase should advance by 2π over one period");
    }

    /**
     * Test phase detection with noisy signals.
     */
    @Test
    public void testPhaseDetectionWithNoise() {
        var samplingRate = 1000.0;
        var duration = 2.0;  // Longer duration for better averaging
        var frequency = 40.0;
        var phaseShift = Math.PI / 3;  // 60 degrees

        var signal1 = generateSineWave(frequency, samplingRate, duration, 0.0);
        var signal2 = generateSineWave(frequency, samplingRate, duration, phaseShift);

        // Add moderate noise
        var noisy1 = addWhiteNoise(signal1, 0.2);
        var noisy2 = addWhiteNoise(signal2, 0.2);

        var detector = new PhaseDetector();
        var phaseDiff = detector.computePhaseDifference(noisy1, noisy2);

        assertEquals(phaseShift, phaseDiff, 0.3,  // More tolerance for noisy signals
            "Phase detection should work with moderate noise");
    }

    /**
     * Test phase locking index computation.
     */
    @Test
    public void testPhaseLockingIndex() {
        var samplingRate = 1000.0;
        var duration = 1.0;
        var frequency = 40.0;

        var detector = new PhaseDetector();

        // Test 1: Perfect phase locking (constant phase difference)
        var signal1 = generateSineWave(frequency, samplingRate, duration, 0.0);
        var signal2 = generateSineWave(frequency, samplingRate, duration, Math.PI / 4);

        var pli1 = detector.computePhaseLockingIndex(signal1, signal2);
        assertTrue(pli1 > 0.9, "Perfect phase locking should have PLI > 0.9");

        // Test 2: No phase locking (random phases)
        var random1 = generateRandomSignal((int) (samplingRate * duration));
        var random2 = generateRandomSignal((int) (samplingRate * duration));

        var pli2 = detector.computePhaseLockingIndex(random1, random2);
        assertTrue(pli2 < 0.5, "Random signals should have PLI < 0.5");
    }

    // ============== Test Helper Methods ==============

    /**
     * Generate sine wave with specified phase offset.
     */
    private double[] generateSineWave(
            double frequency,
            double samplingRate,
            double duration,
            double phaseOffset) {

        int n = (int) (samplingRate * duration);
        var signal = new double[n];

        for (int i = 0; i < n; i++) {
            double t = i / samplingRate;
            signal[i] = Math.sin(2.0 * Math.PI * frequency * t + phaseOffset);
        }

        return signal;
    }

    /**
     * Add white noise to signal.
     */
    private double[] addWhiteNoise(double[] signal, double noiseLevel) {
        var noisy = new double[signal.length];
        var random = new java.util.Random(42);

        for (int i = 0; i < signal.length; i++) {
            noisy[i] = signal[i] + noiseLevel * (random.nextDouble() - 0.5) * 2.0;
        }

        return noisy;
    }

    /**
     * Generate random signal for testing.
     */
    private double[] generateRandomSignal(int length) {
        var signal = new double[length];
        // Use current time as seed to get different random sequences each call
        var random = new java.util.Random();

        for (int i = 0; i < length; i++) {
            signal[i] = random.nextDouble() - 0.5;
        }

        return signal;
    }
}
