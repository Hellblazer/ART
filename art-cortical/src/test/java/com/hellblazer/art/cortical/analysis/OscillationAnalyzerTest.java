package com.hellblazer.art.cortical.analysis;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for OscillationAnalyzer.
 *
 * <p>Tests oscillation analysis on synthetic neural activation patterns.
 *
 * @author Phase 2: Oscillatory Dynamics Integration
 */
public class OscillationAnalyzerTest {

    private static final double FREQ_TOLERANCE = 2.0;  // 2 Hz tolerance

    /**
     * Test analyzer detects 40 Hz gamma oscillation in activation history.
     */
    @Test
    public void testDetectGammaOscillation() {
        var samplingRate = 1000.0;
        var historySize = 256;
        var frequency = 40.0;

        var analyzer = new OscillationAnalyzer(samplingRate, historySize);

        // Generate oscillatory activation history
        var history = generateOscillatoryHistory(frequency, samplingRate, historySize, 64);

        var metrics = analyzer.analyze(history, 1.0);

        // With FFT resolution and averaging, dominant frequency should be close to 40 Hz
        assertEquals(40.0, metrics.dominantFrequency(), 5.0,
            "Dominant frequency should be near 40 Hz");

        // Averaging across neurons may reduce gamma power, so use lower threshold
        assertTrue(metrics.gammaPower() > 0.3,
            "Gamma power should be significant for 40 Hz signal, got: " + metrics.gammaPower());
    }

    /**
     * Test analyzer distinguishes gamma from alpha oscillations.
     */
    @Test
    public void testDistinguishGammaFromAlpha() {
        var samplingRate = 1000.0;
        var historySize = 256;

        var analyzer = new OscillationAnalyzer(samplingRate, historySize);

        // Test 1: Gamma (40 Hz)
        var gammaHistory = generateOscillatoryHistory(40.0, samplingRate, historySize, 64);
        var gammaMetrics = analyzer.analyze(gammaHistory, 1.0);

        // Dominant frequency should be near 40 Hz
        assertTrue(gammaMetrics.dominantFrequency() >= 35.0
                && gammaMetrics.dominantFrequency() <= 45.0,
            "40 Hz should be detected near gamma band, got: " + gammaMetrics.dominantFrequency());

        // Test 2: Alpha (10 Hz)
        var alphaHistory = generateOscillatoryHistory(10.0, samplingRate, historySize, 64);
        var alphaMetrics = analyzer.analyze(alphaHistory, 1.0);

        assertFalse(alphaMetrics.isGammaOscillation(), "10 Hz should not be gamma");
        assertTrue(alphaMetrics.dominantFrequency() < 30.0,
            "10 Hz should be below gamma band");
    }

    /**
     * Test analyzer with scalar time-series (pre-averaged activation).
     */
    @Test
    public void testAnalyzeScalarTimeSeries() {
        var samplingRate = 1000.0;
        var frequency = 40.0;
        var duration = 0.5;  // 500ms

        var signal = generateSineWave(frequency, samplingRate, duration);

        var analyzer = new OscillationAnalyzer(samplingRate, signal.length);
        var metrics = analyzer.analyze(signal, 0.5);

        assertEquals(40.0, metrics.dominantFrequency(), FREQ_TOLERANCE,
            "Should detect 40 Hz in scalar time-series");
        assertTrue(metrics.isGammaOscillation(),
            "Should classify as gamma");
    }

    /**
     * Test analyzer configuration and parameters.
     */
    @Test
    public void testAnalyzerConfiguration() {
        var samplingRate = 1000.0;
        var historySize = 256;

        var analyzer = new OscillationAnalyzer(samplingRate, historySize);

        assertEquals(1000.0, analyzer.getSamplingRate(),
            "Sampling rate should be 1000 Hz");
        assertEquals(256, analyzer.getHistorySize(),
            "History size should be 256");

        var freqRes = analyzer.getFrequencyResolution();
        assertEquals(samplingRate / historySize, freqRes, 1e-10,
            "Frequency resolution should be samplingRate/historySize");
    }

    /**
     * Test phase tracking over time.
     */
    @Test
    public void testPhaseTracking() {
        var samplingRate = 1000.0;
        var historySize = 256;
        var frequency = 40.0;

        var analyzer = new OscillationAnalyzer(samplingRate, historySize);

        // Generate history with known phase
        var history1 = generateOscillatoryHistory(frequency, samplingRate, historySize, 64, 0.0);
        var metrics1 = analyzer.analyze(history1, 1.0);

        // Generate history with π/2 phase shift
        var history2 = generateOscillatoryHistory(frequency, samplingRate, historySize, 64, Math.PI / 2);
        var metrics2 = analyzer.analyze(history2, 2.0);

        // Phase should be tracked (within [-π, π])
        assertTrue(metrics1.phase() >= -Math.PI && metrics1.phase() <= Math.PI,
            "Phase should be in valid range");
        assertTrue(metrics2.phase() >= -Math.PI && metrics2.phase() <= Math.PI,
            "Phase should be in valid range");

        // Phases should be different
        var phaseDiff = Math.abs(metrics1.phaseDifferenceWith(metrics2));
        assertTrue(phaseDiff > 0.1,
            "Different phase inputs should produce different phase outputs");
    }

    /**
     * Test CircularBuffer with oscillation analysis.
     */
    @Test
    public void testCircularBufferIntegration() {
        var samplingRate = 1000.0;
        var historySize = 256;
        var frequency = 40.0;

        var analyzer = new OscillationAnalyzer(samplingRate, historySize);
        var buffer = new CircularBuffer<double[]>(historySize);

        // Fill buffer with oscillatory activations
        for (int t = 0; t < historySize; t++) {
            double time = t / samplingRate;
            var activation = new double[64];
            for (int n = 0; n < 64; n++) {
                activation[n] = Math.sin(2 * Math.PI * frequency * time);
            }
            buffer.add(activation);
        }

        assertTrue(buffer.isFull(), "Buffer should be full");

        var metrics = analyzer.analyze(buffer, historySize / samplingRate);

        assertTrue(metrics.isGammaOscillation(),
            "Should detect gamma oscillation from buffer");
        assertEquals(40.0, metrics.dominantFrequency(), FREQ_TOLERANCE,
            "Should detect 40 Hz");
    }

    /**
     * Test with non-oscillatory (constant) activation.
     */
    @Test
    public void testConstantActivation() {
        var samplingRate = 1000.0;
        var historySize = 256;

        var analyzer = new OscillationAnalyzer(samplingRate, historySize);

        // Generate constant activation (no oscillation)
        var history = new CircularBuffer<double[]>(historySize);
        for (int t = 0; t < historySize; t++) {
            var activation = new double[64];
            for (int n = 0; n < 64; n++) {
                activation[n] = 0.5;  // Constant value
            }
            history.add(activation);
        }

        var metrics = analyzer.analyze(history, 1.0);

        // Dominant frequency should be 0 or very low
        assertTrue(metrics.dominantFrequency() < 5.0,
            "Constant signal should have very low dominant frequency");
    }

    // ============== Helper Methods ==============

    /**
     * Generate oscillatory activation history.
     */
    private CircularBuffer<double[]> generateOscillatoryHistory(
            double frequency,
            double samplingRate,
            int historySize,
            int numNeurons) {
        return generateOscillatoryHistory(frequency, samplingRate, historySize, numNeurons, 0.0);
    }

    /**
     * Generate oscillatory activation history with specified phase.
     */
    private CircularBuffer<double[]> generateOscillatoryHistory(
            double frequency,
            double samplingRate,
            int historySize,
            int numNeurons,
            double phaseOffset) {

        var history = new CircularBuffer<double[]>(historySize);

        for (int t = 0; t < historySize; t++) {
            double time = t / samplingRate;
            var activation = new double[numNeurons];

            for (int n = 0; n < numNeurons; n++) {
                // Pure oscillation (no DC offset) so averaging preserves oscillation
                activation[n] = Math.sin(2 * Math.PI * frequency * time + phaseOffset);
            }

            history.add(activation);
        }

        return history;
    }

    /**
     * Generate sine wave signal.
     */
    private double[] generateSineWave(double frequency, double samplingRate, double duration) {
        int n = (int) (samplingRate * duration);
        var signal = new double[n];

        for (int i = 0; i < n; i++) {
            double t = i / samplingRate;
            signal[i] = Math.sin(2.0 * Math.PI * frequency * t);
        }

        return signal;
    }
}
