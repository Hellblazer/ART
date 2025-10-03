package com.hellblazer.art.cortical.analysis;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for FFT-based spectral analysis.
 *
 * <p>Phase 2A: Mathematical Foundation - Tests FFT processing for oscillation detection.
 *
 * <p>Based on ART Cortical Enhancement Plan Section 3.2 (Phase 2A).
 *
 * @author Phase 2: Oscillatory Dynamics Integration
 */
public class FFTProcessorTest {

    private static final double EPSILON = 1.0;  // 1 Hz resolution tolerance

    /**
     * Test FFT detection of synthetic 40 Hz sine wave (gamma band).
     */
    @Test
    public void testFFTDetects40HzOscillation() {
        // Generate 40 Hz sine wave
        var samplingRate = 1000.0;  // 1000 Hz sampling
        var duration = 1.0;          // 1 second
        var frequency = 40.0;        // 40 Hz target
        var signal = generateSineWave(frequency, samplingRate, duration);

        var fft = new FFTProcessor(samplingRate);
        var spectrum = fft.computePowerSpectrum(signal);

        var peakFreq = spectrum.findPeakFrequency();
        assertEquals(40.0, peakFreq, EPSILON,
            "FFT should detect 40 Hz oscillation within 1 Hz");

        var gammaPower = spectrum.getPowerInBand(30.0, 50.0);
        var totalPower = spectrum.getTotalPower();
        var gammaRatio = gammaPower / totalPower;

        assertTrue(gammaRatio > 0.8,
            "Most power (>80%) should be in gamma band for 40 Hz signal");
    }

    /**
     * Test FFT detection of multiple frequencies (fundamental + harmonics).
     */
    @Test
    public void testFFTDetectsMultipleFrequencies() {
        var samplingRate = 1000.0;
        var duration = 1.0;

        // Generate signal with 40 Hz fundamental + 80 Hz harmonic
        var signal1 = generateSineWave(40.0, samplingRate, duration);
        var signal2 = generateSineWave(80.0, samplingRate, duration);

        var combined = new double[signal1.length];
        for (int i = 0; i < combined.length; i++) {
            combined[i] = signal1[i] + 0.5 * signal2[i];  // Harmonic is weaker
        }

        var fft = new FFTProcessor(samplingRate);
        var spectrum = fft.computePowerSpectrum(combined);

        var peaks = spectrum.findPeaks(2);  // Find top 2 peaks
        assertEquals(2, peaks.size(), "Should find 2 frequency peaks");

        // Fundamental should be stronger
        assertTrue(peaks.get(0).frequency() >= 38.0 && peaks.get(0).frequency() <= 42.0,
            "First peak should be ~40 Hz");
        assertTrue(peaks.get(1).frequency() >= 78.0 && peaks.get(1).frequency() <= 82.0,
            "Second peak should be ~80 Hz");
    }

    /**
     * Test gamma band power computation (30-50 Hz).
     */
    @Test
    public void testGammaBandPowerComputation() {
        var samplingRate = 1000.0;
        var duration = 1.0;

        // Test 1: 40 Hz signal (in gamma band)
        var gammaSignal = generateSineWave(40.0, samplingRate, duration);
        var fft = new FFTProcessor(samplingRate);
        var gammaSpectrum = fft.computePowerSpectrum(gammaSignal);
        var gammaPower1 = gammaSpectrum.getPowerInBand(30.0, 50.0);
        var totalPower1 = gammaSpectrum.getTotalPower();

        assertTrue(gammaPower1 / totalPower1 > 0.8,
            "40 Hz signal should have >80% power in gamma band");

        // Test 2: 10 Hz signal (alpha band, not gamma)
        var alphaSignal = generateSineWave(10.0, samplingRate, duration);
        var alphaSpectrum = fft.computePowerSpectrum(alphaSignal);
        var gammaPower2 = alphaSpectrum.getPowerInBand(30.0, 50.0);
        var totalPower2 = alphaSpectrum.getTotalPower();

        assertTrue(gammaPower2 / totalPower2 < 0.2,
            "10 Hz signal should have <20% power in gamma band");
    }

    /**
     * Test FFT with different signal lengths (power-of-2 padding).
     */
    @Test
    public void testFFTWithVariousSignalLengths() {
        var samplingRate = 1000.0;
        var frequency = 40.0;

        int[] lengths = {100, 256, 500, 1000, 1024};

        for (int length : lengths) {
            var signal = generateSineWave(frequency, samplingRate, length / samplingRate);

            var fft = new FFTProcessor(samplingRate);
            var spectrum = fft.computePowerSpectrum(signal);

            var peakFreq = spectrum.findPeakFrequency();
            assertEquals(40.0, peakFreq, 2.0,  // More tolerance for short signals
                "FFT should detect 40 Hz for signal length " + length);
        }
    }

    /**
     * Test FFT with noisy signal (signal + white noise).
     */
    @Test
    public void testFFTWithNoisySignal() {
        var samplingRate = 1000.0;
        var duration = 2.0;  // Longer duration for better SNR
        var frequency = 40.0;

        var signal = generateSineWave(frequency, samplingRate, duration);
        var noisy = addWhiteNoise(signal, 0.2);  // 20% noise

        var fft = new FFTProcessor(samplingRate);
        var spectrum = fft.computePowerSpectrum(noisy);

        var peakFreq = spectrum.findPeakFrequency();
        assertEquals(40.0, peakFreq, 2.0,
            "FFT should still detect 40 Hz in noisy signal");

        var gammaPower = spectrum.getPowerInBand(30.0, 50.0);
        var totalPower = spectrum.getTotalPower();

        assertTrue(gammaPower / totalPower > 0.3,
            "Even with noise, significant power should be in gamma band");
    }

    /**
     * Test PowerSpectrum record functionality.
     */
    @Test
    public void testPowerSpectrumRecord() {
        var frequencies = new double[]{10.0, 20.0, 30.0, 40.0, 50.0};
        var power = new double[]{0.1, 0.2, 0.3, 0.8, 0.4};

        var spectrum = new PowerSpectrum(frequencies, power);

        // Test peak finding
        assertEquals(40.0, spectrum.findPeakFrequency(),
            "Peak should be at 40 Hz");

        // Test band power
        var bandPower = spectrum.getPowerInBand(30.0, 50.0);
        assertEquals(0.3 + 0.8 + 0.4, bandPower, 1e-10,
            "Band power should sum 30, 40, 50 Hz bins");

        // Test total power
        var totalPower = spectrum.getTotalPower();
        assertEquals(0.1 + 0.2 + 0.3 + 0.8 + 0.4, totalPower, 1e-10,
            "Total power should sum all bins");
    }

    // ============== Test Helper Methods ==============

    /**
     * Generate pure sine wave at specified frequency.
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

    /**
     * Add white noise to signal.
     */
    private double[] addWhiteNoise(double[] signal, double noiseLevel) {
        var noisy = new double[signal.length];
        var random = new java.util.Random(42);  // Fixed seed for reproducibility

        for (int i = 0; i < signal.length; i++) {
            noisy[i] = signal[i] + noiseLevel * (random.nextDouble() - 0.5) * 2.0;
        }

        return noisy;
    }
}
