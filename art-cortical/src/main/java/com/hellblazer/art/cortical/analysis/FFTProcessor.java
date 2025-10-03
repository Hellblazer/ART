package com.hellblazer.art.cortical.analysis;

import org.jtransforms.fft.DoubleFFT_1D;

/**
 * FFT-based spectral analysis for oscillation detection.
 *
 * <p>Uses JTransforms library for efficient FFT computation. Provides:
 * <ul>
 *   <li>Power spectrum computation (frequency → power)</li>
 *   <li>Automatic zero-padding to power-of-2 for efficiency</li>
 *   <li>Frequency bin calculation based on sampling rate</li>
 * </ul>
 *
 * <h2>Mathematical Foundation</h2>
 * <p>Given a real-valued time series signal x[n] of length N:
 * <pre>
 * 1. Zero-pad to nearest power-of-2: N' = 2^⌈log₂(N)⌉
 * 2. Compute FFT: X[k] = Σ x[n]·e^(-2πikn/N')
 * 3. Power spectrum: P[k] = |X[k]|² / N'
 * 4. Frequency bins: f[k] = k · (samplingRate / N')
 * </pre>
 *
 * <h2>Gamma Oscillation Detection</h2>
 * <p>Typical usage for detecting ~40 Hz gamma oscillations:
 * <pre>{@code
 * // 1000 Hz sampling, analyze 1 second windows
 * var fft = new FFTProcessor(1000.0);
 *
 * // Collect activation history
 * double[] activations = layer.getActivationHistory(1000);
 *
 * // Compute spectrum
 * var spectrum = fft.computePowerSpectrum(activations);
 *
 * // Check for gamma oscillations
 * var peakFreq = spectrum.findPeakFrequency();
 * var gammaPower = spectrum.getPowerInBand(30.0, 50.0);
 *
 * if (peakFreq >= 30.0 && peakFreq <= 50.0) {
 *     System.out.println("Gamma oscillation detected at " + peakFreq + " Hz");
 * }
 * }</pre>
 *
 * <h2>Performance Characteristics</h2>
 * <ul>
 *   <li>Time complexity: O(N log N) where N is padded length</li>
 *   <li>Space complexity: O(N) for zero-padding</li>
 *   <li>Recommended signal length: 256-1024 samples (power-of-2)</li>
 *   <li>Frequency resolution: samplingRate / N Hz</li>
 * </ul>
 *
 * @author Phase 2: Oscillatory Dynamics Integration
 */
public class FFTProcessor {

    private final double samplingRate;

    /**
     * Create FFT processor with specified sampling rate.
     *
     * @param samplingRate Sampling rate in Hz (must be > 0)
     * @throws IllegalArgumentException if samplingRate <= 0
     */
    public FFTProcessor(double samplingRate) {
        if (samplingRate <= 0) {
            throw new IllegalArgumentException(
                "samplingRate must be positive, got: " + samplingRate
            );
        }
        this.samplingRate = samplingRate;
    }

    /**
     * Compute power spectrum from time-domain signal.
     *
     * <p>Steps:
     * <ol>
     *   <li>Zero-pad signal to power-of-2 length</li>
     *   <li>Compute real-valued FFT using JTransforms</li>
     *   <li>Calculate power spectrum: |X[k]|²</li>
     *   <li>Map frequency bins: f[k] = k · (samplingRate / N)</li>
     * </ol>
     *
     * @param signal Time-domain signal (arbitrary length)
     * @return Power spectrum with frequency bins
     * @throws IllegalArgumentException if signal is null or empty
     */
    public PowerSpectrum computePowerSpectrum(double[] signal) {
        if (signal == null || signal.length == 0) {
            throw new IllegalArgumentException("signal cannot be null or empty");
        }

        // Zero-pad to power-of-2 for efficient FFT
        var padded = zeroPadToPowerOf2(signal);
        var n = padded.length;

        // Create FFT processor for this size
        var fft = new DoubleFFT_1D(n);

        // Compute real-valued FFT (in-place)
        // After FFT: padded[0]=Re[0], padded[1]=Re[1], ..., padded[n-1]=Re[n/2]
        // For real FFT: padded[2k]=Re[k], padded[2k+1]=Im[k]
        fft.realForward(padded);

        // Extract power spectrum (positive frequencies only)
        var spectrumSize = n / 2;
        var power = new double[spectrumSize];
        var frequencies = new double[spectrumSize];

        // DC component (k=0)
        power[0] = (padded[0] * padded[0]) / n;
        frequencies[0] = 0.0;

        // Positive frequencies (k=1 to n/2-1)
        for (int k = 1; k < spectrumSize; k++) {
            var real = padded[2 * k];
            var imag = padded[2 * k + 1];
            power[k] = (real * real + imag * imag) / n;
            frequencies[k] = k * samplingRate / n;
        }

        return new PowerSpectrum(frequencies, power);
    }

    /**
     * Zero-pad signal to nearest power-of-2 length.
     *
     * <p>Padding improves FFT efficiency (O(N log N) for power-of-2).
     *
     * @param signal Original signal
     * @return Zero-padded signal (length is power-of-2)
     */
    private double[] zeroPadToPowerOf2(double[] signal) {
        int n = signal.length;

        // Find next power-of-2 >= n
        int paddedSize = nextPowerOf2(n);

        if (paddedSize == n) {
            // Already power-of-2, no padding needed
            return signal.clone();  // Clone to avoid modifying input
        }

        // Create padded array (initialized to 0.0)
        var padded = new double[paddedSize];
        System.arraycopy(signal, 0, padded, 0, n);

        return padded;
    }

    /**
     * Find smallest power-of-2 >= n.
     *
     * @param n Input size
     * @return Smallest k such that 2^k >= n
     */
    private int nextPowerOf2(int n) {
        if (n <= 0) {
            return 1;
        }

        // Check if already power-of-2
        if ((n & (n - 1)) == 0) {
            return n;
        }

        // Find next power-of-2
        int power = 1;
        while (power < n) {
            power <<= 1;
        }

        return power;
    }

    /**
     * Get sampling rate.
     *
     * @return Sampling rate in Hz
     */
    public double getSamplingRate() {
        return samplingRate;
    }

    /**
     * Compute frequency resolution for given signal length.
     *
     * <p>Frequency resolution = samplingRate / paddedLength
     *
     * @param signalLength Original signal length
     * @return Frequency resolution in Hz
     */
    public double getFrequencyResolution(int signalLength) {
        int paddedSize = nextPowerOf2(signalLength);
        return samplingRate / paddedSize;
    }
}
