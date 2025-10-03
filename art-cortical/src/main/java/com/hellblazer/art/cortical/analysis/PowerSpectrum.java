package com.hellblazer.art.cortical.analysis;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Power spectrum from FFT analysis.
 *
 * <p>Represents frequency-domain representation of a signal, containing:
 * <ul>
 *   <li>Frequency bins (Hz)</li>
 *   <li>Power at each frequency (magnitude squared)</li>
 * </ul>
 *
 * <p>Provides utilities for:
 * <ul>
 *   <li>Finding dominant frequencies (peaks)</li>
 *   <li>Computing band power (e.g., gamma 30-50 Hz)</li>
 *   <li>Extracting frequency-domain features</li>
 * </ul>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * var fft = new FFTProcessor(1000.0);  // 1000 Hz sampling
 * var spectrum = fft.computePowerSpectrum(signal);
 *
 * var peakFreq = spectrum.findPeakFrequency();
 * var gammaPower = spectrum.getPowerInBand(30.0, 50.0);
 * }</pre>
 *
 * @param frequencies Frequency bins in Hz
 * @param power Power at each frequency (non-negative)
 *
 * @author Phase 2: Oscillatory Dynamics Integration
 */
public record PowerSpectrum(double[] frequencies, double[] power) {

    /**
     * Compact constructor with validation.
     */
    public PowerSpectrum {
        if (frequencies == null || power == null) {
            throw new IllegalArgumentException("frequencies and power cannot be null");
        }
        if (frequencies.length != power.length) {
            throw new IllegalArgumentException(
                "frequencies and power must have same length: %d vs %d"
                .formatted(frequencies.length, power.length)
            );
        }
        if (frequencies.length == 0) {
            throw new IllegalArgumentException("spectrum cannot be empty");
        }
    }

    /**
     * Find frequency with maximum power.
     *
     * @return Frequency (Hz) with highest power
     */
    public double findPeakFrequency() {
        var maxPower = 0.0;
        var peakFreq = 0.0;

        for (int i = 0; i < power.length; i++) {
            if (power[i] > maxPower) {
                maxPower = power[i];
                peakFreq = frequencies[i];
            }
        }

        return peakFreq;
    }

    /**
     * Find top N frequency peaks.
     *
     * @param n Number of peaks to find
     * @return List of peaks sorted by power (descending)
     */
    public List<Peak> findPeaks(int n) {
        if (n <= 0 || n > power.length) {
            throw new IllegalArgumentException(
                "n must be in range [1, %d], got %d".formatted(power.length, n)
            );
        }

        var peaks = new ArrayList<Peak>();
        for (int i = 0; i < power.length; i++) {
            peaks.add(new Peak(frequencies[i], power[i]));
        }

        // Sort by power descending
        peaks.sort(Comparator.comparingDouble(Peak::power).reversed());

        return peaks.subList(0, n);
    }

    /**
     * Compute total power in specified frequency band.
     *
     * @param lowFreq Lower bound (Hz, inclusive)
     * @param highFreq Upper bound (Hz, inclusive)
     * @return Sum of power in frequency band
     */
    public double getPowerInBand(double lowFreq, double highFreq) {
        if (lowFreq > highFreq) {
            throw new IllegalArgumentException(
                "lowFreq (%f) must be <= highFreq (%f)".formatted(lowFreq, highFreq)
            );
        }

        var bandPower = 0.0;
        for (int i = 0; i < frequencies.length; i++) {
            if (frequencies[i] >= lowFreq && frequencies[i] <= highFreq) {
                bandPower += power[i];
            }
        }

        return bandPower;
    }

    /**
     * Compute total power across all frequencies.
     *
     * @return Sum of all power bins
     */
    public double getTotalPower() {
        var total = 0.0;
        for (double p : power) {
            total += p;
        }
        return total;
    }

    /**
     * Get number of frequency bins.
     *
     * @return Spectrum size
     */
    public int size() {
        return frequencies.length;
    }

    /**
     * Frequency peak with power.
     *
     * @param frequency Frequency in Hz
     * @param power Power at this frequency
     */
    public record Peak(double frequency, double power) {
        public Peak {
            if (power < 0) {
                throw new IllegalArgumentException("power cannot be negative: " + power);
            }
        }
    }
}
