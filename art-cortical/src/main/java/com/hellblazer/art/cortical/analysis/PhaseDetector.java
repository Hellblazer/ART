package com.hellblazer.art.cortical.analysis;

import org.jtransforms.fft.DoubleFFT_1D;

/**
 * Phase detection and synchronization analysis for oscillatory signals.
 *
 * <p>Provides methods for:
 * <ul>
 *   <li>Computing phase difference between two signals</li>
 *   <li>Detecting phase synchronization (phase locking)</li>
 *   <li>Instantaneous phase estimation via Hilbert transform</li>
 *   <li>Phase locking index (PLI) computation</li>
 * </ul>
 *
 * <h2>Hilbert Transform Approach</h2>
 * <p>The analytic signal is constructed as:
 * <pre>
 * z(t) = x(t) + i·H[x(t)]
 *
 * where H[x] is the Hilbert transform of x.
 *
 * Instantaneous phase:
 * φ(t) = arctan(H[x(t)] / x(t))
 * </pre>
 *
 * <h2>Phase Synchronization</h2>
 * <p>Two oscillators are phase-synchronized if their phase difference
 * remains bounded:
 * <pre>
 * |φ₁(t) - φ₂(t)| < threshold
 * </pre>
 *
 * <p>Typical threshold: π/4 radians (45 degrees)
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * var detector = new PhaseDetector();
 *
 * // Compute phase difference
 * var phaseDiff = detector.computePhaseDifference(signal1, signal2);
 *
 * // Check if synchronized
 * var synced = detector.isPhaseSynchronized(signal1, signal2, Math.PI / 4);
 *
 * // Compute phase locking strength
 * var pli = detector.computePhaseLockingIndex(signal1, signal2);
 * }</pre>
 *
 * @author Phase 2: Oscillatory Dynamics Integration
 */
public class PhaseDetector {

    /**
     * Compute phase difference between two signals.
     *
     * <p>Uses cross-correlation in frequency domain to find relative phase shift.
     *
     * @param signal1 First signal
     * @param signal2 Second signal (must be same length as signal1)
     * @return Phase difference in radians [-π, π]
     * @throws IllegalArgumentException if signals have different lengths
     */
    public double computePhaseDifference(double[] signal1, double[] signal2) {
        if (signal1 == null || signal2 == null) {
            throw new IllegalArgumentException("signals cannot be null");
        }
        if (signal1.length != signal2.length) {
            throw new IllegalArgumentException(
                "signals must have same length: %d vs %d"
                .formatted(signal1.length, signal2.length)
            );
        }

        // Compute instantaneous phases
        var phase1 = computeInstantaneousPhase(signal1);
        var phase2 = computeInstantaneousPhase(signal2);

        // Compute mean phase difference (circular mean)
        double sumSin = 0.0;
        double sumCos = 0.0;

        for (int i = 0; i < phase1.length; i++) {
            double diff = phase2[i] - phase1[i];
            sumSin += Math.sin(diff);
            sumCos += Math.cos(diff);
        }

        // Circular mean: arctan2(mean(sin), mean(cos))
        return Math.atan2(sumSin / phase1.length, sumCos / phase1.length);
    }

    /**
     * Check if two signals are phase-synchronized.
     *
     * <p>Signals are considered synchronized if their phase difference
     * remains within the specified threshold.
     *
     * @param signal1 First signal
     * @param signal2 Second signal
     * @param threshold Maximum allowed phase difference (radians)
     * @return true if phase-synchronized
     */
    public boolean isPhaseSynchronized(double[] signal1, double[] signal2, double threshold) {
        var phaseDiff = Math.abs(computePhaseDifference(signal1, signal2));
        return phaseDiff < threshold;
    }

    /**
     * Compute instantaneous phase using Hilbert transform.
     *
     * <p>Steps:
     * <ol>
     *   <li>Compute FFT of signal</li>
     *   <li>Zero out negative frequencies (create analytic signal)</li>
     *   <li>Inverse FFT</li>
     *   <li>Phase = arctan2(imag, real)</li>
     * </ol>
     *
     * @param signal Time-domain signal
     * @return Instantaneous phase at each sample (radians, -π to π)
     */
    public double[] computeInstantaneousPhase(double[] signal) {
        if (signal == null || signal.length == 0) {
            throw new IllegalArgumentException("signal cannot be null or empty");
        }

        int n = nextPowerOf2(signal.length);

        // Pad signal to power-of-2
        var padded = new double[n * 2];  // Complex array: [real, imag, real, imag, ...]
        for (int i = 0; i < signal.length; i++) {
            padded[2 * i] = signal[i];      // Real part
            padded[2 * i + 1] = 0.0;        // Imaginary part
        }

        // Compute complex FFT
        var fft = new DoubleFFT_1D(n);
        fft.complexForward(padded);

        // Create analytic signal: zero out negative frequencies
        // Keep DC and positive frequencies, zero negative frequencies
        for (int i = n / 2 + 1; i < n; i++) {
            padded[2 * i] = 0.0;
            padded[2 * i + 1] = 0.0;
        }

        // Scale positive frequencies by 2 (except DC and Nyquist)
        for (int i = 1; i < n / 2; i++) {
            padded[2 * i] *= 2.0;
            padded[2 * i + 1] *= 2.0;
        }

        // Inverse FFT to get analytic signal
        fft.complexInverse(padded, true);  // true = scale by 1/n

        // Compute phase: arctan2(imag, real)
        var phase = new double[signal.length];
        for (int i = 0; i < signal.length; i++) {
            double real = padded[2 * i];
            double imag = padded[2 * i + 1];
            phase[i] = Math.atan2(imag, real);
        }

        return phase;
    }

    /**
     * Compute phase locking index (PLI) between two signals.
     *
     * <p>PLI measures consistency of phase relationship:
     * <pre>
     * PLI = |⟨e^(i·Δφ(t))⟩|
     *
     * where Δφ(t) = φ₂(t) - φ₁(t)
     * </pre>
     *
     * <p>PLI ranges from 0 (no locking) to 1 (perfect locking).
     *
     * @param signal1 First signal
     * @param signal2 Second signal
     * @return Phase locking index [0, 1]
     */
    public double computePhaseLockingIndex(double[] signal1, double[] signal2) {
        if (signal1.length != signal2.length) {
            throw new IllegalArgumentException(
                "signals must have same length: %d vs %d"
                .formatted(signal1.length, signal2.length)
            );
        }

        var phase1 = computeInstantaneousPhase(signal1);
        var phase2 = computeInstantaneousPhase(signal2);

        // Compute complex phase difference: e^(i·Δφ)
        double sumReal = 0.0;
        double sumImag = 0.0;

        for (int i = 0; i < phase1.length; i++) {
            double phaseDiff = phase2[i] - phase1[i];
            sumReal += Math.cos(phaseDiff);
            sumImag += Math.sin(phaseDiff);
        }

        // PLI = magnitude of mean complex phase difference
        double meanReal = sumReal / phase1.length;
        double meanImag = sumImag / phase1.length;

        return Math.sqrt(meanReal * meanReal + meanImag * meanImag);
    }

    /**
     * Find smallest power-of-2 >= n.
     */
    private int nextPowerOf2(int n) {
        if (n <= 0) {
            return 1;
        }
        if ((n & (n - 1)) == 0) {
            return n;
        }
        int power = 1;
        while (power < n) {
            power <<= 1;
        }
        return power;
    }
}
