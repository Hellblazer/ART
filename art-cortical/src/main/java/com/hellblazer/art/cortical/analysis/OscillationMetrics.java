package com.hellblazer.art.cortical.analysis;

/**
 * Oscillation analysis metrics for a layer or signal.
 *
 * <p>Captures frequency-domain characteristics of neural oscillations:
 * <ul>
 *   <li><b>Dominant frequency</b>: Peak frequency in power spectrum (Hz)</li>
 *   <li><b>Gamma power</b>: Power in gamma band (30-50 Hz)</li>
 *   <li><b>Instantaneous phase</b>: Current phase angle (radians)</li>
 *   <li><b>Timestamp</b>: When metrics were computed</li>
 * </ul>
 *
 * <h2>Gamma Oscillations (~40 Hz)</h2>
 * <p>Gamma oscillations are associated with:
 * <ul>
 *   <li>Feature binding and perceptual grouping</li>
 *   <li>ART resonance states</li>
 *   <li>Conscious perception (according to unified theory)</li>
 *   <li>Attention and working memory</li>
 * </ul>
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * var metrics = analyzer.analyze(activationHistory, currentTime);
 *
 * if (metrics.isGammaOscillation()) {
 *     System.out.printf("Gamma oscillation at %.1f Hz (power: %.3f)%n",
 *         metrics.dominantFrequency(), metrics.gammaPower());
 * }
 *
 * // Check for resonance based on gamma activity
 * boolean inResonance = metrics.dominantFrequency() >= 35.0
 *                    && metrics.dominantFrequency() <= 45.0
 *                    && metrics.gammaPower() > 0.5;
 * }</pre>
 *
 * @param dominantFrequency Peak frequency in Hz
 * @param gammaPower Power in gamma band (30-50 Hz), normalized [0, 1]
 * @param phase Instantaneous phase in radians [-π, π]
 * @param timestamp Time when metrics were computed (seconds)
 *
 * @author Phase 2: Oscillatory Dynamics Integration
 */
public record OscillationMetrics(
    double dominantFrequency,
    double gammaPower,
    double phase,
    double timestamp
) {
    /**
     * Gamma band frequency range (Hz).
     */
    public static final double GAMMA_LOW = 30.0;
    public static final double GAMMA_HIGH = 50.0;

    /**
     * Typical gamma center frequency (Hz).
     */
    public static final double GAMMA_CENTER = 40.0;

    /**
     * Compact constructor with validation.
     */
    public OscillationMetrics {
        if (dominantFrequency < 0) {
            throw new IllegalArgumentException(
                "dominantFrequency cannot be negative: " + dominantFrequency
            );
        }
        if (gammaPower < 0 || gammaPower > 1) {
            throw new IllegalArgumentException(
                "gammaPower must be in [0, 1]: " + gammaPower
            );
        }
        if (phase < -Math.PI || phase > Math.PI) {
            throw new IllegalArgumentException(
                "phase must be in [-π, π]: " + phase
            );
        }
    }

    /**
     * Check if dominant frequency is in gamma band.
     *
     * @return true if frequency in [30, 50] Hz
     */
    public boolean isGammaOscillation() {
        return dominantFrequency >= GAMMA_LOW && dominantFrequency <= GAMMA_HIGH;
    }

    /**
     * Check if dominant frequency is near typical gamma (40 Hz).
     *
     * @param tolerance Frequency tolerance (Hz)
     * @return true if within tolerance of 40 Hz
     */
    public boolean isNearGammaCenter(double tolerance) {
        return Math.abs(dominantFrequency - GAMMA_CENTER) <= tolerance;
    }

    /**
     * Check if gamma power exceeds threshold.
     *
     * @param threshold Minimum gamma power [0, 1]
     * @return true if gamma power >= threshold
     */
    public boolean hasStrongGamma(double threshold) {
        return gammaPower >= threshold;
    }

    /**
     * Compute phase difference with another oscillation.
     *
     * <p>Returns phase difference in [-π, π] range.
     *
     * @param other Other oscillation metrics
     * @return Phase difference in radians
     */
    public double phaseDifferenceWith(OscillationMetrics other) {
        double diff = this.phase - other.phase;

        // Normalize to [-π, π]
        while (diff > Math.PI) {
            diff -= 2 * Math.PI;
        }
        while (diff < -Math.PI) {
            diff += 2 * Math.PI;
        }

        return diff;
    }

    /**
     * Check if phase-synchronized with another oscillation.
     *
     * @param other Other oscillation metrics
     * @param threshold Maximum phase difference for synchronization (radians)
     * @return true if phase difference < threshold
     */
    public boolean isPhaseSynchronizedWith(OscillationMetrics other, double threshold) {
        return Math.abs(phaseDifferenceWith(other)) < threshold;
    }

    /**
     * Create metrics indicating no oscillation detected.
     *
     * @param timestamp Current time
     * @return Metrics with zero values
     */
    public static OscillationMetrics none(double timestamp) {
        return new OscillationMetrics(0.0, 0.0, 0.0, timestamp);
    }

    @Override
    public String toString() {
        return "OscillationMetrics[freq=%.1f Hz, gamma=%.3f, phase=%.2f rad, t=%.3f]"
            .formatted(dominantFrequency, gammaPower, phase, timestamp);
    }
}
