package com.hellblazer.art.cortical.resonance;

import com.hellblazer.art.cortical.analysis.CircularBuffer;
import com.hellblazer.art.cortical.analysis.OscillationAnalyzer;
import com.hellblazer.art.cortical.analysis.OscillationMetrics;

/**
 * Enhanced resonance detection with oscillatory dynamics and consciousness metrics.
 *
 * <p>Implements Grossberg (2017) CLEARS framework by combining:
 * <ul>
 *   <li><b>Traditional ART</b>: Feature-expectation match quality and vigilance</li>
 *   <li><b>Oscillatory Dynamics</b>: Gamma band detection and phase synchronization</li>
 *   <li><b>Consciousness Metrics</b>: Heuristic likelihood based on resonance and oscillations</li>
 * </ul>
 *
 * <h2>Architecture</h2>
 * <pre>
 * Bottom-Up Features ──▶ OscillationAnalyzer ──▶ Frequency, Phase, Gamma Power
 *                              │
 *                              ▼
 *                         Match Quality ──▶ ART Resonance (vigilance threshold)
 *                              │
 *                              ▼
 * Top-Down Expectations ──▶ OscillationAnalyzer ──▶ Frequency, Phase, Gamma Power
 *                              │
 *                              ▼
 *                    Phase Synchronization Detection
 *                              │
 *                              ▼
 *                    Consciousness Likelihood Computation
 * </pre>
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * // Create detector with vigilance threshold
 * var detector = new EnhancedResonanceDetector(0.7, 1000.0, 256);
 *
 * // Process activation history for both pathways
 * for (int t = 0; t < 256; t++) {
 *     detector.recordBottomUp(layer4.getActivation());
 *     detector.recordTopDown(layer1.getActivation());
 * }
 *
 * // Detect resonance with oscillatory analysis
 * var state = detector.detectResonance(
 *     bottomUpFeatures,
 *     topDownExpectations,
 *     timestamp
 * );
 *
 * // Check for conscious perception
 * if (state.isLikelyConscious(0.7)) {
 *     System.out.println("Conscious perception detected!");
 *     System.out.println("  Consciousness: " + state.consciousnessLikelihood());
 *     System.out.println("  Phase sync: " + state.phaseSynchronized());
 *     System.out.println("  Gamma: " + state.bothInGamma());
 * }
 * }</pre>
 *
 * @author Phase 2D: Enhanced Resonance Detection
 */
public class EnhancedResonanceDetector {

    private final double vigilanceThreshold;
    private final OscillationAnalyzer bottomUpAnalyzer;
    private final OscillationAnalyzer topDownAnalyzer;
    private final CircularBuffer<double[]> bottomUpHistory;
    private final CircularBuffer<double[]> topDownHistory;

    /**
     * Create enhanced resonance detector with oscillation tracking.
     *
     * @param vigilanceThreshold ART vigilance threshold [0, 1]
     * @param samplingRate Sampling rate in Hz (typically 1000 for 1ms timesteps)
     * @param historySize Number of samples for oscillation analysis (power-of-2 recommended)
     * @throws IllegalArgumentException if parameters invalid
     */
    public EnhancedResonanceDetector(
            double vigilanceThreshold,
            double samplingRate,
            int historySize) {

        if (vigilanceThreshold < 0.0 || vigilanceThreshold > 1.0) {
            throw new IllegalArgumentException(
                "vigilanceThreshold must be in [0, 1]: " + vigilanceThreshold
            );
        }
        if (samplingRate <= 0) {
            throw new IllegalArgumentException("samplingRate must be positive: " + samplingRate);
        }
        if (historySize <= 0) {
            throw new IllegalArgumentException("historySize must be positive: " + historySize);
        }

        this.vigilanceThreshold = vigilanceThreshold;
        this.bottomUpAnalyzer = new OscillationAnalyzer(samplingRate, historySize);
        this.topDownAnalyzer = new OscillationAnalyzer(samplingRate, historySize);
        this.bottomUpHistory = new CircularBuffer<>(historySize);
        this.topDownHistory = new CircularBuffer<>(historySize);
    }

    /**
     * Record bottom-up activation for oscillation analysis.
     *
     * @param activation Bottom-up activation pattern
     */
    public void recordBottomUp(double[] activation) {
        if (activation != null) {
            bottomUpHistory.add(activation.clone());
        }
    }

    /**
     * Record top-down activation for oscillation analysis.
     *
     * @param activation Top-down activation pattern
     */
    public void recordTopDown(double[] activation) {
        if (activation != null) {
            topDownHistory.add(activation.clone());
        }
    }

    /**
     * Detect resonance with oscillatory analysis and consciousness metrics.
     *
     * <p>Steps:
     * <ol>
     *   <li>Compute traditional ART match quality</li>
     *   <li>Determine ART resonance (match &gt;= vigilance)</li>
     *   <li>Analyze oscillations in both pathways (if history available)</li>
     *   <li>Detect phase synchronization</li>
     *   <li>Detect gamma band oscillations</li>
     *   <li>Compute consciousness likelihood</li>
     * </ol>
     *
     * @param bottomUpFeatures Bottom-up feature vector
     * @param topDownExpectations Top-down expectation vector
     * @param timestamp Current time (seconds)
     * @return Enhanced resonance state with consciousness metrics
     */
    public ResonanceState detectResonance(
            double[] bottomUpFeatures,
            double[] topDownExpectations,
            double timestamp) {

        // 1. Traditional ART matching
        var matchQuality = computeMatchQuality(bottomUpFeatures, topDownExpectations);
        var artResonance = matchQuality >= vigilanceThreshold;

        // 2. Oscillatory analysis (if history available)
        OscillationMetrics buMetrics = null;
        OscillationMetrics tdMetrics = null;

        if (bottomUpHistory.isFull() && topDownHistory.isFull()) {
            buMetrics = bottomUpAnalyzer.analyze(bottomUpHistory, timestamp);
            tdMetrics = topDownAnalyzer.analyze(topDownHistory, timestamp);
        }

        // 3. Phase synchronization detection
        var phaseSync = false;
        if (buMetrics != null && tdMetrics != null) {
            var phaseDiff = Math.abs(buMetrics.phaseDifferenceWith(tdMetrics));
            phaseSync = phaseDiff < ResonanceState.PHASE_SYNC_THRESHOLD;
        }

        // 4. Gamma band detection
        var bothInGamma = false;
        if (buMetrics != null && tdMetrics != null) {
            var buInGamma = isInGammaBand(buMetrics.dominantFrequency());
            var tdInGamma = isInGammaBand(tdMetrics.dominantFrequency());
            bothInGamma = buInGamma && tdInGamma;
        }

        // 5. Consciousness likelihood (heuristic)
        var consciousnessLikelihood = computeConsciousnessLikelihood(
            artResonance, phaseSync, bothInGamma, matchQuality
        );

        return new ResonanceState(
            artResonance,
            phaseSync,
            bothInGamma,
            consciousnessLikelihood,
            buMetrics,
            tdMetrics,
            matchQuality,
            timestamp
        );
    }

    /**
     * Compute match quality between bottom-up features and top-down expectations.
     *
     * <p>Uses fuzzy ART match function: |min(features, expectations)| / |features|
     *
     * @param features Bottom-up feature vector
     * @param expectations Top-down expectation vector
     * @return Match quality [0, 1]
     */
    private double computeMatchQuality(double[] features, double[] expectations) {
        if (features == null || expectations == null) {
            return 0.0;
        }

        int dim = Math.min(features.length, expectations.length);
        if (dim == 0) {
            return 0.0;
        }

        // Fuzzy ART match: |min(a, b)| / |a|
        double intersection = 0.0;
        double featureMagnitude = 0.0;

        for (int i = 0; i < dim; i++) {
            intersection += Math.min(features[i], expectations[i]);
            featureMagnitude += features[i];
        }

        return featureMagnitude > 0.0 ? intersection / featureMagnitude : 0.0;
    }

    /**
     * Compute consciousness likelihood based on multiple factors.
     *
     * <p>Formula:
     * <ul>
     *   <li>Base: match quality (if ART resonance)</li>
     *   <li>+0.2 if phase synchronized</li>
     *   <li>+0.3 if both in gamma band</li>
     * </ul>
     *
     * @param artResonance ART vigilance threshold met
     * @param phaseSync Phase synchronization detected
     * @param gammaOscillations Both pathways in gamma band
     * @param matchQuality Feature-expectation match quality
     * @return Consciousness likelihood [0, 1]
     */
    private double computeConsciousnessLikelihood(
            boolean artResonance,
            boolean phaseSync,
            boolean gammaOscillations,
            double matchQuality) {

        if (!artResonance) {
            return 0.0;  // No resonance = no conscious perception
        }

        var likelihood = matchQuality;  // Base: ART match quality

        if (phaseSync) {
            likelihood += 0.2;  // Phase synchronization bonus
        }

        if (gammaOscillations) {
            likelihood += 0.3;  // Gamma oscillation bonus
        }

        return Math.min(1.0, likelihood);
    }

    /**
     * Check if frequency is in gamma band (30-50 Hz).
     *
     * @param frequency Frequency in Hz
     * @return true if in gamma band
     */
    private boolean isInGammaBand(double frequency) {
        return frequency >= ResonanceState.GAMMA_LOW
            && frequency <= ResonanceState.GAMMA_HIGH;
    }

    /**
     * Reset oscillation history buffers.
     */
    public void reset() {
        bottomUpHistory.clear();
        topDownHistory.clear();
    }

    /**
     * Get vigilance threshold.
     *
     * @return Vigilance threshold [0, 1]
     */
    public double getVigilanceThreshold() {
        return vigilanceThreshold;
    }

    /**
     * Check if oscillation analysis is ready (both buffers full).
     *
     * @return true if ready for full oscillation analysis
     */
    public boolean isReady() {
        return bottomUpHistory.isFull() && topDownHistory.isFull();
    }

    @Override
    public String toString() {
        return "EnhancedResonanceDetector[vigilance=" + vigilanceThreshold +
            ", ready=" + isReady() + "]";
    }
}
