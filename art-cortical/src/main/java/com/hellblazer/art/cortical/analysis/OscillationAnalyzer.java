package com.hellblazer.art.cortical.analysis;

/**
 * Oscillation analysis for neural layer activations.
 *
 * <p>Analyzes activation time-series to detect oscillatory dynamics:
 * <ul>
 *   <li>Dominant frequency via FFT power spectrum</li>
 *   <li>Gamma band power (30-50 Hz)</li>
 *   <li>Instantaneous phase via Hilbert transform</li>
 * </ul>
 *
 * <h2>Workflow</h2>
 * <pre>
 * 1. Collect activation history (CircularBuffer)
 * 2. Average across neurons to get scalar time-series
 * 3. Compute FFT power spectrum
 * 4. Extract dominant frequency and gamma power
 * 5. Compute instantaneous phase
 * 6. Return OscillationMetrics
 * </pre>
 *
 * <h2>Configuration</h2>
 * <p>Requires specification of:
 * <ul>
 *   <li>Sampling rate (Hz) - typically 1000 Hz for 1ms timesteps</li>
 *   <li>History size (samples) - typically 256 or 512 (power-of-2)</li>
 * </ul>
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * // Create analyzer (1000 Hz sampling, 256 sample window)
 * var analyzer = new OscillationAnalyzer(1000.0, 256);
 *
 * // Collect activation history
 * var history = new CircularBuffer<double[]>(256);
 * for (int t = 0; t < 256; t++) {
 *     history.add(layer.getActivation());
 * }
 *
 * // Analyze when buffer is full
 * if (history.isFull()) {
 *     var metrics = analyzer.analyze(history, currentTime);
 *
 *     if (metrics.isGammaOscillation()) {
 *         System.out.println("Gamma resonance detected!");
 *     }
 * }
 * }</pre>
 *
 * @author Phase 2: Oscillatory Dynamics Integration
 */
public class OscillationAnalyzer {

    private final FFTProcessor fftProcessor;
    private final PhaseDetector phaseDetector;
    private final double samplingRate;
    private final int historySize;

    /**
     * Create oscillation analyzer with specified parameters.
     *
     * @param samplingRate Sampling rate in Hz (e.g., 1000 Hz for 1ms timesteps)
     * @param historySize Number of samples for analysis window (power-of-2 recommended)
     * @throws IllegalArgumentException if parameters invalid
     */
    public OscillationAnalyzer(double samplingRate, int historySize) {
        if (samplingRate <= 0) {
            throw new IllegalArgumentException(
                "samplingRate must be positive: " + samplingRate
            );
        }
        if (historySize <= 0) {
            throw new IllegalArgumentException(
                "historySize must be positive: " + historySize
            );
        }

        this.samplingRate = samplingRate;
        this.historySize = historySize;
        this.fftProcessor = new FFTProcessor(samplingRate);
        this.phaseDetector = new PhaseDetector();
    }

    /**
     * Analyze activation history to compute oscillation metrics.
     *
     * <p>Steps:
     * <ol>
     *   <li>Extract activation history from buffer</li>
     *   <li>Compute average activation across neurons</li>
     *   <li>FFT analysis for dominant frequency and gamma power</li>
     *   <li>Hilbert transform for instantaneous phase</li>
     * </ol>
     *
     * @param activationHistory Circular buffer of activation snapshots
     * @param timestamp Current time (seconds)
     * @return Oscillation metrics
     * @throws IllegalArgumentException if history not full or wrong size
     */
    public OscillationMetrics analyze(CircularBuffer<double[]> activationHistory, double timestamp) {
        if (!activationHistory.isFull()) {
            throw new IllegalArgumentException(
                "activationHistory must be full (size: %d, capacity: %d)"
                .formatted(activationHistory.size(), activationHistory.capacity())
            );
        }
        if (activationHistory.size() != historySize) {
            throw new IllegalArgumentException(
                "activationHistory size mismatch: expected %d, got %d"
                .formatted(historySize, activationHistory.size())
            );
        }

        // Extract activation history
        var history = activationHistory.toDoubleArray2D();

        // Compute average activation across neurons for each timestep
        var avgActivation = computeAverageActivation(history);

        // Remove DC component for better oscillation detection
        avgActivation = removeDCComponent(avgActivation);

        // FFT analysis
        var spectrum = fftProcessor.computePowerSpectrum(avgActivation);
        var dominantFreq = spectrum.findPeakFrequency();

        // Gamma band power (normalized by total power)
        var gammaPower = spectrum.getPowerInBand(
            OscillationMetrics.GAMMA_LOW,
            OscillationMetrics.GAMMA_HIGH
        );
        var totalPower = spectrum.getTotalPower();
        var normalizedGammaPower = totalPower > 0 ? gammaPower / totalPower : 0.0;

        // Instantaneous phase (at most recent timestep)
        var phases = phaseDetector.computeInstantaneousPhase(avgActivation);
        var currentPhase = phases[phases.length - 1];

        return new OscillationMetrics(
            dominantFreq,
            normalizedGammaPower,
            currentPhase,
            timestamp
        );
    }

    /**
     * Analyze raw activation time-series.
     *
     * <p>Alternative method when activation is already averaged scalar series.
     *
     * @param activationTimeSeries Scalar activation values over time
     * @param timestamp Current time
     * @return Oscillation metrics
     */
    public OscillationMetrics analyze(double[] activationTimeSeries, double timestamp) {
        if (activationTimeSeries == null || activationTimeSeries.length == 0) {
            throw new IllegalArgumentException("activationTimeSeries cannot be null or empty");
        }

        // Remove DC component for better oscillation detection
        var zeroMean = removeDCComponent(activationTimeSeries);

        // FFT analysis
        var spectrum = fftProcessor.computePowerSpectrum(zeroMean);
        var dominantFreq = spectrum.findPeakFrequency();

        // Gamma power
        var gammaPower = spectrum.getPowerInBand(
            OscillationMetrics.GAMMA_LOW,
            OscillationMetrics.GAMMA_HIGH
        );
        var totalPower = spectrum.getTotalPower();
        var normalizedGammaPower = totalPower > 0 ? gammaPower / totalPower : 0.0;

        // Instantaneous phase
        var phases = phaseDetector.computeInstantaneousPhase(activationTimeSeries);
        var currentPhase = phases[phases.length - 1];

        return new OscillationMetrics(
            dominantFreq,
            normalizedGammaPower,
            currentPhase,
            timestamp
        );
    }

    /**
     * Compute average activation across neurons for each timestep.
     *
     * <p>Input: activations[timestep][neuron]
     * <p>Output: avgActivation[timestep] = mean across neurons
     *
     * @param activations 2D array [timestep][neuron]
     * @return Average activation per timestep
     */
    private double[] computeAverageActivation(double[][] activations) {
        int timesteps = activations.length;
        var avgActivation = new double[timesteps];

        for (int t = 0; t < timesteps; t++) {
            double sum = 0.0;
            int count = activations[t].length;

            for (double activation : activations[t]) {
                sum += activation;
            }

            avgActivation[t] = sum / count;
        }

        return avgActivation;
    }

    /**
     * Remove DC component (mean) from signal.
     *
     * <p>Subtracts the mean from each sample to center signal around zero.
     * This improves oscillation detection by removing the DC bias that
     * would otherwise dominate the power spectrum at 0 Hz.
     *
     * @param signal Input signal
     * @return Zero-mean signal
     */
    private double[] removeDCComponent(double[] signal) {
        // Calculate mean
        double mean = 0.0;
        for (double value : signal) {
            mean += value;
        }
        mean /= signal.length;

        // Subtract mean from each sample
        var zeroMean = new double[signal.length];
        for (int i = 0; i < signal.length; i++) {
            zeroMean[i] = signal[i] - mean;
        }

        return zeroMean;
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
     * Get history window size.
     *
     * @return Number of samples in analysis window
     */
    public int getHistorySize() {
        return historySize;
    }

    /**
     * Get frequency resolution.
     *
     * @return Frequency resolution in Hz
     */
    public double getFrequencyResolution() {
        return samplingRate / historySize;
    }

    @Override
    public String toString() {
        return "OscillationAnalyzer[samplingRate=%.1f Hz, historySize=%d, freqRes=%.2f Hz]"
            .formatted(samplingRate, historySize, getFrequencyResolution());
    }
}
