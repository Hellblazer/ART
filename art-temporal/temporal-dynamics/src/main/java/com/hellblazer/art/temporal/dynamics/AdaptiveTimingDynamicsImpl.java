package com.hellblazer.art.temporal.dynamics;

import com.hellblazer.art.temporal.core.TimingState;

/**
 * Adaptive timing dynamics implementation for temporal interval learning.
 * Implements spectral timing theory for learning temporal intervals.
 * Based on Grossberg & Schmajuk (1989) and adapted for Kazerounian & Grossberg (2014).
 */
public class AdaptiveTimingDynamicsImpl {

    private final AdaptiveTimingParameters parameters;
    private double[] timingSpectrum;
    private double[] adaptiveWeights;
    private double[] peakTimes;
    private int dimension;
    private double currentTime;

    public AdaptiveTimingDynamicsImpl(AdaptiveTimingParameters parameters, int dimension) {
        this.parameters = parameters;
        this.dimension = dimension;
        this.timingSpectrum = new double[dimension];
        this.adaptiveWeights = new double[dimension];
        this.peakTimes = new double[dimension];
        this.currentTime = 0.0;

        // Initialize peak times with logarithmic spacing
        initializePeakTimes();
    }

    public TimingState evolve(TimingState currentState, double deltaT) {
        var spectrum = currentState.getSpectrum();
        var weights = currentState.getWeights();
        var result = new double[dimension];

        // Update timing spectrum
        for (int i = 0; i < dimension; i++) {
            // Spectral timing dynamics
            double tau = computeTimeConstant(i);
            double activation = computeActivation(i, currentTime);

            // dx_i/dt = (-x_i + f_i(t)) / tau_i
            double derivative = (-spectrum[i] + activation) / tau;

            // Euler integration
            result[i] = spectrum[i] + deltaT * derivative;

            // Ensure non-negative
            result[i] = Math.max(0.0, result[i]);
        }

        // Update adaptive weights if learning
        if (parameters.isLearningEnabled()) {
            updateAdaptiveWeights(result, deltaT);
        }

        // Advance time
        currentTime += deltaT;

        return new TimingState(result, adaptiveWeights.clone(), currentTime);
    }

    public TimingState getState() {
        return new TimingState(timingSpectrum.clone(), adaptiveWeights.clone(), currentTime);
    }

    public void setState(TimingState state) {
        var spectrum = state.getSpectrum();
        var weights = state.getWeights();
        System.arraycopy(spectrum, 0, timingSpectrum, 0, Math.min(spectrum.length, dimension));
        System.arraycopy(weights, 0, adaptiveWeights, 0, Math.min(weights.length, dimension));
        currentTime = state.getCurrentTime();
    }

    /**
     * Initialize peak times with logarithmic spacing.
     */
    private void initializePeakTimes() {
        double minTime = parameters.getMinInterval();
        double maxTime = parameters.getMaxInterval();

        if (dimension == 1) {
            peakTimes[0] = (minTime + maxTime) / 2.0;
        } else {
            double logMin = Math.log(minTime);
            double logMax = Math.log(maxTime);
            double logStep = (logMax - logMin) / (dimension - 1);

            for (int i = 0; i < dimension; i++) {
                peakTimes[i] = Math.exp(logMin + i * logStep);
            }
        }
    }

    /**
     * Compute time constant for spectral component i.
     */
    private double computeTimeConstant(int i) {
        return peakTimes[i] * parameters.getTimeConstantScale();
    }

    /**
     * Compute activation for spectral component i at time t.
     */
    private double computeActivation(int i, double time) {
        double peakTime = peakTimes[i];
        double width = parameters.getSpectralWidth() * peakTime;

        // Gaussian activation centered at peak time
        double diff = time - peakTime;
        return Math.exp(-diff * diff / (2.0 * width * width));
    }

    /**
     * Update adaptive weights based on learning signal.
     */
    private void updateAdaptiveWeights(double[] spectrum, double deltaT) {
        double learningRate = parameters.getLearningRate();

        for (int i = 0; i < dimension; i++) {
            // Hebbian learning: dW_i/dt = η * x_i * (S - W_i)
            double signal = computeLearningSignal(i);
            double derivative = learningRate * spectrum[i] * (signal - adaptiveWeights[i]);

            adaptiveWeights[i] += deltaT * derivative;

            // Bound weights
            adaptiveWeights[i] = Math.max(0.0, Math.min(1.0, adaptiveWeights[i]));
        }
    }

    /**
     * Compute learning signal for component i.
     */
    private double computeLearningSignal(int i) {
        // Learning signal peaks at specific intervals
        double targetTime = parameters.getTargetInterval();
        if (targetTime > 0) {
            double diff = Math.abs(currentTime - targetTime);
            double width = parameters.getSpectralWidth() * targetTime;
            return Math.exp(-diff * diff / (2.0 * width * width));
        }
        return 0.0;
    }

    /**
     * Reset timing to start of interval.
     */
    public void resetTiming() {
        currentTime = 0.0;
        for (int i = 0; i < dimension; i++) {
            timingSpectrum[i] = 0.0;
        }
    }

    /**
     * Start timing from current state.
     */
    public void startTiming() {
        currentTime = 0.0;
    }

    /**
     * Get expected time to peak response.
     */
    public double getExpectedPeakTime() {
        double weightedSum = 0.0;
        double totalWeight = 0.0;

        for (int i = 0; i < dimension; i++) {
            double weight = adaptiveWeights[i] * timingSpectrum[i];
            weightedSum += weight * peakTimes[i];
            totalWeight += weight;
        }

        if (totalWeight > 0) {
            return weightedSum / totalWeight;
        }
        return 0.0;
    }

    /**
     * Compute timing response at current time.
     */
    public double computeTimingResponse() {
        double response = 0.0;
        for (int i = 0; i < dimension; i++) {
            response += adaptiveWeights[i] * timingSpectrum[i];
        }
        return response;
    }

    /**
     * Check if timing has reached threshold.
     */
    public boolean hasReachedThreshold(double threshold) {
        return computeTimingResponse() >= threshold;
    }

    /**
     * Get timing spectrum.
     */
    public double[] getTimingSpectrum() {
        return timingSpectrum.clone();
    }

    /**
     * Get adaptive weights.
     */
    public double[] getAdaptiveWeights() {
        return adaptiveWeights.clone();
    }

    /**
     * Get peak times.
     */
    public double[] getPeakTimes() {
        return peakTimes.clone();
    }

    /**
     * Get current time.
     */
    public double getCurrentTime() {
        return currentTime;
    }

    /**
     * Set target interval for learning.
     */
    public void setTargetInterval(double interval) {
        parameters.setTargetInterval(interval);
    }

    /**
     * Get dimension.
     */
    public int getDimension() {
        return dimension;
    }

    /**
     * Get parameters.
     */
    public AdaptiveTimingParameters getParameters() {
        return parameters;
    }

    /**
     * Compute timing error relative to target.
     */
    public double computeTimingError() {
        double targetTime = parameters.getTargetInterval();
        if (targetTime > 0) {
            double expectedTime = getExpectedPeakTime();
            return Math.abs(expectedTime - targetTime) / targetTime;
        }
        return 0.0;
    }
}