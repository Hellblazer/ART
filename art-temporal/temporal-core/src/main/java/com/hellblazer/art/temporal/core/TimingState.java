package com.hellblazer.art.temporal.core;

/**
 * State representation for adaptive timing dynamics.
 */
public class TimingState extends State {

    private final double[] spectrum;
    private final double[] weights;
    private final double currentTime;

    public TimingState(double[] spectrum, double[] weights, double currentTime) {
        this.spectrum = spectrum.clone();
        this.weights = weights.clone();
        this.currentTime = currentTime;
    }

    /**
     * Get timing spectrum components.
     */
    public double[] getSpectrum() {
        return spectrum.clone();
    }

    /**
     * Get adaptive weights.
     */
    public double[] getWeights() {
        return weights.clone();
    }

    /**
     * Get current time.
     */
    public double getCurrentTime() {
        return currentTime;
    }

    @Override
    public State add(State other) {
        if (!(other instanceof TimingState)) {
            throw new IllegalArgumentException("Can only add TimingState to TimingState");
        }

        var otherTiming = (TimingState) other;
        if (spectrum.length != otherTiming.spectrum.length) {
            throw new IllegalArgumentException("Dimension mismatch");
        }

        var result = new double[spectrum.length];
        var resultWeights = new double[weights.length];
        for (int i = 0; i < spectrum.length; i++) {
            result[i] = spectrum[i] + otherTiming.spectrum[i];
            resultWeights[i] = weights[i] + otherTiming.weights[i];
        }

        double newTime = currentTime + otherTiming.currentTime;
        return new TimingState(result, resultWeights, newTime);
    }

    @Override
    public State scale(double scalar) {
        var result = new double[spectrum.length];
        var resultWeights = new double[weights.length];
        for (int i = 0; i < spectrum.length; i++) {
            result[i] = spectrum[i] * scalar;
            resultWeights[i] = weights[i] * scalar;
        }
        return new TimingState(result, resultWeights, currentTime * scalar);
    }

    @Override
    public double distance(State other) {
        if (!(other instanceof TimingState)) {
            throw new IllegalArgumentException("Can only compute distance to TimingState");
        }
        var otherTiming = (TimingState) other;
        double sum = 0.0;
        for (int i = 0; i < spectrum.length; i++) {
            double diff = spectrum[i] - otherTiming.spectrum[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    @Override
    public int dimension() {
        return spectrum.length;
    }

    @Override
    public State copy() {
        return new TimingState(spectrum.clone(), weights.clone(), currentTime);
    }

    @Override
    public double[] toArray() {
        return spectrum.clone();
    }

    @Override
    public State fromArray(double[] values) {
        return new TimingState(values, weights.clone(), currentTime);
    }

    /**
     * Check if the state is valid.
     */
    public boolean isValid() {
        if (spectrum == null || weights == null) {
            return false;
        }
        if (spectrum.length != weights.length) {
            return false;
        }
        if (currentTime < 0) {
            return false;
        }
        for (double s : spectrum) {
            if (!Double.isFinite(s) || s < 0) {
                return false;
            }
        }
        for (double w : weights) {
            if (!Double.isFinite(w) || w < 0 || w > 1) {
                return false;
            }
        }
        return true;
    }

    /**
     * Compute total timing response.
     */
    public double computeResponse() {
        double response = 0.0;
        for (int i = 0; i < spectrum.length; i++) {
            response += weights[i] * spectrum[i];
        }
        return response;
    }

    /**
     * Get peak timing component.
     */
    public int getPeakComponent() {
        double maxValue = spectrum[0];
        int maxIndex = 0;
        for (int i = 1; i < spectrum.length; i++) {
            if (spectrum[i] > maxValue) {
                maxValue = spectrum[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}