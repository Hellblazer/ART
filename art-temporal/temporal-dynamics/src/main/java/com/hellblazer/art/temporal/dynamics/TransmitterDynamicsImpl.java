package com.hellblazer.art.temporal.dynamics;

import com.hellblazer.art.temporal.core.TransmitterState;

/**
 * Transmitter habituation dynamics implementation for temporal processing.
 * Implements: dZ_i/dt = ε(1 - Z_i) - Z_i(λ * S_i + μ * S_i²)
 * Based on Kazerounian & Grossberg (2014).
 */
public class TransmitterDynamicsImpl {

    private final TransmitterParameters parameters;
    private double[] transmitters;
    private double[] signals;
    private int dimension;

    public TransmitterDynamicsImpl(TransmitterParameters parameters, int dimension) {
        this.parameters = parameters;
        this.dimension = dimension;
        this.transmitters = new double[dimension];
        this.signals = new double[dimension];

        // Initialize to baseline
        for (int i = 0; i < dimension; i++) {
            transmitters[i] = parameters.getBaselineLevel();
        }
    }

    public TransmitterState evolve(TransmitterState currentState, double deltaT) {
        var current = currentState.getLevels();
        var result = new double[dimension];

        for (int i = 0; i < dimension; i++) {
            // dZ_i/dt = ε(1 - Z_i) - Z_i(λ * S_i + μ * S_i²)
            double recovery = parameters.getRecoveryRate();
            double linearDepletion = parameters.getLinearDepletionRate();
            double quadraticDepletion = parameters.getQuadraticDepletionRate();
            double signal = signals[i];

            // Habituation dynamics
            double derivative = recovery * (1.0 - current[i]) -
                               current[i] * (linearDepletion * signal +
                                           quadraticDepletion * signal * signal);

            // Euler integration
            result[i] = current[i] + deltaT * derivative;

            // Ensure bounds [0, 1]
            result[i] = Math.max(0.0, Math.min(1.0, result[i]));
        }

        return new TransmitterState(result, signals, null);
    }

    public TransmitterState getState() {
        return new TransmitterState(transmitters.clone(), signals.clone(), null);
    }

    public void setState(TransmitterState state) {
        var levels = state.getLevels();
        System.arraycopy(levels, 0, transmitters, 0, Math.min(levels.length, dimension));
    }

    /**
     * Set signal inputs that deplete transmitters.
     */
    public void setSignals(double[] inputSignals) {
        System.arraycopy(inputSignals, 0, signals, 0, Math.min(inputSignals.length, dimension));
    }

    /**
     * Update transmitters based on current signals.
     */
    public void update(double deltaT) {
        var currentState = getState();
        var newState = evolve(currentState, deltaT);
        setState(newState);
    }

    /**
     * Apply habituation based on activation pattern.
     */
    public void habituate(double[] activations, double deltaT) {
        // Convert activations to signals
        for (int i = 0; i < Math.min(activations.length, dimension); i++) {
            signals[i] = Math.max(0, activations[i]);  // Rectify
        }

        // Update transmitter levels
        update(deltaT);
    }

    /**
     * Compute gated output: activation * transmitter.
     */
    public double[] computeGatedOutput(double[] activations) {
        double[] output = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            output[i] = activations[i] * transmitters[i];
        }
        return output;
    }

    /**
     * Reset transmitters to baseline.
     */
    public void reset() {
        for (int i = 0; i < dimension; i++) {
            transmitters[i] = parameters.getBaselineLevel();
            signals[i] = 0.0;
        }
    }

    /**
     * Partial reset with decay factor.
     */
    public void partialReset(double factor) {
        double baseline = parameters.getBaselineLevel();
        for (int i = 0; i < dimension; i++) {
            transmitters[i] = factor * baseline + (1.0 - factor) * transmitters[i];
            signals[i] *= (1.0 - factor);
        }
    }

    /**
     * Get current transmitter levels.
     */
    public double[] getTransmitterLevels() {
        return transmitters.clone();
    }

    /**
     * Get average transmitter level.
     */
    public double getAverageLevel() {
        double sum = 0.0;
        for (double level : transmitters) {
            sum += level;
        }
        return sum / dimension;
    }

    /**
     * Get depletion amount for each unit.
     */
    public double[] getDepletions() {
        double[] depletions = new double[dimension];
        double baseline = parameters.getBaselineLevel();
        for (int i = 0; i < dimension; i++) {
            depletions[i] = baseline - transmitters[i];
        }
        return depletions;
    }

    /**
     * Check if transmitters have recovered.
     */
    public boolean hasRecovered(double threshold) {
        for (double level : transmitters) {
            if (level < threshold) {
                return false;
            }
        }
        return true;
    }

    /**
     * Compute recovery time constant at current state.
     */
    public double computeRecoveryTimeConstant() {
        double avgSignal = 0.0;
        for (double s : signals) {
            avgSignal += s;
        }
        avgSignal /= dimension;

        double recovery = parameters.getRecoveryRate();
        double depletion = parameters.getLinearDepletionRate() * avgSignal +
                          parameters.getQuadraticDepletionRate() * avgSignal * avgSignal;

        if (recovery + depletion > 0) {
            return 1.0 / (recovery + depletion);
        }
        return Double.POSITIVE_INFINITY;
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
    public TransmitterParameters getParameters() {
        return parameters;
    }
}