package com.hellblazer.art.temporal.dynamics;

import com.hellblazer.art.temporal.core.ActivationState;
import com.hellblazer.art.temporal.core.TransmitterState;
import com.hellblazer.art.temporal.core.TimingState;

/**
 * Multi-scale dynamics integrating shunting, transmitter, and timing dynamics.
 * Implements the hierarchical time scales from Kazerounian & Grossberg (2014):
 * - Fast: Shunting dynamics (10-100ms)
 * - Medium: Transmitter habituation (100-1000ms)
 * - Slow: Adaptive timing (1-10s)
 */
public class MultiScaleDynamics {

    private final ShuntingDynamicsImpl shuntingDynamics;
    private final TransmitterDynamicsImpl transmitterDynamics;
    private final AdaptiveTimingDynamicsImpl timingDynamics;
    private final MultiScaleParameters parameters;

    private double currentTime;
    private int updateCount;

    public MultiScaleDynamics(MultiScaleParameters parameters) {
        this.parameters = parameters;

        int dimension = parameters.getDimension();
        this.shuntingDynamics = new ShuntingDynamicsImpl(
            parameters.getShuntingParameters(), dimension
        );
        this.transmitterDynamics = new TransmitterDynamicsImpl(
            parameters.getTransmitterParameters(), dimension
        );
        this.timingDynamics = new AdaptiveTimingDynamicsImpl(
            parameters.getTimingParameters(), dimension
        );

        this.currentTime = 0.0;
        this.updateCount = 0;
    }

    /**
     * Update all dynamics with appropriate time scales.
     */
    public void update(double[] input, double deltaT) {
        // Fast time scale: shunting dynamics
        updateShunting(input, deltaT);

        // Medium time scale: transmitter habituation (every N updates)
        if (updateCount % parameters.getTransmitterUpdateRatio() == 0) {
            updateTransmitters(deltaT * parameters.getTransmitterUpdateRatio());
        }

        // Slow time scale: adaptive timing (every M updates)
        if (updateCount % parameters.getTimingUpdateRatio() == 0) {
            updateTiming(deltaT * parameters.getTimingUpdateRatio());
        }

        currentTime += deltaT;
        updateCount++;
    }

    /**
     * Update shunting dynamics (fast time scale).
     */
    private void updateShunting(double[] input, double deltaT) {
        // Set external input
        shuntingDynamics.setExcitatoryInput(input);

        // Apply competitive dynamics
        var currentState = shuntingDynamics.getState();
        var evolved = shuntingDynamics.evolve(currentState, deltaT);
        shuntingDynamics.setState(evolved);
    }

    /**
     * Update transmitter dynamics (medium time scale).
     */
    private void updateTransmitters(double deltaT) {
        // Get activations from shunting dynamics
        var activations = shuntingDynamics.getActivations();

        // Habituate based on activity
        transmitterDynamics.habituate(activations, deltaT);
    }

    /**
     * Update timing dynamics (slow time scale).
     */
    private void updateTiming(double deltaT) {
        var currentState = timingDynamics.getState();
        var evolved = timingDynamics.evolve(currentState, deltaT);
        timingDynamics.setState(evolved);
    }

    /**
     * Get gated output: activation * transmitter * timing.
     */
    public double[] getGatedOutput() {
        var activations = shuntingDynamics.getActivations();
        var gatedByTransmitter = transmitterDynamics.computeGatedOutput(activations);

        // Further gate by timing response if enabled
        if (parameters.isTimingGatingEnabled()) {
            double timingGate = timingDynamics.computeTimingResponse();
            for (int i = 0; i < gatedByTransmitter.length; i++) {
                gatedByTransmitter[i] *= timingGate;
            }
        }

        return gatedByTransmitter;
    }

    /**
     * Reset all dynamics to initial states.
     */
    public void reset() {
        shuntingDynamics.reset();
        transmitterDynamics.reset();
        timingDynamics.resetTiming();
        currentTime = 0.0;
        updateCount = 0;
    }

    /**
     * Partial reset for new sequence.
     */
    public void partialReset() {
        // Keep some activation memory
        var activations = shuntingDynamics.getActivations();
        for (int i = 0; i < activations.length; i++) {
            activations[i] *= parameters.getResetDecayFactor();
        }
        shuntingDynamics.setState(new ActivationState(activations));

        // Partial transmitter reset
        transmitterDynamics.partialReset(parameters.getResetDecayFactor());

        // Reset timing for new interval
        timingDynamics.resetTiming();
    }

    /**
     * Get current state snapshot.
     */
    public MultiScaleState getState() {
        return new MultiScaleState(
            shuntingDynamics.getState(),
            transmitterDynamics.getState(),
            timingDynamics.getState(),
            currentTime
        );
    }

    /**
     * Set state from snapshot.
     */
    public void setState(MultiScaleState state) {
        shuntingDynamics.setState(state.activationState());
        transmitterDynamics.setState(state.transmitterState());
        timingDynamics.setState(state.timingState());
        currentTime = state.time();
    }

    /**
     * Check convergence across all time scales.
     */
    public boolean hasConverged(double tolerance) {
        // Check fast dynamics
        if (!shuntingDynamics.hasConverged(tolerance)) {
            return false;
        }

        // Check medium dynamics
        if (!transmitterDynamics.hasRecovered(1.0 - tolerance)) {
            return false;
        }

        // Check slow dynamics
        double timingError = timingDynamics.computeTimingError();
        return timingError < tolerance;
    }

    /**
     * Get energy/Lyapunov function for stability analysis.
     */
    public double computeEnergy() {
        double shuntingEnergy = shuntingDynamics.computeEnergy();
        double transmitterEnergy = computeTransmitterEnergy();
        double timingEnergy = computeTimingEnergy();

        return shuntingEnergy + transmitterEnergy + timingEnergy;
    }

    private double computeTransmitterEnergy() {
        var levels = transmitterDynamics.getTransmitterLevels();
        double energy = 0.0;
        for (double level : levels) {
            // Energy increases with depletion
            energy += (1.0 - level) * (1.0 - level);
        }
        return energy;
    }

    private double computeTimingEnergy() {
        var spectrum = timingDynamics.getTimingSpectrum();
        double energy = 0.0;
        for (double component : spectrum) {
            energy += component * component;
        }
        return energy;
    }

    /**
     * Compute statistics for monitoring.
     */
    public DynamicsStatistics computeStatistics() {
        var activations = shuntingDynamics.getActivations();
        var transmitters = transmitterDynamics.getTransmitterLevels();
        var timingResponse = timingDynamics.computeTimingResponse();

        double avgActivation = average(activations);
        double avgTransmitter = average(transmitters);
        double maxActivation = max(activations);
        double minTransmitter = min(transmitters);

        return new DynamicsStatistics(
            avgActivation,
            avgTransmitter,
            timingResponse,
            maxActivation,
            minTransmitter,
            computeEnergy(),
            currentTime
        );
    }

    private double average(double[] array) {
        double sum = 0.0;
        for (double value : array) {
            sum += value;
        }
        return sum / array.length;
    }

    private double max(double[] array) {
        double max = array[0];
        for (double value : array) {
            max = Math.max(max, value);
        }
        return max;
    }

    private double min(double[] array) {
        double min = array[0];
        for (double value : array) {
            min = Math.min(min, value);
        }
        return min;
    }

    // Getters
    public ShuntingDynamicsImpl getShuntingDynamics() {
        return shuntingDynamics;
    }

    public TransmitterDynamicsImpl getTransmitterDynamics() {
        return transmitterDynamics;
    }

    public AdaptiveTimingDynamicsImpl getTimingDynamics() {
        return timingDynamics;
    }

    public double getCurrentTime() {
        return currentTime;
    }

    public MultiScaleParameters getParameters() {
        return parameters;
    }

    /**
     * Record for multi-scale state snapshot.
     */
    public record MultiScaleState(
        ActivationState activationState,
        TransmitterState transmitterState,
        TimingState timingState,
        double time
    ) {}

    /**
     * Record for dynamics statistics.
     */
    public record DynamicsStatistics(
        double averageActivation,
        double averageTransmitter,
        double timingResponse,
        double maxActivation,
        double minTransmitter,
        double totalEnergy,
        double time
    ) {}
}