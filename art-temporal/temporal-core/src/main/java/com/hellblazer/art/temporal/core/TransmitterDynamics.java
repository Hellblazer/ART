package com.hellblazer.art.temporal.core;

import java.util.Map;

/**
 * Transmitter gate dynamics from Kazerounian & Grossberg (2014).
 *
 * Equation: dZ_i/dt = ε(1 - Z_i) - Z_i(λ * S_i + μ * S_i²)
 *
 * Where:
 * - Z_i: Transmitter level for unit i (0 ≤ Z_i ≤ 1)
 * - ε: Recovery rate (0.005 - very slow)
 * - λ: Linear depletion rate (0.1)
 * - μ: Quadratic depletion rate (0.05)
 * - S_i: Input signal strength
 *
 * Equilibrium: Z_eq = ε/(ε + λS + μS²)
 *
 * This creates the primacy gradient effect where early items
 * have more transmitter available than later items.
 */
public final class TransmitterDynamics implements DynamicalSystem<TransmitterState, TransmitterParameters> {

    @Override
    public TransmitterState computeDerivative(TransmitterState state, TransmitterParameters params, double time) {
        var transmitters = state.getTransmitterLevels();
        var signals = state.getPresynapticSignals();
        var derivatives = new double[transmitters.length];

        for (int i = 0; i < transmitters.length; i++) {
            var zi = transmitters[i];
            var si = signals[i];

            // Recovery term: ε(1 - Z_i)
            var recovery = params.getEpsilon() * (1.0 - zi);

            // Depletion term: -Z_i(λ * S_i + μ * S_i²)
            var linearDepletion = params.getLambda() * si;
            var quadraticDepletion = params.getMu() * si * si;
            var depletion = -zi * (linearDepletion + quadraticDepletion);

            derivatives[i] = recovery + depletion;
        }

        // DEBUG: Print derivatives to verify correctness
        // System.err.println("TransmitterDynamics derivatives: " + java.util.Arrays.toString(derivatives));

        // Return state with derivatives in the transmitter levels array
        // The integrator will handle the actual integration
        return new TransmitterState(derivatives, signals, state.getDepletionHistory());
    }

    @Override
    public Matrix getJacobian(TransmitterState state, TransmitterParameters params, double time) {
        var n = state.dimension();
        var jacobian = new Matrix(n, n);
        var transmitters = state.getTransmitterLevels();
        var signals = state.getPresynapticSignals();

        // Diagonal Jacobian for independent transmitter dynamics
        for (int i = 0; i < n; i++) {
            var si = signals[i];
            var zi = transmitters[i];

            // d/dZ_i of [ε(1 - Z_i) - Z_i(λ * S_i + μ * S_i²)]
            // = -ε - (λ * S_i + μ * S_i²)
            var diagonal = -params.getEpsilon() -
                          (params.getLambda() * si + params.getMu() * si * si);

            jacobian.set(i, i, diagonal);
        }

        return jacobian;
    }

    @Override
    public TransmitterState computeEquilibrium(TransmitterParameters params, Map<String, Double> inputs) {
        // Extract dimension from inputs or use default
        var dimension = inputs.getOrDefault("dimension", 100.0).intValue();
        var equilibrium = new double[dimension];
        var signals = new double[dimension];

        for (int i = 0; i < dimension; i++) {
            signals[i] = inputs.getOrDefault("S_" + i, 0.5);

            // Equilibrium: Z_eq = ε / (ε + λS + μS²)
            equilibrium[i] = params.computeEquilibrium(signals[i]);
        }

        return new TransmitterState(equilibrium, signals, new double[dimension]);
    }

    @Override
    public TimeScale getTimeScale() {
        return TimeScale.SLOW; // Transmitter dynamics at 500-5000ms
    }

    @Override
    public void validateParameters(TransmitterParameters parameters) {
        parameters.validate();

        // Additional paper-specific validation
        if (parameters.getEpsilon() > 0.01) {
            System.err.println("Warning: Recovery rate ε > 0.01 is faster than paper specification (0.005)");
        }

        if (parameters.getLambda() < 0.05 || parameters.getLambda() > 0.2) {
            System.err.println("Warning: Linear depletion λ outside typical range [0.05, 0.2]");
        }

        if (parameters.getMu() < 0.01 || parameters.getMu() > 0.1) {
            System.err.println("Warning: Quadratic depletion μ outside typical range [0.01, 0.1]");
        }
    }

    /**
     * Compute the strength of primacy gradient in transmitter distribution.
     * Returns value in [-1, 1] where positive means early items have more transmitter.
     */
    public double computePrimacyGradientStrength(TransmitterState state) {
        var transmitters = state.getTransmitterLevels();
        if (transmitters.length < 2) return 0.0;

        var midpoint = transmitters.length / 2;
        var earlyAvg = 0.0;
        var lateAvg = 0.0;

        for (int i = 0; i < midpoint; i++) {
            earlyAvg += transmitters[i];
        }
        earlyAvg /= midpoint;

        for (int i = midpoint; i < transmitters.length; i++) {
            lateAvg += transmitters[i];
        }
        lateAvg /= (transmitters.length - midpoint);

        return (earlyAvg - lateAvg) / (earlyAvg + lateAvg + 1e-10);
    }

    /**
     * Override step to add proper clamping for transmitter levels.
     */
    @Override
    public TransmitterState step(TransmitterState state, TransmitterParameters parameters, double time, double dt) {
        var derivative = computeDerivative(state, parameters, time);
        @SuppressWarnings("unchecked")
        var scaledDerivative = (TransmitterState) derivative.scale(dt);
        @SuppressWarnings("unchecked")
        var newState = (TransmitterState) state.add(scaledDerivative);

        // Clamp transmitter levels to [0, 1]
        var levels = newState.getTransmitterLevels();
        var clamped = new double[levels.length];
        for (int i = 0; i < levels.length; i++) {
            clamped[i] = Math.max(0.0, Math.min(1.0, levels[i]));
        }

        return new TransmitterState(clamped, newState.getPresynapticSignals(), newState.getDepletionHistory());
    }

    /**
     * Check if transmitter system should trigger reset based on depletion.
     */
    public boolean shouldReset(TransmitterState state, TransmitterParameters params) {
        return state.hasDepletedTransmitters(params.getDepletionThreshold());
    }
}