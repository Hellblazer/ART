package com.hellblazer.art.temporal.core;

import java.util.Map;

/**
 * Core interface for dynamical systems in Temporal ART.
 * Represents any system described by differential equations.
 *
 * Based on Kazerounian & Grossberg (2014) mathematical foundations.
 *
 * @param <S> State type (typically StateVector)
 * @param <P> Parameter type (typically ParameterBundle)
 */
public sealed interface DynamicalSystem<S extends State, P extends Parameters>
    permits ShuntingDynamics, TransmitterDynamics, MaskingFieldDynamics, InstarLearningDynamics {

    /**
     * Compute the derivative dS/dt at the current state.
     * This is the right-hand side of the differential equation.
     *
     * @param state Current state vector
     * @param parameters System parameters
     * @param time Current time
     * @return Derivative of state with respect to time
     */
    S computeDerivative(S state, P parameters, double time);

    /**
     * Advance the system by one time step using the specified integration method.
     * Default implementation uses Euler method.
     *
     * @param state Current state
     * @param parameters System parameters
     * @param time Current time
     * @param dt Time step
     * @return New state after time step
     */
    default S step(S state, P parameters, double time, double dt) {
        var derivative = computeDerivative(state, parameters, time);
        @SuppressWarnings("unchecked")
        S scaledDerivative = (S) derivative.scale(dt);
        @SuppressWarnings("unchecked")
        S newState = (S) state.add(scaledDerivative);
        return newState;
    }

    /**
     * Get the Jacobian matrix for stability analysis.
     * Required for adaptive step sizing and stability monitoring.
     *
     * @param state Current state
     * @param parameters System parameters
     * @param time Current time
     * @return Jacobian matrix ∂f/∂x
     */
    Matrix getJacobian(S state, P parameters, double time);

    /**
     * Compute equilibrium state for given parameters.
     * Returns null if no equilibrium exists or cannot be computed analytically.
     *
     * @param parameters System parameters
     * @param inputs External inputs (if any)
     * @return Equilibrium state or null
     */
    S computeEquilibrium(P parameters, Map<String, Double> inputs);

    /**
     * Check if the system has reached steady state.
     *
     * @param state Current state
     * @param previousState Previous state
     * @param tolerance Convergence tolerance
     * @return true if steady state reached
     */
    default boolean hasConverged(S state, S previousState, double tolerance) {
        return state.distance(previousState) < tolerance;
    }

    /**
     * Get the characteristic time scale of this system.
     * Used for multi-scale time stepping.
     *
     * @return Characteristic time in milliseconds
     */
    TimeScale getTimeScale();

    /**
     * Validate that parameters are within physical/mathematical bounds.
     *
     * @param parameters Parameters to validate
     * @throws IllegalArgumentException if parameters are invalid
     */
    void validateParameters(P parameters);

    /**
     * Get stability properties at current state.
     *
     * @param state Current state
     * @param parameters System parameters
     * @return Stability analysis result
     */
    default StabilityAnalysis analyzeStability(S state, P parameters, double time) {
        var jacobian = getJacobian(state, parameters, time);
        return StabilityAnalysis.fromJacobian(jacobian);
    }

    /**
     * Time scale enumeration for multi-scale dynamics.
     * Based on paper's time scale hierarchy.
     */
    enum TimeScale {
        FAST(10, 100),          // Working memory: 10-100 ms
        MEDIUM(50, 500),        // Masking field: 50-500 ms
        SLOW(500, 5000),        // Transmitter gates: 500-5000 ms
        VERY_SLOW(1000, 10000); // Weight adaptation: 1000-10000 ms

        private final double minMs;
        private final double maxMs;

        TimeScale(double minMs, double maxMs) {
            this.minMs = minMs;
            this.maxMs = maxMs;
        }

        public double getMinMillis() { return minMs; }
        public double getMaxMillis() { return maxMs; }
        public double getTypicalMillis() { return (minMs + maxMs) / 2; }
    }
}