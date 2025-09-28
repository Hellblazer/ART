package com.hellblazer.art.temporal.dynamics;

import com.hellblazer.art.temporal.core.*;

/**
 * Fourth-order Runge-Kutta integrator for high accuracy.
 *
 * RK4 formula:
 * k1 = f(t, y)
 * k2 = f(t + h/2, y + h*k1/2)
 * k3 = f(t + h/2, y + h*k2/2)
 * k4 = f(t + h, y + h*k3)
 * y_new = y + h*(k1 + 2*k2 + 2*k3 + k4)/6
 *
 * Local error is O(h^5), global error is O(h^4).
 */
public class RungeKutta4Integrator<S extends State, P extends Parameters>
        extends NumericalIntegrator<S, P> {

    public RungeKutta4Integrator(DynamicalSystem<S, P> system) {
        super(system);
    }

    @Override
    public S integrate(S initialState, P parameters,
                      double startTime, double endTime,
                      IntegrationCallback<S> callback) {
        var currentState = initialState;
        var currentTime = startTime;
        var dt = stepSizeController.getInitialStepSize();

        while (currentTime < endTime) {
            dt = Math.min(dt, endTime - currentTime);

            var result = step(currentState, parameters, currentTime, dt);
            currentState = result.state();
            currentTime += dt;

            if (callback != null) {
                callback.onStep(currentState, currentTime, dt);
            }

            if (convergenceMonitor.hasConverged(currentState, 1e-6)) {
                if (callback != null) {
                    callback.onConvergence(currentState, currentTime);
                }
                break;
            }
        }

        return currentState;
    }

    @Override
    @SuppressWarnings("unchecked")
    protected IntegrationResult<S> step(S state, P parameters, double time, double dt) {
        // RK4 stages
        var k1 = system.computeDerivative(state, parameters, time);

        var state2 = (S) state.add(k1.scale(dt * 0.5));
        var k2 = system.computeDerivative(state2, parameters, time + dt * 0.5);

        var state3 = (S) state.add(k2.scale(dt * 0.5));
        var k3 = system.computeDerivative(state3, parameters, time + dt * 0.5);

        var state4 = (S) state.add(k3.scale(dt));
        var k4 = system.computeDerivative(state4, parameters, time + dt);

        // Combine stages: y_new = y + h*(k1 + 2*k2 + 2*k3 + k4)/6
        var increment = (S) k1.add(k2.scale(2.0))
                              .add(k3.scale(2.0))
                              .add(k4);
        increment = (S) increment.scale(dt / 6.0);

        var newState = (S) state.add(increment);

        // Estimate error using Richardson extrapolation
        var error = estimateError(state, newState, parameters, time, dt);

        return new IntegrationResult<>(newState, error);
    }

    /**
     * Estimate local truncation error using step doubling.
     */
    @SuppressWarnings("unchecked")
    private double estimateError(S state, S fullStep, P parameters, double time, double dt) {
        // Take two half steps
        var halfDt = dt * 0.5;
        var halfStep1 = step(state, parameters, time, halfDt);
        var halfStep2 = step(halfStep1.state(), parameters, time + halfDt, halfDt);

        // Error estimate: difference between full step and two half steps
        // For RK4, the error is approximately (fullStep - halfStep2) / 15
        var error = fullStep.distance(halfStep2.state()) / 15.0;

        return error;
    }

    /**
     * Create an adaptive RK4 integrator with automatic step size control.
     */
    public static <S extends State, P extends Parameters>
    RungeKutta4Integrator<S, P> adaptive(DynamicalSystem<S, P> system) {
        return new AdaptiveRK4Integrator<>(system);
    }

    /**
     * Adaptive variant with embedded error estimation.
     */
    private static class AdaptiveRK4Integrator<S extends State, P extends Parameters>
            extends RungeKutta4Integrator<S, P> {

        private static final double SAFETY_FACTOR = 0.9;
        private static final double MAX_FACTOR = 4.0;
        private static final double MIN_FACTOR = 0.25;

        public AdaptiveRK4Integrator(DynamicalSystem<S, P> system) {
            super(system);
        }

        @Override
        public S integrate(S initialState, P parameters,
                          double startTime, double endTime,
                          IntegrationCallback<S> callback) {
            // Use adaptive integration with default tolerance
            return integrateAdaptive(initialState, parameters, startTime, endTime,
                                    1e-6, callback);
        }
    }
}