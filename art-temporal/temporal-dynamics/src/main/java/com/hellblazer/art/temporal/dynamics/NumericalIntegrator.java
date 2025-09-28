package com.hellblazer.art.temporal.dynamics;

import com.hellblazer.art.temporal.core.*;

/**
 * Abstract base class for numerical integration methods.
 * Provides common functionality and defines the integration interface.
 *
 * @param <S> State type
 * @param <P> Parameter type
 */
public abstract class NumericalIntegrator<S extends State, P extends Parameters> {

    protected final DynamicalSystem<S, P> system;
    protected final ConvergenceMonitor<S> convergenceMonitor;
    protected final StepSizeController stepSizeController;

    protected NumericalIntegrator(DynamicalSystem<S, P> system) {
        this.system = system;
        this.convergenceMonitor = new ConvergenceMonitor<>();
        this.stepSizeController = new StepSizeController(system.getTimeScale());
    }

    /**
     * Integrate the system from initial state to final time.
     *
     * @param initialState Initial state
     * @param parameters System parameters
     * @param startTime Start time
     * @param endTime End time
     * @param callback Optional callback for intermediate states
     * @return Final state
     */
    public abstract S integrate(S initialState, P parameters,
                                double startTime, double endTime,
                                IntegrationCallback<S> callback);

    /**
     * Single integration step.
     *
     * @param state Current state
     * @param parameters System parameters
     * @param time Current time
     * @param dt Time step
     * @return New state and estimated error
     */
    protected abstract IntegrationResult<S> step(S state, P parameters, double time, double dt);

    /**
     * Adaptive integration with automatic step size control.
     */
    public S integrateAdaptive(S initialState, P parameters,
                               double startTime, double endTime,
                               double tolerance,
                               IntegrationCallback<S> callback) {
        var currentState = initialState;
        var currentTime = startTime;
        var dt = stepSizeController.getInitialStepSize();

        while (currentTime < endTime) {
            // Ensure we don't overshoot
            dt = Math.min(dt, endTime - currentTime);

            // Attempt step
            var result = step(currentState, parameters, currentTime, dt);

            // Check error and adjust step size
            if (result.error() < tolerance) {
                // Accept step
                currentState = result.state();
                currentTime += dt;

                // Check convergence
                if (convergenceMonitor.hasConverged(currentState, tolerance)) {
                    if (callback != null) {
                        callback.onConvergence(currentState, currentTime);
                    }
                    break;
                }

                // Callback for accepted step
                if (callback != null) {
                    callback.onStep(currentState, currentTime, dt);
                }

                // Increase step size if error is very small
                if (result.error() < tolerance * 0.1) {
                    dt = stepSizeController.increaseStepSize(dt);
                }
            } else {
                // Reject step and reduce step size
                dt = stepSizeController.decreaseStepSize(dt, result.error() / tolerance);

                if (dt < stepSizeController.getMinStepSize()) {
                    throw new IntegrationException(
                        "Step size became too small at t=" + currentTime +
                        ", error=" + result.error());
                }
            }
        }

        return currentState;
    }

    /**
     * Result of an integration step.
     */
    public record IntegrationResult<S extends State>(S state, double error) {}

    /**
     * Callback interface for monitoring integration progress.
     */
    public interface IntegrationCallback<S extends State> {
        default void onStep(S state, double time, double dt) {}
        default void onConvergence(S state, double time) {}
        default void onError(String message, double time) {}
    }

    /**
     * Exception thrown when integration fails.
     */
    public static class IntegrationException extends RuntimeException {
        public IntegrationException(String message) {
            super(message);
        }
    }
}