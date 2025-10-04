package com.hellblazer.art.laminar.canonical;

import com.hellblazer.art.temporal.core.DynamicalSystem;
import com.hellblazer.art.temporal.core.Parameters;
import com.hellblazer.art.temporal.core.State;

/**
 * Simple RK4 integrator without adaptive error estimation.
 * Used for pathway temporal dynamics where we want a single integration step
 * without recursive error estimation that can cause stack overflow.
 *
 * @param <S> State type
 * @param <P> Parameters type
 * @author Hal Hildebrand
 */
public class SimpleIntegrator<S extends State, P extends Parameters> {

    private final DynamicalSystem<S, P> system;

    public SimpleIntegrator(DynamicalSystem<S, P> system) {
        this.system = system;
    }

    /**
     * Perform a single RK4 integration step without error estimation.
     *
     * @param state current state
     * @param parameters system parameters
     * @param time current time
     * @param dt time step
     * @return new state
     */
    @SuppressWarnings("unchecked")
    public S step(S state, P parameters, double time, double dt) {
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

        // Clamp transmitter states to [0, 1] if needed
        if (newState instanceof com.hellblazer.art.temporal.core.TransmitterState ts) {
            var levels = ts.getTransmitterLevels();
            var clamped = new double[levels.length];
            for (int i = 0; i < levels.length; i++) {
                clamped[i] = Math.max(0.0, Math.min(1.0, levels[i]));
            }
            newState = (S) new com.hellblazer.art.temporal.core.TransmitterState(
                clamped,
                ts.getPresynapticSignals(),
                ts.getDepletionHistory()
            );
        }

        return newState;
    }

    /**
     * Integrate over a time interval.
     *
     * @param initialState initial state
     * @param parameters system parameters
     * @param startTime start time
     * @param endTime end time
     * @return final state
     */
    public S integrate(S initialState, P parameters, double startTime, double endTime) {
        var currentState = initialState;
        var currentTime = startTime;
        var dt = (endTime - startTime); // Single step for short intervals

        // For very short intervals, use a single step
        if (dt <= 0.1) {
            return step(currentState, parameters, currentTime, dt);
        }

        // For longer intervals, use multiple steps
        var numSteps = (int) Math.ceil(dt / 0.01);
        dt = (endTime - startTime) / numSteps;

        for (int i = 0; i < numSteps; i++) {
            currentState = step(currentState, parameters, currentTime, dt);
            currentTime += dt;
        }

        return currentState;
    }
}