package com.hellblazer.art.cortical.dynamics;

/**
 * Base interface for all neural dynamics models in the cortical architecture.
 * Represents differential equation systems that govern neural activation over time.
 *
 * <p>Implementations include:
 * <ul>
 *   <li>Shunting dynamics (on-center off-surround competition)</li>
 *   <li>Transmitter dynamics (habituation and gating)</li>
 *   <li>Multi-scale temporal dynamics</li>
 * </ul>
 *
 * <p>All dynamics follow continuous-time differential equations that are
 * numerically integrated using methods like Runge-Kutta 4th order (RK4).
 *
 * @author Migrated from art-temporal to art-cortical (Phase 1)
 */
public interface NeuralDynamics {

    /**
     * Update neural state for a single integration time step.
     * Applies the differential equation governing this dynamics model.
     *
     * @param timeStep integration time step (typically 0.01 for stability)
     * @return updated activation values after one time step
     * @throws IllegalArgumentException if timeStep is non-positive
     */
    double[] update(double timeStep);

    /**
     * Reset dynamics to initial state.
     * Clears all internal state and returns activations to baseline.
     * Required for reusing the same dynamics instance across multiple trials.
     */
    void reset();

    /**
     * Get current activation state.
     * Returns a copy to preserve immutability of internal state.
     *
     * @return current activation values (copy of internal state)
     */
    double[] getActivation();

    /**
     * Get the dimensionality of this dynamics system.
     *
     * @return number of neural units governed by this dynamics
     */
    int size();

    /**
     * Check if the dynamics has converged to a stable state.
     * Convergence criteria vary by implementation but typically involve
     * checking if activation changes fall below a threshold.
     *
     * @return true if dynamics has reached stable equilibrium
     */
    default boolean hasConverged() {
        return false; // Default: never converges (continuous dynamics)
    }
}
