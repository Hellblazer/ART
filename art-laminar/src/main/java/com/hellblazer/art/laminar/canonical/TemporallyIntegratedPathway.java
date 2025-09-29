package com.hellblazer.art.laminar.canonical;

import com.hellblazer.art.laminar.core.Pathway;
import com.hellblazer.art.temporal.core.ShuntingState;
import com.hellblazer.art.temporal.core.TransmitterState;

/**
 * Interface for pathways that integrate temporal dynamics (shunting and transmitter gating)
 * into laminar circuit processing.
 *
 * Extends base Pathway with temporal state management and dynamics integration.
 * This follows the decorator pattern, allowing existing pathways to be enhanced
 * with sophisticated temporal processing without modifying their core implementation.
 *
 * @author Hal Hildebrand
 */
public interface TemporallyIntegratedPathway extends Pathway {

    /**
     * Get the current shunting dynamics state.
     * Represents the activation state following the shunting equation:
     * dX_i/dt = -A_i * X_i + (B - X_i) * S_i - X_i * Σ(j≠i) I_ij
     *
     * @return current shunting state
     */
    ShuntingState getShuntingState();

    /**
     * Get the current transmitter dynamics state.
     * Represents neurotransmitter gating following the equation:
     * dZ_i/dt = -C_i * Z_i + D_i * (1 - Z_i) * X_i
     *
     * @return current transmitter state
     */
    TransmitterState getTransmitterState();

    /**
     * Update temporal dynamics by one time step.
     * Integrates both shunting and transmitter dynamics forward in time.
     *
     * @param timeStep the time step for integration (in seconds)
     */
    void updateDynamics(double timeStep);

    /**
     * Check if the temporal dynamics have reached equilibrium.
     * Equilibrium is defined as the state where change rates are below threshold.
     *
     * @param threshold the convergence threshold
     * @return true if dynamics have converged
     */
    boolean hasReachedEquilibrium(double threshold);

    /**
     * Reset temporal dynamics to initial state.
     * Clears all temporal state while preserving connection weights.
     */
    void resetDynamics();

    /**
     * Get the time scale at which this pathway operates.
     * Different pathways may operate at different time scales:
     * - FAST (10-100ms): Bottom-up sensory processing
     * - MEDIUM (50-500ms): Attention shifts, top-down modulation
     * - SLOW (500-5000ms): Learning and weight updates
     *
     * @return the time scale for this pathway
     */
    TimeScale getTimeScale();

    /**
     * Enable or disable temporal dynamics integration.
     * When disabled, pathway behaves as standard laminar pathway.
     * This allows for gradual rollout and A/B testing.
     *
     * @param enabled true to enable temporal dynamics
     */
    void setTemporalDynamicsEnabled(boolean enabled);

    /**
     * Check if temporal dynamics are currently enabled.
     *
     * @return true if temporal dynamics are active
     */
    boolean isTemporalDynamicsEnabled();
}