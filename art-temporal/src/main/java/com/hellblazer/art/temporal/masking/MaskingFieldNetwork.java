/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 *
 * This file is part of Java ART Neural Networks.
 *
 * Java ART Neural Networks is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Java ART Neural Networks is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with Java ART Neural Networks. If not, see <https://www.gnu.org/licenses/>.
 */
package com.hellblazer.art.temporal.masking;

import com.hellblazer.art.temporal.TemporalPattern;
import com.hellblazer.art.temporal.parameters.MaskingParameters;
import com.hellblazer.art.temporal.results.MaskingResult;

/**
 * Multi-scale masking field network implementing the self-similar architecture
 * from Kazerounian & Grossberg 2014.
 *
 * The masking field network performs temporal preprocessing through:
 * - Multi-scale competitive dynamics
 * - Habituative transmitter gates with temporal decay
 * - Boundary detection for sequence chunking
 * - Self-similar structure across scales
 *
 * Mathematical Foundation:
 * Each masking field layer follows competitive dynamics:
 * dx_ij/dt = -α*x_ij + (β - x_ij)*[I_ij - σ*∑(x_kl)] - γ*x_ij*∑(x_ij)
 *
 * Where:
 * - x_ij is activation at position (i,j) in the field
 * - I_ij is input from working memory or lower scale
 * - α is passive decay rate
 * - β is maximum activation
 * - σ controls lateral inhibition
 * - γ controls self-inhibition
 *
 * Habituative transmitter gates:
 * dz_ij/dt = δ*(1 - z_ij) - ε*x_ij*z_ij
 *
 * Where:
 * - z_ij is transmitter gate value
 * - δ is recovery rate
 * - ε is depletion rate
 *
 * @author Hal Hildebrand
 */
public interface MaskingFieldNetwork extends AutoCloseable {

    /**
     * Process a temporal pattern through the masking field network.
     * The pattern is processed through multiple scales to identify
     * chunking boundaries and temporal structure.
     *
     * @param input temporal pattern from working memory
     * @return masking field processing result with boundary information
     */
    MaskingResult process(TemporalPattern input);

    /**
     * Process a single time step of masking field dynamics.
     * This method supports real-time processing where inputs arrive
     * incrementally.
     *
     * @param input current input pattern
     * @param deltaTime time step for dynamics integration
     * @return current masking field state
     */
    MaskingResult processTimeStep(TemporalPattern input, double deltaTime);

    /**
     * Get the number of scales in the masking field hierarchy.
     * Higher scales detect longer temporal structures.
     *
     * @return number of hierarchical scales
     */
    int getScaleCount();

    /**
     * Get activations at a specific scale.
     *
     * @param scale the scale index (0 = finest, higher = coarser)
     * @return activation values at the specified scale
     */
    double[] getScaleActivations(int scale);

    /**
     * Get transmitter gate values at a specific scale.
     *
     * @param scale the scale index
     * @return transmitter gate values at the specified scale
     */
    double[] getScaleTransmitterGates(int scale);

    /**
     * Get all activations across all scales.
     *
     * @return 2D array [scale][position] of activations
     */
    double[][] getAllActivations();

    /**
     * Get all transmitter gate values across all scales.
     *
     * @return 2D array [scale][position] of gate values
     */
    double[][] getAllTransmitterGates();

    /**
     * Detect chunking boundaries in the current activation pattern.
     * Boundaries are detected where masking field activity drops
     * below threshold, indicating natural breakpoints.
     *
     * @return list of boundary positions in the temporal sequence
     */
    int[] detectChunkBoundaries();

    /**
     * Detect boundaries at a specific scale.
     *
     * @param scale the scale at which to detect boundaries
     * @return boundary positions for the specified scale
     */
    int[] detectBoundariesAtScale(int scale);

    /**
     * Check if the masking field has reached steady state.
     * Steady state indicates that competitive dynamics have stabilized.
     *
     * @return true if field has converged to steady state
     */
    boolean hasReachedSteadyState();

    /**
     * Get the time to reach steady state.
     *
     * @return convergence time in temporal units
     */
    double getConvergenceTime();

    /**
     * Reset the masking field network to initial state.
     * Clears all activations and transmitter gates.
     */
    void reset();

    /**
     * Update the network parameters.
     *
     * @param parameters new masking field configuration
     */
    void setParameters(MaskingParameters parameters);

    /**
     * Get the current network parameters.
     *
     * @return current parameter configuration
     */
    MaskingParameters getParameters();

    /**
     * Get the total activation across all scales.
     *
     * @return sum of all activations in the network
     */
    double getTotalActivation();

    /**
     * Get the maximum activation in the network.
     *
     * @return highest activation value across all scales
     */
    double getMaxActivation();

    /**
     * Get the center of mass of activations.
     * This indicates the temporal focus of the masking field.
     *
     * @return center of mass position
     */
    double getActivationCenterOfMass();

    /**
     * Check if competitive dynamics are currently active.
     *
     * @return true if significant competitive activity is present
     */
    boolean hasCompetitiveActivity();

    /**
     * Get the competitive activity level.
     * Higher values indicate stronger competition between field positions.
     *
     * @return competitive activity measure (0.0 to 1.0)
     */
    double getCompetitiveActivityLevel();

    /**
     * Apply external modulation to the masking field.
     * This can represent top-down attention or context effects.
     *
     * @param modulation modulation values for each scale
     */
    void applyModulation(double[][] modulation);

    /**
     * Enable or disable habituative transmitter gates.
     *
     * @param enabled whether to use transmitter gate dynamics
     */
    void setTransmitterGatesEnabled(boolean enabled);

    /**
     * Check if transmitter gates are currently enabled.
     *
     * @return true if transmitter gate dynamics are active
     */
    boolean areTransmitterGatesEnabled();

    /**
     * Get performance metrics for masking field operations.
     *
     * @return performance statistics
     */
    MaskingFieldPerformanceMetrics getPerformanceMetrics();

    /**
     * Reset performance tracking counters.
     */
    void resetPerformanceTracking();

    /**
     * Create a snapshot of the current masking field state.
     *
     * @return immutable state snapshot
     */
    MaskingFieldSnapshot createSnapshot();

    /**
     * Restore masking field from a snapshot.
     *
     * @param snapshot the state to restore
     */
    void restoreSnapshot(MaskingFieldSnapshot snapshot);

    /**
     * Immutable snapshot of masking field state.
     */
    interface MaskingFieldSnapshot {
        /**
         * Get the activations at snapshot time.
         *
         * @return activation values [scale][position]
         */
        double[][] getActivations();

        /**
         * Get the transmitter gate values at snapshot time.
         *
         * @return gate values [scale][position]
         */
        double[][] getTransmitterGates();

        /**
         * Get the timestamp when snapshot was created.
         *
         * @return snapshot creation time
         */
        double getSnapshotTime();

        /**
         * Get the convergence state at snapshot time.
         *
         * @return true if field had converged
         */
        boolean wasConverged();
    }

    /**
     * Performance metrics for masking field operations.
     */
    interface MaskingFieldPerformanceMetrics {
        /**
         * Get the number of processing operations performed.
         *
         * @return processing operation count
         */
        long getProcessingOperations();

        /**
         * Get the number of dynamics update steps.
         *
         * @return dynamics update count
         */
        long getDynamicsUpdates();

        /**
         * Get the total computation time.
         *
         * @return computation time in nanoseconds
         */
        long getComputationTime();

        /**
         * Get the average convergence time.
         *
         * @return mean time to reach steady state
         */
        double getAverageConvergenceTime();

        /**
         * Get the number of SIMD operations performed.
         *
         * @return SIMD operation count
         */
        long getSIMDOperationCount();

        /**
         * Get the current memory usage.
         *
         * @return memory usage in bytes
         */
        long getMemoryUsage();
    }
}