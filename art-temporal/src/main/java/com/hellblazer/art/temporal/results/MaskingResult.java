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
package com.hellblazer.art.temporal.results;

import com.hellblazer.art.temporal.TemporalPattern;

/**
 * Result of masking field network processing, containing activations,
 * boundary information, and competitive dynamics state.
 *
 * @author Hal Hildebrand
 */
public interface MaskingResult {

    /**
     * Get the input temporal pattern that was processed.
     *
     * @return input temporal pattern
     */
    TemporalPattern getInputPattern();

    /**
     * Get the activations across all scales of the masking field.
     *
     * @return activation values [scale][position]
     */
    double[][] getActivations();

    /**
     * Get the transmitter gate values across all scales.
     *
     * @return transmitter gate values [scale][position]
     */
    double[][] getTransmitterGates();

    /**
     * Get the detected chunk boundaries.
     *
     * @return array of boundary positions in the input sequence
     */
    int[] getChunkBoundaries();

    /**
     * Get the processing time to reach steady state.
     *
     * @return convergence time
     */
    double getConvergenceTime();

    /**
     * Check if the masking field reached steady state.
     *
     * @return true if converged
     */
    boolean hasConverged();

    /**
     * Get the maximum activation achieved.
     *
     * @return peak activation value
     */
    double getMaxActivation();

    /**
     * Get the total activation across all scales.
     *
     * @return sum of all activations
     */
    double getTotalActivation();

    /**
     * Get the center of mass of activations.
     *
     * @return temporal center of mass
     */
    double getCenterOfMass();
}