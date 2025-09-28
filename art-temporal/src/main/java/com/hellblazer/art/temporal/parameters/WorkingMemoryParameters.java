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
package com.hellblazer.art.temporal.parameters;

/**
 * Configuration parameters for Item-and-Order Working Memory following
 * the STORE 2 model from Kazerounian & Grossberg 2014.
 *
 * These parameters control the temporal dynamics, capacity, and decay
 * characteristics of working memory during sequence processing.
 *
 * @param capacity maximum number of items that can be stored in working memory
 * @param decayRate passive decay rate (α) for primacy gradients
 * @param maxActivation maximum activation level (β) for stored items
 * @param competitiveRate competitive interaction strength (γ) between items
 * @param primacyThreshold minimum primacy value to maintain items in memory
 * @param temporalResolution time step size for dynamics integration
 * @param inputGain gain factor for new item inputs
 * @param enableCompetition whether to use competitive dynamics between items
 * @param enableNormalization whether to normalize total working memory activation
 * @param enableAdaptiveCapacity whether capacity can adapt based on load
 *
 * Mathematical Context:
 * Working memory dynamics: dx_i/dt = -α*x_i + (β - x_i)*I_i - γ*x_i*∑(x_j)
 * - decayRate (α): Controls how quickly items fade from memory
 * - maxActivation (β): Upper bound on item activation
 * - competitiveRate (γ): Strength of inhibitory interactions between items
 *
 * @author Hal Hildebrand
 */
public record WorkingMemoryParameters(
    int capacity,
    double decayRate,
    double maxActivation,
    double competitiveRate,
    double primacyThreshold,
    double temporalResolution,
    double inputGain,
    boolean enableCompetition,
    boolean enableNormalization,
    boolean enableAdaptiveCapacity
) {

    /**
     * Default working memory parameters based on empirical values from the literature.
     */
    public static final WorkingMemoryParameters DEFAULT = new WorkingMemoryParameters(
        7,      // capacity: Miller's magic number 7±2
        0.1,    // decayRate: moderate temporal decay
        1.0,    // maxActivation: normalized maximum
        0.05,   // competitiveRate: weak competition
        0.01,   // primacyThreshold: low threshold for memory retention
        0.01,   // temporalResolution: fine-grained time steps
        1.0,    // inputGain: linear input scaling
        true,   // enableCompetition: use competitive dynamics
        true,   // enableNormalization: maintain normalized activations
        false   // enableAdaptiveCapacity: fixed capacity
    );

    /**
     * Parameters optimized for short sequences (length 2-5).
     */
    public static final WorkingMemoryParameters SHORT_SEQUENCES = new WorkingMemoryParameters(
        5,      // capacity: smaller for short sequences
        0.05,   // decayRate: slower decay for short sequences
        1.0,    // maxActivation
        0.02,   // competitiveRate: reduced competition
        0.005,  // primacyThreshold: lower threshold
        0.005,  // temporalResolution: finer resolution
        1.2,    // inputGain: slightly higher gain
        true,   // enableCompetition
        true,   // enableNormalization
        false   // enableAdaptiveCapacity
    );

    /**
     * Parameters optimized for long sequences (length 10+).
     */
    public static final WorkingMemoryParameters LONG_SEQUENCES = new WorkingMemoryParameters(
        15,     // capacity: larger for long sequences
        0.2,    // decayRate: faster decay to prevent overflow
        1.0,    // maxActivation
        0.1,    // competitiveRate: stronger competition
        0.02,   // primacyThreshold: higher threshold
        0.02,   // temporalResolution: coarser resolution
        0.8,    // inputGain: reduced gain to prevent saturation
        true,   // enableCompetition
        true,   // enableNormalization
        true    // enableAdaptiveCapacity: allow capacity adaptation
    );

    /**
     * Parameters for real-time processing with minimal delay.
     */
    public static final WorkingMemoryParameters REAL_TIME = new WorkingMemoryParameters(
        10,     // capacity: moderate size
        0.15,   // decayRate: fast decay for real-time
        1.0,    // maxActivation
        0.08,   // competitiveRate: moderate competition
        0.015,  // primacyThreshold
        0.01,   // temporalResolution: balance speed/accuracy
        1.0,    // inputGain
        true,   // enableCompetition
        false,  // enableNormalization: disabled for speed
        false   // enableAdaptiveCapacity: fixed for predictability
    );

    /**
     * Create a builder for working memory parameters.
     *
     * @return new parameter builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Create a copy of these parameters with modified capacity.
     *
     * @param newCapacity the new capacity value
     * @return new parameters with updated capacity
     */
    public WorkingMemoryParameters withCapacity(int newCapacity) {
        return new WorkingMemoryParameters(
            newCapacity, decayRate, maxActivation, competitiveRate,
            primacyThreshold, temporalResolution, inputGain,
            enableCompetition, enableNormalization, enableAdaptiveCapacity
        );
    }

    /**
     * Create a copy of these parameters with modified decay rate.
     *
     * @param newDecayRate the new decay rate value
     * @return new parameters with updated decay rate
     */
    public WorkingMemoryParameters withDecayRate(double newDecayRate) {
        return new WorkingMemoryParameters(
            capacity, newDecayRate, maxActivation, competitiveRate,
            primacyThreshold, temporalResolution, inputGain,
            enableCompetition, enableNormalization, enableAdaptiveCapacity
        );
    }

    /**
     * Validate parameter values for consistency and reasonableness.
     *
     * @throws IllegalArgumentException if parameters are invalid
     */
    public void validate() {
        if (capacity <= 0) {
            throw new IllegalArgumentException("Capacity must be positive: " + capacity);
        }
        if (decayRate < 0 || decayRate > 1) {
            throw new IllegalArgumentException("Decay rate must be in [0,1]: " + decayRate);
        }
        if (maxActivation <= 0) {
            throw new IllegalArgumentException("Max activation must be positive: " + maxActivation);
        }
        if (competitiveRate < 0) {
            throw new IllegalArgumentException("Competitive rate must be non-negative: " + competitiveRate);
        }
        if (primacyThreshold < 0) {
            throw new IllegalArgumentException("Primacy threshold must be non-negative: " + primacyThreshold);
        }
        if (temporalResolution <= 0) {
            throw new IllegalArgumentException("Temporal resolution must be positive: " + temporalResolution);
        }
        if (inputGain <= 0) {
            throw new IllegalArgumentException("Input gain must be positive: " + inputGain);
        }
    }

    /**
     * Get the time constant for decay (1/decayRate).
     *
     * @return decay time constant
     */
    public double getDecayTimeConstant() {
        return decayRate > 0 ? 1.0 / decayRate : Double.POSITIVE_INFINITY;
    }

    /**
     * Calculate the half-life of items in working memory.
     *
     * @return half-life in time units
     */
    public double getHalfLife() {
        return decayRate > 0 ? Math.log(2) / decayRate : Double.POSITIVE_INFINITY;
    }

    /**
     * Get the effective capacity considering competitive dynamics.
     * With strong competition, effective capacity may be lower than nominal capacity.
     *
     * @return estimated effective capacity
     */
    public double getEffectiveCapacity() {
        if (!enableCompetition) return capacity;

        // Competitive dynamics reduce effective capacity
        var competitiveFactor = 1.0 / (1.0 + competitiveRate * capacity);
        return capacity * competitiveFactor;
    }

    /**
     * Builder class for working memory parameters with validation.
     */
    public static class Builder {
        private int capacity = DEFAULT.capacity;
        private double decayRate = DEFAULT.decayRate;
        private double maxActivation = DEFAULT.maxActivation;
        private double competitiveRate = DEFAULT.competitiveRate;
        private double primacyThreshold = DEFAULT.primacyThreshold;
        private double temporalResolution = DEFAULT.temporalResolution;
        private double inputGain = DEFAULT.inputGain;
        private boolean enableCompetition = DEFAULT.enableCompetition;
        private boolean enableNormalization = DEFAULT.enableNormalization;
        private boolean enableAdaptiveCapacity = DEFAULT.enableAdaptiveCapacity;

        public Builder capacity(int capacity) {
            this.capacity = capacity;
            return this;
        }

        public Builder decayRate(double decayRate) {
            this.decayRate = decayRate;
            return this;
        }

        public Builder maxActivation(double maxActivation) {
            this.maxActivation = maxActivation;
            return this;
        }

        public Builder competitiveRate(double competitiveRate) {
            this.competitiveRate = competitiveRate;
            return this;
        }

        public Builder primacyThreshold(double primacyThreshold) {
            this.primacyThreshold = primacyThreshold;
            return this;
        }

        public Builder temporalResolution(double temporalResolution) {
            this.temporalResolution = temporalResolution;
            return this;
        }

        public Builder inputGain(double inputGain) {
            this.inputGain = inputGain;
            return this;
        }

        public Builder enableCompetition(boolean enableCompetition) {
            this.enableCompetition = enableCompetition;
            return this;
        }

        public Builder enableNormalization(boolean enableNormalization) {
            this.enableNormalization = enableNormalization;
            return this;
        }

        public Builder enableAdaptiveCapacity(boolean enableAdaptiveCapacity) {
            this.enableAdaptiveCapacity = enableAdaptiveCapacity;
            return this;
        }

        /**
         * Build and validate the working memory parameters.
         *
         * @return validated parameters
         * @throws IllegalArgumentException if parameters are invalid
         */
        public WorkingMemoryParameters build() {
            var params = new WorkingMemoryParameters(
                capacity, decayRate, maxActivation, competitiveRate,
                primacyThreshold, temporalResolution, inputGain,
                enableCompetition, enableNormalization, enableAdaptiveCapacity
            );
            params.validate();
            return params;
        }
    }
}