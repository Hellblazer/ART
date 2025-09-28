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
 * Configuration parameters for the multi-scale masking field network
 * following Kazerounian & Grossberg 2014.
 *
 * @param scaleCount number of hierarchical scales in the masking field
 * @param fieldSize spatial extent of each masking field scale
 * @param passiveDecayRate passive decay rate (α) for field activations
 * @param maxActivation maximum activation level (β) for field nodes
 * @param lateralInhibition strength of lateral inhibition (σ) between nodes
 * @param selfInhibition strength of self-inhibition (γ) within nodes
 * @param transmitterRecoveryRate recovery rate (δ) for habituative gates
 * @param transmitterDepletionRate depletion rate (ε) for habituative gates
 * @param boundaryThreshold threshold for detecting chunking boundaries
 * @param convergenceThreshold threshold for steady-state detection
 * @param timeStep temporal resolution for dynamics integration
 * @param enableTransmitterGates whether to use habituative transmitter gates
 * @param enableCompetition whether to use competitive dynamics
 * @param enableMultiScale whether to use multi-scale processing
 * @param scaleFactor factor for scaling between hierarchical levels
 *
 * @author Hal Hildebrand
 */
public record MaskingParameters(
    int scaleCount,
    int fieldSize,
    double passiveDecayRate,
    double maxActivation,
    double lateralInhibition,
    double selfInhibition,
    double transmitterRecoveryRate,
    double transmitterDepletionRate,
    double boundaryThreshold,
    double convergenceThreshold,
    double timeStep,
    boolean enableTransmitterGates,
    boolean enableCompetition,
    boolean enableMultiScale,
    double scaleFactor
) {

    /**
     * Default masking field parameters.
     */
    public static final MaskingParameters DEFAULT = new MaskingParameters(
        3,       // scaleCount: 3 hierarchical scales
        20,      // fieldSize: 20 nodes per scale
        0.1,     // passiveDecayRate: moderate decay
        1.0,     // maxActivation: normalized maximum
        0.5,     // lateralInhibition: moderate lateral inhibition
        0.1,     // selfInhibition: weak self-inhibition
        0.2,     // transmitterRecoveryRate: moderate recovery
        0.8,     // transmitterDepletionRate: strong depletion
        0.1,     // boundaryThreshold: low boundary detection threshold
        0.01,    // convergenceThreshold: fine convergence criterion
        0.01,    // timeStep: small integration step
        true,    // enableTransmitterGates: use habituation
        true,    // enableCompetition: use competitive dynamics
        true,    // enableMultiScale: use hierarchical processing
        2.0      // scaleFactor: scale by factor of 2
    );

    /**
     * Parameters optimized for short sequences.
     */
    public static final MaskingParameters SHORT_SEQUENCES = new MaskingParameters(
        2,       // scaleCount: fewer scales for short sequences
        15,      // fieldSize: smaller fields
        0.05,    // passiveDecayRate: slower decay
        1.0,     // maxActivation
        0.3,     // lateralInhibition: weaker inhibition
        0.05,    // selfInhibition
        0.3,     // transmitterRecoveryRate: faster recovery
        0.6,     // transmitterDepletionRate: slower depletion
        0.15,    // boundaryThreshold: higher threshold
        0.005,   // convergenceThreshold: tighter convergence
        0.005,   // timeStep: finer resolution
        true,    // enableTransmitterGates
        true,    // enableCompetition
        true,    // enableMultiScale
        1.5      // scaleFactor: smaller scale factor
    );

    /**
     * Parameters optimized for long sequences.
     */
    public static final MaskingParameters LONG_SEQUENCES = new MaskingParameters(
        4,       // scaleCount: more scales for long sequences
        30,      // fieldSize: larger fields
        0.2,     // passiveDecayRate: faster decay
        1.0,     // maxActivation
        0.7,     // lateralInhibition: stronger inhibition
        0.15,    // selfInhibition
        0.1,     // transmitterRecoveryRate: slower recovery
        1.0,     // transmitterDepletionRate: faster depletion
        0.05,    // boundaryThreshold: lower threshold
        0.02,    // convergenceThreshold: looser convergence
        0.02,    // timeStep: coarser resolution
        true,    // enableTransmitterGates
        true,    // enableCompetition
        true,    // enableMultiScale
        3.0      // scaleFactor: larger scale factor
    );

    /**
     * Parameters for real-time processing.
     */
    public static final MaskingParameters REAL_TIME = new MaskingParameters(
        3,       // scaleCount
        25,      // fieldSize
        0.15,    // passiveDecayRate: moderate for stability
        1.0,     // maxActivation
        0.4,     // lateralInhibition
        0.08,    // selfInhibition
        0.25,    // transmitterRecoveryRate
        0.7,     // transmitterDepletionRate
        0.12,    // boundaryThreshold
        0.015,   // convergenceThreshold: balance speed/accuracy
        0.01,    // timeStep
        true,    // enableTransmitterGates
        true,    // enableCompetition
        true,    // enableMultiScale
        2.5      // scaleFactor
    );

    /**
     * High-performance parameters with reduced computation.
     */
    public static final MaskingParameters HIGH_PERFORMANCE = new MaskingParameters(
        2,       // scaleCount: fewer scales for speed
        16,      // fieldSize: power-of-2 for vectorization
        0.2,     // passiveDecayRate
        1.0,     // maxActivation
        0.6,     // lateralInhibition
        0.1,     // selfInhibition
        0.2,     // transmitterRecoveryRate
        0.8,     // transmitterDepletionRate
        0.1,     // boundaryThreshold
        0.02,    // convergenceThreshold: looser for speed
        0.02,    // timeStep: larger for speed
        false,   // enableTransmitterGates: disabled for speed
        true,    // enableCompetition
        true,    // enableMultiScale
        2.0      // scaleFactor
    );

    /**
     * Create a builder for masking parameters.
     *
     * @return new parameter builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Validate parameter values.
     *
     * @throws IllegalArgumentException if parameters are invalid
     */
    public void validate() {
        if (scaleCount <= 0) {
            throw new IllegalArgumentException("Scale count must be positive: " + scaleCount);
        }
        if (fieldSize <= 0) {
            throw new IllegalArgumentException("Field size must be positive: " + fieldSize);
        }
        if (passiveDecayRate < 0 || passiveDecayRate > 1) {
            throw new IllegalArgumentException("Passive decay rate must be in [0,1]: " + passiveDecayRate);
        }
        if (maxActivation <= 0) {
            throw new IllegalArgumentException("Max activation must be positive: " + maxActivation);
        }
        if (lateralInhibition < 0) {
            throw new IllegalArgumentException("Lateral inhibition must be non-negative: " + lateralInhibition);
        }
        if (selfInhibition < 0) {
            throw new IllegalArgumentException("Self inhibition must be non-negative: " + selfInhibition);
        }
        if (transmitterRecoveryRate < 0 || transmitterRecoveryRate > 1) {
            throw new IllegalArgumentException("Transmitter recovery rate must be in [0,1]: " + transmitterRecoveryRate);
        }
        if (transmitterDepletionRate < 0) {
            throw new IllegalArgumentException("Transmitter depletion rate must be non-negative: " + transmitterDepletionRate);
        }
        if (boundaryThreshold < 0 || boundaryThreshold > 1) {
            throw new IllegalArgumentException("Boundary threshold must be in [0,1]: " + boundaryThreshold);
        }
        if (convergenceThreshold <= 0) {
            throw new IllegalArgumentException("Convergence threshold must be positive: " + convergenceThreshold);
        }
        if (timeStep <= 0) {
            throw new IllegalArgumentException("Time step must be positive: " + timeStep);
        }
        if (scaleFactor <= 1.0) {
            throw new IllegalArgumentException("Scale factor must be > 1.0: " + scaleFactor);
        }
    }

    /**
     * Get the field size at a specific scale.
     *
     * @param scale the scale index (0 = finest)
     * @return field size at the specified scale
     */
    public int getFieldSizeAtScale(int scale) {
        if (scale < 0 || scale >= scaleCount) {
            throw new IllegalArgumentException("Invalid scale: " + scale);
        }
        return (int) (fieldSize / Math.pow(scaleFactor, scale));
    }

    /**
     * Get the time constant for passive decay.
     *
     * @return decay time constant
     */
    public double getDecayTimeConstant() {
        return passiveDecayRate > 0 ? 1.0 / passiveDecayRate : Double.POSITIVE_INFINITY;
    }

    /**
     * Get the estimated convergence time.
     *
     * @return expected time to reach steady state
     */
    public double getEstimatedConvergenceTime() {
        // Empirical formula based on field dynamics
        var baseTime = 3.0 / passiveDecayRate;
        var competitionFactor = enableCompetition ? (1.0 + lateralInhibition) : 1.0;
        var scaleFactor = enableMultiScale ? (1.0 + 0.5 * scaleCount) : 1.0;
        return baseTime * competitionFactor * scaleFactor;
    }

    /**
     * Builder class for masking parameters.
     */
    public static class Builder {
        private int scaleCount = DEFAULT.scaleCount;
        private int fieldSize = DEFAULT.fieldSize;
        private double passiveDecayRate = DEFAULT.passiveDecayRate;
        private double maxActivation = DEFAULT.maxActivation;
        private double lateralInhibition = DEFAULT.lateralInhibition;
        private double selfInhibition = DEFAULT.selfInhibition;
        private double transmitterRecoveryRate = DEFAULT.transmitterRecoveryRate;
        private double transmitterDepletionRate = DEFAULT.transmitterDepletionRate;
        private double boundaryThreshold = DEFAULT.boundaryThreshold;
        private double convergenceThreshold = DEFAULT.convergenceThreshold;
        private double timeStep = DEFAULT.timeStep;
        private boolean enableTransmitterGates = DEFAULT.enableTransmitterGates;
        private boolean enableCompetition = DEFAULT.enableCompetition;
        private boolean enableMultiScale = DEFAULT.enableMultiScale;
        private double scaleFactor = DEFAULT.scaleFactor;

        public Builder scaleCount(int scaleCount) {
            this.scaleCount = scaleCount;
            return this;
        }

        public Builder fieldSize(int fieldSize) {
            this.fieldSize = fieldSize;
            return this;
        }

        public Builder passiveDecayRate(double passiveDecayRate) {
            this.passiveDecayRate = passiveDecayRate;
            return this;
        }

        public Builder maxActivation(double maxActivation) {
            this.maxActivation = maxActivation;
            return this;
        }

        public Builder lateralInhibition(double lateralInhibition) {
            this.lateralInhibition = lateralInhibition;
            return this;
        }

        public Builder selfInhibition(double selfInhibition) {
            this.selfInhibition = selfInhibition;
            return this;
        }

        public Builder transmitterRecoveryRate(double transmitterRecoveryRate) {
            this.transmitterRecoveryRate = transmitterRecoveryRate;
            return this;
        }

        public Builder transmitterDepletionRate(double transmitterDepletionRate) {
            this.transmitterDepletionRate = transmitterDepletionRate;
            return this;
        }

        public Builder boundaryThreshold(double boundaryThreshold) {
            this.boundaryThreshold = boundaryThreshold;
            return this;
        }

        public Builder convergenceThreshold(double convergenceThreshold) {
            this.convergenceThreshold = convergenceThreshold;
            return this;
        }

        public Builder timeStep(double timeStep) {
            this.timeStep = timeStep;
            return this;
        }

        public Builder enableTransmitterGates(boolean enableTransmitterGates) {
            this.enableTransmitterGates = enableTransmitterGates;
            return this;
        }

        public Builder enableCompetition(boolean enableCompetition) {
            this.enableCompetition = enableCompetition;
            return this;
        }

        public Builder enableMultiScale(boolean enableMultiScale) {
            this.enableMultiScale = enableMultiScale;
            return this;
        }

        public Builder scaleFactor(double scaleFactor) {
            this.scaleFactor = scaleFactor;
            return this;
        }

        /**
         * Build and validate the masking parameters.
         *
         * @return validated parameters
         */
        public MaskingParameters build() {
            var params = new MaskingParameters(
                scaleCount, fieldSize, passiveDecayRate, maxActivation,
                lateralInhibition, selfInhibition, transmitterRecoveryRate,
                transmitterDepletionRate, boundaryThreshold, convergenceThreshold,
                timeStep, enableTransmitterGates, enableCompetition,
                enableMultiScale, scaleFactor
            );
            params.validate();
            return params;
        }
    }
}