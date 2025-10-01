package com.hellblazer.art.cortical.dynamics;

/**
 * Parameters for shunting dynamics (on-center off-surround competition).
 * Based on Grossberg (1973) and Kazerounian & Grossberg (2014).
 *
 * <p>Shunting equation:
 * dx_i/dt = -A_i * x_i + (B - x_i) * S_i^+ - (x_i - C) * S_i^-
 *
 * Where:
 * <ul>
 *   <li>A_i: decay rate for unit i</li>
 *   <li>B: activation ceiling (upper bound)</li>
 *   <li>C: activation floor (lower bound)</li>
 *   <li>S_i^+: excitatory input to unit i</li>
 *   <li>S_i^-: inhibitory input to unit i</li>
 * </ul>
 *
 * @param decayRates per-unit decay rates (passive decay, dimension n)
 * @param ceiling activation ceiling (typically 1.0)
 * @param floor activation floor (typically 0.0 or small negative)
 * @param selfExcitation self-excitation strength (recurrent)
 * @param excitatoryStrength lateral excitation strength (on-center)
 * @param inhibitoryStrength lateral inhibition strength (off-surround)
 * @param excitatoryRange spatial range of excitation (narrow)
 * @param inhibitoryRange spatial range of inhibition (broad)
 * @param initialActivation initial activation level (typically 0.0)
 * @param timeStep integration time step (typically 0.01)
 *
 * @author Migrated from art-temporal/temporal-dynamics to art-cortical (Phase 1)
 */
public record ShuntingParameters(
    double[] decayRates,
    double ceiling,
    double floor,
    double selfExcitation,
    double excitatoryStrength,
    double inhibitoryStrength,
    double excitatoryRange,
    double inhibitoryRange,
    double initialActivation,
    double timeStep
) {

    /**
     * Compact canonical constructor with validation.
     */
    public ShuntingParameters {
        if (ceiling <= floor) {
            throw new IllegalArgumentException("Ceiling must be greater than floor: " +
                                               ceiling + " <= " + floor);
        }
        if (timeStep <= 0 || timeStep > 1) {
            throw new IllegalArgumentException("Time step must be in (0, 1]: " + timeStep);
        }
        if (excitatoryRange <= 0 || inhibitoryRange <= 0) {
            throw new IllegalArgumentException(
                "Ranges must be positive: exc=" + excitatoryRange + ", inh=" + inhibitoryRange);
        }
        if (selfExcitation < 0) {
            throw new IllegalArgumentException("Self-excitation must be non-negative: " + selfExcitation);
        }
        if (excitatoryStrength < 0 || inhibitoryStrength < 0) {
            throw new IllegalArgumentException(
                "Strengths must be non-negative: exc=" + excitatoryStrength + ", inh=" + inhibitoryStrength);
        }
        for (var i = 0; i < decayRates.length; i++) {
            if (decayRates[i] < 0) {
                throw new IllegalArgumentException("Decay rate at index " + i + " is negative: " + decayRates[i]);
            }
        }
        // Defensive copy of mutable array
        decayRates = decayRates.clone();
    }

    /**
     * Get decay rate for specific unit.
     */
    public double getDecayRate(int index) {
        return decayRates[index];
    }

    /**
     * Get dimension (number of units).
     */
    public int getDimension() {
        return decayRates.length;
    }

    /**
     * Create default parameters for standard shunting dynamics.
     */
    public static ShuntingParameters defaults(int dimension) {
        return builder(dimension).build();
    }

    /**
     * Create parameters for competitive dynamics (Mexican hat profile).
     * Narrow on-center excitation + broad off-surround inhibition.
     */
    public static ShuntingParameters competitiveDefaults(int dimension) {
        return builder(dimension)
            .ceiling(1.0)
            .floor(0.0)
            .selfExcitation(0.1)
            .excitatoryStrength(0.5)
            .inhibitoryStrength(0.8)
            .excitatoryRange(1.0)    // Narrow on-center
            .inhibitoryRange(3.0)     // Broad off-surround
            .timeStep(0.01)
            .build();
    }

    /**
     * Create parameters for winner-take-all dynamics.
     * Strong global inhibition, no lateral excitation.
     */
    public static ShuntingParameters winnerTakeAllDefaults(int dimension) {
        return builder(dimension)
            .ceiling(1.0)
            .floor(-0.1)
            .selfExcitation(0.2)
            .excitatoryStrength(0.0)   // No lateral excitation
            .inhibitoryStrength(1.5)    // Strong inhibition
            .excitatoryRange(0.5)
            .inhibitoryRange(100.0)     // Essentially global
            .timeStep(0.01)
            .build();
    }

    /**
     * Create a builder for constructing parameters.
     */
    public static Builder builder(int dimension) {
        return new Builder(dimension);
    }

    /**
     * Builder for ShuntingParameters using fluent interface.
     */
    public static class Builder {
        private double[] decayRates;
        private double ceiling = 1.0;
        private double floor = 0.0;
        private double selfExcitation = 0.1;
        private double excitatoryStrength = 0.3;
        private double inhibitoryStrength = 0.5;
        private double excitatoryRange = 2.0;
        private double inhibitoryRange = 5.0;
        private double initialActivation = 0.0;
        private double timeStep = 0.01;

        public Builder(int dimension) {
            this.decayRates = new double[dimension];
            for (var i = 0; i < dimension; i++) {
                decayRates[i] = 1.0;  // Default uniform decay
            }
        }

        public Builder decayRates(double[] rates) {
            this.decayRates = rates.clone();
            return this;
        }

        public Builder uniformDecay(double rate) {
            for (var i = 0; i < decayRates.length; i++) {
                decayRates[i] = rate;
            }
            return this;
        }

        public Builder ceiling(double ceiling) {
            this.ceiling = ceiling;
            return this;
        }

        public Builder floor(double floor) {
            this.floor = floor;
            return this;
        }

        public Builder selfExcitation(double selfExcitation) {
            this.selfExcitation = selfExcitation;
            return this;
        }

        public Builder excitatoryStrength(double strength) {
            this.excitatoryStrength = strength;
            return this;
        }

        public Builder inhibitoryStrength(double strength) {
            this.inhibitoryStrength = strength;
            return this;
        }

        public Builder excitatoryRange(double range) {
            this.excitatoryRange = range;
            return this;
        }

        public Builder inhibitoryRange(double range) {
            this.inhibitoryRange = range;
            return this;
        }

        public Builder initialActivation(double initial) {
            this.initialActivation = initial;
            return this;
        }

        public Builder timeStep(double step) {
            this.timeStep = step;
            return this;
        }

        public ShuntingParameters build() {
            return new ShuntingParameters(
                decayRates,
                ceiling,
                floor,
                selfExcitation,
                excitatoryStrength,
                inhibitoryStrength,
                excitatoryRange,
                inhibitoryRange,
                initialActivation,
                timeStep
            );
        }
    }
}
