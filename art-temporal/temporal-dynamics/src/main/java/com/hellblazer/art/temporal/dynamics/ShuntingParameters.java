package com.hellblazer.art.temporal.dynamics;

/**
 * Parameters for shunting dynamics.
 * Based on Grossberg (1973) and Kazerounian & Grossberg (2014).
 */
public class ShuntingParameters {

    private final double[] decayRates;
    private final double ceiling;
    private final double floor;
    private final double selfExcitation;
    private final double excitatoryStrength;
    private final double inhibitoryStrength;
    private final double excitatoryRange;
    private final double inhibitoryRange;
    private final double initialActivation;
    private final double timeStep;

    private ShuntingParameters(Builder builder) {
        this.decayRates = builder.decayRates;
        this.ceiling = builder.ceiling;
        this.floor = builder.floor;
        this.selfExcitation = builder.selfExcitation;
        this.excitatoryStrength = builder.excitatoryStrength;
        this.inhibitoryStrength = builder.inhibitoryStrength;
        this.excitatoryRange = builder.excitatoryRange;
        this.inhibitoryRange = builder.inhibitoryRange;
        this.initialActivation = builder.initialActivation;
        this.timeStep = builder.timeStep;
        validate();
    }

    /**
     * Create default parameters for standard shunting dynamics.
     */
    public static ShuntingParameters defaults(int dimension) {
        return builder(dimension).build();
    }

    /**
     * Create parameters for competitive dynamics (Mexican hat).
     */
    public static ShuntingParameters competitiveDefaults(int dimension) {
        return builder(dimension)
            .ceiling(1.0)
            .floor(0.0)
            .selfExcitation(0.1)
            .excitatoryStrength(0.5)
            .inhibitoryStrength(0.8)
            .excitatoryRange(1.0)
            .inhibitoryRange(3.0)
            .timeStep(0.01)
            .build();
    }

    /**
     * Create parameters for winner-take-all dynamics.
     */
    public static ShuntingParameters winnerTakeAllDefaults(int dimension) {
        return builder(dimension)
            .ceiling(1.0)
            .floor(-0.1)
            .selfExcitation(0.2)
            .excitatoryStrength(0.0)  // No lateral excitation
            .inhibitoryStrength(1.5)   // Strong global inhibition
            .excitatoryRange(0.5)
            .inhibitoryRange(100.0)     // Essentially global
            .timeStep(0.01)
            .build();
    }

    public static Builder builder(int dimension) {
        return new Builder(dimension);
    }

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
            for (int i = 0; i < dimension; i++) {
                decayRates[i] = 1.0;  // Default uniform decay
            }
        }

        public Builder decayRates(double[] rates) {
            this.decayRates = rates.clone();
            return this;
        }

        public Builder uniformDecay(double rate) {
            for (int i = 0; i < decayRates.length; i++) {
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
            return new ShuntingParameters(this);
        }
    }

    private void validate() {
        if (ceiling <= floor) {
            throw new IllegalArgumentException("Ceiling must be greater than floor");
        }
        if (timeStep <= 0 || timeStep > 1) {
            throw new IllegalArgumentException("Time step must be in (0, 1]");
        }
        if (excitatoryRange <= 0 || inhibitoryRange <= 0) {
            throw new IllegalArgumentException("Ranges must be positive");
        }
        if (selfExcitation < 0) {
            throw new IllegalArgumentException("Self-excitation must be non-negative");
        }
        if (excitatoryStrength < 0 || inhibitoryStrength < 0) {
            throw new IllegalArgumentException("Strengths must be non-negative");
        }
        for (double rate : decayRates) {
            if (rate < 0) {
                throw new IllegalArgumentException("Decay rates must be non-negative");
            }
        }
    }

    // Getters
    public double getDecayRate(int index) {
        return decayRates[index];
    }

    public double[] getDecayRates() {
        return decayRates.clone();
    }

    public double getCeiling() {
        return ceiling;
    }

    public double getFloor() {
        return floor;
    }

    public double getSelfExcitation() {
        return selfExcitation;
    }

    public double getExcitatoryStrength() {
        return excitatoryStrength;
    }

    public double getInhibitoryStrength() {
        return inhibitoryStrength;
    }

    public double getExcitatoryRange() {
        return excitatoryRange;
    }

    public double getInhibitoryRange() {
        return inhibitoryRange;
    }

    public double getInitialActivation() {
        return initialActivation;
    }

    public double getTimeStep() {
        return timeStep;
    }

    public int getDimension() {
        return decayRates.length;
    }
}