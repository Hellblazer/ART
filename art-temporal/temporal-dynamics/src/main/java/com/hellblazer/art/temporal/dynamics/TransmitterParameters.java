package com.hellblazer.art.temporal.dynamics;

/**
 * Parameters for transmitter habituation dynamics.
 * Based on Kazerounian & Grossberg (2014).
 */
public class TransmitterParameters {

    private final double recoveryRate;
    private final double linearDepletionRate;
    private final double quadraticDepletionRate;
    private final double baselineLevel;
    private final double timeConstant;

    private TransmitterParameters(Builder builder) {
        this.recoveryRate = builder.recoveryRate;
        this.linearDepletionRate = builder.linearDepletionRate;
        this.quadraticDepletionRate = builder.quadraticDepletionRate;
        this.baselineLevel = builder.baselineLevel;
        this.timeConstant = builder.timeConstant;
        validate();
    }

    /**
     * Create default parameters from paper.
     */
    public static TransmitterParameters paperDefaults() {
        return builder()
            .recoveryRate(0.01)        // ε = 0.01 (slow recovery)
            .linearDepletionRate(1.0)   // λ = 1.0
            .quadraticDepletionRate(0.5) // μ = 0.5
            .baselineLevel(1.0)
            .timeConstant(500.0)        // 500ms time scale
            .build();
    }

    /**
     * Create parameters for fast habituation.
     */
    public static TransmitterParameters fastHabituation() {
        return builder()
            .recoveryRate(0.001)         // Very slow recovery
            .linearDepletionRate(2.0)    // Strong depletion
            .quadraticDepletionRate(1.0)
            .baselineLevel(1.0)
            .timeConstant(100.0)         // 100ms time scale
            .build();
    }

    /**
     * Create parameters for slow habituation.
     */
    public static TransmitterParameters slowHabituation() {
        return builder()
            .recoveryRate(0.1)           // Faster recovery
            .linearDepletionRate(0.5)    // Weak depletion
            .quadraticDepletionRate(0.1)
            .baselineLevel(1.0)
            .timeConstant(1000.0)        // 1000ms time scale
            .build();
    }

    /**
     * Create parameters for no habituation (constant transmitter).
     */
    public static TransmitterParameters noHabituation() {
        return builder()
            .recoveryRate(1.0)           // Instant recovery
            .linearDepletionRate(0.0)    // No depletion
            .quadraticDepletionRate(0.0)
            .baselineLevel(1.0)
            .timeConstant(1.0)
            .build();
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private double recoveryRate = 0.01;
        private double linearDepletionRate = 1.0;
        private double quadraticDepletionRate = 0.5;
        private double baselineLevel = 1.0;
        private double timeConstant = 500.0;

        public Builder recoveryRate(double rate) {
            this.recoveryRate = rate;
            return this;
        }

        public Builder linearDepletionRate(double rate) {
            this.linearDepletionRate = rate;
            return this;
        }

        public Builder quadraticDepletionRate(double rate) {
            this.quadraticDepletionRate = rate;
            return this;
        }

        public Builder baselineLevel(double level) {
            this.baselineLevel = level;
            return this;
        }

        public Builder timeConstant(double constant) {
            this.timeConstant = constant;
            return this;
        }

        public TransmitterParameters build() {
            return new TransmitterParameters(this);
        }
    }

    private void validate() {
        if (recoveryRate < 0 || recoveryRate > 1) {
            throw new IllegalArgumentException("Recovery rate must be in [0, 1]");
        }
        if (linearDepletionRate < 0) {
            throw new IllegalArgumentException("Linear depletion rate must be non-negative");
        }
        if (quadraticDepletionRate < 0) {
            throw new IllegalArgumentException("Quadratic depletion rate must be non-negative");
        }
        if (baselineLevel < 0 || baselineLevel > 1) {
            throw new IllegalArgumentException("Baseline level must be in [0, 1]");
        }
        if (timeConstant <= 0) {
            throw new IllegalArgumentException("Time constant must be positive");
        }
    }

    // Getters
    public double getRecoveryRate() {
        return recoveryRate;
    }

    public double getLinearDepletionRate() {
        return linearDepletionRate;
    }

    public double getQuadraticDepletionRate() {
        return quadraticDepletionRate;
    }

    public double getBaselineLevel() {
        return baselineLevel;
    }

    public double getTimeConstant() {
        return timeConstant;
    }

    /**
     * Compute effective time step for integration.
     */
    public double getEffectiveTimeStep() {
        return 1.0 / timeConstant;
    }

    /**
     * Compute equilibrium transmitter level for given signal.
     */
    public double computeEquilibrium(double signal) {
        double depletion = linearDepletionRate * signal +
                          quadraticDepletionRate * signal * signal;
        if (recoveryRate + depletion > 0) {
            return recoveryRate / (recoveryRate + depletion);
        }
        return 0.0;
    }

    /**
     * Compute signal threshold for 50% depletion.
     */
    public double computeHalfDepletionThreshold() {
        // At equilibrium: ε(1 - Z) = Z(λS + μS²)
        // For Z = 0.5: ε/2 = 0.5(λS + μS²)
        // λS + μS² = ε
        if (quadraticDepletionRate > 0) {
            double discriminant = linearDepletionRate * linearDepletionRate +
                                 4 * quadraticDepletionRate * recoveryRate;
            if (discriminant >= 0) {
                return (-linearDepletionRate + Math.sqrt(discriminant)) /
                       (2 * quadraticDepletionRate);
            }
        }
        if (linearDepletionRate > 0) {
            return recoveryRate / linearDepletionRate;
        }
        return Double.POSITIVE_INFINITY;
    }
}