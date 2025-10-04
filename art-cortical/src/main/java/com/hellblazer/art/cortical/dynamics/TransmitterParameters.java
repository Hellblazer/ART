package com.hellblazer.art.cortical.dynamics;

/**
 * Parameters for transmitter habituation dynamics (gating mechanism).
 * Based on Kazerounian & Grossberg (2014) Equation 2:
 *
 * <pre>
 * dZ_i/dt = ε(1 - Z_i) - Z_i(λ * S_i + μ * S_i²)
 * </pre>
 *
 * Where:
 * <ul>
 *   <li>Z_i: transmitter level for unit i (range [0,1])</li>
 *   <li>ε: recovery rate (slow, typically 0.01)</li>
 *   <li>λ: linear depletion rate</li>
 *   <li>μ: quadratic depletion rate</li>
 *   <li>S_i: signal/activation level</li>
 * </ul>
 *
 * <p>Habituation mechanism:
 * <ul>
 *   <li>Transmitter depletes with usage (λ and μ terms)</li>
 *   <li>Transmitter recovers slowly toward baseline (ε term)</li>
 *   <li>Creates primacy gradient effect in temporal sequences</li>
 *   <li>Earlier items deplete transmitters more than later items</li>
 * </ul>
 *
 * @param recoveryRate ε - recovery toward baseline (typically 0.01)
 * @param linearDepletionRate λ - linear depletion term (typically 1.0)
 * @param quadraticDepletionRate μ - quadratic depletion term (typically 0.5)
 * @param baselineLevel resting transmitter level (typically 1.0)
 * @param timeConstant temporal scale in ms (typically 500ms)
 *
 * @author Migrated from art-temporal/temporal-dynamics to art-cortical (Phase 1)
 */
public record TransmitterParameters(
    double recoveryRate,
    double linearDepletionRate,
    double quadraticDepletionRate,
    double baselineLevel,
    double timeConstant
) {

    /**
     * Compact canonical constructor with validation.
     */
    public TransmitterParameters {
        if (recoveryRate < 0 || recoveryRate > 1) {
            throw new IllegalArgumentException("Recovery rate must be in [0, 1]: " + recoveryRate);
        }
        if (linearDepletionRate < 0) {
            throw new IllegalArgumentException("Linear depletion rate must be non-negative: " + linearDepletionRate);
        }
        if (quadraticDepletionRate < 0) {
            throw new IllegalArgumentException("Quadratic depletion rate must be non-negative: " + quadraticDepletionRate);
        }
        if (baselineLevel < 0 || baselineLevel > 1) {
            throw new IllegalArgumentException("Baseline level must be in [0, 1]: " + baselineLevel);
        }
        if (timeConstant <= 0) {
            throw new IllegalArgumentException("Time constant must be positive: " + timeConstant);
        }
    }

    /**
     * Create default parameters from Kazerounian & Grossberg (2014) paper.
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
     * Create parameters for fast habituation (strong primacy effect).
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
     * Create parameters for slow habituation (weak primacy effect).
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
     * Create parameters for no habituation (constant transmitter, no primacy).
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

    /**
     * Compute effective time step for integration.
     */
    public double getEffectiveTimeStep() {
        return 1.0 / timeConstant;
    }

    /**
     * Compute equilibrium transmitter level for given signal strength.
     * At equilibrium: dZ/dt = 0, so ε(1 - Z) = Z(λS + μS²)
     */
    public double computeEquilibrium(double signal) {
        var depletion = linearDepletionRate * signal +
                       quadraticDepletionRate * signal * signal;
        if (recoveryRate + depletion > 0) {
            return recoveryRate / (recoveryRate + depletion);
        }
        return 0.0;
    }

    /**
     * Compute signal threshold for 50% depletion (half-maximum).
     * Useful for understanding habituation strength.
     */
    public double computeHalfDepletionThreshold() {
        // At equilibrium with Z = 0.5:
        // ε(1 - 0.5) = 0.5(λS + μS²)
        // ε/2 = 0.5(λS + μS²)
        // ε = λS + μS²

        if (quadraticDepletionRate > 0) {
            var discriminant = linearDepletionRate * linearDepletionRate +
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

    /**
     * Create a builder for constructing parameters.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for TransmitterParameters using fluent interface.
     */
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
            return new TransmitterParameters(
                recoveryRate,
                linearDepletionRate,
                quadraticDepletionRate,
                baselineLevel,
                timeConstant
            );
        }
    }
}
