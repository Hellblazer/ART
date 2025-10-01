package com.hellblazer.art.cortical.temporal;

/**
 * Parameters for working memory based on STORE 2 model from Kazerounian & Grossberg (2014).
 *
 * <p>Controls primacy gradient dynamics through:
 * <ul>
 *   <li>Capacity limits (Miller's 7±2 or Cowan's 4±1)</li>
 *   <li>Shunting dynamics parameters</li>
 *   <li>Transmitter habituation parameters</li>
 *   <li>Primacy gradient strength</li>
 * </ul>
 *
 * <p>The primacy gradient emerges from the interaction between:
 * <ul>
 *   <li>Position-dependent initial activations</li>
 *   <li>Transmitter depletion during encoding</li>
 *   <li>Competitive lateral inhibition</li>
 * </ul>
 *
 * @param capacity Memory capacity in items (3-15, Miller's 7±2)
 * @param decayRate Base decay rate A for shunting dynamics
 * @param maxActivation Upper bound B for activation
 * @param primacyDecayRate Exponential decay factor for primacy gradient
 * @param selfExcitation Self-excitation strength
 * @param lateralInhibition Lateral inhibition strength (competition)
 * @param transmitterRecoveryRate ε for transmitter recovery
 * @param transmitterDepletionLinear λ for linear transmitter depletion
 * @param transmitterDepletionQuadratic μ for quadratic transmitter depletion
 * @param retrievalThreshold Minimum activation for retrieval
 * @param timeStep Integration time step in seconds
 * @param overflowResetEnabled Whether to reset when capacity exceeded
 * @param itemDimension Dimensionality of pattern vectors
 *
 * @author Migrated from art-temporal/temporal-memory to art-cortical (Phase 2)
 */
public record WorkingMemoryParameters(
    int capacity,
    double decayRate,
    double maxActivation,
    double primacyDecayRate,
    double selfExcitation,
    double lateralInhibition,
    double transmitterRecoveryRate,
    double transmitterDepletionLinear,
    double transmitterDepletionQuadratic,
    double retrievalThreshold,
    double timeStep,
    boolean overflowResetEnabled,
    int itemDimension
) {
    /**
     * Compact constructor with comprehensive validation.
     */
    public WorkingMemoryParameters {
        if (capacity < 3 || capacity > 15) {
            throw new IllegalArgumentException(
                "Capacity must be in range [3, 15] (Miller's 7±2): " + capacity);
        }
        if (decayRate <= 0 || decayRate > 1.0) {
            throw new IllegalArgumentException(
                "Decay rate must be in (0, 1]: " + decayRate);
        }
        if (maxActivation <= 0) {
            throw new IllegalArgumentException(
                "Max activation must be positive: " + maxActivation);
        }
        if (primacyDecayRate < 0 || primacyDecayRate > 1.0) {
            throw new IllegalArgumentException(
                "Primacy decay rate must be in [0, 1]: " + primacyDecayRate);
        }
        if (selfExcitation < 0) {
            throw new IllegalArgumentException(
                "Self-excitation must be non-negative: " + selfExcitation);
        }
        if (lateralInhibition < 0) {
            throw new IllegalArgumentException(
                "Lateral inhibition must be non-negative: " + lateralInhibition);
        }
        if (transmitterRecoveryRate <= 0 || transmitterRecoveryRate > 0.1) {
            throw new IllegalArgumentException(
                "Transmitter recovery rate must be in (0, 0.1]: " + transmitterRecoveryRate);
        }
        if (transmitterDepletionLinear < 0 || transmitterDepletionLinear > 1.0) {
            throw new IllegalArgumentException(
                "Linear depletion must be in [0, 1]: " + transmitterDepletionLinear);
        }
        if (transmitterDepletionQuadratic < 0 || transmitterDepletionQuadratic > 1.0) {
            throw new IllegalArgumentException(
                "Quadratic depletion must be in [0, 1]: " + transmitterDepletionQuadratic);
        }
        if (retrievalThreshold < 0 || retrievalThreshold > 1.0) {
            throw new IllegalArgumentException(
                "Retrieval threshold must be in [0, 1]: " + retrievalThreshold);
        }
        if (timeStep <= 0 || timeStep > 0.1) {
            throw new IllegalArgumentException(
                "Time step must be in (0, 0.1]: " + timeStep);
        }
        if (itemDimension < 1) {
            throw new IllegalArgumentException(
                "Item dimension must be positive: " + itemDimension);
        }
    }

    // Derived properties for backward compatibility and convenience

    /**
     * Get primacy gradient strength (alias for primacyDecayRate).
     */
    public double getPrimacyGradient() {
        return primacyDecayRate;
    }

    /**
     * Get recency gradient strength (weaker than primacy).
     */
    public double getRecencyGradient() {
        return primacyDecayRate * 0.5;
    }

    /**
     * Get transmitter baseline level.
     */
    public double getTransmitterBaseline() {
        return 1.0;
    }

    /**
     * Get initial activation level for new items.
     */
    public double getInitialActivation() {
        return maxActivation * 0.8;
    }

    /**
     * Get transmitter recovery rate (epsilon - alias for consistency).
     */
    public double getTransmitterRecovery() {
        return transmitterRecoveryRate;
    }

    /**
     * Get combined transmitter depletion rate.
     */
    public double getTransmitterDepletion() {
        return transmitterDepletionLinear + transmitterDepletionQuadratic;
    }

    /**
     * Get recency boost factor for recent items.
     */
    public double getRecencyBoost() {
        return 1.2;
    }

    /**
     * Get competition strength for lateral interactions.
     */
    public double getCompetitionStrength() {
        return lateralInhibition * 2.0;
    }

    /**
     * Get activation decay rate (alias for backward compatibility).
     */
    public double getActivationDecay() {
        return decayRate;
    }

    /**
     * Get activation ceiling (alias for maxActivation).
     */
    public double ceiling() {
        return maxActivation;
    }

    // Static factory methods for common configurations

    /**
     * Create parameters matching the paper's default specifications.
     * Miller's capacity (7) with primacy gradient dynamics.
     */
    public static WorkingMemoryParameters paperDefaults() {
        return new WorkingMemoryParameters(
            7,      // capacity: Miller's magical number
            0.1,    // decayRate: moderate decay
            1.0,    // maxActivation: normalized
            0.1,    // primacyDecayRate: exponential decay with position
            0.2,    // selfExcitation: moderate self-excitation
            0.3,    // lateralInhibition: competitive dynamics
            0.005,  // transmitterRecoveryRate: very slow recovery (ε)
            0.1,    // transmitterDepletionLinear: linear depletion (λ)
            0.05,   // transmitterDepletionQuadratic: quadratic depletion (μ)
            0.1,    // retrievalThreshold: low threshold
            0.01,   // timeStep: 10ms integration step
            true,   // overflowResetEnabled: reset when full
            10      // itemDimension: default vector dimension
        );
    }

    /**
     * Create parameters for Cowan's 4±1 capacity.
     * Smaller capacity with adjusted competitive dynamics.
     */
    public static WorkingMemoryParameters cowansCapacity() {
        return new WorkingMemoryParameters(
            4,      // capacity: Cowan's 4±1
            0.1,    // decayRate
            1.0,    // maxActivation
            0.1,    // primacyDecayRate
            0.2,    // selfExcitation
            0.3,    // lateralInhibition
            0.005,  // transmitterRecoveryRate
            0.1,    // transmitterDepletionLinear
            0.05,   // transmitterDepletionQuadratic
            0.1,    // retrievalThreshold
            0.01,   // timeStep
            true,   // overflowResetEnabled
            10      // itemDimension
        );
    }

    /**
     * Create parameters for extended capacity (9 items).
     * Requires stronger competition and faster decay.
     */
    public static WorkingMemoryParameters extendedCapacity() {
        return new WorkingMemoryParameters(
            9,      // capacity: extended to 9 items
            0.15,   // decayRate: faster decay for more items
            1.0,    // maxActivation
            0.15,   // primacyDecayRate: stronger gradient needed
            0.15,   // selfExcitation: less self-excitation
            0.4,    // lateralInhibition: more competition
            0.003,  // transmitterRecoveryRate: even slower recovery
            0.15,   // transmitterDepletionLinear: faster depletion
            0.08,   // transmitterDepletionQuadratic
            0.15,   // retrievalThreshold: higher threshold
            0.01,   // timeStep
            true,   // overflowResetEnabled
            10      // itemDimension
        );
    }

    /**
     * Create builder for custom parameter configurations.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for WorkingMemoryParameters (for fluent construction).
     */
    public static class Builder {
        private int capacity = 7;
        private double decayRate = 0.1;
        private double maxActivation = 1.0;
        private double primacyDecayRate = 0.1;
        private double selfExcitation = 0.2;
        private double lateralInhibition = 0.3;
        private double transmitterRecoveryRate = 0.005;
        private double transmitterDepletionLinear = 0.1;
        private double transmitterDepletionQuadratic = 0.05;
        private double retrievalThreshold = 0.1;
        private double timeStep = 0.01;
        private boolean overflowResetEnabled = true;
        private int itemDimension = 10;

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

        public Builder primacyDecayRate(double rate) {
            this.primacyDecayRate = rate;
            return this;
        }

        public Builder selfExcitation(double selfExcitation) {
            this.selfExcitation = selfExcitation;
            return this;
        }

        public Builder lateralInhibition(double lateralInhibition) {
            this.lateralInhibition = lateralInhibition;
            return this;
        }

        public Builder transmitterRecoveryRate(double rate) {
            this.transmitterRecoveryRate = rate;
            return this;
        }

        public Builder transmitterDepletionLinear(double rate) {
            this.transmitterDepletionLinear = rate;
            return this;
        }

        public Builder transmitterDepletionQuadratic(double rate) {
            this.transmitterDepletionQuadratic = rate;
            return this;
        }

        public Builder retrievalThreshold(double threshold) {
            this.retrievalThreshold = threshold;
            return this;
        }

        public Builder timeStep(double timeStep) {
            this.timeStep = timeStep;
            return this;
        }

        public Builder overflowResetEnabled(boolean enabled) {
            this.overflowResetEnabled = enabled;
            return this;
        }

        public Builder itemDimension(int dimension) {
            this.itemDimension = dimension;
            return this;
        }

        public WorkingMemoryParameters build() {
            return new WorkingMemoryParameters(
                capacity, decayRate, maxActivation, primacyDecayRate,
                selfExcitation, lateralInhibition, transmitterRecoveryRate,
                transmitterDepletionLinear, transmitterDepletionQuadratic,
                retrievalThreshold, timeStep, overflowResetEnabled, itemDimension
            );
        }
    }
}
