package com.hellblazer.art.temporal.memory;

import com.hellblazer.art.temporal.core.Parameters;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Parameters for working memory based on STORE 2 model.
 * Controls primacy gradient, capacity, and dynamics.
 */
public class WorkingMemoryParameters implements Parameters {
    private final int capacity;                  // Memory capacity (7±2 items)
    private final double decayRate;              // Base decay rate (A)
    private final double maxActivation;          // Upper bound (B)
    private final double primacyDecayRate;       // How quickly primacy effect decays
    private final double selfExcitation;         // Self-excitation strength
    private final double lateralInhibition;      // Lateral inhibition strength
    private final double transmitterRecoveryRate;    // ε for transmitter recovery
    private final double transmitterDepletionLinear; // λ for linear depletion
    private final double transmitterDepletionQuadratic; // μ for quadratic depletion
    private final double retrievalThreshold;     // Minimum activation for retrieval
    private final double timeStep;               // Integration time step
    private final boolean overflowResetEnabled;  // Reset when capacity exceeded

    private WorkingMemoryParameters(Builder builder) {
        this.capacity = builder.capacity;
        this.decayRate = builder.decayRate;
        this.maxActivation = builder.maxActivation;
        this.primacyDecayRate = builder.primacyDecayRate;
        this.selfExcitation = builder.selfExcitation;
        this.lateralInhibition = builder.lateralInhibition;
        this.transmitterRecoveryRate = builder.transmitterRecoveryRate;
        this.transmitterDepletionLinear = builder.transmitterDepletionLinear;
        this.transmitterDepletionQuadratic = builder.transmitterDepletionQuadratic;
        this.retrievalThreshold = builder.retrievalThreshold;
        this.timeStep = builder.timeStep;
        this.overflowResetEnabled = builder.overflowResetEnabled;
        validate();
    }

    @Override
    public void validate() {
        if (capacity < 3 || capacity > 15) {
            throw new IllegalArgumentException("Capacity should be in range [3, 15] (Miller's 7±2)");
        }
        if (decayRate <= 0 || decayRate > 1.0) {
            throw new IllegalArgumentException("Decay rate must be in (0, 1]");
        }
        if (maxActivation <= 0) {
            throw new IllegalArgumentException("Max activation must be positive");
        }
        if (primacyDecayRate < 0 || primacyDecayRate > 1.0) {
            throw new IllegalArgumentException("Primacy decay rate must be in [0, 1]");
        }
        if (selfExcitation < 0) {
            throw new IllegalArgumentException("Self-excitation must be non-negative");
        }
        if (lateralInhibition < 0) {
            throw new IllegalArgumentException("Lateral inhibition must be non-negative");
        }
        if (transmitterRecoveryRate <= 0 || transmitterRecoveryRate > 0.1) {
            throw new IllegalArgumentException("Transmitter recovery rate must be in (0, 0.1]");
        }
        if (transmitterDepletionLinear < 0 || transmitterDepletionLinear > 1.0) {
            throw new IllegalArgumentException("Linear depletion must be in [0, 1]");
        }
        if (transmitterDepletionQuadratic < 0 || transmitterDepletionQuadratic > 1.0) {
            throw new IllegalArgumentException("Quadratic depletion must be in [0, 1]");
        }
        if (retrievalThreshold < 0 || retrievalThreshold > 1.0) {
            throw new IllegalArgumentException("Retrieval threshold must be in [0, 1]");
        }
        if (timeStep <= 0 || timeStep > 0.1) {
            throw new IllegalArgumentException("Time step must be in (0, 0.1]");
        }
    }

    @Override
    public Optional<Double> getParameter(String name) {
        return Optional.ofNullable(getAllParameters().get(name));
    }

    @Override
    public Map<String, Double> getAllParameters() {
        var params = new HashMap<String, Double>();
        params.put("capacity", (double) capacity);
        params.put("decayRate", decayRate);
        params.put("maxActivation", maxActivation);
        params.put("primacyDecayRate", primacyDecayRate);
        params.put("selfExcitation", selfExcitation);
        params.put("lateralInhibition", lateralInhibition);
        params.put("transmitterRecoveryRate", transmitterRecoveryRate);
        params.put("transmitterDepletionLinear", transmitterDepletionLinear);
        params.put("transmitterDepletionQuadratic", transmitterDepletionQuadratic);
        params.put("retrievalThreshold", retrievalThreshold);
        params.put("timeStep", timeStep);
        params.put("overflowResetEnabled", overflowResetEnabled ? 1.0 : 0.0);
        return params;
    }

    @Override
    public Parameters withParameter(String name, double value) {
        var builder = toBuilder();

        switch (name) {
            case "capacity" -> builder.capacity((int) value);
            case "decayRate" -> builder.decayRate(value);
            case "maxActivation" -> builder.maxActivation(value);
            case "primacyDecayRate" -> builder.primacyDecayRate(value);
            case "selfExcitation" -> builder.selfExcitation(value);
            case "lateralInhibition" -> builder.lateralInhibition(value);
            case "transmitterRecoveryRate" -> builder.transmitterRecoveryRate(value);
            case "transmitterDepletionLinear" -> builder.transmitterDepletionLinear(value);
            case "transmitterDepletionQuadratic" -> builder.transmitterDepletionQuadratic(value);
            case "retrievalThreshold" -> builder.retrievalThreshold(value);
            case "timeStep" -> builder.timeStep(value);
            case "overflowResetEnabled" -> builder.overflowResetEnabled(value > 0.5);
            default -> throw new IllegalArgumentException("Unknown parameter: " + name);
        }

        return builder.build();
    }

    // Getters
    public int getCapacity() { return capacity; }
    public double getDecayRate() { return decayRate; }
    public double getMaxActivation() { return maxActivation; }
    public double getPrimacyDecayRate() { return primacyDecayRate; }
    public double getSelfExcitation() { return selfExcitation; }
    public double getLateralInhibition() { return lateralInhibition; }
    public double getTransmitterRecoveryRate() { return transmitterRecoveryRate; }
    public double getTransmitterDepletionLinear() { return transmitterDepletionLinear; }
    public double getTransmitterDepletionQuadratic() { return transmitterDepletionQuadratic; }
    public double getRetrievalThreshold() { return retrievalThreshold; }
    public double getTimeStep() { return timeStep; }
    public boolean isOverflowResetEnabled() { return overflowResetEnabled; }

    /**
     * Get item dimension (assumed fixed for simplicity).
     */
    public int getItemDimension() {
        return 10; // Default dimension for pattern vectors
    }

    /**
     * Get primacy gradient strength.
     */
    public double getPrimacyGradient() {
        return primacyDecayRate;
    }

    /**
     * Get recency gradient strength.
     */
    public double getRecencyGradient() {
        return primacyDecayRate * 0.5; // Weaker than primacy
    }

    /**
     * Get transmitter baseline level for vectorized operations.
     */
    public double getTransmitterBaseline() {
        return 1.0; // Default baseline transmitter level
    }

    /**
     * Get initial activation level for new items.
     */
    public double getInitialActivation() {
        return maxActivation * 0.8; // 80% of max activation
    }

    /**
     * Get transmitter recovery rate (epsilon).
     */
    public double getTransmitterRecovery() {
        return transmitterRecoveryRate;
    }

    /**
     * Get transmitter depletion rate (combined linear + quadratic).
     */
    public double getTransmitterDepletion() {
        return transmitterDepletionLinear + transmitterDepletionQuadratic;
    }

    /**
     * Get recency boost factor for recent items.
     */
    public double getRecencyBoost() {
        return 1.2; // 20% boost for recent items
    }

    /**
     * Get competition strength for lateral interactions.
     */
    public double getCompetitionStrength() {
        return lateralInhibition * 2.0; // Competition based on inhibition
    }

    /**
     * Get activation decay rate.
     */
    public double getActivationDecay() {
        return decayRate;
    }

    private Builder toBuilder() {
        return new Builder()
            .capacity(capacity)
            .decayRate(decayRate)
            .maxActivation(maxActivation)
            .primacyDecayRate(primacyDecayRate)
            .selfExcitation(selfExcitation)
            .lateralInhibition(lateralInhibition)
            .transmitterRecoveryRate(transmitterRecoveryRate)
            .transmitterDepletionLinear(transmitterDepletionLinear)
            .transmitterDepletionQuadratic(transmitterDepletionQuadratic)
            .retrievalThreshold(retrievalThreshold)
            .timeStep(timeStep)
            .overflowResetEnabled(overflowResetEnabled);
    }

    public static Builder builder() {
        return new Builder();
    }

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

        public WorkingMemoryParameters build() {
            return new WorkingMemoryParameters(this);
        }
    }

    /**
     * Create parameters matching the paper's specifications.
     */
    public static WorkingMemoryParameters paperDefaults() {
        return builder()
            .capacity(7)                        // Miller's magical number
            .decayRate(0.1)                     // From paper
            .maxActivation(1.0)                 // Normalized
            .primacyDecayRate(0.1)              // Exponential decay with position
            .selfExcitation(0.2)                // Moderate self-excitation
            .lateralInhibition(0.3)             // Competitive dynamics
            .transmitterRecoveryRate(0.005)    // Very slow recovery (ε)
            .transmitterDepletionLinear(0.1)   // Linear depletion (λ)
            .transmitterDepletionQuadratic(0.05) // Quadratic depletion (μ)
            .retrievalThreshold(0.1)           // Low threshold
            .timeStep(0.01)                    // 10ms time step
            .overflowResetEnabled(true)        // Reset when full
            .build();
    }

    /**
     * Create parameters for Cowan's 4±1 capacity.
     */
    public static WorkingMemoryParameters cowansCapacity() {
        return (WorkingMemoryParameters) paperDefaults().withParameter("capacity", 4);
    }

    /**
     * Create parameters for extended capacity (9 items).
     */
    public static WorkingMemoryParameters extendedCapacity() {
        return builder()
            .capacity(9)
            .decayRate(0.15)                   // Faster decay for more items
            .maxActivation(1.0)
            .primacyDecayRate(0.15)            // Stronger gradient needed
            .selfExcitation(0.15)              // Less self-excitation
            .lateralInhibition(0.4)            // More competition
            .transmitterRecoveryRate(0.003)   // Even slower recovery
            .transmitterDepletionLinear(0.15)  // Faster depletion
            .transmitterDepletionQuadratic(0.08)
            .retrievalThreshold(0.15)          // Higher threshold
            .timeStep(0.01)
            .overflowResetEnabled(true)
            .build();
    }
}