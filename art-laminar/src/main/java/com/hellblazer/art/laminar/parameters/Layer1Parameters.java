package com.hellblazer.art.laminar.parameters;

/**
 * Parameters for Layer 1 (Top-Down Attentional Priming).
 * Layer 1 contains apical dendrites that receive top-down attentional signals
 * from higher cortical areas.
 *
 * Biological constraints:
 * - Very slow time constants (200-1000ms) for sustained attention
 * - Receives top-down signals from higher cortical areas
 * - Sustained attention effects that persist after input ends
 * - Priming without driving cells directly
 * - Integrates with Layer 2/3 apical dendrites
 * - Long-duration memory traces
 * - Lowest firing rates of all layers
 *
 * @author Hal Hildebrand
 */
public class Layer1Parameters implements LayerParameters {

    private final double timeConstant;          // 200-1000ms range
    private final double primingStrength;       // Strength of priming effect (0-1)
    private final double sustainedDecayRate;    // Very slow decay for persistence
    private final double apicalIntegration;     // Integration strength with Layer 2/3
    private final double attentionShiftRate;    // Rate of attention shifting
    private final double decayRate;
    private final double ceiling;
    private final double floor;
    private final double selfExcitation;
    private final double lateralInhibition;
    private final double maxFiringRate;         // Max biological firing rate (Hz)

    private Layer1Parameters(Builder builder) {
        // Validate Layer 1 specific constraints
        if (builder.timeConstant < 200.0 || builder.timeConstant > 1000.0) {
            throw new IllegalArgumentException(
                "Layer 1 time constant must be 200-1000ms, got: " + builder.timeConstant);
        }
        if (builder.primingStrength < 0.0 || builder.primingStrength > 1.0) {
            throw new IllegalArgumentException(
                "Priming strength must be 0-1, got: " + builder.primingStrength);
        }
        if (builder.sustainedDecayRate < 0.0 || builder.sustainedDecayRate > 0.01) {
            throw new IllegalArgumentException(
                "Sustained decay rate must be 0-0.01 for slow decay, got: " + builder.sustainedDecayRate);
        }
        if (builder.apicalIntegration < 0.0 || builder.apicalIntegration > 1.0) {
            throw new IllegalArgumentException(
                "Apical integration must be 0-1, got: " + builder.apicalIntegration);
        }
        if (builder.attentionShiftRate < 0.0 || builder.attentionShiftRate > 1.0) {
            throw new IllegalArgumentException(
                "Attention shift rate must be 0-1, got: " + builder.attentionShiftRate);
        }
        if (builder.floor > builder.ceiling) {
            throw new IllegalArgumentException("Floor must be <= ceiling");
        }
        if (builder.maxFiringRate <= 0.0 || builder.maxFiringRate > 50.0) {
            throw new IllegalArgumentException(
                "Layer 1 max firing rate must be 0-50Hz, got: " + builder.maxFiringRate);
        }

        this.timeConstant = builder.timeConstant;
        this.primingStrength = builder.primingStrength;
        this.sustainedDecayRate = builder.sustainedDecayRate;
        this.apicalIntegration = builder.apicalIntegration;
        this.attentionShiftRate = builder.attentionShiftRate;
        this.decayRate = 1.0 / builder.timeConstant; // Inverse relationship
        this.ceiling = builder.ceiling;
        this.floor = builder.floor;
        this.selfExcitation = builder.selfExcitation;
        this.lateralInhibition = builder.lateralInhibition;
        this.maxFiringRate = builder.maxFiringRate;
    }

    @Override
    public double getDecayRate() {
        return decayRate;
    }

    @Override
    public double getCeiling() {
        return ceiling;
    }

    @Override
    public double getFloor() {
        return floor;
    }

    @Override
    public double getSelfExcitation() {
        return selfExcitation;
    }

    @Override
    public double getLateralInhibition() {
        return lateralInhibition;
    }

    public double getTimeConstant() {
        return timeConstant;
    }

    public double getPrimingStrength() {
        return primingStrength;
    }

    public double getSustainedDecayRate() {
        return sustainedDecayRate;
    }

    public double getApicalIntegration() {
        return apicalIntegration;
    }

    public double getAttentionShiftRate() {
        return attentionShiftRate;
    }

    public double getMaxFiringRate() {
        return maxFiringRate;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private double timeConstant = 500.0;           // Mid-range default
        private double primingStrength = 0.3;          // 30% priming effect
        private double sustainedDecayRate = 0.001;     // Very slow decay
        private double apicalIntegration = 0.5;        // 50% integration
        private double attentionShiftRate = 0.3;       // Moderate shift rate
        private double ceiling = 1.0;
        private double floor = 0.0;
        private double selfExcitation = 0.05;          // Very weak self-excitation
        private double lateralInhibition = 0.05;       // Very weak lateral inhibition
        private double maxFiringRate = 30.0;           // Low firing rate for Layer 1

        public Builder timeConstant(double timeConstant) {
            if (timeConstant <= 0) {
                throw new IllegalArgumentException("Time constant must be positive");
            }
            this.timeConstant = timeConstant;
            return this;
        }

        public Builder primingStrength(double primingStrength) {
            this.primingStrength = primingStrength;
            return this;
        }

        public Builder sustainedDecayRate(double sustainedDecayRate) {
            this.sustainedDecayRate = sustainedDecayRate;
            return this;
        }

        public Builder apicalIntegration(double apicalIntegration) {
            this.apicalIntegration = apicalIntegration;
            return this;
        }

        public Builder attentionShiftRate(double attentionShiftRate) {
            this.attentionShiftRate = attentionShiftRate;
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

        public Builder lateralInhibition(double lateralInhibition) {
            this.lateralInhibition = lateralInhibition;
            return this;
        }

        public Builder maxFiringRate(double maxFiringRate) {
            this.maxFiringRate = maxFiringRate;
            return this;
        }

        public Layer1Parameters build() {
            return new Layer1Parameters(this);
        }
    }

    @Override
    public String toString() {
        return String.format("Layer1Parameters[timeConstant=%.1fms, priming=%.2f, sustainedDecay=%.4f, " +
                "apicalInteg=%.2f, shiftRate=%.2f, decay=%.4f, ceiling=%.2f, floor=%.2f, " +
                "selfExc=%.2f, latInhib=%.2f, maxRate=%.1fHz]",
            timeConstant, primingStrength, sustainedDecayRate,
            apicalIntegration, attentionShiftRate, decayRate, ceiling, floor,
            selfExcitation, lateralInhibition, maxFiringRate);
    }
}