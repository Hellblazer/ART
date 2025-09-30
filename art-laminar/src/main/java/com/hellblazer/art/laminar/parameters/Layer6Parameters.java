package com.hellblazer.art.laminar.parameters;

/**
 * Parameters for Layer 6 (Feedback Modulation).
 * Layer 6 provides modulatory feedback to Layer 4 and thalamus.
 *
 * CRITICAL: Implements ART matching rule - modulatory only!
 * - Cannot fire cells alone (requires bottom-up + top-down)
 * - On-center, off-surround dynamics
 * - Top-down expectation generation
 * - Attentional gain control
 *
 * Biological constraints:
 * - Slow time constants (100-500ms) for sustained modulation
 * - Modulatory signals that cannot drive responses alone
 * - Implements center-surround organization
 * - Lower firing rates than other layers
 *
 * @author Hal Hildebrand
 */
public class Layer6Parameters implements LayerParameters {

    private final double timeConstant;          // 100-500ms range
    private final double onCenterWeight;        // On-center excitation strength
    private final double offSurroundStrength;   // Off-surround inhibition strength
    private final double modulationThreshold;   // Threshold for modulatory effect
    private final double attentionalGain;       // Gain for attentional modulation
    private final double decayRate;
    private final double ceiling;
    private final double floor;
    private final double selfExcitation;
    private final double lateralInhibition;
    private final double maxFiringRate;         // Max biological firing rate (Hz)

    private Layer6Parameters(Builder builder) {
        // Validate Layer 6 specific constraints
        if (builder.timeConstant < 100.0 || builder.timeConstant > 500.0) {
            throw new IllegalArgumentException(
                "Layer 6 time constant must be 100-500ms, got: " + builder.timeConstant);
        }
        if (builder.onCenterWeight < 0.0) {
            throw new IllegalArgumentException(
                "On-center weight must be non-negative, got: " + builder.onCenterWeight);
        }
        if (builder.offSurroundStrength < 0.0) {
            throw new IllegalArgumentException(
                "Off-surround strength must be non-negative, got: " + builder.offSurroundStrength);
        }
        if (builder.modulationThreshold < 0.0 || builder.modulationThreshold > 1.0) {
            throw new IllegalArgumentException(
                "Modulation threshold must be 0-1, got: " + builder.modulationThreshold);
        }
        if (builder.attentionalGain < 0.0) {
            throw new IllegalArgumentException(
                "Attentional gain must be non-negative, got: " + builder.attentionalGain);
        }
        if (builder.floor > builder.ceiling) {
            throw new IllegalArgumentException("Floor must be <= ceiling");
        }
        if (builder.maxFiringRate <= 0.0 || builder.maxFiringRate > 100.0) {
            throw new IllegalArgumentException(
                "Layer 6 max firing rate must be 0-100Hz, got: " + builder.maxFiringRate);
        }

        this.timeConstant = builder.timeConstant;
        this.onCenterWeight = builder.onCenterWeight;
        this.offSurroundStrength = builder.offSurroundStrength;
        this.modulationThreshold = builder.modulationThreshold;
        this.attentionalGain = builder.attentionalGain;
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

    public double getOnCenterWeight() {
        return onCenterWeight;
    }

    public double getOffSurroundStrength() {
        return offSurroundStrength;
    }

    public double getModulationThreshold() {
        return modulationThreshold;
    }

    public double getAttentionalGain() {
        return attentionalGain;
    }

    public double getMaxFiringRate() {
        return maxFiringRate;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private double timeConstant = 200.0;          // Mid-range default
        private double onCenterWeight = 1.0;          // Unity on-center
        private double offSurroundStrength = 0.2;     // Weak off-surround
        private double modulationThreshold = 0.1;     // Low threshold for modulation
        private double attentionalGain = 1.0;         // Unity gain default
        private double ceiling = 1.0;
        private double floor = 0.0;
        private double selfExcitation = 0.1;          // Weak self-excitation
        private double lateralInhibition = 0.3;       // Moderate lateral inhibition
        private double maxFiringRate = 50.0;          // Lower firing rate for Layer 6

        public Builder timeConstant(double timeConstant) {
            if (timeConstant <= 0) {
                throw new IllegalArgumentException("Time constant must be positive");
            }
            this.timeConstant = timeConstant;
            return this;
        }

        public Builder onCenterWeight(double onCenterWeight) {
            this.onCenterWeight = onCenterWeight;
            return this;
        }

        public Builder offSurroundStrength(double offSurroundStrength) {
            this.offSurroundStrength = offSurroundStrength;
            return this;
        }

        public Builder modulationThreshold(double modulationThreshold) {
            this.modulationThreshold = modulationThreshold;
            return this;
        }

        public Builder attentionalGain(double attentionalGain) {
            this.attentionalGain = attentionalGain;
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

        public Layer6Parameters build() {
            return new Layer6Parameters(this);
        }
    }

    @Override
    public String toString() {
        return String.format("Layer6Parameters[timeConstant=%.1fms, onCenter=%.2f, offSurround=%.2f, " +
                "modThresh=%.2f, attnGain=%.2f, decay=%.3f, ceiling=%.2f, floor=%.2f, " +
                "selfExc=%.2f, latInhib=%.2f, maxRate=%.1fHz]",
            timeConstant, onCenterWeight, offSurroundStrength,
            modulationThreshold, attentionalGain, decayRate, ceiling, floor,
            selfExcitation, lateralInhibition, maxFiringRate);
    }
}