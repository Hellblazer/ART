package com.hellblazer.art.laminar.parameters;

/**
 * Parameters for Layer 5 (Output to Higher Areas).
 * Layer 5 projects processed signals from Layer 2/3 to higher cortical areas.
 *
 * Biological constraints:
 * - Medium time constants (50-200ms)
 * - Receives input from Layer 2/3 pyramidal cells
 * - Amplification/gating for salient features
 * - Output normalization for stable signaling
 * - Category signal generation
 * - Burst firing capability for important signals
 *
 * @author Hal Hildebrand
 */
public class Layer5Parameters implements LayerParameters {

    private final double timeConstant;           // 50-200ms range
    private final double amplificationGain;      // Signal amplification factor
    private final double outputGain;             // Output scaling factor
    private final double outputNormalization;    // Normalization strength
    private final double categoryThreshold;      // Threshold for category detection
    private final double burstThreshold;         // Threshold for burst firing
    private final double burstAmplification;     // Amplification during burst
    private final double decayRate;
    private final double ceiling;
    private final double floor;
    private final double selfExcitation;
    private final double lateralInhibition;
    private final double maxFiringRate;          // Max biological firing rate (Hz)

    private Layer5Parameters(Builder builder) {
        // Validate Layer 5 specific constraints
        if (builder.timeConstant < 50.0 || builder.timeConstant > 200.0) {
            throw new IllegalArgumentException(
                "Layer 5 time constant must be 50-200ms, got: " + builder.timeConstant);
        }
        if (builder.amplificationGain < 0.0) {
            throw new IllegalArgumentException(
                "Amplification gain must be non-negative, got: " + builder.amplificationGain);
        }
        if (builder.outputGain < 0.0) {
            throw new IllegalArgumentException(
                "Output gain must be non-negative, got: " + builder.outputGain);
        }
        if (builder.categoryThreshold < 0.0 || builder.categoryThreshold > 1.0) {
            throw new IllegalArgumentException(
                "Category threshold must be 0-1, got: " + builder.categoryThreshold);
        }
        if (builder.burstThreshold < 0.0 || builder.burstThreshold > 1.0) {
            throw new IllegalArgumentException(
                "Burst threshold must be 0-1, got: " + builder.burstThreshold);
        }
        if (builder.burstAmplification < 1.0) {
            throw new IllegalArgumentException(
                "Burst amplification must be >= 1.0, got: " + builder.burstAmplification);
        }
        if (builder.floor > builder.ceiling) {
            throw new IllegalArgumentException("Floor must be <= ceiling");
        }
        if (builder.maxFiringRate <= 0.0 || builder.maxFiringRate > 200.0) {
            throw new IllegalArgumentException(
                "Max firing rate must be 0-200Hz, got: " + builder.maxFiringRate);
        }

        this.timeConstant = builder.timeConstant;
        this.amplificationGain = builder.amplificationGain;
        this.outputGain = builder.outputGain;
        this.outputNormalization = builder.outputNormalization;
        this.categoryThreshold = builder.categoryThreshold;
        this.burstThreshold = builder.burstThreshold;
        this.burstAmplification = builder.burstAmplification;
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

    public double getAmplificationGain() {
        return amplificationGain;
    }

    public double getOutputGain() {
        return outputGain;
    }

    public double getOutputNormalization() {
        return outputNormalization;
    }

    public double getCategoryThreshold() {
        return categoryThreshold;
    }

    public double getBurstThreshold() {
        return burstThreshold;
    }

    public double getBurstAmplification() {
        return burstAmplification;
    }

    public double getMaxFiringRate() {
        return maxFiringRate;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private double timeConstant = 100.0;           // Mid-range default
        private double amplificationGain = 1.5;        // Moderate amplification
        private double outputGain = 1.0;               // Unity gain default
        private double outputNormalization = 0.01;     // Weak normalization
        private double categoryThreshold = 0.5;        // Mid-threshold
        private double burstThreshold = 0.8;           // High threshold for bursting
        private double burstAmplification = 2.0;       // Double during burst
        private double ceiling = 1.0;
        private double floor = 0.0;
        private double selfExcitation = 0.2;
        private double lateralInhibition = 0.1;        // Weak lateral inhibition
        private double maxFiringRate = 100.0;          // 100Hz max

        public Builder timeConstant(double timeConstant) {
            if (timeConstant <= 0) {
                throw new IllegalArgumentException("Time constant must be positive");
            }
            this.timeConstant = timeConstant;
            return this;
        }

        public Builder amplificationGain(double amplificationGain) {
            this.amplificationGain = amplificationGain;
            return this;
        }

        public Builder outputGain(double outputGain) {
            this.outputGain = outputGain;
            return this;
        }

        public Builder outputNormalization(double outputNormalization) {
            this.outputNormalization = outputNormalization;
            return this;
        }

        public Builder categoryThreshold(double categoryThreshold) {
            this.categoryThreshold = categoryThreshold;
            return this;
        }

        public Builder burstThreshold(double burstThreshold) {
            this.burstThreshold = burstThreshold;
            return this;
        }

        public Builder burstAmplification(double burstAmplification) {
            this.burstAmplification = burstAmplification;
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

        public Layer5Parameters build() {
            return new Layer5Parameters(this);
        }
    }

    @Override
    public String toString() {
        return String.format("Layer5Parameters[timeConstant=%.1fms, ampGain=%.2f, outGain=%.2f, " +
                "outNorm=%.3f, catThresh=%.2f, burstThresh=%.2f, burstAmp=%.2f, " +
                "decay=%.3f, ceiling=%.2f, floor=%.2f, selfExc=%.2f, latInhib=%.2f, maxRate=%.1fHz]",
            timeConstant, amplificationGain, outputGain, outputNormalization,
            categoryThreshold, burstThreshold, burstAmplification,
            decayRate, ceiling, floor, selfExcitation, lateralInhibition, maxFiringRate);
    }
}