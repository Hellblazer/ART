package com.hellblazer.art.laminar.parameters;

/**
 * Parameters for Layer 4 (Thalamic Driving Input).
 * Layer 4 receives driving input from the thalamus and initiates cortical processing.
 *
 * Biological constraints:
 * - Fast time constants (10-50ms) for rapid response
 * - Strong driving signals that can fire cells
 * - Simple feedforward processing initially
 * - No lateral inhibition in basic implementation
 *
 * @author Hal Hildebrand
 */
public class Layer4Parameters implements LayerParameters {

    private final double timeConstant;      // 10-50ms range
    private final double drivingStrength;   // 0-1, strength of thalamic drive
    private final double decayRate;
    private final double ceiling;
    private final double floor;
    private final double selfExcitation;
    private final double lateralInhibition;

    private Layer4Parameters(Builder builder) {
        // Validate Layer 4 specific constraints
        if (builder.timeConstant < 10.0 || builder.timeConstant > 50.0) {
            throw new IllegalArgumentException(
                "Layer 4 time constant must be 10-50ms, got: " + builder.timeConstant);
        }
        if (builder.drivingStrength < 0.0 || builder.drivingStrength > 1.0) {
            throw new IllegalArgumentException(
                "Driving strength must be 0-1, got: " + builder.drivingStrength);
        }
        if (builder.floor > builder.ceiling) {
            throw new IllegalArgumentException(
                "Floor must be <= ceiling");
        }

        this.timeConstant = builder.timeConstant;
        this.drivingStrength = builder.drivingStrength;
        this.decayRate = 1.0 / builder.timeConstant; // Inverse relationship
        this.ceiling = builder.ceiling;
        this.floor = builder.floor;
        this.selfExcitation = builder.selfExcitation;
        this.lateralInhibition = builder.lateralInhibition;
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

    public double getDrivingStrength() {
        return drivingStrength;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private double timeConstant = 25.0;      // Mid-range default
        private double drivingStrength = 0.8;    // Strong driving
        private double ceiling = 1.0;
        private double floor = 0.0;
        private double selfExcitation = 0.3;
        private double lateralInhibition = 0.0;  // No lateral inhibition initially

        public Builder timeConstant(double timeConstant) {
            if (timeConstant <= 0) {
                throw new IllegalArgumentException("Time constant must be positive");
            }
            this.timeConstant = timeConstant;
            return this;
        }

        public Builder drivingStrength(double drivingStrength) {
            this.drivingStrength = drivingStrength;
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

        public Layer4Parameters build() {
            return new Layer4Parameters(this);
        }
    }

    @Override
    public String toString() {
        return String.format("Layer4Parameters[timeConstant=%.1fms, driving=%.2f, decay=%.3f, " +
                "ceiling=%.2f, floor=%.2f, selfExc=%.2f, latInhib=%.2f]",
            timeConstant, drivingStrength, decayRate, ceiling, floor,
            selfExcitation, lateralInhibition);
    }
}