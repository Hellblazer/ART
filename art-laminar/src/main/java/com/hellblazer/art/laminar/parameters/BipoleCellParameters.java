package com.hellblazer.art.laminar.parameters;

/**
 * Parameters for Bipole Cell Network configuration.
 * Controls the three-way firing logic, horizontal connections, and boundary completion.
 *
 * @author Hal Hildebrand
 */
public record BipoleCellParameters(
    int networkSize,
    double strongDirectThreshold,    // Threshold for Condition 1: Strong direct input alone
    double weakDirectThreshold,      // Threshold for Condition 3: Weak direct + one side
    double horizontalThreshold,      // Threshold for horizontal inputs
    int maxHorizontalRange,          // Maximum range for horizontal connections (5-20 units)
    double distanceSigma,            // Sigma for exponential distance decay
    double maxWeight,                // Maximum connection weight
    boolean orientationSelectivity,  // Enable orientation-selective connections
    double orientationSigma,         // Sigma for orientation tuning
    double timeConstant              // Time constant for bipole dynamics (30-150ms)
) {

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private int networkSize = 100;
        private double strongDirectThreshold = 0.8;
        private double weakDirectThreshold = 0.3;
        private double horizontalThreshold = 0.5;
        private int maxHorizontalRange = 10;
        private double distanceSigma = 5.0;
        private double maxWeight = 1.0;
        private boolean orientationSelectivity = true;
        private double orientationSigma = Math.PI / 4;  // 45 degrees
        private double timeConstant = 0.05;  // 50ms default

        public Builder networkSize(int networkSize) {
            this.networkSize = networkSize;
            return this;
        }

        public Builder strongDirectThreshold(double strongDirectThreshold) {
            this.strongDirectThreshold = strongDirectThreshold;
            return this;
        }

        public Builder weakDirectThreshold(double weakDirectThreshold) {
            this.weakDirectThreshold = weakDirectThreshold;
            return this;
        }

        public Builder horizontalThreshold(double horizontalThreshold) {
            this.horizontalThreshold = horizontalThreshold;
            return this;
        }

        public Builder maxHorizontalRange(int maxHorizontalRange) {
            this.maxHorizontalRange = maxHorizontalRange;
            return this;
        }

        public Builder distanceSigma(double distanceSigma) {
            this.distanceSigma = distanceSigma;
            return this;
        }

        public Builder maxWeight(double maxWeight) {
            this.maxWeight = maxWeight;
            return this;
        }

        public Builder orientationSelectivity(boolean orientationSelectivity) {
            this.orientationSelectivity = orientationSelectivity;
            return this;
        }

        public Builder orientationSigma(double orientationSigma) {
            this.orientationSigma = orientationSigma;
            return this;
        }

        public Builder timeConstant(double timeConstant) {
            this.timeConstant = timeConstant;
            return this;
        }

        public BipoleCellParameters build() {
            return new BipoleCellParameters(
                networkSize,
                strongDirectThreshold,
                weakDirectThreshold,
                horizontalThreshold,
                maxHorizontalRange,
                distanceSigma,
                maxWeight,
                orientationSelectivity,
                orientationSigma,
                timeConstant
            );
        }
    }
}