package com.hellblazer.art.cortical.network;

/**
 * Parameters for Bipole Cell Network configuration.
 * Controls the three-way firing logic, horizontal connections, and boundary completion.
 *
 * <p>Bipole cells implement boundary completion via three firing conditions:
 * <ol>
 *   <li>Strong direct bottom-up activation alone</li>
 *   <li>Simultaneous horizontal inputs from BOTH sides (collinear)</li>
 *   <li>Both bottom-up AND horizontal inputs present</li>
 * </ol>
 *
 * <p>Biological constraints:
 * <ul>
 *   <li>Time constant: 30-150ms (Layer 1 dynamics)</li>
 *   <li>Horizontal range: 5-20 units (cortical distance)</li>
 *   <li>Exponential distance decay with sigma ~5 units</li>
 *   <li>Optional orientation selectivity (45-90 degree tuning)</li>
 * </ul>
 *
 * @param networkSize Total number of bipole cells in network
 * @param strongDirectThreshold Threshold for Condition 1 (strong direct input alone)
 * @param weakDirectThreshold Threshold for Condition 3 (weak direct + one side)
 * @param horizontalThreshold Threshold for horizontal inputs
 * @param maxHorizontalRange Maximum range for horizontal connections (5-20 units)
 * @param distanceSigma Sigma for exponential distance decay
 * @param maxWeight Maximum connection weight
 * @param orientationSelectivity Enable orientation-selective connections
 * @param orientationSigma Sigma for orientation tuning (radians)
 * @param timeConstant Time constant for bipole dynamics in seconds (0.03-0.15)
 *
 * @author Migrated from art-laminar to art-cortical (Phase 3, Milestone 2)
 */
public record BipoleCellParameters(
    int networkSize,
    double strongDirectThreshold,
    double weakDirectThreshold,
    double horizontalThreshold,
    int maxHorizontalRange,
    double distanceSigma,
    double maxWeight,
    boolean orientationSelectivity,
    double orientationSigma,
    double timeConstant
) {

    /**
     * Canonical constructor with validation.
     */
    public BipoleCellParameters {
        if (networkSize <= 0) {
            throw new IllegalArgumentException("Network size must be positive, got: " + networkSize);
        }
        if (strongDirectThreshold < 0.0 || strongDirectThreshold > 1.0) {
            throw new IllegalArgumentException(
                "Strong direct threshold must be 0-1, got: " + strongDirectThreshold);
        }
        if (weakDirectThreshold < 0.0 || weakDirectThreshold > 1.0) {
            throw new IllegalArgumentException(
                "Weak direct threshold must be 0-1, got: " + weakDirectThreshold);
        }
        if (horizontalThreshold < 0.0 || horizontalThreshold > 1.0) {
            throw new IllegalArgumentException(
                "Horizontal threshold must be 0-1, got: " + horizontalThreshold);
        }
        if (maxHorizontalRange < 1 || maxHorizontalRange > 50) {
            throw new IllegalArgumentException(
                "Max horizontal range must be 1-50, got: " + maxHorizontalRange);
        }
        if (distanceSigma <= 0.0) {
            throw new IllegalArgumentException(
                "Distance sigma must be positive, got: " + distanceSigma);
        }
        if (maxWeight <= 0.0) {
            throw new IllegalArgumentException(
                "Max weight must be positive, got: " + maxWeight);
        }
        if (orientationSigma <= 0.0) {
            throw new IllegalArgumentException(
                "Orientation sigma must be positive, got: " + orientationSigma);
        }
        if (timeConstant <= 0.0) {
            throw new IllegalArgumentException(
                "Time constant must be positive, got: " + timeConstant);
        }
    }

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
