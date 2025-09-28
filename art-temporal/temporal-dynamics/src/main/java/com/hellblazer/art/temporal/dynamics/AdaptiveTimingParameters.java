package com.hellblazer.art.temporal.dynamics;

/**
 * Parameters for adaptive timing dynamics.
 * Based on spectral timing theory (Grossberg & Schmajuk, 1989).
 */
public class AdaptiveTimingParameters {

    private final double minInterval;
    private final double maxInterval;
    private final double spectralWidth;
    private final double timeConstantScale;
    private final double learningRate;
    private double targetInterval;
    private final boolean learningEnabled;

    private AdaptiveTimingParameters(Builder builder) {
        this.minInterval = builder.minInterval;
        this.maxInterval = builder.maxInterval;
        this.spectralWidth = builder.spectralWidth;
        this.timeConstantScale = builder.timeConstantScale;
        this.learningRate = builder.learningRate;
        this.targetInterval = builder.targetInterval;
        this.learningEnabled = builder.learningEnabled;
        validate();
    }

    /**
     * Create default parameters for speech timing (50-500ms).
     */
    public static AdaptiveTimingParameters speechDefaults() {
        return builder()
            .minInterval(0.05)       // 50ms
            .maxInterval(0.5)        // 500ms
            .spectralWidth(0.2)      // 20% of peak time
            .timeConstantScale(0.1)  // Fast dynamics
            .learningRate(0.1)
            .learningEnabled(true)
            .build();
    }

    /**
     * Create parameters for music timing (100ms - 2s).
     */
    public static AdaptiveTimingParameters musicDefaults() {
        return builder()
            .minInterval(0.1)        // 100ms
            .maxInterval(2.0)        // 2 seconds
            .spectralWidth(0.15)     // Tighter timing
            .timeConstantScale(0.05) // Very fast dynamics
            .learningRate(0.2)       // Faster learning
            .learningEnabled(true)
            .build();
    }

    /**
     * Create parameters for motor timing (200ms - 5s).
     */
    public static AdaptiveTimingParameters motorDefaults() {
        return builder()
            .minInterval(0.2)        // 200ms
            .maxInterval(5.0)        // 5 seconds
            .spectralWidth(0.25)     // Broader timing
            .timeConstantScale(0.2)  // Slower dynamics
            .learningRate(0.05)      // Slower learning
            .learningEnabled(true)
            .build();
    }

    /**
     * Create parameters for interval timing (1s - 60s).
     */
    public static AdaptiveTimingParameters intervalDefaults() {
        return builder()
            .minInterval(1.0)        // 1 second
            .maxInterval(60.0)       // 1 minute
            .spectralWidth(0.3)      // Broad timing
            .timeConstantScale(0.5)  // Very slow dynamics
            .learningRate(0.01)      // Very slow learning
            .learningEnabled(true)
            .build();
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private double minInterval = 0.05;
        private double maxInterval = 1.0;
        private double spectralWidth = 0.2;
        private double timeConstantScale = 0.1;
        private double learningRate = 0.1;
        private double targetInterval = 0.0;
        private boolean learningEnabled = true;

        public Builder minInterval(double interval) {
            this.minInterval = interval;
            return this;
        }

        public Builder maxInterval(double interval) {
            this.maxInterval = interval;
            return this;
        }

        public Builder spectralWidth(double width) {
            this.spectralWidth = width;
            return this;
        }

        public Builder timeConstantScale(double scale) {
            this.timeConstantScale = scale;
            return this;
        }

        public Builder learningRate(double rate) {
            this.learningRate = rate;
            return this;
        }

        public Builder targetInterval(double interval) {
            this.targetInterval = interval;
            return this;
        }

        public Builder learningEnabled(boolean enabled) {
            this.learningEnabled = enabled;
            return this;
        }

        public AdaptiveTimingParameters build() {
            return new AdaptiveTimingParameters(this);
        }
    }

    private void validate() {
        if (minInterval <= 0 || maxInterval <= 0) {
            throw new IllegalArgumentException("Intervals must be positive");
        }
        if (minInterval >= maxInterval) {
            throw new IllegalArgumentException("Min interval must be less than max interval");
        }
        if (spectralWidth <= 0 || spectralWidth > 1) {
            throw new IllegalArgumentException("Spectral width must be in (0, 1]");
        }
        if (timeConstantScale <= 0 || timeConstantScale > 1) {
            throw new IllegalArgumentException("Time constant scale must be in (0, 1]");
        }
        if (learningRate < 0 || learningRate > 1) {
            throw new IllegalArgumentException("Learning rate must be in [0, 1]");
        }
        if (targetInterval < 0) {
            throw new IllegalArgumentException("Target interval must be non-negative");
        }
    }

    // Getters
    public double getMinInterval() {
        return minInterval;
    }

    public double getMaxInterval() {
        return maxInterval;
    }

    public double getSpectralWidth() {
        return spectralWidth;
    }

    public double getTimeConstantScale() {
        return timeConstantScale;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public double getTargetInterval() {
        return targetInterval;
    }

    public boolean isLearningEnabled() {
        return learningEnabled;
    }

    /**
     * Set target interval (mutable for online learning).
     */
    public void setTargetInterval(double interval) {
        if (interval < 0) {
            throw new IllegalArgumentException("Target interval must be non-negative");
        }
        this.targetInterval = interval;
    }

    /**
     * Get number of spectral components needed for given resolution.
     */
    public int computeRequiredDimension(double resolution) {
        double logRange = Math.log(maxInterval / minInterval);
        double logResolution = Math.log(1.0 + resolution);
        return (int) Math.ceil(logRange / logResolution) + 1;
    }

    /**
     * Check if interval is within timing range.
     */
    public boolean isInRange(double interval) {
        return interval >= minInterval && interval <= maxInterval;
    }
}