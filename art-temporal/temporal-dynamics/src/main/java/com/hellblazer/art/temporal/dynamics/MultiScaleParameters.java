package com.hellblazer.art.temporal.dynamics;

/**
 * Parameters for multi-scale dynamics integration.
 * Coordinates the different time scales in temporal ART.
 */
public class MultiScaleParameters {

    private final int dimension;
    private final ShuntingParameters shuntingParameters;
    private final TransmitterParameters transmitterParameters;
    private final AdaptiveTimingParameters timingParameters;
    private final int transmitterUpdateRatio;
    private final int timingUpdateRatio;
    private final double resetDecayFactor;
    private final boolean timingGatingEnabled;

    private MultiScaleParameters(Builder builder) {
        this.dimension = builder.dimension;
        this.shuntingParameters = builder.shuntingParameters;
        this.transmitterParameters = builder.transmitterParameters;
        this.timingParameters = builder.timingParameters;
        this.transmitterUpdateRatio = builder.transmitterUpdateRatio;
        this.timingUpdateRatio = builder.timingUpdateRatio;
        this.resetDecayFactor = builder.resetDecayFactor;
        this.timingGatingEnabled = builder.timingGatingEnabled;
        validate();
    }

    /**
     * Create default parameters for temporal ART.
     */
    public static MultiScaleParameters defaults(int dimension) {
        return builder(dimension)
            .shuntingParameters(ShuntingParameters.competitiveDefaults(dimension))
            .transmitterParameters(TransmitterParameters.paperDefaults())
            .timingParameters(AdaptiveTimingParameters.speechDefaults())
            .transmitterUpdateRatio(10)  // Update every 10 shunting steps
            .timingUpdateRatio(100)      // Update every 100 shunting steps
            .resetDecayFactor(0.3)
            .timingGatingEnabled(false)
            .build();
    }

    /**
     * Create parameters for speech processing.
     */
    public static MultiScaleParameters speechDefaults(int dimension) {
        return builder(dimension)
            .shuntingParameters(ShuntingParameters.competitiveDefaults(dimension))
            .transmitterParameters(TransmitterParameters.fastHabituation())
            .timingParameters(AdaptiveTimingParameters.speechDefaults())
            .transmitterUpdateRatio(5)
            .timingUpdateRatio(50)
            .resetDecayFactor(0.2)
            .timingGatingEnabled(true)
            .build();
    }

    /**
     * Create parameters for list learning.
     */
    public static MultiScaleParameters listLearningDefaults(int dimension) {
        return builder(dimension)
            .shuntingParameters(ShuntingParameters.winnerTakeAllDefaults(dimension))
            .transmitterParameters(TransmitterParameters.slowHabituation())
            .timingParameters(AdaptiveTimingParameters.intervalDefaults())
            .transmitterUpdateRatio(20)
            .timingUpdateRatio(200)
            .resetDecayFactor(0.5)
            .timingGatingEnabled(false)
            .build();
    }

    /**
     * Create parameters for motor sequence learning.
     */
    public static MultiScaleParameters motorDefaults(int dimension) {
        return builder(dimension)
            .shuntingParameters(ShuntingParameters.competitiveDefaults(dimension))
            .transmitterParameters(TransmitterParameters.paperDefaults())
            .timingParameters(AdaptiveTimingParameters.motorDefaults())
            .transmitterUpdateRatio(10)
            .timingUpdateRatio(100)
            .resetDecayFactor(0.1)
            .timingGatingEnabled(true)
            .build();
    }

    public static Builder builder(int dimension) {
        return new Builder(dimension);
    }

    public static class Builder {
        private final int dimension;
        private ShuntingParameters shuntingParameters;
        private TransmitterParameters transmitterParameters;
        private AdaptiveTimingParameters timingParameters;
        private int transmitterUpdateRatio = 10;
        private int timingUpdateRatio = 100;
        private double resetDecayFactor = 0.3;
        private boolean timingGatingEnabled = false;

        public Builder(int dimension) {
            this.dimension = dimension;
            // Set defaults
            this.shuntingParameters = ShuntingParameters.defaults(dimension);
            this.transmitterParameters = TransmitterParameters.paperDefaults();
            this.timingParameters = AdaptiveTimingParameters.speechDefaults();
        }

        public Builder shuntingParameters(ShuntingParameters params) {
            this.shuntingParameters = params;
            return this;
        }

        public Builder transmitterParameters(TransmitterParameters params) {
            this.transmitterParameters = params;
            return this;
        }

        public Builder timingParameters(AdaptiveTimingParameters params) {
            this.timingParameters = params;
            return this;
        }

        public Builder transmitterUpdateRatio(int ratio) {
            this.transmitterUpdateRatio = ratio;
            return this;
        }

        public Builder timingUpdateRatio(int ratio) {
            this.timingUpdateRatio = ratio;
            return this;
        }

        public Builder resetDecayFactor(double factor) {
            this.resetDecayFactor = factor;
            return this;
        }

        public Builder timingGatingEnabled(boolean enabled) {
            this.timingGatingEnabled = enabled;
            return this;
        }

        public MultiScaleParameters build() {
            return new MultiScaleParameters(this);
        }
    }

    private void validate() {
        if (dimension <= 0) {
            throw new IllegalArgumentException("Dimension must be positive");
        }
        if (transmitterUpdateRatio <= 0 || timingUpdateRatio <= 0) {
            throw new IllegalArgumentException("Update ratios must be positive");
        }
        if (timingUpdateRatio < transmitterUpdateRatio) {
            throw new IllegalArgumentException(
                "Timing update ratio should be >= transmitter update ratio (slower dynamics)"
            );
        }
        if (resetDecayFactor < 0 || resetDecayFactor > 1) {
            throw new IllegalArgumentException("Reset decay factor must be in [0, 1]");
        }
        if (shuntingParameters.getDimension() != dimension) {
            throw new IllegalArgumentException("Shunting dimension mismatch");
        }
    }

    // Getters
    public int getDimension() {
        return dimension;
    }

    public ShuntingParameters getShuntingParameters() {
        return shuntingParameters;
    }

    public TransmitterParameters getTransmitterParameters() {
        return transmitterParameters;
    }

    public AdaptiveTimingParameters getTimingParameters() {
        return timingParameters;
    }

    public int getTransmitterUpdateRatio() {
        return transmitterUpdateRatio;
    }

    public int getTimingUpdateRatio() {
        return timingUpdateRatio;
    }

    public double getResetDecayFactor() {
        return resetDecayFactor;
    }

    public boolean isTimingGatingEnabled() {
        return timingGatingEnabled;
    }

    /**
     * Compute effective time step for transmitter dynamics.
     */
    public double getTransmitterTimeStep() {
        return shuntingParameters.getTimeStep() * transmitterUpdateRatio;
    }

    /**
     * Compute effective time step for timing dynamics.
     */
    public double getTimingTimeStep() {
        return shuntingParameters.getTimeStep() * timingUpdateRatio;
    }

    /**
     * Get time scale separation factor.
     */
    public double getTimeScaleSeparation() {
        return (double) timingUpdateRatio / transmitterUpdateRatio;
    }

    /**
     * Get number of scales (3: fast/medium/slow).
     */
    public int getNumberOfScales() {
        return 3;
    }

    /**
     * Get time constants for each scale.
     */
    public double[] getTimeConstants() {
        return new double[]{
            shuntingParameters.getTimeStep(),
            getTransmitterTimeStep(),
            getTimingTimeStep()
        };
    }

    /**
     * Get weights for each scale.
     */
    public double[] getScaleWeights() {
        return new double[]{1.0, 0.8, 0.6}; // Fast, medium, slow weights
    }

    /**
     * Get parameters for specific scale.
     */
    public Object getScaleParameters(int scale) {
        return switch (scale) {
            case 0 -> shuntingParameters;
            case 1 -> transmitterParameters;
            case 2 -> timingParameters;
            default -> throw new IllegalArgumentException("Invalid scale: " + scale);
        };
    }

    /**
     * Get filter strength for scale.
     */
    public double getFilterStrength(int scale) {
        return switch (scale) {
            case 0 -> 1.0;  // No filtering for fast scale
            case 1 -> 0.7;  // Medium filtering for transmitter scale
            case 2 -> 0.5;  // Strong filtering for slow scale
            default -> throw new IllegalArgumentException("Invalid scale: " + scale);
        };
    }

    /**
     * Get filter cutoff for scale.
     */
    public double getFilterCutoff(int scale) {
        return switch (scale) {
            case 0 -> 1.0;   // High cutoff for fast scale
            case 1 -> 0.1;   // Medium cutoff for transmitter scale
            case 2 -> 0.01;  // Low cutoff for slow scale
            default -> throw new IllegalArgumentException("Invalid scale: " + scale);
        };
    }

    /**
     * Get input gain for scale.
     */
    public double getInputGain(int scale) {
        return switch (scale) {
            case 0 -> 1.0;   // Full gain for fast scale
            case 1 -> 0.5;   // Reduced gain for transmitter scale
            case 2 -> 0.2;   // Low gain for slow scale
            default -> throw new IllegalArgumentException("Invalid scale: " + scale);
        };
    }

    /**
     * Get forward coupling strength.
     */
    public double getForwardCoupling() {
        return 0.3;  // Default forward coupling between scales
    }

    /**
     * Get backward coupling strength.
     */
    public double getBackwardCoupling() {
        return 0.1;  // Default backward coupling between scales
    }

    /**
     * Get coupling strength for specific scale.
     */
    public double getCouplingStrength(int scale) {
        return switch (scale) {
            case 0 -> 0.5;   // Strong coupling for fast scale
            case 1 -> 0.3;   // Medium coupling for transmitter scale
            case 2 -> 0.1;   // Weak coupling for slow scale
            default -> throw new IllegalArgumentException("Invalid scale: " + scale);
        };
    }
}