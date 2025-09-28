package com.hellblazer.art.laminar.impl;

import com.hellblazer.art.laminar.parameters.ILayerParameters;

/**
 * Default implementation of layer parameters.
 *
 * @author Hal Hildebrand
 */
public record DefaultLayerParameters(
        double decayRate,
        double upperBound,
        double lowerBound,
        double activationThreshold,
        double saturationLevel,
        boolean useNormalization,
        NormalizationType normalizationType,
        double timeConstant,
        int integrationSteps,
        double noiseLevel,
        NoiseType noiseType
) implements ILayerParameters {

    public static final DefaultLayerParameters DEFAULT = new DefaultLayerParameters(
            0.1,    // decayRate
            1.0,    // upperBound
            0.0,    // lowerBound
            0.1,    // activationThreshold
            1.0,    // saturationLevel
            true,   // useNormalization
            NormalizationType.L2,
            1.0,    // timeConstant
            10,     // integrationSteps
            0.0,    // noiseLevel
            NoiseType.NONE
    );

    @Override
    public double getDecayRate() {
        return decayRate;
    }

    @Override
    public double getUpperBound() {
        return upperBound;
    }

    @Override
    public double getLowerBound() {
        return lowerBound;
    }

    @Override
    public double getActivationThreshold() {
        return activationThreshold;
    }

    @Override
    public double getSaturationLevel() {
        return saturationLevel;
    }

    @Override
    public boolean useNormalization() {
        return useNormalization;
    }

    @Override
    public NormalizationType getNormalizationType() {
        return normalizationType;
    }

    @Override
    public double getTimeConstant() {
        return timeConstant;
    }

    @Override
    public int getIntegrationSteps() {
        return integrationSteps;
    }

    @Override
    public double getNoiseLevel() {
        return noiseLevel;
    }

    @Override
    public NoiseType getNoiseType() {
        return noiseType;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private double decayRate = DEFAULT.decayRate;
        private double upperBound = DEFAULT.upperBound;
        private double lowerBound = DEFAULT.lowerBound;
        private double activationThreshold = DEFAULT.activationThreshold;
        private double saturationLevel = DEFAULT.saturationLevel;
        private boolean useNormalization = DEFAULT.useNormalization;
        private NormalizationType normalizationType = DEFAULT.normalizationType;
        private double timeConstant = DEFAULT.timeConstant;
        private int integrationSteps = DEFAULT.integrationSteps;
        private double noiseLevel = DEFAULT.noiseLevel;
        private NoiseType noiseType = DEFAULT.noiseType;

        public Builder withDecayRate(double decayRate) {
            this.decayRate = decayRate;
            return this;
        }

        public Builder withUpperBound(double upperBound) {
            this.upperBound = upperBound;
            return this;
        }

        public Builder withLowerBound(double lowerBound) {
            this.lowerBound = lowerBound;
            return this;
        }

        public Builder withActivationThreshold(double threshold) {
            this.activationThreshold = threshold;
            return this;
        }

        public Builder withSaturationLevel(double saturation) {
            this.saturationLevel = saturation;
            return this;
        }

        public Builder withNormalization(boolean useNormalization, NormalizationType type) {
            this.useNormalization = useNormalization;
            this.normalizationType = type;
            return this;
        }

        public Builder withTimeConstant(double timeConstant) {
            this.timeConstant = timeConstant;
            return this;
        }

        public Builder withIntegrationSteps(int steps) {
            this.integrationSteps = steps;
            return this;
        }

        public Builder withNoise(double level, NoiseType type) {
            this.noiseLevel = level;
            this.noiseType = type;
            return this;
        }

        public DefaultLayerParameters build() {
            return new DefaultLayerParameters(decayRate, upperBound, lowerBound,
                    activationThreshold, saturationLevel, useNormalization, normalizationType,
                    timeConstant, integrationSteps, noiseLevel, noiseType);
        }
    }
}