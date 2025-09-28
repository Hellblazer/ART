package com.hellblazer.art.hybrid.pan.parameters;

import java.util.Arrays;

/**
 * CNN architecture configuration.
 */
public record CNNConfig(
    int inputSize,
    int outputFeatures,
    String architecture,
    int numLayers,
    int[] filterSizes
) {
    public CNNConfig {
        if (inputSize <= 0) {
            throw new IllegalArgumentException("Input size must be positive");
        }
        if (outputFeatures <= 0) {
            throw new IllegalArgumentException("Output features must be positive");
        }
        if (numLayers <= 0) {
            throw new IllegalArgumentException("Number of layers must be positive");
        }
        if (filterSizes == null || filterSizes.length != numLayers) {
            throw new IllegalArgumentException("Filter sizes must match number of layers");
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof CNNConfig other)) return false;
        return inputSize == other.inputSize &&
               outputFeatures == other.outputFeatures &&
               numLayers == other.numLayers &&
               architecture.equals(other.architecture) &&
               Arrays.equals(filterSizes, other.filterSizes);
    }

    @Override
    public int hashCode() {
        int result = inputSize;
        result = 31 * result + outputFeatures;
        result = 31 * result + architecture.hashCode();
        result = 31 * result + numLayers;
        result = 31 * result + Arrays.hashCode(filterSizes);
        return result;
    }

    public static CNNConfig simple() {
        return new CNNConfig(784, 128, "simple", 2, new int[]{32, 64});
    }

    public static CNNConfig mnist() {
        return new CNNConfig(784, 128, "mnist", 2, new int[]{32, 64});
    }

    public static CNNConfig omniglot() {
        return new CNNConfig(784, 256, "deeper", 3, new int[]{32, 64, 128});
    }
}