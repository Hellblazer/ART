package com.hellblazer.art.hybrid.pan.parameters;

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