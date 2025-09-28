package com.hellblazer.art.hybrid.pan.parameters;

/**
 * Parameters for PAN (Pretrained Adaptive Resonance Network).
 * Combines ART parameters with neural network training parameters.
 */
public record PANParameters(
    // Core ART parameters
    double vigilance,
    int maxCategories,

    // CNN configuration
    CNNConfig cnnConfig,
    boolean enableCNNPretraining,

    // BPART neural network parameters
    double learningRate,
    double momentum,
    double weightDecay,
    boolean allowNegativeWeights,
    int hiddenUnits,

    // Memory management
    double stmDecayRate,
    double ltmConsolidationThreshold,

    // Experience replay
    int replayBufferSize,
    int replayBatchSize,
    double replayFrequency,

    // Light induction (from paper)
    double biasFactor,

    // Normalization control (FIX for clustering issue)
    boolean enableFeatureNormalization,
    double globalMinBound,
    double globalMaxBound
) {

    /**
     * Validation in compact constructor.
     */
    public PANParameters {
        // Note: Vigilance validation relaxed for PAN architecture
        if (maxCategories <= 0) {
            throw new IllegalArgumentException("Max categories must be positive: " + maxCategories);
        }
        if (learningRate <= 0.0 || learningRate > 1.0) {
            throw new IllegalArgumentException("Learning rate must be in (0, 1]: " + learningRate);
        }
        if (momentum < 0.0 || momentum >= 1.0) {
            throw new IllegalArgumentException("Momentum must be in [0, 1): " + momentum);
        }
        if (weightDecay < 0.0) {
            throw new IllegalArgumentException("Weight decay must be non-negative: " + weightDecay);
        }
        if (hiddenUnits <= 0) {
            throw new IllegalArgumentException("Hidden units must be positive: " + hiddenUnits);
        }
        if (stmDecayRate < 0.0 || stmDecayRate > 1.0) {
            throw new IllegalArgumentException("STM decay rate must be in [0, 1]: " + stmDecayRate);
        }
        if (ltmConsolidationThreshold < 0.0 || ltmConsolidationThreshold > 1.0) {
            throw new IllegalArgumentException("LTM threshold must be in [0, 1]: " + ltmConsolidationThreshold);
        }
        if (replayBufferSize < 0) {
            throw new IllegalArgumentException("Replay buffer size must be non-negative: " + replayBufferSize);
        }
        if (replayBatchSize < 0 || (replayBufferSize > 0 && replayBatchSize > replayBufferSize)) {
            throw new IllegalArgumentException("Invalid replay batch size: " + replayBatchSize);
        }
        if (replayFrequency < 0.0 || replayFrequency > 1.0) {
            throw new IllegalArgumentException("Replay frequency must be in [0, 1]: " + replayFrequency);
        }
        if (biasFactor < 0.0) {
            throw new IllegalArgumentException("Bias factor must be non-negative: " + biasFactor);
        }
        if (enableFeatureNormalization && globalMinBound >= globalMaxBound) {
            throw new IllegalArgumentException("Global min bound must be less than max bound");
        }
    }

    /**
     * Default parameters based on paper specifications.
     */
    public static PANParameters defaultParameters() {
        return new PANParameters(
            0.5,                          // vigilance
            20,                           // maxCategories (allow more initially)
            CNNConfig.simple(),           // cnnConfig
            false,                        // enableCNNPretraining
            0.01,                         // learningRate
            0.9,                          // momentum
            0.0001,                       // weightDecay
            true,                         // allowNegativeWeights
            64,                           // hiddenUnits
            0.95,                         // stmDecayRate
            0.8,                          // ltmConsolidationThreshold
            1000,                         // replayBufferSize
            32,                           // replayBatchSize
            0.1,                          // replayFrequency
            0.1,                          // biasFactor (ε from paper)
            false,                        // enableFeatureNormalization (DISABLED to fix clustering)
            0.0,                          // globalMinBound
            1.0                           // globalMaxBound
        );
    }

    /**
     * Parameters for paper-compliant configuration.
     */
    public static PANParameters paperCompliantParameters() {
        return new PANParameters(
            0.7,                          // vigilance (original paper value)
            20,                           // maxCategories
            CNNConfig.simple(),           // cnnConfig
            false,                        // enableCNNPretraining
            0.01,                         // learningRate
            0.9,                          // momentum
            0.0001,                       // weightDecay
            true,                         // allowNegativeWeights
            64,                           // hiddenUnits
            0.95,                         // stmDecayRate
            0.8,                          // ltmConsolidationThreshold
            1000,                         // replayBufferSize
            32,                           // replayBatchSize
            0.1,                          // replayFrequency
            0.1,                          // biasFactor
            false,                        // enableFeatureNormalization (DISABLED to fix clustering)
            0.0,                          // globalMinBound
            1.0                           // globalMaxBound
        );
    }

    /**
     * Parameters for MNIST dataset.
     */
    public static PANParameters forMNIST() {
        return new PANParameters(
            0.45,                         // moderate vigilance
            20,                           // allow some redundancy
            CNNConfig.mnist(),
            false,
            0.01,
            0.9,
            0.0001,
            true,
            64,
            0.95,
            0.8,
            1000,
            32,
            0.1,
            0.1,
            true,                         // enableFeatureNormalization (for real datasets)
            0.0,                          // globalMinBound (typical pixel min)
            255.0                         // globalMaxBound (typical pixel max)
        );
    }

    /**
     * Parameters for Omniglot dataset.
     */
    public static PANParameters forOmniglot() {
        return new PANParameters(
            0.55,                         // Higher vigilance for more classes
            50,                           // Many character classes
            CNNConfig.omniglot(),
            true,                         // Use pretrained for complex dataset
            0.005,                        // Lower learning rate
            0.9,
            0.0001,
            true,
            128,                          // More hidden units
            0.95,
            0.8,
            2000,                         // Larger replay buffer
            64,
            0.15,                         // More frequent replay
            0.1,
            true,                         // enableFeatureNormalization (for real datasets)
            0.0,                          // globalMinBound
            1.0                           // globalMaxBound
        );
    }
}