package com.hellblazer.art.hybrid.parameters;

/**
 * Immutable parameters for PAN (Pretrained Adaptive Resonance Network).
 * Combines traditional ART parameters with PAN-specific neural network parameters.
 *
 * @param vigilance ART vigilance parameter (ρ) in range [0, 1]
 * @param learningRate ART learning rate (β) in range [0, 1]
 * @param choice ART choice parameter (α) for activation, α ≥ 0
 * @param cnnInputSize Input dimension (e.g., 784 for MNIST 28x28)
 * @param cnnOutputFeatures CNN feature vector dimension
 * @param cnnArchitecture CNN architecture type ("simple", "deeper")
 * @param usePretrained Whether to use pretrained CNN weights
 * @param pretrainedPath Path to pretrained weights (optional)
 * @param hiddenUnits Hidden layer size in BPART nodes
 * @param backpropLearningRate Learning rate for backpropagation
 * @param momentum Momentum for SGD optimization
 * @param weightDecay Weight decay for regularization
 * @param allowNegativeWeights Allow negative weights (PAN innovation)
 * @param stmSize Short-term memory buffer size
 * @param stmDecayRate STM decay rate for consistency checks
 * @param ltmTransferThreshold Threshold for STM to LTM transfer
 * @param experiencePoolSize Experience replay pool size
 * @param replayProbability Probability of experience replay
 * @param replayBatchSize Batch size for experience replay
 * @param bpartEpochs Local epochs per BPART update
 * @param useComplementCoding Use complement coding [x, 1-x]
 * @param useL2Normalization Use L2 normalization
 * @param dropoutRate Dropout rate for regularization
 * @param enableGPU Enable GPU acceleration
 * @param vectorBatchSize Batch size for vectorized operations
 * @param targetAccuracy Target accuracy (91.3% from paper)
 * @param maxCategories Maximum categories allowed (2-6 from paper)
 * @param biasFactor Light induction bias factor (ε from paper)
 * @param locationConfidenceThreshold Location confidence threshold
 */
public record PANParameters(
    double vigilance,
    double learningRate,
    double choice,
    int cnnInputSize,
    int cnnOutputFeatures,
    String cnnArchitecture,
    boolean usePretrained,
    String pretrainedPath,
    int hiddenUnits,
    float backpropLearningRate,
    float momentum,
    float weightDecay,
    boolean allowNegativeWeights,
    int stmSize,
    float stmDecayRate,
    float ltmTransferThreshold,
    int experiencePoolSize,
    float replayProbability,
    int replayBatchSize,
    int bpartEpochs,
    boolean useComplementCoding,
    boolean useL2Normalization,
    float dropoutRate,
    boolean enableGPU,
    int vectorBatchSize,
    float targetAccuracy,
    int maxCategories,
    float biasFactor,
    float locationConfidenceThreshold
) {

    /**
     * Constructor with validation.
     */
    public PANParameters {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in range [0, 1], got: " + vigilance);
        }
        if (learningRate < 0.0 || learningRate > 1.0) {
            throw new IllegalArgumentException("Learning rate must be in range [0, 1], got: " + learningRate);
        }
        if (choice < 0.0) {
            throw new IllegalArgumentException("Choice must be non-negative, got: " + choice);
        }
        if (cnnInputSize <= 0 || cnnOutputFeatures <= 0) {
            throw new IllegalArgumentException("CNN dimensions must be positive");
        }
        if (hiddenUnits <= 0 || stmSize <= 0) {
            throw new IllegalArgumentException("Hidden units and STM size must be positive");
        }
        if (backpropLearningRate <= 0 || backpropLearningRate > 1) {
            throw new IllegalArgumentException("Backprop learning rate must be in (0, 1]");
        }
        if (momentum < 0 || momentum >= 1) {
            throw new IllegalArgumentException("Momentum must be in [0, 1)");
        }
        if (experiencePoolSize < 0 || replayBatchSize < 0) {
            throw new IllegalArgumentException("Pool and batch sizes must be non-negative");
        }
        if (replayBatchSize > experiencePoolSize && experiencePoolSize > 0) {
            throw new IllegalArgumentException("Replay batch size cannot exceed pool size");
        }
        if (targetAccuracy < 0 || targetAccuracy > 1) {
            throw new IllegalArgumentException("Target accuracy must be in [0, 1]");
        }
        if (maxCategories <= 0) {
            throw new IllegalArgumentException("Max categories must be positive");
        }
        if (biasFactor < 0) {
            throw new IllegalArgumentException("Bias factor must be non-negative");
        }
        if (locationConfidenceThreshold < 0 || locationConfidenceThreshold > 1) {
            throw new IllegalArgumentException("Location confidence must be in [0, 1]");
        }
    }

    /**
     * Create default parameters for PAN.
     */
    public static PANParameters defaultParameters() {
        return new PANParameters(
            0.7,      // vigilance
            0.8,      // learningRate
            0.01,     // choice
            784,      // cnnInputSize (28x28)
            128,      // cnnOutputFeatures
            "simple", // cnnArchitecture
            true,     // usePretrained
            null,     // pretrainedPath
            64,       // hiddenUnits
            0.01f,    // backpropLearningRate
            0.9f,     // momentum
            0.0001f,  // weightDecay
            true,     // allowNegativeWeights (key PAN innovation)
            10,       // stmSize
            0.95f,    // stmDecayRate
            0.8f,     // ltmTransferThreshold
            1000,     // experiencePoolSize
            0.1f,     // replayProbability
            32,       // replayBatchSize
            5,        // bpartEpochs
            false,    // useComplementCoding (disabled for PAN)
            true,     // useL2Normalization
            0.1f,     // dropoutRate
            false,    // enableGPU (default to CPU)
            64,       // vectorBatchSize
            0.913f,   // targetAccuracy (91.3% from paper)
            6,        // maxCategories (2-6 from paper)
            0.1f,     // biasFactor (ε from paper)
            0.8f      // locationConfidenceThreshold
        );
    }

    /**
     * Create parameters optimized for MNIST dataset.
     */
    public static PANParameters forMNIST() {
        return defaultParameters();
    }

    /**
     * Create parameters optimized for Omniglot dataset.
     */
    public static PANParameters forOmniglot() {
        var defaults = defaultParameters();
        return new PANParameters(
            0.75,     // Higher vigilance for more distinct categories
            defaults.learningRate(),
            defaults.choice(),
            784,      // Same input size as MNIST
            256,      // More features for complex alphabet
            "deeper", // Deeper architecture for complexity
            defaults.usePretrained(),
            defaults.pretrainedPath(),
            defaults.hiddenUnits(),
            defaults.backpropLearningRate(),
            defaults.momentum(),
            defaults.weightDecay(),
            defaults.allowNegativeWeights(),
            defaults.stmSize(),
            defaults.stmDecayRate(),
            defaults.ltmTransferThreshold(),
            defaults.experiencePoolSize(),
            defaults.replayProbability(),
            defaults.replayBatchSize(),
            defaults.bpartEpochs(),
            defaults.useComplementCoding(),
            defaults.useL2Normalization(),
            defaults.dropoutRate(),
            defaults.enableGPU(),
            defaults.vectorBatchSize(),
            defaults.targetAccuracy(),
            50,       // More categories for many character classes
            defaults.biasFactor(),
            defaults.locationConfidenceThreshold()
        );
    }
}