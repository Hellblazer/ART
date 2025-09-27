# ART Hybrid Algorithms

This module contains hybrid ART algorithms that combine Adaptive Resonance Theory with deep learning techniques.

## PAN (Pretrained Adaptive Resonance Network)

PAN is an implementation of the paper "Pretrained back propagation based adaptive resonance theory network for adaptive learning" by Zhang et al. (2023). It combines CNN feature extraction with ART's stability-plasticity capabilities.

### Key Features

- **CNN Preprocessing**: Automatic feature extraction from raw input data
- **BPART Nodes**: Backpropagation-enabled ART nodes allowing negative weights
- **Dual Memory System**: STM (Short-Term Memory) and LTM (Long-Term Memory)
- **Experience Replay**: Prevents catastrophic forgetting in continual learning
- **Supervised Learning**: Support for both supervised and unsupervised modes

### Architecture

PAN implements the `ARTAlgorithm` interface directly rather than extending `BaseART`, providing full control over the learning process:

```java
PAN
├── CNNPreprocessor         # Feature extraction
├── BPARTWeight[]          # Neural network weight vectors
├── DualMemoryManager      # STM/LTM management
└── ExperienceReplayBuffer # Continual learning support
```

### Usage

#### Basic Unsupervised Learning

```java
// Create parameters
PANParameters params = PANParameters.defaultParameters();

// Initialize PAN
try (PAN pan = new PAN(params)) {
    // Create input pattern (e.g., 28x28 image as 784-dimensional vector)
    double[] imageData = new double[784];
    // ... populate imageData ...
    Pattern input = new DenseVector(imageData);

    // Learn the pattern
    ActivationResult result = pan.learn(input, params);

    // Check result
    if (result instanceof ActivationResult.Success success) {
        System.out.println("Learned as category: " + success.categoryIndex());
        System.out.println("Activation: " + success.activationValue());
    }
}
```

#### Supervised Learning

```java
// Create input and target patterns
Pattern input = new DenseVector(imageData);
Pattern target = new DenseVector(oneHotLabel);  // e.g., [0,0,1,0,0] for class 2

// Supervised learning
ActivationResult result = pan.learnSupervised(input, target, params);
```

#### Prediction

```java
// Predict without learning
ActivationResult prediction = pan.predict(input, params);

if (prediction instanceof ActivationResult.Success success) {
    int category = success.categoryIndex();
    double confidence = success.activationValue();
}
```

#### Batch Processing

```java
// Process multiple patterns in parallel
List<Pattern> batch = Arrays.asList(pattern1, pattern2, pattern3);

// Batch learning
List<ActivationResult> results = pan.learnBatch(batch, params);

// Batch prediction
List<ActivationResult> predictions = pan.predictBatch(batch, params);
```

### Parameters

PAN uses immutable parameter records with validation:

```java
// Default parameters based on paper
PANParameters params = PANParameters.defaultParameters();

// MNIST-specific parameters
PANParameters mnistParams = PANParameters.forMNIST();

// Omniglot-specific parameters
PANParameters omniglotParams = PANParameters.forOmniglot();

// Custom parameters
PANParameters custom = new PANParameters(
    0.7,    // vigilance
    10,     // maxCategories
    CNNConfig.simple(),  // CNN configuration
    false,  // enableCNNPretraining
    0.01,   // learningRate
    0.9,    // momentum
    0.0001, // weightDecay
    true,   // allowNegativeWeights (key innovation)
    64,     // hiddenUnits
    0.95,   // stmDecayRate
    0.8,    // ltmConsolidationThreshold
    1000,   // replayBufferSize
    32,     // replayBatchSize
    0.1,    // replayFrequency
    0.1     // biasFactor (light induction from paper)
);
```

### CNN Configuration

The CNN preprocessor can be configured for different architectures:

```java
// Simple 2-layer CNN
CNNConfig simple = CNNConfig.simple();

// MNIST-optimized
CNNConfig mnist = CNNConfig.mnist();

// Deeper architecture for complex datasets
CNNConfig omniglot = CNNConfig.omniglot();

// Custom configuration
CNNConfig custom = new CNNConfig(
    784,    // inputSize (28x28)
    256,    // outputFeatures
    "deep", // architecture name
    4,      // numLayers
    new int[]{32, 64, 128, 256}  // filterSizes per layer
);
```

### Performance Tracking

PAN tracks various performance metrics:

```java
// Get performance statistics
Map<String, Object> stats = pan.getPerformanceStats();

// Available metrics:
// - totalSamples: Number of patterns processed
// - correctPredictions: For supervised learning
// - accuracy: Classification accuracy
// - averageLoss: Training loss
// - trainingTimeMs: Total training time
// - categoryCount: Number of categories created
// - memoryUsageBytes: Estimated memory usage

// Reset tracking
pan.resetPerformanceTracking();
```

### Memory Management

PAN includes sophisticated memory management:

- **STM (Short-Term Memory)**: Recent patterns with decay
- **LTM (Long-Term Memory)**: Consolidated stable patterns
- **Experience Replay**: Prevents catastrophic forgetting

```java
// Clear all learned patterns
pan.clear();

// Resource management (implements AutoCloseable)
try (PAN pan = new PAN(params)) {
    // Use PAN
} // Automatically releases resources
```

### Performance Claims

Based on the paper, PAN should achieve:
- **91.3% accuracy** on MNIST+Omniglot dataset
- **2-6 categories** vs 11-18 for traditional ART
- Improved continual learning without catastrophic forgetting

### Implementation Details

#### BPARTWeight
Immutable weight vector implementing backpropagation:
- Forward weights: Input → Hidden connections
- Backward weights: Hidden → Output connections
- Supports negative weights (unlike traditional ART)
- Gradient-based updates with momentum

#### CNNPreprocessor
Handles feature extraction:
- Pure Java implementation with Vector API
- Float/double conversion for framework compatibility
- Configurable architecture (layers, filters)
- Optional pretrained weight loading

#### DualMemoryManager
Manages STM/LTM networks:
- STM buffer with configurable decay
- LTM consolidation based on frequency
- Enhanced vigilance checking
- Pattern similarity computation

#### ExperienceReplayBuffer
Enables continual learning:
- Reservoir sampling for uniform distribution
- Prioritized replay based on reward
- Configurable buffer size and batch size

### Testing

The module includes comprehensive tests:

```bash
# Run all tests
mvn test -pl art-hybrid

# Run specific test class
mvn test -pl art-hybrid -Dtest=PANTest

# Run integration tests
mvn test -pl art-hybrid -Dtest=PANIntegrationTest
```

### Requirements

- Java 24+
- Maven 3.9.1+
- Vector API (jdk.incubator.vector module)

### Future Enhancements

- GPU acceleration support
- ONNX model import/export
- Pretrained weight loading
- Additional CNN architectures
- Distributed training support

### References

Zhang, L., et al. (2023). "Pretrained back propagation based adaptive resonance theory network for adaptive learning"

### License

GNU Affero General Public License V3