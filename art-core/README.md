# ART Core Module

The `art-core` module provides the foundational implementations of Adaptive Resonance Theory neural networks. This module contains reference implementations that prioritize correctness, clarity, and scientific accuracy.

## Overview

The core module serves as the theoretical foundation and reference implementation for all ART algorithms. It provides:

- **Base ART Algorithms**: Complete implementations of all major ART variants
- **Mathematical Foundations**: Core data structures and mathematical operations
- **Parameter Management**: Comprehensive parameter classes with validation
- **Result Types**: Detailed result structures for analysis and debugging
- **Utility Classes**: Helper functions for data processing and analysis

## Module Structure

```
art-core/
├── algorithms/           # Core ART algorithm implementations
│   ├── ART2.java         # ART-2 preprocessing algorithm
│   ├── ARTA.java         # Attention-based ART
│   ├── ARTE.java         # Enhanced ART with extensions
│   ├── ARTSTAR.java      # Star-shaped category regions
│   ├── BayesianART.java  # Probabilistic ART with uncertainty
│   ├── EllipsoidART.java # Ellipsoidal category boundaries
│   ├── FuzzyART.java     # Fuzzy logic pattern recognition
│   ├── GaussianART.java  # Gaussian distribution modeling
│   └── HypersphereART.java # Hyperspherical coordinate system
├── artmap/               # Supervised learning implementations
│   ├── ARTMAP.java       # Basic supervised ARTMAP
│   ├── ARTMAPParameters.java
│   ├── ARTMAPResult.java
│   ├── DeepARTMAP.java   # Hierarchical multi-layer ARTMAP
│   ├── DeepARTMAPParameters.java
│   ├── DeepARTMAPResult.java
│   └── SimpleARTMAP.java # Simplified ARTMAP variant
├── parameters/           # Algorithm parameter classes
│   ├── ART2Parameters.java
│   ├── ARTAParameters.java
│   ├── ARTEParameters.java
│   ├── ARTSTARParameters.java
│   ├── BayesianParameters.java
│   ├── EllipsoidParameters.java
│   ├── FuzzyParameters.java
│   ├── GaussianParameters.java
│   └── HypersphereParameters.java
├── results/              # Algorithm result structures
│   ├── ActivationResult.java
│   ├── BayesianActivationResult.java
│   ├── CategoryResult.java
│   ├── EllipsoidActivationResult.java
│   └── MatchResult.java
├── utils/                # Utility classes
│   ├── DataBounds.java   # Data normalization and bounds
│   ├── Matrix.java       # Matrix operations
│   └── SimpleVector.java # Basic vector operations
└── weights/              # Weight vector implementations
    ├── ART2Weight.java
    ├── ARTAWeight.java
    ├── ARTEWeight.java
    ├── ARTSTARWeight.java
    ├── BayesianWeight.java
    ├── EllipsoidWeight.java
    ├── FuzzyWeight.java
    ├── GaussianWeight.java
    └── HypersphereWeight.java
```

## Key Classes

### BaseART - Abstract Foundation

All ART algorithms extend `BaseART`:

```java
public abstract class BaseART<P extends Parameters> implements AutoCloseable {
    // Core learning method - must be implemented by subclasses
    public abstract int stepFit(Pattern input);
    
    // Common prediction and analysis methods
    public CategoryResult predict(Pattern input);
    public ActivationResult activate(Pattern input);
    public int getCategoryCount();
    public void reset();
}
```

### Pattern Interface

The fundamental data representation:

```java
public interface Pattern {
    double getValue(int index);
    int getDimension();
    double[] toArray();
    Pattern normalize();
    Pattern complement();
}
```

### DenseVector - Primary Pattern Implementation

```java
public class DenseVector implements Pattern {
    // Constructors
    public DenseVector(double[] values);
    public DenseVector(float[] values);
    public DenseVector(int dimension);
    
    // Mathematical operations
    public DenseVector add(DenseVector other);
    public DenseVector subtract(DenseVector other);
    public double dotProduct(DenseVector other);
    public double norm();
    public DenseVector normalize();
    
    // ART-specific operations
    public DenseVector fuzzyIntersection(DenseVector other);
    public DenseVector complement();
}
```

## Algorithm Implementations

### FuzzyART - General Pattern Recognition

The most widely used ART variant:

```java
public class FuzzyART extends BaseART<FuzzyParameters> {
    public FuzzyART(FuzzyParameters parameters);
    
    // Core learning
    public int stepFit(Pattern input);
    
    // Advanced analysis
    public ActivationResult[] activateAll(Pattern input);
    public FuzzyWeight getWeight(int categoryIndex);
    public void pruneUnusedCategories();
}
```

**Key Features:**
- Fuzzy set theory for analog input processing
- Complement coding for stable learning
- Choice function for category competition
- Vigilance test for category acceptance

**Usage Example:**
```java
var parameters = FuzzyParameters.of(0.8, 0.1, 0.001);
try (var fuzzyART = new FuzzyART(parameters)) {
    var pattern = new DenseVector(new double[]{0.8, 0.2, 0.9});
    int category = fuzzyART.stepFit(pattern);
    System.out.printf("Pattern assigned to category %d%n", category);
}
```

### BayesianART - Uncertainty Quantification

Probabilistic ART with confidence estimates:

```java
public class BayesianART extends BaseART<BayesianParameters> {
    public BayesianART(BayesianParameters parameters);
    
    // Enhanced prediction with uncertainty
    public BayesianActivationResult predictWithUncertainty(Pattern input);
    public double calculateEntropy(Pattern input);
    public double[] getPosteriorProbabilities(Pattern input);
}
```

**Key Features:**
- Bayesian inference framework
- Uncertainty quantification
- Confidence intervals
- Prior probability learning

### HypersphereART - Geometric Clustering

Uses hyperspherical coordinate system:

```java
public class HypersphereART extends BaseART<HypersphereParameters> {
    public HypersphereART(HypersphereParameters parameters);
    
    // Geometric analysis
    public double calculateDistance(Pattern input, int categoryIndex);
    public double[] getCenterVector(int categoryIndex);
    public double getRadius(int categoryIndex);
}
```

**Key Features:**
- Euclidean distance calculations
- Adaptive radius learning
- Geometric category representation
- Rotation invariant processing

### GaussianART - Statistical Modeling

Statistical pattern recognition:

```java
public class GaussianART extends BaseART<GaussianParameters> {
    public GaussianART(GaussianParameters parameters);
    
    // Statistical methods
    public double[] getMean(int categoryIndex);
    public double[] getVariance(int categoryIndex);
    public double calculateLikelihood(Pattern input, int categoryIndex);
}
```

**Key Features:**
- Gaussian probability distributions
- Maximum likelihood estimation
- Statistical significance testing
- Covariance matrix learning

## ARTMAP - Supervised Learning

### Basic ARTMAP

Input-output pattern association:

```java
public class ARTMAP implements AutoCloseable {
    public ARTMAP(ARTMAPParameters parameters);
    
    // Supervised learning
    public ARTMAPResult learn(Pattern input, Pattern target);
    public ARTMAPResult predict(Pattern input);
    
    // Network components
    public BaseART getArtA();  // Input processing
    public BaseART getArtB();  // Output processing
}
```

**Key Features:**
- Dual ART architecture (ART-A and ART-B)
- Match tracking for stable learning
- Associative map field
- Predictive capability

### DeepARTMAP - Hierarchical Learning

Multi-layer hierarchical processing:

```java
public class DeepARTMAP extends ARTMAP {
    public DeepARTMAP(DeepARTMAPParameters parameters);
    
    // Hierarchical processing
    public DeepARTMAPResult learnHierarchical(Pattern input, Pattern target);
    public ActivationResult[] getLayerActivations(Pattern input);
    public int getLayerCount();
}
```

**Key Features:**
- Multi-layer processing hierarchy
- Feature abstraction across layers
- Hierarchical category formation
- Layer-specific vigilance parameters

## Parameter Management

### Parameter Classes

Each algorithm has a dedicated parameter class:

```java
// Example: FuzzyParameters
public class FuzzyParameters implements Parameters {
    // Factory methods
    public static FuzzyParameters of(double vigilance, double learningRate, double bias);
    public static FuzzyParameters conservative(int dimensions);
    public static FuzzyParameters aggressive(int dimensions);
    
    // Builder pattern
    public static Builder builder();
    
    // Parameter validation
    private void validateParameters();
}
```

### Parameter Presets

Pre-configured parameter sets for common use cases:

```java
// Conservative learning (stable, slower)
var conservative = FuzzyParameters.conservative(inputDimensions);

// Aggressive learning (faster, less stable)
var aggressive = FuzzyParameters.aggressive(inputDimensions);

// Custom parameters
var custom = FuzzyParameters.builder()
    .vigilance(0.85)        // Pattern selectivity
    .learningRate(0.1)      // Adaptation speed
    .bias(0.001)           // Stability factor
    .maxCategories(1000)    // Memory limit
    .build();
```

## Result Types

### CategoryResult

Basic classification result:

```java
public class CategoryResult {
    public int getCategory();           // Winning category
    public double getActivation();      // Activation strength
    public double getVigilanceMatch();  // Vigilance test result
    public boolean isNewCategory();     // Category creation flag
}
```

### ActivationResult

Detailed activation analysis:

```java
public class ActivationResult extends CategoryResult {
    public double[] getAllActivations();    // All category activations
    public int[] getRankedCategories();     // Sorted by activation
    public double getCompetitionRatio();    // Winner margin
    public Map<String, Double> getMetrics(); // Algorithm-specific data
}
```

### BayesianActivationResult

Enhanced result with uncertainty:

```java
public class BayesianActivationResult extends ActivationResult {
    public double getConfidence();       // Classification confidence
    public double getEntropy();          // Information entropy
    public double[] getProbabilities();  // Full probability distribution
    public double getVariance();         // Prediction variance
}
```

## Utility Classes

### DataBounds - Data Preprocessing

```java
public class DataBounds {
    public static DataBounds from(Collection<Pattern> patterns);
    
    // Normalization
    public Pattern normalize(Pattern input);
    public List<Pattern> normalizeAll(List<Pattern> patterns);
    
    // Statistics
    public Pattern getMeanValues();
    public Pattern getStandardDeviation();
}
```

### ScikitClusterer - Compatibility Layer

```java
public class ScikitClusterer {
    public ScikitClusterer(BaseART art);
    
    // Scikit-learn compatible interface
    public ScikitClusterer fit(double[][] X);
    public int[] predict(double[][] X);
    public int[] fit_predict(double[][] X);
}
```

## Usage Patterns

### Basic Learning Workflow

```java
// 1. Create parameters
var parameters = FuzzyParameters.builder()
    .vigilance(0.8)
    .learningRate(0.1)
    .build();

// 2. Initialize algorithm
try (var art = new FuzzyART(parameters)) {
    // 3. Train with data
    for (var pattern : trainingData) {
        int category = art.stepFit(pattern);
        System.out.printf("Pattern -> Category %d%n", category);
    }
    
    // 4. Analyze results
    System.out.printf("Created %d categories%n", art.getCategoryCount());
    
    // 5. Make predictions
    for (var testPattern : testData) {
        var result = art.predict(testPattern);
        System.out.printf("Prediction: Category %d (confidence: %.3f)%n",
            result.getCategory(), result.getActivation());
    }
}
```

### Supervised Learning Workflow

```java
// 1. Configure ARTMAP
var parameters = ARTMAPParameters.builder()
    .vigilanceAB(0.9)
    .matchTracking(true)
    .build();

// 2. Train with input-output pairs
try (var artmap = new ARTMAP(parameters)) {
    for (int i = 0; i < inputs.size(); i++) {
        var result = artmap.learn(inputs.get(i), outputs.get(i));
        System.out.printf("Learned association %d%n", i);
    }
    
    // 3. Make predictions
    for (var input : testInputs) {
        var prediction = artmap.predict(input);
        System.out.printf("Predicted output: %s%n", 
            Arrays.toString(prediction.getPredictedOutput().toArray()));
    }
}
```

### Bayesian Analysis Workflow

```java
// 1. Create Bayesian ART
var parameters = BayesianParameters.conservative(inputDimensions);
try (var bayesianART = new BayesianART(parameters)) {
    
    // 2. Train with data
    for (var pattern : trainingData) {
        bayesianART.stepFit(pattern);
    }
    
    // 3. Analyze with uncertainty
    for (var testPattern : testData) {
        var result = bayesianART.predictWithUncertainty(testPattern);
        
        System.out.printf("Category: %d%n", result.getCategory());
        System.out.printf("Confidence: %.3f%n", result.getConfidence());
        System.out.printf("Entropy: %.3f%n", result.getEntropy());
        System.out.printf("Probabilities: %s%n", 
            Arrays.toString(result.getProbabilities()));
    }
}
```

## Testing and Validation

The art-core module includes comprehensive testing:

- **Comprehensive test coverage** covering all algorithms and edge cases
- **Unit tests** for individual components
- **Integration tests** for complete workflows  
- **Property-based tests** for mathematical invariants
- **Performance tests** for algorithmic complexity

### Running Tests

```bash
# All core tests
mvn test -pl art-core

# Specific algorithm tests
mvn test -pl art-core -Dtest=FuzzyARTTest
mvn test -pl art-core -Dtest=BayesianARTTest

# Integration tests
mvn test -pl art-core -Dtest=*IntegrationTest
```

## Performance Characteristics

The art-core module prioritizes correctness over performance:

- **Single-threaded execution** for predictable behavior
- **Precise floating-point arithmetic** for scientific accuracy
- **Comprehensive validation** for parameter checking
- **Memory-efficient data structures** for moderate datasets

For high-performance applications, consider using `art-performance` module implementations that extend these core classes.

## Best Practices

### Parameter Selection

```java
// Start conservative for stable learning
var parameters = FuzzyParameters.conservative(dimensions);

// Adjust based on data characteristics
if (needsHighSelectivity) {
    parameters = parameters.toBuilder()
        .vigilance(0.9)  // Higher selectivity
        .build();
}

if (needsFastAdaptation) {
    parameters = parameters.toBuilder()
        .learningRate(0.3)  // Faster learning
        .build();
}
```

### Memory Management

```java
// Always use try-with-resources
try (var art = new FuzzyART(parameters)) {
    // Training and prediction
    processData(art);
} // Automatic resource cleanup

// For long-running processes
art.pruneUnusedCategories();  // Remove unused categories
art.compactMemory();         // Defragment memory (if available)
```

### Error Handling

```java
try {
    var art = new FuzzyART(parameters);
    var result = art.stepFit(pattern);
} catch (DimensionMismatchException e) {
    System.err.printf("Dimension mismatch: expected %d, got %d%n",
        e.getExpectedDimension(), e.getActualDimension());
} catch (CategoryLimitException e) {
    System.err.printf("Category limit exceeded: %d%n", e.getMaxCategories());
} catch (ARTException e) {
    System.err.printf("ART error: %s%n", e.getMessage());
}
```

The art-core module provides the scientific foundation for all ART implementations, ensuring mathematical correctness and theoretical compliance with published ART algorithms.