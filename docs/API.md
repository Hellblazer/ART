# ART API Documentation

## Overview

The ART API is designed with performance, type safety, and ease of use in mind. The API is organized into two main modules: `art-core` for base implementations and `art-performance` for vectorized, high-performance variants.

## Core Concepts

### Pattern Representation

All input data in ART is represented using the `Pattern` interface:

```java
public interface Pattern {
    double getValue(int index);
    int getDimension();
    double[] toArray();
    Pattern normalize();
    Pattern complement();
}
```

#### DenseVector Implementation

The primary implementation of `Pattern`:

```java
public class DenseVector implements Pattern {
    public DenseVector(double[] values)
    public DenseVector(float[] values)
    public DenseVector(int dimension)  // Zero-initialized
    
    // Access methods
    public double getValue(int index)
    public int getDimension()
    public double[] toArray()
    
    // Mathematical operations
    public DenseVector add(DenseVector other)
    public DenseVector subtract(DenseVector other)
    public DenseVector multiply(double scalar)
    public double dotProduct(DenseVector other)
    public double norm()
    public DenseVector normalize()
    
    // ART-specific operations
    public DenseVector fuzzyIntersection(DenseVector other)
    public DenseVector complement()
}
```

**Usage Example:**

```java
// Create patterns
var pattern1 = new DenseVector(new double[]{0.8, 0.2, 0.9, 0.1});
var pattern2 = new DenseVector(new float[]{0.7f, 0.3f, 0.8f, 0.2f});

// Mathematical operations
var sum = pattern1.add(pattern2);
var normalized = pattern1.normalize();
var intersection = pattern1.fuzzyIntersection(pattern2);

System.out.printf("Norm: %.3f%n", pattern1.norm());
System.out.printf("Dot product: %.3f%n", pattern1.dotProduct(pattern2));
```

## art-core Module API

### Base ART Algorithms

All ART algorithms extend the abstract `BaseART` class:

```java
public abstract class BaseART<P extends Parameters> implements AutoCloseable {
    protected BaseART(P parameters)
    
    // Core learning method
    public abstract int stepFit(Pattern input)
    
    // Prediction methods
    public CategoryResult predict(Pattern input)
    public ActivationResult activate(Pattern input)
    
    // Network state
    public int getCategoryCount()
    public boolean isEmpty()
    public void reset()
    
    // Resource management
    public void close()
}
```

### FuzzyART

The most commonly used ART variant for general pattern recognition:

```java
public class FuzzyART extends BaseART<FuzzyParameters> {
    public FuzzyART(FuzzyParameters parameters)
    
    // Learning
    public int stepFit(Pattern input)
    public CategoryResult learnPattern(Pattern input)
    
    // Prediction
    public CategoryResult predict(Pattern input)
    public ActivationResult[] activateAll(Pattern input)
    
    // Network inspection
    public FuzzyWeight getWeight(int categoryIndex)
    public double getVigilance()
    public double getLearningRate()
    
    // Category management
    public boolean hasCategory(int index)
    public void pruneUnusedCategories()
}
```

**Parameters:**

```java
public class FuzzyParameters implements Parameters {
    // Factory methods
    public static FuzzyParameters of(double vigilance, double learningRate, double bias)
    public static FuzzyParameters conservative(int dimensions)
    public static FuzzyParameters aggressive(int dimensions)
    
    // Builder pattern
    public static Builder builder()
    
    public static class Builder {
        public Builder vigilance(double vigilance)          // 0.0-1.0
        public Builder learningRate(double learningRate)   // 0.0-1.0  
        public Builder bias(double bias)                    // Small positive
        public Builder maxCategories(int maxCategories)    // Memory limit
        public FuzzyParameters build()
    }
    
    // Access methods
    public double getVigilance()
    public double getLearningRate()
    public double getBias()
    public int getMaxCategories()
}
```

**Usage Example:**

```java
// Create network with custom parameters
var parameters = FuzzyParameters.builder()
    .vigilance(0.85)           // High selectivity
    .learningRate(0.1)         // Moderate learning speed
    .bias(0.001)              // Prevent division by zero
    .maxCategories(1000)       // Memory allocation
    .build();

try (var fuzzyART = new FuzzyART(parameters)) {
    // Training phase
    var patterns = Arrays.asList(
        new DenseVector(new double[]{0.8, 0.2, 0.9}),
        new DenseVector(new double[]{0.1, 0.9, 0.2}),
        new DenseVector(new double[]{0.7, 0.3, 0.8})
    );
    
    for (var pattern : patterns) {
        int category = fuzzyART.stepFit(pattern);
        System.out.printf("Pattern %s -> Category %d%n", 
            Arrays.toString(pattern.toArray()), category);
    }
    
    // Prediction phase
    var testPattern = new DenseVector(new double[]{0.75, 0.25, 0.85});
    var result = fuzzyART.predict(testPattern);
    
    System.out.printf("Predicted category: %d (confidence: %.3f)%n",
        result.getCategory(), result.getActivation());
        
    // Network analysis
    System.out.printf("Total categories: %d%n", fuzzyART.getCategoryCount());
    
    // Inspect category weights
    for (int i = 0; i < fuzzyART.getCategoryCount(); i++) {
        var weight = fuzzyART.getWeight(i);
        System.out.printf("Category %d weight: %s%n", i, 
            Arrays.toString(weight.getValues()));
    }
}
```

### HypersphereART

Geometric clustering using hyperspherical coordinates:

```java
public class HypersphereART extends BaseART<HypersphereParameters> {
    public HypersphereART(HypersphereParameters parameters)
    
    // Core methods inherited from BaseART
    public int stepFit(Pattern input)
    public CategoryResult predict(Pattern input)
    
    // Hypersphere-specific methods
    public double calculateDistance(Pattern input, int categoryIndex)
    public HypersphereWeight getHypersphere(int categoryIndex)
    public double[] getCenterVector(int categoryIndex)
    public double getRadius(int categoryIndex)
}
```

**Parameters:**

```java
public class HypersphereParameters implements Parameters {
    public static HypersphereParameters of(double vigilance, double learningRate, boolean adaptive)
    
    public static class Builder {
        public Builder vigilance(double vigilance)
        public Builder learningRate(double learningRate)
        public Builder adaptiveRadii(boolean adaptive)      // Dynamic radius adjustment
        public Builder initialRadius(double radius)         // Starting radius
        public Builder minRadius(double minRadius)          // Minimum allowed radius
        public Builder maxRadius(double maxRadius)          // Maximum allowed radius
        public HypersphereParameters build()
    }
}
```

### BayesianART

Probabilistic learning with uncertainty quantification:

```java
public class BayesianART extends BaseART<BayesianParameters> {
    public BayesianART(BayesianParameters parameters)
    
    // Enhanced prediction with uncertainty
    public BayesianActivationResult predictWithUncertainty(Pattern input)
    public double calculateEntropy(Pattern input)
    public double getConfidence(int categoryIndex, Pattern input)
    
    // Probabilistic methods
    public double[] getPosteriorProbabilities(Pattern input)
    public double getPriorProbability(int categoryIndex)
    public void updatePriors()
}
```

**Bayesian Results:**

```java
public class BayesianActivationResult extends ActivationResult {
    public double getConfidence()        // Classification confidence [0,1]
    public double getEntropy()          // Information entropy
    public double[] getProbabilities()  // Full probability distribution
    public double getVariance()         // Prediction variance
}
```

### ARTMAP (Supervised Learning)

Supervised learning with input-output mapping:

```java
public class ARTMAP implements AutoCloseable {
    public ARTMAP(ARTMAPParameters parameters)
    
    // Supervised learning
    public ARTMAPResult learn(Pattern input, Pattern target)
    public ARTMAPResult predict(Pattern input)
    
    // Network components
    public BaseART getArtA()           // Input processing network
    public BaseART getArtB()           // Output processing network  
    public Map<Integer, Integer> getMapField()  // Category associations
    
    // Match tracking
    public void setMatchTracking(boolean enabled)
    public boolean isMatchTrackingEnabled()
}
```

**ARTMAP Parameters:**

```java
public class ARTMAPParameters {
    public static ARTMAPParameters of(double vigilanceAB, double baseline, boolean matchTracking)
    
    public static class Builder {
        public Builder vigilanceAB(double vigilance)        // Inter-ART vigilance
        public Builder baseline(double baseline)            // Minimum activation
        public Builder matchTracking(boolean enabled)       // Enable match tracking
        public Builder maxSearchAttempts(int attempts)      // Search iteration limit
        public ARTMAPParameters build()
    }
}
```

**Usage Example:**

```java
var parameters = ARTMAPParameters.builder()
    .vigilanceAB(0.9)
    .baseline(0.001)
    .matchTracking(true)
    .maxSearchAttempts(100)
    .build();

try (var artmap = new ARTMAP(parameters)) {
    // Training data
    var inputs = Arrays.asList(
        new DenseVector(new double[]{0.8, 0.2, 0.9}),
        new DenseVector(new double[]{0.1, 0.9, 0.2}),
        new DenseVector(new double[]{0.7, 0.3, 0.8})
    );
    
    var targets = Arrays.asList(
        new DenseVector(new double[]{1.0}),  // Class 1
        new DenseVector(new double[]{2.0}),  // Class 2  
        new DenseVector(new double[]{1.0})   // Class 1
    );
    
    // Supervised training
    for (int i = 0; i < inputs.size(); i++) {
        var result = artmap.learn(inputs.get(i), targets.get(i));
        System.out.printf("Training sample %d: input category %d -> output category %d%n",
            i, result.getInputCategory(), result.getOutputCategory());
    }
    
    // Prediction
    var testInput = new DenseVector(new double[]{0.75, 0.25, 0.85});
    var prediction = artmap.predict(testInput);
    
    System.out.printf("Prediction: class %.0f (confidence: %.3f)%n",
        prediction.getPredictedOutput().getValue(0), 
        prediction.getConfidence());
}
```

## art-performance Module API

### Vectorized Implementations

High-performance SIMD-optimized implementations that extend the core algorithms:

#### VectorizedFuzzyART

```java
public class VectorizedFuzzyART extends FuzzyART {
    public VectorizedFuzzyART(VectorizedParameters parameters)
    
    // Batch processing methods
    public List<Integer> learnBatch(List<Pattern> patterns)
    public List<CategoryResult> predictBatch(List<Pattern> patterns)
    
    // Performance monitoring
    public VectorizedPerformanceStats getPerformanceStats()
    public boolean isSIMDEnabled()
    public int getParallelThreads()
    
    // Memory management
    public void compactMemory()
    public long getMemoryUsage()
}
```

#### VectorizedParameters

Enhanced parameter class with performance tuning options:

```java
public class VectorizedParameters extends FuzzyParameters {
    public static Builder builder()
    
    public static class Builder extends FuzzyParameters.Builder {
        // Performance settings
        public Builder enableSIMD(boolean enabled)           // Vector API
        public Builder parallelThreads(int threads)          // Thread pool size
        public Builder batchSize(int size)                   // Optimal batch size
        public Builder memoryStrategy(MemoryStrategy strategy) // Memory management
        
        // Algorithm parameters (inherited)
        public Builder vigilance(double vigilance)
        public Builder learningRate(double learningRate)
        public Builder inputDimensions(int dimensions)
        public Builder maxCategories(int categories)
        
        public VectorizedParameters build()
    }
    
    // Performance accessors
    public boolean isSIMDEnabled()
    public int getParallelThreads()
    public int getBatchSize()
    public MemoryStrategy getMemoryStrategy()
}
```

**Memory Strategies:**

```java
public enum MemoryStrategy {
    CONSERVATIVE,    // Minimize memory usage
    BALANCED,        // Balance speed and memory
    PERFORMANCE,     // Maximize performance
    POOL_REUSE       // Buffer pooling for repeated operations
}
```

**Usage Example:**

```java
var parameters = VectorizedParameters.builder()
    .vigilance(0.85)
    .learningRate(0.1)
    .inputDimensions(64)
    .maxCategories(10000)
    .enableSIMD(true)
    .parallelThreads(Runtime.getRuntime().availableProcessors())
    .batchSize(256)
    .memoryStrategy(MemoryStrategy.PERFORMANCE)
    .build();

try (var vectorizedART = new VectorizedFuzzyART(parameters)) {
    // Generate large dataset
    var patterns = IntStream.range(0, 10000)
        .mapToObj(i -> generateRandomPattern(64))
        .collect(Collectors.toList());
    
    // Batch learning for maximum throughput
    var startTime = System.nanoTime();
    var categories = vectorizedART.learnBatch(patterns);
    var duration = System.nanoTime() - startTime;
    
    // Performance analysis
    var stats = vectorizedART.getPerformanceStats();
    System.out.printf("Processed %d patterns in %.3f seconds%n", 
        patterns.size(), duration / 1e9);
    System.out.printf("Throughput: %.1f patterns/sec%n", 
        patterns.size() / (duration / 1e9));
    System.out.printf("SIMD speedup: %.1fx%n", stats.getSIMDSpeedup());
    System.out.printf("Memory usage: %.1f MB%n", stats.getMemoryUsageMB());
}
```

### VectorizedARTMAP

High-performance supervised learning:

```java
public class VectorizedARTMAP extends ARTMAP {
    public VectorizedARTMAP(VectorizedARTMAPParameters parameters)
    
    // Batch supervised learning
    public List<ARTMAPResult> learnBatch(List<Pattern> inputs, List<Pattern> targets)
    public List<ARTMAPResult> predictBatch(List<Pattern> inputs)
    
    // Parallel processing
    public CompletableFuture<ARTMAPResult> learnAsync(Pattern input, Pattern target)
    public Stream<ARTMAPResult> learnStream(Stream<Pair<Pattern, Pattern>> data)
}
```

## Results and Data Types

### CategoryResult

Basic classification result:

```java
public class CategoryResult {
    public int getCategory()              // Winning category index
    public double getActivation()         // Activation strength [0,1]
    public double getVigilanceMatch()     // Vigilance test result [0,1]
    public boolean isNewCategory()        // Whether a new category was created
    public long getTimestampNanos()       // Processing timestamp
}
```

### ActivationResult

Detailed activation information:

```java
public class ActivationResult extends CategoryResult {
    public double[] getAllActivations()   // All category activations
    public int[] getRankedCategories()    // Categories sorted by activation
    public double getCompetitionRatio()   // Winner vs runner-up ratio
    public Map<String, Double> getMetrics() // Algorithm-specific metrics
}
```

### ARTMAPResult

Supervised learning result:

```java
public class ARTMAPResult {
    public int getInputCategory()         // ART-A category
    public int getOutputCategory()        // ART-B category
    public Pattern getPredictedOutput()   // Predicted output pattern
    public double getConfidence()         // Prediction confidence
    public boolean wasMatchTrackingTriggered()  // Match tracking status
    public int getSearchIterations()      // Number of search iterations
}
```

## Utilities and Helper Classes

### DataBounds

Data normalization and bounds checking:

```java
public class DataBounds {
    public static DataBounds from(Collection<Pattern> patterns)
    public static DataBounds from(Pattern... patterns)
    
    // Normalization
    public Pattern normalize(Pattern input)
    public List<Pattern> normalizeAll(List<Pattern> patterns)
    
    // Bounds checking
    public boolean isWithinBounds(Pattern input)
    public Pattern clampToBounds(Pattern input)
    
    // Statistics
    public Pattern getMinValues()
    public Pattern getMaxValues()  
    public Pattern getMeanValues()
    public Pattern getStandardDeviation()
}
```

### ScikitClusterer

Compatibility layer for scikit-learn style clustering:

```java
public class ScikitClusterer {
    public ScikitClusterer(BaseART art)
    
    // Scikit-learn compatible interface
    public ScikitClusterer fit(double[][] X)
    public int[] predict(double[][] X)
    public int[] fit_predict(double[][] X)
    
    // Clustering metrics
    public double[] cluster_centers_()     // Category centroids
    public int[] labels_()                // Pattern labels
    public int n_clusters_()              // Number of clusters
    public double inertia_()              // Within-cluster sum of squares
}
```

## Error Handling

### Common Exceptions

```java
// Algorithm-specific exceptions
public class ARTException extends RuntimeException
public class VigilanceException extends ARTException
public class DimensionMismatchException extends ARTException
public class CategoryLimitException extends ARTException

// Performance-specific exceptions  
public class SIMDUnsupportedException extends ARTException
public class MemoryExhaustedException extends ARTException
```

### Exception Handling Examples

```java
try {
    var art = new FuzzyART(parameters);
    var result = art.stepFit(pattern);
} catch (DimensionMismatchException e) {
    System.err.printf("Pattern dimension %d doesn't match expected %d%n",
        e.getActualDimension(), e.getExpectedDimension());
} catch (CategoryLimitException e) {
    System.err.printf("Maximum categories (%d) exceeded%n", e.getMaxCategories());
    // Consider increasing maxCategories or using category pruning
} catch (ARTException e) {
    System.err.printf("ART algorithm error: %s%n", e.getMessage());
}
```

## Thread Safety

### Thread Safety Guarantees

- **Immutable Classes**: `Pattern`, `Parameters`, and `Result` classes are immutable
- **Thread-Safe Algorithms**: All ART implementations are thread-safe for concurrent prediction
- **Learning Synchronization**: Learning operations (`stepFit`, `learn`) require external synchronization
- **Vectorized Performance**: `art-performance` classes include optimized thread-safe implementations

### Concurrent Usage Patterns

```java
// Safe: Concurrent prediction after training
var art = new FuzzyART(parameters);
// ... training phase (single-threaded) ...

// Multiple threads can safely predict concurrently
var patterns = getTestPatterns();
var results = patterns.parallelStream()
    .map(art::predict)
    .collect(Collectors.toList());

// Safe: Using vectorized implementations for concurrent learning
var vectorizedART = new VectorizedFuzzyART(vectorizedParameters);
var categories = patterns.parallelStream()  // Built-in thread safety
    .map(vectorizedART::learn)
    .collect(Collectors.toList());
```

## Performance Guidelines

### Best Practices

1. **Choose the Right Implementation**:
   - Use `art-core` for small datasets (<1000 patterns)
   - Use `art-performance` for large datasets (>1000 patterns)
   - Enable SIMD for numerical computations

2. **Memory Management**:
   - Use try-with-resources for automatic cleanup
   - Monitor memory usage with performance stats
   - Consider memory strategies for different use cases

3. **Batch Processing**:
   - Process patterns in batches for better throughput
   - Use optimal batch sizes (typically 64-512 patterns)
   - Leverage parallel streams for I/O-bound operations

4. **Parameter Tuning**:
   - Start with conservative parameters for stable learning
   - Increase vigilance for more selective categories
   - Adjust learning rate based on data characteristics

### Performance Monitoring

```java
// Enable detailed performance monitoring
var monitor = new ARTPerformanceMonitor();
var monitoredART = new MonitoredVectorizedART(vectorizedART, monitor);

// Process data
var results = monitoredART.learnBatch(patterns);

// Analyze performance
var metrics = monitor.getDetailedMetrics();
System.out.printf("Throughput: %.1f patterns/sec%n", metrics.getThroughput());
System.out.printf("Memory efficiency: %.1f%%%n", metrics.getMemoryEfficiency());
System.out.printf("SIMD utilization: %.1f%%%n", metrics.getSIMDUtilization());
System.out.printf("Thread utilization: %.1f%%%n", metrics.getThreadUtilization());
```

This comprehensive API documentation covers all major classes, methods, and usage patterns in the ART library. For additional examples and advanced usage patterns, see the test classes and benchmark implementations in the source code.