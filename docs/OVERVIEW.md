# ART Algorithms Overview

This document provides detailed information about each Adaptive Resonance Theory algorithm implementation.

## Table of Contents
- [Unsupervised Learning Algorithms](#unsupervised-learning-algorithms)
  - [DualVigilanceART](#dualvigilanceart)
  - [TopoART](#topoart)
  - [FuzzyART](#fuzzyart)
  - [BayesianART](#bayesianart)
  - [GaussianART](#gaussianart)
  - [HypersphereART](#hypersphereart)
  - [ART-2](#art-2)
- [Supervised Learning Algorithms](#supervised-learning-algorithms)
  - [ARTMAP](#artmap)
  - [DeepARTMAP](#deepartmap)
- [Temporal Processing Algorithms](#temporal-processing-algorithms)
  - [TemporalART](#temporalart)

---

## Unsupervised Learning Algorithms

### DualVigilanceART

**DualVigilanceART** introduces a dual-threshold system that significantly improves noise handling and cluster boundary definition. It uses two vigilance parameters to distinguish between core cluster patterns and boundary/noise patterns.

#### Key Features

- **Dual vigilance thresholds**: Upper vigilance (ρ) for core patterns, lower vigilance (ρ_lb) for boundary detection
- **Boundary node mechanism**: Patterns failing upper but passing lower vigilance become boundary nodes
- **Improved noise isolation**: Noisy patterns are isolated without proliferating categories
- **Cluster integrity**: Maintains clean cluster cores while handling outliers gracefully

#### Basic Usage

```java
import com.hellblazer.art.core.algorithms.DualVigilanceART;
import com.hellblazer.art.core.parameters.DualVigilanceParameters;

// Configure dual vigilance parameters
var params = new DualVigilanceParameters(
    0.4,    // Lower vigilance (ρ_lb) - boundary threshold
    0.7,    // Upper vigilance (ρ) - core threshold  
    0.1,    // Learning rate (β)
    0.001,  // Choice parameter
    1000    // Max categories
);

var dualART = new DualVigilanceART();

// Train with patterns
var corePattern = new DenseVector(new double[]{0.9, 0.9});
var boundaryPattern = new DenseVector(new double[]{0.3, 0.3});
var noisePattern = new DenseVector(new double[]{0.1, 0.1});

// Learn patterns - automatically determines core vs boundary
var result1 = dualART.stepFit(corePattern, params);
var result2 = dualART.stepFit(boundaryPattern, params);
var result3 = dualART.stepFit(noisePattern, params);

// Check boundary node status
System.out.printf("Core pattern -> Category %d (boundary: %s)%n", 
    result1.categoryIndex(), dualART.isBoundaryNode(result1.categoryIndex()));
System.out.printf("Boundary nodes: %d/%d categories%n",
    dualART.getBoundaryNodeCount(), dualART.getCategoryCount());
```

#### When to Use DualVigilanceART

DualVigilanceART is particularly effective for:
- **Noisy datasets**: Isolates noise without creating excessive categories
- **Outlier detection**: Boundary nodes naturally identify outliers
- **Cluster boundary analysis**: Distinguishes core from peripheral patterns
- **Robust clustering**: Maintains cluster purity despite noise

#### Parameter Guidelines

| Parameter | Range | Description | Tuning Tips |
|-----------|--------|-------------|-------------|
| `ρ_lb` (lower) | 0.0-1.0 | Boundary detection threshold | Lower = more permissive boundaries |
| `ρ` (upper) | ρ_lb-1.0 | Core pattern threshold | Higher = tighter clusters |
| `β` (learning) | 0.0-1.0 | Weight adaptation rate | Lower = more stable learning |

**Important**: Lower vigilance must be less than upper vigilance (ρ_lb < ρ)

---

### TopoART

**TopoART** is a hierarchical ART architecture that learns topological structure through edge formation between neurons. This makes it particularly effective for clustering problems where the spatial relationships between patterns matter.

#### Key Features

- **Dual-component architecture**: Combines pattern matching with topological edge formation
- **Permanence mechanism**: Neurons become permanent when they reach a stability threshold
- **Connected component clustering**: Groups permanent neurons based on learned topology
- **Complement coding**: Automatically transforms input patterns for improved stability

#### Basic Usage

```java
import com.hellblazer.art.core.algorithms.TopoART;
import com.hellblazer.art.core.parameters.TopoARTParameters;

// Configure TopoART parameters
var params = TopoARTParameters.builder()
    .vigilanceA(0.85)        // Pattern selectivity component A
    .vigilanceB(0.85)        // Pattern selectivity component B  
    .learningRate(0.1)       // Speed of weight adaptation
    .phi(3)                  // Permanence threshold
    .build();

var topoART = new TopoART(params);

// Train with patterns
var patterns = Arrays.asList(
    new DenseVector(new double[]{0.8, 0.2}),
    new DenseVector(new double[]{0.7, 0.3}),
    new DenseVector(new double[]{0.2, 0.8})
);

// Learn patterns and form topology
for (var pattern : patterns) {
    var result = topoART.stepFit(pattern);
    System.out.printf("Pattern learned -> Neuron %d (permanent: %s)%n", 
                     result.getNeuron(), result.isPermanent());
}

// Extract learned clusters
var clusters = topoART.getClusters();
System.out.printf("Found %d clusters%n", clusters.size());

clusters.forEach((component, cluster) -> {
    System.out.printf("Cluster %d: %d neurons, %d edges%n",
                     component, 
                     cluster.getNeuronIndices().size(),
                     cluster.getEdgeCount());
});
```

#### Advanced Configuration

```java
// Fine-tune for specific clustering behavior
var advancedParams = TopoARTParameters.builder()
    .vigilanceA(0.9)         // High selectivity for precise clusters
    .vigilanceB(0.7)         // Lower selectivity allows broader grouping
    .learningRate(0.05)      // Slower learning for stability
    .phi(5)                  // Higher threshold for permanence
    .maxIterations(1000)     // Extended training
    .complementCoding(true)  // Enable complement coding (default)
    .build();

var preciseTopoART = new TopoART(advancedParams);

// Monitor training progress
for (int iteration = 0; iteration < patterns.size(); iteration++) {
    var result = preciseTopoART.stepFit(patterns.get(iteration));
    
    if (result.isResonance()) {
        System.out.printf("Iteration %d: Resonance achieved%n", iteration);
    }
    
    if (result.isPermanent()) {
        System.out.printf("Iteration %d: Neuron %d became permanent%n", 
                         iteration, result.getNeuron());
    }
}
```

#### Understanding TopoART Behavior

TopoART creates edges between the best and second-best matching neurons during learning, which can result in highly connected topological structures. This is correct algorithm behavior:

```java
// Analysis of topological structure
var metrics = topoART.getTopologyMetrics();
System.out.printf("Total neurons: %d%n", metrics.getTotalNeurons());
System.out.printf("Permanent neurons: %d%n", metrics.getPermanentNeurons());
System.out.printf("Total edges: %d%n", metrics.getTotalEdges());
System.out.printf("Connectivity ratio: %.3f%n", metrics.getConnectivityRatio());

// Cluster analysis
var clusterAnalysis = topoART.analyzeClusterStructure();
clusterAnalysis.getClusters().forEach(cluster -> {
    System.out.printf("Cluster size: %d, density: %.3f, diameter: %d%n",
                     cluster.getSize(), 
                     cluster.getDensity(),
                     cluster.getDiameter());
});
```

#### Parameter Guidelines

| Parameter | Range | Description | Tuning Tips |
|-----------|--------|-------------|-------------|
| `vigilanceA` | 0.0-1.0 | Component A selectivity | Higher = more precise clusters |
| `vigilanceB` | 0.0-1.0 | Component B selectivity | Can differ from A for asymmetric learning |
| `learningRate` | 0.0-1.0 | Weight adaptation speed | Lower = more stable, slower convergence |
| `phi` | 1+ | Permanence threshold | Higher = fewer but more stable permanent neurons |

---

### FuzzyART

**FuzzyART** extends ART to handle analog input patterns using fuzzy set theory operations. It's the most general-purpose ART algorithm for continuous-valued data.

#### Key Features
- Complement coding for pattern normalization
- Fuzzy AND/OR operations for pattern matching
- Fast learning with stable category formation
- Handles both binary and continuous inputs

#### When to Use
- General pattern recognition tasks
- When input ranges vary significantly
- Need for fast, stable learning
- Mixed binary/analog data

---

### BayesianART

**BayesianART** incorporates probabilistic reasoning into the ART framework, providing uncertainty estimates alongside classifications.

#### Key Features
- Bayesian inference for category selection
- Confidence and entropy estimates
- Probabilistic weight updates
- Uncertainty quantification

#### Basic Usage

```java
var bayesianART = new BayesianART(BayesianParameters.conservative(4));
var result = bayesianART.stepFit(pattern);

// Get uncertainty estimates
double confidence = result.getConfidence();
double entropy = result.getEntropy();
System.out.printf("Classification confidence: %.3f (entropy: %.3f)%n", confidence, entropy);
```

#### When to Use
- Need confidence estimates with predictions
- Safety-critical applications
- Small sample sizes
- When understanding model uncertainty is important

---

### GaussianART

**GaussianART** models categories as Gaussian distributions, ideal for normally-distributed data clusters.

#### Key Features
- Gaussian probability distributions
- Mean and variance tracking
- Statistical distance metrics
- Adaptive variance estimation

#### When to Use
- Naturally Gaussian-distributed data
- Need for statistical modeling
- Continuous sensor data
- Quality control applications

---

### HypersphereART

**HypersphereART** represents categories as hyperspheres in n-dimensional space, providing rotation-invariant pattern recognition.

#### Key Features
- Hyperspherical geometry
- Rotation and scale invariance
- Radius-based vigilance testing
- Geometric distance metrics

#### When to Use
- Geometric pattern recognition
- Rotation-invariant clustering
- Spatial data analysis
- Computer vision applications

---

### ART-2

**ART-2** extends ART-1 with preprocessing layers for handling continuous inputs with noise suppression.

#### Key Features
- Built-in normalization layers
- Noise suppression mechanisms
- Contrast enhancement
- Multi-stage processing

#### When to Use
- Noisy analog inputs
- Need for preprocessing
- Signal processing applications
- When normalization is critical

---

## Supervised Learning Algorithms

### ARTMAP

**ARTMAP** combines two ART modules for supervised learning, mapping input patterns to output categories.

#### Key Features
- Input-output association learning
- Match tracking for error correction
- Incremental learning capability
- Fast stable learning

#### Basic Usage

```java
import com.hellblazer.art.core.artmap.ARTMAP;
import com.hellblazer.art.core.artmap.ARTMAPParameters;

// Create supervised learning network
var artmapParams = ARTMAPParameters.of(0.9, 0.001, true);  // vigilance, baseline, match tracking
var artmap = new ARTMAP(artmapParams);

// Train with input-output pairs
var input = new DenseVector(new double[]{0.8, 0.2, 0.9});
var target = new DenseVector(new double[]{1.0});  // Class label

var result = artmap.learn(input, target);
System.out.printf("Learned association: input -> category %d%n", result.getCategory());

// Make predictions
var prediction = artmap.predict(input);
System.out.printf("Prediction confidence: %.3f%n", prediction.getActivation());
```

#### When to Use
- Classification tasks
- Online learning requirements
- Need for incremental learning
- Pattern-to-pattern mapping

---

### DeepARTMAP

**DeepARTMAP** extends ARTMAP with hierarchical multi-layer processing for complex feature learning.

#### Key Features
- Multi-layer architecture
- Hierarchical feature extraction
- Layer-wise vigilance control
- Deep representation learning

#### Basic Usage

```java
var deepParams = DeepARTMAPParameters.builder()
    .layers(3)                    // 3-layer hierarchy
    .vigilanceSchedule(0.9, 0.7, 0.5)  // Decreasing vigilance per layer
    .build();

var deepARTMAP = new DeepARTMAP(deepParams);
var hierarchicalResult = deepARTMAP.learnHierarchical(inputPattern, targetPattern);

// Access per-layer activations
for (int layer = 0; layer < 3; layer++) {
    var activation = hierarchicalResult.getLayerActivation(layer);
    System.out.printf("Layer %d activation: %.3f%n", layer, activation);
}
```

#### When to Use
- Complex pattern relationships
- Need for feature hierarchy
- Large-scale classification
- When single-layer ARTMAP is insufficient

---

## Temporal Processing Algorithms

### TemporalART

**TemporalART** implements temporal sequence processing based on Kazerounian & Grossberg (2014), providing working memory, temporal chunking, and sequence learning capabilities.

#### Key Features

- **Working memory**: STORE 2 model with primacy and recency gradients
- **Multi-scale chunking**: Item, chunk, and list scales with asymmetric inhibition
- **Transmitter habituation**: Activity-dependent gating for temporal segmentation
- **Time scale separation**: Fast (10-100ms), medium (50-500ms), slow (500-5000ms) dynamics
- **Cognitive phenomena**: Reproduces Miller's 7±2, serial position effects, phone number chunking

#### Architecture

The temporal system consists of three main components:

1. **Working Memory** - Maintains sequences with position-dependent activation
2. **Masking Field** - Multi-scale competitive dynamics for chunk formation
3. **Transmitter Gates** - Habituation mechanisms for reset and segmentation

#### Basic Usage

```java
import com.hellblazer.art.temporal.integration.TemporalART;
import com.hellblazer.art.temporal.integration.TemporalARTParameters;

// Configure for speech processing
var params = TemporalARTParameters.speechDefaults();
var temporalART = new TemporalART(params);

// Process sequence of patterns
List<double[]> sequence = Arrays.asList(
    new double[]{1.0, 0.0, 0.0},  // First item
    new double[]{0.0, 1.0, 0.0},  // Second item
    new double[]{0.0, 0.0, 1.0}   // Third item
);

temporalART.processSequence(sequence);

// Retrieve learned chunks
var categories = temporalART.getCategories();
var statistics = temporalART.getStatistics();
```

#### Parameter Configuration

```java
// Different processing modes
var speechParams = TemporalARTParameters.speechDefaults();      // Speech segmentation
var listParams = TemporalARTParameters.listLearningDefaults();  // List learning
var customParams = TemporalARTParameters.builder()
    .workingMemoryCapacity(7)
    .primacyGradientStrength(0.3)
    .chunkSizePreference(3)
    .transmitterRecoveryRate(0.01)
    .build();
```

#### Performance

Temporal processing includes vectorized implementations with significant speedups:
- Working Memory operations: 14x faster
- Shunting dynamics: 1.53x faster
- Full SIMD optimization via Java Vector API

#### Applications

- **Speech processing**: Word segmentation, phoneme clustering
- **Sequence learning**: Pattern sequences, temporal associations
- **Cognitive modeling**: Memory span, serial recall, chunking
- **Time series**: Temporal pattern recognition

#### Mathematical Foundation

Based on the paper ["Real-time learning of predictive recognition categories that chunk sequences of items stored in working memory"](https://doi.org/10.3389/fpsyg.2014.01053) by Kazerounian & Grossberg (2014). The implementation includes:

- Shunting on-center off-surround dynamics (Equation 1)
- Transmitter habituation dynamics (Equation 2)
- Item node activation (Equation 3)
- List chunk activation (Equation 4)

All equations are validated in the temporal-validation module with 95% fidelity to the original paper.

---

## Performance Considerations

All algorithms have vectorized counterparts in the `art-performance` module:
- `VectorizedFuzzyART`, `VectorizedHypersphereART`, etc.
- 6-8x performance improvement over base implementations
- Automatic fallback for edge cases
- Full API compatibility

### Choosing the Right Algorithm

| Task | Recommended Algorithm | Why |
|------|----------------------|-----|
| General clustering | FuzzyART | Versatile, well-tested |
| Noisy data | DualVigilanceART | Boundary node mechanism |
| Spatial relationships | TopoART | Topology learning |
| Need confidence | BayesianART | Uncertainty quantification |
| Classification | ARTMAP | Supervised learning |
| Complex patterns | DeepARTMAP | Hierarchical features |
| Geometric data | HypersphereART | Rotation invariance |
| Statistical modeling | GaussianART | Gaussian distributions |