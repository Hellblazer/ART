# TopoART Implementation Guide for Claude Code

## Executive Summary

This document provides a comprehensive guide for implementing TopoART (Topology Learning Hierarchical ART Network) based on the paper by Marko Tscherepanow (2010). TopoART combines Adaptive Resonance Theory (ART) with topology learning to enable stable online clustering of stationary and non-stationary data while creating hierarchical representations at different levels of detail.

## Core Architecture Overview

TopoART consists of:
- **Two parallel Fuzzy ART-like components** (TopoART a and TopoART b)
- **Shared input layer** (F0) performing complement coding
- **Topology learning** through edge connections between neurons
- **Noise filtering** mechanism using node candidates and permanent nodes
- **Hierarchical representation** with Component B providing finer detail than Component A

## Implementation Phases

### Phase 1: Core Data Structures and Foundation
**Goal:** Establish basic building blocks with comprehensive testing

#### 1.1 Data Structures to Implement

```java
// Neuron.java
public class Neuron {
    private double[] weights;        // Weight vector (size 2d for complement coding)
    private int counter;             // Number of samples learned
    private Set<Integer> edges;      // Connected neuron indices
    private boolean isPermanent;     // true if counter >= φ
    
    // Constructor, getters, setters
}

// Layer.java
public abstract class Layer {
    protected List<Neuron> neurons;
    // Abstract methods for layer operations
}

// TopoARTComponent.java
public class TopoARTComponent {
    private Layer F0, F1, F2;
    private double vigilance;        // ρa or ρb
    private double learningRate;     // βsbm for second-best
    private int phi;                 // Permanence threshold
    private int tau;                 // Cleanup cycle period
    
    // Learning methods
}
```

#### 1.2 Tests for Phase 1

```java
// Test: Neuron creation and initialization
@Test
public void testNeuronCreation() {
    Neuron n = new Neuron(10); // dimension 5, complement coded to 10
    assertNotNull(n.getWeights());
    assertEquals(10, n.getWeights().length);
    assertEquals(0, n.getCounter());
    assertFalse(n.isPermanent());
}

// Test: Layer structure
@Test
public void testLayerStructure() {
    TopoARTComponent component = new TopoARTComponent(5, 0.92, 0.6, 5, 100);
    assertNotNull(component.getF0Layer());
    assertNotNull(component.getF1Layer());
    assertNotNull(component.getF2Layer());
}

// Test: Edge management
@Test
public void testEdgeOperations() {
    Neuron n1 = new Neuron(10);
    Neuron n2 = new Neuron(10);
    n1.addEdge(1); // Connect to neuron at index 1
    assertTrue(n1.hasEdgeTo(1));
}
```

### Phase 2: Mathematical Operations
**Goal:** Implement and test all mathematical functions

#### 2.1 Mathematical Functions to Implement

```java
// MathOperations.java
public class MathOperations {
    
    // Complement coding: x → [x, 1-x]
    public static double[] complementCode(double[] input) {
        double[] coded = new double[input.length * 2];
        for (int i = 0; i < input.length; i++) {
            coded[i] = input[i];
            coded[input.length + i] = 1.0 - input[i];
        }
        return coded;
    }
    
    // Component-wise minimum
    public static double[] componentWiseMin(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = Math.min(a[i], b[i]);
        }
        return result;
    }
    
    // City block norm (L1 norm)
    public static double cityBlockNorm(double[] vector) {
        double sum = 0;
        for (double v : vector) {
            sum += Math.abs(v);
        }
        return sum;
    }
    
    // Activation function (choice function)
    // z_i = |x ∧ w_i|₁ / (α + |w_i|₁)
    public static double activation(double[] input, double[] weights, double alpha) {
        double[] minVec = componentWiseMin(input, weights);
        double numerator = cityBlockNorm(minVec);
        double denominator = alpha + cityBlockNorm(weights);
        return numerator / denominator;
    }
    
    // Match function
    // |x ∧ w|₁ / |x|₁ ≥ ρ
    public static boolean matchFunction(double[] input, double[] weights, double vigilance) {
        double[] minVec = componentWiseMin(input, weights);
        double ratio = cityBlockNorm(minVec) / cityBlockNorm(input);
        return ratio >= vigilance;
    }
    
    // Category size calculation
    public static double categorySize(double[] weights) {
        int d = weights.length / 2;
        double size = 0;
        for (int j = 0; j < d; j++) {
            size += (1 - weights[d + j] - weights[j]);
        }
        return size;
    }
}
```

#### 2.2 Tests for Phase 2

```java
// Test: Complement coding
@Test
public void testComplementCoding() {
    double[] input = {0.3, 0.7, 0.5};
    double[] coded = MathOperations.complementCode(input);
    assertEquals(6, coded.length);
    assertEquals(0.3, coded[0], 1e-10);
    assertEquals(0.7, coded[1], 1e-10);
    assertEquals(0.5, coded[2], 1e-10);
    assertEquals(0.7, coded[3], 1e-10);  // 1 - 0.3
    assertEquals(0.3, coded[4], 1e-10);  // 1 - 0.7
    assertEquals(0.5, coded[5], 1e-10);  // 1 - 0.5
}

// Test: Activation function
@Test
public void testActivation() {
    double[] input = {0.3, 0.7, 0.5, 0.7, 0.3, 0.5};
    double[] weights = {0.2, 0.6, 0.4, 0.8, 0.4, 0.6};
    double alpha = 0.001;
    double activation = MathOperations.activation(input, weights, alpha);
    assertTrue(activation > 0 && activation <= 1);
}

// Test: Match function
@Test
public void testMatchFunction() {
    double[] input = {0.3, 0.7, 0.7, 0.3};
    double[] weights = {0.25, 0.65, 0.75, 0.35};
    double vigilance = 0.9;
    boolean matches = MathOperations.matchFunction(input, weights, vigilance);
    assertNotNull(matches);
}
```

### Phase 3: Single Component Learning Algorithm
**Goal:** Implement the core learning loop for a single TopoART component

#### 3.1 Learning Algorithm Implementation

```java
// TopoARTComponent.java - Learning methods
public class TopoARTComponent {
    
    // Find best and second-best matching neurons
    public MatchResult findBestMatches(double[] input) {
        double bestActivation = -1;
        double secondBestActivation = -1;
        int bestIndex = -1;
        int secondBestIndex = -1;
        
        for (int i = 0; i < F2.neurons.size(); i++) {
            double activation = MathOperations.activation(
                input, F2.neurons.get(i).getWeights(), alpha
            );
            
            if (activation > bestActivation) {
                secondBestActivation = bestActivation;
                secondBestIndex = bestIndex;
                bestActivation = activation;
                bestIndex = i;
            } else if (activation > secondBestActivation) {
                secondBestActivation = activation;
                secondBestIndex = i;
            }
        }
        
        return new MatchResult(bestIndex, secondBestIndex, 
                                bestActivation, secondBestActivation);
    }
    
    // Process single input
    public LearningResult learn(double[] input) {
        // 1. Apply complement coding
        double[] coded = MathOperations.complementCode(input);
        
        // 2. Find best matches
        MatchResult matches = findBestMatches(coded);
        
        // 3. Check resonance for best match
        boolean resonance = false;
        if (matches.bestIndex >= 0) {
            Neuron bestNeuron = F2.neurons.get(matches.bestIndex);
            resonance = MathOperations.matchFunction(
                coded, bestNeuron.getWeights(), vigilance
            );
        }
        
        // 4. If no resonance, try other neurons or create new
        if (!resonance) {
            // Implementation of vigilance reset and new neuron creation
            Neuron newNeuron = new Neuron(coded.length);
            newNeuron.setWeights(coded.clone());
            F2.neurons.add(newNeuron);
            matches.bestIndex = F2.neurons.size() - 1;
            resonance = true;
        }
        
        // 5. Update weights if resonance
        if (resonance) {
            // Update best matching neuron (fast learning, β = 1)
            Neuron bestNeuron = F2.neurons.get(matches.bestIndex);
            double[] newWeights = MathOperations.componentWiseMin(
                coded, bestNeuron.getWeights()
            );
            bestNeuron.setWeights(newWeights);
            bestNeuron.incrementCounter();
            
            // Update second best if exists (partial learning)
            if (matches.secondBestIndex >= 0) {
                Neuron secondBest = F2.neurons.get(matches.secondBestIndex);
                if (MathOperations.matchFunction(coded, secondBest.getWeights(), vigilance)) {
                    updateSecondBest(coded, secondBest, learningRateSecond);
                    // Create edge between best and second best
                    bestNeuron.addEdge(matches.secondBestIndex);
                    secondBest.addEdge(matches.bestIndex);
                }
            }
        }
        
        return new LearningResult(resonance, matches.bestIndex);
    }
    
    // Cleanup: Remove node candidates with counter < phi
    public void cleanup() {
        Iterator<Neuron> iterator = F2.neurons.iterator();
        while (iterator.hasNext()) {
            Neuron neuron = iterator.next();
            if (neuron.getCounter() < phi) {
                iterator.remove();
            } else {
                neuron.setPermanent(true);
            }
        }
    }
}
```

#### 3.2 Tests for Phase 3

```java
// Test: Best matching neuron selection
@Test
public void testBestMatchSelection() {
    TopoARTComponent component = new TopoARTComponent(3, 0.9, 0.6, 5, 100);
    double[] input = {0.3, 0.7, 0.5};
    
    // Add some neurons
    component.addNeuron(new double[]{0.25, 0.65, 0.45, 0.75, 0.35, 0.55});
    component.addNeuron(new double[]{0.35, 0.75, 0.55, 0.65, 0.25, 0.45});
    
    MatchResult matches = component.findBestMatches(
        MathOperations.complementCode(input)
    );
    
    assertNotEquals(-1, matches.bestIndex);
    assertTrue(matches.bestActivation > 0);
}

// Test: Weight update
@Test
public void testWeightUpdate() {
    TopoARTComponent component = new TopoARTComponent(2, 0.9, 0.6, 5, 100);
    double[] input = {0.5, 0.5};
    double[] initialWeights = {0.7, 0.7, 0.3, 0.3};
    
    component.addNeuron(initialWeights.clone());
    LearningResult result = component.learn(input);
    
    assertTrue(result.resonance);
    
    // Weights should be min(input, weights) after fast learning
    double[] expectedWeights = {0.5, 0.5, 0.5, 0.5};
    assertArrayEquals(expectedWeights, 
                      component.getNeuron(0).getWeights(), 1e-10);
}

// Test: New neuron creation
@Test
public void testNewNeuronCreation() {
    TopoARTComponent component = new TopoARTComponent(2, 0.99, 0.6, 5, 100);
    int initialSize = component.getNeuronCount();
    
    double[] input = {0.1, 0.1};  // Very different from any existing neurons
    component.learn(input);
    
    assertEquals(initialSize + 1, component.getNeuronCount());
}

// Test: Node cleanup
@Test
public void testNodeCleanup() {
    TopoARTComponent component = new TopoARTComponent(2, 0.9, 0.6, 3, 100);
    
    // Add neurons with different counters
    Neuron n1 = new Neuron(4);
    n1.setCounter(5);  // Will become permanent
    Neuron n2 = new Neuron(4);
    n2.setCounter(2);  // Will be removed
    
    component.addNeuron(n1);
    component.addNeuron(n2);
    
    component.cleanup();
    
    assertEquals(1, component.getNeuronCount());
    assertTrue(component.getNeuron(0).isPermanent());
}
```

### Phase 4: Dual Component System
**Goal:** Implement the full TopoART network with two interacting components

#### 4.1 Full TopoART Implementation

```java
// TopoART.java
public class TopoART {
    private TopoARTComponent componentA;
    private TopoARTComponent componentB;
    private int learningCycle = 0;
    
    public TopoART(int dimension, double vigilanceA, 
                   double learningRateSecond, int phi, int tau) {
        this.componentA = new TopoARTComponent(
            dimension, vigilanceA, learningRateSecond, phi, tau
        );
        
        // Component B has higher vigilance (Equation 11)
        double vigilanceB = 0.5 * (vigilanceA + 1);
        this.componentB = new TopoARTComponent(
            dimension, vigilanceB, learningRateSecond, phi, tau
        );
    }
    
    // Main learning method
    public void learn(double[] input) {
        // 1. Process through Component A
        LearningResult resultA = componentA.learn(input);
        
        // 2. If resonance in A and counter >= phi, propagate to B
        if (resultA.resonance && resultA.bestIndex >= 0) {
            Neuron bestNeuronA = componentA.getNeuron(resultA.bestIndex);
            if (bestNeuronA.getCounter() >= componentA.getPhi()) {
                componentB.learn(input);
            }
        }
        
        // 3. Periodic cleanup
        learningCycle++;
        if (learningCycle % componentA.getTau() == 0) {
            componentA.cleanup();
            componentB.cleanup();
        }
    }
    
    // Get clusters from connected components
    public List<Cluster> getClusters(boolean useComponentB) {
        TopoARTComponent component = useComponentB ? componentB : componentA;
        return extractConnectedComponents(component);
    }
    
    // Extract connected components using DFS/BFS
    private List<Cluster> extractConnectedComponents(TopoARTComponent component) {
        List<Cluster> clusters = new ArrayList<>();
        Set<Integer> visited = new HashSet<>();
        
        for (int i = 0; i < component.getNeuronCount(); i++) {
            if (!visited.contains(i) && component.getNeuron(i).isPermanent()) {
                Cluster cluster = new Cluster();
                dfs(component, i, visited, cluster);
                clusters.add(cluster);
            }
        }
        
        return clusters;
    }
    
    private void dfs(TopoARTComponent component, int nodeIndex, 
                     Set<Integer> visited, Cluster cluster) {
        visited.add(nodeIndex);
        cluster.addNeuron(nodeIndex);
        
        Neuron neuron = component.getNeuron(nodeIndex);
        for (int neighbor : neuron.getEdges()) {
            if (!visited.contains(neighbor) && 
                component.getNeuron(neighbor).isPermanent()) {
                dfs(component, neighbor, visited, cluster);
            }
        }
    }
}
```

#### 4.2 Tests for Phase 4

```java
// Test: Inter-component communication
@Test
public void testInterComponentPropagation() {
    TopoART network = new TopoART(3, 0.9, 0.6, 3, 100);
    
    // Train with enough samples to make a neuron permanent
    double[] input = {0.5, 0.5, 0.5};
    for (int i = 0; i < 3; i++) {
        network.learn(input);
    }
    
    // Component B should have received input
    assertTrue(network.getComponentB().getNeuronCount() > 0);
}

// Test: Hierarchical representation
@Test
public void testHierarchicalClustering() {
    TopoART network = new TopoART(2, 0.85, 0.6, 3, 100);
    
    // Train with two distinct clusters
    for (int i = 0; i < 10; i++) {
        network.learn(new double[]{0.2 + Math.random() * 0.1, 
                                    0.2 + Math.random() * 0.1});
        network.learn(new double[]{0.7 + Math.random() * 0.1, 
                                    0.7 + Math.random() * 0.1});
    }
    
    List<Cluster> clustersA = network.getClusters(false);
    List<Cluster> clustersB = network.getClusters(true);
    
    // Component B should have more detailed clustering
    assertTrue(clustersB.size() >= clustersA.size());
}

// Test: Stability-plasticity
@Test
public void testStabilityPlasticity() {
    TopoART network = new TopoART(2, 0.9, 0.6, 5, 100);
    
    // Train with initial data
    double[] input1 = {0.3, 0.3};
    for (int i = 0; i < 10; i++) {
        network.learn(input1);
    }
    
    // Get initial clusters
    List<Cluster> initialClusters = network.getClusters(false);
    int initialNeuronCount = network.getComponentA().getNeuronCount();
    
    // Train with same data again
    for (int i = 0; i < 10; i++) {
        network.learn(input1);
    }
    
    // Should maintain stability (no new neurons for same data)
    assertEquals(initialNeuronCount, network.getComponentA().getNeuronCount());
    
    // Train with new data
    double[] input2 = {0.8, 0.8};
    network.learn(input2);
    
    // Should show plasticity (new neuron for new data)
    assertTrue(network.getComponentA().getNeuronCount() > initialNeuronCount);
}
```

### Phase 5: Advanced Features and Optimization
**Goal:** Implement performance optimizations and advanced features

#### 5.1 Performance Optimizations

```java
// OptimizedTopoART.java
public class OptimizedTopoART extends TopoART {
    private KDTree neuronIndex;  // For efficient nearest neighbor search
    private double[] cachedSizes;  // Cache category sizes
    private boolean[] dirtyFlags;  // Track which sizes need recalculation
    
    // Parallel processing for large datasets
    public void learnBatch(List<double[]> inputs) {
        inputs.parallelStream().forEach(this::learn);
    }
    
    // Efficient activation calculation using vectorization
    public double[] computeActivationsBatch(double[] input) {
        // Use vector operations library (e.g., Apache Commons Math)
        // for efficient batch computation
    }
}
```

#### 5.2 Validation and Metrics

```java
// ValidationMetrics.java
public class ValidationMetrics {
    
    // Jaccard coefficient
    public static double jaccardCoefficient(List<Cluster> predicted, 
                                             List<Cluster> actual) {
        // Implementation
    }
    
    // Rand index
    public static double randIndex(List<Cluster> predicted, 
                                   List<Cluster> actual) {
        // Implementation
    }
    
    // Visualization helper
    public static void visualizeClusters(TopoART network, 
                                          String outputPath) {
        // Generate visualization using a plotting library
    }
}
```

## Critical Implementation Details

### 1. Parameter Settings
- **α (alpha)**: Set to 0.001 (not critical, just needs to be slightly > 0)
- **ρa (vigilance A)**: Controls category size, typically 0.85-0.95
- **ρb (vigilance B)**: Automatically set as (ρa + 1) / 2
- **βsbm (second best learning rate)**: Typically 0.2-0.6
- **φ (permanence threshold)**: Typically 3-6
- **τ (cleanup cycle)**: Typically 100-200 cycles

### 2. Key Invariants to Maintain
1. **Categories never shrink**: Weight updates only use minimum operation
2. **Permanent nodes are never removed**: Once counter ≥ φ, node persists
3. **Edges between permanent nodes are stable**: Never deleted
4. **Component B has higher vigilance than A**: ρb > ρa always
5. **Input normalization**: All inputs must be in [0,1] range

### 3. Testing Requirements
Each phase MUST have:
- Unit tests with >90% code coverage
- Integration tests for component interactions
- Property-based tests for invariants
- Performance benchmarks
- Visual validation for 2D test cases

### 4. Error Handling
- Validate input ranges [0,1]
- Handle empty datasets gracefully
- Prevent division by zero in activation function
- Check parameter validity (ρ ∈ (0,1], φ > 0, τ > 0)

## Validation Datasets

### 1. Synthetic Dataset (from paper)
```java
public class SyntheticDataGenerator {
    // Generate 2D Gaussian clusters
    public static List<double[]> generateGaussianClusters(
        int samplesPerCluster, double[][] centers, double[][] covariances
    ) {
        // Implementation using random number generation
    }
    
    // Generate ring-shaped clusters
    public static List<double[]> generateRingClusters(
        int samplesPerRing, double[] center, double innerRadius, 
        double outerRadius
    ) {
        // Implementation
    }
    
    // Generate sinusoidal pattern
    public static List<double[]> generateSinusoidalPattern(
        int samples, double amplitude, double frequency
    ) {
        // Implementation
    }
    
    // Add uniform noise
    public static void addUniformNoise(List<double[]> data, 
                                        double noiseRatio) {
        // Add noise samples
    }
}
```

### 2. Non-stationary Test
```java
@Test
public void testNonStationaryLearning() {
    TopoART network = new TopoART(2, 0.92, 0.6, 6, 100);
    
    // Phase 1: Learn cluster A
    List<double[]> clusterA = generateGaussianClusters(1000, 
        new double[][]{{0.2, 0.2}}, new double[][]{{0.01, 0.01}});
    clusterA.forEach(network::learn);
    
    // Phase 2: Learn cluster B (different location)
    List<double[]> clusterB = generateGaussianClusters(1000,
        new double[][]{{0.8, 0.8}}, new double[][]{{0.01, 0.01}});
    clusterB.forEach(network::learn);
    
    // Verify both clusters are maintained
    List<Cluster> clusters = network.getClusters(false);
    assertEquals(2, clusters.size());
}
```

## Success Criteria

1. **Functional Requirements**
   - ✅ Implements all mathematical operations from the paper
   - ✅ Dual-component architecture working correctly
   - ✅ Topology learning creates meaningful clusters
   - ✅ Noise filtering reduces sensitivity
   - ✅ Hierarchical representation at two detail levels

2. **Performance Requirements**
   - ✅ O(n*m) time complexity (n samples, m neurons)
   - ✅ Handles 100,000+ samples efficiently
   - ✅ Real-time learning capability (<100ms per sample)
   - ✅ Memory efficient (bounded by max neurons)

3. **Quality Requirements**
   - ✅ All tests passing (unit, integration, validation)
   - ✅ >90% code coverage
   - ✅ Clustering metrics match paper results (±5%)
   - ✅ Clean, documented, maintainable code
   - ✅ No compilation warnings

## Development Timeline

- **Phase 1**: 2-3 days - Core data structures
- **Phase 2**: 2 days - Mathematical operations  
- **Phase 3**: 3-4 days - Single component learning
- **Phase 4**: 3-4 days - Dual component system
- **Phase 5**: 2-3 days - Optimization and validation

**Total**: ~15 days for complete, tested implementation

## Additional Resources

- Original paper: "TopoART: A Topology Learning Hierarchical ART Network" (Tscherepanow, 2010)
- Related: Fuzzy ART (Carpenter et al., 1991)
- Related: SOINN (Furao & Hasegawa, 2006)
- Test datasets: Available in paper's experimental section

## Notes for Claude Code

1. **Start with Phase 1** - Get the foundation right before proceeding
2. **Test everything** - Write tests before implementation
3. **Compile frequently** - Ensure each component compiles before moving on
4. **Use version control** - Commit after each successful phase
5. **Document as you go** - Include JavaDoc/comments for all public methods
6. **Benchmark regularly** - Track performance from the beginning
7. **Visualize results** - Use 2D test cases to verify clustering visually

This implementation guide provides all necessary information to build a robust, well-tested TopoART implementation. Follow the phases sequentially, ensure all tests pass at each phase, and maintain the key invariants throughout development.
