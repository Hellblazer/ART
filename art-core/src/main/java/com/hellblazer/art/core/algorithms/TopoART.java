package com.hellblazer.art.core.algorithms;

import com.hellblazer.art.core.BaseART;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.parameters.TopoARTParameters;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.core.results.MatchResult;
import com.hellblazer.art.core.topological.Cluster;
import com.hellblazer.art.core.topological.Neuron;
import com.hellblazer.art.core.topological.TopoARTComponent;
import com.hellblazer.art.core.utils.MathOperations;
import com.hellblazer.art.core.weights.TopoARTWeight;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * TopoART (Topology Learning Hierarchical ART Network) implementation.
 * 
 * TopoART consists of two parallel Fuzzy ART-like components with different vigilance levels:
 * - Component A: Lower vigilance for broader categorization
 * - Component B: Higher vigilance for finer detail categorization
 * 
 * Key features:
 * - Topology learning through neuron connections
 * - Hierarchical representation at two levels of detail
 * - Noise filtering through permanence mechanism
 * - Non-stationary data handling
 * - Compatible with BaseART template for DeepARTMAP integration
 * 
 * Based on: Tscherepanow, M. (2010). "TopoART: A Topology Learning Hierarchical ART Network"
 * 
 * @see BaseART for the template method framework
 * @see TopoARTWeight for topology-aware weight vectors
 * @see TopoARTParameters for algorithm parameters
 */
public final class TopoART extends BaseART {
    
    private final TopoARTComponent componentA;
    private final TopoARTComponent componentB;
    private final TopoARTParameters parameters;
    private int learningCycle = 0;
    
    /**
     * Create a new TopoART network with specified parameters.
     * 
     * @param parameters the network parameters
     * @throws NullPointerException if parameters is null
     */
    public TopoART(TopoARTParameters parameters) {
        super(); // Initialize BaseART with empty categories
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        this.parameters = parameters;
        
        // Create component A with specified vigilance
        this.componentA = new TopoARTComponent(
            parameters.inputDimension(),
            parameters.vigilanceA(),
            parameters.learningRateSecond(),
            parameters.phi(),
            parameters.tau(),
            parameters.alpha()
        );
        
        // Create component B with higher vigilance: ρB = (ρA + 1) / 2
        this.componentB = new TopoARTComponent(
            parameters.inputDimension(),
            parameters.vigilanceB(),
            parameters.learningRateSecond(),
            parameters.phi(),
            parameters.tau(),
            parameters.alpha()
        );
    }
    
    
    /**
     * Get connected component clusters from the specified component.
     * 
     * @param useComponentB if true, extract clusters from component B; otherwise component A
     * @return list of clusters representing connected components
     */
    public List<Cluster> getClusters(boolean useComponentB) {
        var component = useComponentB ? componentB : componentA;
        return extractConnectedComponents(component);
    }
    
    /**
     * Extract connected components using depth-first search.
     * Only considers permanent neurons.
     * 
     * @param component the component to analyze
     * @return list of clusters
     */
    private List<Cluster> extractConnectedComponents(TopoARTComponent component) {
        var clusters = new ArrayList<Cluster>();
        var visited = new HashSet<Integer>();
        
        // Only consider permanent neurons
        for (int i = 0; i < component.getNeuronCount(); i++) {
            var neuron = component.getNeuron(i);
            if (!visited.contains(i) && neuron.isPermanent()) {
                var cluster = new Cluster();
                depthFirstSearch(component, i, visited, cluster);
                if (!cluster.isEmpty()) {
                    clusters.add(cluster);
                }
            }
        }
        
        return clusters;
    }
    
    /**
     * Depth-first search to find connected components.
     * 
     * @param component the component containing neurons
     * @param nodeIndex the current neuron index
     * @param visited set of visited neuron indices
     * @param cluster the current cluster being built
     */
    private void depthFirstSearch(TopoARTComponent component, int nodeIndex, 
                                 Set<Integer> visited, Cluster cluster) {
        visited.add(nodeIndex);
        cluster.addNeuron(nodeIndex);
        
        var neuron = component.getNeuron(nodeIndex);
        for (int neighborIndex : neuron.getEdges()) {
            if (!visited.contains(neighborIndex) && 
                neighborIndex < component.getNeuronCount() &&
                component.getNeuron(neighborIndex).isPermanent()) {
                depthFirstSearch(component, neighborIndex, visited, cluster);
            }
        }
    }
    
    /**
     * Get component A.
     * 
     * @return component A (lower vigilance)
     */
    public TopoARTComponent getComponentA() {
        return componentA;
    }
    
    /**
     * Get component B.
     * 
     * @return component B (higher vigilance)
     */
    public TopoARTComponent getComponentB() {
        return componentB;
    }
    
    /**
     * Get the network parameters.
     * 
     * @return the parameters used to create this network
     */
    public TopoARTParameters getParameters() {
        return parameters;
    }
    
    /**
     * Get the current learning cycle count.
     * 
     * @return the number of patterns presented so far
     */
    public int getLearningCycle() {
        return learningCycle;
    }
    
    /**
     * Reset the learning cycle counter.
     * Useful for testing or restarting learning phases.
     */
    public void resetLearningCycle() {
        learningCycle = 0;
    }
    
    /**
     * Clear both components, removing all learned patterns.
     * Resets the network to its initial state.
     */
    public void clearComponents() {
        componentA.clear();
        componentB.clear();
        learningCycle = 0;
    }
    
    /**
     * Get network statistics.
     * 
     * @return NetworkStats with current network state
     */
    public NetworkStats getStats() {
        return new NetworkStats(
            componentA.getNeuronCount(),
            componentB.getNeuronCount(),
            countPermanentNeurons(componentA),
            countPermanentNeurons(componentB),
            learningCycle
        );
    }
    
    /**
     * Count permanent neurons in a component.
     * 
     * @param component the component to analyze
     * @return number of permanent neurons
     */
    private long countPermanentNeurons(TopoARTComponent component) {
        return component.getNeurons().stream()
            .filter(Neuron::isPermanent)
            .count();
    }
    
    /**
     * Network statistics record.
     * 
     * @param neuronsA number of neurons in component A
     * @param neuronsB number of neurons in component B
     * @param permanentA number of permanent neurons in component A
     * @param permanentB number of permanent neurons in component B
     * @param cycles number of learning cycles completed
     */
    public record NetworkStats(int neuronsA, int neuronsB, 
                              long permanentA, long permanentB, int cycles) {
        @Override
        public String toString() {
            return String.format("NetworkStats{A: %d neurons (%d permanent), B: %d neurons (%d permanent), cycles: %d}",
                               neuronsA, permanentA, neuronsB, permanentB, cycles);
        }
    }
    
    @Override
    public String toString() {
        var stats = getStats();
        return String.format("TopoART{%s, vigilance: A=%.3f B=%.3f}",
                           stats, parameters.vigilanceA(), parameters.vigilanceB());
    }
    
    // ========================================================================
    // BaseART Template Method Implementations
    // ========================================================================
    
    /**
     * Calculate activation for TopoART using component A.
     * 
     * For TopoART, we use component A as the primary component for BaseART integration.
     * The activation is calculated using fuzzy choice function on complement-coded input.
     * 
     * @param input the input pattern
     * @param weight the category weight (must be TopoARTWeight)  
     * @param parameters the algorithm parameters (must be TopoARTParameters)
     * @return the activation value
     */
    @Override
    protected double calculateActivation(Pattern input, WeightVector weight, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(weight, "Weight cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof TopoARTParameters topoParams)) {
            throw new IllegalArgumentException("Parameters must be TopoARTParameters, got: " + 
                parameters.getClass().getSimpleName());
        }
        
        if (!(weight instanceof TopoARTWeight topoWeight)) {
            throw new IllegalArgumentException("Weight must be TopoARTWeight, got: " + 
                weight.getClass().getSimpleName());
        }
        
        // Convert Pattern to double array for TopoART component processing
        var inputArray = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            inputArray[i] = input.get(i);
        }
        
        // Apply complement coding
        var complementInput = MathOperations.complementCode(inputArray);
        var weightArray = topoWeight.toArray();
        
        // Calculate fuzzy choice function: |I ∧ w| / (α + |w|)
        var intersection = MathOperations.componentWiseMin(complementInput, weightArray);
        var intersectionNorm = MathOperations.cityBlockNorm(intersection);
        var weightNorm = MathOperations.cityBlockNorm(weightArray);
        
        return intersectionNorm / (topoParams.alpha() + weightNorm);
    }
    
    /**
     * Check vigilance for TopoART using component A vigilance criterion.
     * 
     * @param input the input pattern
     * @param weight the category weight (must be TopoARTWeight)
     * @param parameters the algorithm parameters (must be TopoARTParameters)
     * @return match result indicating acceptance or rejection
     */
    @Override
    protected MatchResult checkVigilance(Pattern input, WeightVector weight, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(weight, "Weight cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof TopoARTParameters topoParams)) {
            throw new IllegalArgumentException("Parameters must be TopoARTParameters, got: " + 
                parameters.getClass().getSimpleName());
        }
        
        if (!(weight instanceof TopoARTWeight topoWeight)) {
            throw new IllegalArgumentException("Weight must be TopoARTWeight, got: " + 
                weight.getClass().getSimpleName());
        }
        
        // Convert Pattern to double array
        var inputArray = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            inputArray[i] = input.get(i);
        }
        
        // Apply complement coding
        var complementInput = MathOperations.complementCode(inputArray);
        var weightArray = topoWeight.toArray();
        
        // Calculate match function: |I ∧ w| / |I|
        var intersection = MathOperations.componentWiseMin(complementInput, weightArray);
        var intersectionNorm = MathOperations.cityBlockNorm(intersection);
        var inputNorm = MathOperations.cityBlockNorm(complementInput);
        
        double matchValue = (inputNorm > 0) ? intersectionNorm / inputNorm : 0.0;
        
        // Use component A vigilance (primary component for BaseART integration)
        double vigilance = topoWeight.isComponentB() ? topoParams.vigilanceB() : topoParams.vigilanceA();
        
        if (matchValue >= vigilance) {
            return new MatchResult.Accepted(matchValue, vigilance);
        } else {
            return new MatchResult.Rejected(matchValue, vigilance);
        }
    }
    
    /**
     * Update weights using fuzzy min learning rule.
     * 
     * @param input the input pattern
     * @param currentWeight the current weight (must be TopoARTWeight)
     * @param parameters the algorithm parameters (must be TopoARTParameters)
     * @return updated weight vector
     */
    @Override
    protected WeightVector updateWeights(Pattern input, WeightVector currentWeight, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(currentWeight, "Current weight cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof TopoARTParameters topoParams)) {
            throw new IllegalArgumentException("Parameters must be TopoARTParameters, got: " + 
                parameters.getClass().getSimpleName());
        }
        
        if (!(currentWeight instanceof TopoARTWeight topoWeight)) {
            throw new IllegalArgumentException("Weight must be TopoARTWeight, got: " + 
                currentWeight.getClass().getSimpleName());
        }
        
        // Convert Pattern to double array
        var inputArray = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            inputArray[i] = input.get(i);
        }
        
        // Apply complement coding
        var complementInput = MathOperations.complementCode(inputArray);
        var currentWeightArray = topoWeight.toArray();
        
        // Fuzzy min learning: w^new = β(I ∧ w^old) + (1-β)w^old
        var learningRate = topoParams.learningRateSecond();
        var intersection = MathOperations.componentWiseMin(complementInput, currentWeightArray);
        var newWeights = new double[currentWeightArray.length];
        
        for (int i = 0; i < newWeights.length; i++) {
            newWeights[i] = learningRate * intersection[i] + (1.0 - learningRate) * currentWeightArray[i];
        }
        
        // Update counter and permanence
        int newCounter = topoWeight.getCounter() + 1;
        return topoWeight.withUpdatedWeights(newWeights).withUpdatedCounter(newCounter, topoParams.phi());
    }
    
    /**
     * Create initial weight vector from input pattern using complement coding.
     * 
     * @param input the input pattern
     * @param parameters the algorithm parameters (must be TopoARTParameters)
     * @return new TopoARTWeight initialized from input
     */
    @Override
    protected WeightVector createInitialWeight(Pattern input, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        // Parameters can be null for initial weight creation
        
        // Create TopoARTWeight from input (uses component A by default for BaseART integration)
        return new TopoARTWeight(input, false); // false = component A
    }
    
    /**
     * Learn from a single input pattern.
     * 
     * Main TopoART learning algorithm:
     * 1. Process input through component A
     * 2. If resonance and neuron is permanent (counter >= φ), also process through component B
     * 3. Perform periodic cleanup every τ cycles
     * 
     * @param input the input pattern with values in [0, 1]
     * @throws NullPointerException if input is null
     * @throws IllegalArgumentException if input dimension doesn't match or values outside [0, 1]
     */
    public void learn(double[] input) {
        Objects.requireNonNull(input, "Input cannot be null");
        
        if (input.length != parameters.inputDimension()) {
            throw new IllegalArgumentException(
                String.format("Input dimension must be %d, got %d", 
                            parameters.inputDimension(), input.length));
        }
        
        // Validate input range [0, 1]
        MathOperations.validateRange(input, 0.0, 1.0, "Input");
        
        // Process through component A
        var resultA = componentA.learn(input);
        
        // If resonance achieved in A and neuron is permanent, propagate to B
        if (resultA.isSuccessful() && resultA.bestIndex() >= 0) {
            var bestNeuronA = componentA.getNeuron(resultA.bestIndex());
            if (bestNeuronA.getCounter() >= parameters.phi()) {
                // Neuron is permanent - also train component B
                componentB.learn(input);
            }
        }
        
        // Increment learning cycle counter
        learningCycle++;
        
        // Perform periodic cleanup every τ cycles
        if (learningCycle % parameters.tau() == 0) {
            componentA.cleanup();
            componentB.cleanup();
        }
    }
}