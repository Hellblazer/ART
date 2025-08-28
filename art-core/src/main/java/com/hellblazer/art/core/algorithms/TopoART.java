package com.hellblazer.art.core.algorithms;

import com.hellblazer.art.core.parameters.TopoARTParameters;
import com.hellblazer.art.core.topological.Cluster;
import com.hellblazer.art.core.topological.Neuron;
import com.hellblazer.art.core.topological.TopoARTComponent;
import com.hellblazer.art.core.utils.MathOperations;
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
 * 
 * Based on: Tscherepanow, M. (2010). "TopoART: A Topology Learning Hierarchical ART Network"
 */
public final class TopoART {
    
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
    public void clear() {
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
}