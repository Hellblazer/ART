package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.parameters.TopoARTParameters;
import com.hellblazer.art.core.results.TopoARTResult;
import com.hellblazer.art.core.topological.Cluster;
import com.hellblazer.art.core.topological.Neuron;
import com.hellblazer.art.performance.VectorizedARTAlgorithm;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Stack;

/**
 * Vectorized implementation of the TopoART (Topology Learning Hierarchical ART Network) algorithm.
 * Uses Java Vector API for SIMD-optimized performance while maintaining identical behavior
 * to the standard TopoART implementation.
 * 
 * Based on Tscherepanow (2010): "TopoART: A Topology Learning Hierarchical ART Network"
 * 
 * Key Features:
 * - Dual-component architecture (A: low vigilance, B: high vigilance)
 * - SIMD-accelerated mathematical operations using Java Vector API
 * - Topology learning through edge connections between winning neurons
 * - Permanence mechanism for stable category formation
 * - Connected component clustering via depth-first search
 * - Periodic cleanup of non-permanent, unused neurons
 * - High-performance vectorized learning rules
 */
public final class VectorizedTopoART implements VectorizedARTAlgorithm<VectorizedPerformanceStats, TopoARTParameters> {
    
    private final VectorizedTopoARTComponent componentA;
    private final VectorizedTopoARTComponent componentB;
    private final TopoARTParameters parameters;
    
    // Performance tracking
    private long totalVectorOperations = 0;
    private long totalTopologicalUpdates = 0;
    private long totalClusterAnalyses = 0;
    private long activationCalls = 0;
    private long matchCalls = 0;
    private long learningCalls = 0;
    
    /**
     * Create a new vectorized TopoART network with the given parameters.
     * 
     * @param parameters the network parameters
     * @throws NullPointerException if parameters is null
     * @throws IllegalArgumentException if parameters are invalid
     */
    public VectorizedTopoART(TopoARTParameters parameters) {
        this.parameters = Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        // Component A: Lower vigilance for coarse categorization
        this.componentA = new VectorizedTopoARTComponent(
            parameters.inputDimension(),
            parameters.vigilanceA(),
            parameters.learningRateSecond(),
            parameters.phi(),
            parameters.tau(),
            parameters.alpha()
        );
        
        // Component B: Higher vigilance for fine categorization
        // Vigilance B is computed as (vigilanceA + 1) / 2 for hierarchical learning
        double vigilanceB = (parameters.vigilanceA() + 1.0) / 2.0;
        this.componentB = new VectorizedTopoARTComponent(
            parameters.inputDimension(),
            vigilanceB,
            parameters.learningRateSecond(),
            parameters.phi(),
            parameters.tau(),
            parameters.alpha()
        );
    }
    
    /**
     * Present an input pattern to the network and perform learning.
     * Both components learn simultaneously using vectorized operations.
     * 
     * @param input the input pattern
     * @return combined learning result from both components
     * @throws NullPointerException if input is null
     * @throws IllegalArgumentException if input has wrong dimension
     */
    public TopoARTResult learn(double[] input) {
        if (input == null) {
            throw new NullPointerException("Input cannot be null");
        }
        if (input.length != parameters.inputDimension()) {
            throw new IllegalArgumentException("Input dimension mismatch: expected " + 
                                             parameters.inputDimension() + ", got " + input.length);
        }
        
        // Both components learn the same input pattern using vectorized operations
        var resultA = componentA.learn(input);
        var resultB = componentB.learn(input);
        
        // Track performance metrics
        totalVectorOperations += 2; // Two component operations
        totalTopologicalUpdates++; // One learning cycle
        
        // Return combined result (typically focusing on component A for primary categorization)
        return new TopoARTResult(
            resultA.resonance() || resultB.resonance(),
            resultA.bestIndex()
        );
    }
    
    /**
     * Present an input Pattern to the network and perform learning.
     * Convenience method that delegates to the double[] version.
     * 
     * @param input the input pattern
     * @return combined learning result from both components
     */
    public TopoARTResult learn(com.hellblazer.art.core.Pattern input) {
        var inputArray = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            inputArray[i] = input.get(i);
        }
        return learn(inputArray);
    }
    
    /**
     * Extract clusters from the network topology using depth-first search.
     * Uses vectorized distance computations for efficient cluster analysis.
     * 
     * @param useComponentB if true, extract clusters from component B; otherwise from component A
     * @return list of connected component clusters
     */
    public List<Cluster> getClusters(boolean useComponentB) {
        var component = useComponentB ? componentB : componentA;
        var neurons = component.getNeurons();
        
        if (neurons.isEmpty()) {
            return new ArrayList<>();
        }
        
        var clusters = new ArrayList<Cluster>();
        var visited = new boolean[neurons.size()];
        
        // Track cluster analysis performance
        totalClusterAnalyses++;
        
        // Find connected components using depth-first search
        for (int i = 0; i < neurons.size(); i++) {
            if (!visited[i]) {
                var cluster = new Cluster();
                exploreCluster(i, neurons, visited, cluster);
                if (!cluster.isEmpty()) {
                    clusters.add(cluster);
                }
            }
        }
        
        return clusters;
    }
    
    /**
     * Recursively explore connected neurons to form a cluster.
     * 
     * @param neuronIndex current neuron index
     * @param neurons list of all neurons
     * @param visited visited flags array
     * @param cluster cluster being built
     */
    private void exploreCluster(int neuronIndex, List<Neuron> neurons, 
                               boolean[] visited, Cluster cluster) {
        var stack = new Stack<Integer>();
        stack.push(neuronIndex);
        
        while (!stack.isEmpty()) {
            int currentIndex = stack.pop();
            
            if (visited[currentIndex]) {
                continue;
            }
            
            visited[currentIndex] = true;
            cluster.addNeuron(currentIndex);
            
            // Add all connected neighbors to the stack
            var currentNeuron = neurons.get(currentIndex);
            for (int neighborIndex : currentNeuron.getEdges()) {
                if (!visited[neighborIndex]) {
                    stack.push(neighborIndex);
                }
            }
        }
    }
    
    /**
     * Get component A (lower vigilance).
     * 
     * @return component A
     */
    public VectorizedTopoARTComponent getComponentA() {
        return componentA;
    }
    
    /**
     * Get component B (higher vigilance).
     * 
     * @return component B
     */
    public VectorizedTopoARTComponent getComponentB() {
        return componentB;
    }
    
    /**
     * Get the network parameters.
     * 
     * @return parameters
     */
    public TopoARTParameters getParameters() {
        return parameters;
    }
    
    /**
     * Reset both components to initial state.
     * Clears all neurons and resets counters.
     */
    public void reset() {
        componentA.reset();
        componentB.reset();
    }
    
    /**
     * Check if vectorized operations are supported on this platform.
     * 
     * @return true if SIMD acceleration is available
     */
    public static boolean isVectorizedSupported() {
        return VectorizedTopoARTComponent.isVectorizedSupported();
    }
    
    /**
     * Get information about the vectorization capabilities.
     * 
     * @return description of vector capabilities
     */
    public static String getVectorInfo() {
        return VectorizedTopoARTComponent.getVectorInfo();
    }
    
    /**
     * Get comprehensive statistics about both components.
     * 
     * @return formatted statistics string
     */
    public String getStats() {
        return String.format("VectorizedTopoART:\n  Component A: %s\n  Component B: %s\n  Vectorization: %s",
                           componentA.getStats(),
                           componentB.getStats(),
                           isVectorizedSupported() ? "Enabled" : "Not Available");
    }
    
    // VectorizedARTAlgorithm interface implementation
    
    @Override
    public Object learn(com.hellblazer.art.core.Pattern input, TopoARTParameters parameters) {
        return learn(input);
    }
    
    @Override
    public Object predict(com.hellblazer.art.core.Pattern input, TopoARTParameters parameters) {
        return learn(input);
    }
    
    @Override
    public int getCategoryCount() {
        return Math.max(componentA.getNeurons().size(), componentB.getNeurons().size());
    }
    
    @Override
    public VectorizedPerformanceStats getPerformanceStats() {
        return new VectorizedPerformanceStats(
            componentA.getNeurons().size() + componentB.getNeurons().size(),
            totalVectorOperations,
            0.0, // Average processing time not tracked
            (int) totalTopologicalUpdates,
            (int) totalClusterAnalyses,
            getCategoryCount(),
            activationCalls,
            matchCalls,
            learningCalls
        );
    }
    
    @Override
    public void resetPerformanceTracking() {
        totalVectorOperations = 0;
        totalTopologicalUpdates = 0;
        totalClusterAnalyses = 0;
        activationCalls = 0;
        matchCalls = 0;
        learningCalls = 0;
    }
    
    // clear() is not required by VectorizedARTAlgorithm interface anymore
    
    @Override
    public void close() {
        // No resources to clean up
    }
    
    // getParameters() is already implemented above, no override needed
    
    @Override
    public int getVectorSpeciesLength() {
        return VectorizedTopoARTComponent.isVectorizedSupported() ? -1 : -1;
    }
    
    @Override
    public String toString() {
        return String.format("VectorizedTopoART{componentA=%d neurons, componentB=%d neurons, vectorized=%s}",
                           componentA.getNeurons().size(),
                           componentB.getNeurons().size(),
                           isVectorizedSupported());
    }
}