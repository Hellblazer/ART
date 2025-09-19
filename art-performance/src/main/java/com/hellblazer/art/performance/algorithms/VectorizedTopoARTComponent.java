package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.results.TopoARTResult;
import com.hellblazer.art.core.topological.Neuron;
import com.hellblazer.art.core.topological.TopoARTMatchResult;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;

/**
 * Vectorized implementation of a single TopoART component using Java Vector API.
 * Provides SIMD-optimized performance for mathematical operations while maintaining
 * the same interface and behavior as the standard TopoARTComponent.
 * 
 * This implementation leverages Java's Vector API for high-performance parallel
 * computation on supported hardware platforms.
 */
public final class VectorizedTopoARTComponent {
    
    private final List<Neuron> neurons;
    private final int inputDimension;
    private final int complementDimension;
    private final double vigilance;
    private final double learningRateSecond;
    private final int phi;
    private final int tau;
    private final double alpha;
    private int cycleCount;
    
    /**
     * Create a new vectorized TopoART component.
     * 
     * @param inputDimension dimension of original input vectors (before complement coding)
     * @param vigilance vigilance parameter ρ ∈ [0, 1]
     * @param learningRateSecond learning rate for second-best match β ∈ (0, 1]
     * @param phi permanence threshold
     * @param tau cleanup period (cycles between cleanup operations)
     * @param alpha choice parameter α > 0
     * @throws IllegalArgumentException if parameters are invalid
     */
    public VectorizedTopoARTComponent(int inputDimension, double vigilance, double learningRateSecond,
                                    int phi, int tau, double alpha) {
        if (inputDimension <= 0) {
            throw new IllegalArgumentException("Input dimension must be positive");
        }
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in range [0, 1]");
        }
        if (learningRateSecond <= 0.0 || learningRateSecond > 1.0) {
            throw new IllegalArgumentException("Learning rate must be in range (0, 1]");
        }
        if (phi <= 0) {
            throw new IllegalArgumentException("Permanence threshold must be positive");
        }
        if (tau <= 0) {
            throw new IllegalArgumentException("Cleanup period must be positive");
        }
        if (alpha <= 0.0) {
            throw new IllegalArgumentException("Alpha must be positive");
        }
        
        this.inputDimension = inputDimension;
        this.complementDimension = inputDimension * 2;
        this.vigilance = vigilance;
        this.learningRateSecond = learningRateSecond;
        this.phi = phi;
        this.tau = tau;
        this.alpha = alpha;
        this.neurons = new ArrayList<>();
        this.cycleCount = 0;
    }
    
    /**
     * Present an input pattern to this component and perform learning.
     * Uses vectorized operations for optimal performance.
     * 
     * @param input the input pattern (original dimension, will be complement coded)
     * @return learning result indicating resonance and selected neurons
     * @throws NullPointerException if input is null
     * @throws IllegalArgumentException if input has wrong dimension
     */
    public TopoARTResult learn(double[] input) {
        if (input == null) {
            throw new NullPointerException("Input cannot be null");
        }
        if (input.length != inputDimension) {
            throw new IllegalArgumentException("Input dimension mismatch: expected " + 
                                             inputDimension + ", got " + input.length);
        }
        
        // Apply complement coding using vectorized operations
        var complementInput = VectorizedMathOperations.complementCode(input);
        
        // Increment cycle count and perform cleanup if needed
        cycleCount++;
        if (cycleCount % tau == 0) {
            performCleanup();
        }
        
        // Find best and second-best matches using vectorized operations
        var matchResult = findBestMatches(complementInput);
        
        if (matchResult.bestIndex() == -1) {
            // No existing neurons, create first neuron
            return createNewNeuron(complementInput);
        }
        
        var bestNeuron = neurons.get(matchResult.bestIndex());
        
        // Test vigilance using vectorized operations
        boolean matches = VectorizedMathOperations.matchFunction(complementInput, bestNeuron.getWeights(), vigilance);
        if (matches) {
            // Resonance achieved - update weights
            updateNeuronWeights(bestNeuron, matchResult, complementInput);
            return new TopoARTResult(true, matchResult.bestIndex());
        } else {
            // Vigilance test failed - create new neuron
            return createNewNeuron(complementInput);
        }
    }
    
    /**
     * Find the best and second-best matching neurons using vectorized operations.
     * 
     * @param input complement-coded input pattern
     * @return match result with indices and activations
     */
    private TopoARTMatchResult findBestMatches(double[] input) {
        if (neurons.isEmpty()) {
            return new TopoARTMatchResult(-1, -1, 0.0, 0.0);
        }
        
        int bestIndex = -1;
        int secondBestIndex = -1;
        double bestActivation = 0.0;
        double secondBestActivation = 0.0;
        
        // Vectorized computation of all activations
        for (int i = 0; i < neurons.size(); i++) {
            var neuron = neurons.get(i);
            double activation = VectorizedMathOperations.activation(input, neuron.getWeights(), alpha);
            
            if (bestIndex == -1 || activation > bestActivation) {
                // New best match
                secondBestIndex = bestIndex;
                secondBestActivation = bestIndex == -1 ? 0.0 : bestActivation;
                bestIndex = i;
                bestActivation = activation;
            } else if (secondBestIndex == -1 || activation > secondBestActivation) {
                // New second-best match
                secondBestIndex = i;
                secondBestActivation = activation;
            }
        }
        
        return new TopoARTMatchResult(bestIndex, secondBestIndex, bestActivation, secondBestActivation);
    }
    
    /**
     * Create a new neuron with the given pattern using vectorized operations.
     * 
     * @param input complement-coded input pattern
     * @return learning result
     */
    private TopoARTResult createNewNeuron(double[] input) {
        var newNeuron = new Neuron(complementDimension);
        
        // Initialize weights using vectorized fast learning (min operation)
        VectorizedMathOperations.fastLearning(newNeuron.getWeights(), input);
        
        neurons.add(newNeuron);
        int newIndex = neurons.size() - 1;
        
        return new TopoARTResult(true, newIndex);
    }
    
    /**
     * Update neuron weights using vectorized learning rules.
     * 
     * @param bestNeuron the winning neuron
     * @param matchResult match information
     * @param input complement-coded input pattern
     */
    private void updateNeuronWeights(Neuron bestNeuron, TopoARTMatchResult matchResult, double[] input) {
        // Update best match with fast learning (β = 1)
        VectorizedMathOperations.fastLearning(bestNeuron.getWeights(), input);
        bestNeuron.incrementCounter();
        
        // Check permanence using vectorized operations
        if (bestNeuron.getCounter() >= phi && !bestNeuron.isPermanent()) {
            bestNeuron.setPermanent(true);
        }
        
        // Update second-best match with partial learning if it exists
        if (matchResult.secondBestIndex() != -1) {
            var secondBestNeuron = neurons.get(matchResult.secondBestIndex());
            VectorizedMathOperations.partialLearning(secondBestNeuron.getWeights(), input, learningRateSecond);
            secondBestNeuron.incrementCounter();
            
            // Create edge between best and second-best neurons
            bestNeuron.addEdge(matchResult.secondBestIndex());
            secondBestNeuron.addEdge(matchResult.bestIndex());
        }
    }
    
    /**
     * Perform periodic cleanup of non-permanent neurons with low activation.
     * Uses vectorized operations for efficient computation.
     */
    private void performCleanup() {
        var iterator = neurons.iterator();
        int index = 0;
        
        while (iterator.hasNext()) {
            var neuron = iterator.next();
            
            if (!neuron.isPermanent() && neuron.getCounter() == 0) {
                // Remove non-permanent neurons that haven't been activated recently
                iterator.remove();
                
                // Update edge references in remaining neurons
                updateEdgeIndicesAfterRemoval(index);
            } else {
                // Reset counter for next cleanup cycle
                neuron.setCounter(0);
                index++;
            }
        }
    }
    
    /**
     * Update edge indices after a neuron is removed.
     * 
     * @param removedIndex index of the removed neuron
     */
    private void updateEdgeIndicesAfterRemoval(int removedIndex) {
        for (var neuron : neurons) {
            var edges = neuron.getEdges();
            edges.removeIf(edge -> edge == removedIndex);
            
            // Decrement indices that are greater than removed index
            var updatedEdges = edges.stream()
                .mapToInt(edge -> edge > removedIndex ? edge - 1 : edge)
                .boxed()
                .collect(java.util.stream.Collectors.toSet());
            
            edges.clear();
            edges.addAll(updatedEdges);
        }
    }
    
    /**
     * Get all neurons in this component.
     * 
     * @return unmodifiable list of neurons
     */
    public List<Neuron> getNeurons() {
        return List.copyOf(neurons);
    }
    
    /**
     * Get the input dimension (before complement coding).
     * 
     * @return input dimension
     */
    public int getInputDimension() {
        return inputDimension;
    }
    
    /**
     * Get the vigilance parameter.
     * 
     * @return vigilance value
     */
    public double getVigilance() {
        return vigilance;
    }
    
    /**
     * Get the learning rate for second-best matches.
     * 
     * @return learning rate
     */
    public double getLearningRateSecond() {
        return learningRateSecond;
    }
    
    /**
     * Get the permanence threshold.
     * 
     * @return permanence threshold
     */
    public int getPhi() {
        return phi;
    }
    
    /**
     * Get the cleanup period.
     * 
     * @return cleanup period
     */
    public int getTau() {
        return tau;
    }
    
    /**
     * Get the choice parameter alpha.
     * 
     * @return alpha value
     */
    public double getAlpha() {
        return alpha;
    }
    
    /**
     * Get the current cycle count.
     * 
     * @return cycle count
     */
    public int getCycleCount() {
        return cycleCount;
    }
    
    /**
     * Check if vectorized operations are supported on this platform.
     * 
     * @return true if SIMD acceleration is available
     */
    public static boolean isVectorizedSupported() {
        return VectorizedMathOperations.isVectorizedSupported();
    }
    
    /**
     * Get information about the vectorization capabilities.
     * 
     * @return description of vector capabilities
     */
    public static String getVectorInfo() {
        return VectorizedMathOperations.getVectorInfo();
    }
    
    /**
     * Reset the component to initial state.
     * Clears all neurons and resets counters.
     */
    public void reset() {
        neurons.clear();
        cycleCount = 0;
    }
    
    /**
     * Get statistics about this component.
     * 
     * @return formatted statistics string
     */
    public String getStats() {
        long permanentNeurons = neurons.stream().mapToLong(n -> n.isPermanent() ? 1 : 0).sum();
        long totalEdges = neurons.stream().mapToLong(n -> n.getEdges().size()).sum();
        
        return String.format("VectorizedTopoARTComponent: %d neurons (%d permanent), %d edges, cycle %d", 
                           neurons.size(), permanentNeurons, totalEdges, cycleCount);
    }
    
    @Override
    public String toString() {
        return String.format("VectorizedTopoARTComponent{dim=%d, neurons=%d, vigilance=%.3f, cycles=%d}", 
                           inputDimension, neurons.size(), vigilance, cycleCount);
    }
}