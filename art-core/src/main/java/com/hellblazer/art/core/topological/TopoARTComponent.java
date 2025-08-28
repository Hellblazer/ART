package com.hellblazer.art.core.topological;

import com.hellblazer.art.core.results.TopoARTResult;
import com.hellblazer.art.core.utils.MathOperations;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;

/**
 * A single TopoART component (A or B).
 * Implements the core learning algorithm for one component of the TopoART network.
 */
public final class TopoARTComponent {
    
    private final List<Neuron> neurons;
    private final int inputDimension;
    private final double vigilance;
    private final double learningRateSecond;
    private final int phi;
    private final int tau;
    private final double alpha;
    
    /**
     * Create a new TopoART component.
     * 
     * @param inputDimension the dimension of input vectors (before complement coding)
     * @param vigilance the vigilance parameter (ρ ∈ [0, 1])
     * @param learningRateSecond the learning rate for second-best neurons (β ∈ [0, 1])
     * @param phi the permanence threshold (φ > 0)
     * @param tau the cleanup cycle period (τ > 0)
     * @throws IllegalArgumentException if any parameter is invalid
     */
    public TopoARTComponent(int inputDimension, double vigilance, double learningRateSecond, 
                           int phi, int tau) {
        this(inputDimension, vigilance, learningRateSecond, phi, tau, 0.001);
    }
    
    /**
     * Create a new TopoART component with specified alpha.
     * 
     * @param inputDimension the dimension of input vectors (before complement coding)
     * @param vigilance the vigilance parameter (ρ ∈ [0, 1])
     * @param learningRateSecond the learning rate for second-best neurons (β ∈ [0, 1])
     * @param phi the permanence threshold (φ > 0)
     * @param tau the cleanup cycle period (τ > 0)
     * @param alpha the choice parameter (α ≥ 0)
     * @throws IllegalArgumentException if any parameter is invalid
     */
    public TopoARTComponent(int inputDimension, double vigilance, double learningRateSecond, 
                           int phi, int tau, double alpha) {
        if (inputDimension <= 0) {
            throw new IllegalArgumentException("Input dimension must be positive, got: " + inputDimension);
        }
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in [0, 1], got: " + vigilance);
        }
        if (learningRateSecond < 0.0 || learningRateSecond > 1.0) {
            throw new IllegalArgumentException("Learning rate second must be in [0, 1], got: " + learningRateSecond);
        }
        if (phi <= 0) {
            throw new IllegalArgumentException("Phi must be positive, got: " + phi);
        }
        if (tau <= 0) {
            throw new IllegalArgumentException("Tau must be positive, got: " + tau);
        }
        if (alpha < 0.0) {
            throw new IllegalArgumentException("Alpha must be non-negative, got: " + alpha);
        }
        
        this.inputDimension = inputDimension;
        this.vigilance = vigilance;
        this.learningRateSecond = learningRateSecond;
        this.phi = phi;
        this.tau = tau;
        this.alpha = alpha;
        this.neurons = new ArrayList<>();
    }
    
    /**
     * Find the best and second-best matching neurons for the given input.
     * 
     * @param complementCodedInput the complement-coded input vector
     * @return TopoARTMatchResult with best and second-best matches
     * @throws NullPointerException if input is null
     */
    public TopoARTMatchResult findBestMatches(double[] complementCodedInput) {
        Objects.requireNonNull(complementCodedInput, "Input cannot be null");
        
        if (neurons.isEmpty()) {
            return TopoARTMatchResult.noMatch();
        }
        
        double bestActivation = -1.0;
        double secondBestActivation = -1.0;
        int bestIndex = -1;
        int secondBestIndex = -1;
        
        for (int i = 0; i < neurons.size(); i++) {
            var neuron = neurons.get(i);
            var activation = MathOperations.activation(complementCodedInput, neuron.getWeights(), alpha);
            
            if (activation > bestActivation) {
                // New best - demote current best to second best
                secondBestActivation = bestActivation;
                secondBestIndex = bestIndex;
                bestActivation = activation;
                bestIndex = i;
            } else if (activation > secondBestActivation) {
                // New second best
                secondBestActivation = activation;
                secondBestIndex = i;
            }
        }
        
        if (bestIndex < 0) {
            return TopoARTMatchResult.noMatch();
        } else if (secondBestIndex < 0) {
            return TopoARTMatchResult.singleMatch(bestIndex, bestActivation);
        } else {
            return new TopoARTMatchResult(bestIndex, secondBestIndex, bestActivation, secondBestActivation);
        }
    }
    
    /**
     * Process a single input through this component.
     * 
     * @param input the input vector (will be complement-coded internally)
     * @return TopoARTResult indicating whether resonance occurred
     * @throws NullPointerException if input is null
     * @throws IllegalArgumentException if input dimension doesn't match or values outside [0, 1]
     */
    public TopoARTResult learn(double[] input) {
        Objects.requireNonNull(input, "Input cannot be null");
        
        if (input.length != inputDimension) {
            throw new IllegalArgumentException(
                String.format("Input dimension must be %d, got %d", inputDimension, input.length));
        }
        
        // Validate input range
        MathOperations.validateRange(input, 0.0, 1.0, "Input");
        
        // Apply complement coding
        var complementCoded = MathOperations.complementCode(input);
        
        // Find best matches
        var matches = findBestMatches(complementCoded);
        
        // If no neurons exist, create first neuron
        if (!matches.hasBestMatch()) {
            var newNeuron = new Neuron(complementCoded);
            newNeuron.incrementCounter();
            checkPermanence(newNeuron);
            neurons.add(newNeuron);
            return TopoARTResult.success(0);
        }
        
        // Test vigilance for best match
        var bestNeuron = neurons.get(matches.bestIndex());
        var matchesBest = MathOperations.matchFunction(complementCoded, bestNeuron.getWeights(), vigilance);
        
        if (matchesBest) {
            // Resonance with best match - perform fast learning (β = 1)
            var updatedWeights = MathOperations.fastLearning(complementCoded, bestNeuron.getWeights());
            bestNeuron.setWeights(updatedWeights);
            bestNeuron.incrementCounter();
            checkPermanence(bestNeuron);
            
            // Check second-best match if available
            if (matches.hasSecondBestMatch()) {
                var secondBestNeuron = neurons.get(matches.secondBestIndex());
                
                // Always create bidirectional edges between best and second-best when both exist
                bestNeuron.addEdge(matches.secondBestIndex());
                secondBestNeuron.addEdge(matches.bestIndex());
                
                // Update second-best with partial learning if it also passes vigilance
                var matchesSecond = MathOperations.matchFunction(complementCoded, secondBestNeuron.getWeights(), vigilance);
                if (matchesSecond) {
                    var partialUpdatedWeights = MathOperations.partialLearning(
                        complementCoded, secondBestNeuron.getWeights(), learningRateSecond);
                    secondBestNeuron.setWeights(partialUpdatedWeights);
                }
            }
            
            return TopoARTResult.success(matches.bestIndex());
        } else {
            // No resonance - create new neuron
            var newNeuron = new Neuron(complementCoded);
            newNeuron.incrementCounter();
            checkPermanence(newNeuron);
            neurons.add(newNeuron);
            return TopoARTResult.success(neurons.size() - 1);
        }
    }
    
    /**
     * Check if a neuron should be marked as permanent based on its counter.
     * 
     * @param neuron the neuron to check
     */
    private void checkPermanence(Neuron neuron) {
        if (neuron.getCounter() >= phi) {
            neuron.setPermanent(true);
        }
    }
    
    /**
     * Perform cleanup - remove neurons with counter < phi and set permanence.
     * Called periodically every tau learning cycles.
     */
    public void cleanup() {
        // First, identify which neurons to keep and create index mapping
        var keptNeurons = new ArrayList<Neuron>();
        var oldToNewIndex = new HashMap<Integer, Integer>();
        
        for (int i = 0; i < neurons.size(); i++) {
            var neuron = neurons.get(i);
            if (neuron.getCounter() >= phi) {
                neuron.setPermanent(true);
                oldToNewIndex.put(i, keptNeurons.size());
                keptNeurons.add(neuron);
            }
        }
        
        // Update edge indices for kept neurons
        for (var neuron : keptNeurons) {
            var updatedEdges = new HashSet<Integer>();
            for (int oldEdgeIndex : neuron.getEdges()) {
                if (oldToNewIndex.containsKey(oldEdgeIndex)) {
                    updatedEdges.add(oldToNewIndex.get(oldEdgeIndex));
                }
            }
            neuron.getEdges().clear();
            neuron.getEdges().addAll(updatedEdges);
        }
        
        // Replace the neurons list with kept neurons
        neurons.clear();
        neurons.addAll(keptNeurons);
    }
    
    /**
     * Add a neuron to this component (for testing).
     * 
     * @param weights the weight vector for the neuron
     * @throws NullPointerException if weights is null
     */
    public void addNeuron(double[] weights) {
        Objects.requireNonNull(weights, "Weights cannot be null");
        neurons.add(new Neuron(weights));
    }
    
    /**
     * Add a neuron to this component (for testing).
     * 
     * @param neuron the neuron to add
     * @throws NullPointerException if neuron is null
     */
    public void addNeuron(Neuron neuron) {
        Objects.requireNonNull(neuron, "Neuron cannot be null");
        neurons.add(neuron);
    }
    
    /**
     * Get the number of neurons in this component.
     * 
     * @return the neuron count
     */
    public int getNeuronCount() {
        return neurons.size();
    }
    
    /**
     * Get a neuron by index.
     * 
     * @param index the neuron index
     * @return the neuron at the specified index
     * @throws IndexOutOfBoundsException if index is invalid
     */
    public Neuron getNeuron(int index) {
        return neurons.get(index);
    }
    
    /**
     * Get a copy of all neurons in this component.
     * 
     * @return list of neurons (defensive copy)
     */
    public List<Neuron> getNeurons() {
        return new ArrayList<>(neurons);
    }
    
    /**
     * Get the vigilance parameter.
     * 
     * @return the vigilance value
     */
    public double getVigilance() {
        return vigilance;
    }
    
    /**
     * Get the learning rate for second-best neurons.
     * 
     * @return the learning rate
     */
    public double getLearningRateSecond() {
        return learningRateSecond;
    }
    
    /**
     * Get the permanence threshold.
     * 
     * @return the phi value
     */
    public int getPhi() {
        return phi;
    }
    
    /**
     * Get the cleanup cycle period.
     * 
     * @return the tau value
     */
    public int getTau() {
        return tau;
    }
    
    /**
     * Get the choice parameter.
     * 
     * @return the alpha value
     */
    public double getAlpha() {
        return alpha;
    }
    
    /**
     * Get the input dimension (before complement coding).
     * 
     * @return the input dimension
     */
    public int getInputDimension() {
        return inputDimension;
    }
    
    /**
     * Clear all neurons from this component.
     */
    public void clear() {
        neurons.clear();
    }
    
    @Override
    public String toString() {
        return String.format("TopoARTComponent{neurons=%d, ρ=%.3f, βₛᵦₘ=%.3f, φ=%d, τ=%d}", 
                           neurons.size(), vigilance, learningRateSecond, phi, tau);
    }
}