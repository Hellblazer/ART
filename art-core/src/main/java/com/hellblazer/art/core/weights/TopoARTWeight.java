package com.hellblazer.art.core.weights;

import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.topological.Neuron;
import java.util.Objects;
import java.util.Arrays;

/**
 * Weight vector implementation for TopoART that encapsulates neuron topology information.
 * 
 * TopoART uses dual-component architecture with topology learning through edge formation.
 * This weight vector represents a neuron in the topological structure with its weights
 * and connection information.
 */
public final class TopoARTWeight implements WeightVector {
    
    private final double[] weights;
    private final int neuronIndex;
    private final boolean componentB;
    private final int counter;
    private final boolean permanent;
    
    /**
     * Create a new TopoARTWeight from a neuron.
     * 
     * @param neuron the neuron containing weights and topology info
     * @param neuronIndex the index of this neuron in its component
     * @param componentB true if this is from component B, false for component A
     */
    public TopoARTWeight(Neuron neuron, int neuronIndex, boolean componentB) {
        Objects.requireNonNull(neuron, "Neuron cannot be null");
        if (neuronIndex < 0) {
            throw new IllegalArgumentException("Neuron index must be non-negative");
        }
        
        this.weights = Arrays.copyOf(neuron.getWeights(), neuron.getWeights().length);
        this.neuronIndex = neuronIndex;
        this.componentB = componentB;
        this.counter = neuron.getCounter();
        this.permanent = neuron.isPermanent();
    }
    
    /**
     * Create a new TopoARTWeight from input pattern (for initial weight creation).
     * 
     * @param input the input pattern
     * @param componentB true if this is for component B
     */
    public TopoARTWeight(Pattern input, boolean componentB) {
        Objects.requireNonNull(input, "Input cannot be null");
        
        // Apply complement coding: [x1, x2, ..., xn] -> [x1, x2, ..., xn, 1-x1, 1-x2, ..., 1-xn]
        this.weights = new double[input.dimension() * 2];
        for (int i = 0; i < input.dimension(); i++) {
            this.weights[i] = input.get(i);
            this.weights[i + input.dimension()] = 1.0 - input.get(i);
        }
        
        this.neuronIndex = -1; // New weight, no index yet
        this.componentB = componentB;
        this.counter = 1;
        this.permanent = false;
    }
    
    /**
     * Create a TopoARTWeight with specific values.
     * 
     * @param weights the weight values (will be copied)
     * @param neuronIndex the neuron index
     * @param componentB true if from component B
     * @param counter the activation counter
     * @param permanent true if neuron is permanent
     */
    public TopoARTWeight(double[] weights, int neuronIndex, boolean componentB, int counter, boolean permanent) {
        Objects.requireNonNull(weights, "Weights cannot be null");
        if (neuronIndex < -1) {
            throw new IllegalArgumentException("Neuron index must be >= -1");
        }
        if (counter < 0) {
            throw new IllegalArgumentException("Counter must be non-negative");
        }
        
        this.weights = Arrays.copyOf(weights, weights.length);
        this.neuronIndex = neuronIndex;
        this.componentB = componentB;
        this.counter = counter;
        this.permanent = permanent;
    }
    
    @Override
    public double get(int index) {
        if (index < 0 || index >= weights.length) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for length " + weights.length);
        }
        return weights[index];
    }
    
    @Override
    public int dimension() {
        return weights.length;
    }
    
    @Override
    public double l1Norm() {
        double sum = 0.0;
        for (double weight : weights) {
            sum += Math.abs(weight);
        }
        return sum;
    }
    
    @Override
    public WeightVector update(Pattern input, Object parameters) {
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof com.hellblazer.art.core.parameters.TopoARTParameters topoParams)) {
            throw new IllegalArgumentException("Parameters must be TopoARTParameters for TopoARTWeight");
        }
        
        // Convert Pattern to double array
        var inputArray = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            inputArray[i] = input.get(i);
        }
        
        // Apply complement coding
        var complementInput = com.hellblazer.art.core.utils.MathOperations.complementCode(inputArray);
        
        // Fuzzy min learning: w^new = β(I ∧ w^old) + (1-β)w^old
        var learningRate = topoParams.learningRateSecond();
        var intersection = com.hellblazer.art.core.utils.MathOperations.componentWiseMin(complementInput, weights);
        var newWeights = new double[weights.length];
        
        for (int i = 0; i < newWeights.length; i++) {
            newWeights[i] = learningRate * intersection[i] + (1.0 - learningRate) * weights[i];
        }
        
        // Update counter and permanence
        int newCounter = counter + 1;
        return new TopoARTWeight(newWeights, neuronIndex, componentB, newCounter, newCounter >= topoParams.phi());
    }
    
    public double[] toArray() {
        return Arrays.copyOf(weights, weights.length);
    }
    
    /**
     * Get the neuron index in the component.
     * @return the neuron index, or -1 if not assigned
     */
    public int getNeuronIndex() {
        return neuronIndex;
    }
    
    /**
     * Check if this weight is from component B.
     * @return true if from component B, false if from component A
     */
    public boolean isComponentB() {
        return componentB;
    }
    
    /**
     * Get the activation counter.
     * @return the number of times this neuron has been activated
     */
    public int getCounter() {
        return counter;
    }
    
    /**
     * Check if this neuron is permanent.
     * @return true if neuron is permanent (counter >= phi)
     */
    public boolean isPermanent() {
        return permanent;
    }
    
    /**
     * Create a new TopoARTWeight with updated counter and permanence.
     * 
     * @param newCounter the new counter value
     * @param phi the permanence threshold
     * @return new TopoARTWeight instance
     */
    public TopoARTWeight withUpdatedCounter(int newCounter, int phi) {
        return new TopoARTWeight(weights, neuronIndex, componentB, newCounter, newCounter >= phi);
    }
    
    /**
     * Create a new TopoARTWeight with updated weights (for learning).
     * 
     * @param newWeights the new weight values
     * @return new TopoARTWeight instance
     */
    public TopoARTWeight withUpdatedWeights(double[] newWeights) {
        return new TopoARTWeight(newWeights, neuronIndex, componentB, counter, permanent);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof TopoARTWeight other)) return false;
        
        return neuronIndex == other.neuronIndex &&
               componentB == other.componentB &&
               counter == other.counter &&
               permanent == other.permanent &&
               Arrays.equals(weights, other.weights);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(Arrays.hashCode(weights), neuronIndex, componentB, counter, permanent);
    }
    
    @Override
    public String toString() {
        return String.format("TopoARTWeight[neuron=%d, component=%s, counter=%d, permanent=%s, dim=%d]",
                neuronIndex, componentB ? "B" : "A", counter, permanent, weights.length);
    }
}