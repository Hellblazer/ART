package com.hellblazer.art.core.topological;

import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

/**
 * Represents a neuron in the TopoART network.
 * Each neuron maintains its weight vector, learning counter, permanence status,
 * and connections (edges) to other neurons for topology learning.
 */
public final class Neuron {
    
    private final double[] weights;
    private final Set<Integer> edges;
    private int counter;
    private boolean isPermanent;
    
    /**
     * Create a new neuron with specified dimension.
     * Weight vector is initialized with ones (standard for ART networks).
     * 
     * @param dimension the dimension of the weight vector
     * @throws IllegalArgumentException if dimension <= 0
     */
    public Neuron(int dimension) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("Dimension must be positive, got: " + dimension);
        }
        
        this.weights = new double[dimension];
        java.util.Arrays.fill(this.weights, 1.0);
        this.edges = new HashSet<>();
        this.counter = 0;
        this.isPermanent = false;
    }
    
    /**
     * Create a new neuron with specified initial weights.
     * 
     * @param initialWeights the initial weight vector (will be copied)
     * @throws NullPointerException if initialWeights is null
     * @throws IllegalArgumentException if initialWeights is empty
     */
    public Neuron(double[] initialWeights) {
        Objects.requireNonNull(initialWeights, "Initial weights cannot be null");
        
        if (initialWeights.length == 0) {
            throw new IllegalArgumentException("Initial weights cannot be empty");
        }
        
        this.weights = initialWeights.clone();
        this.edges = new HashSet<>();
        this.counter = 0;
        this.isPermanent = false;
    }
    
    /**
     * Get a copy of the weight vector.
     * 
     * @return copy of the weight vector
     */
    public double[] getWeights() {
        return weights.clone();
    }
    
    /**
     * Set the weights of this neuron.
     * 
     * @param newWeights the new weight vector (will be copied)
     * @throws NullPointerException if newWeights is null
     * @throws IllegalArgumentException if newWeights has different length than current weights
     */
    public void setWeights(double[] newWeights) {
        Objects.requireNonNull(newWeights, "New weights cannot be null");
        
        if (newWeights.length != weights.length) {
            throw new IllegalArgumentException(
                String.format("Weight vector length must match: expected %d, got %d", 
                            weights.length, newWeights.length));
        }
        
        System.arraycopy(newWeights, 0, weights, 0, weights.length);
    }
    
    /**
     * Get the current learning counter value.
     * The counter tracks how many times this neuron has been the best match.
     * 
     * @return the counter value
     */
    public int getCounter() {
        return counter;
    }
    
    /**
     * Set the learning counter value.
     * 
     * @param counter the new counter value
     * @throws IllegalArgumentException if counter < 0
     */
    public void setCounter(int counter) {
        if (counter < 0) {
            throw new IllegalArgumentException("Counter cannot be negative, got: " + counter);
        }
        this.counter = counter;
    }
    
    /**
     * Increment the learning counter by one.
     * Called each time this neuron is selected as the best match.
     */
    public void incrementCounter() {
        counter++;
    }
    
    /**
     * Check if this neuron is permanent.
     * A neuron becomes permanent when its counter reaches the permanence threshold φ
     * and survives the cleanup process.
     * 
     * @return true if permanent, false if still a candidate
     */
    public boolean isPermanent() {
        return isPermanent;
    }
    
    /**
     * Set the permanence status of this neuron.
     * This is typically called during the cleanup process.
     * 
     * @param permanent the new permanence status
     */
    public void setPermanent(boolean permanent) {
        this.isPermanent = permanent;
    }
    
    /**
     * Get a copy of the edge set.
     * Edges represent connections to other neurons in the topology.
     * 
     * @return copy of the edge set containing neuron indices
     */
    public Set<Integer> getEdges() {
        return new HashSet<>(edges);
    }
    
    /**
     * Add an edge to another neuron.
     * Creates a connection in the topology graph.
     * 
     * @param neuronIndex the index of the neuron to connect to
     * @throws IllegalArgumentException if neuronIndex < 0
     */
    public void addEdge(int neuronIndex) {
        if (neuronIndex < 0) {
            throw new IllegalArgumentException("Neuron index cannot be negative, got: " + neuronIndex);
        }
        edges.add(neuronIndex);
    }
    
    /**
     * Remove an edge to another neuron.
     * 
     * @param neuronIndex the index of the neuron to disconnect from
     * @return true if the edge was removed, false if it didn't exist
     */
    public boolean removeEdge(int neuronIndex) {
        return edges.remove(neuronIndex);
    }
    
    /**
     * Check if this neuron has an edge to another neuron.
     * 
     * @param neuronIndex the index of the neuron to check connection to
     * @return true if connected, false otherwise
     */
    public boolean hasEdgeTo(int neuronIndex) {
        return edges.contains(neuronIndex);
    }
    
    /**
     * Get the number of edges (connections) this neuron has.
     * 
     * @return the number of edges
     */
    public int getEdgeCount() {
        return edges.size();
    }
    
    /**
     * Clear all edges from this neuron.
     * Removes all topology connections.
     */
    public void clearEdges() {
        edges.clear();
    }
    
    /**
     * Get the dimension of this neuron's weight vector.
     * 
     * @return the weight vector dimension
     */
    public int getDimension() {
        return weights.length;
    }
    
    /**
     * Create a deep copy of this neuron.
     * The copy will have the same weights, counter, permanence status, and edges.
     * 
     * @return a deep copy of this neuron
     */
    public Neuron copy() {
        var copy = new Neuron(weights);
        copy.counter = this.counter;
        copy.isPermanent = this.isPermanent;
        copy.edges.addAll(this.edges);
        return copy;
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        var other = (Neuron) obj;
        return counter == other.counter &&
               isPermanent == other.isPermanent &&
               java.util.Arrays.equals(weights, other.weights) &&
               Objects.equals(edges, other.edges);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(java.util.Arrays.hashCode(weights), edges, counter, isPermanent);
    }
    
    @Override
    public String toString() {
        return String.format("Neuron{dim=%d, counter=%d, permanent=%s, edges=%d, weights=[%s]}", 
                           weights.length, counter, isPermanent, edges.size(),
                           formatWeights());
    }
    
    /**
     * Format the weights for display, showing first few elements.
     */
    private String formatWeights() {
        if (weights.length <= 4) {
            var sb = new StringBuilder();
            for (int i = 0; i < weights.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(String.format("%.3f", weights[i]));
            }
            return sb.toString();
        } else {
            return String.format("%.3f, %.3f, ..., %.3f, %.3f", 
                               weights[0], weights[1], 
                               weights[weights.length - 2], weights[weights.length - 1]);
        }
    }
}