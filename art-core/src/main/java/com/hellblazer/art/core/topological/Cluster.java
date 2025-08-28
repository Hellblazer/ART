package com.hellblazer.art.core.topological;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Represents a cluster of connected neurons in TopoART topology.
 * Clusters are formed by connected components in the neuron graph.
 */
public final class Cluster {
    
    private final Set<Integer> neuronIndices;
    
    /**
     * Create a new empty cluster.
     */
    public Cluster() {
        this.neuronIndices = new HashSet<>();
    }
    
    /**
     * Create a cluster with initial neuron indices.
     * 
     * @param neuronIndices the initial set of neuron indices
     * @throws NullPointerException if neuronIndices is null
     */
    public Cluster(Set<Integer> neuronIndices) {
        Objects.requireNonNull(neuronIndices, "Neuron indices cannot be null");
        this.neuronIndices = new HashSet<>(neuronIndices);
    }
    
    /**
     * Add a neuron to this cluster.
     * 
     * @param neuronIndex the index of the neuron to add
     * @throws IllegalArgumentException if neuronIndex < 0
     */
    public void addNeuron(int neuronIndex) {
        if (neuronIndex < 0) {
            throw new IllegalArgumentException("Neuron index cannot be negative, got: " + neuronIndex);
        }
        neuronIndices.add(neuronIndex);
    }
    
    /**
     * Remove a neuron from this cluster.
     * 
     * @param neuronIndex the index of the neuron to remove
     * @return true if the neuron was removed, false if it wasn't in the cluster
     */
    public boolean removeNeuron(int neuronIndex) {
        return neuronIndices.remove(neuronIndex);
    }
    
    /**
     * Check if this cluster contains a specific neuron.
     * 
     * @param neuronIndex the index of the neuron to check
     * @return true if the neuron is in this cluster
     */
    public boolean containsNeuron(int neuronIndex) {
        return neuronIndices.contains(neuronIndex);
    }
    
    /**
     * Get a copy of the neuron indices in this cluster.
     * 
     * @return copy of the neuron indices set
     */
    public Set<Integer> getNeuronIndices() {
        return new HashSet<>(neuronIndices);
    }
    
    /**
     * Get the number of neurons in this cluster.
     * 
     * @return the cluster size
     */
    public int size() {
        return neuronIndices.size();
    }
    
    /**
     * Check if this cluster is empty.
     * 
     * @return true if the cluster contains no neurons
     */
    public boolean isEmpty() {
        return neuronIndices.isEmpty();
    }
    
    /**
     * Clear all neurons from this cluster.
     */
    public void clear() {
        neuronIndices.clear();
    }
    
    /**
     * Get the neuron indices as a sorted list.
     * 
     * @return sorted list of neuron indices
     */
    public List<Integer> getSortedNeuronIndices() {
        var sortedList = new ArrayList<>(neuronIndices);
        sortedList.sort(Integer::compareTo);
        return sortedList;
    }
    
    /**
     * Merge another cluster into this cluster.
     * 
     * @param other the cluster to merge
     * @throws NullPointerException if other is null
     */
    public void merge(Cluster other) {
        Objects.requireNonNull(other, "Other cluster cannot be null");
        neuronIndices.addAll(other.neuronIndices);
    }
    
    /**
     * Create a copy of this cluster.
     * 
     * @return a deep copy of this cluster
     */
    public Cluster copy() {
        return new Cluster(neuronIndices);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        var other = (Cluster) obj;
        return Objects.equals(neuronIndices, other.neuronIndices);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(neuronIndices);
    }
    
    @Override
    public String toString() {
        if (isEmpty()) {
            return "Cluster{empty}";
        }
        
        var sortedIndices = getSortedNeuronIndices();
        if (sortedIndices.size() <= 5) {
            return String.format("Cluster{neurons=%s}", sortedIndices);
        } else {
            return String.format("Cluster{size=%d, neurons=[%d, %d, ..., %d, %d]}", 
                               size(), 
                               sortedIndices.get(0), sortedIndices.get(1),
                               sortedIndices.get(sortedIndices.size() - 2), 
                               sortedIndices.get(sortedIndices.size() - 1));
        }
    }
}