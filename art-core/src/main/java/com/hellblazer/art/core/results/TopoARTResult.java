package com.hellblazer.art.core.results;

/**
 * Result of a learning operation in TopoART algorithm.
 * Indicates whether resonance was achieved and which neuron was selected.
 * 
 * @param resonance true if the input resonated with a neuron (vigilance test passed)
 * @param bestIndex the index of the neuron that resonated (-1 if no resonance)
 */
public record TopoARTResult(boolean resonance, int bestIndex) {
    
    /**
     * Constructor with validation.
     */
    public TopoARTResult {
        if (resonance && bestIndex < 0) {
            throw new IllegalArgumentException("If resonance is true, bestIndex must be >= 0, got: " + bestIndex);
        }
        if (!resonance && bestIndex >= 0) {
            throw new IllegalArgumentException("If resonance is false, bestIndex must be -1, got: " + bestIndex);
        }
    }
    
    /**
     * Create a TopoARTResult indicating successful resonance.
     * 
     * @param bestIndex the index of the resonating neuron
     * @return TopoARTResult with resonance = true
     * @throws IllegalArgumentException if bestIndex < 0
     */
    public static TopoARTResult success(int bestIndex) {
        if (bestIndex < 0) {
            throw new IllegalArgumentException("Best index must be >= 0 for successful resonance, got: " + bestIndex);
        }
        return new TopoARTResult(true, bestIndex);
    }
    
    /**
     * Create a TopoARTResult indicating no resonance.
     * 
     * @return TopoARTResult with resonance = false and bestIndex = -1
     */
    public static TopoARTResult failure() {
        return new TopoARTResult(false, -1);
    }
    
    /**
     * Check if learning was successful (resonance achieved).
     * 
     * @return true if resonance occurred
     */
    public boolean isSuccessful() {
        return resonance;
    }
    
    /**
     * Check if a valid neuron index is available.
     * 
     * @return true if bestIndex >= 0
     */
    public boolean hasValidIndex() {
        return bestIndex >= 0;
    }
    
    @Override
    public String toString() {
        if (resonance) {
            return String.format("TopoARTResult{resonance=true, neuron=%d}", bestIndex);
        } else {
            return "TopoARTResult{resonance=false}";
        }
    }
}