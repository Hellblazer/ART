package com.hellblazer.art.core.topological;

/**
 * Result of finding best and second-best matching neurons in TopoART.
 * Contains indices and activation values for the best and second-best matches.
 * 
 * @param bestIndex the index of the best matching neuron (-1 if no neurons exist)
 * @param secondBestIndex the index of the second-best matching neuron (-1 if < 2 neurons exist)
 * @param bestActivation the activation value of the best matching neuron
 * @param secondBestActivation the activation value of the second-best matching neuron
 */
public record TopoARTMatchResult(int bestIndex, int secondBestIndex, 
                                double bestActivation, double secondBestActivation) {
    
    /**
     * Constructor with validation.
     */
    public TopoARTMatchResult {
        if (bestIndex < -1) {
            throw new IllegalArgumentException("Best index must be >= -1, got: " + bestIndex);
        }
        if (secondBestIndex < -1) {
            throw new IllegalArgumentException("Second best index must be >= -1, got: " + secondBestIndex);
        }
        if (bestActivation < 0.0) {
            throw new IllegalArgumentException("Best activation must be non-negative, got: " + bestActivation);
        }
        if (secondBestActivation < 0.0) {
            throw new IllegalArgumentException("Second best activation must be non-negative, got: " + secondBestActivation);
        }
        if (bestIndex >= 0 && secondBestIndex >= 0 && bestIndex == secondBestIndex) {
            throw new IllegalArgumentException("Best and second best indices cannot be the same: " + bestIndex);
        }
    }
    
    /**
     * Create a TopoARTMatchResult with no matches found.
     * 
     * @return TopoARTMatchResult indicating no neurons available
     */
    public static TopoARTMatchResult noMatch() {
        return new TopoARTMatchResult(-1, -1, 0.0, 0.0);
    }
    
    /**
     * Create a TopoARTMatchResult with only one match found.
     * 
     * @param bestIndex the index of the only matching neuron
     * @param bestActivation the activation value of the only matching neuron
     * @return TopoARTMatchResult with only best match
     */
    public static TopoARTMatchResult singleMatch(int bestIndex, double bestActivation) {
        return new TopoARTMatchResult(bestIndex, -1, bestActivation, 0.0);
    }
    
    /**
     * Check if a best match was found.
     * 
     * @return true if bestIndex >= 0
     */
    public boolean hasBestMatch() {
        return bestIndex >= 0;
    }
    
    /**
     * Check if a second-best match was found.
     * 
     * @return true if secondBestIndex >= 0
     */
    public boolean hasSecondBestMatch() {
        return secondBestIndex >= 0;
    }
    
    /**
     * Check if both best and second-best matches were found.
     * 
     * @return true if both matches are available
     */
    public boolean hasBothMatches() {
        return hasBestMatch() && hasSecondBestMatch();
    }
    
    @Override
    public String toString() {
        if (!hasBestMatch()) {
            return "TopoARTMatchResult{no matches}";
        } else if (!hasSecondBestMatch()) {
            return String.format("TopoARTMatchResult{best=%d(%.3f)}", bestIndex, bestActivation);
        } else {
            return String.format("TopoARTMatchResult{best=%d(%.3f), second=%d(%.3f)}", 
                               bestIndex, bestActivation, secondBestIndex, secondBestActivation);
        }
    }
}