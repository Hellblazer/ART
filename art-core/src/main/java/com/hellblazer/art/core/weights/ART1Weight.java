package com.hellblazer.art.core.weights;

import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.Pattern;
import java.util.Arrays;

/**
 * Weight structure for ART1 binary pattern recognition algorithm.
 * 
 * ART1 uses dual weight vectors:
 * - Bottom-up weights: Used for category choice/activation calculation 
 * - Top-down weights: Used for match criterion and learning
 */
public class ART1Weight implements WeightVector {
    
    private final double[] bottomUpWeights;
    private final int[] topDownWeights;
    
    public ART1Weight(double[] bottomUpWeights, int[] topDownWeights) {
        if (bottomUpWeights.length != topDownWeights.length) {
            throw new IllegalArgumentException("Bottom-up and top-down weights must have same dimension");
        }
        this.bottomUpWeights = Arrays.copyOf(bottomUpWeights, bottomUpWeights.length);
        this.topDownWeights = Arrays.copyOf(topDownWeights, topDownWeights.length);
    }
    
    /**
     * Get the bottom-up weights used for category choice activation.
     * @return copy of bottom-up weight vector
     */
    public double[] getBottomUpWeights() {
        return Arrays.copyOf(bottomUpWeights, bottomUpWeights.length);
    }
    
    /**
     * Get the top-down weights used for match criterion.
     * @return copy of top-down weight vector (binary)
     */
    public int[] getTopDownWeights() {
        return Arrays.copyOf(topDownWeights, topDownWeights.length);
    }
    
    /**
     * Get the dimension of the weight vectors.
     * @return dimension
     */
    public int getDimension() {
        return bottomUpWeights.length;
    }
    
    /**
     * Compute dot product of input with bottom-up weights for activation.
     * @param input binary input pattern
     * @return activation value
     */
    public double computeActivation(double[] input) {
        if (input.length != bottomUpWeights.length) {
            throw new IllegalArgumentException("Input dimension mismatch");
        }
        
        var activation = 0.0;
        for (int i = 0; i < input.length; i++) {
            activation += input[i] * bottomUpWeights[i];
        }
        return activation;
    }
    
    /**
     * Compute match criterion between input and top-down weights.
     * @param input binary input pattern
     * @return match ratio (0.0 to 1.0)
     */
    public double computeMatch(double[] input) {
        if (input.length != topDownWeights.length) {
            throw new IllegalArgumentException("Input dimension mismatch");
        }
        
        var matchCount = 0;
        for (int i = 0; i < input.length; i++) {
            // Binary AND operation: input[i] != 0 AND topDownWeights[i] != 0
            if (input[i] != 0.0 && topDownWeights[i] != 0) {
                matchCount++;
            }
        }
        return (double) matchCount / input.length;
    }
    
    /**
     * Create new weight from input pattern for uncommitted node.
     * @param input binary input pattern
     * @param L uncommitted node bias parameter
     * @return new weight
     */
    public static ART1Weight createNew(double[] input, double L) {
        var dimension = input.length;
        
        // Top-down weights equal input (but as binary integers)
        var topDown = new int[dimension];
        for (int i = 0; i < dimension; i++) {
            topDown[i] = input[i] != 0.0 ? 1 : 0;
        }
        
        // Bottom-up weights = L/(L-1+dim) * input
        var scalingFactor = L / (L - 1.0 + dimension);
        var bottomUp = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            bottomUp[i] = scalingFactor * input[i];
        }
        
        return new ART1Weight(bottomUp, topDown);
    }
    
    /**
     * Update weight using ART1 learning rule.
     * @param input binary input pattern
     * @param L uncommitted node bias parameter
     * @return updated weight
     */
    public ART1Weight update(double[] input, double L) {
        var dimension = input.length;
        
        // New top-down weights = bitwise AND of input and old top-down weights
        var newTopDown = new int[dimension];
        var nonzeroCount = 0;
        for (int i = 0; i < dimension; i++) {
            newTopDown[i] = (input[i] != 0.0 && topDownWeights[i] != 0) ? 1 : 0;
            if (newTopDown[i] != 0) {
                nonzeroCount++;
            }
        }
        
        // New bottom-up weights = L/(L-1+count) * new_top_down
        var scalingFactor = L / (L - 1.0 + nonzeroCount);
        var newBottomUp = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            newBottomUp[i] = scalingFactor * newTopDown[i];
        }
        
        return new ART1Weight(newBottomUp, newTopDown);
    }
    
    // WeightVector interface methods
    
    @Override
    public double get(int index) {
        if (index < 0 || index >= bottomUpWeights.length) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for dimension " + bottomUpWeights.length);
        }
        return bottomUpWeights[index];
    }
    
    @Override
    public int dimension() {
        return bottomUpWeights.length;
    }
    
    @Override
    public double l1Norm() {
        var norm = 0.0;
        for (var weight : bottomUpWeights) {
            norm += Math.abs(weight);
        }
        return norm;
    }
    
    /**
     * Compute the activation value for ART1 choice function.
     * 
     * @param input the input pattern
     * @param params the ART1 parameters
     * @return the activation value
     */
    public double computeActivation(Pattern input, com.hellblazer.art.core.parameters.ART1Parameters params) {
        // Convert Pattern to array
        var inputArray = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            inputArray[i] = input.get(i);
        }
        
        // Calculate |I ∧ w| (intersection L1 norm)
        var intersectionNorm = 0.0;
        for (int i = 0; i < inputArray.length; i++) {
            // Binary AND: min(input[i], topDownWeights[i])
            var intersection = Math.min(inputArray[i], topDownWeights[i]);
            intersectionNorm += intersection;
        }
        
        // Calculate |w| (weight L1 norm)
        var weightNorm = 0.0;
        for (var weight : topDownWeights) {
            weightNorm += weight;
        }
        
        // Choice function: T_j = |I ∧ w_j| / (L + |w_j|)
        return intersectionNorm / (params.L() + weightNorm);
    }
    
    /**
     * Check the match criterion against vigilance parameter.
     * 
     * @param input the input pattern
     * @param params the ART1 parameters
     * @return MatchResult indicating acceptance or rejection
     */
    public com.hellblazer.art.core.results.MatchResult checkMatch(Pattern input, com.hellblazer.art.core.parameters.ART1Parameters params) {
        // Convert Pattern to array
        var inputArray = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            inputArray[i] = input.get(i);
        }
        
        // Calculate |I ∧ w| (intersection L1 norm)
        var intersectionNorm = 0.0;
        for (int i = 0; i < inputArray.length; i++) {
            var intersection = Math.min(inputArray[i], topDownWeights[i]);
            intersectionNorm += intersection;
        }
        
        // Calculate |I| (input L1 norm)
        var inputNorm = input.l1Norm();
        
        // Match criterion: |I ∧ w| / |I| >= ρ
        var matchRatio = intersectionNorm / inputNorm;
        
        if (matchRatio >= params.vigilance()) {
            return new com.hellblazer.art.core.results.MatchResult.Accepted(matchRatio, params.vigilance());
        } else {
            return new com.hellblazer.art.core.results.MatchResult.Rejected(matchRatio, params.vigilance());
        }
    }
    
    /**
     * Create an initial weight from an input pattern.
     * 
     * @param input the input pattern
     * @return new ART1Weight
     */
    public static ART1Weight fromInput(Pattern input) {
        var dimension = input.dimension();
        var topDown = new int[dimension];
        var bottomUp = new double[dimension];
        
        // Initialize weights from input pattern
        for (int i = 0; i < dimension; i++) {
            var value = input.get(i);
            topDown[i] = value != 0.0 ? 1 : 0;
            bottomUp[i] = value; // Will be normalized during activation
        }
        
        return new ART1Weight(bottomUp, topDown);
    }
    
    @Override
    public WeightVector update(Pattern input, Object parameters) {
        if (!(parameters instanceof com.hellblazer.art.core.parameters.ART1Parameters params)) {
            throw new IllegalArgumentException("Parameters must be ART1Parameters");
        }
        
        // Convert Pattern to array for learning update
        var inputArray = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            inputArray[i] = input.get(i);
        }
        
        return update(inputArray, params.L());
    }
    
    @Override
    public String toString() {
        return String.format("ART1Weight{bottomUp=%s, topDown=%s}", 
            Arrays.toString(bottomUpWeights), Arrays.toString(topDownWeights));
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof ART1Weight other)) return false;
        return Arrays.equals(bottomUpWeights, other.bottomUpWeights) &&
               Arrays.equals(topDownWeights, other.topDownWeights);
    }
    
    @Override
    public int hashCode() {
        return Arrays.hashCode(bottomUpWeights) * 31 + Arrays.hashCode(topDownWeights);
    }
}