package com.hellblazer.art.core.algorithms;

import com.hellblazer.art.core.BaseART;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.parameters.ART1Parameters;
import com.hellblazer.art.core.results.MatchResult;
import com.hellblazer.art.core.weights.ART1Weight;

import java.util.Objects;

/**
 * ART1 Binary Pattern Recognition Algorithm.
 * 
 * Implementation of the classic ART1 algorithm for clustering binary data,
 * based on Carpenter & Grossberg (1987). ART1 is exclusively designed for 
 * binary pattern recognition and uses dual weight vectors (bottom-up and top-down)
 * for activation and matching.
 * 
 * Key features:
 * - Binary data clustering only
 * - Vigilance-based resonance control  
 * - Dual weight system (bottom-up/top-down)
 * - Self-organizing competitive learning
 * 
 * @see BaseART for the template method framework
 * @see ART1Weight for dual weight vector implementation
 * @see ART1Parameters for algorithm parameters (vigilance, L)
 */
public final class ART1 extends BaseART {
    
    /**
     * Create a new ART1 network with no initial categories.
     */
    public ART1() {
        super();
    }
    
    /**
     * Validate that input pattern contains only binary values (0 or 1).
     * @param pattern input pattern to validate
     * @throws IllegalArgumentException if pattern contains non-binary values
     */
    private void validateBinaryPattern(Pattern pattern) {
        for (int i = 0; i < pattern.dimension(); i++) {
            var value = pattern.get(i);
            if (value != 0.0 && value != 1.0) {
                throw new IllegalArgumentException("ART1 requires binary input patterns, found: " + value + " at index " + i);
            }
        }
    }
    
    /**
     * Validate that parameters are of the correct type for ART1.
     * This prophylactic validation prevents runtime errors and ensures type safety.
     * 
     * @param parameters the parameters to validate
     * @return the validated ART1Parameters
     * @throws IllegalArgumentException if parameters are not ART1Parameters
     * @throws NullPointerException if parameters are null
     */
    private ART1Parameters validateParameters(Object parameters) {
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof ART1Parameters)) {
            var actualType = parameters.getClass().getName();
            throw new IllegalArgumentException(
                "ART1 requires ART1Parameters but received: " + actualType + 
                ". Please provide valid ART1Parameters with vigilance and L values."
            );
        }
        
        return (ART1Parameters) parameters;
    }
    
    /**
     * Calculate the activation value for a category using ART1 bottom-up weights.
     * 
     * Choice function: T_j = |I ∧ w_j| / (L + |w_j|)
     * Where:
     * - I is the binary input vector
     * - w_j is the bottom-up weight vector for category j
     * - ∧ is the logical AND operation (element-wise minimum)
     * - |·| is the L1 norm (sum of values)
     * - L is the choice parameter (typically > 0)
     * 
     * @param input the input vector (must be binary)
     * @param weight the category weight vector (must be ART1Weight)
     * @param parameters the algorithm parameters (must be ART1Parameters)
     * @return the activation value for this category
     * @throws IllegalArgumentException if parameters are not ART1Parameters or weight is not ART1Weight
     * @throws NullPointerException if any parameter is null
     */
    @Override
    protected double calculateActivation(Pattern input, WeightVector weight, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(weight, "Weight vector cannot be null");
        
        var art1Params = validateParameters(parameters);
        
        if (!(weight instanceof ART1Weight art1Weight)) {
            throw new IllegalArgumentException("Weight vector must be ART1Weight, got: " + 
                weight.getClass().getSimpleName());
        }
        
        validateBinaryPattern(input);
        
        return art1Weight.computeActivation(input, art1Params);
    }
    
    /**
     * Test whether the input matches the category according to the vigilance criterion.
     * 
     * Vigilance test: |I ∧ w_j| / |I| ≥ ρ
     * Where:
     * - I is the binary input vector
     * - w_j is the top-down weight vector for category j
     * - ∧ is the logical AND operation (element-wise minimum)
     * - |·| is the L1 norm (sum of values)
     * - ρ is the vigilance parameter
     * 
     * @param input the input vector (must be binary)
     * @param weight the category weight vector (must be ART1Weight)
     * @param parameters the algorithm parameters (must be ART1Parameters)
     * @return MatchResult.Accepted if vigilance test passes, MatchResult.Rejected otherwise
     * @throws IllegalArgumentException if parameters are not ART1Parameters or weight is not ART1Weight
     * @throws NullPointerException if any parameter is null
     */
    @Override
    protected MatchResult checkVigilance(Pattern input, WeightVector weight, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(weight, "Weight vector cannot be null");
        
        var art1Params = validateParameters(parameters);
        
        if (!(weight instanceof ART1Weight art1Weight)) {
            throw new IllegalArgumentException("Weight vector must be ART1Weight, got: " + 
                weight.getClass().getSimpleName());
        }
        
        validateBinaryPattern(input);
        
        return art1Weight.checkMatch(input, art1Params);
    }
    
    /**
     * Update the category weight using the ART1 learning rule.
     * 
     * Learning rule: w_j^(new) = I ∧ w_j^(old)
     * Where:
     * - I is the binary input vector
     * - w_j is the category weight vector
     * - ∧ is the logical AND operation (element-wise minimum)
     * 
     * This method delegates to ART1Weight.update() which implements the learning rule
     * for both bottom-up and top-down weights.
     * 
     * @param input the input vector (must be binary)
     * @param currentWeight the current category weight (must be ART1Weight)
     * @param parameters the algorithm parameters (must be ART1Parameters)
     * @return the updated weight vector
     * @throws IllegalArgumentException if parameters are not ART1Parameters or weight is not ART1Weight
     * @throws NullPointerException if any parameter is null
     */
    @Override
    protected WeightVector updateWeights(Pattern input, WeightVector currentWeight, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(currentWeight, "Current weight cannot be null");
        
        validateParameters(parameters); // Validates and throws if wrong type
        
        if (!(currentWeight instanceof ART1Weight)) {
            throw new IllegalArgumentException("Weight vector must be ART1Weight, got: " + 
                currentWeight.getClass().getSimpleName());
        }
        
        validateBinaryPattern(input);
        
        // Delegate to ART1Weight.update() which implements the learning rule
        return currentWeight.update(input, parameters);
    }
    
    /**
     * Create an initial weight vector for a new category based on the input.
     * 
     * For ART1, the initial weight is set to the binary input vector.
     * This follows the principle that a new category should initially represent
     * exactly the input pattern that caused its creation.
     * 
     * @param input the input vector that will become the first example of this category (must be binary)
     * @param parameters the algorithm parameters (unused but required by interface)
     * @return the initial weight vector
     * @throws NullPointerException if input is null
     * @throws IllegalArgumentException if input is not binary
     */
    @Override
    protected WeightVector createInitialWeight(Pattern input, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        validateBinaryPattern(input);
        
        // Validate parameters if provided (may be null for initial weight creation)
        if (parameters != null) {
            validateParameters(parameters);
        }
        
        return ART1Weight.fromInput(input);
    }
    
    /**
     * Get a string representation of this ART1 network.
     * @return string showing the class name and number of categories
     */
    @Override
    public String toString() {
        return String.format("ART1{categories=%d}", getCategoryCount());
    }
}