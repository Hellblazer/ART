package com.hellblazer.art.core;

import java.util.Objects;

/**
 * FuzzyART implementation using the BaseART template framework.
 * 
 * FuzzyART is a neural network architecture based on Adaptive Resonance Theory (ART)
 * that performs unsupervised learning and pattern recognition using fuzzy set theory.
 * This implementation uses complement coding to handle both binary and analog inputs.
 * 
 * Key Features:
 * - Complement coding: 2D input becomes 4D [x1, x2, 1-x1, 1-x2]
 * - Choice function: T_j = |I ∧ w_j| / (α + |w_j|)
 * - Vigilance test: |I ∧ w_j| / |I| ≥ ρ 
 * - Fuzzy min learning: w_j^(new) = β(I ∧ w_j^(old)) + (1-β)w_j^(old)
 * 
 * @see BaseART for the template method framework
 * @see FuzzyWeight for complement-coded weight vectors
 * @see FuzzyParameters for algorithm parameters (ρ, α, β)
 */
public final class FuzzyART extends BaseART {
    
    /**
     * Create a new FuzzyART network with no initial categories.
     */
    public FuzzyART() {
        super();
    }
    
    /**
     * Calculate the activation value for a category using the FuzzyART choice function.
     * 
     * Choice function: T_j = |I ∧ w_j| / (α + |w_j|)
     * Where:
     * - I is the complement-coded input vector
     * - w_j is the weight vector for category j
     * - ∧ is the fuzzy AND operation (element-wise minimum)
     * - |·| is the L1 norm (sum of absolute values)
     * - α is the choice parameter
     * 
     * @param input the input vector (will be complement-coded internally)
     * @param weight the category weight vector (must be FuzzyWeight)
     * @param parameters the algorithm parameters (must be FuzzyParameters)
     * @return the activation value for this category
     * @throws IllegalArgumentException if parameters are not FuzzyParameters or weight is not FuzzyWeight
     * @throws NullPointerException if any parameter is null
     */
    @Override
    protected double calculateActivation(Vector input, WeightVector weight, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(weight, "Weight vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof FuzzyParameters fuzzyParams)) {
            throw new IllegalArgumentException("Parameters must be FuzzyParameters, got: " + 
                parameters.getClass().getSimpleName());
        }
        
        if (!(weight instanceof FuzzyWeight fuzzyWeight)) {
            throw new IllegalArgumentException("Weight vector must be FuzzyWeight, got: " + 
                weight.getClass().getSimpleName());
        }
        
        // Convert input to complement-coded form to match weight dimensions
        var complementCoded = FuzzyWeight.fromInput(input);
        var inputVector = Vector.of(complementCoded.data());
        var weightVector = Vector.of(fuzzyWeight.data());
        
        // Calculate fuzzy intersection: I ∧ w_j (element-wise minimum)
        var intersection = inputVector.min(weightVector);
        var intersectionNorm = intersection.l1Norm();
        
        // Choice function: T_j = |I ∧ w_j| / (α + |w_j|)
        var denominator = fuzzyParams.alpha() + weightVector.l1Norm();
        
        // Avoid division by zero (shouldn't happen with valid parameters)
        if (denominator == 0.0) {
            throw new IllegalStateException("Division by zero in choice function: α + |w_j| = 0");
        }
        
        return intersectionNorm / denominator;
    }
    
    /**
     * Test whether the input matches the category according to the vigilance criterion.
     * 
     * Vigilance test: |I ∧ w_j| / |I| ≥ ρ
     * Where:
     * - I is the complement-coded input vector
     * - w_j is the weight vector for the category
     * - ∧ is the fuzzy AND operation (element-wise minimum)
     * - |·| is the L1 norm
     * - ρ is the vigilance parameter
     * 
     * @param input the input vector (will be complement-coded internally)
     * @param weight the category weight vector (must be FuzzyWeight)
     * @param parameters the algorithm parameters (must be FuzzyParameters)
     * @return MatchResult.Accepted if vigilance test passes, MatchResult.Rejected otherwise
     * @throws IllegalArgumentException if parameters are not FuzzyParameters or weight is not FuzzyWeight
     * @throws NullPointerException if any parameter is null
     */
    @Override
    protected MatchResult checkVigilance(Vector input, WeightVector weight, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(weight, "Weight vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof FuzzyParameters fuzzyParams)) {
            throw new IllegalArgumentException("Parameters must be FuzzyParameters, got: " + 
                parameters.getClass().getSimpleName());
        }
        
        if (!(weight instanceof FuzzyWeight fuzzyWeight)) {
            throw new IllegalArgumentException("Weight vector must be FuzzyWeight, got: " + 
                weight.getClass().getSimpleName());
        }
        
        // Convert input to complement-coded form to match weight dimensions
        var complementCoded = FuzzyWeight.fromInput(input);
        var inputVector = Vector.of(complementCoded.data());
        var weightVector = Vector.of(fuzzyWeight.data());
        
        // Calculate fuzzy intersection: I ∧ w_j
        var intersection = inputVector.min(weightVector);
        var intersectionNorm = intersection.l1Norm();
        
        // Calculate input norm for the match function
        var inputNorm = inputVector.l1Norm();
        
        // Avoid division by zero
        if (inputNorm == 0.0) {
            throw new IllegalStateException("Division by zero in vigilance test: |I| = 0");
        }
        
        // Match function: |I ∧ w_j| / |I|
        var matchRatio = intersectionNorm / inputNorm;
        
        // Test against vigilance parameter
        if (matchRatio >= fuzzyParams.vigilance()) {
            return new MatchResult.Accepted(matchRatio, fuzzyParams.vigilance());
        } else {
            return new MatchResult.Rejected(matchRatio, fuzzyParams.vigilance());
        }
    }
    
    /**
     * Update the category weight using the FuzzyART learning rule.
     * 
     * Fuzzy min learning rule: w_j^(new) = β(I ∧ w_j^(old)) + (1-β)w_j^(old)
     * Where:
     * - β is the learning rate parameter
     * - I is the complement-coded input vector
     * - w_j is the category weight vector
     * - ∧ is the fuzzy AND operation (element-wise minimum)
     * 
     * This method delegates to FuzzyWeight.update() which implements the learning rule.
     * 
     * @param input the input vector (will be complement-coded internally)
     * @param currentWeight the current category weight (must be FuzzyWeight)
     * @param parameters the algorithm parameters (must be FuzzyParameters)
     * @return the updated weight vector
     * @throws IllegalArgumentException if parameters are not FuzzyParameters or weight is not FuzzyWeight
     * @throws NullPointerException if any parameter is null
     */
    @Override
    protected WeightVector updateWeights(Vector input, WeightVector currentWeight, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(currentWeight, "Current weight cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        if (!(parameters instanceof FuzzyParameters)) {
            throw new IllegalArgumentException("Parameters must be FuzzyParameters, got: " + 
                parameters.getClass().getSimpleName());
        }
        
        if (!(currentWeight instanceof FuzzyWeight)) {
            throw new IllegalArgumentException("Weight vector must be FuzzyWeight, got: " + 
                currentWeight.getClass().getSimpleName());
        }
        
        // Convert input to complement-coded form
        var complementCoded = FuzzyWeight.fromInput(input);
        var inputVector = Vector.of(complementCoded.data());
        
        // Delegate to FuzzyWeight.update() which implements the fuzzy min learning rule
        return currentWeight.update(inputVector, parameters);
    }
    
    /**
     * Create an initial weight vector for a new category based on the input.
     * 
     * For FuzzyART, the initial weight is set to the complement-coded input vector.
     * This follows the principle that a new category should initially represent
     * exactly the input pattern that caused its creation.
     * 
     * Complement coding transforms 2D input [x1, x2] into 4D [x1, x2, 1-x1, 1-x2].
     * 
     * @param input the input vector that will become the first example of this category
     * @param parameters the algorithm parameters (unused but required by interface)
     * @return the initial weight vector with complement coding applied
     * @throws NullPointerException if input is null
     */
    @Override
    protected WeightVector createInitialWeight(Vector input, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        // parameters can be null for initial weight creation
        
        // Create FuzzyWeight with complement coding applied
        return FuzzyWeight.fromInput(input);
    }
    
    /**
     * Get a string representation of this FuzzyART network.
     * @return string showing the class name and number of categories
     */
    @Override
    public String toString() {
        return "FuzzyART{categories=" + getCategoryCount() + "}";
    }
}