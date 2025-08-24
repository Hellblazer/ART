package com.hellblazer.art.core;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Abstract base class implementing the template method pattern for ART algorithms.
 * Provides the common ART algorithm structure while allowing specific implementations
 * to customize activation, vigilance, and learning behaviors.
 */
public abstract class BaseART {
    
    private final List<WeightVector> categories;
    
    /**
     * Create a new BaseART instance with no initial categories.
     */
    protected BaseART() {
        this.categories = new ArrayList<>();
    }
    
    /**
     * Create a new BaseART instance with initial categories.
     * @param initialCategories the initial categories (will be copied)
     */
    protected BaseART(List<? extends WeightVector> initialCategories) {
        Objects.requireNonNull(initialCategories, "Initial categories cannot be null");
        this.categories = new ArrayList<>(initialCategories);
    }
    
    /**
     * Main template method implementing the ART algorithm.
     * This method orchestrates the complete ART learning cycle:
     * 1. Handle empty category case
     * 2. Calculate activations for all existing categories
     * 3. Find winner through winner-take-all competition
     * 4. Test vigilance criterion
     * 5. Update weights or create new category
     * 
     * @param input the input vector
     * @param parameters the algorithm parameters
     * @return the result of the activation process
     */
    public final ActivationResult stepFit(Pattern input, Object parameters) {
        Objects.requireNonNull(input, "Input vector cannot be null");
        Objects.requireNonNull(parameters, "Parameters cannot be null");
        
        // Step 1: Handle empty categories - create first category
        if (categories.isEmpty()) {
            var newWeight = createInitialWeight(input, parameters);
            categories.add(newWeight);
            return new ActivationResult.Success(0, 1.0, newWeight);
        }
        
        // Step 2: Calculate activations for all categories
        var activations = new double[categories.size()];
        for (int i = 0; i < categories.size(); i++) {
            activations[i] = calculateActivation(input, categories.get(i), parameters);
        }
        
        // Step 3: Find winner through competition (highest activation)
        var winnerResult = findWinner(activations);
        var winnerIndex = winnerResult.winnerIndex();
        var winnerWeight = categories.get(winnerIndex);
        
        // Step 4: Test vigilance criterion
        var matchResult = checkVigilance(input, winnerWeight, parameters);
        
        // Step 5: Update or create based on vigilance test
        if (matchResult.isAccepted()) {
            // Learn: update the winning category
            var updatedWeight = updateWeights(input, winnerWeight, parameters);
            categories.set(winnerIndex, updatedWeight);
            return new ActivationResult.Success(winnerIndex, winnerResult.winnerActivation(), updatedWeight);
        } else {
            // Vigilance failed: create new category
            var newWeight = createInitialWeight(input, parameters);
            categories.add(newWeight);
            var newIndex = categories.size() - 1;
            return new ActivationResult.Success(newIndex, 1.0, newWeight);
        }
    }
    
    /**
     * Calculate the activation value for a specific category given an input.
     * This is algorithm-specific (e.g., choice function in FuzzyART).
     * 
     * @param input the input vector
     * @param weight the category weight vector
     * @param parameters the algorithm parameters
     * @return the activation value for this category
     */
    protected abstract double calculateActivation(Pattern input, WeightVector weight, Object parameters);
    
    /**
     * Test whether the input matches the category well enough according to vigilance.
     * This is algorithm-specific (e.g., vigilance test in FuzzyART).
     * 
     * @param input the input vector
     * @param weight the category weight vector
     * @param parameters the algorithm parameters
     * @return the match result (accepted or rejected)
     */
    protected abstract MatchResult checkVigilance(Pattern input, WeightVector weight, Object parameters);
    
    /**
     * Update the category weight based on the input using the learning rule.
     * This is algorithm-specific (e.g., fuzzy min learning in FuzzyART).
     * 
     * @param input the input vector
     * @param currentWeight the current category weight
     * @param parameters the algorithm parameters
     * @return the updated weight vector
     */
    protected abstract WeightVector updateWeights(Pattern input, WeightVector currentWeight, Object parameters);
    
    /**
     * Create an initial weight vector for a new category based on the input.
     * This is algorithm-specific (e.g., complement coding initialization in FuzzyART).
     * 
     * @param input the input vector that will become the first example of this category
     * @param parameters the algorithm parameters
     * @return the initial weight vector for the new category
     */
    protected abstract WeightVector createInitialWeight(Pattern input, Object parameters);
    
    /**
     * Find the winner among categories using winner-take-all competition.
     * Default implementation selects the category with highest activation.
     * 
     * @param activations the activation values for all categories
     * @return the winner result with index and activation value
     */
    protected CategoryResult findWinner(double[] activations) {
        if (activations.length == 0) {
            throw new IllegalArgumentException("Cannot find winner with no categories");
        }
        
        int winnerIndex = 0;
        double maxActivation = activations[0];
        
        for (int i = 1; i < activations.length; i++) {
            if (activations[i] > maxActivation) {
                maxActivation = activations[i];
                winnerIndex = i;
            }
        }
        
        return CategoryResult.of(winnerIndex, categories, activations);
    }
    
    /**
     * Get an unmodifiable view of the categories.
     * @return unmodifiable list of category weight vectors
     */
    public final List<WeightVector> getCategories() {
        return Collections.unmodifiableList(categories);
    }
    
    /**
     * Get the number of categories.
     * @return the number of categories
     */
    public final int getCategoryCount() {
        return categories.size();
    }
    
    /**
     * Get a specific category weight by index.
     * @param index the category index
     * @return the weight vector for that category
     * @throws IndexOutOfBoundsException if index is invalid
     */
    public final WeightVector getCategory(int index) {
        if (index < 0 || index >= categories.size()) {
            throw new IndexOutOfBoundsException("Category index " + index + 
                " out of bounds for " + categories.size() + " categories");
        }
        return categories.get(index);
    }
    
    /**
     * Clear all categories (reset the network).
     */
    public final void clear() {
        categories.clear();
    }
    
    /**
     * Replace all categories with a new list (for subclass use).
     * @param newCategories the new categories to replace with
     */
    protected final void replaceAllCategories(List<WeightVector> newCategories) {
        Objects.requireNonNull(newCategories, "New categories cannot be null");
        categories.clear();
        categories.addAll(newCategories);
    }
    
    /**
     * Get a string representation showing the number of categories.
     * @return string representation
     */
    @Override
    public String toString() {
        return getClass().getSimpleName() + "{categories=" + categories.size() + "}";
    }
}