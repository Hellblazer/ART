package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;

/**
 * Result of a learning operation in vectorized ART algorithms.
 * Contains information about the learning outcome including the category,
 * weight vector, and success status.
 */
public class LearningResult {
    private final boolean successful;
    private final int categoryIndex;
    private final Pattern input;
    private final WeightVector weight;
    private final double activationValue;
    
    /**
     * Create a successful learning result.
     * 
     * @param categoryIndex the index of the category that learned
     * @param input the input pattern that was learned
     * @param weight the updated weight vector
     * @param activationValue the activation value
     */
    public LearningResult(int categoryIndex, Pattern input, WeightVector weight, double activationValue) {
        this.successful = true;
        this.categoryIndex = categoryIndex;
        this.input = input;
        this.weight = weight;
        this.activationValue = activationValue;
    }
    
    /**
     * Create a failed learning result.
     */
    public LearningResult() {
        this.successful = false;
        this.categoryIndex = -1;
        this.input = null;
        this.weight = null;
        this.activationValue = Double.NaN;
    }
    
    /**
     * Check if the learning was successful.
     * 
     * @return true if learning was successful, false otherwise
     */
    public boolean wasSuccessful() {
        return successful;
    }
    
    /**
     * Get the category index that learned.
     * 
     * @return the category index, or -1 if unsuccessful
     */
    public int getCategoryIndex() {
        return categoryIndex;
    }
    
    /**
     * Get the input pattern that was learned.
     * 
     * @return the input pattern, or null if unsuccessful
     */
    public Pattern getInput() {
        return input;
    }
    
    /**
     * Get the updated weight vector.
     * 
     * @return the weight vector, or null if unsuccessful
     */
    public WeightVector getWeight() {
        return weight;
    }
    
    /**
     * Get the activation value.
     * 
     * @return the activation value, or NaN if unsuccessful
     */
    public double getActivationValue() {
        return activationValue;
    }
    
    /**
     * Factory method for creating a successful learning result.
     */
    public static LearningResult success(int categoryIndex, Pattern input, WeightVector weight, double activationValue) {
        return new LearningResult(categoryIndex, input, weight, activationValue);
    }
    
    /**
     * Factory method for creating a failed learning result.
     */
    public static LearningResult failure() {
        return new LearningResult();
    }
    
    @Override
    public String toString() {
        if (successful) {
            return String.format("LearningResult{successful=true, categoryIndex=%d, activationValue=%.4f}", 
                               categoryIndex, activationValue);
        } else {
            return "LearningResult{successful=false}";
        }
    }
}