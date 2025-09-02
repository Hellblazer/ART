package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.Pattern;

/**
 * Result of a prediction operation in vectorized ART algorithms.
 * Contains information about the prediction outcome including the category
 * and success status.
 */
public class PredictionResult {
    private final boolean successful;
    private final int categoryIndex;
    private final Pattern input;
    private final double activationValue;
    private final double confidence;
    
    /**
     * Create a successful prediction result.
     * 
     * @param categoryIndex the index of the predicted category
     * @param input the input pattern that was predicted
     * @param activationValue the activation value
     * @param confidence the confidence of the prediction
     */
    public PredictionResult(int categoryIndex, Pattern input, double activationValue, double confidence) {
        this.successful = true;
        this.categoryIndex = categoryIndex;
        this.input = input;
        this.activationValue = activationValue;
        this.confidence = confidence;
    }
    
    /**
     * Create a failed prediction result.
     */
    public PredictionResult() {
        this.successful = false;
        this.categoryIndex = -1;
        this.input = null;
        this.activationValue = Double.NaN;
        this.confidence = 0.0;
    }
    
    /**
     * Check if the prediction was successful.
     * 
     * @return true if prediction was successful, false otherwise
     */
    public boolean wasSuccessful() {
        return successful;
    }
    
    /**
     * Get the predicted category index.
     * 
     * @return the category index, or -1 if unsuccessful
     */
    public int getCategoryIndex() {
        return categoryIndex;
    }
    
    /**
     * Get the input pattern that was predicted.
     * 
     * @return the input pattern, or null if unsuccessful
     */
    public Pattern getInput() {
        return input;
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
     * Get the prediction confidence.
     * 
     * @return the confidence value, or 0.0 if unsuccessful
     */
    public double getConfidence() {
        return confidence;
    }
    
    /**
     * Factory method for creating a successful prediction result.
     */
    public static PredictionResult success(int categoryIndex, Pattern input, double activationValue, double confidence) {
        return new PredictionResult(categoryIndex, input, activationValue, confidence);
    }
    
    /**
     * Factory method for creating a failed prediction result.
     */
    public static PredictionResult failure() {
        return new PredictionResult();
    }
    
    @Override
    public String toString() {
        if (successful) {
            return String.format("PredictionResult{successful=true, categoryIndex=%d, activationValue=%.4f, confidence=%.4f}", 
                               categoryIndex, activationValue, confidence);
        } else {
            return "PredictionResult{successful=false}";
        }
    }
}