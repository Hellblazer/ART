package com.hellblazer.art.nlp.config;

/**
 * Configuration for syntactic channel (SalienceAwareART for grammar).
 * 
 * Default vigilance: 0.75 (range: 0.70-0.85) as per TARGET_VISION.md
 */
public class SyntacticConfig {
    private boolean enabled = true;
    private double vigilance = 0.75;
    private double learningRate = 0.1;
    private int maxTokens = 100;
    private boolean useNormalization = true;
    private String featureSet = "FULL_SYNTAX";
    
    // Getters and setters
    public boolean isEnabled() { 
        return enabled; 
    }
    
    public void setEnabled(boolean enabled) { 
        this.enabled = enabled; 
    }
    
    public double getVigilance() { 
        return vigilance; 
    }
    
    public void setVigilance(double vigilance) {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in [0.0, 1.0]: " + vigilance);
        }
        this.vigilance = vigilance; 
    }
    
    public double getLearningRate() { 
        return learningRate; 
    }
    
    public void setLearningRate(double learningRate) {
        if (learningRate < 0.0 || learningRate > 1.0) {
            throw new IllegalArgumentException("Learning rate must be in [0.0, 1.0]: " + learningRate);
        }
        this.learningRate = learningRate; 
    }
    
    public int getMaxTokens() { 
        return maxTokens; 
    }
    
    public void setMaxTokens(int maxTokens) {
        if (maxTokens < 1) {
            throw new IllegalArgumentException("Max tokens must be positive: " + maxTokens);
        }
        this.maxTokens = maxTokens; 
    }
    
    public boolean isUseNormalization() { 
        return useNormalization; 
    }
    
    public void setUseNormalization(boolean useNormalization) { 
        this.useNormalization = useNormalization; 
    }
    
    public String getFeatureSet() { 
        return featureSet; 
    }
    
    public void setFeatureSet(String featureSet) { 
        this.featureSet = featureSet; 
    }
    
    /**
     * Create default syntactic configuration.
     * 
     * @return default SyntacticConfig
     */
    public static SyntacticConfig defaults() {
        return new SyntacticConfig();
    }
    
    @Override
    public String toString() {
        return String.format("SyntacticConfig{enabled=%s, vigilance=%.3f, learningRate=%.3f, maxTokens=%d, useNormalization=%s, featureSet='%s'}",
                           enabled, vigilance, learningRate, maxTokens, useNormalization, featureSet);
    }
}