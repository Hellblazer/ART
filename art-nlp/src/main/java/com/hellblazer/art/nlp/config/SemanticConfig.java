package com.hellblazer.art.nlp.config;

/**
 * Configuration for semantic channel (FuzzyART for concepts).
 * 
 * Default vigilance: 0.85 (range: 0.70-0.95) as per TARGET_VISION.md
 */
public class SemanticConfig {
    private boolean enabled = true;
    private double vigilance = 0.85;
    private double learningRate = 0.1;
    private String modelPath = "models/cc.en.300.vec.gz";
    private int cacheSize = 10000;
    private int maxCategories = 10000;
    
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
    
    public String getModelPath() { 
        return modelPath; 
    }
    
    public void setModelPath(String modelPath) { 
        this.modelPath = modelPath; 
    }
    
    public int getCacheSize() { 
        return cacheSize; 
    }
    
    public void setCacheSize(int cacheSize) {
        if (cacheSize < 0) {
            throw new IllegalArgumentException("Cache size must be non-negative: " + cacheSize);
        }
        this.cacheSize = cacheSize; 
    }
    
    public int getMaxCategories() { 
        return maxCategories; 
    }
    
    public void setMaxCategories(int maxCategories) {
        if (maxCategories < 1) {
            throw new IllegalArgumentException("Max categories must be positive: " + maxCategories);
        }
        this.maxCategories = maxCategories; 
    }
    
    /**
     * Create default semantic configuration.
     * 
     * @return default SemanticConfig
     */
    public static SemanticConfig defaults() {
        return new SemanticConfig();
    }
    
    @Override
    public String toString() {
        return String.format("SemanticConfig{enabled=%s, vigilance=%.3f, learningRate=%.3f, modelPath='%s', cacheSize=%d, maxCategories=%d}",
                           enabled, vigilance, learningRate, modelPath, cacheSize, maxCategories);
    }
}