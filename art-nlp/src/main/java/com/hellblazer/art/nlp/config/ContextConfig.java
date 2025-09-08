package com.hellblazer.art.nlp.config;

/**
 * Configuration for context channel (TopoART for temporal relations).
 * 
 * Default vigilance: 0.85 (range: 0.80-0.95) as per TARGET_VISION.md
 */
public class ContextConfig {
    private boolean enabled = true;
    private double vigilance = 0.85;
    private double learningRate = 0.1;
    private int windowSize = 200;  // tokens
    private double neighbourhoodRadius = 0.2;
    private boolean extractTemporalFeatures = true;
    private boolean extractSpatialFeatures = true;
    private boolean extractCausalFeatures = true;
    
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
    
    public int getWindowSize() { 
        return windowSize; 
    }
    
    public void setWindowSize(int windowSize) {
        if (windowSize < 1) {
            throw new IllegalArgumentException("Window size must be positive: " + windowSize);
        }
        this.windowSize = windowSize; 
    }
    
    public double getNeighbourhoodRadius() { 
        return neighbourhoodRadius; 
    }
    
    public void setNeighbourhoodRadius(double neighbourhoodRadius) {
        if (neighbourhoodRadius < 0.0 || neighbourhoodRadius > 1.0) {
            throw new IllegalArgumentException("Neighbourhood radius must be in [0.0, 1.0]: " + neighbourhoodRadius);
        }
        this.neighbourhoodRadius = neighbourhoodRadius; 
    }
    
    public boolean isExtractTemporalFeatures() { 
        return extractTemporalFeatures; 
    }
    
    public void setExtractTemporalFeatures(boolean extractTemporalFeatures) { 
        this.extractTemporalFeatures = extractTemporalFeatures; 
    }
    
    public boolean isExtractSpatialFeatures() { 
        return extractSpatialFeatures; 
    }
    
    public void setExtractSpatialFeatures(boolean extractSpatialFeatures) { 
        this.extractSpatialFeatures = extractSpatialFeatures; 
    }
    
    public boolean isExtractCausalFeatures() { 
        return extractCausalFeatures; 
    }
    
    public void setExtractCausalFeatures(boolean extractCausalFeatures) { 
        this.extractCausalFeatures = extractCausalFeatures; 
    }
    
    /**
     * Create default context configuration.
     * 
     * @return default ContextConfig
     */
    public static ContextConfig defaults() {
        return new ContextConfig();
    }
    
    @Override
    public String toString() {
        return String.format("ContextConfig{enabled=%s, vigilance=%.3f, learningRate=%.3f, windowSize=%d, neighbourhoodRadius=%.3f, temporal=%s, spatial=%s, causal=%s}",
                           enabled, vigilance, learningRate, windowSize, neighbourhoodRadius, 
                           extractTemporalFeatures, extractSpatialFeatures, extractCausalFeatures);
    }
}