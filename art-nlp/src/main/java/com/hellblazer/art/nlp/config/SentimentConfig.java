package com.hellblazer.art.nlp.config;

/**
 * Configuration for sentiment channel (FuzzyART for emotions).
 * 
 * Default vigilance: 0.50 (range: 0.40-0.70) as per TARGET_VISION.md
 */
public class SentimentConfig {
    private boolean enabled = true;
    private double vigilance = 0.50;
    private double learningRate = 0.1;
    private String lexiconPath = "lexicons/sentiment/vader-lexicon.txt";
    private boolean useNegationHandling = true;
    private boolean useIntensityBooster = true;
    private double negationScalar = -0.74;
    private double[] intensityBoosts = {0.293, 0.215, 0.190}; // very, really, quite
    
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
    
    public String getLexiconPath() { 
        return lexiconPath; 
    }
    
    public void setLexiconPath(String lexiconPath) { 
        this.lexiconPath = lexiconPath; 
    }
    
    public boolean isUseNegationHandling() { 
        return useNegationHandling; 
    }
    
    public void setUseNegationHandling(boolean useNegationHandling) { 
        this.useNegationHandling = useNegationHandling; 
    }
    
    public boolean isUseIntensityBooster() { 
        return useIntensityBooster; 
    }
    
    public void setUseIntensityBooster(boolean useIntensityBooster) { 
        this.useIntensityBooster = useIntensityBooster; 
    }
    
    public double getNegationScalar() { 
        return negationScalar; 
    }
    
    public void setNegationScalar(double negationScalar) { 
        this.negationScalar = negationScalar; 
    }
    
    public double[] getIntensityBoosts() { 
        return intensityBoosts.clone(); 
    }
    
    public void setIntensityBoosts(double[] intensityBoosts) { 
        this.intensityBoosts = intensityBoosts.clone(); 
    }
    
    /**
     * Create default sentiment configuration.
     * 
     * @return default SentimentConfig
     */
    public static SentimentConfig defaults() {
        return new SentimentConfig();
    }
    
    @Override
    public String toString() {
        return String.format("SentimentConfig{enabled=%s, vigilance=%.3f, learningRate=%.3f, lexiconPath='%s', negationHandling=%s, intensityBooster=%s}",
                           enabled, vigilance, learningRate, lexiconPath, useNegationHandling, useIntensityBooster);
    }
}