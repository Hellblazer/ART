package com.hellblazer.art.nlp.config;

import java.util.EnumSet;
import java.util.Set;

/**
 * Configuration for entity channel (FuzzyARTMAP for named entities).
 * 
 * Default vigilance: 0.80 (range: 0.75-0.85) as per TARGET_VISION.md
 */
public class EntityConfig {
    
    public enum EntityType {
        PERSON("en-ner-person.bin"),
        LOCATION("en-ner-location.bin"), 
        ORGANIZATION("en-ner-organization.bin");
        
        private final String modelFileName;
        
        EntityType(String modelFileName) {
            this.modelFileName = modelFileName;
        }
        
        public String getModelFileName() {
            return modelFileName;
        }
    }
    
    public enum FeatureMode {
        COUNT_BASED,
        DENSITY_BASED,
        COMPREHENSIVE
    }
    
    private boolean enabled = true;
    private double vigilance = 0.80;
    private double learningRate = 0.1;
    private Set<EntityType> enabledEntityTypes = EnumSet.allOf(EntityType.class);
    private FeatureMode featureMode = FeatureMode.COUNT_BASED;
    private boolean useNormalization = true;
    private int maxEntitiesPerText = 50;
    
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
    
    public Set<EntityType> getEnabledEntityTypes() { 
        return enabledEntityTypes; 
    }
    
    public void setEnabledEntityTypes(Set<EntityType> enabledEntityTypes) { 
        this.enabledEntityTypes = EnumSet.copyOf(enabledEntityTypes); 
    }
    
    public FeatureMode getFeatureMode() { 
        return featureMode; 
    }
    
    public void setFeatureMode(FeatureMode featureMode) { 
        this.featureMode = featureMode; 
    }
    
    public boolean isUseNormalization() { 
        return useNormalization; 
    }
    
    public void setUseNormalization(boolean useNormalization) { 
        this.useNormalization = useNormalization; 
    }
    
    public int getMaxEntitiesPerText() { 
        return maxEntitiesPerText; 
    }
    
    public void setMaxEntitiesPerText(int maxEntitiesPerText) {
        if (maxEntitiesPerText < 1) {
            throw new IllegalArgumentException("Max entities per text must be positive: " + maxEntitiesPerText);
        }
        this.maxEntitiesPerText = maxEntitiesPerText; 
    }
    
    /**
     * Create default entity configuration.
     * 
     * @return default EntityConfig
     */
    public static EntityConfig defaults() {
        return new EntityConfig();
    }
    
    @Override
    public String toString() {
        return String.format("EntityConfig{enabled=%s, vigilance=%.3f, learningRate=%.3f, entityTypes=%s, featureMode=%s, useNormalization=%s, maxEntities=%d}",
                           enabled, vigilance, learningRate, enabledEntityTypes, featureMode, useNormalization, maxEntitiesPerText);
    }
}