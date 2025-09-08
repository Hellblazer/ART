package com.hellblazer.art.nlp.config;

import java.util.Objects;

/**
 * Configuration class for ART-NLP channels using builder pattern.
 * Provides unified configuration for all channel types.
 */
public final class ChannelConfig {
    
    // Core ART parameters
    private final double vigilance;
    private final double learningRate;
    private final int maxCategories;
    
    // Performance parameters
    private final int maxTokensPerInput;
    private final boolean parallelProcessing;
    private final int threadPoolSize;
    
    // Feature processing
    private final boolean enableComplementCoding;
    private final boolean enableNormalization;
    private final double noiseReduction;
    
    // Channel-specific settings
    private final String channelName;
    private final boolean enableLearning;
    private final boolean enablePersistence;
    
    private ChannelConfig(Builder builder) {
        this.vigilance = builder.vigilance;
        this.learningRate = builder.learningRate;
        this.maxCategories = builder.maxCategories;
        this.maxTokensPerInput = builder.maxTokensPerInput;
        this.parallelProcessing = builder.parallelProcessing;
        this.threadPoolSize = builder.threadPoolSize;
        this.enableComplementCoding = builder.enableComplementCoding;
        this.enableNormalization = builder.enableNormalization;
        this.noiseReduction = builder.noiseReduction;
        this.channelName = builder.channelName;
        this.enableLearning = builder.enableLearning;
        this.enablePersistence = builder.enablePersistence;
    }
    
    // Getters
    public double getVigilance() { return vigilance; }
    public double getLearningRate() { return learningRate; }
    public int getMaxCategories() { return maxCategories; }
    public int getMaxTokensPerInput() { return maxTokensPerInput; }
    public boolean isParallelProcessing() { return parallelProcessing; }
    public int getThreadPoolSize() { return threadPoolSize; }
    public boolean isEnableComplementCoding() { return enableComplementCoding; }
    public boolean isEnableNormalization() { return enableNormalization; }
    public double getNoiseReduction() { return noiseReduction; }
    public String getChannelName() { return channelName; }
    public boolean isEnableLearning() { return enableLearning; }
    public boolean isEnablePersistence() { return enablePersistence; }
    
    // Default configurations
    public static ChannelConfig defaultConfig() {
        return new Builder().build();
    }
    
    public static ChannelConfig semanticConfig() {
        return new Builder()
            .vigilance(0.85)
            .learningRate(0.1)
            .maxTokensPerInput(512)
            .channelName("semantic")
            .build();
    }
    
    public static ChannelConfig sentimentConfig() {
        return new Builder()
            .vigilance(0.75)
            .learningRate(0.15)
            .maxTokensPerInput(256)
            .channelName("sentiment")
            .build();
    }
    
    public static ChannelConfig entityConfig() {
        return new Builder()
            .vigilance(0.9)
            .learningRate(0.05)
            .maxTokensPerInput(1024)
            .channelName("entity")
            .build();
    }
    
    public static ChannelConfig contextConfig() {
        return new Builder()
            .vigilance(0.8)
            .learningRate(0.1)
            .maxTokensPerInput(2048)
            .channelName("context")
            .build();
    }
    
    public static ChannelConfig syntacticConfig() {
        return new Builder()
            .vigilance(0.95)
            .learningRate(0.05)
            .maxTokensPerInput(512)
            .channelName("syntactic")
            .build();
    }
    
    public static Builder builder() {
        return new Builder();
    }
    
    public static class Builder {
        private double vigilance = 0.8;
        private double learningRate = 0.1;
        private int maxCategories = 1000;
        private int maxTokensPerInput = 512;
        private boolean parallelProcessing = true;
        private int threadPoolSize = Runtime.getRuntime().availableProcessors();
        private boolean enableComplementCoding = true;
        private boolean enableNormalization = true;
        private double noiseReduction = 0.01;
        private String channelName = "default";
        private boolean enableLearning = true;
        private boolean enablePersistence = true;
        
        public Builder vigilance(double vigilance) {
            if (vigilance < 0.0 || vigilance > 1.0) {
                throw new IllegalArgumentException("Vigilance must be in [0.0, 1.0]: " + vigilance);
            }
            this.vigilance = vigilance;
            return this;
        }
        
        public Builder learningRate(double learningRate) {
            if (learningRate <= 0.0 || learningRate > 1.0) {
                throw new IllegalArgumentException("Learning rate must be in (0.0, 1.0]: " + learningRate);
            }
            this.learningRate = learningRate;
            return this;
        }
        
        public Builder maxCategories(int maxCategories) {
            if (maxCategories <= 0) {
                throw new IllegalArgumentException("Max categories must be positive: " + maxCategories);
            }
            this.maxCategories = maxCategories;
            return this;
        }
        
        public Builder maxTokensPerInput(int maxTokensPerInput) {
            if (maxTokensPerInput <= 0) {
                throw new IllegalArgumentException("Max tokens per input must be positive: " + maxTokensPerInput);
            }
            this.maxTokensPerInput = maxTokensPerInput;
            return this;
        }
        
        public Builder parallelProcessing(boolean parallelProcessing) {
            this.parallelProcessing = parallelProcessing;
            return this;
        }
        
        public Builder threadPoolSize(int threadPoolSize) {
            if (threadPoolSize <= 0) {
                throw new IllegalArgumentException("Thread pool size must be positive: " + threadPoolSize);
            }
            this.threadPoolSize = threadPoolSize;
            return this;
        }
        
        public Builder enableComplementCoding(boolean enableComplementCoding) {
            this.enableComplementCoding = enableComplementCoding;
            return this;
        }
        
        public Builder enableNormalization(boolean enableNormalization) {
            this.enableNormalization = enableNormalization;
            return this;
        }
        
        public Builder noiseReduction(double noiseReduction) {
            if (noiseReduction < 0.0 || noiseReduction >= 1.0) {
                throw new IllegalArgumentException("Noise reduction must be in [0.0, 1.0): " + noiseReduction);
            }
            this.noiseReduction = noiseReduction;
            return this;
        }
        
        public Builder channelName(String channelName) {
            if (channelName == null || channelName.trim().isEmpty()) {
                throw new IllegalArgumentException("Channel name cannot be null or empty");
            }
            this.channelName = channelName.trim();
            return this;
        }
        
        public Builder enableLearning(boolean enableLearning) {
            this.enableLearning = enableLearning;
            return this;
        }
        
        public Builder enablePersistence(boolean enablePersistence) {
            this.enablePersistence = enablePersistence;
            return this;
        }
        
        public ChannelConfig build() {
            return new ChannelConfig(this);
        }
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        var that = (ChannelConfig) o;
        return Double.compare(that.vigilance, vigilance) == 0 &&
               Double.compare(that.learningRate, learningRate) == 0 &&
               maxCategories == that.maxCategories &&
               maxTokensPerInput == that.maxTokensPerInput &&
               parallelProcessing == that.parallelProcessing &&
               threadPoolSize == that.threadPoolSize &&
               enableComplementCoding == that.enableComplementCoding &&
               enableNormalization == that.enableNormalization &&
               Double.compare(that.noiseReduction, noiseReduction) == 0 &&
               enableLearning == that.enableLearning &&
               enablePersistence == that.enablePersistence &&
               Objects.equals(channelName, that.channelName);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(vigilance, learningRate, maxCategories, maxTokensPerInput,
                           parallelProcessing, threadPoolSize, enableComplementCoding,
                           enableNormalization, noiseReduction, channelName,
                           enableLearning, enablePersistence);
    }
    
    @Override
    public String toString() {
        return "ChannelConfig{" +
                "channelName='" + channelName + '\'' +
                ", vigilance=" + vigilance +
                ", learningRate=" + learningRate +
                ", maxCategories=" + maxCategories +
                ", maxTokensPerInput=" + maxTokensPerInput +
                ", parallelProcessing=" + parallelProcessing +
                ", threadPoolSize=" + threadPoolSize +
                ", enableComplementCoding=" + enableComplementCoding +
                ", enableNormalization=" + enableNormalization +
                ", noiseReduction=" + noiseReduction +
                ", enableLearning=" + enableLearning +
                ", enablePersistence=" + enablePersistence +
                '}';
    }
}