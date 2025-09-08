package com.hellblazer.art.nlp.config;

/**
 * Configuration builder for NLP processor.
 * 
 * Provides comprehensive configuration for all channels and performance settings
 * as specified in TARGET_VISION.md and API_DESIGN.md.
 */
public class NLPConfig {
    
    private final SemanticConfig semantic;
    private final SyntacticConfig syntactic;
    private final ContextConfig context;
    private final EntityConfig entity;
    private final SentimentConfig sentiment;
    private final PerformanceConfig performance;
    
    private NLPConfig(Builder builder) {
        this.semantic = builder.semantic;
        this.syntactic = builder.syntactic;
        this.context = builder.context;
        this.entity = builder.entity;
        this.sentiment = builder.sentiment;
        this.performance = builder.performance;
    }
    
    // Getters
    public SemanticConfig getSemanticConfig() { 
        return semantic; 
    }
    
    public SyntacticConfig getSyntacticConfig() { 
        return syntactic; 
    }
    
    public ContextConfig getContextConfig() { 
        return context; 
    }
    
    public EntityConfig getEntityConfig() { 
        return entity; 
    }
    
    public SentimentConfig getSentimentConfig() { 
        return sentiment; 
    }
    
    public PerformanceConfig getPerformanceConfig() { 
        return performance; 
    }
    
    /**
     * Check if a specific channel is enabled.
     * 
     * @param channelName Channel name (semantic, syntactic, context, entity, sentiment)
     * @return true if channel is enabled
     */
    public boolean isChannelEnabled(String channelName) {
        if (channelName == null || channelName.trim().isEmpty()) {
            return false;
        }
        return switch (channelName) {
            case "semantic" -> semantic.isEnabled();
            case "syntactic" -> syntactic.isEnabled();
            case "context" -> context.isEnabled();
            case "entity" -> entity.isEnabled();
            case "sentiment" -> sentiment.isEnabled();
            default -> false;
        };
    }
    
    /**
     * Builder for NLPConfig with fluent API.
     */
    public static class Builder {
        private SemanticConfig semantic = SemanticConfig.defaults();
        private SyntacticConfig syntactic = SyntacticConfig.defaults();
        private ContextConfig context = ContextConfig.defaults();
        private EntityConfig entity = EntityConfig.defaults();
        private SentimentConfig sentiment = SentimentConfig.defaults();
        private PerformanceConfig performance = PerformanceConfig.defaults();
        
        /**
         * Set semantic channel vigilance parameter.
         * 
         * @param vigilance Vigilance value (0.70-0.95)
         * @return this builder
         */
        public Builder withSemanticVigilance(double vigilance) {
            semantic.setVigilance(vigilance);
            return this;
        }
        
        /**
         * Set syntactic channel vigilance parameter.
         * 
         * @param vigilance Vigilance value (0.70-0.85)
         * @return this builder
         */
        public Builder withSyntacticVigilance(double vigilance) {
            syntactic.setVigilance(vigilance);
            return this;
        }
        
        /**
         * Set context channel vigilance parameter.
         * 
         * @param vigilance Vigilance value (0.80-0.95)
         * @return this builder
         */
        public Builder withContextVigilance(double vigilance) {
            context.setVigilance(vigilance);
            return this;
        }
        
        /**
         * Set entity channel vigilance parameter.
         * 
         * @param vigilance Vigilance value (0.75-0.85)
         * @return this builder
         */
        public Builder withEntityVigilance(double vigilance) {
            entity.setVigilance(vigilance);
            return this;
        }
        
        /**
         * Set sentiment channel vigilance parameter.
         * 
         * @param vigilance Vigilance value (0.40-0.70)
         * @return this builder
         */
        public Builder withSentimentVigilance(double vigilance) {
            sentiment.setVigilance(vigilance);
            return this;
        }
        
        /**
         * Set FastText model path for semantic processing.
         * 
         * @param modelPath Path to FastText model file
         * @return this builder
         */
        public Builder withFastTextModel(String modelPath) {
            semantic.setModelPath(modelPath);
            return this;
        }
        
        /**
         * Set context window size for context channel.
         * 
         * @param tokens Number of tokens in context window
         * @return this builder
         */
        public Builder withContextWindowSize(int tokens) {
            context.setWindowSize(tokens);
            return this;
        }
        
        /**
         * Set thread pool size for parallel processing.
         * 
         * @param threads Number of threads
         * @return this builder
         */
        public Builder withThreadPoolSize(int threads) {
            performance.setThreadPoolSize(threads);
            return this;
        }
        
        /**
         * Enable specific channel.
         * 
         * @param channel Channel name (semantic, syntactic, context, entity, sentiment)
         * @return this builder
         */
        public Builder enableChannel(String channel) {
            switch (channel) {
                case "semantic" -> semantic.setEnabled(true);
                case "syntactic" -> syntactic.setEnabled(true);
                case "context" -> context.setEnabled(true);
                case "entity" -> entity.setEnabled(true);
                case "sentiment" -> sentiment.setEnabled(true);
            }
            return this;
        }
        
        /**
         * Disable specific channel.
         * 
         * @param channel Channel name (semantic, syntactic, context, entity, sentiment)
         * @return this builder
         */
        public Builder disableChannel(String channel) {
            switch (channel) {
                case "semantic" -> semantic.setEnabled(false);
                case "syntactic" -> syntactic.setEnabled(false);
                case "context" -> context.setEnabled(false);
                case "entity" -> entity.setEnabled(false);
                case "sentiment" -> sentiment.setEnabled(false);
            }
            return this;
        }
        
        /**
         * Enable all channels.
         * 
         * @return this builder
         */
        public Builder enableAllChannels() {
            semantic.setEnabled(true);
            syntactic.setEnabled(true);
            context.setEnabled(true);
            entity.setEnabled(true);
            sentiment.setEnabled(true);
            return this;
        }
        
        /**
         * Disable all channels.
         * 
         * @return this builder
         */
        public Builder disableAllChannels() {
            semantic.setEnabled(false);
            syntactic.setEnabled(false);
            context.setEnabled(false);
            entity.setEnabled(false);
            sentiment.setEnabled(false);
            return this;
        }
        
        /**
         * Build immutable NLPConfig.
         * 
         * @return configured NLPConfig instance
         */
        public NLPConfig build() {
            return new NLPConfig(this);
        }
    }
    
    /**
     * Create new builder instance.
     * 
     * @return new NLPConfig.Builder
     */
    public static Builder builder() {
        return new Builder();
    }
}