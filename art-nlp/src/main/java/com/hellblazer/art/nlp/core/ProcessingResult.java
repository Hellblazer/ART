package com.hellblazer.art.nlp.core;

import java.util.Map;
import java.util.List;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.Objects;
import java.util.Collections;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.nlp.processor.ChannelResult;

/**
 * CANONICAL ProcessingResult - Multi-channel NLP processing result.
 * Thread-safe immutable result container with builder pattern.
 * 
 * Enhanced for multi-channel processing with consensus results and feature fusion.
 */
public final class ProcessingResult {
    // Original text being processed
    private final String text;
    
    // Multi-channel processing results
    private final Map<String, ChannelResult> channelResults;
    
    // Channel name -> category ID mapping (REQUIRED format for compatibility)
    private final Map<String, Integer> channelCategories;
    
    // Consensus classification result
    private final int category;
    private final double confidence;
    
    // Fused feature vector from all channels
    private final DenseVector fusedFeatures;
    
    // Extracted entities (may be empty)
    private final List<Entity> entities;
    
    // Overall sentiment (may be null)
    private final SentimentScore sentiment;
    
    // Processing metrics
    private final long processingTimeMs;
    private final int tokenCount;
    
    // Consensus metadata
    private final Map<String, Object> consensusMetadata;
    
    // Additional metadata
    private final Map<String, Object> metadata;
    
    // Processing status
    private final boolean success;
    private final String errorMessage;

    private ProcessingResult(Builder builder) {
        this.text = builder.text;
        this.channelResults = builder.channelResults != null ? Map.copyOf(builder.channelResults) : Map.of();
        this.channelCategories = Map.copyOf(builder.channelCategories);
        this.category = builder.category;
        this.confidence = builder.confidence;
        this.fusedFeatures = builder.fusedFeatures;
        this.entities = List.copyOf(builder.entities);
        this.sentiment = builder.sentiment;
        this.processingTimeMs = builder.processingTimeMs;
        this.tokenCount = builder.tokenCount;
        this.consensusMetadata = builder.consensusMetadata != null ? Map.copyOf(builder.consensusMetadata) : Map.of();
        this.metadata = Map.copyOf(builder.metadata);
        this.success = builder.success;
        this.errorMessage = builder.errorMessage;
    }
    
    /**
     * Get category ID for specific channel.
     * @param channel Channel name
     * @return Category ID or null if channel not processed
     */
    public Integer getCategoryForChannel(String channel) {
        return channelCategories.get(channel);
    }
    
    /**
     * Get the original text that was processed.
     */
    public String getText() {
        return text;
    }
    
    /**
     * Get results from all channels.
     */
    public Map<String, ChannelResult> getChannelResults() {
        return channelResults;
    }
    
    /**
     * Get result for a specific channel.
     */
    public ChannelResult getChannelResult(String channelId) {
        return channelResults.get(channelId);
    }
    
    /**
     * Get consensus classification category.
     */
    public int getCategory() {
        return category;
    }
    
    /**
     * Get consensus classification confidence.
     */
    public double getConfidence() {
        return confidence;
    }
    
    /**
     * Get fused feature vector from all channels.
     */
    public DenseVector getFusedFeatures() {
        return fusedFeatures;
    }
    
    /**
     * Get consensus metadata.
     */
    public Map<String, Object> getConsensusMetadata() {
        return consensusMetadata;
    }
    
    /**
     * Check if processing was successful.
     */
    public boolean isSuccess() {
        return success;
    }
    
    /**
     * Get error message if processing failed.
     */
    public String getErrorMessage() {
        return errorMessage;
    }
    
    /**
     * Get all channel categories.
     * @return Immutable copy of channel -> category mapping
     */
    public Map<String, Integer> getAllCategories() {
        return channelCategories; // Already immutable from Map.copyOf
    }
    
    /**
     * Get all channel names that were processed.
     */
    public java.util.Set<String> getChannelNames() {
        return channelCategories.keySet();
    }
    
    /**
     * Check if channel was processed and has a category.
     */
    public boolean hasChannel(String channel) {
        return channelCategories.containsKey(channel);
    }
    
    /**
     * Get extracted entities.
     */
    public List<Entity> getEntities() {
        return entities; // Already immutable from List.copyOf
    }
    
    /**
     * Get entities of specific type.
     */
    public List<Entity> getEntitiesByType(String type) {
        return entities.stream()
                      .filter(entity -> entity.getType().equals(type))
                      .toList();
    }
    
    /**
     * Get overall sentiment score.
     */
    public SentimentScore getSentiment() {
        return sentiment;
    }
    
    /**
     * Get processing time in milliseconds.
     */
    public long getProcessingTimeMs() {
        return processingTimeMs;
    }
    
    /**
     * Get token count.
     */
    public int getTokenCount() {
        return tokenCount;
    }
    
    /**
     * Get metadata value.
     */
    public Object getMetadata(String key) {
        return metadata.get(key);
    }
    
    /**
     * Get all metadata.
     */
    public Map<String, Object> getAllMetadata() {
        return metadata; // Already immutable from Map.copyOf
    }
    
    /**
     * Check if processing was degraded (had failures).
     */
    public boolean isDegraded() {
        return Boolean.TRUE.equals(getMetadata("degraded"));
    }
    
    /**
     * Get failed channel if any.
     */
    public String getFailedChannel() {
        return (String) getMetadata("failed_channel");
    }
    
    /**
     * Get error message from metadata if any.
     */
    public String getErrorMessageFromMetadata() {
        return (String) getMetadata("error");
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        ProcessingResult that = (ProcessingResult) obj;
        return processingTimeMs == that.processingTimeMs &&
               tokenCount == that.tokenCount &&
               Objects.equals(channelCategories, that.channelCategories) &&
               Objects.equals(entities, that.entities) &&
               Objects.equals(sentiment, that.sentiment) &&
               Objects.equals(metadata, that.metadata);
    }

    @Override
    public int hashCode() {
        return Objects.hash(channelCategories, entities, sentiment, processingTimeMs, tokenCount, metadata);
    }

    @Override
    public String toString() {
        return String.format("ProcessingResult{channels=%d, categories=%s, entities=%d, sentiment=%s, time=%dms, tokens=%d}",
                           channelCategories.size(), channelCategories, entities.size(), 
                           sentiment != null ? sentiment.getSentiment() : "none", processingTimeMs, tokenCount);
    }

    /**
     * Builder for ProcessingResult with fluent API.
     */
    public static class Builder {
        // Multi-channel fields
        private String text = "";
        private final Map<String, ChannelResult> channelResults = new HashMap<>();
        private final Map<String, Integer> channelCategories = new HashMap<>();
        private int category = -1;
        private double confidence = 0.0;
        private DenseVector fusedFeatures = null;
        private final Map<String, Object> consensusMetadata = new HashMap<>();
        
        // Legacy fields
        private final List<Entity> entities = new ArrayList<>();
        private SentimentScore sentiment = null;
        private long processingTimeMs = 0;
        private int tokenCount = 0;
        private final Map<String, Object> metadata = new HashMap<>();
        private boolean success = true;
        private String errorMessage = null;
        
        /**
         * Set the original text.
         */
        public Builder withText(String text) {
            this.text = Objects.requireNonNull(text, "text cannot be null");
            return this;
        }
        
        /**
         * Set the original text (alias for withText).
         */
        public Builder text(String text) {
            return withText(text);
        }
        
        /**
         * Add channel result.
         */
        public Builder withChannelResult(String channelId, ChannelResult result) {
            channelResults.put(Objects.requireNonNull(channelId, "channelId cannot be null"), 
                              Objects.requireNonNull(result, "result cannot be null"));
            return this;
        }
        
        /**
         * Add multiple channel results.
         */
        public Builder withChannelResults(Map<String, ChannelResult> results) {
            if (results != null) {
                channelResults.putAll(results);
            }
            return this;
        }
        
        /**
         * Add multiple channel results (alias for withChannelResults).
         */
        public Builder channelResults(Map<String, ChannelResult> results) {
            return withChannelResults(results);
        }
        
        /**
         * Add channel category result.
         * @param channel Channel name
         * @param category Category ID
         */
        public Builder withChannelCategory(String channel, int category) {
            channelCategories.put(Objects.requireNonNull(channel, "channel cannot be null"), category);
            return this;
        }
        
        /**
         * Set consensus category.
         */
        public Builder withCategory(int category) {
            this.category = category;
            return this;
        }
        
        /**
         * Set consensus category (alias for withCategory).
         */
        public Builder category(int category) {
            return withCategory(category);
        }
        
        /**
         * Set consensus confidence.
         */
        public Builder withConfidence(double confidence) {
            if (confidence < 0.0 || confidence > 1.0) {
                throw new IllegalArgumentException("Confidence must be in [0.0, 1.0]: " + confidence);
            }
            this.confidence = confidence;
            return this;
        }
        
        /**
         * Set consensus confidence (alias for withConfidence).
         */
        public Builder confidence(double confidence) {
            return withConfidence(confidence);
        }
        
        /**
         * Set fused features.
         */
        public Builder withFusedFeatures(DenseVector features) {
            this.fusedFeatures = features;
            return this;
        }
        
        /**
         * Set fused features (alias for withFusedFeatures).
         */
        public Builder fusedFeatures(DenseVector features) {
            return withFusedFeatures(features);
        }
        
        /**
         * Add consensus metadata.
         */
        public Builder withConsensusMetadata(String key, Object value) {
            consensusMetadata.put(Objects.requireNonNull(key, "key cannot be null"), value);
            return this;
        }
        
        /**
         * Add multiple consensus metadata entries.
         */
        public Builder withConsensusMetadata(Map<String, Object> metadata) {
            if (metadata != null) {
                consensusMetadata.putAll(metadata);
            }
            return this;
        }
        
        /**
         * Add multiple consensus metadata entries (alias).
         */
        public Builder consensusMetadata(Map<String, Object> metadata) {
            return withConsensusMetadata(metadata);
        }
        
        /**
         * Set processing success status.
         */
        public Builder withSuccess(boolean success) {
            this.success = success;
            return this;
        }
        
        /**
         * Set error message.
         */
        public Builder withErrorMessage(String errorMessage) {
            this.errorMessage = errorMessage;
            return this;
        }
        
        /**
         * Add multiple channel categories.
         */
        public Builder withChannelCategories(Map<String, Integer> categories) {
            if (categories != null) {
                channelCategories.putAll(categories);
            }
            return this;
        }
        
        /**
         * Add entity.
         */
        public Builder withEntity(Entity entity) {
            entities.add(Objects.requireNonNull(entity, "entity cannot be null"));
            return this;
        }
        
        /**
         * Add entities.
         */
        public Builder withEntities(List<Entity> entities) {
            if (entities != null) {
                this.entities.addAll(entities);
            }
            return this;
        }
        
        /**
         * Set sentiment score.
         */
        public Builder withSentiment(SentimentScore sentiment) {
            this.sentiment = sentiment;
            return this;
        }
        
        /**
         * Set processing time.
         */
        public Builder withProcessingTime(long ms) {
            if (ms < 0) {
                throw new IllegalArgumentException("Processing time cannot be negative: " + ms);
            }
            this.processingTimeMs = ms;
            return this;
        }
        
        /**
         * Set processing time (alias for withProcessingTime).
         */
        public Builder processingTimeMs(long ms) {
            return withProcessingTime(ms);
        }
        
        /**
         * Set token count.
         */
        public Builder withTokenCount(int count) {
            if (count < 0) {
                throw new IllegalArgumentException("Token count cannot be negative: " + count);
            }
            this.tokenCount = count;
            return this;
        }
        
        /**
         * Add metadata.
         */
        public Builder withMetadata(String key, Object value) {
            metadata.put(Objects.requireNonNull(key, "metadata key cannot be null"), value);
            return this;
        }
        
        /**
         * Add multiple metadata entries.
         */
        public Builder withMetadata(Map<String, Object> metadata) {
            if (metadata != null) {
                this.metadata.putAll(metadata);
            }
            return this;
        }
        
        /**
         * Mark result as degraded due to channel failure.
         */
        public Builder markDegraded(String failedChannel, String errorMessage) {
            return withSuccess(false)
                  .withErrorMessage(errorMessage)
                  .withMetadata("degraded", true)
                  .withMetadata("failed_channel", failedChannel)
                  .withMetadata("error", errorMessage);
        }
        
        /**
         * Build immutable ProcessingResult.
         */
        public ProcessingResult build() {
            return new ProcessingResult(this);
        }
    }
    
    /**
     * Create new builder.
     */
    public static Builder builder() {
        return new Builder();
    }
    
    /**
     * Create empty result.
     */
    public static ProcessingResult empty() {
        return new Builder().build();
    }
    
    /**
     * Create failed result with error information.
     */
    public static ProcessingResult failed(String errorMessage) {
        return new Builder()
            .withSuccess(false)
            .withErrorMessage(errorMessage)
            .build();
    }
    
    /**
     * Create degraded result with error information.
     */
    public static ProcessingResult degraded(String failedChannel, String errorMessage, 
                                          Map<String, Integer> partialResults) {
        var builder = new Builder()
            .withChannelCategories(partialResults != null ? partialResults : Map.of())
            .markDegraded(failedChannel, errorMessage);
        return builder.build();
    }
}