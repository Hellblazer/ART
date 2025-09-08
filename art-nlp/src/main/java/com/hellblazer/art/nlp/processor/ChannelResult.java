package com.hellblazer.art.nlp.processor;

import java.util.Map;
import java.util.Objects;

/**
 * Result of processing text through a single channel.
 * Immutable record containing classification results and metadata.
 */
public record ChannelResult(
    String channelId,
    boolean success,
    int category,
    double confidence,
    long processingTimeMs,
    String errorMessage,
    Map<String, Object> metadata
) {
    
    public ChannelResult {
        Objects.requireNonNull(channelId, "channelId cannot be null");
        
        if (confidence < 0.0 || confidence > 1.0) {
            throw new IllegalArgumentException("Confidence must be in [0.0, 1.0]: " + confidence);
        }
        
        if (processingTimeMs < 0) {
            throw new IllegalArgumentException("Processing time cannot be negative: " + processingTimeMs);
        }
    }
    
    /**
     * Create a successful channel result.
     */
    public static ChannelResult success(String channelId, int category, double confidence, long processingTimeMs) {
        return new ChannelResult(channelId, true, category, confidence, processingTimeMs, null, Map.of());
    }
    
    /**
     * Create a successful channel result with metadata.
     */
    public static ChannelResult success(String channelId, int category, double confidence, 
                                       long processingTimeMs, Map<String, Object> metadata) {
        return new ChannelResult(channelId, true, category, confidence, processingTimeMs, null, 
                                Objects.requireNonNull(metadata, "metadata cannot be null"));
    }
    
    /**
     * Create a failed channel result.
     */
    public static ChannelResult failed(String channelId, String errorMessage) {
        return new ChannelResult(channelId, false, -1, 0.0, 0L, 
                                Objects.requireNonNull(errorMessage, "errorMessage cannot be null"), Map.of());
    }
    
    /**
     * Create a failed channel result with processing time.
     */
    public static ChannelResult failed(String channelId, String errorMessage, long processingTimeMs) {
        return new ChannelResult(channelId, false, -1, 0.0, processingTimeMs, 
                                Objects.requireNonNull(errorMessage, "errorMessage cannot be null"), Map.of());
    }
    
    /**
     * Create a failed channel result with processing time and metadata.
     */
    public static ChannelResult failed(String channelId, String errorMessage, long processingTimeMs, Map<String, Object> metadata) {
        return new ChannelResult(channelId, false, -1, 0.0, processingTimeMs, 
                                Objects.requireNonNull(errorMessage, "errorMessage cannot be null"), 
                                Objects.requireNonNull(metadata, "metadata cannot be null"));
    }
    
    /**
     * Check if this result represents a successful classification.
     */
    public boolean isSuccess() {
        return success;
    }
    
    /**
     * Check if this result represents a failed classification.
     */
    public boolean isFailure() {
        return !success;
    }
    
    /**
     * Get metadata value by key.
     */
    @SuppressWarnings("unchecked")
    public <T> T getMetadata(String key, Class<T> type) {
        Objects.requireNonNull(key, "key cannot be null");
        Objects.requireNonNull(type, "type cannot be null");
        
        var value = metadata.get(key);
        if (value == null) {
            return null;
        }
        
        if (type.isInstance(value)) {
            return (T) value;
        }
        
        throw new ClassCastException("Metadata value for key '" + key + 
                                   "' is not of expected type " + type.getSimpleName());
    }
    
    /**
     * Check if metadata contains a specific key.
     */
    public boolean hasMetadata(String key) {
        return metadata.containsKey(key);
    }
    
    @Override
    public String toString() {
        if (success) {
            return String.format("ChannelResult{channel='%s', category=%d, confidence=%.3f, time=%dms}", 
                               channelId, category, confidence, processingTimeMs);
        } else {
            return String.format("ChannelResult{channel='%s', FAILED: %s, time=%dms}", 
                               channelId, errorMessage, processingTimeMs);
        }
    }
}