package com.hellblazer.art.nlp.processor;

import java.util.Map;
import java.util.Objects;

/**
 * Result of consensus decision making across multiple channels.
 * Contains the final classification decision and supporting metadata.
 */
public record ConsensusResult(
    int category,
    double confidence,
    String strategy,
    Map<String, Double> channelContributions,
    Map<String, Object> metadata
) {
    
    public ConsensusResult {
        Objects.requireNonNull(strategy, "strategy cannot be null");
        Objects.requireNonNull(channelContributions, "channelContributions cannot be null");
        Objects.requireNonNull(metadata, "metadata cannot be null");
        
        if (confidence < 0.0 || confidence > 1.0) {
            throw new IllegalArgumentException("Confidence must be in [0.0, 1.0]: " + confidence);
        }
        
        // Validate channel contributions sum to reasonable range
        var totalContribution = channelContributions.values().stream()
            .mapToDouble(Double::doubleValue)
            .sum();
        
        if (totalContribution < 0.0 || totalContribution > 1.1) { // Allow small floating point errors
            throw new IllegalArgumentException("Channel contributions should sum to ~1.0, got: " + totalContribution);
        }
    }
    
    /**
     * Create a consensus result.
     */
    public static ConsensusResult create(int category, double confidence, String strategy,
                                        Map<String, Double> channelContributions) {
        return new ConsensusResult(category, confidence, strategy, channelContributions, Map.of());
    }
    
    /**
     * Create a consensus result with metadata.
     */
    public static ConsensusResult create(int category, double confidence, String strategy,
                                        Map<String, Double> channelContributions,
                                        Map<String, Object> metadata) {
        return new ConsensusResult(category, confidence, strategy, channelContributions, metadata);
    }
    
    /**
     * Get contribution for a specific channel.
     */
    public double getChannelContribution(String channelId) {
        return channelContributions.getOrDefault(channelId, 0.0);
    }
    
    /**
     * Get the channel with highest contribution.
     */
    public String getDominantChannel() {
        return channelContributions.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse("none");
    }
    
    /**
     * Check if consensus was reached with high confidence.
     */
    public boolean isHighConfidence(double threshold) {
        return confidence >= threshold;
    }
    
    /**
     * Get consensus quality based on contribution distribution.
     * Returns value between 0.0 (poor consensus) and 1.0 (perfect consensus).
     */
    public double getConsensusQuality() {
        if (channelContributions.isEmpty()) {
            return 0.0;
        }
        
        // Calculate entropy of contributions to measure consensus quality
        var entropy = channelContributions.values().stream()
            .mapToDouble(Double::doubleValue)
            .filter(contrib -> contrib > 0.0)
            .map(contrib -> -contrib * Math.log(contrib))
            .sum();
        
        // Normalize entropy to [0, 1] range (higher is better consensus)
        var maxEntropy = Math.log(channelContributions.size());
        return maxEntropy > 0 ? 1.0 - (entropy / maxEntropy) : 1.0;
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
    
    @Override
    public String toString() {
        return String.format("ConsensusResult{category=%d, confidence=%.3f, strategy='%s', " +
                           "dominant='%s', quality=%.3f}", 
                           category, confidence, strategy, getDominantChannel(), getConsensusQuality());
    }
}