package com.hellblazer.art.nlp.processor;

import java.util.Map;
import java.util.Objects;

/**
 * Performance metrics for multi-channel processing.
 * Contains aggregate metrics and per-channel statistics.
 */
public record MultiChannelMetrics(
    int totalProcessed,
    int successfulProcessed,
    int activeChannels,
    double overallSuccessRate,
    double averageProcessingTimeMs,
    Map<String, Object> channelStatistics,
    Map<String, Double> channelContributions
) {
    
    public MultiChannelMetrics {
        Objects.requireNonNull(channelStatistics, "channelStatistics cannot be null");
        Objects.requireNonNull(channelContributions, "channelContributions cannot be null");
        
        if (totalProcessed < 0) {
            throw new IllegalArgumentException("totalProcessed cannot be negative: " + totalProcessed);
        }
        
        if (successfulProcessed < 0 || successfulProcessed > totalProcessed) {
            throw new IllegalArgumentException("successfulProcessed must be in [0, totalProcessed]: " + 
                                             successfulProcessed);
        }
        
        if (activeChannels < 0) {
            throw new IllegalArgumentException("activeChannels cannot be negative: " + activeChannels);
        }
        
        if (overallSuccessRate < 0.0 || overallSuccessRate > 1.0) {
            throw new IllegalArgumentException("overallSuccessRate must be in [0.0, 1.0]: " + overallSuccessRate);
        }
        
        if (averageProcessingTimeMs < 0.0) {
            throw new IllegalArgumentException("averageProcessingTimeMs cannot be negative: " + 
                                             averageProcessingTimeMs);
        }
    }
    
    /**
     * Get failure rate.
     */
    public double getFailureRate() {
        return 1.0 - overallSuccessRate;
    }
    
    /**
     * Get processing throughput (items per second).
     */
    public double getThroughput() {
        return averageProcessingTimeMs > 0 ? 1000.0 / averageProcessingTimeMs : 0.0;
    }
    
    /**
     * Check if processing performance is considered good.
     */
    public boolean isPerformanceGood() {
        return overallSuccessRate >= 0.8 && averageProcessingTimeMs <= 1000.0;
    }
    
    /**
     * Get the most contributing channel.
     */
    public String getTopPerformingChannel() {
        return channelContributions.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse("none");
    }
    
    /**
     * Get contribution for a specific channel.
     */
    public double getChannelContribution(String channelId) {
        return channelContributions.getOrDefault(channelId, 0.0);
    }
    
    /**
     * Calculate channel diversity index.
     * Higher values indicate better balance across channels.
     */
    public double getChannelDiversity() {
        if (channelContributions.isEmpty()) {
            return 0.0;
        }
        
        // Calculate Shannon diversity index
        var entropy = channelContributions.values().stream()
            .mapToDouble(Double::doubleValue)
            .filter(contrib -> contrib > 0.0)
            .map(contrib -> -contrib * Math.log(contrib))
            .sum();
        
        var maxEntropy = Math.log(channelContributions.size());
        return maxEntropy > 0 ? entropy / maxEntropy : 0.0;
    }
    
    /**
     * Get efficiency score (success rate weighted by processing speed).
     */
    public double getEfficiencyScore() {
        var speedScore = averageProcessingTimeMs > 0 ? 
            Math.min(1.0, 100.0 / averageProcessingTimeMs) : 0.0;
        return (overallSuccessRate + speedScore) / 2.0;
    }
    
    /**
     * Check if system is under-utilizing channels.
     */
    public boolean isUnderUtilized() {
        if (activeChannels <= 1) {
            return false;
        }
        
        var maxContribution = channelContributions.values().stream()
            .mapToDouble(Double::doubleValue)
            .max()
            .orElse(0.0);
        
        // If one channel dominates with >80% contribution, system may be under-utilized
        return maxContribution > 0.8;
    }
    
    /**
     * Get performance summary as formatted string.
     */
    public String getPerformanceSummary() {
        return String.format(
            "MultiChannel Performance: %d/%d processed (%.1f%% success), " +
            "%.1fms avg time, %d channels, top: %s (%.1f%% contrib)",
            successfulProcessed, totalProcessed, overallSuccessRate * 100,
            averageProcessingTimeMs, activeChannels,
            getTopPerformingChannel(), getChannelContribution(getTopPerformingChannel()) * 100
        );
    }
    
    /**
     * Create empty metrics for testing.
     */
    public static MultiChannelMetrics empty() {
        return new MultiChannelMetrics(0, 0, 0, 0.0, 0.0, Map.of(), Map.of());
    }
    
    @Override
    public String toString() {
        return String.format(
            "MultiChannelMetrics{processed=%d, success=%.1f%%, avgTime=%.1fms, " +
            "channels=%d, diversity=%.3f, efficiency=%.3f}",
            totalProcessed, overallSuccessRate * 100, averageProcessingTimeMs,
            activeChannels, getChannelDiversity(), getEfficiencyScore()
        );
    }
}