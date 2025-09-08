package com.hellblazer.art.nlp.processor.optimization;

import com.hellblazer.art.nlp.processor.ChannelResult;
import com.hellblazer.art.nlp.processor.ConsensusResult;
import com.hellblazer.art.nlp.processor.fusion.FeatureFusionStrategy;
import com.hellblazer.art.nlp.processor.consensus.ConsensusStrategy;

import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Performance optimization framework for ART-NLP processing.
 * Provides caching, performance monitoring, and adaptive optimization strategies.
 */
public class PerformanceOptimizer {
    private static final Logger log = LoggerFactory.getLogger(PerformanceOptimizer.class);
    
    private final Map<String, CacheEntry> resultCache;
    private final Map<String, ChannelPerformanceMetrics> channelMetrics;
    private final Map<String, StrategyPerformanceMetrics> strategyMetrics;
    private final OptimizationConfig config;
    private final AtomicLong cacheHits = new AtomicLong(0);
    private final AtomicLong cacheMisses = new AtomicLong(0);
    private final AtomicReference<OptimizationStrategy> currentStrategy = new AtomicReference<>(OptimizationStrategy.ADAPTIVE);
    
    public PerformanceOptimizer() {
        this(OptimizationConfig.defaultConfig());
    }
    
    public PerformanceOptimizer(OptimizationConfig config) {
        Objects.requireNonNull(config, "config cannot be null");
        this.config = config;
        this.resultCache = new ConcurrentHashMap<>(config.maxCacheSize());
        this.channelMetrics = new ConcurrentHashMap<>();
        this.strategyMetrics = new ConcurrentHashMap<>();
    }
    
    /**
     * Optimize channel processing based on historical performance.
     */
    public Map<String, ChannelResult> optimizeChannelProcessing(String inputText, 
                                                               Map<String, ChannelResult> channelResults) {
        Objects.requireNonNull(inputText, "inputText cannot be null");
        Objects.requireNonNull(channelResults, "channelResults cannot be null");
        
        var cacheKey = generateCacheKey(inputText, channelResults.keySet());
        
        // Try cache first
        if (config.enableCaching()) {
            var cached = getCachedResult(cacheKey);
            if (cached != null) {
                cacheHits.incrementAndGet();
                log.debug("Cache hit for input (length={})", inputText.length());
                return cached;
            }
            cacheMisses.incrementAndGet();
        }
        
        // Record channel performance metrics
        var optimizedResults = new HashMap<String, ChannelResult>();
        for (var entry : channelResults.entrySet()) {
            var channelId = entry.getKey();
            var result = entry.getValue();
            
            recordChannelPerformance(channelId, result);
            
            // Apply channel-specific optimizations
            var optimizedResult = optimizeChannelResult(channelId, result);
            optimizedResults.put(channelId, optimizedResult);
        }
        
        // Cache the results
        if (config.enableCaching()) {
            cacheResults(cacheKey, optimizedResults);
        }
        
        return optimizedResults;
    }
    
    /**
     * Recommend optimal consensus strategy based on performance metrics.
     */
    public String recommendConsensusStrategy(Map<String, ChannelResult> channelResults) {
        Objects.requireNonNull(channelResults, "channelResults cannot be null");
        
        var numChannels = channelResults.size();
        var avgConfidence = channelResults.values().stream()
            .filter(ChannelResult::isSuccess)
            .mapToDouble(ChannelResult::confidence)
            .average()
            .orElse(0.0);
        
        var confidenceVariance = calculateConfidenceVariance(channelResults);
        
        // Adaptive strategy selection based on data characteristics
        if (numChannels >= 5 && confidenceVariance > 0.05) {
            return "HierarchicalConsensus"; // Good for many channels with varying confidence
        } else if (avgConfidence > 0.8 && numChannels >= 3) {
            return "AttentionConsensus"; // Good for high-confidence multi-channel scenarios
        } else if (numChannels <= 2) {
            return "WeightedVoting"; // Simple strategy for few channels
        } else {
            return "MajorityVoting"; // Fallback strategy
        }
    }
    
    /**
     * Recommend optimal fusion strategy based on channel characteristics.
     */
    public String recommendFusionStrategy(Map<String, ChannelResult> channelResults) {
        Objects.requireNonNull(channelResults, "channelResults cannot be null");
        
        var successfulChannels = channelResults.values().stream()
            .filter(ChannelResult::isSuccess)
            .toList();
        
        if (successfulChannels.size() >= 4) {
            return "PCA"; // Good for high-dimensional data reduction
        } else if (successfulChannels.size() >= 2) {
            return "AttentionFusion"; // Good for learning channel relationships
        } else {
            return "SimpleConcat"; // Fallback for single channel
        }
    }
    
    /**
     * Record consensus strategy performance.
     */
    public void recordConsensusPerformance(String strategyName, ConsensusResult result, long processingTimeMs) {
        Objects.requireNonNull(strategyName, "strategyName cannot be null");
        Objects.requireNonNull(result, "result cannot be null");
        
        var metrics = strategyMetrics.computeIfAbsent(strategyName, k -> new StrategyPerformanceMetrics());
        metrics.recordExecution(result.confidence(), processingTimeMs);
        
        log.debug("Recorded {} performance: confidence={:.3f}, time={}ms", 
                 strategyName, result.confidence(), processingTimeMs);
    }
    
    /**
     * Record fusion strategy performance.
     */
    public void recordFusionPerformance(String strategyName, Object result, long processingTimeMs) {
        Objects.requireNonNull(strategyName, "strategyName cannot be null");
        
        var metrics = strategyMetrics.computeIfAbsent("Fusion_" + strategyName, k -> new StrategyPerformanceMetrics());
        var confidence = result != null ? 1.0 : 0.0; // Simple success/failure metric
        metrics.recordExecution(confidence, processingTimeMs);
        
        log.debug("Recorded fusion {} performance: success={}, time={}ms", 
                 strategyName, result != null, processingTimeMs);
    }
    
    /**
     * Get performance statistics.
     */
    public PerformanceStats getPerformanceStats() {
        var channelStats = new HashMap<String, ChannelPerformanceMetrics>();
        channelMetrics.forEach((k, v) -> channelStats.put(k, v.copy()));
        
        var strategyStats = new HashMap<String, StrategyPerformanceMetrics>();
        strategyMetrics.forEach((k, v) -> strategyStats.put(k, v.copy()));
        
        var cacheStats = new CacheStats(
            cacheHits.get(),
            cacheMisses.get(),
            resultCache.size(),
            config.maxCacheSize()
        );
        
        return new PerformanceStats(channelStats, strategyStats, cacheStats, currentStrategy.get());
    }
    
    /**
     * Clear all cached results and metrics.
     */
    public void clearCache() {
        resultCache.clear();
        cacheHits.set(0);
        cacheMisses.set(0);
        log.debug("Performance cache cleared");
    }
    
    /**
     * Clear all performance metrics.
     */
    public void clearMetrics() {
        channelMetrics.clear();
        strategyMetrics.clear();
        log.debug("Performance metrics cleared");
    }
    
    /**
     * Update optimization strategy.
     */
    public void setOptimizationStrategy(OptimizationStrategy strategy) {
        Objects.requireNonNull(strategy, "strategy cannot be null");
        var previous = currentStrategy.getAndSet(strategy);
        log.debug("Optimization strategy changed from {} to {}", previous, strategy);
    }
    
    /**
     * Get current optimization strategy.
     */
    public OptimizationStrategy getOptimizationStrategy() {
        return currentStrategy.get();
    }
    
    // Private helper methods
    
    private String generateCacheKey(String inputText, Set<String> channelIds) {
        return String.format("input_%d_channels_%s", 
                           inputText.hashCode(), 
                           String.join(",", new TreeSet<>(channelIds)));
    }
    
    private Map<String, ChannelResult> getCachedResult(String cacheKey) {
        var entry = resultCache.get(cacheKey);
        if (entry != null && !entry.isExpired(config.cacheExpirationMs())) {
            return entry.results();
        } else if (entry != null) {
            resultCache.remove(cacheKey); // Remove expired entry
        }
        return null;
    }
    
    private void cacheResults(String cacheKey, Map<String, ChannelResult> results) {
        if (resultCache.size() >= config.maxCacheSize()) {
            // Simple LRU: remove oldest entries
            var oldestKey = resultCache.entrySet().stream()
                .min(Map.Entry.comparingByValue((a, b) -> a.timestamp().compareTo(b.timestamp())))
                .map(Map.Entry::getKey)
                .orElse(null);
            
            if (oldestKey != null) {
                resultCache.remove(oldestKey);
            }
        }
        
        resultCache.put(cacheKey, new CacheEntry(new HashMap<>(results), Instant.now()));
    }
    
    private void recordChannelPerformance(String channelId, ChannelResult result) {
        var metrics = channelMetrics.computeIfAbsent(channelId, k -> new ChannelPerformanceMetrics());
        metrics.recordExecution(result.isSuccess(), result.confidence(), result.processingTimeMs());
    }
    
    private ChannelResult optimizeChannelResult(String channelId, ChannelResult result) {
        // For now, return the result as-is
        // Future optimizations could include confidence adjustment, timeout handling, etc.
        return result;
    }
    
    private double calculateConfidenceVariance(Map<String, ChannelResult> channelResults) {
        var confidences = channelResults.values().stream()
            .filter(ChannelResult::isSuccess)
            .mapToDouble(ChannelResult::confidence)
            .toArray();
        
        if (confidences.length <= 1) {
            return 0.0;
        }
        
        var mean = Arrays.stream(confidences).average().orElse(0.0);
        var variance = Arrays.stream(confidences)
            .map(conf -> Math.pow(conf - mean, 2))
            .average()
            .orElse(0.0);
        
        return variance;
    }
    
    // Inner classes for data structures
    
    private record CacheEntry(Map<String, ChannelResult> results, Instant timestamp) {
        boolean isExpired(long expirationMs) {
            return Instant.now().toEpochMilli() - timestamp.toEpochMilli() > expirationMs;
        }
    }
    
    /**
     * Performance metrics for individual channels.
     */
    public static class ChannelPerformanceMetrics {
        private final AtomicLong totalExecutions = new AtomicLong(0);
        private final AtomicLong successfulExecutions = new AtomicLong(0);
        private final AtomicLong totalProcessingTime = new AtomicLong(0);
        private final AtomicReference<Double> avgConfidence = new AtomicReference<>(0.0);
        private volatile double minProcessingTime = Double.MAX_VALUE;
        private volatile double maxProcessingTime = Double.MIN_VALUE;
        
        void recordExecution(boolean success, double confidence, long processingTimeMs) {
            totalExecutions.incrementAndGet();
            if (success) {
                successfulExecutions.incrementAndGet();
                // Update running average confidence
                var currentAvg = avgConfidence.get();
                var count = successfulExecutions.get();
                var newAvg = ((currentAvg * (count - 1)) + confidence) / count;
                avgConfidence.set(newAvg);
            }
            
            totalProcessingTime.addAndGet(processingTimeMs);
            
            // Update min/max processing times
            synchronized(this) {
                if (processingTimeMs < minProcessingTime) minProcessingTime = processingTimeMs;
                if (processingTimeMs > maxProcessingTime) maxProcessingTime = processingTimeMs;
            }
        }
        
        public long getTotalExecutions() { return totalExecutions.get(); }
        public long getSuccessfulExecutions() { return successfulExecutions.get(); }
        public double getSuccessRate() { 
            var total = totalExecutions.get();
            return total > 0 ? (double) successfulExecutions.get() / total : 0.0;
        }
        public double getAverageConfidence() { return avgConfidence.get(); }
        public double getAverageProcessingTime() {
            var total = totalExecutions.get();
            return total > 0 ? (double) totalProcessingTime.get() / total : 0.0;
        }
        public double getMinProcessingTime() { 
            return minProcessingTime == Double.MAX_VALUE ? 0.0 : minProcessingTime; 
        }
        public double getMaxProcessingTime() { 
            return maxProcessingTime == Double.MIN_VALUE ? 0.0 : maxProcessingTime; 
        }
        
        ChannelPerformanceMetrics copy() {
            var copy = new ChannelPerformanceMetrics();
            copy.totalExecutions.set(this.totalExecutions.get());
            copy.successfulExecutions.set(this.successfulExecutions.get());
            copy.totalProcessingTime.set(this.totalProcessingTime.get());
            copy.avgConfidence.set(this.avgConfidence.get());
            copy.minProcessingTime = this.minProcessingTime;
            copy.maxProcessingTime = this.maxProcessingTime;
            return copy;
        }
    }
    
    /**
     * Performance metrics for strategies (consensus/fusion).
     */
    public static class StrategyPerformanceMetrics {
        private final AtomicLong totalExecutions = new AtomicLong(0);
        private final AtomicLong totalProcessingTime = new AtomicLong(0);
        private final AtomicReference<Double> avgConfidence = new AtomicReference<>(0.0);
        private volatile double minProcessingTime = Double.MAX_VALUE;
        private volatile double maxProcessingTime = Double.MIN_VALUE;
        
        void recordExecution(double confidence, long processingTimeMs) {
            var count = totalExecutions.incrementAndGet();
            totalProcessingTime.addAndGet(processingTimeMs);
            
            // Update running average confidence
            var currentAvg = avgConfidence.get();
            var newAvg = ((currentAvg * (count - 1)) + confidence) / count;
            avgConfidence.set(newAvg);
            
            // Update min/max processing times
            synchronized(this) {
                if (processingTimeMs < minProcessingTime) minProcessingTime = processingTimeMs;
                if (processingTimeMs > maxProcessingTime) maxProcessingTime = processingTimeMs;
            }
        }
        
        public long getTotalExecutions() { return totalExecutions.get(); }
        public double getAverageConfidence() { return avgConfidence.get(); }
        public double getAverageProcessingTime() {
            var total = totalExecutions.get();
            return total > 0 ? (double) totalProcessingTime.get() / total : 0.0;
        }
        public double getMinProcessingTime() { 
            return minProcessingTime == Double.MAX_VALUE ? 0.0 : minProcessingTime; 
        }
        public double getMaxProcessingTime() { 
            return maxProcessingTime == Double.MIN_VALUE ? 0.0 : maxProcessingTime; 
        }
        
        StrategyPerformanceMetrics copy() {
            var copy = new StrategyPerformanceMetrics();
            copy.totalExecutions.set(this.totalExecutions.get());
            copy.totalProcessingTime.set(this.totalProcessingTime.get());
            copy.avgConfidence.set(this.avgConfidence.get());
            copy.minProcessingTime = this.minProcessingTime;
            copy.maxProcessingTime = this.maxProcessingTime;
            return copy;
        }
    }
    
    /**
     * Cache performance statistics.
     */
    public record CacheStats(long hits, long misses, int currentSize, int maxSize) {
        public double getHitRate() {
            var total = hits + misses;
            return total > 0 ? (double) hits / total : 0.0;
        }
        
        public double getUtilization() {
            return maxSize > 0 ? (double) currentSize / maxSize : 0.0;
        }
    }
    
    /**
     * Complete performance statistics.
     */
    public record PerformanceStats(
        Map<String, ChannelPerformanceMetrics> channelMetrics,
        Map<String, StrategyPerformanceMetrics> strategyMetrics,
        CacheStats cacheStats,
        OptimizationStrategy currentStrategy
    ) {}
    
    /**
     * Optimization strategies.
     */
    public enum OptimizationStrategy {
        NONE,           // No optimization
        CACHE_ONLY,     // Only caching optimization
        ADAPTIVE,       // Adaptive strategy selection
        AGGRESSIVE      // Aggressive optimization (may sacrifice accuracy for speed)
    }
    
    /**
     * Configuration for performance optimization.
     */
    public record OptimizationConfig(
        boolean enableCaching,
        int maxCacheSize,
        long cacheExpirationMs,
        boolean enableMetrics,
        boolean enableAdaptiveStrategies
    ) {
        public OptimizationConfig {
            if (maxCacheSize < 0) {
                throw new IllegalArgumentException("maxCacheSize must be non-negative: " + maxCacheSize);
            }
            if (cacheExpirationMs < 0) {
                throw new IllegalArgumentException("cacheExpirationMs must be non-negative: " + cacheExpirationMs);
            }
        }
        
        public static OptimizationConfig defaultConfig() {
            return new OptimizationConfig(
                true,      // enableCaching
                1000,      // maxCacheSize
                300000,    // cacheExpirationMs (5 minutes)
                true,      // enableMetrics
                true       // enableAdaptiveStrategies
            );
        }
        
        public static OptimizationConfig disabledConfig() {
            return new OptimizationConfig(false, 0, 0, false, false);
        }
        
        public static OptimizationConfig cacheOnlyConfig(int maxCacheSize, long expirationMs) {
            return new OptimizationConfig(true, maxCacheSize, expirationMs, true, false);
        }
    }
}