package com.hellblazer.art.nlp.processor.adaptation;

import com.hellblazer.art.nlp.processor.ChannelResult;
import com.hellblazer.art.nlp.processor.ConsensusResult;

import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Adaptive weight manager that learns optimal channel weights from historical performance.
 * Uses machine learning techniques to continuously improve channel weighting based on
 * success patterns, channel correlations, and context-specific performance.
 */
public class AdaptiveWeightManager {
    private static final Logger log = LoggerFactory.getLogger(AdaptiveWeightManager.class);
    
    private final Map<String, ChannelWeightHistory> channelHistories;
    private final Map<String, Map<String, ChannelCorrelation>> channelCorrelations;
    private final Map<String, ContextualWeights> contextualWeights;
    private final AdaptationConfig config;
    private final AtomicLong totalAdaptations = new AtomicLong(0);
    
    public AdaptiveWeightManager() {
        this(AdaptationConfig.defaultConfig());
    }
    
    public AdaptiveWeightManager(AdaptationConfig config) {
        Objects.requireNonNull(config, "config cannot be null");
        this.config = config;
        this.channelHistories = new ConcurrentHashMap<>();
        this.channelCorrelations = new ConcurrentHashMap<>();
        this.contextualWeights = new ConcurrentHashMap<>();
    }
    
    /**
     * Learn from processing results and update adaptive weights.
     */
    public void learn(Map<String, ChannelResult> channelResults, ConsensusResult consensusResult, String context) {
        Objects.requireNonNull(channelResults, "channelResults cannot be null");
        Objects.requireNonNull(consensusResult, "consensusResult cannot be null");
        
        var contextKey = context != null ? context : "default";
        
        // Update individual channel histories
        for (var entry : channelResults.entrySet()) {
            var channelId = entry.getKey();
            var result = entry.getValue();
            
            updateChannelHistory(channelId, result, consensusResult);
        }
        
        // Update channel correlations
        if (config.enableCorrelationLearning()) {
            updateChannelCorrelations(channelResults, consensusResult);
        }
        
        // Update contextual weights
        if (config.enableContextualLearning()) {
            updateContextualWeights(contextKey, channelResults, consensusResult);
        }
        
        totalAdaptations.incrementAndGet();
        
        log.debug("Learned from {} channels, consensus confidence: {:.3f}, context: {}", 
                 channelResults.size(), consensusResult.confidence(), contextKey);
    }
    
    /**
     * Get current adaptive weights for channels, optionally considering context.
     */
    public Map<String, Double> getAdaptiveWeights(Set<String> channelIds, String context) {
        Objects.requireNonNull(channelIds, "channelIds cannot be null");
        
        var weights = new HashMap<String, Double>();
        var contextKey = context != null ? context : "default";
        
        // Start with base weights from individual channel performance
        for (var channelId : channelIds) {
            var baseWeight = getBaseChannelWeight(channelId);
            weights.put(channelId, baseWeight);
        }
        
        // Apply contextual adjustments
        if (config.enableContextualLearning()) {
            applyContextualAdjustments(weights, contextKey);
        }
        
        // Apply correlation-based adjustments
        if (config.enableCorrelationLearning()) {
            applyCorrelationAdjustments(weights, channelIds);
        }
        
        // Normalize weights
        normalizeWeights(weights);
        
        log.debug("Generated adaptive weights for {} channels in context '{}': {}", 
                 channelIds.size(), contextKey, weights);
        
        return weights;
    }
    
    /**
     * Get channel correlations - how well channels work together.
     */
    public Map<String, Map<String, Double>> getChannelCorrelations() {
        var correlations = new HashMap<String, Map<String, Double>>();
        
        for (var entry1 : channelCorrelations.entrySet()) {
            var channelId1 = entry1.getKey();
            var correlationMap = new HashMap<String, Double>();
            
            for (var entry2 : entry1.getValue().entrySet()) {
                var channelId2 = entry2.getKey();
                var correlation = entry2.getValue();
                correlationMap.put(channelId2, correlation.getCorrelationScore());
            }
            
            correlations.put(channelId1, correlationMap);
        }
        
        return correlations;
    }
    
    /**
     * Get adaptation statistics.
     */
    public AdaptationStats getAdaptationStats() {
        var channelStats = new HashMap<String, ChannelAdaptationStats>();
        
        for (var entry : channelHistories.entrySet()) {
            var channelId = entry.getKey();
            var history = entry.getValue();
            
            channelStats.put(channelId, new ChannelAdaptationStats(
                channelId,
                history.getTotalObservations(),
                history.getSuccessRate(),
                history.getAverageConfidence(),
                history.getCurrentWeight(),
                history.getWeightTrend()
            ));
        }
        
        return new AdaptationStats(
            totalAdaptations.get(),
            channelStats,
            getChannelCorrelations(),
            contextualWeights.keySet()
        );
    }
    
    /**
     * Reset all learned weights and correlations.
     */
    public void resetLearning() {
        channelHistories.clear();
        channelCorrelations.clear();
        contextualWeights.clear();
        totalAdaptations.set(0);
        log.debug("Reset all adaptive learning data");
    }
    
    /**
     * Export learned weights for persistence.
     */
    public Map<String, Object> exportWeights() {
        var export = new HashMap<String, Object>();
        
        // Export channel histories
        var channelData = new HashMap<String, Object>();
        for (var entry : channelHistories.entrySet()) {
            channelData.put(entry.getKey(), entry.getValue().exportData());
        }
        export.put("channelHistories", channelData);
        
        // Export correlations
        export.put("correlations", getChannelCorrelations());
        
        // Export contextual weights
        var contextData = new HashMap<String, Object>();
        for (var entry : contextualWeights.entrySet()) {
            contextData.put(entry.getKey(), entry.getValue().exportData());
        }
        export.put("contextualWeights", contextData);
        
        export.put("totalAdaptations", totalAdaptations.get());
        export.put("exportTime", Instant.now().toString());
        
        return export;
    }
    
    // Private helper methods
    
    private void updateChannelHistory(String channelId, ChannelResult result, ConsensusResult consensusResult) {
        var history = channelHistories.computeIfAbsent(channelId, k -> new ChannelWeightHistory());
        
        // Calculate performance score based on channel contribution to consensus
        var contributionWeight = consensusResult.channelContributions().getOrDefault(channelId, 0.0);
        var performanceScore = result.isSuccess() ? result.confidence() * contributionWeight : 0.0;
        
        history.addObservation(performanceScore, result.processingTimeMs());
        
        // Update weight using exponential moving average
        if (config.adaptationStrategy() == AdaptationStrategy.EXPONENTIAL_MOVING_AVERAGE) {
            var currentWeight = history.getCurrentWeight();
            var learningRate = config.learningRate();
            var newWeight = (1 - learningRate) * currentWeight + learningRate * performanceScore;
            history.updateWeight(newWeight);
        }
    }
    
    private void updateChannelCorrelations(Map<String, ChannelResult> channelResults, ConsensusResult consensusResult) {
        var successfulChannels = channelResults.entrySet().stream()
            .filter(entry -> entry.getValue().isSuccess())
            .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
        
        // Update pairwise correlations for successful channels
        for (var entry1 : successfulChannels.entrySet()) {
            var channel1 = entry1.getKey();
            var result1 = entry1.getValue();
            
            var correlationMap = channelCorrelations.computeIfAbsent(channel1, k -> new ConcurrentHashMap<>());
            
            for (var entry2 : successfulChannels.entrySet()) {
                var channel2 = entry2.getKey();
                var result2 = entry2.getValue();
                
                if (!channel1.equals(channel2)) {
                    var correlation = correlationMap.computeIfAbsent(channel2, k -> new ChannelCorrelation());
                    
                    // Simple correlation based on confidence similarity and joint success
                    var confidenceSimilarity = 1.0 - Math.abs(result1.confidence() - result2.confidence());
                    var categorySimilarity = result1.category() == result2.category() ? 1.0 : 0.0;
                    var jointScore = (confidenceSimilarity + categorySimilarity) / 2.0;
                    
                    correlation.addObservation(jointScore);
                }
            }
        }
    }
    
    private void updateContextualWeights(String context, Map<String, ChannelResult> channelResults, ConsensusResult consensusResult) {
        var contextWeights = contextualWeights.computeIfAbsent(context, k -> new ContextualWeights());
        
        for (var entry : channelResults.entrySet()) {
            var channelId = entry.getKey();
            var result = entry.getValue();
            
            var contributionWeight = consensusResult.channelContributions().getOrDefault(channelId, 0.0);
            var contextScore = result.isSuccess() ? result.confidence() * contributionWeight : 0.0;
            
            contextWeights.updateChannelWeight(channelId, contextScore);
        }
    }
    
    private double getBaseChannelWeight(String channelId) {
        var history = channelHistories.get(channelId);
        if (history != null) {
            return history.getCurrentWeight();
        }
        return config.defaultChannelWeight();
    }
    
    private void applyContextualAdjustments(Map<String, Double> weights, String context) {
        var contextWeights = contextualWeights.get(context);
        if (contextWeights != null) {
            for (var entry : weights.entrySet()) {
                var channelId = entry.getKey();
                var contextualAdjustment = contextWeights.getChannelWeight(channelId);
                var adjustedWeight = entry.getValue() * (1.0 + contextualAdjustment * config.contextualInfluence());
                weights.put(channelId, Math.max(0.01, adjustedWeight)); // Minimum weight threshold
            }
        }
    }
    
    private void applyCorrelationAdjustments(Map<String, Double> weights, Set<String> channelIds) {
        if (channelIds.size() < 2) return;
        
        var adjustments = new HashMap<String, Double>();
        
        for (var channelId : channelIds) {
            var correlationMap = channelCorrelations.get(channelId);
            if (correlationMap != null) {
                var totalCorrelation = 0.0;
                var correlationCount = 0;
                
                for (var otherChannelId : channelIds) {
                    if (!channelId.equals(otherChannelId)) {
                        var correlation = correlationMap.get(otherChannelId);
                        if (correlation != null) {
                            totalCorrelation += correlation.getCorrelationScore();
                            correlationCount++;
                        }
                    }
                }
                
                if (correlationCount > 0) {
                    var avgCorrelation = totalCorrelation / correlationCount;
                    adjustments.put(channelId, avgCorrelation * config.correlationInfluence());
                }
            }
        }
        
        // Apply correlation adjustments
        for (var entry : adjustments.entrySet()) {
            var channelId = entry.getKey();
            var adjustment = entry.getValue();
            var currentWeight = weights.get(channelId);
            weights.put(channelId, Math.max(0.01, currentWeight * (1.0 + adjustment)));
        }
    }
    
    private void normalizeWeights(Map<String, Double> weights) {
        var totalWeight = weights.values().stream().mapToDouble(Double::doubleValue).sum();
        
        if (totalWeight > 0) {
            weights.replaceAll((k, v) -> v / totalWeight);
        } else {
            // Equal weights fallback
            var equalWeight = 1.0 / weights.size();
            weights.replaceAll((k, v) -> equalWeight);
        }
    }
    
    // Inner classes for data structures
    
    /**
     * Tracks historical performance and weight evolution for a single channel.
     */
    private static class ChannelWeightHistory {
        private final List<Double> performanceHistory = new ArrayList<>();
        private final List<Long> processingTimes = new ArrayList<>();
        private double currentWeight = 1.0;
        private final int maxHistorySize = 100;
        
        void addObservation(double performanceScore, long processingTime) {
            performanceHistory.add(performanceScore);
            processingTimes.add(processingTime);
            
            // Keep history bounded
            if (performanceHistory.size() > maxHistorySize) {
                performanceHistory.removeFirst();
                processingTimes.removeFirst();
            }
        }
        
        void updateWeight(double newWeight) {
            this.currentWeight = Math.max(0.01, Math.min(10.0, newWeight)); // Bounds: [0.01, 10.0]
        }
        
        double getCurrentWeight() {
            return currentWeight;
        }
        
        long getTotalObservations() {
            return performanceHistory.size();
        }
        
        double getSuccessRate() {
            if (performanceHistory.isEmpty()) return 0.0;
            return (double) performanceHistory.stream().mapToLong(score -> score > 0 ? 1 : 0).sum() / performanceHistory.size();
        }
        
        double getAverageConfidence() {
            return performanceHistory.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        }
        
        String getWeightTrend() {
            if (performanceHistory.size() < 3) return "INSUFFICIENT_DATA";
            
            var recent = performanceHistory.subList(Math.max(0, performanceHistory.size() - 3), performanceHistory.size());
            var isIncreasing = recent.get(1) > recent.get(0) && recent.get(2) > recent.get(1);
            var isDecreasing = recent.get(1) < recent.get(0) && recent.get(2) < recent.get(1);
            
            if (isIncreasing) return "INCREASING";
            if (isDecreasing) return "DECREASING";
            return "STABLE";
        }
        
        Map<String, Object> exportData() {
            return Map.of(
                "currentWeight", currentWeight,
                "totalObservations", getTotalObservations(),
                "successRate", getSuccessRate(),
                "averageConfidence", getAverageConfidence(),
                "trend", getWeightTrend()
            );
        }
    }
    
    /**
     * Tracks correlation between two channels.
     */
    private static class ChannelCorrelation {
        private final List<Double> correlationHistory = new ArrayList<>();
        private final int maxHistorySize = 50;
        
        void addObservation(double correlation) {
            correlationHistory.add(correlation);
            
            if (correlationHistory.size() > maxHistorySize) {
                correlationHistory.removeFirst();
            }
        }
        
        double getCorrelationScore() {
            return correlationHistory.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        }
    }
    
    /**
     * Context-specific channel weights.
     */
    private static class ContextualWeights {
        private final Map<String, Double> channelWeights = new ConcurrentHashMap<>();
        
        void updateChannelWeight(String channelId, double score) {
            channelWeights.merge(channelId, score, (existing, newScore) -> {
                // Use exponential moving average
                return 0.7 * existing + 0.3 * newScore;
            });
        }
        
        double getChannelWeight(String channelId) {
            return channelWeights.getOrDefault(channelId, 0.0);
        }
        
        Map<String, Object> exportData() {
            return new HashMap<>(channelWeights);
        }
    }
    
    /**
     * Statistics about individual channel adaptation.
     */
    public record ChannelAdaptationStats(
        String channelId,
        long totalObservations,
        double successRate,
        double averageConfidence,
        double currentWeight,
        String weightTrend
    ) {}
    
    /**
     * Complete adaptation statistics.
     */
    public record AdaptationStats(
        long totalAdaptations,
        Map<String, ChannelAdaptationStats> channelStats,
        Map<String, Map<String, Double>> channelCorrelations,
        Set<String> knownContexts
    ) {}
    
    /**
     * Configuration for adaptive weight management.
     */
    public record AdaptationConfig(
        double learningRate,
        double defaultChannelWeight,
        double contextualInfluence,
        double correlationInfluence,
        boolean enableCorrelationLearning,
        boolean enableContextualLearning,
        AdaptationStrategy adaptationStrategy
    ) {
        public AdaptationConfig {
            if (learningRate <= 0.0 || learningRate > 1.0) {
                throw new IllegalArgumentException("Learning rate must be in (0.0, 1.0]: " + learningRate);
            }
            if (defaultChannelWeight <= 0.0) {
                throw new IllegalArgumentException("Default channel weight must be positive: " + defaultChannelWeight);
            }
            Objects.requireNonNull(adaptationStrategy, "adaptationStrategy cannot be null");
        }
        
        public static AdaptationConfig defaultConfig() {
            return new AdaptationConfig(
                0.1,                                    // learningRate
                1.0,                                    // defaultChannelWeight
                0.2,                                    // contextualInfluence
                0.15,                                   // correlationInfluence
                true,                                   // enableCorrelationLearning
                true,                                   // enableContextualLearning
                AdaptationStrategy.EXPONENTIAL_MOVING_AVERAGE  // adaptationStrategy
            );
        }
        
        public static AdaptationConfig conservativeConfig() {
            return new AdaptationConfig(
                0.05,                                   // lower learning rate
                1.0,
                0.1,                                    // lower contextual influence
                0.05,                                   // lower correlation influence
                false,                                  // disable correlation learning
                true,                                   // keep contextual learning
                AdaptationStrategy.EXPONENTIAL_MOVING_AVERAGE
            );
        }
        
        public static AdaptationConfig aggressiveConfig() {
            return new AdaptationConfig(
                0.3,                                    // higher learning rate
                1.0,
                0.4,                                    // higher contextual influence
                0.3,                                    // higher correlation influence
                true,
                true,
                AdaptationStrategy.EXPONENTIAL_MOVING_AVERAGE
            );
        }
    }
    
    /**
     * Adaptation strategies for weight updates.
     */
    public enum AdaptationStrategy {
        EXPONENTIAL_MOVING_AVERAGE,    // EMA-based weight updates
        REINFORCEMENT_LEARNING,        // Q-learning style updates (future extension)
        BAYESIAN_OPTIMIZATION          // Bayesian optimization (future extension)
    }
}