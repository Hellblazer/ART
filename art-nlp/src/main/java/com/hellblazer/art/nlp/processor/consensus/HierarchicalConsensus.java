package com.hellblazer.art.nlp.processor.consensus;

import com.hellblazer.art.nlp.processor.ChannelResult;
import com.hellblazer.art.nlp.processor.ConsensusResult;

import java.util.*;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Hierarchical consensus strategy that processes channels in tiers.
 * Higher-tier channels (semantic) have more influence than lower-tier channels (syntactic).
 * Implements cascading decision making with confidence propagation.
 */
public class HierarchicalConsensus implements ConsensusStrategy {
    private static final Logger log = LoggerFactory.getLogger(HierarchicalConsensus.class);
    
    private final Map<String, Integer> channelTiers;
    private final double confidenceThreshold;
    private final boolean requireHighTierAgreement;
    private final double tierWeightMultiplier;
    
    /**
     * Create hierarchical consensus with default tiers.
     */
    public HierarchicalConsensus() {
        this(createDefaultTiers(), 0.6, true, 2.0);
    }
    
    /**
     * Create hierarchical consensus with custom configuration.
     * 
     * @param channelTiers Map of channel IDs to tier levels (higher number = higher tier)
     * @param confidenceThreshold Minimum confidence required for tier consensus
     * @param requireHighTierAgreement Whether high-tier channels must agree
     * @param tierWeightMultiplier Multiplier applied to higher tier weights
     */
    public HierarchicalConsensus(Map<String, Integer> channelTiers, 
                                double confidenceThreshold,
                                boolean requireHighTierAgreement,
                                double tierWeightMultiplier) {
        if (confidenceThreshold < 0.0 || confidenceThreshold > 1.0) {
            throw new IllegalArgumentException("Confidence threshold must be in [0.0, 1.0]: " + confidenceThreshold);
        }
        if (tierWeightMultiplier <= 0.0) {
            throw new IllegalArgumentException("Tier weight multiplier must be positive: " + tierWeightMultiplier);
        }
        
        this.channelTiers = new HashMap<>(channelTiers);
        this.confidenceThreshold = confidenceThreshold;
        this.requireHighTierAgreement = requireHighTierAgreement;
        this.tierWeightMultiplier = tierWeightMultiplier;
    }
    
    @Override
    public ConsensusResult computeConsensus(Map<String, ChannelResult> channelResults,
                                          Map<String, Double> channelWeights) {
        Objects.requireNonNull(channelResults, "channelResults cannot be null");
        Objects.requireNonNull(channelWeights, "channelWeights cannot be null");
        
        // Filter successful results
        var validResults = channelResults.entrySet().stream()
            .filter(entry -> entry.getValue().isSuccess())
            .filter(entry -> entry.getValue().confidence() >= confidenceThreshold)
            .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
        
        if (validResults.isEmpty()) {
            log.debug("No valid channel results for hierarchical consensus");
            return ConsensusResult.create(-1, 0.0, getStrategyName(), 
                                        calculateContributions(channelResults, channelWeights));
        }
        
        // Group channels by tier
        var tierGroups = groupChannelsByTier(validResults);
        var maxTier = tierGroups.keySet().stream().max(Integer::compareTo).orElse(0);
        
        // Find the actual max tier across all channels (including filtered ones)
        var actualMaxTier = channelResults.keySet().stream()
            .mapToInt(channelId -> channelTiers.getOrDefault(channelId, 0))
            .max().orElse(0);
        
        log.debug("Processing {} tiers (max tier: {})", tierGroups.size(), maxTier);
        
        // Process tiers from highest to lowest
        ConsensusResult bestResult = null;
        var metadata = new HashMap<String, Object>();
        var allContributions = new HashMap<String, Double>();
        
        for (var tier = maxTier; tier >= 0; tier--) {
            var tierChannels = tierGroups.get(tier);
            if (tierChannels == null || tierChannels.isEmpty()) {
                continue;
            }
            
            log.debug("Processing tier {}: channels {}", tier, tierChannels.keySet());
            
            // Compute tier consensus
            var tierResult = computeTierConsensus(tierChannels, channelWeights, tier);
            
            if (tierResult != null && tierResult.confidence() >= confidenceThreshold) {
                // Check if high-tier agreement is required
                if (requireHighTierAgreement && tier == maxTier) {
                    bestResult = tierResult;
                    metadata.put("decidingTier", tier);
                    metadata.put("highTierDecision", tier == actualMaxTier);
                    break;
                } else if (!requireHighTierAgreement || bestResult == null) {
                    // Use best available result
                    if (bestResult == null || tierResult.confidence() > bestResult.confidence()) {
                        bestResult = tierResult;
                        metadata.put("decidingTier", tier);
                        metadata.put("highTierDecision", tier == actualMaxTier);
                    }
                }
            }
            
            // Accumulate contributions
            if (tierResult != null) {
                tierResult.channelContributions().forEach((channel, contribution) -> 
                    allContributions.merge(channel, contribution, Double::sum));
            }
        }
        
        if (bestResult == null) {
            log.debug("No tier reached confidence threshold");
            return ConsensusResult.create(-1, 0.0, getStrategyName(),
                                        calculateContributions(channelResults, channelWeights));
        }
        
        // Add hierarchical metadata
        metadata.put("tierGroups", tierGroups.keySet());
        metadata.put("maxTier", maxTier);
        metadata.put("tierWeightMultiplier", tierWeightMultiplier);
        metadata.put("channelTiers", new HashMap<>(channelTiers));
        
        log.debug("Hierarchical consensus: category={}, confidence={:.3f}, deciding tier={}", 
                 bestResult.category(), bestResult.confidence(), metadata.get("decidingTier"));
        
        // Use the final calculated contributions for all channels
        var finalContributions = calculateContributions(channelResults, channelWeights);
        
        return ConsensusResult.create(
            bestResult.category(), 
            bestResult.confidence(),
            getStrategyName(),
            finalContributions,
            metadata
        );
    }
    
    /**
     * Group channels by their assigned tiers.
     */
    private Map<Integer, Map<String, ChannelResult>> groupChannelsByTier(
            Map<String, ChannelResult> validResults) {
        var tierGroups = new HashMap<Integer, Map<String, ChannelResult>>();
        
        for (var entry : validResults.entrySet()) {
            var channelId = entry.getKey();
            var result = entry.getValue();
            var tier = channelTiers.getOrDefault(channelId, 0); // Default to tier 0
            
            tierGroups.computeIfAbsent(tier, k -> new HashMap<>())
                     .put(channelId, result);
        }
        
        return tierGroups;
    }
    
    /**
     * Compute consensus within a single tier using weighted voting.
     */
    private ConsensusResult computeTierConsensus(Map<String, ChannelResult> tierChannels,
                                               Map<String, Double> channelWeights,
                                               int tier) {
        var categoryVotes = new HashMap<Integer, Double>();
        var categoryChannels = new HashMap<Integer, Set<String>>();
        var totalWeight = 0.0;
        var contributions = new HashMap<String, Double>();
        
        for (var entry : tierChannels.entrySet()) {
            var channelId = entry.getKey();
            var result = entry.getValue();
            var baseWeight = channelWeights.getOrDefault(channelId, 1.0);
            
            // Apply tier multiplier - higher tiers get exponentially more weight
            var tierMultiplier = Math.pow(tierWeightMultiplier, tier);
            var effectiveWeight = baseWeight * result.confidence() * tierMultiplier;
            
            var category = result.category();
            categoryVotes.merge(category, effectiveWeight, Double::sum);
            categoryChannels.computeIfAbsent(category, k -> new HashSet<>()).add(channelId);
            
            totalWeight += effectiveWeight;
            contributions.put(channelId, effectiveWeight);
        }
        
        if (categoryVotes.isEmpty()) {
            return null;
        }
        
        // Find winning category
        var winningEntry = categoryVotes.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .orElse(null);
        
        if (winningEntry == null) {
            return null;
        }
        
        var winningCategory = winningEntry.getKey();
        var winningWeight = winningEntry.getValue();
        var confidence = totalWeight > 0 ? winningWeight / totalWeight : 0.0;
        
        // Normalize contributions
        final var finalTotalWeight = totalWeight;
        if (finalTotalWeight > 0) {
            contributions.replaceAll((k, v) -> v / finalTotalWeight);
        }
        
        var metadata = Map.<String, Object>of(
            "tier", tier,
            "tierMultiplier", Math.pow(tierWeightMultiplier, tier),
            "categoryVotes", new HashMap<>(categoryVotes),
            "supportingChannels", categoryChannels.get(winningCategory),
            "tierChannels", tierChannels.keySet()
        );
        
        return ConsensusResult.create(winningCategory, confidence, getStrategyName() + "_Tier" + tier,
                                    contributions, metadata);
    }
    
    /**
     * Calculate channel contributions across all tiers.
     */
    private Map<String, Double> calculateContributions(Map<String, ChannelResult> channelResults,
                                                      Map<String, Double> channelWeights) {
        var contributions = new HashMap<String, Double>();
        var totalWeight = 0.0;
        
        for (var entry : channelResults.entrySet()) {
            var channelId = entry.getKey();
            var result = entry.getValue();
            
            if (result.isSuccess() && result.confidence() >= confidenceThreshold) {
                var baseWeight = channelWeights.getOrDefault(channelId, 1.0);
                var tier = channelTiers.getOrDefault(channelId, 0);
                var tierMultiplier = Math.pow(tierWeightMultiplier, tier);
                var effectiveWeight = baseWeight * result.confidence() * tierMultiplier;
                
                contributions.put(channelId, effectiveWeight);
                totalWeight += effectiveWeight;
            } else {
                contributions.put(channelId, 0.0);
            }
        }
        
        // Normalize contributions
        final var finalTotalWeight = totalWeight;
        if (finalTotalWeight > 0) {
            contributions.replaceAll((k, v) -> v / finalTotalWeight);
        }
        
        return contributions;
    }
    
    /**
     * Create default channel tier assignments.
     */
    private static Map<String, Integer> createDefaultTiers() {
        return Map.of(
            "semantic", 3,      // Highest tier - word embeddings
            "fasttext", 3,      // Alias for semantic
            "entity", 2,        // Mid-high tier - named entities
            "ner", 2,           // Alias for entity
            "syntactic", 1,     // Mid tier - syntax patterns
            "pos", 1,           // Alias for syntactic
            "lexical", 0        // Base tier - word-level features
        );
    }
    
    @Override
    public String getStrategyName() {
        return "HierarchicalConsensus";
    }
    
    @Override
    public int getMinimumRequiredChannels() {
        return 1;
    }
    
    /**
     * Get channel tier assignments.
     */
    public Map<String, Integer> getChannelTiers() {
        return new HashMap<>(channelTiers);
    }
    
    /**
     * Get confidence threshold.
     */
    public double getConfidenceThreshold() {
        return confidenceThreshold;
    }
    
    /**
     * Check if high-tier agreement is required.
     */
    public boolean isRequireHighTierAgreement() {
        return requireHighTierAgreement;
    }
    
    /**
     * Get tier weight multiplier.
     */
    public double getTierWeightMultiplier() {
        return tierWeightMultiplier;
    }
    
    @Override
    public String toString() {
        return String.format("HierarchicalConsensus{tiers=%d, threshold=%.2f, requireHighTier=%s, multiplier=%.1f}",
                           channelTiers.size(), confidenceThreshold, requireHighTierAgreement, tierWeightMultiplier);
    }
}