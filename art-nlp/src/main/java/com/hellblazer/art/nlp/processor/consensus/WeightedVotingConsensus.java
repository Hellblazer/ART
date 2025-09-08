package com.hellblazer.art.nlp.processor.consensus;

import com.hellblazer.art.nlp.processor.ChannelResult;
import com.hellblazer.art.nlp.processor.ConsensusResult;

import java.util.*;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Weighted voting consensus strategy.
 * Combines channel results using weighted voting where each channel's vote
 * is weighted by its assigned weight and confidence.
 */
public class WeightedVotingConsensus implements ConsensusStrategy {
    private static final Logger log = LoggerFactory.getLogger(WeightedVotingConsensus.class);
    
    private final double confidenceThreshold;
    private final boolean requireMajority;
    
    /**
     * Create weighted voting consensus with default parameters.
     */
    public WeightedVotingConsensus() {
        this(0.5, false);
    }
    
    /**
     * Create weighted voting consensus with custom parameters.
     * 
     * @param confidenceThreshold Minimum confidence required for a vote to count
     * @param requireMajority Whether to require majority agreement
     */
    public WeightedVotingConsensus(double confidenceThreshold, boolean requireMajority) {
        if (confidenceThreshold < 0.0 || confidenceThreshold > 1.0) {
            throw new IllegalArgumentException("Confidence threshold must be in [0.0, 1.0]: " + confidenceThreshold);
        }
        
        this.confidenceThreshold = confidenceThreshold;
        this.requireMajority = requireMajority;
    }
    
    @Override
    public ConsensusResult computeConsensus(Map<String, ChannelResult> channelResults,
                                          Map<String, Double> channelWeights) {
        Objects.requireNonNull(channelResults, "channelResults cannot be null");
        Objects.requireNonNull(channelWeights, "channelWeights cannot be null");
        
        // Filter successful results that meet confidence threshold
        var validResults = channelResults.entrySet().stream()
            .filter(entry -> entry.getValue().isSuccess())
            .filter(entry -> entry.getValue().confidence() >= confidenceThreshold)
            .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
        
        if (validResults.isEmpty()) {
            log.debug("No valid channel results for consensus");
            return ConsensusResult.create(-1, 0.0, getStrategyName(), 
                                        calculateContributions(channelResults, channelWeights));
        }
        
        // Calculate weighted votes for each category
        var categoryVotes = new HashMap<Integer, Double>();
        var categoryChannels = new HashMap<Integer, Set<String>>();
        var totalWeight = 0.0;
        
        for (var entry : validResults.entrySet()) {
            var channelId = entry.getKey();
            var result = entry.getValue();
            var weight = channelWeights.getOrDefault(channelId, 1.0);
            
            // Weight by channel weight and result confidence
            var effectiveWeight = weight * result.confidence();
            var category = result.category();
            
            categoryVotes.merge(category, effectiveWeight, Double::sum);
            categoryChannels.computeIfAbsent(category, k -> new HashSet<>()).add(channelId);
            totalWeight += effectiveWeight;
        }
        
        // Find winning category
        var winningEntry = categoryVotes.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .orElseThrow(() -> new IllegalStateException("No votes found"));
        
        var winningCategory = winningEntry.getKey();
        var winningWeight = winningEntry.getValue();
        var winningChannels = categoryChannels.get(winningCategory);
        
        // Calculate consensus confidence
        var consensusConfidence = totalWeight > 0 ? winningWeight / totalWeight : 0.0;
        
        // Check majority requirement
        if (requireMajority && consensusConfidence <= 0.5) {
            log.debug("Majority consensus not achieved: {} confidence", consensusConfidence);
            return ConsensusResult.create(-1, consensusConfidence, getStrategyName(),
                                        calculateContributions(channelResults, channelWeights));
        }
        
        // Create metadata
        var metadata = Map.<String, Object>of(
            "totalVotes", totalWeight,
            "winningVote", winningWeight,
            "categoryVotes", new HashMap<>(categoryVotes),
            "supportingChannels", new HashSet<>(winningChannels),
            "validChannels", validResults.keySet()
        );
        
        var contributions = calculateContributions(channelResults, channelWeights);
        
        log.debug("Consensus reached: category={}, confidence={:.3f}, channels={}", 
                 winningCategory, consensusConfidence, winningChannels);
        
        return ConsensusResult.create(winningCategory, consensusConfidence, getStrategyName(),
                                    contributions, metadata);
    }
    
    /**
     * Calculate channel contributions to the final decision.
     */
    private Map<String, Double> calculateContributions(Map<String, ChannelResult> channelResults,
                                                      Map<String, Double> channelWeights) {
        var contributions = new HashMap<String, Double>();
        var totalWeight = 0.0;
        
        // Calculate total effective weight
        for (var entry : channelResults.entrySet()) {
            var channelId = entry.getKey();
            var result = entry.getValue();
            
            if (result.isSuccess() && result.confidence() >= confidenceThreshold) {
                var weight = channelWeights.getOrDefault(channelId, 1.0);
                var effectiveWeight = weight * result.confidence();
                totalWeight += effectiveWeight;
            }
        }
        
        // Calculate normalized contributions
        for (var entry : channelResults.entrySet()) {
            var channelId = entry.getKey();
            var result = entry.getValue();
            
            if (result.isSuccess() && result.confidence() >= confidenceThreshold && totalWeight > 0) {
                var weight = channelWeights.getOrDefault(channelId, 1.0);
                var effectiveWeight = weight * result.confidence();
                contributions.put(channelId, effectiveWeight / totalWeight);
            } else {
                contributions.put(channelId, 0.0);
            }
        }
        
        return contributions;
    }
    
    @Override
    public String getStrategyName() {
        return "WeightedVoting";
    }
    
    @Override
    public int getMinimumRequiredChannels() {
        return requireMajority ? 2 : 1;
    }
    
    /**
     * Get confidence threshold.
     */
    public double getConfidenceThreshold() {
        return confidenceThreshold;
    }
    
    /**
     * Check if majority is required.
     */
    public boolean isRequireMajority() {
        return requireMajority;
    }
    
    @Override
    public String toString() {
        return String.format("WeightedVotingConsensus{threshold=%.2f, requireMajority=%s}",
                           confidenceThreshold, requireMajority);
    }
}