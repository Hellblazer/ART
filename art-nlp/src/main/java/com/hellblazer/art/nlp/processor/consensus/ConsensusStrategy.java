package com.hellblazer.art.nlp.processor.consensus;

import com.hellblazer.art.nlp.processor.ChannelResult;
import com.hellblazer.art.nlp.processor.ConsensusResult;

import java.util.Map;

/**
 * Strategy interface for computing consensus across multiple channel results.
 * Implementations define different approaches to combining channel classifications.
 */
public interface ConsensusStrategy {
    
    /**
     * Compute consensus result from multiple channel results.
     * 
     * @param channelResults Results from individual channels
     * @param channelWeights Weights for each channel (higher = more influence)
     * @return Consensus decision with confidence and metadata
     */
    ConsensusResult computeConsensus(Map<String, ChannelResult> channelResults,
                                   Map<String, Double> channelWeights);
    
    /**
     * Get strategy name for identification and logging.
     */
    String getStrategyName();
    
    /**
     * Check if this strategy requires all channels to succeed.
     */
    default boolean requiresAllChannels() {
        return false;
    }
    
    /**
     * Get minimum number of successful channels required.
     */
    default int getMinimumRequiredChannels() {
        return 1;
    }
}