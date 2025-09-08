package com.hellblazer.art.nlp.processor.fusion;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.nlp.processor.ChannelResult;

import java.util.Map;

/**
 * Strategy interface for fusing features from multiple channels.
 * Implementations define different approaches to combining channel feature vectors.
 */
public interface FeatureFusionStrategy {
    
    /**
     * Fuse features from multiple channel results into a single vector.
     * 
     * @param channelResults Results from individual channels
     * @return Fused feature vector, or null if fusion fails
     */
    DenseVector fuseFeatures(Map<String, ChannelResult> channelResults);
    
    /**
     * Get strategy name for identification and logging.
     */
    String getStrategyName();
    
    /**
     * Get expected output dimension for the fused vector.
     * Returns -1 if dimension depends on input channels.
     */
    default int getOutputDimension() {
        return -1;
    }
    
    /**
     * Check if this strategy requires all channels to provide features.
     */
    default boolean requiresAllChannels() {
        return false;
    }
    
    /**
     * Get minimum number of successful channels required for fusion.
     */
    default int getMinimumRequiredChannels() {
        return 1;
    }
}