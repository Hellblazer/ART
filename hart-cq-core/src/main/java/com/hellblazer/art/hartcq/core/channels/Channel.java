package com.hellblazer.art.hartcq.core.channels;

import com.hellblazer.art.hartcq.ProcessingWindow;
import com.hellblazer.art.hartcq.Token;

/**
 * Base interface for all HART-CQ processing channels.
 * Each channel processes the token window in a specific way and returns structured output.
 */
public interface Channel {
    
    /**
     * Processes a processing window and returns structured channel output.
     * @param window ProcessingWindow containing tokens and metadata
     * @return ChannelOutput with features, type, timing, and validity information
     */
    ChannelOutput process(ProcessingWindow window);
    
    /**
     * Processes a window of tokens and returns a feature vector.
     * @param tokens Array of tokens (size 20)
     * @return Feature vector representing the processed window
     * @deprecated Use process(ProcessingWindow) instead for structured output
     */
    @Deprecated
    default float[] processWindow(Token[] tokens) {
        // Convert Token[] to ProcessingWindow for backward compatibility
        var tokenList = java.util.Arrays.asList(tokens);
        var window = new ProcessingWindow(tokenList, System.currentTimeMillis());
        var result = process(window);
        return result.features();
    }
    
    /**
     * Gets the dimensionality of the feature vector this channel produces.
     * @return The number of dimensions in the output vector
     */
    int getOutputDimension();
    
    /**
     * Gets the name of this channel.
     * @return Channel name
     */
    String getName();
    
    /**
     * Gets the type of this channel.
     * @return ChannelType enum value
     */
    ChannelType getChannelType();
    
    /**
     * Resets any internal state of the channel.
     */
    void reset();
    
    /**
     * Checks if this channel is deterministic (same input produces same output).
     * @return true if deterministic
     */
    boolean isDeterministic();
}