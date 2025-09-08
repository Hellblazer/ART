package com.hellblazer.art.nlp.channels.base;

/**
 * AbstractNLPChannel - Alias for BaseChannel for backward compatibility.
 * 
 * This class exists to support legacy imports. All new code should use BaseChannel directly.
 * The functionality is identical to BaseChannel.
 * 
 * @deprecated Use {@link BaseChannel} instead
 */
@Deprecated
public abstract class AbstractNLPChannel extends BaseChannel {
    
    /**
     * Constructor matching BaseChannel signature.
     * 
     * @param channelName The name of the channel
     * @param vigilance The vigilance parameter [0.0, 1.0]
     */
    protected AbstractNLPChannel(String channelName, double vigilance) {
        super(channelName, vigilance);
    }
}