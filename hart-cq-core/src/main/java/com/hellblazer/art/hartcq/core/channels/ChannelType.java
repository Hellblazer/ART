package com.hellblazer.art.hartcq.core.channels;

/**
 * Enumeration of all available channel types in the HART-CQ system.
 * Each channel type represents a different aspect of language processing.
 */
public enum ChannelType {
    /**
     * Positional encoding channel - provides position awareness using sinusoidal encoding.
     * CRITICAL: Essential for maintaining positional information in the model.
     */
    POSITIONAL("Positional", 64, true),
    
    /**
     * Word embedding channel - Word2Vec integration for COMPREHENSION ONLY.
     * NEVER used for generation to prevent hallucination.
     */
    WORD("Word", 128, true),
    
    /**
     * Context tracking channel - maintains historical information across windows.
     * Tracks previous 5 windows for context awareness.
     */
    CONTEXT("Context", 40, false),
    
    /**
     * Structural analysis channel - analyzes sentence structure and grammar.
     * Tracks grammatical patterns and syntactic relationships.
     */
    STRUCTURAL("Structural", 56, true),
    
    /**
     * Semantic meaning channel - extracts semantic features and topic coherence.
     * Analyzes meaning relationships and semantic transitions.
     */
    SEMANTIC("Semantic", 48, true),
    
    /**
     * Temporal sequence channel - tracks time-based patterns and sequence ordering.
     * Maintains temporal relationships within and across windows.
     */
    TEMPORAL("Temporal", 32, false);
    
    private final String displayName;
    private final int defaultDimension;
    private final boolean isDeterministic;
    
    ChannelType(String displayName, int defaultDimension, boolean isDeterministic) {
        this.displayName = displayName;
        this.defaultDimension = defaultDimension;
        this.isDeterministic = isDeterministic;
    }
    
    /**
     * Gets the human-readable display name for this channel type.
     */
    public String getDisplayName() {
        return displayName;
    }
    
    /**
     * Gets the default output dimension for this channel type.
     */
    public int getDefaultDimension() {
        return defaultDimension;
    }
    
    /**
     * Returns true if this channel type is deterministic (same input produces same output).
     */
    public boolean isDeterministic() {
        return isDeterministic;
    }
    
    /**
     * Returns true if this channel is for comprehension only (like Word channel).
     */
    public boolean isComprehensionOnly() {
        return this == WORD;
    }
    
    /**
     * Gets the total dimension when all channels are combined.
     */
    public static int getTotalDimension() {
        int total = 0;
        for (ChannelType type : values()) {
            total += type.defaultDimension;
        }
        return total;
    }
    
    @Override
    public String toString() {
        return displayName + " (" + defaultDimension + "D)";
    }
}