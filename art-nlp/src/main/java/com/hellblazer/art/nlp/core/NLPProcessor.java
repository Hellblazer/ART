package com.hellblazer.art.nlp.core;

import java.io.InputStream;
import java.util.concurrent.CompletableFuture;

/**
 * Main entry point for ART-NLP text processing.
 * 
 * This interface defines the primary API for multi-channel NLP processing 
 * using Adaptive Resonance Theory algorithms. All implementations must be 
 * thread-safe and support both synchronous and asynchronous processing.
 * 
 * CRITICAL: This interface matches the exact specification from API_DESIGN.md
 */
public interface NLPProcessor extends AutoCloseable {
    
    /**
     * Process a single text input through all channels.
     * 
     * @param text Input text to process
     * @return Multi-channel processing results
     * @throws IllegalArgumentException if text is null or blank
     */
    ProcessingResult process(String text);
    
    /**
     * Process text asynchronously.
     * 
     * @param text Input text to process
     * @return Future containing processing results
     */
    default CompletableFuture<ProcessingResult> processAsync(String text) {
        return CompletableFuture.supplyAsync(() -> process(text));
    }
    
    /**
     * Process streaming text with callback for each chunk.
     * 
     * @param stream Text stream source
     * @param callback Result callback for each processed chunk
     * @throws IllegalArgumentException if parameters are null
     */
    void processStream(InputStream stream, ResultCallback callback);
    
    /**
     * Process document with metadata.
     * 
     * @param document Document with content and metadata
     * @return Complete document analysis with enriched metadata
     * @throws IllegalArgumentException if document is null
     */
    DocumentAnalysis processDocument(Document document);
    
    /**
     * Get current processing statistics across all channels.
     * 
     * @return Comprehensive performance and accuracy metrics
     */
    ProcessingStats getStatistics();
    
    /**
     * Reset all ART networks and clear categories.
     * 
     * WARNING: This operation is destructive and cannot be undone.
     * All learned patterns will be lost.
     */
    void reset();
    
    /**
     * Reset specific channel.
     * 
     * @param channelName Name of channel to reset
     * @throws IllegalArgumentException if channel doesn't exist
     */
    void resetChannel(String channelName);
    
    /**
     * Enable or disable a specific channel.
     * 
     * @param channelName Name of channel to modify
     * @param enabled Whether channel should be enabled
     * @return true if channel state was changed, false if already in desired state
     */
    boolean setChannelEnabled(String channelName, boolean enabled);
    
    /**
     * Get names of all available channels.
     * 
     * @return Set of channel names
     */
    java.util.Set<String> getChannelNames();
    
    /**
     * Get names of currently enabled channels.
     * 
     * @return Set of enabled channel names
     */
    java.util.Set<String> getEnabledChannelNames();
    
    /**
     * Check if processor is ready for processing.
     * 
     * @return true if all enabled channels are initialized and ready
     */
    boolean isReady();
    
    /**
     * Save processor state to persistent storage.
     * This includes all channel states and configuration.
     */
    void saveState();
    
    /**
     * Load processor state from persistent storage.
     * This restores all channel states and configuration.
     */
    void loadState();
    
    /**
     * Shutdown processor and cleanup resources.
     * This method is called automatically by close().
     */
    void shutdown();
    
    /**
     * Close processor resources.
     * Equivalent to shutdown() but implements AutoCloseable.
     */
    @Override
    default void close() {
        shutdown();
    }
    
    /**
     * Callback interface for streaming text processing.
     */
    @FunctionalInterface
    interface ResultCallback {
        /**
         * Called when a text chunk has been processed.
         * 
         * @param result Processing result for the chunk
         */
        void onResult(ProcessingResult result);
        
        /**
         * Called when an error occurs during processing.
         * Default implementation logs the error.
         * 
         * @param error The error that occurred
         */
        default void onError(Throwable error) {
            System.err.println("Processing error: " + error.getMessage());
        }
        
        /**
         * Called when streaming is complete.
         * Default implementation does nothing.
         */
        default void onComplete() {
            // Default: no action
        }
    }
}