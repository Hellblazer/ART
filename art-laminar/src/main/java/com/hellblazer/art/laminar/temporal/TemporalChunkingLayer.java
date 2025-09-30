package com.hellblazer.art.laminar.temporal;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.core.Layer;

import java.util.List;

/**
 * Interface for layers with temporal chunking capabilities.
 * Integrates MaskingField-style chunking into laminar layer processing.
 *
 * Based on Grossberg & Kazerounian (2016) LIST PARSE model for
 * multi-scale temporal grouping and working memory organization.
 *
 * @author Hal Hildebrand
 */
public interface TemporalChunkingLayer extends Layer {

    /**
     * Get the current temporal chunks formed in this layer.
     *
     * @return list of active temporal chunks
     */
    List<TemporalChunk> getTemporalChunks();

    /**
     * Process input with temporal chunking.
     * Groups sequential patterns into coherent temporal chunks.
     *
     * @param input the input pattern
     * @param timeStep the time step for this processing cycle
     * @return processed pattern with chunking applied
     */
    Pattern processWithChunking(Pattern input, double timeStep);

    /**
     * Get the chunking state for this layer.
     *
     * @return current chunking state
     */
    ChunkingState getChunkingState();

    /**
     * Update temporal chunking dynamics.
     * Evolves chunk formation and decay over time.
     *
     * @param timeStep the integration time step
     */
    void updateChunkingDynamics(double timeStep);

    /**
     * Check if a new chunk should be formed based on current state.
     *
     * @return true if chunk formation conditions are met
     */
    boolean shouldFormChunk();

    /**
     * Form a new temporal chunk from recent activations.
     *
     * @return the newly formed chunk, or null if formation failed
     */
    TemporalChunk formChunk();

    /**
     * Reset all temporal chunks and chunking state.
     */
    void resetChunking();

    /**
     * Get the temporal context accumulated across chunks.
     * This provides memory of recent processing history.
     *
     * @return temporal context pattern
     */
    Pattern getTemporalContext();

    /**
     * Set chunking parameters for this layer.
     *
     * @param params chunking parameters
     */
    void setChunkingParameters(ChunkingParameters params);

    /**
     * Get statistics about chunking behavior.
     *
     * @return chunking statistics
     */
    ChunkingStatistics getChunkingStatistics();

    /**
     * Get complete layer state including current activation and temporal context.
     * Combines instantaneous activation with chunked temporal history.
     *
     * @return layer state with both activation and temporal context
     */
    LayerState getLayerState();

    /**
     * Set the context weight for combining activation with temporal context.
     *
     * @param weight Weight for temporal context (0.0 = only activation, 1.0 = only context)
     */
    void setContextWeight(double weight);

    /**
     * Get the current context weight.
     *
     * @return context weight
     */
    double getContextWeight();
}