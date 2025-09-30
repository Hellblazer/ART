package com.hellblazer.art.laminar.temporal;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.temporal.memory.WorkingMemory;

import java.util.ArrayList;
import java.util.List;

/**
 * Bridges WorkingMemory's primacy gradient with layer temporal chunking.
 *
 * Integrates two complementary temporal mechanisms:
 * 1. WorkingMemory: Position-dependent primacy gradient (early items stronger)
 * 2. Layer Chunking: Coherence-based grouping of sequential patterns
 *
 * This creates a two-level temporal representation:
 * - Item level: Individual patterns with primacy weighting
 * - Chunk level: Coherent groups of items
 *
 * Based on Grossberg & Kazerounian (2016) integration of
 * STORE 2 (working memory) and LIST PARSE (chunking).
 *
 * @author Hal Hildebrand
 */
public class WorkingMemoryLayerBridge {

    private final WorkingMemory workingMemory;
    private final TemporalChunkingLayer chunkingLayer;
    private final double itemDuration;

    private List<Pattern> processedSequence;
    private int sequencePosition;

    public WorkingMemoryLayerBridge(WorkingMemory workingMemory,
                                    TemporalChunkingLayer chunkingLayer,
                                    double itemDuration) {
        this.workingMemory = workingMemory;
        this.chunkingLayer = chunkingLayer;
        this.itemDuration = itemDuration;
        this.processedSequence = new ArrayList<>();
        this.sequencePosition = 0;
    }

    /**
     * Process a pattern through both working memory and layer chunking.
     *
     * Flow:
     * 1. Store in WorkingMemory (applies primacy gradient)
     * 2. Process through layer (chunking dynamics)
     * 3. Update temporal context from chunks
     *
     * @param input Input pattern
     * @return Processed pattern with both primacy and chunking effects
     */
    public Pattern processItem(Pattern input) {
        // Convert to array for WorkingMemory
        double[] inputArray = patternToArray(input);

        // Store in working memory (applies primacy gradient)
        workingMemory.storeItem(inputArray, itemDuration);

        // Process through chunking layer
        var processed = chunkingLayer.processWithChunking(input, itemDuration);

        // Track sequence
        processedSequence.add(processed);
        sequencePosition++;

        return processed;
    }

    /**
     * Process a complete sequence with integrated dynamics.
     *
     * @param patterns Sequence of patterns
     * @return Final layer state after processing
     */
    public LayerState processSequence(List<Pattern> patterns) {
        reset();

        for (var pattern : patterns) {
            processItem(pattern);
        }

        return getLayerState();
    }

    /**
     * Get working memory activations with primacy gradient.
     * These reflect the STORE 2 model's position-dependent encoding.
     */
    public double[] getWorkingMemoryActivations() {
        return workingMemory.getState().getPrimacyWeights();
    }

    /**
     * Get temporal chunks formed by the layer.
     * These reflect LIST PARSE model's coherence-based grouping.
     */
    public List<TemporalChunk> getTemporalChunks() {
        return chunkingLayer.getTemporalChunks();
    }

    /**
     * Get integrated layer state combining:
     * - Current activation
     * - Temporal context from chunks
     * - Primacy gradient from working memory
     */
    public LayerState getLayerState() {
        return chunkingLayer.getLayerState();
    }

    /**
     * Get processed sequence history.
     */
    public List<Pattern> getProcessedSequence() {
        return new ArrayList<>(processedSequence);
    }

    /**
     * Create a combined representation that includes both
     * primacy-weighted activations and chunk context.
     *
     * @return Pattern combining working memory and chunk information
     */
    public Pattern getCombinedRepresentation() {
        // Get working memory activations (primacy gradient)
        double[] wmActivations = getWorkingMemoryActivations();

        // Get temporal context from chunks
        var context = chunkingLayer.getTemporalContext();

        // Combine: 50% working memory primacy, 50% chunk context
        int dimension = Math.min(wmActivations.length, context.dimension());
        double[] combined = new double[dimension];

        for (int i = 0; i < dimension; i++) {
            combined[i] = 0.5 * (i < wmActivations.length ? wmActivations[i] : 0.0) +
                         0.5 * context.get(i);
        }

        return new DenseVector(combined);
    }

    /**
     * Reset both working memory and chunking state.
     */
    public void reset() {
        workingMemory.reset();
        chunkingLayer.resetChunking();
        processedSequence.clear();
        sequencePosition = 0;
    }

    /**
     * Statistics combining both working memory and chunking metrics.
     */
    public record IntegratedStatistics(
        double[] primacyWeights,
        double[] recencyWeights,
        int chunksFormed,
        int activeChunks,
        double averageChunkSize,
        double averageCoherence,
        int sequenceLength,
        double primacyGradientStrength
    ) {
        public static IntegratedStatistics from(WorkingMemoryLayerBridge bridge) {
            var wmState = bridge.workingMemory.getState();
            var primacyWeights = wmState.getPrimacyWeights();
            var recencyWeights = wmState.getRecencyWeights();
            var chunkStats = bridge.chunkingLayer.getChunkingStatistics();

            // Compute primacy gradient strength (difference between first and last)
            double primacyStrength = 0.0;
            if (primacyWeights.length >= 2) {
                primacyStrength = primacyWeights[0] - primacyWeights[primacyWeights.length - 1];
            }

            return new IntegratedStatistics(
                primacyWeights,
                recencyWeights,
                chunkStats.totalChunks(),
                chunkStats.activeChunks(),
                chunkStats.averageChunkSize(),
                chunkStats.averageCoherence(),
                bridge.sequencePosition,
                primacyStrength
            );
        }
    }

    public IntegratedStatistics getStatistics() {
        return IntegratedStatistics.from(this);
    }

    // Helper methods

    private double[] patternToArray(Pattern pattern) {
        double[] array = new double[pattern.dimension()];
        for (int i = 0; i < pattern.dimension(); i++) {
            array[i] = pattern.get(i);
        }
        return array;
    }

    // Getters

    public WorkingMemory getWorkingMemory() {
        return workingMemory;
    }

    public TemporalChunkingLayer getChunkingLayer() {
        return chunkingLayer;
    }

    public int getSequencePosition() {
        return sequencePosition;
    }
}