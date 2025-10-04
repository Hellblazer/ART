package com.hellblazer.art.cortical.temporal;

import java.util.ArrayList;
import java.util.List;

/**
 * Coordinates temporal processing pipeline: WorkingMemory → MaskingField.
 *
 * Implements LIST PARSE temporal chunking from Kazerounian & Grossberg (2014):
 * 1. Items stored in working memory with primacy gradient
 * 2. Multi-scale masking field forms chunks (item/chunk/list scales)
 * 3. Chunks output as temporally organized representations
 *
 * Classic examples:
 * - Phone numbers: "555-1234" → chunks as [555][1234]
 * - Sequences: "ABCDEFG" → chunks by similarity/grouping
 *
 * @author Hal Hildebrand
 */
public class TemporalProcessor {
    private final WorkingMemory workingMemory;
    private final MaskingField maskingField;
    private final double integrationTimeStep;

    public TemporalProcessor(WorkingMemoryParameters wmParams,
                             MaskingFieldParameters mfParams,
                             double timeStep) {
        this.workingMemory = new WorkingMemory(wmParams);
        this.maskingField = new MaskingField(mfParams);
        this.integrationTimeStep = timeStep;
    }

    /**
     * Convenience constructor with standard time step (0.01).
     */
    public TemporalProcessor(WorkingMemoryParameters wmParams,
                             MaskingFieldParameters mfParams) {
        this(wmParams, mfParams, 0.01);
    }

    /**
     * Process single item through temporal pipeline.
     *
     * @param item Input pattern
     * @return Temporal processing result
     */
    public TemporalResult processItem(double[] item) {
        // Stage 1: Add to working memory (primacy gradient applied)
        workingMemory.storeItem(item, integrationTimeStep);
        var wmState = workingMemory.getState();

        // Stage 2: Form chunks via masking field
        // Use the combined pattern from working memory as input to masking field
        var wmPattern = wmState.getCombinedPattern();
        var mfState = maskingField.update(wmPattern, integrationTimeStep);

        return new TemporalResult(wmState, mfState, maskingField.getActiveChunks());
    }

    /**
     * Process sequence of items.
     */
    public List<TemporalResult> processSequence(List<double[]> items) {
        var results = new ArrayList<TemporalResult>();
        for (var item : items) {
            results.add(processItem(item));
        }
        return results;
    }

    /**
     * Get current active chunks.
     */
    public List<ListChunk> getActiveChunks() {
        return maskingField.getActiveChunks();
    }

    /**
     * Get working memory state.
     */
    public WorkingMemoryState getWorkingMemoryState() {
        return workingMemory.getState();
    }

    /**
     * Get masking field state.
     */
    public MaskingFieldState getMaskingFieldState() {
        return maskingField.getState();
    }

    /**
     * Reset temporal processor state.
     */
    public void reset() {
        workingMemory.reset();
        maskingField.reset();
    }

    /**
     * Result of temporal processing.
     */
    public record TemporalResult(
        WorkingMemoryState workingMemoryState,
        MaskingFieldState maskingFieldState,
        List<ListChunk> activeChunks
    ) {
        /**
         * Get number of active chunks.
         */
        public int chunkCount() {
            return activeChunks.size();
        }

        /**
         * Check if any chunks were formed.
         */
        public boolean hasChunks() {
            return !activeChunks.isEmpty();
        }

        /**
         * Get total items across all chunks.
         */
        public int totalChunkedItems() {
            return activeChunks.stream()
                .mapToInt(ListChunk::size)
                .sum();
        }
    }
}
