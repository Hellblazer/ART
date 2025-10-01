package com.hellblazer.art.cortical.temporal;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Immutable state representation for the masking field.
 * Tracks activations of item nodes and list chunks.
 *
 * Part of the LIST PARSE multi-scale temporal chunking system
 * (Kazerounian & Grossberg, 2014).
 *
 * @author Hal Hildebrand
 */
public record MaskingFieldState(
    double[] itemActivations,
    double[] chunkActivations,
    List<Integer> winningNodes,
    int activeItemCount
) {
    /**
     * Canonical constructor with defensive copies.
     */
    public MaskingFieldState {
        itemActivations = itemActivations.clone();
        chunkActivations = chunkActivations.clone();
        winningNodes = new ArrayList<>(winningNodes);
    }

    /**
     * Create initial empty state.
     */
    public static MaskingFieldState create(int maxItemNodes) {
        return new MaskingFieldState(
            new double[maxItemNodes],
            new double[maxItemNodes / 2], // Chunks are typically fewer
            new ArrayList<>(),
            0
        );
    }

    /**
     * Create new state with updated item activations.
     */
    public MaskingFieldState withItemActivations(double[] newActivations) {
        return new MaskingFieldState(
            newActivations,
            chunkActivations,
            winningNodes,
            activeItemCount
        );
    }

    /**
     * Create new state with updated chunk activations.
     */
    public MaskingFieldState withChunkActivations(double[] newActivations) {
        return new MaskingFieldState(
            itemActivations,
            newActivations,
            winningNodes,
            activeItemCount
        );
    }

    /**
     * Create new state with updated winning nodes.
     */
    public MaskingFieldState withWinningNodes(List<Integer> newWinners) {
        return new MaskingFieldState(
            itemActivations,
            chunkActivations,
            newWinners,
            activeItemCount
        );
    }

    /**
     * Create new state with cleared winners.
     */
    public MaskingFieldState withClearedWinners() {
        return new MaskingFieldState(
            itemActivations,
            chunkActivations,
            new ArrayList<>(),
            activeItemCount
        );
    }

    /**
     * Create new state with updated active item count.
     */
    public MaskingFieldState withActiveItemCount(int count) {
        return new MaskingFieldState(
            itemActivations,
            chunkActivations,
            winningNodes,
            Math.max(0, Math.min(count, itemActivations.length))
        );
    }

    /**
     * Create new state with incremented active item count.
     */
    public MaskingFieldState withIncrementedItemCount() {
        return withActiveItemCount(activeItemCount + 1);
    }

    /**
     * Create new state with decremented active item count.
     */
    public MaskingFieldState withDecrementedItemCount() {
        return withActiveItemCount(activeItemCount - 1);
    }

    /**
     * Get item activation at index (safe access).
     */
    public double getItemActivation(int index) {
        if (index >= 0 && index < itemActivations.length) {
            return itemActivations[index];
        }
        return 0.0;
    }

    /**
     * Get chunk activation at index (safe access).
     */
    public double getChunkActivation(int index) {
        if (index >= 0 && index < chunkActivations.length) {
            return chunkActivations[index];
        }
        return 0.0;
    }

    /**
     * Get total activation across all items.
     */
    public double getTotalActivation() {
        var total = 0.0;
        for (var activation : itemActivations) {
            total += activation;
        }
        return total;
    }

    /**
     * Compute contrast enhancement metric.
     */
    public double computeContrast() {
        if (itemActivations.length == 0) return 0.0;

        var max = 0.0;
        var mean = 0.0;

        for (var activation : itemActivations) {
            max = Math.max(max, activation);
            mean += activation;
        }

        mean /= itemActivations.length;

        if (mean == 0.0) return 0.0;
        return (max - mean) / mean;
    }

    /**
     * Get maximum capacity of item nodes.
     */
    public int getItemNodeCapacity() {
        return itemActivations.length;
    }

    /**
     * Get count of chunks.
     */
    public int getChunkCount() {
        return chunkActivations.length;
    }

    /**
     * Get activation statistics.
     */
    public ActivationStatistics getStatistics() {
        var maxItem = 0.0;
        var meanItem = 0.0;
        var activeItems = 0;

        for (var activation : itemActivations) {
            maxItem = Math.max(maxItem, activation);
            meanItem += activation;
            if (activation > 0.01) activeItems++;
        }
        meanItem /= itemActivations.length;

        var maxChunk = 0.0;
        var meanChunk = 0.0;
        var activeChunks = 0;

        for (var activation : chunkActivations) {
            maxChunk = Math.max(maxChunk, activation);
            meanChunk += activation;
            if (activation > 0.01) activeChunks++;
        }
        meanChunk /= chunkActivations.length;

        return new ActivationStatistics(
            maxItem, meanItem, activeItems,
            maxChunk, meanChunk, activeChunks,
            computeContrast()
        );
    }

    /**
     * Override to provide defensive copy of item activations.
     */
    @Override
    public double[] itemActivations() {
        return itemActivations.clone();
    }

    /**
     * Override to provide defensive copy of chunk activations.
     */
    @Override
    public double[] chunkActivations() {
        return chunkActivations.clone();
    }

    /**
     * Override to provide defensive copy of winning nodes.
     */
    @Override
    public List<Integer> winningNodes() {
        return new ArrayList<>(winningNodes);
    }

    /**
     * Override equals to handle array comparison.
     */
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof MaskingFieldState other)) return false;
        return Arrays.equals(itemActivations, other.itemActivations) &&
               Arrays.equals(chunkActivations, other.chunkActivations) &&
               winningNodes.equals(other.winningNodes) &&
               activeItemCount == other.activeItemCount;
    }

    /**
     * Override hashCode to handle array hashing.
     */
    @Override
    public int hashCode() {
        var result = Arrays.hashCode(itemActivations);
        result = 31 * result + Arrays.hashCode(chunkActivations);
        result = 31 * result + winningNodes.hashCode();
        result = 31 * result + activeItemCount;
        return result;
    }

    /**
     * Activation statistics record.
     */
    public record ActivationStatistics(
        double maxItemActivation,
        double meanItemActivation,
        int activeItemCount,
        double maxChunkActivation,
        double meanChunkActivation,
        int activeChunkCount,
        double contrast
    ) {}
}
