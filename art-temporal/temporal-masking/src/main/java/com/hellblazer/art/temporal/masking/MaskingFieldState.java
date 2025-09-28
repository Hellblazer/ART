package com.hellblazer.art.temporal.masking;

import java.util.ArrayList;
import java.util.List;

/**
 * State representation for the masking field.
 * Tracks activations of item nodes and list chunks.
 */
public class MaskingFieldState {
    private double[] itemActivations;
    private double[] chunkActivations;
    private List<Integer> winningNodes;
    private double totalActivation;
    private int activeItemCount;

    public MaskingFieldState(int maxItemNodes) {
        this.itemActivations = new double[maxItemNodes];
        this.chunkActivations = new double[maxItemNodes / 2]; // Chunks are typically fewer
        this.winningNodes = new ArrayList<>();
        this.totalActivation = 0.0;
        this.activeItemCount = 0;  // Initialize to 0, not maxItemNodes
    }

    /**
     * Copy constructor.
     */
    private MaskingFieldState(MaskingFieldState other) {
        this.itemActivations = other.itemActivations.clone();
        this.chunkActivations = other.chunkActivations.clone();
        this.winningNodes = new ArrayList<>(other.winningNodes);
        this.totalActivation = other.totalActivation;
        this.activeItemCount = other.activeItemCount;
    }

    /**
     * Update item activations.
     */
    public void setItemActivations(double[] activations) {
        System.arraycopy(activations, 0, itemActivations, 0,
                        Math.min(activations.length, itemActivations.length));
        updateTotalActivation();
    }

    /**
     * Update chunk activations.
     */
    public void setChunkActivations(double[] activations) {
        System.arraycopy(activations, 0, chunkActivations, 0,
                        Math.min(activations.length, chunkActivations.length));
    }

    /**
     * Set winning nodes after competition.
     */
    public void setWinningNodes(List<Integer> winners) {
        this.winningNodes = new ArrayList<>(winners);
    }

    /**
     * Clear winning nodes.
     */
    public void clearWinners() {
        winningNodes.clear();
    }

    /**
     * Update total activation.
     */
    private void updateTotalActivation() {
        totalActivation = 0.0;
        for (double activation : itemActivations) {
            totalActivation += activation;
        }
    }

    /**
     * Get item activation at index.
     */
    public double getItemActivation(int index) {
        if (index >= 0 && index < itemActivations.length) {
            return itemActivations[index];
        }
        return 0.0;
    }

    /**
     * Get chunk activation at index.
     */
    public double getChunkActivation(int index) {
        if (index >= 0 && index < chunkActivations.length) {
            return chunkActivations[index];
        }
        return 0.0;
    }

    /**
     * Compute contrast enhancement metric.
     */
    public double computeContrast() {
        if (itemActivations.length == 0) return 0.0;

        double max = 0.0;
        double mean = 0.0;

        for (double activation : itemActivations) {
            max = Math.max(max, activation);
            mean += activation;
        }

        mean /= itemActivations.length;

        if (mean == 0.0) return 0.0;
        return (max - mean) / mean;
    }

    /**
     * Create a copy of this state.
     */
    public MaskingFieldState copy() {
        return new MaskingFieldState(this);
    }

    // Getters
    public double[] getItemActivations() {
        return itemActivations.clone();
    }

    public double[] getChunkActivations() {
        return chunkActivations.clone();
    }

    public List<Integer> getWinningNodes() {
        return new ArrayList<>(winningNodes);
    }

    public double getTotalActivation() {
        return totalActivation;
    }

    /**
     * Get count of active item nodes.
     */
    public int getItemNodeCount() {
        return activeItemCount;
    }

    /**
     * Get count of active item nodes based on actual storage.
     * This is a fallback method that should be synchronized with the actual node count.
     */
    public int getActualItemNodeCount(java.util.List<?> actualItemNodes) {
        return actualItemNodes.size();
    }

    /**
     * Set the count of active item nodes.
     */
    public void setActiveItemCount(int count) {
        this.activeItemCount = Math.max(0, Math.min(count, itemActivations.length));
    }

    /**
     * Increment the active item count.
     */
    public void incrementActiveItemCount() {
        if (activeItemCount < itemActivations.length) {
            activeItemCount++;
        }
    }

    /**
     * Decrement the active item count.
     */
    public void decrementActiveItemCount() {
        if (activeItemCount > 0) {
            activeItemCount--;
        }
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
        double maxItem = 0.0;
        double meanItem = 0.0;
        int activeItems = 0;

        for (double activation : itemActivations) {
            maxItem = Math.max(maxItem, activation);
            meanItem += activation;
            if (activation > 0.01) activeItems++;
        }
        meanItem /= itemActivations.length;

        double maxChunk = 0.0;
        double meanChunk = 0.0;
        int activeChunks = 0;

        for (double activation : chunkActivations) {
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
     * Activation statistics.
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