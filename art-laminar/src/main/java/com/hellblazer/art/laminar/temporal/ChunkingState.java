package com.hellblazer.art.laminar.temporal;

import com.hellblazer.art.core.Pattern;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

/**
 * State representation for temporal chunking in a layer.
 * Tracks activation history and chunk formation dynamics.
 *
 * @author Hal Hildebrand
 */
public class ChunkingState {

    private final Deque<ActivationSnapshot> activationHistory;
    private final List<TemporalChunk> activeChunks;
    private final int maxHistorySize;
    private double currentTime;
    private int nextChunkId;

    public ChunkingState(int maxHistorySize) {
        this.activationHistory = new ArrayDeque<>(maxHistorySize);
        this.activeChunks = new ArrayList<>();
        this.maxHistorySize = maxHistorySize;
        this.currentTime = 0.0;
        this.nextChunkId = 0;
    }

    /**
     * Add new activation to history.
     */
    public void addActivation(Pattern pattern, double activation, double time) {
        // Add to history
        activationHistory.addLast(new ActivationSnapshot(pattern, activation, time));

        // Maintain maximum history size
        while (activationHistory.size() > maxHistorySize) {
            activationHistory.removeFirst();
        }

        currentTime = time;
    }

    /**
     * Get recent activation history.
     */
    public List<ActivationSnapshot> getRecentHistory(int count) {
        int size = Math.min(count, activationHistory.size());
        List<ActivationSnapshot> recent = new ArrayList<>(size);

        int startIndex = activationHistory.size() - size;
        int index = 0;

        for (var snapshot : activationHistory) {
            if (index >= startIndex) {
                recent.add(snapshot);
            }
            index++;
        }

        return recent;
    }

    /**
     * Add a newly formed chunk.
     */
    public void addChunk(TemporalChunk chunk) {
        activeChunks.add(chunk);
    }

    /**
     * Get all active chunks.
     */
    public List<TemporalChunk> getActiveChunks() {
        return new ArrayList<>(activeChunks);
    }

    /**
     * Remove inactive chunks (strength below threshold).
     */
    public void pruneInactiveChunks(double threshold) {
        activeChunks.removeIf(chunk -> !chunk.isActive(threshold));
    }

    /**
     * Apply decay to all chunks.
     */
    public void decayChunks(double decayRate) {
        for (var chunk : activeChunks) {
            chunk.decay(decayRate, currentTime);
        }
    }

    /**
     * Get total number of items across all chunks.
     */
    public int getTotalChunkedItems() {
        return activeChunks.stream()
            .mapToInt(TemporalChunk::size)
            .sum();
    }

    /**
     * Get average chunk size.
     */
    public double getAverageChunkSize() {
        if (activeChunks.isEmpty()) return 0.0;

        return (double) getTotalChunkedItems() / activeChunks.size();
    }

    /**
     * Get next chunk ID.
     */
    public int getNextChunkId() {
        return nextChunkId++;
    }

    /**
     * Clear all state.
     */
    public void reset() {
        activationHistory.clear();
        activeChunks.clear();
        currentTime = 0.0;
        nextChunkId = 0;
    }

    /**
     * Get current time.
     */
    public double getCurrentTime() {
        return currentTime;
    }

    /**
     * Get history size.
     */
    public int getHistorySize() {
        return activationHistory.size();
    }

    /**
     * Snapshot of layer activation at a point in time.
     */
    public record ActivationSnapshot(
        Pattern pattern,
        double activation,
        double time
    ) {}
}