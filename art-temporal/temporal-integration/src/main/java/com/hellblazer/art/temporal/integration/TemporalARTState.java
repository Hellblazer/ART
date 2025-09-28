package com.hellblazer.art.temporal.integration;

import com.hellblazer.art.temporal.memory.WorkingMemoryState;
import com.hellblazer.art.temporal.masking.MaskingFieldState;

import java.util.List;

/**
 * Complete state of the temporal ART system.
 */
public record TemporalARTState(
    WorkingMemoryState workingMemoryState,
    MaskingFieldState maskingFieldState,
    List<TemporalCategory> categories,
    double currentTime,
    boolean learningEnabled
) {

    /**
     * Get total number of stored items across all components.
     */
    public int getTotalItemCount() {
        return workingMemoryState.getItemCount() +
               maskingFieldState.getItemNodeCount();
    }

    /**
     * Get number of active categories.
     */
    public int getActiveCategoryCount() {
        return (int) categories.stream()
            .filter(c -> c.getStrength() > 0.1)
            .count();
    }

    /**
     * Check if system is in resonance.
     */
    public boolean isInResonance() {
        return !maskingFieldState.getWinningNodes().isEmpty();
    }

    /**
     * Get memory utilization ratio.
     */
    public double getMemoryUtilization() {
        int maxCapacity = workingMemoryState.getCapacity();
        int currentUsage = workingMemoryState.getItemCount();
        return (double) currentUsage / maxCapacity;
    }

    /**
     * Get chunking efficiency.
     */
    public double getChunkingEfficiency() {
        int totalItems = maskingFieldState.getItemNodeCount();
        int numChunks = maskingFieldState.getChunkCount();
        if (numChunks == 0) return 0.0;
        return (double) totalItems / numChunks;
    }

    /**
     * Get system statistics summary.
     */
    public String getSummary() {
        return String.format(
            "TemporalART[time=%.3f, categories=%d, wm_items=%d, chunks=%d, resonance=%s, learning=%s]",
            currentTime,
            categories.size(),
            workingMemoryState.getItemCount(),
            maskingFieldState.getChunkCount(),
            isInResonance() ? "YES" : "NO",
            learningEnabled ? "ON" : "OFF"
        );
    }
}