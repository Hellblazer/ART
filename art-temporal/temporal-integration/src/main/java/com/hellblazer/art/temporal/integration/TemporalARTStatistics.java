package com.hellblazer.art.temporal.integration;

/**
 * Statistical information about temporal ART performance.
 */
public record TemporalARTStatistics(
    int categoryCount,
    int workingMemoryItems,
    int chunkCount,
    int itemNodeCount,
    double averageChunkSize,
    double compressionRatio
) {

    /**
     * Get memory efficiency metric.
     */
    public double getMemoryEfficiency() {
        if (workingMemoryItems == 0) return 1.0;
        return (double) categoryCount / workingMemoryItems;
    }

    /**
     * Get chunking effectiveness.
     */
    public double getChunkingEffectiveness() {
        if (itemNodeCount == 0) return 0.0;
        if (chunkCount == 0) return 0.0;
        return averageChunkSize / 7.0;  // Normalized to Miller's magic number
    }

    /**
     * Get overall system efficiency.
     */
    public double getOverallEfficiency() {
        return (getMemoryEfficiency() + getChunkingEffectiveness() + compressionRatio / 10.0) / 3.0;
    }

    @Override
    public String toString() {
        return String.format(
            "Statistics[categories=%d, wm=%d, chunks=%d, nodes=%d, avgChunk=%.2f, compression=%.2f, efficiency=%.2f]",
            categoryCount,
            workingMemoryItems,
            chunkCount,
            itemNodeCount,
            averageChunkSize,
            compressionRatio,
            getOverallEfficiency()
        );
    }
}