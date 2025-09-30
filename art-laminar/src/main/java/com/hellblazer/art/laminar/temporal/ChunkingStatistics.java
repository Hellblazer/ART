package com.hellblazer.art.laminar.temporal;

/**
 * Statistics about temporal chunking behavior.
 *
 * @author Hal Hildebrand
 */
public record ChunkingStatistics(
    int totalChunks,
    int activeChunks,
    double averageChunkSize,
    double averageCoherence,
    int totalChunkedItems,
    int historySize,
    double chunkFormationRate  // Chunks per second
) {

    /**
     * Create statistics from chunking state.
     */
    public static ChunkingStatistics from(ChunkingState state, double timeWindow) {
        var chunks = state.getActiveChunks();

        int totalChunks = chunks.size();
        int activeChunks = (int) chunks.stream()
            .filter(c -> c.isActive(0.1))
            .count();

        double avgSize = state.getAverageChunkSize();

        double avgCoherence = chunks.isEmpty() ? 0.0 :
            chunks.stream()
                .mapToDouble(TemporalChunk::getCoherence)
                .average()
                .orElse(0.0);

        int totalItems = state.getTotalChunkedItems();
        int historySize = state.getHistorySize();

        double formationRate = timeWindow > 0 ?
            totalChunks / timeWindow : 0.0;

        return new ChunkingStatistics(
            totalChunks,
            activeChunks,
            avgSize,
            avgCoherence,
            totalItems,
            historySize,
            formationRate
        );
    }

    /**
     * Empty statistics.
     */
    public static ChunkingStatistics empty() {
        return new ChunkingStatistics(0, 0, 0.0, 0.0, 0, 0, 0.0);
    }
}