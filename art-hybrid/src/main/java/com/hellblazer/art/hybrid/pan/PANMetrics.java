package com.hellblazer.art.hybrid.pan;

/**
 * Performance metrics for PAN algorithm validation
 */
public record PANMetrics(
    int totalSamples,
    int correctPredictions,
    int categoryCount,
    long totalProcessingTimeNs,
    long memoryUsageBytes,
    boolean gpuEnabled
) {

    public float getAccuracy() {
        return totalSamples > 0 ? (float) correctPredictions / totalSamples : 0;
    }

    public double getAverageProcessingTimeMs() {
        return totalSamples > 0 ? (totalProcessingTimeNs / 1_000_000.0) / totalSamples : 0;
    }

    public String getMemoryUsageMB() {
        return String.format("%.2f MB", memoryUsageBytes / (1024.0 * 1024.0));
    }

    /**
     * Check if metrics meet paper's claimed performance
     */
    public boolean meetsTargetPerformance(float targetAccuracy, int maxCategories) {
        return getAccuracy() >= targetAccuracy && categoryCount <= maxCategories;
    }

    @Override
    public String toString() {
        return String.format(
            "PANMetrics[accuracy=%.1f%%, categories=%d, avgTime=%.2fms, memory=%s, gpu=%b]",
            getAccuracy() * 100,
            categoryCount,
            getAverageProcessingTimeMs(),
            getMemoryUsageMB(),
            gpuEnabled
        );
    }
}