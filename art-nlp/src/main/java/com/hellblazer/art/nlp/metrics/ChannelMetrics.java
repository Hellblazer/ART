package com.hellblazer.art.nlp.metrics;

import java.time.Instant;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Thread-safe metrics collection for channels.
 * Tracks performance and operational statistics.
 */
public final class ChannelMetrics {
    private final String channelName;
    private final AtomicInteger totalClassifications = new AtomicInteger(0);
    private final AtomicInteger categoriesCreated = new AtomicInteger(0);
    private final AtomicLong totalProcessingTimeMs = new AtomicLong(0);
    private final AtomicLong maxProcessingTimeMs = new AtomicLong(0);
    private final AtomicInteger errors = new AtomicInteger(0);
    private final Instant startTime;
    
    // Category management stats
    private final AtomicInteger currentCategoryCount = new AtomicInteger(0);
    private final AtomicInteger maxCategoriesReached = new AtomicInteger(0);
    private final AtomicInteger categoriesPruned = new AtomicInteger(0);
    
    public ChannelMetrics(String channelName) {
        this.channelName = channelName;
        this.startTime = Instant.now();
    }
    
    public String getChannelName() {
        return channelName;
    }
    
    public void recordClassification(long processingTimeMs) {
        totalClassifications.incrementAndGet();
        totalProcessingTimeMs.addAndGet(processingTimeMs);
        maxProcessingTimeMs.updateAndGet(current -> Math.max(current, processingTimeMs));
    }
    
    public void recordCategoryCreated() {
        categoriesCreated.incrementAndGet();
        var newCount = currentCategoryCount.incrementAndGet();
        maxCategoriesReached.updateAndGet(current -> Math.max(current, newCount));
    }
    
    public void recordCategoryPruned() {
        categoriesPruned.incrementAndGet();
        currentCategoryCount.decrementAndGet();
    }
    
    public void recordError() {
        errors.incrementAndGet();
    }
    
    public void updateCurrentCategoryCount(int count) {
        currentCategoryCount.set(count);
        maxCategoriesReached.updateAndGet(current -> Math.max(current, count));
    }
    
    public int getTotalClassifications() {
        return totalClassifications.get();
    }
    
    public int getCategoriesCreated() {
        return categoriesCreated.get();
    }
    
    public long getTotalProcessingTimeMs() {
        return totalProcessingTimeMs.get();
    }
    
    public long getMaxProcessingTimeMs() {
        return maxProcessingTimeMs.get();
    }
    
    public double getAverageProcessingTimeMs() {
        var total = totalClassifications.get();
        if (total == 0) return 0.0;
        return (double) totalProcessingTimeMs.get() / total;
    }
    
    public int getErrors() {
        return errors.get();
    }
    
    public int getCurrentCategoryCount() {
        return currentCategoryCount.get();
    }
    
    public int getMaxCategoriesReached() {
        return maxCategoriesReached.get();
    }
    
    public int getCategoriesPruned() {
        return categoriesPruned.get();
    }
    
    public double getClassificationsPerSecond() {
        var uptimeSeconds = java.time.Duration.between(startTime, Instant.now()).toSeconds();
        if (uptimeSeconds == 0) return 0.0;
        return (double) totalClassifications.get() / uptimeSeconds;
    }
    
    public double getErrorRate() {
        var total = totalClassifications.get();
        if (total == 0) return 0.0;
        return (double) errors.get() / total;
    }
    
    public Instant getStartTime() {
        return startTime;
    }
    
    /**
     * Reset all metrics (for testing or restart scenarios).
     */
    public void reset() {
        totalClassifications.set(0);
        categoriesCreated.set(0);
        totalProcessingTimeMs.set(0);
        maxProcessingTimeMs.set(0);
        errors.set(0);
        currentCategoryCount.set(0);
        maxCategoriesReached.set(0);
        categoriesPruned.set(0);
    }
    
    @Override
    public String toString() {
        return String.format("ChannelMetrics{channel='%s', classifications=%d, categories=%d/%d, " +
                           "avgTime=%.2fms, errors=%d, rate=%.1f/sec}",
                           channelName, getTotalClassifications(), getCurrentCategoryCount(), 
                           getMaxCategoriesReached(), getAverageProcessingTimeMs(), 
                           getErrors(), getClassificationsPerSecond());
    }
}