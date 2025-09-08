package com.hellblazer.art.nlp.core;

import java.time.Duration;
import java.time.Instant;
import java.util.Map;
import java.util.Collections;
import java.util.HashMap;
import java.util.Objects;

/**
 * Comprehensive processing statistics for NLP processor performance monitoring.
 * Thread-safe immutable statistics container.
 */
public final class ProcessingStats {
    private final Instant startTime;
    private final Instant lastUpdate;
    private final Duration uptime;
    private final long totalProcessed;
    private final long successfulProcessed;
    private final long failedProcessed;
    private final double averageProcessingTimeMs;
    private final double throughputPerSecond;
    private final Map<String, ChannelStats> channelStatistics;
    private final Map<String, Object> systemMetrics;
    private final ErrorStatistics errorStats;
    
    private ProcessingStats(Builder builder) {
        this.startTime = Objects.requireNonNull(builder.startTime, "startTime cannot be null");
        this.lastUpdate = Objects.requireNonNull(builder.lastUpdate, "lastUpdate cannot be null");
        this.uptime = Duration.between(startTime, lastUpdate);
        this.totalProcessed = builder.totalProcessed;
        this.successfulProcessed = builder.successfulProcessed;
        this.failedProcessed = builder.failedProcessed;
        this.averageProcessingTimeMs = builder.averageProcessingTimeMs;
        this.throughputPerSecond = builder.throughputPerSecond;
        this.channelStatistics = Collections.unmodifiableMap(new HashMap<>(builder.channelStatistics));
        this.systemMetrics = Collections.unmodifiableMap(new HashMap<>(builder.systemMetrics));
        this.errorStats = builder.errorStats;
    }
    
    /**
     * Get processor start time.
     */
    public Instant getStartTime() {
        return startTime;
    }
    
    /**
     * Get time of last statistics update.
     */
    public Instant getLastUpdate() {
        return lastUpdate;
    }
    
    /**
     * Get processor uptime.
     */
    public Duration getUptime() {
        return uptime;
    }
    
    /**
     * Get total number of texts processed.
     */
    public long getTotalProcessed() {
        return totalProcessed;
    }
    
    /**
     * Get number of successfully processed texts.
     */
    public long getSuccessfulProcessed() {
        return successfulProcessed;
    }
    
    /**
     * Get number of failed processing attempts.
     */
    public long getFailedProcessed() {
        return failedProcessed;
    }
    
    /**
     * Get average processing time in milliseconds.
     */
    public double getAverageProcessingTimeMs() {
        return averageProcessingTimeMs;
    }
    
    /**
     * Get processing throughput (texts per second).
     */
    public double getThroughputPerSecond() {
        return throughputPerSecond;
    }
    
    /**
     * Get success rate (0.0 to 1.0).
     */
    public double getSuccessRate() {
        return totalProcessed > 0 ? (double) successfulProcessed / totalProcessed : 0.0;
    }
    
    /**
     * Get failure rate (0.0 to 1.0).
     */
    public double getFailureRate() {
        return totalProcessed > 0 ? (double) failedProcessed / totalProcessed : 0.0;
    }
    
    /**
     * Get per-channel statistics.
     */
    public Map<String, ChannelStats> getChannelStatistics() {
        return channelStatistics;
    }
    
    /**
     * Get statistics for specific channel.
     */
    public ChannelStats getChannelStats(String channelName) {
        return channelStatistics.get(channelName);
    }
    
    /**
     * Get system metrics (memory, CPU, etc.).
     */
    public Map<String, Object> getSystemMetrics() {
        return systemMetrics;
    }
    
    /**
     * Get system metric value.
     */
    public Object getSystemMetric(String key) {
        return systemMetrics.get(key);
    }
    
    /**
     * Get error statistics.
     */
    public ErrorStatistics getErrorStatistics() {
        return errorStats;
    }
    
    @Override
    public String toString() {
        return String.format("ProcessingStats{processed=%d, success=%.2f%%, throughput=%.1f/s, uptime=%s}",
                           totalProcessed, getSuccessRate() * 100, throughputPerSecond, 
                           formatDuration(uptime));
    }
    
    private String formatDuration(Duration duration) {
        var hours = duration.toHours();
        var minutes = duration.toMinutesPart();
        var seconds = duration.toSecondsPart();
        return String.format("%02d:%02d:%02d", hours, minutes, seconds);
    }
    
    /**
     * Per-channel statistics.
     */
    public record ChannelStats(
        String channelName,
        long totalClassifications,
        long successfulClassifications,
        int currentCategories,
        double averageProcessingTimeMs,
        double successRate,
        Map<String, Object> channelSpecificMetrics
    ) {
        public ChannelStats {
            channelSpecificMetrics = channelSpecificMetrics != null ? 
                Collections.unmodifiableMap(new HashMap<>(channelSpecificMetrics)) : 
                Collections.emptyMap();
        }
        
        public double getFailureRate() {
            return 1.0 - successRate;
        }
        
        public Object getChannelMetric(String key) {
            return channelSpecificMetrics.get(key);
        }
    }
    
    /**
     * Error statistics.
     */
    public record ErrorStatistics(
        Map<String, Long> errorTypeCounts,
        String mostCommonErrorType,
        long totalErrors,
        Instant lastErrorTime,
        String lastErrorMessage
    ) {
        public ErrorStatistics {
            errorTypeCounts = errorTypeCounts != null ? 
                Collections.unmodifiableMap(new HashMap<>(errorTypeCounts)) : 
                Collections.emptyMap();
        }
        
        public long getErrorCount(String errorType) {
            return errorTypeCounts.getOrDefault(errorType, 0L);
        }
        
        public double getErrorRate(long totalProcessed) {
            return totalProcessed > 0 ? (double) totalErrors / totalProcessed : 0.0;
        }
    }
    
    /**
     * Builder for ProcessingStats.
     */
    public static class Builder {
        private Instant startTime = Instant.now();
        private Instant lastUpdate = Instant.now();
        private long totalProcessed = 0;
        private long successfulProcessed = 0;
        private long failedProcessed = 0;
        private double averageProcessingTimeMs = 0.0;
        private double throughputPerSecond = 0.0;
        private final Map<String, ChannelStats> channelStatistics = new HashMap<>();
        private final Map<String, Object> systemMetrics = new HashMap<>();
        private ErrorStatistics errorStats = new ErrorStatistics(
            Collections.emptyMap(), null, 0, null, null);
        
        /**
         * Set processor start time.
         */
        public Builder withStartTime(Instant startTime) {
            this.startTime = startTime;
            return this;
        }
        
        /**
         * Set last update time.
         */
        public Builder withLastUpdate(Instant lastUpdate) {
            this.lastUpdate = lastUpdate;
            return this;
        }
        
        /**
         * Set processing counts.
         */
        public Builder withProcessingCounts(long total, long successful, long failed) {
            this.totalProcessed = total;
            this.successfulProcessed = successful;
            this.failedProcessed = failed;
            return this;
        }
        
        /**
         * Set average processing time.
         */
        public Builder withAverageProcessingTime(double averageMs) {
            this.averageProcessingTimeMs = averageMs;
            return this;
        }
        
        /**
         * Set throughput.
         */
        public Builder withThroughput(double throughputPerSecond) {
            this.throughputPerSecond = throughputPerSecond;
            return this;
        }
        
        /**
         * Add channel statistics.
         */
        public Builder withChannelStats(String channelName, ChannelStats stats) {
            this.channelStatistics.put(Objects.requireNonNull(channelName), 
                                     Objects.requireNonNull(stats));
            return this;
        }
        
        /**
         * Add channel statistics.
         */
        public Builder withChannelStatistics(Map<String, ChannelStats> stats) {
            if (stats != null) {
                this.channelStatistics.putAll(stats);
            }
            return this;
        }
        
        /**
         * Add system metric.
         */
        public Builder withSystemMetric(String key, Object value) {
            this.systemMetrics.put(Objects.requireNonNull(key), value);
            return this;
        }
        
        /**
         * Add system metrics.
         */
        public Builder withSystemMetrics(Map<String, Object> metrics) {
            if (metrics != null) {
                this.systemMetrics.putAll(metrics);
            }
            return this;
        }
        
        /**
         * Set error statistics.
         */
        public Builder withErrorStatistics(ErrorStatistics errorStats) {
            this.errorStats = Objects.requireNonNull(errorStats);
            return this;
        }
        
        /**
         * Build immutable ProcessingStats.
         */
        public ProcessingStats build() {
            return new ProcessingStats(this);
        }
    }
    
    /**
     * Create new builder.
     */
    public static Builder builder() {
        return new Builder();
    }
    
    /**
     * Create empty statistics.
     */
    public static ProcessingStats empty() {
        return new Builder().build();
    }
}