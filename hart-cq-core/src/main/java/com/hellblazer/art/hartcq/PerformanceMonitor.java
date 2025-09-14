package com.hellblazer.art.hartcq;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.OperatingSystemMXBean;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Consumer;

/**
 * Performance monitoring system for HART-CQ processing.
 * Tracks throughput (must achieve >100 sentences/sec), latency monitoring,
 * and resource utilization with real-time reporting capabilities.
 */
public class PerformanceMonitor implements AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(PerformanceMonitor.class);
    
    // Performance counters
    private final LongAdder sentencesProcessed = new LongAdder();
    private final LongAdder tokensProcessed = new LongAdder();
    private final LongAdder windowsProcessed = new LongAdder();
    private final LongAdder errorsEncountered = new LongAdder();
    
    // Timing tracking
    private final AtomicLong totalProcessingTimeNanos = new AtomicLong(0);
    private final AtomicLong minProcessingTimeNanos = new AtomicLong(Long.MAX_VALUE);
    private final AtomicLong maxProcessingTimeNanos = new AtomicLong(0);
    private final AtomicLong startTimeNanos = new AtomicLong(System.nanoTime());
    
    // System monitoring
    private final OperatingSystemMXBean osBean;
    private final MemoryMXBean memoryBean;
    private final ScheduledExecutorService monitoringExecutor;
    
    // Configuration
    private final int targetThroughputSentencesPerSecond;
    private final int reportIntervalSeconds;
    private final Consumer<PerformanceReport> reportConsumer;
    
    // State
    private volatile boolean monitoring = false;
    private ScheduledFuture<?> reportingTask;
    
    /**
     * Creates a PerformanceMonitor with default settings.
     */
    public PerformanceMonitor() {
        this(100, 30, null);
    }
    
    /**
     * Creates a PerformanceMonitor with custom configuration.
     * 
     * @param targetThroughputSentencesPerSecond target throughput (must be >100)
     * @param reportIntervalSeconds interval for performance reports
     * @param reportConsumer consumer for performance reports (null for logging only)
     */
    public PerformanceMonitor(int targetThroughputSentencesPerSecond, 
                             int reportIntervalSeconds,
                             Consumer<PerformanceReport> reportConsumer) {
        this.targetThroughputSentencesPerSecond = Math.max(1, targetThroughputSentencesPerSecond);
        this.reportIntervalSeconds = Math.max(1, reportIntervalSeconds);
        this.reportConsumer = reportConsumer;
        
        this.osBean = ManagementFactory.getOperatingSystemMXBean();
        this.memoryBean = ManagementFactory.getMemoryMXBean();
        this.monitoringExecutor = Executors.newSingleThreadScheduledExecutor(r -> {
            var thread = new Thread(r, "HART-CQ-Monitor");
            thread.setDaemon(true);
            return thread;
        });
    }
    
    /**
     * Starts performance monitoring and reporting.
     */
    public void startMonitoring() {
        if (monitoring) {
            return;
        }
        
        monitoring = true;
        startTimeNanos.set(System.nanoTime());
        
        reportingTask = monitoringExecutor.scheduleAtFixedRate(
            this::generateAndReportPerformance,
            reportIntervalSeconds,
            reportIntervalSeconds,
            TimeUnit.SECONDS
        );
        
        logger.info("Performance monitoring started. Target: {} sentences/sec, Report interval: {} seconds",
                   targetThroughputSentencesPerSecond, reportIntervalSeconds);
    }
    
    /**
     * Stops performance monitoring.
     */
    public void stopMonitoring() {
        if (!monitoring) {
            return;
        }
        
        monitoring = false;
        
        if (reportingTask != null) {
            reportingTask.cancel(false);
            reportingTask = null;
        }
        
        // Generate final report
        generateAndReportPerformance();
        
        logger.info("Performance monitoring stopped");
    }
    
    /**
     * Records the processing of sentences.
     * 
     * @param count number of sentences processed
     */
    public void recordSentencesProcessed(int count) {
        sentencesProcessed.add(count);
    }
    
    /**
     * Records the processing of tokens.
     * 
     * @param count number of tokens processed
     */
    public void recordTokensProcessed(int count) {
        tokensProcessed.add(count);
    }
    
    /**
     * Records the processing of windows.
     * 
     * @param count number of windows processed
     */
    public void recordWindowsProcessed(int count) {
        windowsProcessed.add(count);
    }
    
    /**
     * Records an error occurrence.
     */
    public void recordError() {
        errorsEncountered.increment();
    }
    
    /**
     * Records processing time for a single operation.
     * 
     * @param processingTimeNanos time taken in nanoseconds
     */
    public void recordProcessingTime(long processingTimeNanos) {
        totalProcessingTimeNanos.addAndGet(processingTimeNanos);
        
        // Update min/max with thread-safe operations
        minProcessingTimeNanos.updateAndGet(current -> Math.min(current, processingTimeNanos));
        maxProcessingTimeNanos.updateAndGet(current -> Math.max(current, processingTimeNanos));
    }
    
    /**
     * Records a complete window processing operation.
     * 
     * @param sentenceCount sentences in the window
     * @param tokenCount tokens in the window
     * @param processingTimeNanos time taken to process
     */
    public void recordWindowProcessing(int sentenceCount, int tokenCount, long processingTimeNanos) {
        recordSentencesProcessed(sentenceCount);
        recordTokensProcessed(tokenCount);
        recordWindowsProcessed(1);
        recordProcessingTime(processingTimeNanos);
    }
    
    /**
     * Gets current throughput in sentences per second.
     * 
     * @return current sentences per second
     */
    public double getCurrentThroughputSentencesPerSecond() {
        var elapsedNanos = System.nanoTime() - startTimeNanos.get();
        var elapsedSeconds = elapsedNanos / 1_000_000_000.0;
        
        if (elapsedSeconds < 0.001) { // Avoid division by zero
            return 0.0;
        }
        
        return sentencesProcessed.sum() / elapsedSeconds;
    }
    
    /**
     * Gets current throughput in tokens per second.
     * 
     * @return current tokens per second
     */
    public double getCurrentThroughputTokensPerSecond() {
        var elapsedNanos = System.nanoTime() - startTimeNanos.get();
        var elapsedSeconds = elapsedNanos / 1_000_000_000.0;
        
        if (elapsedSeconds < 0.001) {
            return 0.0;
        }
        
        return tokensProcessed.sum() / elapsedSeconds;
    }
    
    /**
     * Gets average processing time per operation in milliseconds.
     * 
     * @return average processing time in milliseconds
     */
    public double getAverageProcessingTimeMs() {
        long operations = sentencesProcessed.sum() + windowsProcessed.sum();
        if (operations == 0) {
            return 0.0;
        }
        
        return (totalProcessingTimeNanos.get() / (double) operations) / 1_000_000.0;
    }
    
    /**
     * Checks if the current throughput meets the target.
     * 
     * @return true if current throughput >= target throughput
     */
    public boolean isMeetingThroughputTarget() {
        return getCurrentThroughputSentencesPerSecond() >= targetThroughputSentencesPerSecond;
    }
    
    /**
     * Gets current system CPU usage.
     * 
     * @return CPU usage as a percentage [0.0, 1.0]
     */
    public double getCpuUsage() {
        if (osBean instanceof com.sun.management.OperatingSystemMXBean sunOsBean) {
            return sunOsBean.getProcessCpuLoad();
        }
        return osBean.getSystemLoadAverage() / osBean.getAvailableProcessors();
    }
    
    /**
     * Gets current memory usage.
     * 
     * @return memory usage as a percentage [0.0, 1.0]
     */
    public double getMemoryUsage() {
        var heapMemory = memoryBean.getHeapMemoryUsage();
        return (double) heapMemory.getUsed() / heapMemory.getMax();
    }
    
    /**
     * Generates a comprehensive performance report.
     * 
     * @return current performance report
     */
    public PerformanceReport generatePerformanceReport() {
        var elapsedNanos = System.nanoTime() - startTimeNanos.get();
        var elapsedSeconds = elapsedNanos / 1_000_000_000.0;
        
        return new PerformanceReport(
            System.currentTimeMillis(),
            elapsedSeconds,
            sentencesProcessed.sum(),
            tokensProcessed.sum(),
            windowsProcessed.sum(),
            errorsEncountered.sum(),
            getCurrentThroughputSentencesPerSecond(),
            getCurrentThroughputTokensPerSecond(),
            getAverageProcessingTimeMs(),
            minProcessingTimeNanos.get() == Long.MAX_VALUE ? 0 : minProcessingTimeNanos.get() / 1_000_000.0,
            maxProcessingTimeNanos.get() / 1_000_000.0,
            getCpuUsage(),
            getMemoryUsage(),
            targetThroughputSentencesPerSecond,
            isMeetingThroughputTarget()
        );
    }
    
    /**
     * Generates and reports performance statistics.
     */
    private void generateAndReportPerformance() {
        try {
            var report = generatePerformanceReport();
            
            // Log the report
            logger.info("Performance Report: {}", report);
            
            // Send to consumer if provided
            if (reportConsumer != null) {
                reportConsumer.accept(report);
            }
            
            // Warn if not meeting throughput target
            if (!report.isMeetingTarget()) {
                logger.warn("WARNING: Throughput below target! Current: {:.1f} sentences/sec, Target: {} sentences/sec",
                          report.getCurrentThroughputSentencesPerSecond(), 
                          report.getTargetThroughputSentencesPerSecond());
            }
            
        } catch (Exception e) {
            logger.error("Error generating performance report", e);
        }
    }
    
    /**
     * Resets all performance counters and timing statistics.
     */
    public void reset() {
        sentencesProcessed.reset();
        tokensProcessed.reset();
        windowsProcessed.reset();
        errorsEncountered.reset();
        totalProcessingTimeNanos.set(0);
        minProcessingTimeNanos.set(Long.MAX_VALUE);
        maxProcessingTimeNanos.set(0);
        startTimeNanos.set(System.nanoTime());
        
        logger.info("Performance monitor reset");
    }
    
    @Override
    public void close() {
        stopMonitoring();
        monitoringExecutor.shutdown();
        
        try {
            if (!monitoringExecutor.awaitTermination(5, TimeUnit.SECONDS)) {
                monitoringExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            monitoringExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }
        
        logger.info("PerformanceMonitor closed");
    }
    
    /**
     * Performance report containing comprehensive statistics.
     */
    public static class PerformanceReport {
        private final long timestamp;
        private final double elapsedTimeSeconds;
        private final long totalSentencesProcessed;
        private final long totalTokensProcessed;
        private final long totalWindowsProcessed;
        private final long totalErrors;
        private final double currentThroughputSentencesPerSecond;
        private final double currentThroughputTokensPerSecond;
        private final double averageProcessingTimeMs;
        private final double minProcessingTimeMs;
        private final double maxProcessingTimeMs;
        private final double cpuUsage;
        private final double memoryUsage;
        private final int targetThroughputSentencesPerSecond;
        private final boolean meetingTarget;
        
        public PerformanceReport(long timestamp, double elapsedTimeSeconds,
                               long totalSentencesProcessed, long totalTokensProcessed,
                               long totalWindowsProcessed, long totalErrors,
                               double currentThroughputSentencesPerSecond,
                               double currentThroughputTokensPerSecond,
                               double averageProcessingTimeMs,
                               double minProcessingTimeMs, double maxProcessingTimeMs,
                               double cpuUsage, double memoryUsage,
                               int targetThroughputSentencesPerSecond, boolean meetingTarget) {
            this.timestamp = timestamp;
            this.elapsedTimeSeconds = elapsedTimeSeconds;
            this.totalSentencesProcessed = totalSentencesProcessed;
            this.totalTokensProcessed = totalTokensProcessed;
            this.totalWindowsProcessed = totalWindowsProcessed;
            this.totalErrors = totalErrors;
            this.currentThroughputSentencesPerSecond = currentThroughputSentencesPerSecond;
            this.currentThroughputTokensPerSecond = currentThroughputTokensPerSecond;
            this.averageProcessingTimeMs = averageProcessingTimeMs;
            this.minProcessingTimeMs = minProcessingTimeMs;
            this.maxProcessingTimeMs = maxProcessingTimeMs;
            this.cpuUsage = cpuUsage;
            this.memoryUsage = memoryUsage;
            this.targetThroughputSentencesPerSecond = targetThroughputSentencesPerSecond;
            this.meetingTarget = meetingTarget;
        }
        
        // Getters
        public long getTimestamp() { return timestamp; }
        public double getElapsedTimeSeconds() { return elapsedTimeSeconds; }
        public long getTotalSentencesProcessed() { return totalSentencesProcessed; }
        public long getTotalTokensProcessed() { return totalTokensProcessed; }
        public long getTotalWindowsProcessed() { return totalWindowsProcessed; }
        public long getTotalErrors() { return totalErrors; }
        public double getCurrentThroughputSentencesPerSecond() { return currentThroughputSentencesPerSecond; }
        public double getCurrentThroughputTokensPerSecond() { return currentThroughputTokensPerSecond; }
        public double getAverageProcessingTimeMs() { return averageProcessingTimeMs; }
        public double getMinProcessingTimeMs() { return minProcessingTimeMs; }
        public double getMaxProcessingTimeMs() { return maxProcessingTimeMs; }
        public double getCpuUsage() { return cpuUsage; }
        public double getMemoryUsage() { return memoryUsage; }
        public int getTargetThroughputSentencesPerSecond() { return targetThroughputSentencesPerSecond; }
        public boolean isMeetingTarget() { return meetingTarget; }
        
        /**
         * Gets error rate as a percentage of total operations.
         * @return error rate [0.0, 1.0]
         */
        public double getErrorRate() {
            var totalOps = totalSentencesProcessed + totalWindowsProcessed;
            return totalOps > 0 ? (double) totalErrors / totalOps : 0.0;
        }
        
        @Override
        public String toString() {
            return String.format(
                "PerformanceReport[sentences=%d (%.1f/sec), tokens=%d (%.1f/sec), " +
                "windows=%d, errors=%d (%.2f%%), avgTime=%.2fms, cpu=%.1f%%, mem=%.1f%%, " +
                "target=%s]",
                totalSentencesProcessed, currentThroughputSentencesPerSecond,
                totalTokensProcessed, currentThroughputTokensPerSecond,
                totalWindowsProcessed, totalErrors, getErrorRate() * 100,
                averageProcessingTimeMs, cpuUsage * 100, memoryUsage * 100,
                meetingTarget ? "MET" : "MISSED"
            );
        }
    }
}