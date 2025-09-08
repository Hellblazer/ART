package com.hellblazer.art.nlp.streaming;

import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.OperatingSystemMXBean;
import java.time.Duration;
import java.time.Instant;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Real-time metrics and monitoring system for streaming ART processing.
 * Provides comprehensive performance tracking, alerting, and health monitoring.
 */
public class StreamingMetrics implements AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(StreamingMetrics.class);
    
    private final String metricsId;
    private final MetricsConfig config;
    private final ScheduledExecutorService scheduler;
    private final Map<String, Counter> counters = new ConcurrentHashMap<>();
    private final Map<String, Gauge> gauges = new ConcurrentHashMap<>();
    private final Map<String, Timer> timers = new ConcurrentHashMap<>();
    private final Map<String, Histogram> histograms = new ConcurrentHashMap<>();
    private final Map<String, HealthCheck> healthChecks = new ConcurrentHashMap<>();
    private final AtomicLong startTime = new AtomicLong(System.currentTimeMillis());
    
    private volatile boolean monitoring = false;
    
    public StreamingMetrics(String metricsId, MetricsConfig config) {
        this.metricsId = metricsId;
        this.config = config;
        this.scheduler = Executors.newScheduledThreadPool(2, 
            r -> Thread.ofVirtual().name("metrics-" + metricsId + "-").factory().newThread(r));
    }
    
    /**
     * Configuration for metrics system.
     */
    public record MetricsConfig(
        Duration reportingInterval,
        Duration healthCheckInterval,
        boolean enableJVMMetrics,
        boolean enableSystemMetrics,
        int histogramBuckets,
        double[] histogramPercentiles,
        AlertConfig alertConfig
    ) {
        public static MetricsConfig defaultConfig() {
            return new MetricsConfig(
                Duration.ofSeconds(30),
                Duration.ofSeconds(10),
                true,
                true,
                64,
                new double[]{0.5, 0.75, 0.95, 0.99, 0.999},
                AlertConfig.defaultConfig()
            );
        }
    }
    
    /**
     * Alert configuration for thresholds and notifications.
     */
    public record AlertConfig(
        double errorRateThreshold,
        Duration latencyThreshold,
        long memoryUsageThreshold,
        double cpuUsageThreshold,
        boolean enableAlerting
    ) {
        public static AlertConfig defaultConfig() {
            return new AlertConfig(
                0.05, // 5% error rate
                Duration.ofSeconds(1),
                500 * 1024 * 1024, // 500MB
                0.8, // 80% CPU
                true
            );
        }
    }
    
    /**
     * Counter for tracking event counts.
     */
    public static class Counter {
        private final LongAdder count = new LongAdder();
        private final AtomicLong lastUpdate = new AtomicLong();
        
        public void increment() {
            count.increment();
            lastUpdate.set(System.currentTimeMillis());
        }
        
        public void increment(long delta) {
            count.add(delta);
            lastUpdate.set(System.currentTimeMillis());
        }
        
        public long get() {
            return count.sum();
        }
        
        public long getLastUpdate() {
            return lastUpdate.get();
        }
    }
    
    /**
     * Gauge for tracking current values.
     */
    public static class Gauge {
        private final Supplier<Double> valueSupplier;
        private volatile double lastValue;
        private final AtomicLong lastUpdate = new AtomicLong();
        
        public Gauge(Supplier<Double> valueSupplier) {
            this.valueSupplier = valueSupplier;
        }
        
        public double get() {
            var value = valueSupplier.get();
            lastValue = value;
            lastUpdate.set(System.currentTimeMillis());
            return value;
        }
        
        public double getLastValue() {
            return lastValue;
        }
        
        public long getLastUpdate() {
            return lastUpdate.get();
        }
    }
    
    /**
     * Timer for tracking operation durations.
     */
    public static class Timer {
        private final LongAdder totalTime = new LongAdder();
        private final LongAdder count = new LongAdder();
        private final AtomicLong minTime = new AtomicLong(Long.MAX_VALUE);
        private final AtomicLong maxTime = new AtomicLong(Long.MIN_VALUE);
        
        public void record(Duration duration) {
            var nanos = duration.toNanos();
            totalTime.add(nanos);
            count.increment();
            
            // Update min/max
            minTime.updateAndGet(current -> Math.min(current, nanos));
            maxTime.updateAndGet(current -> Math.max(current, nanos));
        }
        
        public double getMean() {
            var c = count.sum();
            return c > 0 ? totalTime.sum() / (double) c : 0.0;
        }
        
        public long getCount() {
            return count.sum();
        }
        
        public Duration getMin() {
            var min = minTime.get();
            return min == Long.MAX_VALUE ? Duration.ZERO : Duration.ofNanos(min);
        }
        
        public Duration getMax() {
            var max = maxTime.get();
            return max == Long.MIN_VALUE ? Duration.ZERO : Duration.ofNanos(max);
        }
    }
    
    /**
     * Histogram for tracking value distributions.
     */
    public static class Histogram {
        private final long[] buckets;
        private final double[] percentiles;
        private final LongAdder count = new LongAdder();
        private final LongAdder sum = new LongAdder();
        private final AtomicLong min = new AtomicLong(Long.MAX_VALUE);
        private final AtomicLong max = new AtomicLong(Long.MIN_VALUE);
        
        public Histogram(int bucketCount, double[] percentiles) {
            this.buckets = new long[bucketCount];
            this.percentiles = percentiles.clone();
        }
        
        public void record(long value) {
            count.increment();
            sum.add(value);
            
            min.updateAndGet(current -> Math.min(current, value));
            max.updateAndGet(current -> Math.max(current, value));
            
            // Simple bucket assignment (could be more sophisticated)
            var bucketIndex = Math.min((int) (value * buckets.length / (max.get() + 1)), buckets.length - 1);
            synchronized (buckets) {
                buckets[bucketIndex]++;
            }
        }
        
        public long getCount() {
            return count.sum();
        }
        
        public double getMean() {
            var c = count.sum();
            return c > 0 ? sum.sum() / (double) c : 0.0;
        }
        
        public long getMin() {
            var minVal = min.get();
            return minVal == Long.MAX_VALUE ? 0 : minVal;
        }
        
        public long getMax() {
            var maxVal = max.get();
            return maxVal == Long.MIN_VALUE ? 0 : maxVal;
        }
        
        public double[] getPercentiles() {
            // Simplified percentile calculation
            var totalCount = count.sum();
            if (totalCount == 0) return new double[percentiles.length];
            
            var result = new double[percentiles.length];
            synchronized (buckets) {
                for (int i = 0; i < percentiles.length; i++) {
                    var targetCount = (long) (totalCount * percentiles[i]);
                    var runningCount = 0L;
                    
                    for (int j = 0; j < buckets.length; j++) {
                        runningCount += buckets[j];
                        if (runningCount >= targetCount) {
                            result[i] = j * (max.get() + 1.0) / buckets.length;
                            break;
                        }
                    }
                }
            }
            return result;
        }
    }
    
    /**
     * Health check for monitoring system health.
     */
    public static class HealthCheck {
        private final String name;
        private final Supplier<Boolean> check;
        private volatile boolean lastResult = true;
        private final AtomicLong lastCheck = new AtomicLong();
        private volatile String lastError;
        
        public HealthCheck(String name, Supplier<Boolean> check) {
            this.name = name;
            this.check = check;
        }
        
        public boolean execute() {
            try {
                var result = check.get();
                lastResult = result;
                lastError = null;
                lastCheck.set(System.currentTimeMillis());
                return result;
            } catch (Exception e) {
                lastResult = false;
                lastError = e.getMessage();
                lastCheck.set(System.currentTimeMillis());
                logger.warn("Health check '{}' failed: {}", name, e.getMessage());
                return false;
            }
        }
        
        public boolean getLastResult() {
            return lastResult;
        }
        
        public String getLastError() {
            return lastError;
        }
        
        public long getLastCheck() {
            return lastCheck.get();
        }
        
        public String getName() {
            return name;
        }
    }
    
    /**
     * Metrics snapshot for reporting.
     */
    public record MetricsSnapshot(
        String metricsId,
        Instant timestamp,
        long uptime,
        Map<String, Long> counters,
        Map<String, Double> gauges,
        Map<String, TimerStats> timers,
        Map<String, HistogramStats> histograms,
        Map<String, HealthStatus> healthChecks,
        SystemMetrics systemMetrics
    ) {}
    
    public record TimerStats(long count, double mean, Duration min, Duration max) {}
    
    public record HistogramStats(long count, double mean, long min, long max, double[] percentiles) {}
    
    public record HealthStatus(boolean healthy, String lastError, Instant lastCheck) {}
    
    public record SystemMetrics(
        long usedMemory,
        long maxMemory,
        double cpuUsage,
        int activeThreads,
        long gcTime
    ) {}
    
    // API Methods
    
    public void startMonitoring() {
        if (monitoring) return;
        
        monitoring = true;
        startTime.set(System.currentTimeMillis());
        
        // Schedule periodic reporting
        scheduler.scheduleAtFixedRate(
            this::reportMetrics,
            config.reportingInterval().toMillis(),
            config.reportingInterval().toMillis(),
            TimeUnit.MILLISECONDS
        );
        
        // Schedule health checks
        scheduler.scheduleAtFixedRate(
            this::runHealthChecks,
            config.healthCheckInterval().toMillis(),
            config.healthCheckInterval().toMillis(),
            TimeUnit.MILLISECONDS
        );
        
        logger.info("Started metrics monitoring for {}", metricsId);
    }
    
    public void stopMonitoring() {
        monitoring = false;
        logger.info("Stopped metrics monitoring for {}", metricsId);
    }
    
    public Counter counter(String name) {
        return counters.computeIfAbsent(name, k -> new Counter());
    }
    
    public Gauge gauge(String name, Supplier<Double> valueSupplier) {
        return gauges.computeIfAbsent(name, k -> new Gauge(valueSupplier));
    }
    
    public Timer timer(String name) {
        return timers.computeIfAbsent(name, k -> new Timer());
    }
    
    public Histogram histogram(String name) {
        return histograms.computeIfAbsent(name, k -> new Histogram(config.histogramBuckets(), config.histogramPercentiles()));
    }
    
    public void addHealthCheck(String name, Supplier<Boolean> check) {
        healthChecks.put(name, new HealthCheck(name, check));
    }
    
    public MetricsSnapshot getSnapshot() {
        var timestamp = Instant.now();
        var uptime = System.currentTimeMillis() - startTime.get();
        
        var counterSnapshot = counters.entrySet().stream()
            .collect(java.util.stream.Collectors.toMap(
                Map.Entry::getKey,
                e -> e.getValue().get(),
                (v1, v2) -> v1,
                ConcurrentHashMap::new));
        
        var gaugeSnapshot = gauges.entrySet().stream()
            .collect(java.util.stream.Collectors.toMap(
                Map.Entry::getKey,
                e -> e.getValue().get(),
                (v1, v2) -> v1,
                ConcurrentHashMap::new));
        
        var timerSnapshot = timers.entrySet().stream()
            .collect(java.util.stream.Collectors.toMap(
                Map.Entry::getKey,
                e -> {
                    var timer = e.getValue();
                    return new TimerStats(timer.getCount(), timer.getMean(), timer.getMin(), timer.getMax());
                },
                (v1, v2) -> v1,
                ConcurrentHashMap::new));
        
        var histogramSnapshot = histograms.entrySet().stream()
            .collect(java.util.stream.Collectors.toMap(
                Map.Entry::getKey,
                e -> {
                    var hist = e.getValue();
                    return new HistogramStats(hist.getCount(), hist.getMean(), 
                        hist.getMin(), hist.getMax(), hist.getPercentiles());
                },
                (v1, v2) -> v1,
                ConcurrentHashMap::new));
        
        var healthSnapshot = healthChecks.entrySet().stream()
            .collect(java.util.stream.Collectors.toMap(
                Map.Entry::getKey,
                e -> {
                    var health = e.getValue();
                    return new HealthStatus(health.getLastResult(), 
                        health.getLastError(), Instant.ofEpochMilli(health.getLastCheck()));
                },
                (v1, v2) -> v1,
                ConcurrentHashMap::new));
        
        var systemMetrics = collectSystemMetrics();
        
        return new MetricsSnapshot(metricsId, timestamp, uptime, counterSnapshot, 
            gaugeSnapshot, timerSnapshot, histogramSnapshot, healthSnapshot, systemMetrics);
    }
    
    private void reportMetrics() {
        if (!monitoring) return;
        
        var snapshot = getSnapshot();
        logger.info("Metrics report for {}: uptime={}s, counters={}, timers={}, health={}",
            metricsId,
            snapshot.uptime() / 1000,
            snapshot.counters().size(),
            snapshot.timers().size(),
            snapshot.healthChecks().values().stream().allMatch(HealthStatus::healthy)
        );
        
        // Check alerts
        if (config.alertConfig().enableAlerting()) {
            checkAlerts(snapshot);
        }
    }
    
    private void runHealthChecks() {
        healthChecks.values().forEach(HealthCheck::execute);
    }
    
    private void checkAlerts(MetricsSnapshot snapshot) {
        var alerts = config.alertConfig();
        
        // Check error rate
        var totalOperations = snapshot.counters().getOrDefault("operations.total", 0L);
        var errorCount = snapshot.counters().getOrDefault("operations.errors", 0L);
        if (totalOperations > 0) {
            var errorRate = errorCount / (double) totalOperations;
            if (errorRate > alerts.errorRateThreshold()) {
                logger.warn("ALERT: Error rate {}% exceeds threshold {}%", 
                    errorRate * 100, alerts.errorRateThreshold() * 100);
            }
        }
        
        // Check latency
        var processingTimer = snapshot.timers().get("processing.time");
        if (processingTimer != null && Duration.ofNanos((long) processingTimer.mean()).compareTo(alerts.latencyThreshold()) > 0) {
            logger.warn("ALERT: Average processing time {}ms exceeds threshold {}ms",
                Duration.ofNanos((long) processingTimer.mean()).toMillis(),
                alerts.latencyThreshold().toMillis());
        }
        
        // Check system metrics
        var system = snapshot.systemMetrics();
        if (system.usedMemory() > alerts.memoryUsageThreshold()) {
            logger.warn("ALERT: Memory usage {}MB exceeds threshold {}MB",
                system.usedMemory() / (1024 * 1024),
                alerts.memoryUsageThreshold() / (1024 * 1024));
        }
        
        if (system.cpuUsage() > alerts.cpuUsageThreshold()) {
            logger.warn("ALERT: CPU usage {}% exceeds threshold {}%",
                system.cpuUsage() * 100,
                alerts.cpuUsageThreshold() * 100);
        }
    }
    
    private SystemMetrics collectSystemMetrics() {
        if (!config.enableSystemMetrics()) {
            return new SystemMetrics(0, 0, 0.0, 0, 0);
        }
        
        var runtime = Runtime.getRuntime();
        var usedMemory = runtime.totalMemory() - runtime.freeMemory();
        var maxMemory = runtime.maxMemory();
        
        // Real CPU usage monitoring via MXBeans
        var cpuUsage = getCpuUsage();
        
        var activeThreads = Thread.activeCount();
        
        // Real GC time monitoring via MXBeans
        var gcTime = getTotalGcTime();
        
        return new SystemMetrics(usedMemory, maxMemory, cpuUsage, activeThreads, gcTime);
    }
    
    /**
     * Get current CPU usage as a percentage (0.0 to 1.0).
     * Returns -1.0 if CPU usage is not available.
     */
    private double getCpuUsage() {
        try {
            var osBean = ManagementFactory.getOperatingSystemMXBean();
            
            // Try to access platform-specific CPU usage
            try {
                var method = osBean.getClass().getMethod("getProcessCpuLoad");
                var cpuLoad = (Double) method.invoke(osBean);
                if (cpuLoad >= 0.0 && cpuLoad <= 1.0) {
                    return cpuLoad;
                }
            } catch (Exception ignored) {
                // Platform-specific CPU load not available
            }
            
            // Fallback to system load average (approximation)
            var loadAverage = osBean.getSystemLoadAverage();
            if (loadAverage >= 0.0) {
                var processors = osBean.getAvailableProcessors();
                return Math.min(loadAverage / processors, 1.0);
            }
            
        } catch (Exception e) {
            logger.debug("Failed to get CPU usage", e);
        }
        return -1.0; // Indicates unavailable
    }
    
    /**
     * Get total garbage collection time across all GC collectors in milliseconds.
     */
    private long getTotalGcTime() {
        try {
            var gcBeans = ManagementFactory.getGarbageCollectorMXBeans();
            var totalTime = 0L;
            for (var gcBean : gcBeans) {
                var collectionTime = gcBean.getCollectionTime();
                if (collectionTime > 0) {
                    totalTime += collectionTime;
                }
            }
            return totalTime;
        } catch (Exception e) {
            logger.debug("Failed to get GC time", e);
            return 0L;
        }
    }
    
    @Override
    public void close() {
        stopMonitoring();
        scheduler.shutdown();
        try {
            if (!scheduler.awaitTermination(5, TimeUnit.SECONDS)) {
                scheduler.shutdownNow();
            }
        } catch (InterruptedException e) {
            scheduler.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
}