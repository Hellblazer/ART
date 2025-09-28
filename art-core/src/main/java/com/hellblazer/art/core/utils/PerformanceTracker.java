/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 *
 * This file is part of Java ART Neural Networks.
 *
 * Java ART Neural Networks is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Java ART Neural Networks is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with Java ART Neural Networks. If not, see <https://www.gnu.org/licenses/>.
 */
package com.hellblazer.art.core.utils;

import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;

/**
 * Interface for tracking performance metrics in hybrid ART neural networks.
 *
 * PerformanceTracker provides comprehensive monitoring of algorithm performance,
 * resource usage, and operational metrics. It enables profiling, optimization,
 * and monitoring of ART systems in production environments.
 *
 * Key tracking capabilities:
 * - Execution time measurements
 * - Memory usage monitoring
 * - Throughput and latency tracking
 * - Resource utilization metrics
 * - Algorithm-specific performance indicators
 * - Statistical analysis and reporting
 * - Performance trend analysis
 *
 * @author Hal Hildebrand
 * @since 1.0
 */
public interface PerformanceTracker extends AutoCloseable {

    /**
     * Types of performance metrics that can be tracked.
     */
    enum MetricType {
        /** Execution time measurements */
        TIMING,
        /** Memory usage metrics */
        MEMORY,
        /** CPU utilization */
        CPU,
        /** Throughput measurements */
        THROUGHPUT,
        /** Latency measurements */
        LATENCY,
        /** Algorithm accuracy */
        ACCURACY,
        /** Learning rate and convergence */
        CONVERGENCE,
        /** Resource utilization */
        RESOURCE,
        /** Custom application metrics */
        CUSTOM,
        /** Error and exception tracking */
        ERROR
    }

    /**
     * Aggregation methods for metric values.
     */
    enum AggregationMethod {
        /** Sum of all values */
        SUM,
        /** Average of all values */
        AVERAGE,
        /** Minimum value */
        MIN,
        /** Maximum value */
        MAX,
        /** Median value */
        MEDIAN,
        /** 95th percentile */
        P95,
        /** 99th percentile */
        P99,
        /** Standard deviation */
        STDDEV,
        /** Count of occurrences */
        COUNT,
        /** Rate per second */
        RATE
    }

    /**
     * Performance measurement contexts.
     */
    enum Context {
        /** Training/learning operations */
        TRAINING,
        /** Prediction/inference operations */
        INFERENCE,
        /** Initialization operations */
        INITIALIZATION,
        /** Memory management operations */
        MEMORY_MANAGEMENT,
        /** Data preprocessing */
        PREPROCESSING,
        /** Model serialization */
        SERIALIZATION,
        /** Validation operations */
        VALIDATION,
        /** Overall system operations */
        SYSTEM
    }

    /**
     * Start tracking a performance measurement.
     *
     * @param metricName unique name for the metric
     * @param metricType type of metric being tracked
     * @param context operational context
     * @return measurement ID for stopping the measurement
     */
    String startMeasurement(String metricName, MetricType metricType, Context context);

    /**
     * Stop tracking a performance measurement.
     *
     * @param measurementId the measurement ID returned by startMeasurement
     * @return the measured duration
     */
    Duration stopMeasurement(String measurementId);

    /**
     * Record a single metric value.
     *
     * @param metricName name of the metric
     * @param value metric value
     * @param metricType type of metric
     * @param context operational context
     * @param timestamp when the measurement was taken
     */
    void recordMetric(String metricName, double value, MetricType metricType,
                     Context context, Instant timestamp);

    /**
     * Record a metric value with current timestamp.
     *
     * @param metricName name of the metric
     * @param value metric value
     * @param metricType type of metric
     * @param context operational context
     */
    default void recordMetric(String metricName, double value, MetricType metricType, Context context) {
        recordMetric(metricName, value, metricType, context, Instant.now());
    }

    /**
     * Record multiple metrics at once.
     *
     * @param metrics map of metric names to values
     * @param metricType type for all metrics
     * @param context operational context
     * @param timestamp when measurements were taken
     */
    default void recordMetrics(Map<String, Double> metrics, MetricType metricType,
                              Context context, Instant timestamp) {
        metrics.forEach((name, value) -> recordMetric(name, value, metricType, context, timestamp));
    }

    /**
     * Record timing for a code block using try-with-resources.
     *
     * @param metricName name of the timing metric
     * @param context operational context
     * @return timer resource that stops measurement when closed
     */
    Timer startTimer(String metricName, Context context);

    /**
     * Increment a counter metric.
     *
     * @param counterName name of the counter
     * @param increment amount to increment by
     * @param context operational context
     */
    void incrementCounter(String counterName, long increment, Context context);

    /**
     * Increment counter by 1.
     *
     * @param counterName name of the counter
     * @param context operational context
     */
    default void incrementCounter(String counterName, Context context) {
        incrementCounter(counterName, 1L, context);
    }

    /**
     * Record memory usage snapshot.
     *
     * @param context operational context
     * @return memory usage statistics
     */
    MemorySnapshot recordMemoryUsage(Context context);

    /**
     * Get current performance statistics for a metric.
     *
     * @param metricName name of the metric
     * @param aggregationMethod how to aggregate multiple values
     * @return optional performance statistics
     */
    Optional<PerformanceStatistics> getStatistics(String metricName, AggregationMethod aggregationMethod);

    /**
     * Get performance statistics for all metrics.
     *
     * @return map of metric names to their statistics
     */
    Map<String, PerformanceStatistics> getAllStatistics();

    /**
     * Get performance statistics filtered by context.
     *
     * @param context the operational context to filter by
     * @return map of metric names to their statistics
     */
    Map<String, PerformanceStatistics> getStatistics(Context context);

    /**
     * Get performance statistics filtered by metric type.
     *
     * @param metricType the metric type to filter by
     * @return map of metric names to their statistics
     */
    Map<String, PerformanceStatistics> getStatistics(MetricType metricType);

    /**
     * Generate a comprehensive performance report.
     *
     * @param timeRange optional time range for the report
     * @param includeDetails whether to include detailed metrics
     * @return performance report
     */
    PerformanceReport generateReport(Optional<TimeRange> timeRange, boolean includeDetails);

    /**
     * Get performance trends over time.
     *
     * @param metricName name of the metric
     * @param timeWindow time window for trend analysis
     * @return performance trend analysis
     */
    Optional<PerformanceTrend> getTrend(String metricName, Duration timeWindow);

    /**
     * Set up performance alerts for metric thresholds.
     *
     * @param metricName name of the metric to monitor
     * @param threshold threshold value for alerting
     * @param comparison comparison operator (GREATER_THAN, LESS_THAN, etc.)
     * @param alertHandler handler to call when threshold is exceeded
     * @return alert ID for future reference
     */
    String addAlert(String metricName, double threshold, ThresholdComparison comparison,
                   AlertHandler alertHandler);

    /**
     * Remove a performance alert.
     *
     * @param alertId the alert ID to remove
     * @return true if alert was found and removed
     */
    boolean removeAlert(String alertId);

    /**
     * Enable or disable metric collection for specific contexts.
     *
     * @param context the context to enable/disable
     * @param enabled whether to enable tracking for this context
     */
    void setContextEnabled(Context context, boolean enabled);

    /**
     * Check if metric collection is enabled for a context.
     *
     * @param context the context to check
     * @return true if tracking is enabled
     */
    boolean isContextEnabled(Context context);

    /**
     * Set sampling rate for metric collection (to reduce overhead).
     *
     * @param samplingRate sampling rate between 0.0 and 1.0
     */
    void setSamplingRate(double samplingRate);

    /**
     * Get current sampling rate.
     *
     * @return current sampling rate
     */
    double getSamplingRate();

    /**
     * Clear all collected metrics and reset counters.
     */
    void reset();

    /**
     * Clear metrics older than specified age.
     *
     * @param maxAge maximum age of metrics to retain
     * @return number of metrics cleared
     */
    long clearOldMetrics(Duration maxAge);

    /**
     * Export performance data for external analysis.
     *
     * @param format export format (JSON, CSV, etc.)
     * @param timeRange optional time range to export
     * @return exported data
     */
    Optional<Object> exportData(String format, Optional<TimeRange> timeRange);

    /**
     * Get the overhead of performance tracking itself.
     *
     * @return tracking overhead statistics
     */
    TrackingOverhead getTrackingOverhead();

    /**
     * Enable or disable performance tracking globally.
     *
     * @param enabled whether to enable tracking
     */
    void setEnabled(boolean enabled);

    /**
     * Check if performance tracking is enabled.
     *
     * @return true if tracking is enabled
     */
    boolean isEnabled();

    /**
     * Release resources used by the performance tracker.
     */
    @Override
    void close();

    /**
     * Timer resource for measuring execution time.
     */
    interface Timer extends AutoCloseable {
        /** Get elapsed time since timer started */
        Duration getElapsed();

        /** Stop the timer and record the measurement */
        @Override
        void close();

        /** Get the metric name being timed */
        String getMetricName();

        /** Get the context for this timer */
        Context getContext();
    }

    /**
     * Memory usage snapshot.
     */
    interface MemorySnapshot {
        /** Get timestamp of snapshot */
        Instant getTimestamp();

        /** Get total heap memory used in bytes */
        long getHeapUsed();

        /** Get maximum heap memory available in bytes */
        long getHeapMax();

        /** Get non-heap memory used in bytes */
        long getNonHeapUsed();

        /** Get number of active threads */
        int getThreadCount();

        /** Get number of garbage collection cycles */
        long getGcCount();

        /** Get time spent in garbage collection */
        Duration getGcTime();

        /** Calculate heap utilization percentage */
        default double getHeapUtilization() {
            return getHeapMax() > 0 ? (double) getHeapUsed() / getHeapMax() * 100.0 : 0.0;
        }
    }

    /**
     * Performance statistics for a metric.
     */
    interface PerformanceStatistics {
        /** Get metric name */
        String getMetricName();

        /** Get metric type */
        MetricType getMetricType();

        /** Get sample count */
        long getSampleCount();

        /** Get minimum value */
        double getMin();

        /** Get maximum value */
        double getMax();

        /** Get average value */
        double getAverage();

        /** Get median value */
        double getMedian();

        /** Get standard deviation */
        double getStandardDeviation();

        /** Get 95th percentile */
        double getP95();

        /** Get 99th percentile */
        double getP99();

        /** Get sum of all values */
        double getSum();

        /** Get rate per second */
        double getRate();

        /** Get time range of measurements */
        TimeRange getTimeRange();

        /** Get last recorded value */
        Optional<Double> getLastValue();

        /** Get last measurement time */
        Optional<Instant> getLastMeasurementTime();
    }

    /**
     * Time range specification.
     */
    record TimeRange(Instant start, Instant end) {
        public TimeRange {
            if (start.isAfter(end)) {
                throw new IllegalArgumentException("Start time must be before end time");
            }
        }

        /** Get duration of this time range */
        public Duration getDuration() {
            return Duration.between(start, end);
        }

        /** Check if timestamp is within this range */
        public boolean contains(Instant timestamp) {
            return !timestamp.isBefore(start) && !timestamp.isAfter(end);
        }
    }

    /**
     * Trend directions for performance analysis.
     */
    enum TrendDirection {
        IMPROVING, DEGRADING, STABLE, OSCILLATING, UNKNOWN
    }

    /**
     * Comprehensive performance report.
     */
    interface PerformanceReport {
        /** Get report generation time */
        Instant getGenerationTime();

        /** Get time range covered by report */
        TimeRange getTimeRange();

        /** Get summary statistics */
        Map<String, Object> getSummary();

        /** Get detailed metrics by context */
        Map<Context, Map<String, PerformanceStatistics>> getDetailedMetrics();

        /** Get memory usage analysis */
        MemoryAnalysis getMemoryAnalysis();

        /** Get timing analysis */
        TimingAnalysis getTimingAnalysis();

        /** Get throughput analysis */
        ThroughputAnalysis getThroughputAnalysis();

        /** Get identified performance issues */
        List<PerformanceIssue> getIssues();

        /** Get recommendations for improvement */
        List<String> getRecommendations();

        /** Export report in specified format */
        Optional<Object> export(String format);
    }

    /**
     * Performance trend analysis.
     */
    interface PerformanceTrend {
        /** Get metric name */
        String getMetricName();

        /** Get trend direction */
        TrendDirection getDirection();

        /** Get trend strength (0.0 to 1.0) */
        double getStrength();

        /** Get correlation coefficient */
        double getCorrelation();

        /** Get slope of trend line */
        double getSlope();

        /** Get predicted future values */
        List<Double> getPredictions(int futurePoints);

        /** Get confidence in predictions */
        double getPredictionConfidence();
    }

    /**
     * Alert handler for performance thresholds.
     */
    interface AlertHandler {
        /** Handle alert when threshold is exceeded */
        void onAlert(String metricName, double currentValue, double threshold,
                    ThresholdComparison comparison, Context context);

        /** Handle alert resolution when metric returns to normal */
        default void onAlertResolved(String metricName, double currentValue, double threshold,
                                   ThresholdComparison comparison, Context context) {
            // Default: no action
        }
    }

    /**
     * Threshold comparison operators for alerts.
     */
    enum ThresholdComparison {
        GREATER_THAN, LESS_THAN, EQUAL_TO, NOT_EQUAL_TO,
        GREATER_THAN_OR_EQUAL, LESS_THAN_OR_EQUAL
    }

    /**
     * Performance tracking overhead analysis.
     */
    interface TrackingOverhead {
        /** Get time overhead per measurement */
        Duration getTimeOverheadPerMeasurement();

        /** Get memory overhead in bytes */
        long getMemoryOverhead();

        /** Get CPU overhead percentage */
        double getCpuOverhead();

        /** Get total measurements processed */
        long getTotalMeasurements();

        /** Get tracking efficiency score */
        double getEfficiencyScore();
    }

    /**
     * Memory usage analysis.
     */
    interface MemoryAnalysis {
        /** Get peak memory usage */
        long getPeakMemoryUsage();

        /** Get average memory usage */
        double getAverageMemoryUsage();

        /** Get memory usage trend */
        TrendDirection getMemoryTrend();

        /** Get garbage collection impact */
        GcAnalysis getGcAnalysis();

        /** Get memory leak indicators */
        List<String> getMemoryLeakIndicators();

        /** Garbage collection analysis */
        interface GcAnalysis {
            /** Get total GC time */
            Duration getTotalGcTime();

            /** Get average GC pause time */
            Duration getAverageGcPause();

            /** Get GC frequency */
            double getGcFrequency();

            /** Get GC efficiency */
            double getGcEfficiency();
        }
    }

    /**
     * Timing analysis.
     */
    interface TimingAnalysis {
        /** Get slowest operations */
        List<String> getSlowestOperations();

        /** Get operations with highest variance */
        List<String> getHighVarianceOperations();

        /** Get timing bottlenecks */
        List<String> getBottlenecks();

        /** Get timing efficiency by context */
        Map<Context, Double> getEfficiencyByContext();
    }

    /**
     * Throughput analysis.
     */
    interface ThroughputAnalysis {
        /** Get peak throughput achieved */
        double getPeakThroughput();

        /** Get average throughput */
        double getAverageThroughput();

        /** Get throughput trend */
        TrendDirection getThroughputTrend();

        /** Get throughput by time of day */
        Map<Integer, Double> getThroughputByHour();

        /** Get operations per second by context */
        Map<Context, Double> getOpsPerSecondByContext();
    }

    /**
     * Performance issue identification.
     */
    interface PerformanceIssue {
        /** Get issue severity */
        IssueSeverity getSeverity();

        /** Get issue category */
        IssueCategory getCategory();

        /** Get issue description */
        String getDescription();

        /** Get affected metrics */
        List<String> getAffectedMetrics();

        /** Get suggested resolution */
        String getSuggestedResolution();

        /** Get issue detection time */
        Instant getDetectionTime();

        /** Issue severity levels */
        enum IssueSeverity {
            LOW, MEDIUM, HIGH, CRITICAL
        }

        /** Issue categories */
        enum IssueCategory {
            MEMORY_LEAK, HIGH_LATENCY, LOW_THROUGHPUT, CPU_BOTTLENECK,
            RESOURCE_CONTENTION, CONFIGURATION, ALGORITHM_INEFFICIENCY
        }
    }
}