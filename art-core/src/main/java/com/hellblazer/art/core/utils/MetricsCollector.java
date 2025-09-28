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
import java.util.function.Supplier;

/**
 * Interface for collecting and aggregating metrics from hybrid ART neural networks.
 *
 * MetricsCollector provides a unified system for gathering various types of
 * metrics from ART algorithms, preprocessing pipelines, and hybrid ML components.
 * It supports real-time metric collection, batch processing, and integration
 * with monitoring systems.
 *
 * Key collection capabilities:
 * - Real-time metric streaming
 * - Batch metric processing
 * - Custom metric definitions
 * - Metric transformation and aggregation
 * - Time-series data management
 * - Multi-dimensional metric labeling
 * - Distributed metric collection
 *
 * @author Hal Hildebrand
 * @since 1.0
 */
public interface MetricsCollector extends AutoCloseable {

    /**
     * Metric value types supported by the collector.
     */
    enum MetricValueType {
        /** Numeric gauge value (current measurement) */
        GAUGE,
        /** Monotonically increasing counter */
        COUNTER,
        /** Timer/duration measurements */
        TIMER,
        /** Histogram of value distributions */
        HISTOGRAM,
        /** Set of unique values */
        SET,
        /** Boolean/binary values */
        BOOLEAN,
        /** String/categorical values */
        STRING,
        /** Vector/array values */
        VECTOR,
        /** Custom complex objects */
        OBJECT
    }

    /**
     * Metric collection modes.
     */
    enum CollectionMode {
        /** Collect metrics immediately when recorded */
        IMMEDIATE,
        /** Buffer metrics and collect in batches */
        BATCHED,
        /** Collect metrics asynchronously */
        ASYNC,
        /** Sample metrics at regular intervals */
        SAMPLED,
        /** Collect metrics on-demand only */
        ON_DEMAND
    }

    /**
     * Metric data retention policies.
     */
    enum RetentionPolicy {
        /** Keep metrics for specified duration */
        TIME_BASED,
        /** Keep specified number of metric points */
        COUNT_BASED,
        /** Keep metrics until memory limit */
        MEMORY_BASED,
        /** Keep all metrics indefinitely */
        UNLIMITED,
        /** Custom retention logic */
        CUSTOM
    }

    /**
     * Register a new metric for collection.
     *
     * @param metricName unique name for the metric
     * @param valueType type of values this metric will contain
     * @param description human-readable description
     * @param labels optional labels for metric categorization
     * @return metric registration ID
     */
    String registerMetric(String metricName, MetricValueType valueType,
                         String description, Map<String, String> labels);

    /**
     * Register a metric with default labels.
     *
     * @param metricName unique name for the metric
     * @param valueType type of values this metric will contain
     * @param description human-readable description
     * @return metric registration ID
     */
    default String registerMetric(String metricName, MetricValueType valueType, String description) {
        return registerMetric(metricName, valueType, description, Map.of());
    }

    /**
     * Unregister a metric to stop collection.
     *
     * @param metricId the metric registration ID
     * @return true if metric was found and unregistered
     */
    boolean unregisterMetric(String metricId);

    /**
     * Collect a single metric value.
     *
     * @param metricName name of the metric
     * @param value the metric value
     * @param timestamp when the value was measured
     * @param labels additional labels for this measurement
     */
    void collect(String metricName, Object value, Instant timestamp, Map<String, String> labels);

    /**
     * Collect a metric value with current timestamp.
     *
     * @param metricName name of the metric
     * @param value the metric value
     * @param labels additional labels for this measurement
     */
    default void collect(String metricName, Object value, Map<String, String> labels) {
        collect(metricName, value, Instant.now(), labels);
    }

    /**
     * Collect a metric value with no additional labels.
     *
     * @param metricName name of the metric
     * @param value the metric value
     */
    default void collect(String metricName, Object value) {
        collect(metricName, value, Map.of());
    }

    /**
     * Collect multiple metrics atomically.
     *
     * @param metricBatch batch of metrics to collect
     * @param timestamp common timestamp for all metrics
     */
    void collectBatch(Map<String, Object> metricBatch, Instant timestamp);

    /**
     * Collect multiple metrics with current timestamp.
     *
     * @param metricBatch batch of metrics to collect
     */
    default void collectBatch(Map<String, Object> metricBatch) {
        collectBatch(metricBatch, Instant.now());
    }

    /**
     * Set up periodic collection from a metric supplier.
     *
     * @param metricName name of the metric
     * @param supplier function that provides metric values
     * @param interval collection interval
     * @param labels labels for collected values
     * @return collection job ID for cancellation
     */
    String scheduleCollection(String metricName, Supplier<Object> supplier,
                             Duration interval, Map<String, String> labels);

    /**
     * Cancel a scheduled collection job.
     *
     * @param jobId the collection job ID
     * @return true if job was found and cancelled
     */
    boolean cancelCollection(String jobId);

    /**
     * Retrieve collected metric values within a time range.
     *
     * @param metricName name of the metric
     * @param startTime start of time range (inclusive)
     * @param endTime end of time range (exclusive)
     * @param labels optional label filters
     * @return list of metric values in chronological order
     */
    List<MetricValue> getMetricValues(String metricName, Instant startTime, Instant endTime,
                                     Map<String, String> labels);

    /**
     * Retrieve recent metric values.
     *
     * @param metricName name of the metric
     * @param count maximum number of recent values to return
     * @param labels optional label filters
     * @return list of recent metric values, most recent first
     */
    List<MetricValue> getRecentValues(String metricName, int count, Map<String, String> labels);

    /**
     * Get the latest value for a metric.
     *
     * @param metricName name of the metric
     * @param labels optional label filters
     * @return optional latest metric value
     */
    Optional<MetricValue> getLatestValue(String metricName, Map<String, String> labels);

    /**
     * Get aggregated statistics for a metric over a time range.
     *
     * @param metricName name of the metric
     * @param startTime start of time range
     * @param endTime end of time range
     * @param aggregations list of aggregation functions to apply
     * @param labels optional label filters
     * @return aggregated metric statistics
     */
    MetricAggregation getAggregation(String metricName, Instant startTime, Instant endTime,
                                   List<AggregationFunction> aggregations,
                                   Map<String, String> labels);

    /**
     * Query metrics using flexible criteria.
     *
     * @param query metric query specification
     * @return query result containing matching metrics
     */
    MetricQueryResult query(MetricQuery query);

    /**
     * Get information about all registered metrics.
     *
     * @return map of metric names to their information
     */
    Map<String, MetricInfo> getRegisteredMetrics();

    /**
     * Get information about a specific metric.
     *
     * @param metricName name of the metric
     * @return optional metric information
     */
    Optional<MetricInfo> getMetricInfo(String metricName);

    /**
     * Set the collection mode for all metrics.
     *
     * @param mode the collection mode to use
     */
    void setCollectionMode(CollectionMode mode);

    /**
     * Get the current collection mode.
     *
     * @return current collection mode
     */
    CollectionMode getCollectionMode();

    /**
     * Set the retention policy for metric data.
     *
     * @param policy the retention policy to apply
     * @param parameter policy-specific parameter (duration, count, etc.)
     */
    void setRetentionPolicy(RetentionPolicy policy, Object parameter);

    /**
     * Get the current retention policy.
     *
     * @return current retention policy information
     */
    RetentionPolicyInfo getRetentionPolicy();

    /**
     * Flush any buffered metrics immediately.
     *
     * @return future that completes when flush is done
     */
    CompletableFuture<Void> flush();

    /**
     * Clear all collected metric data.
     */
    void clear();

    /**
     * Clear metric data older than specified age.
     *
     * @param maxAge maximum age of data to retain
     * @return number of metric values removed
     */
    long clearOldData(Duration maxAge);

    /**
     * Export collected metrics in specified format.
     *
     * @param format export format (JSON, CSV, Prometheus, etc.)
     * @param timeRange optional time range to export
     * @param metricFilter optional metric name filter
     * @return exported metric data
     */
    Optional<Object> export(String format, Optional<TimeRange> timeRange,
                          Optional<String> metricFilter);

    /**
     * Import metrics from external source.
     *
     * @param data metric data to import
     * @param format data format
     * @return number of metrics successfully imported
     */
    int importMetrics(Object data, String format);

    /**
     * Get collector statistics and health information.
     *
     * @return collector statistics
     */
    CollectorStatistics getStatistics();

    /**
     * Enable or disable metric collection.
     *
     * @param enabled whether to enable collection
     */
    void setEnabled(boolean enabled);

    /**
     * Check if metric collection is enabled.
     *
     * @return true if collection is enabled
     */
    boolean isEnabled();

    /**
     * Set up metric streaming to external systems.
     *
     * @param streamName unique name for the stream
     * @param destination stream destination configuration
     * @param metricFilter filter for which metrics to stream
     * @return stream ID for management
     */
    String setupStream(String streamName, StreamDestination destination, MetricFilter metricFilter);

    /**
     * Stop a metric stream.
     *
     * @param streamId the stream ID to stop
     * @return true if stream was found and stopped
     */
    boolean stopStream(String streamId);

    /**
     * Release resources used by the metrics collector.
     */
    @Override
    void close();

    /**
     * Represents a collected metric value with metadata.
     */
    interface MetricValue {
        /** Get the metric name */
        String getMetricName();

        /** Get the metric value */
        Object getValue();

        /** Get measurement timestamp */
        Instant getTimestamp();

        /** Get metric labels */
        Map<String, String> getLabels();

        /** Get metric value type */
        MetricValueType getValueType();

        /** Get value as double (for numeric types) */
        default Optional<Double> getAsDouble() {
            if (getValue() instanceof Number n) {
                return Optional.of(n.doubleValue());
            }
            return Optional.empty();
        }

        /** Get value as string */
        default String getAsString() {
            return getValue().toString();
        }

        /** Check if value matches label criteria */
        default boolean matchesLabels(Map<String, String> labelFilter) {
            return labelFilter.entrySet().stream()
                             .allMatch(entry -> entry.getValue().equals(getLabels().get(entry.getKey())));
        }
    }

    /**
     * Information about a registered metric.
     */
    interface MetricInfo {
        /** Get metric name */
        String getName();

        /** Get metric value type */
        MetricValueType getValueType();

        /** Get metric description */
        String getDescription();

        /** Get default labels */
        Map<String, String> getDefaultLabels();

        /** Get registration timestamp */
        Instant getRegistrationTime();

        /** Get number of values collected */
        long getValueCount();

        /** Get last collection timestamp */
        Optional<Instant> getLastCollectionTime();

        /** Get memory usage for this metric */
        long getMemoryUsage();

        /** Check if metric is actively being collected */
        boolean isActive();
    }

    /**
     * Aggregated metric statistics.
     */
    interface MetricAggregation {
        /** Get metric name */
        String getMetricName();

        /** Get time range of aggregation */
        TimeRange getTimeRange();

        /** Get aggregated values by function */
        Map<AggregationFunction, Double> getValues();

        /** Get sample count */
        long getSampleCount();

        /** Get aggregation labels */
        Map<String, String> getLabels();

        /** Get specific aggregated value */
        default Optional<Double> getValue(AggregationFunction function) {
            return Optional.ofNullable(getValues().get(function));
        }
    }

    /**
     * Aggregation functions for metric statistics.
     */
    enum AggregationFunction {
        SUM, AVERAGE, MIN, MAX, MEDIAN, COUNT, STDDEV,
        P90, P95, P99, RATE, FIRST, LAST
    }

    /**
     * Query specification for flexible metric retrieval.
     */
    interface MetricQuery {
        /** Get metric name pattern (supports wildcards) */
        String getMetricPattern();

        /** Get time range for query */
        TimeRange getTimeRange();

        /** Get label filters */
        Map<String, String> getLabelFilters();

        /** Get aggregation functions to apply */
        List<AggregationFunction> getAggregations();

        /** Get result limit */
        Optional<Integer> getLimit();

        /** Get result ordering */
        Optional<QueryOrdering> getOrdering();

        /** Query result ordering */
        enum QueryOrdering {
            TIMESTAMP_ASC, TIMESTAMP_DESC, VALUE_ASC, VALUE_DESC, METRIC_NAME
        }
    }

    /**
     * Result of a metric query.
     */
    interface MetricQueryResult {
        /** Get matching metric values */
        List<MetricValue> getValues();

        /** Get aggregated results if requested */
        Map<String, MetricAggregation> getAggregations();

        /** Get query execution time */
        Duration getExecutionTime();

        /** Get number of total matches (before limit) */
        long getTotalMatches();

        /** Check if result was truncated due to limit */
        boolean isTruncated();
    }

    /**
     * Time range specification for queries.
     */
    record TimeRange(Instant start, Instant end) {
        public TimeRange {
            if (start.isAfter(end)) {
                throw new IllegalArgumentException("Start must be before end");
            }
        }

        /** Get duration of this range */
        public Duration getDuration() {
            return Duration.between(start, end);
        }

        /** Check if timestamp is in range */
        public boolean contains(Instant timestamp) {
            return !timestamp.isBefore(start) && !timestamp.isAfter(end);
        }

        /** Create range for last N duration */
        public static TimeRange lastDuration(Duration duration) {
            var end = Instant.now();
            var start = end.minus(duration);
            return new TimeRange(start, end);
        }
    }

    /**
     * Retention policy information.
     */
    interface RetentionPolicyInfo {
        /** Get retention policy type */
        RetentionPolicy getPolicy();

        /** Get policy parameter */
        Object getParameter();

        /** Get estimated data size */
        long getEstimatedDataSize();

        /** Get oldest retained data timestamp */
        Optional<Instant> getOldestDataTime();

        /** Get retention effectiveness */
        double getRetentionEffectiveness();
    }

    /**
     * Collector performance and health statistics.
     */
    interface CollectorStatistics {
        /** Get total metrics collected */
        long getTotalMetricsCollected();

        /** Get collection rate (metrics per second) */
        double getCollectionRate();

        /** Get memory usage */
        long getMemoryUsage();

        /** Get active metric count */
        int getActiveMetricCount();

        /** Get scheduled collection job count */
        int getScheduledJobCount();

        /** Get collection errors */
        long getCollectionErrors();

        /** Get flush statistics */
        FlushStatistics getFlushStatistics();

        /** Get retention statistics */
        RetentionStatistics getRetentionStatistics();

        /** Flush operation statistics */
        interface FlushStatistics {
            /** Get total flush operations */
            long getTotalFlushes();

            /** Get average flush time */
            Duration getAverageFlushTime();

            /** Get last flush time */
            Optional<Instant> getLastFlushTime();

            /** Get flush errors */
            long getFlushErrors();
        }

        /** Data retention statistics */
        interface RetentionStatistics {
            /** Get total retention operations */
            long getTotalRetentions();

            /** Get total data points removed */
            long getTotalDataPointsRemoved();

            /** Get last retention time */
            Optional<Instant> getLastRetentionTime();

            /** Get retention efficiency */
            double getRetentionEfficiency();
        }
    }

    /**
     * Stream destination configuration.
     */
    interface StreamDestination {
        /** Get destination type */
        String getType();

        /** Get destination configuration */
        Map<String, Object> getConfiguration();

        /** Get stream format */
        String getFormat();

        /** Get buffer size for streaming */
        int getBufferSize();

        /** Get stream flush interval */
        Duration getFlushInterval();
    }

    /**
     * Filter for selecting metrics to stream or process.
     */
    interface MetricFilter {
        /** Get metric name patterns to include */
        List<String> getIncludePatterns();

        /** Get metric name patterns to exclude */
        List<String> getExcludePatterns();

        /** Get label filters */
        Map<String, String> getLabelFilters();

        /** Get value type filters */
        List<MetricValueType> getValueTypeFilters();

        /** Check if metric matches filter */
        boolean matches(String metricName, Map<String, String> labels, MetricValueType valueType);
    }
}