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
import java.util.function.Function;

/**
 * Interface for monitoring convergence of learning algorithms and optimization processes
 * in hybrid ART neural networks.
 *
 * ConvergenceMonitor tracks various metrics and criteria to determine when learning
 * algorithms have converged to stable solutions. This is essential for adaptive
 * learning systems that need to know when to stop training, switch strategies,
 * or trigger consolidation processes.
 *
 * Key monitoring capabilities:
 * - Loss function convergence
 * - Parameter stability tracking
 * - Performance metric plateaus
 * - Early stopping criteria
 * - Multi-objective convergence
 * - Convergence rate analysis
 * - Overfitting detection
 *
 * @author Hal Hildebrand
 * @since 1.0
 */
public interface ConvergenceMonitor extends AutoCloseable {

    /**
     * Types of convergence criteria.
     */
    enum ConvergenceType {
        /** Loss function convergence */
        LOSS_CONVERGENCE,
        /** Parameter change convergence */
        PARAMETER_CONVERGENCE,
        /** Performance metric convergence */
        PERFORMANCE_CONVERGENCE,
        /** Gradient magnitude convergence */
        GRADIENT_CONVERGENCE,
        /** Relative improvement convergence */
        RELATIVE_IMPROVEMENT,
        /** Absolute improvement convergence */
        ABSOLUTE_IMPROVEMENT,
        /** Statistical convergence */
        STATISTICAL_CONVERGENCE,
        /** Custom user-defined convergence */
        CUSTOM
    }

    /**
     * Convergence detection strategies.
     */
    enum ConvergenceStrategy {
        /** Simple threshold-based detection */
        THRESHOLD,
        /** Moving average-based detection */
        MOVING_AVERAGE,
        /** Trend analysis-based detection */
        TREND_ANALYSIS,
        /** Statistical test-based detection */
        STATISTICAL_TEST,
        /** Plateau detection */
        PLATEAU_DETECTION,
        /** Early stopping with patience */
        EARLY_STOPPING,
        /** Ensemble of multiple strategies */
        ENSEMBLE
    }

    /**
     * Monitor states for tracking convergence progress.
     */
    enum MonitorState {
        /** Initial state, not enough data */
        INITIALIZING,
        /** Actively monitoring for convergence */
        MONITORING,
        /** Convergence detected */
        CONVERGED,
        /** Divergence detected */
        DIVERGED,
        /** Oscillating around solution */
        OSCILLATING,
        /** Stalled with no progress */
        STALLED,
        /** Monitoring stopped/paused */
        STOPPED
    }

    /**
     * Add a convergence criterion to monitor.
     *
     * @param name unique name for the criterion
     * @param type type of convergence to monitor
     * @param strategy detection strategy to use
     * @param threshold convergence threshold value
     * @param windowSize number of recent observations to consider
     * @return criterion ID for future reference
     */
    String addCriterion(String name, ConvergenceType type, ConvergenceStrategy strategy,
                       double threshold, int windowSize);

    /**
     * Remove a convergence criterion.
     *
     * @param criterionId the criterion ID to remove
     * @return true if criterion was found and removed
     */
    boolean removeCriterion(String criterionId);

    /**
     * Update a metric value for convergence monitoring.
     *
     * @param metricName name of the metric
     * @param value current metric value
     * @param timestamp when the value was observed
     */
    void updateMetric(String metricName, double value, Instant timestamp);

    /**
     * Update a metric value with current timestamp.
     *
     * @param metricName name of the metric
     * @param value current metric value
     */
    default void updateMetric(String metricName, double value) {
        updateMetric(metricName, value, Instant.now());
    }

    /**
     * Update multiple metrics at once.
     *
     * @param metrics map of metric names to values
     * @param timestamp when the values were observed
     */
    default void updateMetrics(Map<String, Double> metrics, Instant timestamp) {
        metrics.forEach((name, value) -> updateMetric(name, value, timestamp));
    }

    /**
     * Update multiple metrics with current timestamp.
     *
     * @param metrics map of metric names to values
     */
    default void updateMetrics(Map<String, Double> metrics) {
        updateMetrics(metrics, Instant.now());
    }

    /**
     * Check if any convergence criteria have been met.
     *
     * @return true if convergence has been detected
     */
    boolean hasConverged();

    /**
     * Check if specific convergence criterion has been met.
     *
     * @param criterionId the criterion ID to check
     * @return true if the specific criterion has converged
     */
    boolean hasConverged(String criterionId);

    /**
     * Check if divergence has been detected.
     *
     * @return true if divergence is detected
     */
    boolean hasDiverged();

    /**
     * Get the current monitoring state.
     *
     * @return current monitor state
     */
    MonitorState getState();

    /**
     * Get detailed convergence status for all criteria.
     *
     * @return convergence status report
     */
    ConvergenceStatus getConvergenceStatus();

    /**
     * Get convergence status for a specific criterion.
     *
     * @param criterionId the criterion ID
     * @return optional convergence status for the criterion
     */
    Optional<CriterionStatus> getCriterionStatus(String criterionId);

    /**
     * Get the time elapsed since monitoring started.
     *
     * @return duration since monitoring began
     */
    Duration getElapsedTime();

    /**
     * Get the number of iterations/updates processed.
     *
     * @return total number of metric updates
     */
    long getIterationCount();

    /**
     * Estimate time to convergence based on current trends.
     *
     * @return optional estimated time to convergence
     */
    Optional<Duration> estimateTimeToConvergence();

    /**
     * Get convergence rate for a specific metric.
     *
     * @param metricName the metric name
     * @return optional convergence rate (units per iteration)
     */
    Optional<Double> getConvergenceRate(String metricName);

    /**
     * Set early stopping patience (iterations to wait after best result).
     *
     * @param patience number of iterations to wait
     */
    void setEarlyStoppingPatience(int patience);

    /**
     * Get current early stopping patience.
     *
     * @return early stopping patience value
     */
    int getEarlyStoppingPatience();

    /**
     * Enable or disable overfitting detection.
     *
     * @param enabled whether to enable overfitting detection
     */
    void setOverfittingDetectionEnabled(boolean enabled);

    /**
     * Check if overfitting has been detected.
     *
     * @return true if overfitting is detected
     */
    boolean isOverfittingDetected();

    /**
     * Set custom convergence function for advanced convergence detection.
     *
     * @param criterionId the criterion ID
     * @param function custom convergence detection function
     */
    void setCustomConvergenceFunction(String criterionId,
                                    Function<List<Double>, Boolean> function);

    /**
     * Reset monitoring state and clear all metric history.
     */
    void reset();

    /**
     * Pause monitoring (stop updating but retain state).
     */
    void pause();

    /**
     * Resume monitoring after pause.
     */
    void resume();

    /**
     * Get monitoring configuration.
     *
     * @return current configuration settings
     */
    MonitorConfiguration getConfiguration();

    /**
     * Update monitoring configuration.
     *
     * @param configuration new configuration
     */
    void updateConfiguration(MonitorConfiguration configuration);

    /**
     * Export monitoring history for analysis.
     *
     * @param format export format (implementation-specific)
     * @return optional exported data
     */
    Optional<Object> exportHistory(String format);

    /**
     * Get memory usage of the monitor.
     *
     * @return optional memory usage in bytes
     */
    Optional<Long> getMemoryUsage();

    /**
     * Release resources used by the monitor.
     */
    @Override
    void close();

    /**
     * Overall convergence status across all criteria.
     */
    interface ConvergenceStatus {
        /** Check if overall convergence achieved */
        boolean isConverged();

        /** Check if divergence detected */
        boolean isDiverged();

        /** Get current monitor state */
        MonitorState getState();

        /** Get status for each criterion */
        Map<String, CriterionStatus> getCriterionStatuses();

        /** Get overall progress estimate [0.0, 1.0] */
        double getProgress();

        /** Get best metric values achieved so far */
        Map<String, Double> getBestValues();

        /** Get current metric values */
        Map<String, Double> getCurrentValues();

        /** Get convergence summary message */
        String getSummaryMessage();

        /** Get time when convergence was achieved */
        Optional<Instant> getConvergenceTime();

        /** Get iteration when convergence was achieved */
        Optional<Long> getConvergenceIteration();
    }

    /**
     * Status for individual convergence criterion.
     */
    interface CriterionStatus {
        /** Get criterion ID */
        String getCriterionId();

        /** Get criterion name */
        String getName();

        /** Get convergence type */
        ConvergenceType getType();

        /** Get detection strategy */
        ConvergenceStrategy getStrategy();

        /** Check if criterion has converged */
        boolean isConverged();

        /** Get current metric value */
        double getCurrentValue();

        /** Get target threshold */
        double getThreshold();

        /** Get progress toward threshold [0.0, 1.0] */
        double getProgress();

        /** Get recent metric values */
        List<Double> getRecentValues();

        /** Get convergence rate estimate */
        Optional<Double> getConvergenceRate();

        /** Get time since last improvement */
        Duration getTimeSinceImprovement();

        /** Get iterations since last improvement */
        long getIterationsSinceImprovement();

        /** Check if criterion is stalled */
        boolean isStalled();
    }

    /**
     * Configuration for convergence monitoring.
     */
    interface MonitorConfiguration {
        /** Get maximum number of iterations to monitor */
        Optional<Long> getMaxIterations();

        /** Get maximum time to monitor */
        Optional<Duration> getMaxTime();

        /** Get minimum number of iterations before convergence check */
        int getMinIterations();

        /** Get history retention policy */
        HistoryRetentionPolicy getHistoryRetention();

        /** Get overfitting detection settings */
        OverfittingDetectionSettings getOverfittingDetection();

        /** Get logging/reporting configuration */
        ReportingConfiguration getReporting();

        /** History retention policies */
        interface HistoryRetentionPolicy {
            /** Maximum number of values to retain per metric */
            int getMaxHistorySize();

            /** Whether to compress old history */
            boolean isCompressionEnabled();

            /** Compression strategy for old data */
            CompressionStrategy getCompressionStrategy();

            enum CompressionStrategy {
                NONE, SAMPLING, MOVING_AVERAGE, EXPONENTIAL_DECAY
            }
        }

        /** Overfitting detection configuration */
        interface OverfittingDetectionSettings {
            /** Whether overfitting detection is enabled */
            boolean isEnabled();

            /** Validation metric name for overfitting detection */
            String getValidationMetric();

            /** Training metric name for comparison */
            String getTrainingMetric();

            /** Threshold for validation/training gap */
            double getGapThreshold();

            /** Window size for overfitting detection */
            int getWindowSize();
        }

        /** Reporting and logging configuration */
        interface ReportingConfiguration {
            /** Whether to enable periodic reporting */
            boolean isPeriodicReportingEnabled();

            /** Reporting frequency (iterations) */
            int getReportingFrequency();

            /** Metrics to include in reports */
            List<String> getReportedMetrics();

            /** Whether to log convergence events */
            boolean isEventLoggingEnabled();
        }
    }

    /**
     * Builder for creating convergence monitors with fluent API.
     */
    interface Builder {
        /** Set maximum iterations */
        Builder withMaxIterations(long maxIterations);

        /** Set maximum monitoring time */
        Builder withMaxTime(Duration maxTime);

        /** Set minimum iterations before convergence checking */
        Builder withMinIterations(int minIterations);

        /** Add a loss convergence criterion */
        Builder withLossConvergence(String name, double threshold, int windowSize);

        /** Add a parameter convergence criterion */
        Builder withParameterConvergence(String name, double threshold, int windowSize);

        /** Add early stopping with patience */
        Builder withEarlyStopping(int patience);

        /** Enable overfitting detection */
        Builder withOverfittingDetection(String validationMetric, String trainingMetric,
                                       double gapThreshold);

        /** Enable periodic reporting */
        Builder withPeriodicReporting(int frequency);

        /** Build the convergence monitor */
        ConvergenceMonitor build();
    }

    /**
     * Create a new convergence monitor builder.
     *
     * @return new builder instance
     */
    static Builder builder() {
        // Default implementation would be provided by implementing classes
        throw new UnsupportedOperationException("Must be implemented by concrete classes");
    }
}