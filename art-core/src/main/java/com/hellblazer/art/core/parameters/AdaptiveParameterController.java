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
package com.hellblazer.art.core.parameters;

import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;

/**
 * Interface for adaptive parameter control in hybrid ART neural networks.
 *
 * AdaptiveParameterController automatically adjusts algorithm parameters
 * based on performance feedback, data characteristics, and learning progress.
 * This enables ART systems to self-optimize and adapt to changing conditions
 * without manual parameter tuning.
 *
 * Key capabilities:
 * - Automatic parameter optimization
 * - Performance-based parameter adaptation
 * - Multi-objective parameter balancing
 * - Learning schedule management
 * - Dynamic parameter constraints
 * - Parameter sensitivity analysis
 * - Adaptive hyperparameter search
 *
 * @author Hal Hildebrand
 * @since 1.0
 */
public interface AdaptiveParameterController extends AutoCloseable {

    /**
     * Parameter adaptation strategies.
     */
    enum AdaptationStrategy {
        /** Gradient-based optimization */
        GRADIENT_BASED,
        /** Evolutionary/genetic algorithm optimization */
        EVOLUTIONARY,
        /** Bayesian optimization */
        BAYESIAN,
        /** Grid search optimization */
        GRID_SEARCH,
        /** Random search optimization */
        RANDOM_SEARCH,
        /** Reinforcement learning-based adaptation */
        REINFORCEMENT_LEARNING,
        /** Multi-armed bandit approach */
        BANDIT,
        /** Schedule-based adaptation */
        SCHEDULED,
        /** Rule-based adaptation */
        RULE_BASED,
        /** Ensemble of multiple strategies */
        ENSEMBLE
    }

    /**
     * Parameter update policies.
     */
    enum UpdatePolicy {
        /** Update parameters immediately when triggered */
        IMMEDIATE,
        /** Batch updates at regular intervals */
        BATCH,
        /** Smooth/gradual parameter changes */
        GRADUAL,
        /** Conservative updates with validation */
        CONSERVATIVE,
        /** Aggressive updates for rapid adaptation */
        AGGRESSIVE,
        /** Conditional updates based on criteria */
        CONDITIONAL
    }

    /**
     * Parameter constraint types.
     */
    enum ConstraintType {
        /** Hard bounds that cannot be violated */
        HARD_BOUNDS,
        /** Soft bounds with penalty */
        SOFT_BOUNDS,
        /** Relative constraints between parameters */
        RELATIVE,
        /** Conditional constraints based on other parameters */
        CONDITIONAL,
        /** Dynamic constraints that change over time */
        DYNAMIC,
        /** Statistical constraints based on distributions */
        STATISTICAL
    }

    /**
     * Register a parameter for adaptive control.
     *
     * @param parameterName unique name for the parameter
     * @param initialValue initial parameter value
     * @param constraints parameter constraints and bounds
     * @param adaptationStrategy strategy for adapting this parameter
     * @return parameter registration ID
     */
    String registerParameter(String parameterName, Object initialValue,
                            ParameterConstraints constraints, AdaptationStrategy adaptationStrategy);

    /**
     * Unregister a parameter from adaptive control.
     *
     * @param parameterId the parameter registration ID
     * @return true if parameter was found and unregistered
     */
    boolean unregisterParameter(String parameterId);

    /**
     * Get the current value of a parameter.
     *
     * @param parameterName name of the parameter
     * @return optional current parameter value
     */
    Optional<Object> getParameterValue(String parameterName);

    /**
     * Get current values of all registered parameters.
     *
     * @return map of parameter names to their current values
     */
    Map<String, Object> getAllParameterValues();

    /**
     * Manually set a parameter value (overrides adaptive control temporarily).
     *
     * @param parameterName name of the parameter
     * @param value new parameter value
     * @param overrideDuration how long to override adaptive control
     */
    void setParameterValue(String parameterName, Object value, Duration overrideDuration);

    /**
     * Update the controller with performance feedback.
     *
     * @param performanceMetrics map of metric names to values
     * @param timestamp when the metrics were measured
     */
    void updatePerformance(Map<String, Double> performanceMetrics, Instant timestamp);

    /**
     * Update with performance feedback using current timestamp.
     *
     * @param performanceMetrics map of metric names to values
     */
    default void updatePerformance(Map<String, Double> performanceMetrics) {
        updatePerformance(performanceMetrics, Instant.now());
    }

    /**
     * Trigger parameter adaptation based on current performance.
     *
     * @param adaptationContext context for adaptation decisions
     * @return adaptation result with changes made
     */
    AdaptationResult adapt(AdaptationContext adaptationContext);

    /**
     * Get the adaptation strategy for a specific parameter.
     *
     * @param parameterName name of the parameter
     * @return optional adaptation strategy
     */
    Optional<AdaptationStrategy> getAdaptationStrategy(String parameterName);

    /**
     * Update the adaptation strategy for a parameter.
     *
     * @param parameterName name of the parameter
     * @param newStrategy new adaptation strategy
     * @return true if parameter was found and updated
     */
    boolean setAdaptationStrategy(String parameterName, AdaptationStrategy newStrategy);

    /**
     * Get the update policy used by this controller.
     *
     * @return current update policy
     */
    UpdatePolicy getUpdatePolicy();

    /**
     * Set the update policy for parameter changes.
     *
     * @param policy new update policy
     */
    void setUpdatePolicy(UpdatePolicy policy);

    /**
     * Enable or disable adaptive control globally.
     *
     * @param enabled whether to enable adaptive control
     */
    void setAdaptationEnabled(boolean enabled);

    /**
     * Check if adaptive control is enabled.
     *
     * @return true if adaptation is enabled
     */
    boolean isAdaptationEnabled();

    /**
     * Enable or disable adaptation for a specific parameter.
     *
     * @param parameterName name of the parameter
     * @param enabled whether to enable adaptation for this parameter
     */
    void setParameterAdaptationEnabled(String parameterName, boolean enabled);

    /**
     * Check if adaptation is enabled for a specific parameter.
     *
     * @param parameterName name of the parameter
     * @return true if adaptation is enabled for this parameter
     */
    boolean isParameterAdaptationEnabled(String parameterName);

    /**
     * Get parameter adaptation history.
     *
     * @param parameterName name of the parameter
     * @param timeRange optional time range for history
     * @return list of parameter changes over time
     */
    List<ParameterChange> getParameterHistory(String parameterName, Optional<TimeRange> timeRange);

    /**
     * Get optimization suggestions for parameter tuning.
     *
     * @param targetMetrics metrics to optimize for
     * @param optimizationGoals optimization objectives
     * @return list of optimization suggestions
     */
    List<OptimizationSuggestion> getOptimizationSuggestions(List<String> targetMetrics,
                                                           List<OptimizationGoal> optimizationGoals);

    /**
     * Perform parameter sensitivity analysis.
     *
     * @param parameterName name of the parameter
     * @param perturbationRange range of values to test
     * @param targetMetric metric to measure sensitivity against
     * @return sensitivity analysis result
     */
    SensitivityAnalysis performSensitivityAnalysis(String parameterName, ParameterRange perturbationRange,
                                                  String targetMetric);

    /**
     * Set up automatic parameter scheduling.
     *
     * @param parameterName name of the parameter
     * @param schedule parameter change schedule
     * @return schedule ID for management
     */
    String scheduleParameterChanges(String parameterName, ParameterSchedule schedule);

    /**
     * Cancel a parameter schedule.
     *
     * @param scheduleId the schedule ID
     * @return true if schedule was found and cancelled
     */
    boolean cancelSchedule(String scheduleId);

    /**
     * Get controller statistics and performance.
     *
     * @return controller statistics
     */
    ControllerStatistics getStatistics();

    /**
     * Validate current parameter configuration.
     *
     * @return list of validation issues (empty if valid)
     */
    List<String> validateConfiguration();

    /**
     * Reset all parameters to their initial values.
     */
    void reset();

    /**
     * Export parameter configuration and history.
     *
     * @param format export format (JSON, CSV, etc.)
     * @return optional exported data
     */
    Optional<Object> exportConfiguration(String format);

    /**
     * Import parameter configuration.
     *
     * @param configurationData configuration data to import
     * @param format data format
     * @return true if import was successful
     */
    boolean importConfiguration(Object configurationData, String format);

    /**
     * Release resources used by the controller.
     */
    @Override
    void close();

    /**
     * Parameter constraints and bounds.
     */
    interface ParameterConstraints {
        /** Get constraint type */
        ConstraintType getType();

        /** Get minimum allowed value */
        Optional<Object> getMinValue();

        /** Get maximum allowed value */
        Optional<Object> getMaxValue();

        /** Get allowed discrete values */
        Optional<List<Object>> getAllowedValues();

        /** Get custom constraint function */
        Optional<Function<Object, Boolean>> getCustomConstraint();

        /** Get constraint parameters */
        Map<String, Object> getConstraintParameters();

        /** Check if value satisfies constraints */
        boolean isValid(Object value);

        /** Get constraint violation message */
        Optional<String> getViolationMessage(Object value);

        /** Project value to satisfy constraints */
        Object projectToValid(Object value);
    }

    /**
     * Time range specification.
     */
    record TimeRange(Instant start, Instant end) {
        public TimeRange {
            if (start.isAfter(end)) {
                throw new IllegalArgumentException("Start must be before end");
            }
        }

        public Duration getDuration() {
            return Duration.between(start, end);
        }

        public boolean contains(Instant timestamp) {
            return !timestamp.isBefore(start) && !timestamp.isAfter(end);
        }
    }

    /**
     * Parameter value range for optimization and analysis.
     */
    interface ParameterRange {
        /** Get minimum value in range */
        Object getMinValue();

        /** Get maximum value in range */
        Object getMaxValue();

        /** Get number of steps/samples in range */
        int getSteps();

        /** Get specific values to sample */
        List<Object> getSampleValues();

        /** Check if value is in range */
        boolean contains(Object value);
    }

    /**
     * Context for adaptation decisions.
     */
    interface AdaptationContext {
        /** Get current performance metrics */
        Map<String, Double> getCurrentMetrics();

        /** Get performance trends */
        Map<String, PerformanceTrend> getTrends();

        /** Get available computational resources */
        Map<String, Double> getAvailableResources();

        /** Get adaptation goals and priorities */
        List<OptimizationGoal> getGoals();

        /** Get environmental factors */
        Map<String, Object> getEnvironmentInfo();

        /** Get user preferences or constraints */
        Map<String, Object> getUserPreferences();

        /** Performance trend information */
        interface PerformanceTrend {
            /** Get trend direction */
            TrendDirection getDirection();

            /** Get trend strength (0-1) */
            double getStrength();

            /** Get trend duration */
            Duration getDuration();

            /** Get recent values */
            List<Double> getRecentValues();

            enum TrendDirection {
                IMPROVING, DEGRADING, STABLE, OSCILLATING, UNKNOWN
            }
        }
    }

    /**
     * Result of parameter adaptation.
     */
    interface AdaptationResult {
        /** Check if adaptation was successful */
        boolean isSuccessful();

        /** Get parameters that were changed */
        Map<String, ParameterChange> getChanges();

        /** Get expected performance improvement */
        Map<String, Double> getExpectedImprovement();

        /** Get adaptation reasoning */
        String getReasoning();

        /** Get adaptation time */
        Duration getAdaptationTime();

        /** Get any warnings or issues */
        List<String> getWarnings();
    }

    /**
     * Record of a parameter change.
     */
    interface ParameterChange {
        /** Get parameter name */
        String getParameterName();

        /** Get old value */
        Object getOldValue();

        /** Get new value */
        Object getNewValue();

        /** Get change timestamp */
        Instant getTimestamp();

        /** Get reason for change */
        String getReason();

        /** Get adaptation strategy used */
        AdaptationStrategy getStrategy();

        /** Get performance before change */
        Map<String, Double> getPerformanceBefore();

        /** Get performance after change */
        Optional<Map<String, Double>> getPerformanceAfter();

        /** Calculate relative change magnitude */
        default double getChangeMagnitude() {
            if (getOldValue() instanceof Number oldNum && getNewValue() instanceof Number newNum) {
                double oldVal = oldNum.doubleValue();
                double newVal = newNum.doubleValue();
                return oldVal != 0.0 ? Math.abs((newVal - oldVal) / oldVal) : 0.0;
            }
            return 0.0;
        }
    }

    /**
     * Optimization goal specification.
     */
    interface OptimizationGoal {
        /** Get goal type */
        GoalType getType();

        /** Get target metric name */
        String getMetricName();

        /** Get target value */
        Optional<Double> getTargetValue();

        /** Get goal weight/importance */
        double getWeight();

        /** Get tolerance for goal achievement */
        double getTolerance();

        /** Get optimization direction */
        OptimizationDirection getDirection();

        /** Goal types */
        enum GoalType {
            MINIMIZE, MAXIMIZE, TARGET, STABILIZE, BALANCE
        }

        /** Optimization directions */
        enum OptimizationDirection {
            MINIMIZE, MAXIMIZE, NONE
        }
    }

    /**
     * Optimization suggestion for parameter tuning.
     */
    interface OptimizationSuggestion {
        /** Get parameter to adjust */
        String getParameterName();

        /** Get suggested new value */
        Object getSuggestedValue();

        /** Get expected improvement */
        Map<String, Double> getExpectedImprovement();

        /** Get confidence in suggestion */
        double getConfidence();

        /** Get suggestion reasoning */
        String getReasoning();

        /** Get implementation priority */
        int getPriority();

        /** Get potential risks */
        List<String> getRisks();
    }

    /**
     * Parameter sensitivity analysis result.
     */
    interface SensitivityAnalysis {
        /** Get parameter name analyzed */
        String getParameterName();

        /** Get sensitivity coefficient */
        double getSensitivityCoefficient();

        /** Get sensitivity curve data points */
        List<SensitivityPoint> getSensitivityCurve();

        /** Get optimal parameter value */
        Optional<Object> getOptimalValue();

        /** Get parameter importance ranking */
        double getImportanceRanking();

        /** Get analysis summary */
        String getSummary();

        /** Sensitivity data point */
        interface SensitivityPoint {
            /** Get parameter value */
            Object getParameterValue();

            /** Get corresponding metric value */
            double getMetricValue();

            /** Get confidence in measurement */
            double getConfidence();
        }
    }

    /**
     * Parameter change schedule.
     */
    interface ParameterSchedule {
        /** Get schedule type */
        ScheduleType getType();

        /** Get scheduled parameter changes */
        List<ScheduledChange> getChanges();

        /** Get schedule start time */
        Instant getStartTime();

        /** Get schedule end time */
        Optional<Instant> getEndTime();

        /** Get repeat configuration */
        Optional<RepeatConfiguration> getRepeatConfig();

        /** Schedule types */
        enum ScheduleType {
            LINEAR, EXPONENTIAL, LOGARITHMIC, STEP, CUSTOM
        }

        /** Scheduled parameter change */
        interface ScheduledChange {
            /** Get target parameter value */
            Object getTargetValue();

            /** Get change time */
            Instant getChangeTime();

            /** Get transition duration */
            Duration getTransitionDuration();

            /** Get change condition */
            Optional<Function<Map<String, Double>, Boolean>> getCondition();
        }

        /** Schedule repeat configuration */
        interface RepeatConfiguration {
            /** Get repeat interval */
            Duration getInterval();

            /** Get maximum repetitions */
            Optional<Integer> getMaxRepetitions();

            /** Get repeat until condition */
            Optional<Function<Map<String, Double>, Boolean>> getUntilCondition();
        }
    }

    /**
     * Controller performance statistics.
     */
    interface ControllerStatistics {
        /** Get total adaptations performed */
        long getTotalAdaptations();

        /** Get successful adaptations */
        long getSuccessfulAdaptations();

        /** Get average adaptation time */
        Duration getAverageAdaptationTime();

        /** Get parameter statistics by name */
        Map<String, ParameterStatistics> getParameterStatistics();

        /** Get adaptation frequency */
        double getAdaptationFrequency();

        /** Get performance improvement achieved */
        Map<String, Double> getPerformanceImprovement();

        /** Get controller efficiency metrics */
        Map<String, Double> getEfficiencyMetrics();

        /** Statistics for individual parameters */
        interface ParameterStatistics {
            /** Get parameter name */
            String getParameterName();

            /** Get number of changes */
            long getChangeCount();

            /** Get average change magnitude */
            double getAverageChangeMagnitude();

            /** Get parameter stability (lower is more stable) */
            double getStability();

            /** Get correlation with performance metrics */
            Map<String, Double> getPerformanceCorrelations();

            /** Get last change time */
            Optional<Instant> getLastChangeTime();
        }
    }
}