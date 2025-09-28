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

import com.hellblazer.art.core.Context;
import com.hellblazer.art.core.State;

import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Interface for defining hybrid learning strategies in ART neural networks.
 *
 * HybridStrategy coordinates the integration of traditional ART algorithms
 * with modern machine learning approaches, defining how different components
 * interact, when to switch between methods, and how to combine their outputs
 * for optimal performance.
 *
 * Key strategy capabilities:
 * - Component coordination and orchestration
 * - Dynamic method selection based on context
 * - Performance-based strategy adaptation
 * - Resource allocation between components
 * - Integration pattern management
 * - Strategy optimization and tuning
 *
 * @param <S> the type of states handled by this strategy
 *
 * @author Hal Hildebrand
 * @since 1.0
 */
public interface HybridStrategy<S extends State<?>> {

    /**
     * Types of hybrid integration patterns.
     */
    enum IntegrationPattern {
        /** Sequential processing through components */
        SEQUENTIAL,
        /** Parallel processing with result combination */
        PARALLEL,
        /** Hierarchical processing with delegation */
        HIERARCHICAL,
        /** Ensemble approach with voting/averaging */
        ENSEMBLE,
        /** Pipeline with conditional branching */
        PIPELINE,
        /** Adaptive switching between methods */
        ADAPTIVE_SWITCHING,
        /** Cooperative multi-agent approach */
        COOPERATIVE
    }

    /**
     * Strategy adaptation modes.
     */
    enum AdaptationMode {
        /** Static strategy, no adaptation */
        STATIC,
        /** Adapt based on performance metrics */
        PERFORMANCE_BASED,
        /** Adapt based on data characteristics */
        DATA_DRIVEN,
        /** Adapt based on resource constraints */
        RESOURCE_AWARE,
        /** Adapt based on user feedback */
        FEEDBACK_DRIVEN,
        /** Continuous reinforcement learning adaptation */
        REINFORCEMENT_LEARNING,
        /** Multi-objective optimization adaptation */
        MULTI_OBJECTIVE
    }

    /**
     * Component types in hybrid systems.
     */
    enum ComponentType {
        /** Traditional ART neural network */
        ART_NETWORK,
        /** Deep neural network */
        DEEP_NETWORK,
        /** Preprocessing component */
        PREPROCESSOR,
        /** Feature extractor */
        FEATURE_EXTRACTOR,
        /** Classifier/predictor */
        CLASSIFIER,
        /** Ensemble method */
        ENSEMBLE,
        /** Memory system */
        MEMORY_SYSTEM,
        /** Optimizer */
        OPTIMIZER,
        /** Validator */
        VALIDATOR
    }

    /**
     * Get the integration pattern used by this strategy.
     *
     * @return the integration pattern
     */
    IntegrationPattern getIntegrationPattern();

    /**
     * Get the adaptation mode for this strategy.
     *
     * @return the adaptation mode
     */
    AdaptationMode getAdaptationMode();

    /**
     * Get all components managed by this strategy.
     *
     * @return map of component IDs to their information
     */
    Map<String, ComponentInfo> getComponents();

    /**
     * Add a component to the hybrid strategy.
     *
     * @param componentId unique identifier for the component
     * @param component the component implementation
     * @param componentType type of the component
     * @param weight initial weight/priority for the component
     * @param configuration component-specific configuration
     */
    void addComponent(String componentId, Object component, ComponentType componentType,
                     double weight, Map<String, Object> configuration);

    /**
     * Remove a component from the strategy.
     *
     * @param componentId the component identifier
     * @return true if component was found and removed
     */
    boolean removeComponent(String componentId);

    /**
     * Update component weight/priority.
     *
     * @param componentId the component identifier
     * @param newWeight the new weight value
     * @return true if component was found and updated
     */
    boolean updateComponentWeight(String componentId, double newWeight);

    /**
     * Execute the hybrid strategy for a given state and context.
     *
     * @param state the input state to process
     * @param context the execution context
     * @return strategy execution result
     */
    StrategyResult<S> execute(S state, Context context);

    /**
     * Execute the strategy for multiple states (batch processing).
     *
     * @param states the input states to process
     * @param context the execution context
     * @return list of strategy results in the same order
     */
    default List<StrategyResult<S>> executeBatch(List<S> states, Context context) {
        return states.stream()
                    .map(state -> execute(state, context))
                    .toList();
    }

    /**
     * Update the strategy based on performance feedback.
     *
     * @param performanceMetrics recent performance measurements
     * @param adaptationContext context for adaptation decisions
     */
    void adapt(Map<String, Double> performanceMetrics, AdaptationContext adaptationContext);

    /**
     * Get the current strategy configuration.
     *
     * @return strategy configuration
     */
    StrategyConfiguration getConfiguration();

    /**
     * Update the strategy configuration.
     *
     * @param configuration new configuration to apply
     */
    void updateConfiguration(StrategyConfiguration configuration);

    /**
     * Validate the strategy configuration and components.
     *
     * @return list of validation issues (empty if valid)
     */
    List<String> validate();

    /**
     * Get performance statistics for the strategy and its components.
     *
     * @return performance statistics
     */
    StrategyPerformance getPerformanceStatistics();

    /**
     * Reset the strategy to its initial state.
     */
    void reset();

    /**
     * Get optimization recommendations for improving strategy performance.
     *
     * @return list of optimization recommendations
     */
    List<OptimizationRecommendation> getOptimizationRecommendations();

    /**
     * Enable or disable specific components temporarily.
     *
     * @param componentId the component identifier
     * @param enabled whether to enable the component
     */
    void setComponentEnabled(String componentId, boolean enabled);

    /**
     * Check if a component is currently enabled.
     *
     * @param componentId the component identifier
     * @return true if component is enabled
     */
    boolean isComponentEnabled(String componentId);

    /**
     * Get memory usage of the strategy and its components.
     *
     * @return memory usage in bytes
     */
    long getMemoryUsage();

    /**
     * Information about a component in the hybrid strategy.
     */
    interface ComponentInfo {
        /** Get component identifier */
        String getId();

        /** Get component type */
        ComponentType getType();

        /** Get component weight/priority */
        double getWeight();

        /** Get component configuration */
        Map<String, Object> getConfiguration();

        /** Check if component is enabled */
        boolean isEnabled();

        /** Get component implementation */
        Object getComponent();

        /** Get last execution time */
        Optional<java.time.Instant> getLastExecutionTime();

        /** Get execution count */
        long getExecutionCount();

        /** Get average execution time */
        java.time.Duration getAverageExecutionTime();

        /** Get success rate */
        double getSuccessRate();
    }

    /**
     * Result of strategy execution.
     */
    interface StrategyResult<S extends State<?>> {
        /** Get the processed state */
        S getProcessedState();

        /** Get execution success status */
        boolean isSuccessful();

        /** Get error message if execution failed */
        Optional<String> getErrorMessage();

        /** Get results from individual components */
        Map<String, Object> getComponentResults();

        /** Get execution time */
        java.time.Duration getExecutionTime();

        /** Get confidence in the result */
        double getConfidence();

        /** Get execution trace for debugging */
        List<ExecutionStep> getExecutionTrace();

        /** Get resource usage during execution */
        ResourceUsage getResourceUsage();

        /** Individual execution step in the strategy */
        interface ExecutionStep {
            /** Get component that executed this step */
            String getComponentId();

            /** Get step start time */
            java.time.Instant getStartTime();

            /** Get step duration */
            java.time.Duration getDuration();

            /** Get step result */
            Object getResult();

            /** Check if step was successful */
            boolean isSuccessful();

            /** Get step-specific metrics */
            Map<String, Object> getMetrics();
        }

        /** Resource usage during execution */
        interface ResourceUsage {
            /** Get CPU time used */
            java.time.Duration getCpuTime();

            /** Get memory allocated */
            long getMemoryAllocated();

            /** Get I/O operations performed */
            long getIoOperations();

            /** Get network requests made */
            long getNetworkRequests();
        }
    }

    /**
     * Configuration for the hybrid strategy.
     */
    interface StrategyConfiguration {
        /** Get integration pattern */
        IntegrationPattern getIntegrationPattern();

        /** Get adaptation mode */
        AdaptationMode getAdaptationMode();

        /** Get execution timeout */
        java.time.Duration getExecutionTimeout();

        /** Get maximum parallel components */
        int getMaxParallelComponents();

        /** Get component selection criteria */
        ComponentSelectionCriteria getSelectionCriteria();

        /** Get adaptation settings */
        AdaptationSettings getAdaptationSettings();

        /** Get resource limits */
        ResourceLimits getResourceLimits();

        /** Component selection criteria */
        interface ComponentSelectionCriteria {
            /** Minimum confidence threshold for component results */
            double getMinConfidence();

            /** Maximum execution time for components */
            java.time.Duration getMaxExecutionTime();

            /** Required component types for execution */
            List<ComponentType> getRequiredComponents();

            /** Optional component types */
            List<ComponentType> getOptionalComponents();
        }

        /** Adaptation behavior settings */
        interface AdaptationSettings {
            /** Frequency of adaptation */
            java.time.Duration getAdaptationInterval();

            /** Performance window for adaptation decisions */
            java.time.Duration getPerformanceWindow();

            /** Minimum performance change to trigger adaptation */
            double getMinPerformanceChange();

            /** Learning rate for weight updates */
            double getLearningRate();

            /** Exploration vs exploitation balance */
            double getExplorationRate();
        }

        /** Resource usage limits */
        interface ResourceLimits {
            /** Maximum memory usage */
            long getMaxMemoryUsage();

            /** Maximum CPU time per execution */
            java.time.Duration getMaxCpuTime();

            /** Maximum concurrent executions */
            int getMaxConcurrentExecutions();

            /** Maximum component chain length */
            int getMaxChainLength();
        }
    }

    /**
     * Context for strategy adaptation decisions.
     */
    interface AdaptationContext {
        /** Get current performance metrics */
        Map<String, Double> getCurrentMetrics();

        /** Get historical performance trends */
        Map<String, List<Double>> getPerformanceTrends();

        /** Get available resources */
        Map<String, Double> getAvailableResources();

        /** Get user preferences or constraints */
        Map<String, Object> getUserPreferences();

        /** Get environment characteristics */
        Map<String, Object> getEnvironmentInfo();

        /** Get adaptation goals */
        List<AdaptationGoal> getGoals();

        /** Adaptation goal specification */
        interface AdaptationGoal {
            /** Get goal type */
            GoalType getType();

            /** Get target value */
            double getTargetValue();

            /** Get goal weight/importance */
            double getWeight();

            /** Get tolerance for goal achievement */
            double getTolerance();

            /** Goal types */
            enum GoalType {
                MAXIMIZE_ACCURACY, MINIMIZE_LATENCY, MINIMIZE_MEMORY,
                MAXIMIZE_THROUGHPUT, MINIMIZE_ERRORS, BALANCE_TRADE_OFFS
            }
        }
    }

    /**
     * Performance statistics for the strategy.
     */
    interface StrategyPerformance {
        /** Get total executions */
        long getTotalExecutions();

        /** Get success rate */
        double getSuccessRate();

        /** Get average execution time */
        java.time.Duration getAverageExecutionTime();

        /** Get average confidence */
        double getAverageConfidence();

        /** Get component performance breakdown */
        Map<String, ComponentPerformance> getComponentPerformance();

        /** Get adaptation statistics */
        AdaptationStatistics getAdaptationStatistics();

        /** Get resource utilization */
        Map<String, Double> getResourceUtilization();

        /** Get performance trends */
        Map<String, PerformanceTrend> getTrends();

        /** Performance statistics for individual components */
        interface ComponentPerformance {
            /** Get component execution count */
            long getExecutionCount();

            /** Get component success rate */
            double getSuccessRate();

            /** Get average execution time */
            java.time.Duration getAverageExecutionTime();

            /** Get current weight */
            double getCurrentWeight();

            /** Get weight change over time */
            List<WeightChange> getWeightHistory();

            /** Weight change record */
            record WeightChange(java.time.Instant timestamp, double oldWeight, double newWeight, String reason) {}
        }

        /** Strategy adaptation statistics */
        interface AdaptationStatistics {
            /** Get total adaptations performed */
            long getTotalAdaptations();

            /** Get last adaptation time */
            Optional<java.time.Instant> getLastAdaptationTime();

            /** Get adaptation frequency */
            double getAdaptationFrequency();

            /** Get adaptation effectiveness */
            double getAdaptationEffectiveness();

            /** Get adaptation triggers */
            Map<String, Long> getAdaptationTriggers();
        }

        /** Performance trend analysis */
        interface PerformanceTrend {
            /** Get trend direction */
            TrendDirection getDirection();

            /** Get trend strength */
            double getStrength();

            /** Get slope of trend line */
            double getSlope();

            /** Get correlation coefficient */
            double getCorrelation();

            /** Trend directions */
            enum TrendDirection {
                IMPROVING, DEGRADING, STABLE, OSCILLATING, UNKNOWN
            }
        }
    }

    /**
     * Optimization recommendation for strategy improvement.
     */
    interface OptimizationRecommendation {
        /** Get recommendation type */
        RecommendationType getType();

        /** Get recommendation description */
        String getDescription();

        /** Get expected impact */
        double getExpectedImpact();

        /** Get implementation difficulty */
        DifficultyLevel getDifficulty();

        /** Get recommendation parameters */
        Map<String, Object> getParameters();

        /** Get recommendation priority */
        int getPriority();

        /** Recommendation types */
        enum RecommendationType {
            ADJUST_WEIGHTS, ADD_COMPONENT, REMOVE_COMPONENT,
            CHANGE_PATTERN, TUNE_PARAMETERS, INCREASE_RESOURCES,
            OPTIMIZE_PIPELINE, ENABLE_PARALLELISM
        }

        /** Implementation difficulty levels */
        enum DifficultyLevel {
            LOW, MEDIUM, HIGH, VERY_HIGH
        }
    }
}