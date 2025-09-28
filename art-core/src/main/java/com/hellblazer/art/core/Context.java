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
package com.hellblazer.art.core;

import java.time.Instant;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * Execution context for hybrid ART neural network operations.
 *
 * The Context interface provides environmental information and execution state
 * that influences how ART algorithms process patterns, make decisions, and
 * adapt their behavior. This includes learning mode, performance constraints,
 * temporal context, and integration settings with external systems.
 *
 * Contexts in hybrid ART systems support:
 * - Multi-modal learning environments
 * - Performance and resource constraints
 * - Temporal and sequential dependencies
 * - Integration with external ML pipelines
 * - Dynamic parameter adjustment
 *
 * @author Hal Hildebrand
 * @since 1.0
 */
public interface Context {

    /**
     * Learning modes supported by hybrid ART systems.
     */
    enum LearningMode {
        /** Traditional unsupervised ART learning */
        UNSUPERVISED,
        /** Supervised learning with labeled data */
        SUPERVISED,
        /** Reinforcement learning with rewards */
        REINFORCEMENT,
        /** Hybrid mode combining multiple approaches */
        HYBRID,
        /** Semi-supervised learning with partial labels */
        SEMI_SUPERVISED,
        /** Transfer learning from pre-trained models */
        TRANSFER,
        /** Continual learning with experience replay */
        CONTINUAL
    }

    /**
     * Execution priorities for balancing performance vs. accuracy.
     */
    enum ExecutionPriority {
        /** Optimize for accuracy regardless of performance */
        ACCURACY,
        /** Balance accuracy and performance */
        BALANCED,
        /** Optimize for speed, accept reduced accuracy */
        SPEED,
        /** Minimize resource usage */
        RESOURCE_EFFICIENT
    }

    /**
     * Get the current learning mode.
     *
     * @return the active learning mode, never null
     */
    LearningMode getLearningMode();

    /**
     * Get the execution priority for this context.
     *
     * @return the execution priority, never null
     */
    ExecutionPriority getExecutionPriority();

    /**
     * Get the context creation timestamp.
     *
     * @return creation time, never null
     */
    Instant getCreationTime();

    /**
     * Get the maximum execution time allowed for operations in this context.
     *
     * @return optional timeout in milliseconds
     */
    Optional<Long> getTimeoutMillis();

    /**
     * Get the maximum memory usage allowed for operations in this context.
     *
     * @return optional memory limit in bytes
     */
    Optional<Long> getMemoryLimitBytes();

    /**
     * Check if SIMD vectorization is enabled in this context.
     *
     * @return true if SIMD operations should be used
     */
    default boolean isSIMDEnabled() {
        return true;
    }

    /**
     * Check if parallel processing is enabled in this context.
     *
     * @return true if parallel operations should be used
     */
    default boolean isParallelEnabled() {
        return true;
    }

    /**
     * Get the number of threads to use for parallel operations.
     *
     * @return thread count, or empty if system default should be used
     */
    default Optional<Integer> getThreadCount() {
        return Optional.empty();
    }

    /**
     * Get context-specific parameters.
     * These can override algorithm defaults for this execution context.
     *
     * @return unmodifiable map of parameters, never null but may be empty
     */
    Map<String, Object> getParameters();

    /**
     * Get a specific parameter value by key.
     *
     * @param key the parameter key
     * @param type the expected parameter type
     * @param <T> the parameter type
     * @return optional parameter value of the specified type
     */
    default <T> Optional<T> getParameter(String key, Class<T> type) {
        var value = getParameters().get(key);
        if (value != null && type.isInstance(value)) {
            return Optional.of(type.cast(value));
        }
        return Optional.empty();
    }

    /**
     * Get context metadata.
     * Used for storing arbitrary information about the execution environment.
     *
     * @return unmodifiable map of metadata, never null but may be empty
     */
    Map<String, Object> getMetadata();

    /**
     * Get a specific metadata value by key.
     *
     * @param key the metadata key
     * @return optional metadata value
     */
    default Optional<Object> getMetadata(String key) {
        return Optional.ofNullable(getMetadata().get(key));
    }

    /**
     * Get the set of enabled features for this context.
     * Features control optional functionality like debugging, profiling, etc.
     *
     * @return unmodifiable set of enabled features, never null but may be empty
     */
    default Set<String> getEnabledFeatures() {
        return Set.of();
    }

    /**
     * Check if a specific feature is enabled in this context.
     *
     * @param feature the feature name
     * @return true if the feature is enabled
     */
    default boolean isFeatureEnabled(String feature) {
        return getEnabledFeatures().contains(feature);
    }

    /**
     * Get the random seed for deterministic operations.
     * Used to ensure reproducible results when needed.
     *
     * @return optional random seed
     */
    default Optional<Long> getRandomSeed() {
        return Optional.empty();
    }

    /**
     * Check if this context requires deterministic execution.
     * When true, algorithms should use the random seed and avoid non-deterministic operations.
     *
     * @return true if deterministic execution is required
     */
    default boolean isDeterministic() {
        return getRandomSeed().isPresent();
    }

    /**
     * Get the debug level for this context.
     * Higher values indicate more verbose debugging output.
     *
     * @return debug level (0 = no debug, higher = more verbose)
     */
    default int getDebugLevel() {
        return 0;
    }

    /**
     * Check if debugging is enabled in this context.
     *
     * @return true if debug level > 0
     */
    default boolean isDebugEnabled() {
        return getDebugLevel() > 0;
    }

    /**
     * Get the session or experiment identifier for this context.
     * Used for tracking and correlating operations across a session.
     *
     * @return optional session identifier
     */
    default Optional<String> getSessionId() {
        return Optional.empty();
    }

    /**
     * Create a derived context with modified parameters.
     * The new context inherits all settings from this context except for the specified changes.
     *
     * @param modifications map of parameters to modify
     * @return new context with applied modifications
     */
    default Context withParameters(Map<String, Object> modifications) {
        // Default implementation: create a simple derived context
        var originalParams = getParameters();
        var newParams = new java.util.HashMap<String, Object>();
        newParams.putAll(originalParams);
        newParams.putAll(modifications);

        return new Context() {
            @Override
            public LearningMode getLearningMode() {
                return Context.this.getLearningMode();
            }

            @Override
            public ExecutionPriority getExecutionPriority() {
                return Context.this.getExecutionPriority();
            }

            @Override
            public Instant getCreationTime() {
                return Context.this.getCreationTime();
            }

            @Override
            public Optional<Long> getTimeoutMillis() {
                return Context.this.getTimeoutMillis();
            }

            @Override
            public Optional<Long> getMemoryLimitBytes() {
                return Context.this.getMemoryLimitBytes();
            }

            @Override
            public Map<String, Object> getParameters() {
                return newParams;
            }

            @Override
            public Map<String, Object> getMetadata() {
                return Context.this.getMetadata();
            }
        };
    }

    /**
     * Check if this context is still valid and usable.
     * Contexts may become invalid due to timeouts, resource exhaustion, or cancellation.
     *
     * @return true if context is valid and operations should proceed
     */
    default boolean isValid() {
        var timeout = getTimeoutMillis();
        if (timeout.isPresent()) {
            long elapsed = System.currentTimeMillis() - getCreationTime().toEpochMilli();
            return elapsed < timeout.get();
        }
        return true;
    }
}