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

import java.time.Duration;
import java.time.Instant;
import java.util.Map;
import java.util.Optional;

/**
 * Represents a state transition in hybrid ART neural networks.
 *
 * Transitions capture the movement from one state to another, including
 * the context in which the transition occurred, timing information,
 * and any associated rewards or costs. This is essential for temporal
 * learning, sequential pattern recognition, and reinforcement learning
 * in hybrid ART systems.
 *
 * Transitions support:
 * - Temporal sequence learning
 * - Reinforcement learning with rewards
 * - Causal relationship modeling
 * - Experience replay and continual learning
 * - Multi-modal state transitions
 *
 * @param <S> the type of states involved in this transition
 * @param <C> the type of context information
 *
 * @author Hal Hildebrand
 * @since 1.0
 */
public interface Transition<S extends State<?>, C extends Context> {

    /**
     * Types of transitions supported in hybrid ART systems.
     */
    enum TransitionType {
        /** Normal state-to-state transition */
        NORMAL,
        /** Transition that creates a new category/cluster */
        CREATION,
        /** Transition that merges existing categories */
        MERGE,
        /** Transition that splits a category */
        SPLIT,
        /** Transition representing a reward/punishment signal */
        REWARD,
        /** Transition representing an error correction */
        CORRECTION,
        /** Transition from external input/environment */
        EXTERNAL,
        /** Transition representing memory consolidation */
        CONSOLIDATION
    }

    /**
     * Get the source state of this transition.
     *
     * @return the starting state, never null
     */
    S getSourceState();

    /**
     * Get the target state of this transition.
     *
     * @return the ending state, never null
     */
    S getTargetState();

    /**
     * Get the context in which this transition occurred.
     *
     * @return the execution context, never null
     */
    C getContext();

    /**
     * Get the type of this transition.
     *
     * @return the transition type, never null
     */
    TransitionType getType();

    /**
     * Get the timestamp when this transition started.
     *
     * @return start time, never null
     */
    Instant getStartTime();

    /**
     * Get the timestamp when this transition completed.
     *
     * @return end time, never null
     */
    Instant getEndTime();

    /**
     * Get the duration of this transition.
     *
     * @return transition duration, never null
     */
    default Duration getDuration() {
        return Duration.between(getStartTime(), getEndTime());
    }

    /**
     * Get the probability or confidence of this transition.
     * Used in probabilistic and Markov models.
     *
     * @return probability value in range [0.0, 1.0]
     */
    default double getProbability() {
        return 1.0;
    }

    /**
     * Get the reward associated with this transition.
     * Used in reinforcement learning scenarios.
     *
     * @return optional reward value (positive for rewards, negative for penalties)
     */
    default Optional<Double> getReward() {
        return Optional.empty();
    }

    /**
     * Get the cost or energy required for this transition.
     * Used for optimizing transition paths and resource usage.
     *
     * @return transition cost (>= 0.0), default is 1.0
     */
    default double getCost() {
        return 1.0;
    }

    /**
     * Get additional properties or metadata for this transition.
     *
     * @return unmodifiable map of properties, never null but may be empty
     */
    default Map<String, Object> getProperties() {
        return Map.of();
    }

    /**
     * Get a specific property value by key.
     *
     * @param key the property key
     * @param type the expected property type
     * @param <T> the property type
     * @return optional property value of the specified type
     */
    default <T> Optional<T> getProperty(String key, Class<T> type) {
        var value = getProperties().get(key);
        if (value != null && type.isInstance(value)) {
            return Optional.of(type.cast(value));
        }
        return Optional.empty();
    }

    /**
     * Calculate the similarity between this transition and another.
     * Considers both state similarity and contextual similarity.
     *
     * @param other the transition to compare with
     * @return similarity value in range [0.0, 1.0] where 1.0 is identical
     * @throws IllegalArgumentException if other is null
     */
    default double similarity(Transition<S, C> other) {
        if (other == null) {
            throw new IllegalArgumentException("Cannot compare with null transition");
        }

        // Default implementation: average of source and target state similarities
        @SuppressWarnings("unchecked")
        double sourceSimilarity = ((State<Object>)getSourceState()).similarity((State<Object>)other.getSourceState());
        @SuppressWarnings("unchecked")
        double targetSimilarity = ((State<Object>)getTargetState()).similarity((State<Object>)other.getTargetState());
        return (sourceSimilarity + targetSimilarity) / 2.0;
    }

    /**
     * Check if this transition is valid and well-formed.
     *
     * @return true if transition is valid (states exist, times are consistent, etc.)
     */
    default boolean isValid() {
        return getSourceState() != null &&
               getTargetState() != null &&
               getContext() != null &&
               getType() != null &&
               getStartTime() != null &&
               getEndTime() != null &&
               !getEndTime().isBefore(getStartTime()) &&
               getProbability() >= 0.0 && getProbability() <= 1.0 &&
               getCost() >= 0.0;
    }

    /**
     * Check if this transition represents a successful outcome.
     * Used in reinforcement learning and performance evaluation.
     *
     * @return true if transition led to a positive outcome
     */
    default boolean isSuccessful() {
        return getReward().map(r -> r > 0.0).orElse(true);
    }

    /**
     * Check if this transition represents a terminal transition.
     * Terminal transitions end a sequence or episode.
     *
     * @return true if this is a terminal transition
     */
    default boolean isTerminal() {
        return getTargetState().isTerminal();
    }

    /**
     * Get the unique identifier for this transition.
     * Default implementation combines source and target state IDs with timestamp.
     *
     * @return unique transition identifier, never null
     */
    default String getId() {
        return String.format("%s->%s@%d",
                getSourceState().getId(),
                getTargetState().getId(),
                getStartTime().toEpochMilli());
    }

    /**
     * Create a reverse transition (target to source).
     * Useful for bidirectional learning and backtracking.
     *
     * @return reverse transition with swapped states and adjusted properties
     */
    default Transition<S, C> reverse() {
        var original = this;
        return new Transition<S, C>() {
            @Override
            public S getSourceState() {
                return original.getTargetState();
            }

            @Override
            public S getTargetState() {
                return original.getSourceState();
            }

            @Override
            public C getContext() {
                return original.getContext();
            }

            @Override
            public TransitionType getType() {
                return original.getType();
            }

            @Override
            public Instant getStartTime() {
                return original.getEndTime();
            }

            @Override
            public Instant getEndTime() {
                return original.getStartTime();
            }

            @Override
            public double getProbability() {
                return original.getProbability();
            }

            @Override
            public Optional<Double> getReward() {
                return original.getReward().map(r -> -r); // Reverse reward
            }

            @Override
            public double getCost() {
                return original.getCost();
            }

            @Override
            public Map<String, Object> getProperties() {
                return original.getProperties();
            }

            @Override
            public String getId() {
                return original.getId() + "_reverse";
            }
        };
    }

    /**
     * Get hash code based on source state, target state, and start time.
     *
     * @return hash code value
     */
    @Override
    int hashCode();

    /**
     * Check equality with another transition.
     * Should consider source, target, context, and timing.
     *
     * @param obj the object to compare with
     * @return true if transitions are equivalent
     */
    @Override
    boolean equals(Object obj);

    /**
     * Get a human-readable string representation.
     *
     * @return string representation including transition details
     */
    @Override
    String toString();
}