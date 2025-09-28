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

import java.util.Map;
import java.util.Optional;

/**
 * Generic state representation for hybrid ART neural networks.
 *
 * This interface represents any state within an ART system that can maintain
 * temporal context, support various learning modes, and integrate with both
 * traditional ART mechanisms and modern machine learning approaches.
 *
 * States in hybrid ART systems are characterized by:
 * - Temporal context and sequential dependencies
 * - Multiple representation formats (vector, symbolic, hybrid)
 * - Learning mode awareness (supervised, unsupervised, reinforcement)
 * - Integration capabilities with external ML models
 *
 * @param <T> the type of data contained in this state
 *
 * @author Hal Hildebrand
 * @since 1.0
 */
public interface State<T> extends Comparable<State<T>> {

    /**
     * Get the primary data content of this state.
     *
     * @return the state's data content, never null
     */
    T getData();

    /**
     * Get the temporal timestamp of this state.
     * Used for sequential learning and temporal ordering.
     *
     * @return timestamp in milliseconds since epoch
     */
    long getTimestamp();

    /**
     * Get the unique identifier for this state.
     *
     * @return unique state identifier, never null
     */
    String getId();

    /**
     * Get the confidence or certainty associated with this state.
     * Used in probabilistic reasoning and uncertainty quantification.
     *
     * @return confidence value in range [0.0, 1.0]
     */
    default double getConfidence() {
        return 1.0;
    }

    /**
     * Get the activation level of this state.
     * Represents how "active" or relevant this state currently is.
     *
     * @return activation level in range [0.0, 1.0]
     */
    default double getActivation() {
        return 1.0;
    }

    /**
     * Calculate similarity to another state.
     * Implementation should be symmetric: s1.similarity(s2) == s2.similarity(s1)
     *
     * @param other the state to compare with
     * @return similarity value in range [0.0, 1.0] where 1.0 is identical
     * @throws IllegalArgumentException if other is null or incompatible type
     */
    double similarity(State<T> other);

    /**
     * Calculate distance to another state.
     * Implementation should satisfy metric properties and be consistent with similarity.
     * Typically: distance = 1.0 - similarity
     *
     * @param other the state to measure distance to
     * @return distance value in range [0.0, 1.0] where 0.0 is identical
     * @throws IllegalArgumentException if other is null or incompatible type
     */
    default double distance(State<T> other) {
        return 1.0 - similarity(other);
    }

    /**
     * Check if this state can transition to another state.
     * Used for validating state transitions in sequential learning.
     *
     * @param target the target state
     * @return true if transition is valid and allowed
     */
    default boolean canTransitionTo(State<T> target) {
        return target != null && target.getData() != null;
    }

    /**
     * Get metadata associated with this state.
     * Allows storing arbitrary key-value pairs for extensibility.
     *
     * @return unmodifiable map of metadata, never null but may be empty
     */
    default Map<String, Object> getMetadata() {
        return Map.of();
    }

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
     * Check if this state is in a valid/consistent state.
     *
     * @return true if state is valid and usable
     */
    default boolean isValid() {
        return getData() != null && getId() != null;
    }

    /**
     * Create a deep copy of this state.
     *
     * @return a new state instance with same content
     */
    State<T> copy();

    /**
     * Compare states for temporal ordering.
     * Default implementation compares by timestamp.
     *
     * @param other the state to compare with
     * @return negative if this state is earlier, positive if later, 0 if same time
     */
    @Override
    default int compareTo(State<T> other) {
        return Long.compare(this.getTimestamp(), other.getTimestamp());
    }

    /**
     * Get a vector representation of this state for mathematical operations.
     * Used for integration with vectorized ART algorithms.
     *
     * @return vector representation, never null
     */
    default double[] toVector() {
        // Default implementation for non-vectorizable states
        return new double[]{getActivation(), getConfidence()};
    }

    /**
     * Get the dimensionality of this state's vector representation.
     *
     * @return number of dimensions in vector representation
     */
    default int getDimensions() {
        return toVector().length;
    }

    /**
     * Create a normalized version of this state.
     * Useful for maintaining consistent activation levels.
     *
     * @return normalized state with values in appropriate ranges
     */
    default State<T> normalize() {
        return this; // Default: no normalization needed
    }

    /**
     * Check if this state represents a terminal state.
     * Used in reinforcement learning scenarios.
     *
     * @return true if this is a terminal state
     */
    default boolean isTerminal() {
        return false;
    }

    /**
     * Get the hash code based on state content and timestamp.
     * Should be consistent with equals() implementation.
     *
     * @return hash code value
     */
    @Override
    int hashCode();

    /**
     * Check equality with another state.
     * Should consider both content and temporal aspects.
     *
     * @param obj the object to compare with
     * @return true if states are equivalent
     */
    @Override
    boolean equals(Object obj);

    /**
     * Get a human-readable string representation.
     *
     * @return string representation including key state information
     */
    @Override
    String toString();
}