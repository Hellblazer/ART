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
import java.util.Collection;
import java.util.List;
import java.util.Optional;
import java.util.function.Predicate;
import java.util.stream.Stream;

/**
 * Memory management system for state transitions in hybrid ART neural networks.
 *
 * TransitionMemory maintains a repository of observed transitions, supporting
 * both short-term working memory and long-term consolidated memory. It enables
 * experience replay, temporal pattern recognition, and sequential learning
 * by efficiently storing, retrieving, and analyzing transition histories.
 *
 * Key capabilities:
 * - Efficient storage and retrieval of transitions
 * - Memory consolidation from STM to LTM
 * - Experience replay for continual learning
 * - Temporal pattern analysis and prediction
 * - Memory capacity management and garbage collection
 * - Query and search functionality
 *
 * @param <S> the type of states stored in transitions
 *
 * @author Hal Hildebrand
 * @since 1.0
 */
public interface TransitionMemory<S extends State<?>> extends AutoCloseable {

    /**
     * Memory types supported by the transition memory system.
     */
    enum MemoryType {
        /** Short-term working memory with fast access and limited capacity */
        SHORT_TERM,
        /** Long-term memory with large capacity and persistent storage */
        LONG_TERM,
        /** Both short-term and long-term memory */
        BOTH
    }

    /**
     * Retrieval strategies for memory queries.
     */
    enum RetrievalStrategy {
        /** Most recent transitions first */
        TEMPORAL,
        /** Highest probability/confidence transitions first */
        CONFIDENCE,
        /** Most similar transitions first */
        SIMILARITY,
        /** Random sampling */
        RANDOM,
        /** Most frequently accessed transitions */
        FREQUENCY
    }

    /**
     * Store a new transition in memory.
     *
     * @param transition the transition to store
     * @param memoryType the type of memory to store in
     * @throws IllegalArgumentException if transition is null or invalid
     */
    void store(Transition<S, ?> transition, MemoryType memoryType);

    /**
     * Store a new transition using default memory allocation strategy.
     * Recent transitions typically go to STM, important ones may go directly to LTM.
     *
     * @param transition the transition to store
     * @throws IllegalArgumentException if transition is null or invalid
     */
    default void store(Transition<S, ?> transition) {
        store(transition, MemoryType.SHORT_TERM);
    }

    /**
     * Store multiple transitions efficiently.
     *
     * @param transitions the transitions to store
     * @param memoryType the type of memory to store in
     */
    default void storeAll(Collection<Transition<S, ?>> transitions, MemoryType memoryType) {
        transitions.forEach(t -> store(t, memoryType));
    }

    /**
     * Retrieve a specific transition by ID.
     *
     * @param transitionId the transition identifier
     * @return optional transition if found
     */
    Optional<Transition<S, ?>> retrieve(String transitionId);

    /**
     * Retrieve transitions involving a specific state.
     *
     * @param state the state to search for
     * @param memoryType which memory types to search in
     * @return list of transitions involving the state (as source or target)
     */
    List<Transition<S, ?>> retrieveByState(S state, MemoryType memoryType);

    /**
     * Retrieve transitions between two specific states.
     *
     * @param sourceState the source state
     * @param targetState the target state
     * @param memoryType which memory types to search in
     * @return list of transitions from source to target
     */
    List<Transition<S, ?>> retrieveByStates(S sourceState, S targetState, MemoryType memoryType);

    /**
     * Retrieve transitions within a time range.
     *
     * @param startTime the earliest transition start time (inclusive)
     * @param endTime the latest transition end time (exclusive)
     * @param memoryType which memory types to search in
     * @return list of transitions within the time range
     */
    List<Transition<S, ?>> retrieveByTimeRange(Instant startTime, Instant endTime, MemoryType memoryType);

    /**
     * Retrieve the most recent transitions.
     *
     * @param count maximum number of transitions to retrieve
     * @param memoryType which memory types to search in
     * @return list of recent transitions, most recent first
     */
    List<Transition<S, ?>> retrieveRecent(int count, MemoryType memoryType);

    /**
     * Query transitions using a custom predicate.
     *
     * @param predicate the filter condition
     * @param memoryType which memory types to search in
     * @param strategy the retrieval strategy for ordering results
     * @param maxResults maximum number of results to return
     * @return list of matching transitions
     */
    List<Transition<S, ?>> query(Predicate<Transition<S, ?>> predicate,
                                  MemoryType memoryType,
                                  RetrievalStrategy strategy,
                                  int maxResults);

    /**
     * Get a stream of all transitions for custom processing.
     *
     * @param memoryType which memory types to include
     * @return stream of transitions
     */
    Stream<Transition<S, ?>> stream(MemoryType memoryType);

    /**
     * Sample transitions for experience replay.
     * This method intelligently selects transitions to balance recent experiences
     * with important historical patterns.
     *
     * @param sampleSize number of transitions to sample
     * @param strategy the sampling strategy to use
     * @return sampled transitions for replay
     */
    List<Transition<S, ?>> sample(int sampleSize, RetrievalStrategy strategy);

    /**
     * Consolidate transitions from short-term to long-term memory.
     * This process typically involves importance-based selection and
     * compression of redundant transitions.
     *
     * @param consolidationThreshold transitions older than this are candidates for consolidation
     * @return number of transitions consolidated
     */
    int consolidate(Duration consolidationThreshold);

    /**
     * Remove old or unimportant transitions to free memory.
     * This is typically called when memory usage exceeds capacity limits.
     *
     * @param memoryType which memory types to clean
     * @param retentionCriteria predicate determining which transitions to keep
     * @return number of transitions removed
     */
    int cleanup(MemoryType memoryType, Predicate<Transition<S, ?>> retentionCriteria);

    /**
     * Get the current number of stored transitions.
     *
     * @param memoryType which memory types to count
     * @return total number of transitions
     */
    long size(MemoryType memoryType);

    /**
     * Get the total number of transitions stored.
     *
     * @return total transition count across all memory types
     */
    default long totalSize() {
        return size(MemoryType.BOTH);
    }

    /**
     * Check if the memory is empty.
     *
     * @param memoryType which memory types to check
     * @return true if no transitions are stored
     */
    default boolean isEmpty(MemoryType memoryType) {
        return size(memoryType) == 0;
    }

    /**
     * Get the memory capacity limits.
     *
     * @param memoryType which memory type
     * @return optional capacity limit, empty if unlimited
     */
    Optional<Long> getCapacity(MemoryType memoryType);

    /**
     * Get the current memory usage as a fraction of capacity.
     *
     * @param memoryType which memory type
     * @return usage fraction [0.0, 1.0], or -1.0 if unlimited capacity
     */
    default double getUsage(MemoryType memoryType) {
        var capacity = getCapacity(memoryType);
        if (capacity.isEmpty()) {
            return -1.0; // Unlimited
        }
        return (double) size(memoryType) / capacity.get();
    }

    /**
     * Clear all transitions from memory.
     *
     * @param memoryType which memory types to clear
     */
    void clear(MemoryType memoryType);

    /**
     * Clear all transitions from all memory types.
     */
    default void clearAll() {
        clear(MemoryType.BOTH);
    }

    /**
     * Get memory statistics for monitoring and debugging.
     *
     * @return memory statistics including size, capacity, hit rates, etc.
     */
    MemoryStatistics getStatistics();

    /**
     * Validate the integrity of stored transitions.
     * Checks for consistency, duplicate IDs, temporal ordering, etc.
     *
     * @return list of validation issues found (empty if all valid)
     */
    default List<String> validate() {
        return List.of(); // Default: assume valid
    }

    /**
     * Export transitions to an external format for persistence or analysis.
     *
     * @param memoryType which memory types to export
     * @param format the export format (implementation-specific)
     * @return exported data in the specified format
     */
    default Optional<Object> export(MemoryType memoryType, String format) {
        return Optional.empty(); // Default: not supported
    }

    /**
     * Import transitions from an external source.
     *
     * @param data the transition data to import
     * @param format the data format (implementation-specific)
     * @param memoryType where to store the imported transitions
     * @return number of transitions successfully imported
     */
    default int importTransitions(Object data, String format, MemoryType memoryType) {
        return 0; // Default: not supported
    }

    /**
     * Memory statistics for monitoring transition memory performance.
     */
    interface MemoryStatistics {
        /** Get the timestamp when statistics were collected */
        Instant getTimestamp();

        /** Get the number of transitions in each memory type */
        long getShortTermSize();
        long getLongTermSize();

        /** Get capacity information */
        Optional<Long> getShortTermCapacity();
        Optional<Long> getLongTermCapacity();

        /** Get access statistics */
        long getTotalRetrievals();
        long getCacheHits();
        long getCacheMisses();

        /** Get consolidation statistics */
        long getTotalConsolidations();
        Instant getLastConsolidation();

        /** Get cleanup statistics */
        long getTotalCleanups();
        long getTotalTransitionsRemoved();

        /** Calculate hit rate */
        default double getHitRate() {
            long total = getCacheHits() + getCacheMisses();
            return total > 0 ? (double) getCacheHits() / total : 0.0;
        }

        /** Get memory efficiency (useful transitions / total capacity) */
        default double getEfficiency() {
            long totalStored = getShortTermSize() + getLongTermSize();
            long totalCapacity = getShortTermCapacity().orElse(Long.MAX_VALUE) +
                               getLongTermCapacity().orElse(Long.MAX_VALUE);
            return totalCapacity > 0 ? (double) totalStored / totalCapacity : 0.0;
        }
    }

    /**
     * Release memory resources and perform cleanup.
     * Called automatically when the memory system is no longer needed.
     */
    @Override
    void close();
}