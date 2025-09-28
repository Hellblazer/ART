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
package com.hellblazer.art.temporal.memory;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.temporal.TemporalPattern;
import com.hellblazer.art.temporal.parameters.WorkingMemoryParameters;

/**
 * Item-and-Order Working Memory implementation following the STORE 2 model from:
 *
 * Kazerounian, S., & Grossberg, S. (2014). Real-time learning of predictive recognition
 * categories that chunk sequences of items stored in working memory.
 * Frontiers in Psychology, 5, 1053. https://doi.org/10.3389/fpsyg.2014.01053
 *
 * This working memory maintains both the content (items) and temporal order
 * of sequences using primacy gradients that decay over time. The model supports:
 * - Real-time sequence storage with temporal gradients
 * - Primacy-based item activation and decay
 * - Dynamic capacity management
 * - Integration with masking field networks
 *
 * Mathematical Foundation:
 * The working memory uses shunting dynamics with primacy gradients:
 * dx_i/dt = -α*x_i + (β - x_i)*I_i - γ*x_i*∑(x_j)
 *
 * Where:
 * - x_i is the activation of item i
 * - α is the passive decay rate
 * - β is the maximum activation level
 * - I_i is the input to item i
 * - γ controls competitive interactions
 *
 * @author Hal Hildebrand
 */
public interface ItemOrderWorkingMemory extends AutoCloseable {

    /**
     * Store a new item in working memory.
     * The item is added with maximum primacy and existing items undergo
     * temporal decay according to the primacy gradient.
     *
     * @param item the pattern to store
     * @param timestamp the temporal position of the item
     */
    void storeItem(Pattern item, double timestamp);

    /**
     * Store a complete temporal sequence in working memory.
     * Items are stored in temporal order with appropriate primacy values.
     *
     * @param sequence the temporal pattern to store
     */
    default void storeSequence(TemporalPattern sequence) {
        var items = sequence.getSequence();
        for (int i = 0; i < items.size(); i++) {
            storeItem(items.get(i), i);
        }
    }

    /**
     * Get the current contents of working memory as a temporal pattern.
     * Items are ordered by their temporal position and include primacy values.
     *
     * @return current working memory contents
     */
    TemporalPattern getCurrentContents();

    /**
     * Get the primacy values for all items currently in working memory.
     * Primacy values indicate the temporal strength/activation of each item.
     *
     * @return array of primacy values in temporal order
     */
    double[] getPrimacyValues();

    /**
     * Get the temporal positions of items in working memory.
     *
     * @return array of timestamps in temporal order
     */
    double[] getTemporalPositions();

    /**
     * Update the working memory state by advancing time.
     * This applies temporal decay to all stored items according to
     * the primacy gradient dynamics.
     *
     * @param deltaTime the time step for updating dynamics
     */
    void updateDynamics(double deltaTime);

    /**
     * Clear all items from working memory.
     * Resets the memory to empty state.
     */
    void clear();

    /**
     * Check if working memory is currently empty.
     *
     * @return true if no items are stored
     */
    boolean isEmpty();

    /**
     * Get the number of items currently in working memory.
     *
     * @return current item count
     */
    int getItemCount();

    /**
     * Get the maximum capacity of working memory.
     *
     * @return maximum number of items that can be stored
     */
    int getCapacity();

    /**
     * Check if working memory is at full capacity.
     *
     * @return true if no more items can be stored
     */
    default boolean isFull() {
        return getItemCount() >= getCapacity();
    }

    /**
     * Get items within a specific primacy threshold.
     * Only items with primacy values above the threshold are returned.
     *
     * @param threshold minimum primacy value
     * @return temporal pattern containing items above threshold
     */
    TemporalPattern getItemsAboveThreshold(double threshold);

    /**
     * Get the most recent items up to a specified count.
     *
     * @param count maximum number of recent items to retrieve
     * @return temporal pattern containing the most recent items
     */
    TemporalPattern getRecentItems(int count);

    /**
     * Get the item with maximum primacy value.
     *
     * @return the item with highest temporal activation
     */
    Pattern getMostSalientItem();

    /**
     * Calculate the total activation in working memory.
     * This is the sum of all primacy values.
     *
     * @return total working memory activation
     */
    double getTotalActivation();

    /**
     * Get the average primacy value across all items.
     *
     * @return mean primacy value
     */
    default double getAveragePrimacy() {
        var count = getItemCount();
        return count > 0 ? getTotalActivation() / count : 0.0;
    }

    /**
     * Check if a specific item is currently in working memory.
     *
     * @param item the pattern to search for
     * @return true if the item is present
     */
    boolean containsItem(Pattern item);

    /**
     * Get the primacy value for a specific item.
     *
     * @param item the pattern to query
     * @return primacy value, or 0.0 if item not found
     */
    double getItemPrimacy(Pattern item);

    /**
     * Set the parameters for working memory dynamics.
     *
     * @param parameters the working memory configuration
     */
    void setParameters(WorkingMemoryParameters parameters);

    /**
     * Get the current working memory parameters.
     *
     * @return current parameter configuration
     */
    WorkingMemoryParameters getParameters();

    /**
     * Create a snapshot of the current working memory state.
     * This is useful for visualization and debugging.
     *
     * @return immutable snapshot of current state
     */
    WorkingMemorySnapshot createSnapshot();

    /**
     * Restore working memory from a snapshot.
     *
     * @param snapshot the state to restore
     */
    void restoreSnapshot(WorkingMemorySnapshot snapshot);

    /**
     * Get performance statistics for working memory operations.
     *
     * @return performance metrics
     */
    WorkingMemoryPerformanceMetrics getPerformanceMetrics();

    /**
     * Reset performance tracking counters.
     */
    void resetPerformanceTracking();

    /**
     * Immutable snapshot of working memory state.
     */
    interface WorkingMemorySnapshot {
        /**
         * Get the stored temporal pattern.
         *
         * @return temporal pattern at snapshot time
         */
        TemporalPattern getStoredPattern();

        /**
         * Get the primacy values at snapshot time.
         *
         * @return array of primacy values
         */
        double[] getPrimacyValues();

        /**
         * Get the timestamp when snapshot was created.
         *
         * @return snapshot creation time
         */
        double getSnapshotTime();

        /**
         * Get the total activation at snapshot time.
         *
         * @return total working memory activation
         */
        double getTotalActivation();
    }

    /**
     * Performance metrics for working memory operations.
     */
    interface WorkingMemoryPerformanceMetrics {
        /**
         * Get the number of store operations performed.
         *
         * @return store operation count
         */
        long getStoreOperations();

        /**
         * Get the number of dynamics updates performed.
         *
         * @return update operation count
         */
        long getUpdateOperations();

        /**
         * Get the total time spent in dynamics calculations.
         *
         * @return computation time in nanoseconds
         */
        long getComputationTime();

        /**
         * Get the current memory usage.
         *
         * @return memory usage in bytes
         */
        long getMemoryUsage();

        /**
         * Get the average access time for memory operations.
         *
         * @return average access time in nanoseconds
         */
        double getAverageAccessTime();
    }
}