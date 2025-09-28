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
package com.hellblazer.art.temporal;

import com.hellblazer.art.core.Pattern;
import java.util.List;

/**
 * Represents a temporal pattern consisting of a sequence of items with temporal structure.
 * This interface provides temporal sequence processing capabilities as described in
 * Kazerounian & Grossberg 2014.
 *
 * A temporal pattern captures both the content (what items are present) and the order
 * (when items occur in sequence) for processing by temporal ART algorithms.
 *
 * Key Features:
 * - Sequential item representation with temporal ordering
 * - Support for variable-length sequences
 * - Integration with working memory and masking field processing
 * - Chunking and subsequence extraction capabilities
 *
 * @author Hal Hildebrand
 */
public interface TemporalPattern {

    /**
     * Get the sequence of items in temporal order.
     * Each item represents a single time step in the sequence.
     *
     * @return ordered list of pattern items (each item is a Pattern)
     */
    List<Pattern> getSequence();

    /**
     * Get the number of items in the temporal sequence.
     *
     * @return sequence length
     */
    default int getSequenceLength() {
        return getSequence().size();
    }

    /**
     * Get a specific item from the sequence by temporal position.
     *
     * @param timeStep the temporal position (0-based index)
     * @return the pattern at the specified time step
     * @throws IndexOutOfBoundsException if timeStep is invalid
     */
    default Pattern getItemAt(int timeStep) {
        return getSequence().get(timeStep);
    }

    /**
     * Extract a subsequence from this temporal pattern.
     *
     * @param startTime starting time step (inclusive)
     * @param endTime ending time step (exclusive)
     * @return new temporal pattern containing the subsequence
     * @throws IndexOutOfBoundsException if time bounds are invalid
     */
    TemporalPattern getSubsequence(int startTime, int endTime);

    /**
     * Check if this temporal pattern is empty (no items).
     *
     * @return true if sequence is empty
     */
    default boolean isEmpty() {
        return getSequenceLength() == 0;
    }

    /**
     * Get the dimensionality of individual items in the sequence.
     * All items in a temporal pattern should have the same dimensionality.
     *
     * @return dimensionality of sequence items, or 0 if sequence is empty
     */
    default int getItemDimensionality() {
        return isEmpty() ? 0 : getItemAt(0).dimension();
    }

    /**
     * Get the total dimensionality of the temporal pattern.
     * This is the dimensionality of individual items.
     * Note: This differs from flattened representations where dimensionality
     * would be itemDim * sequenceLength.
     *
     * @return the dimensionality of individual items
     */
    default int getDimensionality() {
        return getItemDimensionality();
    }

    /**
     * Validate that all items in the sequence have consistent dimensionality.
     *
     * @return true if all items have the same dimensionality
     */
    default boolean isValid() {
        if (isEmpty()) return true;

        var expectedDim = getItemDimensionality();
        return getSequence().stream()
                           .mapToInt(Pattern::dimension)
                           .allMatch(dim -> dim == expectedDim);
    }

    /**
     * Create a string representation showing sequence structure.
     * Format: [item1, item2, ..., itemN] with length info.
     *
     * @return descriptive string representation
     */
    default String toSequenceString() {
        if (isEmpty()) {
            return "TemporalPattern[empty]";
        }

        var sb = new StringBuilder();
        sb.append("TemporalPattern[length=").append(getSequenceLength())
          .append(", itemDim=").append(getItemDimensionality())
          .append(", sequence=[");

        var sequence = getSequence();
        for (int i = 0; i < sequence.size(); i++) {
            if (i > 0) sb.append(", ");
            if (i < 3 || i >= sequence.size() - 2) {
                sb.append(sequence.get(i).toString());
            } else if (i == 3) {
                sb.append("...");
            }
        }
        sb.append("]]");

        return sb.toString();
    }
}