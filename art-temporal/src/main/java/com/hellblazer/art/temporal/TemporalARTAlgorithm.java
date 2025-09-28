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

import com.hellblazer.art.core.ARTAlgorithm;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.temporal.results.TemporalResult;
import java.util.List;

/**
 * Base interface for temporal ART algorithms that process sequential patterns using
 * masking fields and working memory as described in:
 *
 * Kazerounian, S., & Grossberg, S. (2014). Real-time learning of predictive recognition
 * categories that chunk sequences of items stored in working memory.
 * Frontiers in Psychology, 5, 1053. https://doi.org/10.3389/fpsyg.2014.01053
 *
 * Temporal ART algorithms extend basic ART processing with:
 * - Item-and-order working memory (STORE 2 model)
 * - Masking field networks for temporal preprocessing
 * - Sequence chunking and predictive category formation
 * - Real-time learning of temporal patterns
 *
 * The temporal processing pipeline:
 * 1. Sequences enter working memory with primacy gradients
 * 2. Masking fields apply habituative gating and competitive dynamics
 * 3. Temporal chunks are extracted and fed to underlying ART algorithms
 * 4. Categories learn both content and temporal structure
 *
 * @param <P> the type of parameters used by this temporal algorithm
 *
 * @author Hal Hildebrand
 */
public interface TemporalARTAlgorithm<P> extends ARTAlgorithm<P> {

    // === Temporal Learning Interface ===

    /**
     * Learn from a temporal pattern sequence.
     * The sequence is processed through working memory and masking fields
     * before category formation in the underlying ART algorithm.
     *
     * @param temporalPattern the temporal pattern to learn
     * @return temporal learning result with chunking information
     */
    TemporalResult learnTemporal(TemporalPattern temporalPattern);

    /**
     * Predict the category for a temporal pattern sequence.
     * Uses the same temporal processing pipeline as learning but without
     * weight updates.
     *
     * @param temporalPattern the temporal pattern to classify
     * @return temporal prediction result
     */
    TemporalResult predictTemporal(TemporalPattern temporalPattern);

    /**
     * Process a real-time sequence item by item.
     * This method supports online temporal processing where items arrive
     * sequentially and chunks are formed dynamically.
     *
     * @param item the next item in the sequence
     * @return temporal result, possibly containing a new chunk
     */
    TemporalResult processSequenceItem(Pattern item);

    /**
     * Reset the temporal processing state.
     * Clears working memory, resets masking fields, and prepares for
     * a new sequence.
     */
    void resetTemporalState();

    // === Sequence Chunking Interface ===

    /**
     * Get the current temporal chunks stored in memory.
     * Chunks represent learned subsequences that have formed stable categories.
     *
     * @return list of temporal patterns representing learned chunks
     */
    List<TemporalPattern> getTemporalChunks();

    /**
     * Get the number of temporal chunks learned.
     *
     * @return chunk count
     */
    default int getChunkCount() {
        return getTemporalChunks().size();
    }

    /**
     * Check if a temporal pattern would create a new chunk.
     * This method performs temporal processing without learning to
     * determine if the pattern would form a new category.
     *
     * @param temporalPattern the pattern to test
     * @return true if a new chunk would be created
     */
    boolean wouldCreateNewChunk(TemporalPattern temporalPattern);

    // === Working Memory Interface ===

    /**
     * Get the current contents of working memory.
     * Working memory maintains recently processed items with primacy gradients.
     *
     * @return current working memory contents as temporal pattern
     */
    TemporalPattern getWorkingMemoryContents();

    /**
     * Check if working memory is currently active.
     *
     * @return true if working memory contains items
     */
    default boolean isWorkingMemoryActive() {
        var contents = getWorkingMemoryContents();
        return contents != null && !contents.isEmpty();
    }

    /**
     * Get the maximum capacity of working memory.
     *
     * @return maximum number of items that can be stored in working memory
     */
    int getWorkingMemoryCapacity();

    // === Masking Field Interface ===

    /**
     * Get the current activations in the masking field network.
     * The masking field applies competitive dynamics and habituative gating
     * to temporal patterns.
     *
     * @return current masking field activations (multi-scale)
     */
    double[][] getMaskingFieldActivations();

    /**
     * Check if the masking field is currently active.
     *
     * @return true if masking field has non-zero activations
     */
    default boolean isMaskingFieldActive() {
        var activations = getMaskingFieldActivations();
        if (activations == null) return false;

        for (var scale : activations) {
            for (var activation : scale) {
                if (activation > 0.0) return true;
            }
        }
        return false;
    }

    // === Temporal Parameters ===

    /**
     * Get the temporal-specific parameters.
     * These parameters control working memory, masking fields, and chunking behavior.
     *
     * @return temporal parameters object
     */
    P getTemporalParameters();

    /**
     * Update temporal parameters.
     * This allows dynamic adjustment of temporal processing characteristics.
     *
     * @param parameters new temporal parameters
     */
    void setTemporalParameters(P parameters);

    // === Batch Processing ===

    /**
     * Learn from multiple temporal patterns in batch mode.
     * This method can optimize processing for multiple sequences.
     *
     * @param temporalPatterns list of temporal patterns to learn
     * @return list of temporal results for each pattern
     */
    default List<TemporalResult> learnTemporalBatch(List<TemporalPattern> temporalPatterns) {
        return temporalPatterns.stream()
                              .map(this::learnTemporal)
                              .toList();
    }

    /**
     * Predict categories for multiple temporal patterns in batch mode.
     *
     * @param temporalPatterns list of temporal patterns to classify
     * @return list of temporal results for each pattern
     */
    default List<TemporalResult> predictTemporalBatch(List<TemporalPattern> temporalPatterns) {
        return temporalPatterns.stream()
                              .map(this::predictTemporal)
                              .toList();
    }

    // === Integration with Base ART ===

    /**
     * Learn from a regular (non-temporal) pattern.
     * This method provides compatibility with base ART interface.
     * The pattern is treated as a single-item temporal sequence.
     *
     * @param pattern the pattern to learn
     * @param parameters the algorithm parameters
     * @return activation result from underlying ART algorithm
     */
    @Override
    default ActivationResult learn(Pattern pattern, P parameters) {
        var temporalPattern = createSingleItemTemporalPattern(pattern);
        var temporalResult = learnTemporal(temporalPattern);
        return temporalResult.getActivationResult();
    }

    /**
     * Predict category for a regular (non-temporal) pattern.
     *
     * @param pattern the pattern to classify
     * @param parameters the algorithm parameters
     * @return activation result from underlying ART algorithm
     */
    @Override
    default ActivationResult predict(Pattern pattern, P parameters) {
        var temporalPattern = createSingleItemTemporalPattern(pattern);
        var temporalResult = predictTemporal(temporalPattern);
        return temporalResult.getActivationResult();
    }

    /**
     * Helper method to create a single-item temporal pattern.
     *
     * @param pattern the pattern to wrap
     * @return temporal pattern containing single item
     */
    private TemporalPattern createSingleItemTemporalPattern(Pattern pattern) {
        return new TemporalPattern() {
            @Override
            public List<Pattern> getSequence() {
                return List.of(pattern);
            }

            @Override
            public TemporalPattern getSubsequence(int startTime, int endTime) {
                if (startTime == 0 && endTime == 1) return this;
                throw new IndexOutOfBoundsException("Invalid subsequence bounds for single-item pattern");
            }
        };
    }
}