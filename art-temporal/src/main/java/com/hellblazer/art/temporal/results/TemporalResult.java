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
package com.hellblazer.art.temporal.results;

import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.temporal.TemporalPattern;
import java.util.List;
import java.util.Optional;

/**
 * Result of temporal pattern processing including sequence chunking information,
 * working memory state, and masking field activations.
 *
 * This result extends basic ART processing with temporal-specific information
 * from the masking field network and working memory processing described in
 * Kazerounian & Grossberg 2014.
 *
 * @author Hal Hildebrand
 */
public interface TemporalResult {

    /**
     * Get the underlying ART activation result.
     * This provides access to the basic category information (winning category,
     * activation values, etc.) from the underlying ART algorithm.
     *
     * @return the activation result from the base ART algorithm
     */
    ActivationResult getActivationResult();

    /**
     * Get the temporal chunks that were identified during processing.
     * Chunks represent subsequences that formed coherent categories.
     *
     * @return list of temporal chunks (may be empty)
     */
    List<TemporalPattern> getIdentifiedChunks();

    /**
     * Get the primary chunk that was formed or activated.
     * For sequences that form multiple chunks, this is typically the
     * longest or most salient chunk.
     *
     * @return the primary chunk, or empty if no chunks were formed
     */
    Optional<TemporalPattern> getPrimaryChunk();

    /**
     * Check if new chunks were created during this processing.
     *
     * @return true if new temporal chunks were learned
     */
    boolean hasNewChunks();

    /**
     * Get the number of new chunks created.
     *
     * @return count of newly created chunks
     */
    default int getNewChunkCount() {
        return hasNewChunks() ? getIdentifiedChunks().size() : 0;
    }

    /**
     * Get the working memory state after processing.
     * This shows what items remain in working memory with their primacy values.
     *
     * @return current working memory contents
     */
    TemporalPattern getWorkingMemoryState();

    /**
     * Get the masking field activations at different scales.
     * Returns a multi-dimensional array where first dimension is scale
     * and second dimension is spatial position within that scale.
     *
     * @return masking field activations [scale][position]
     */
    double[][] getMaskingFieldActivations();

    /**
     * Get the habituative transmitter gate values.
     * These values control the flow of information through the masking field.
     *
     * @return transmitter gate values [scale][position]
     */
    double[][] getTransmitterGateValues();

    /**
     * Get the temporal processing time for this result.
     * This is the simulated time or processing steps required.
     *
     * @return processing time in algorithm-specific units
     */
    double getProcessingTime();

    /**
     * Check if temporal resonance was achieved.
     * Resonance occurs when the temporal pattern forms a stable chunk
     * that satisfies the vigilance criteria.
     *
     * @return true if temporal resonance was achieved
     */
    boolean hasTemporalResonance();

    /**
     * Get the resonance quality measure.
     * Higher values indicate stronger, more stable temporal resonance.
     *
     * @return resonance quality (0.0 to 1.0)
     */
    double getResonanceQuality();

    /**
     * Check if the sequence required chunking boundary detection.
     * Some sequences may be processed as single units without chunking.
     *
     * @return true if chunking boundary detection was performed
     */
    boolean requiredChunking();

    /**
     * Get chunk boundary positions within the original sequence.
     * These positions indicate where the masking field identified
     * natural breakpoints in the temporal pattern.
     *
     * @return list of boundary positions (indices in original sequence)
     */
    List<Integer> getChunkBoundaries();

    /**
     * Get temporal prediction information if this was a prediction result.
     * For learning results, this may be empty.
     *
     * @return temporal prediction details
     */
    Optional<TemporalPrediction> getPrediction();

    /**
     * Get performance metrics for temporal processing.
     * This includes timing, memory usage, and computational efficiency metrics.
     *
     * @return performance metrics
     */
    TemporalPerformanceMetrics getPerformanceMetrics();

    /**
     * Check if this result represents successful learning.
     * Learning is successful if the pattern was either matched to an existing
     * category or formed a new category with adequate vigilance.
     *
     * @return true if learning was successful
     */
    default boolean isSuccessful() {
        return getActivationResult() instanceof ActivationResult.Success;
    }

    /**
     * Get a summary description of the temporal processing result.
     *
     * @return human-readable description of the result
     */
    default String getResultSummary() {
        var sb = new StringBuilder();
        sb.append("TemporalResult[");

        if (isSuccessful() && getActivationResult() instanceof ActivationResult.Success success) {
            sb.append("category=").append(success.categoryIndex());
        } else {
            sb.append("no_category");
        }

        var chunkCount = getIdentifiedChunks().size();
        if (chunkCount > 0) {
            sb.append(", chunks=").append(chunkCount);
            if (hasNewChunks()) {
                sb.append("(").append(getNewChunkCount()).append(" new)");
            }
        }

        if (hasTemporalResonance()) {
            sb.append(", resonance=").append(String.format("%.3f", getResonanceQuality()));
        }

        sb.append(", time=").append(String.format("%.2f", getProcessingTime()));
        sb.append("]");

        return sb.toString();
    }

    /**
     * Nested interface for temporal prediction information.
     */
    interface TemporalPrediction {
        /**
         * Get the predicted next items in the sequence.
         *
         * @return predicted continuation patterns
         */
        List<TemporalPattern> getPredictedContinuations();

        /**
         * Get confidence values for each prediction.
         *
         * @return confidence values (0.0 to 1.0)
         */
        List<Double> getPredictionConfidences();

        /**
         * Get the prediction horizon (how far ahead predictions extend).
         *
         * @return number of future time steps predicted
         */
        int getPredictionHorizon();
    }

    /**
     * Performance metrics for temporal processing.
     */
    interface TemporalPerformanceMetrics {
        /**
         * Get the working memory processing time.
         *
         * @return time spent in working memory operations
         */
        double getWorkingMemoryTime();

        /**
         * Get the masking field processing time.
         *
         * @return time spent in masking field operations
         */
        double getMaskingFieldTime();

        /**
         * Get the chunking processing time.
         *
         * @return time spent in chunking operations
         */
        double getChunkingTime();

        /**
         * Get the total memory usage during processing.
         *
         * @return memory usage in bytes
         */
        long getMemoryUsage();

        /**
         * Get the number of SIMD operations performed.
         *
         * @return SIMD operation count
         */
        long getSIMDOperationCount();
    }
}