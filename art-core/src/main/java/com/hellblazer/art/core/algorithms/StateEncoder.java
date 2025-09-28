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
package com.hellblazer.art.core.algorithms;

import com.hellblazer.art.core.Context;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.State;

import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Interface for encoding states with contextual information in hybrid ART systems.
 *
 * StateEncoder is responsible for transforming raw states into encoded representations
 * that can be processed by ART algorithms. This includes feature extraction,
 * dimensionality reduction, normalization, and contextual augmentation.
 *
 * Key capabilities:
 * - Multi-modal state encoding (visual, textual, numerical)
 * - Context-aware feature extraction
 * - Temporal sequence encoding
 * - Incremental encoding for online learning
 * - Batch encoding for efficient processing
 *
 * @param <S> the type of states to encode
 * @param <C> the type of context information
 *
 * @author Hal Hildebrand
 * @since 1.0
 */
public interface StateEncoder<S extends State<?>, C extends Context> {

    /**
     * Encoding strategies supported by state encoders.
     */
    enum EncodingStrategy {
        /** Basic feature extraction without context */
        BASIC,
        /** Context-aware encoding incorporating environmental information */
        CONTEXTUAL,
        /** Temporal encoding preserving sequential dependencies */
        TEMPORAL,
        /** Hierarchical encoding with multi-level representations */
        HIERARCHICAL,
        /** Adaptive encoding that adjusts based on data characteristics */
        ADAPTIVE,
        /** Ensemble encoding combining multiple strategies */
        ENSEMBLE
    }

    /**
     * Normalization methods for encoded features.
     */
    enum NormalizationMethod {
        /** No normalization applied */
        NONE,
        /** Min-max normalization to [0,1] range */
        MIN_MAX,
        /** Z-score normalization (zero mean, unit variance) */
        Z_SCORE,
        /** L1 normalization (unit L1 norm) */
        L1,
        /** L2 normalization (unit L2 norm) */
        L2,
        /** ART complement coding [x, 1-x] */
        COMPLEMENT_CODING
    }

    /**
     * Encode a single state with the given context.
     *
     * @param state the state to encode
     * @param context the execution context providing encoding parameters
     * @return encoded pattern representing the state
     * @throws IllegalArgumentException if state or context is null or invalid
     */
    Pattern encode(S state, C context);

    /**
     * Encode multiple states efficiently using batch processing.
     *
     * @param states the states to encode
     * @param context the execution context
     * @return list of encoded patterns in the same order as input states
     * @throws IllegalArgumentException if states or context is null
     */
    default List<Pattern> encodeBatch(List<S> states, C context) {
        return states.stream()
                    .map(state -> encode(state, context))
                    .toList();
    }

    /**
     * Encode a temporal sequence of states preserving sequential dependencies.
     *
     * @param stateSequence the sequence of states in temporal order
     * @param context the execution context
     * @return encoded pattern representing the entire sequence
     * @throws IllegalArgumentException if sequence or context is null
     */
    default Pattern encodeSequence(List<S> stateSequence, C context) {
        // Default implementation: encode each state and concatenate
        var encodedStates = encodeBatch(stateSequence, context);
        return concatenatePatterns(encodedStates);
    }

    /**
     * Get the encoding strategy used by this encoder.
     *
     * @return the encoding strategy, never null
     */
    EncodingStrategy getStrategy();

    /**
     * Get the normalization method applied during encoding.
     *
     * @return the normalization method, never null
     */
    NormalizationMethod getNormalizationMethod();

    /**
     * Get the dimensionality of encoded patterns produced by this encoder.
     * May return -1 if dimensionality is variable or unknown.
     *
     * @return encoded pattern dimension, or -1 if variable
     */
    int getEncodedDimension();

    /**
     * Check if this encoder supports incremental/online encoding.
     * Incremental encoders can adapt their encoding based on previously seen states.
     *
     * @return true if incremental encoding is supported
     */
    default boolean supportsIncrementalEncoding() {
        return false;
    }

    /**
     * Update the encoder with a new state for incremental learning.
     * Only meaningful if supportsIncrementalEncoding() returns true.
     *
     * @param state the new state to incorporate into encoding knowledge
     * @param context the execution context
     */
    default void updateIncremental(S state, C context) {
        // Default: no-op for non-incremental encoders
    }

    /**
     * Get encoding quality metrics for the last encoded state(s).
     * Useful for monitoring encoding performance and detecting issues.
     *
     * @return optional encoding metrics map
     */
    default Optional<Map<String, Double>> getEncodingMetrics() {
        return Optional.empty();
    }

    /**
     * Check if the encoder can handle the given state type.
     *
     * @param state the state to check
     * @return true if this encoder can process the state
     */
    default boolean canEncode(S state) {
        return state != null && state.isValid();
    }

    /**
     * Get the set of features extracted during encoding.
     * Useful for feature analysis and interpretability.
     *
     * @return optional list of feature names/descriptions
     */
    default Optional<List<String>> getFeatureNames() {
        return Optional.empty();
    }

    /**
     * Get the importance or weight of each feature in the encoded representation.
     *
     * @return optional array of feature importance values
     */
    default Optional<double[]> getFeatureWeights() {
        return Optional.empty();
    }

    /**
     * Create a derived encoder with modified parameters.
     *
     * @param modifications map of parameter names to new values
     * @return new encoder with applied modifications
     */
    default StateEncoder<S, C> withParameters(Map<String, Object> modifications) {
        return this; // Default: immutable encoder
    }

    /**
     * Validate that the encoder is properly configured and ready to use.
     *
     * @return list of validation issues (empty if valid)
     */
    default List<String> validate() {
        return List.of(); // Default: assume valid
    }

    /**
     * Get memory usage statistics for this encoder.
     *
     * @return optional memory usage in bytes
     */
    default Optional<Long> getMemoryUsage() {
        return Optional.empty();
    }

    /**
     * Reset the encoder to its initial state.
     * Clears any incremental learning progress and resets internal state.
     */
    default void reset() {
        // Default: no-op for stateless encoders
    }

    /**
     * Create a copy of this encoder with the same configuration.
     *
     * @return new encoder instance with identical settings
     */
    StateEncoder<S, C> copy();

    /**
     * Helper method to concatenate multiple patterns into a single pattern.
     * Used by default implementations of sequence encoding.
     *
     * @param patterns the patterns to concatenate
     * @return concatenated pattern
     */
    default Pattern concatenatePatterns(List<Pattern> patterns) {
        if (patterns == null || patterns.isEmpty()) {
            throw new IllegalArgumentException("Cannot concatenate null or empty pattern list");
        }

        if (patterns.size() == 1) {
            return patterns.get(0);
        }

        // Calculate total dimension
        int totalDim = patterns.stream()
                              .mapToInt(Pattern::dimension)
                              .sum();

        // Create concatenated array
        double[] concatenated = new double[totalDim];
        int offset = 0;

        for (var pattern : patterns) {
            for (int i = 0; i < pattern.dimension(); i++) {
                concatenated[offset + i] = pattern.get(i);
            }
            offset += pattern.dimension();
        }

        // Return as Pattern - assumes there's a DenseVector implementation
        return new com.hellblazer.art.core.DenseVector(concatenated);
    }

    /**
     * Helper method to normalize a pattern using the specified method.
     *
     * @param pattern the pattern to normalize
     * @param method the normalization method
     * @return normalized pattern
     */
    default Pattern normalizePattern(Pattern pattern, NormalizationMethod method) {
        if (pattern == null) {
            throw new IllegalArgumentException("Cannot normalize null pattern");
        }

        return switch (method) {
            case NONE -> pattern;
            case MIN_MAX -> normalizeMinMax(pattern);
            case Z_SCORE -> normalizeZScore(pattern);
            case L1 -> normalizeL1(pattern);
            case L2 -> normalizeL2(pattern);
            case COMPLEMENT_CODING -> applyComplementCoding(pattern);
        };
    }

    /**
     * Apply min-max normalization to range [0,1].
     */
    private Pattern normalizeMinMax(Pattern pattern) {
        double min = Double.MAX_VALUE;
        double max = -Double.MAX_VALUE;

        // Find min and max
        for (int i = 0; i < pattern.dimension(); i++) {
            double value = pattern.get(i);
            min = Math.min(min, value);
            max = Math.max(max, value);
        }

        // Avoid division by zero
        if (max == min) {
            return pattern;
        }

        // Normalize
        double[] normalized = new double[pattern.dimension()];
        double range = max - min;
        for (int i = 0; i < pattern.dimension(); i++) {
            normalized[i] = (pattern.get(i) - min) / range;
        }

        return new com.hellblazer.art.core.DenseVector(normalized);
    }

    /**
     * Apply Z-score normalization (zero mean, unit variance).
     */
    private Pattern normalizeZScore(Pattern pattern) {
        // Calculate mean
        double mean = 0.0;
        for (int i = 0; i < pattern.dimension(); i++) {
            mean += pattern.get(i);
        }
        mean /= pattern.dimension();

        // Calculate standard deviation
        double variance = 0.0;
        for (int i = 0; i < pattern.dimension(); i++) {
            double diff = pattern.get(i) - mean;
            variance += diff * diff;
        }
        double stdDev = Math.sqrt(variance / pattern.dimension());

        // Avoid division by zero
        if (stdDev == 0.0) {
            return pattern;
        }

        // Normalize
        double[] normalized = new double[pattern.dimension()];
        for (int i = 0; i < pattern.dimension(); i++) {
            normalized[i] = (pattern.get(i) - mean) / stdDev;
        }

        return new com.hellblazer.art.core.DenseVector(normalized);
    }

    /**
     * Apply L1 normalization (unit L1 norm).
     */
    private Pattern normalizeL1(Pattern pattern) {
        double sum = 0.0;
        for (int i = 0; i < pattern.dimension(); i++) {
            sum += Math.abs(pattern.get(i));
        }

        if (sum == 0.0) {
            return pattern;
        }

        double[] normalized = new double[pattern.dimension()];
        for (int i = 0; i < pattern.dimension(); i++) {
            normalized[i] = pattern.get(i) / sum;
        }

        return new com.hellblazer.art.core.DenseVector(normalized);
    }

    /**
     * Apply L2 normalization (unit L2 norm).
     */
    private Pattern normalizeL2(Pattern pattern) {
        double sumSquares = 0.0;
        for (int i = 0; i < pattern.dimension(); i++) {
            double value = pattern.get(i);
            sumSquares += value * value;
        }

        double norm = Math.sqrt(sumSquares);
        if (norm == 0.0) {
            return pattern;
        }

        double[] normalized = new double[pattern.dimension()];
        for (int i = 0; i < pattern.dimension(); i++) {
            normalized[i] = pattern.get(i) / norm;
        }

        return new com.hellblazer.art.core.DenseVector(normalized);
    }

    /**
     * Apply ART complement coding: [x, 1-x].
     */
    private Pattern applyComplementCoding(Pattern pattern) {
        double[] complementCoded = new double[pattern.dimension() * 2];

        for (int i = 0; i < pattern.dimension(); i++) {
            double value = pattern.get(i);
            complementCoded[i] = value;
            complementCoded[i + pattern.dimension()] = 1.0 - value;
        }

        return new com.hellblazer.art.core.DenseVector(complementCoded);
    }
}