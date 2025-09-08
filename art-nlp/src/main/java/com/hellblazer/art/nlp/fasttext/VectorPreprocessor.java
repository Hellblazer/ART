package com.hellblazer.art.nlp.fasttext;

import java.util.*;
import java.util.function.Function;

import com.hellblazer.art.core.DenseVector;

/**
 * Vector preprocessing utilities for FastText embeddings.
 * Provides normalization, complement coding, and aggregation operations
 * required for ART neural networks.
 */
public final class VectorPreprocessor {

    /**
     * Normalization strategies for vectors.
     */
    public enum NormalizationType {
        /** No normalization */
        NONE,
        /** L1 normalization (sum of absolute values = 1) */
        L1,
        /** L2 normalization (unit vector) */
        L2,
        /** Min-Max normalization to [0,1] range */
        MIN_MAX,
        /** Z-score normalization (mean=0, std=1) */
        Z_SCORE
    }

    /**
     * Aggregation strategies for multiple vectors.
     */
    public enum AggregationType {
        /** Element-wise mean */
        MEAN,
        /** Element-wise sum */
        SUM,
        /** Element-wise maximum */
        MAX,
        /** Element-wise minimum */
        MIN,
        /** Concatenate vectors */
        CONCAT,
        /** Weighted average */
        WEIGHTED_MEAN
    }

    /**
     * Apply L1 normalization (Manhattan normalization).
     * Normalizes so sum of absolute values equals 1.
     */
    public static DenseVector normalizeL1(DenseVector vector) {
        if (vector == null) {
            return null;
        }

        var values = vector.values();
        var sum = 0.0;
        
        for (var value : values) {
            sum += Math.abs(value);
        }

        if (sum == 0.0) {
            return vector; // Zero vector remains zero
        }

        var normalized = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            normalized[i] = values[i] / sum;
        }

        return new DenseVector(normalized);
    }

    /**
     * Apply L2 normalization (Euclidean normalization).
     * Normalizes to unit vector.
     */
    public static DenseVector normalizeL2(DenseVector vector) {
        if (vector == null) {
            return null;
        }

        var values = vector.values();
        var sumSquares = 0.0;
        
        for (var value : values) {
            sumSquares += value * value;
        }

        var norm = Math.sqrt(sumSquares);
        if (norm == 0.0) {
            return vector; // Zero vector remains zero
        }

        var normalized = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            normalized[i] = values[i] / norm;
        }

        return new DenseVector(normalized);
    }

    /**
     * Apply Min-Max normalization to [0,1] range.
     */
    public static DenseVector normalizeMinMax(DenseVector vector) {
        if (vector == null) {
            return null;
        }

        var values = vector.values();
        var min = Double.POSITIVE_INFINITY;
        var max = Double.NEGATIVE_INFINITY;

        for (var value : values) {
            if (value < min) min = value;
            if (value > max) max = value;
        }

        var range = max - min;
        if (range == 0.0) {
            return vector; // Constant vector remains unchanged
        }

        var normalized = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            normalized[i] = (values[i] - min) / range;
        }

        return new DenseVector(normalized);
    }

    /**
     * Apply Z-score normalization (standardization).
     */
    public static DenseVector normalizeZScore(DenseVector vector) {
        if (vector == null) {
            return null;
        }

        var values = vector.values();
        var n = values.length;
        
        // Calculate mean
        var sum = 0.0;
        for (var value : values) {
            sum += value;
        }
        var mean = sum / n;

        // Calculate standard deviation
        var sumSquareDiffs = 0.0;
        for (var value : values) {
            var diff = value - mean;
            sumSquareDiffs += diff * diff;
        }
        var std = Math.sqrt(sumSquareDiffs / n);

        if (std == 0.0) {
            return vector; // Constant vector remains unchanged
        }

        var normalized = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            normalized[i] = (values[i] - mean) / std;
        }

        return new DenseVector(normalized);
    }

    /**
     * Apply specified normalization type.
     */
    public static DenseVector normalize(DenseVector vector, NormalizationType type) {
        return switch (type) {
            case NONE -> vector;
            case L1 -> normalizeL1(vector);
            case L2 -> normalizeL2(vector);
            case MIN_MAX -> normalizeMinMax(vector);
            case Z_SCORE -> normalizeZScore(vector);
        };
    }

    /**
     * Apply complement coding for ART algorithms.
     * Returns [x, 1-x] where x is normalized to [0,1].
     */
    public static DenseVector complementCode(DenseVector vector) {
        if (vector == null) {
            return null;
        }

        // First normalize to [0,1] range
        var normalized = normalizeMinMax(vector);
        var values = normalized.values();
        
        // Create complement coded vector [x, 1-x]
        var complemented = new double[values.length * 2];
        
        for (int i = 0; i < values.length; i++) {
            complemented[i] = values[i];                    // Original value
            complemented[i + values.length] = 1.0 - values[i]; // Complement
        }

        return new DenseVector(complemented);
    }

    /**
     * Aggregate multiple vectors using mean averaging.
     */
    public static DenseVector aggregateMean(List<DenseVector> vectors) {
        if (vectors == null || vectors.isEmpty()) {
            return null;
        }

        var first = vectors.get(0);
        if (first == null) {
            return null;
        }

        var dimensions = first.dimension();
        var result = new double[dimensions];
        var count = 0;

        for (var vector : vectors) {
            if (vector != null && vector.dimension() == dimensions) {
                var values = vector.values();
                for (int i = 0; i < dimensions; i++) {
                    result[i] += values[i];
                }
                count++;
            }
        }

        if (count == 0) {
            return null;
        }

        // Average the accumulated values
        for (int i = 0; i < dimensions; i++) {
            result[i] /= count;
        }

        return new DenseVector(result);
    }

    /**
     * Aggregate multiple vectors using weighted averaging.
     */
    public static DenseVector aggregateWeightedMean(List<DenseVector> vectors, List<Double> weights) {
        if (vectors == null || vectors.isEmpty() || weights == null || weights.size() != vectors.size()) {
            return null;
        }

        var first = vectors.get(0);
        if (first == null) {
            return null;
        }

        var dimensions = first.dimension();
        var result = new double[dimensions];
        var totalWeight = 0.0;

        for (int v = 0; v < vectors.size(); v++) {
            var vector = vectors.get(v);
            var weight = weights.get(v);
            
            if (vector != null && vector.dimension() == dimensions && weight > 0) {
                var values = vector.values();
                for (int i = 0; i < dimensions; i++) {
                    result[i] += values[i] * weight;
                }
                totalWeight += weight;
            }
        }

        if (totalWeight == 0.0) {
            return null;
        }

        // Normalize by total weight
        for (int i = 0; i < dimensions; i++) {
            result[i] /= totalWeight;
        }

        return new DenseVector(result);
    }

    /**
     * Concatenate multiple vectors into a single vector.
     */
    public static DenseVector concatenate(List<DenseVector> vectors) {
        if (vectors == null || vectors.isEmpty()) {
            return null;
        }

        // Calculate total dimensions
        var totalDimensions = 0;
        for (var vector : vectors) {
            if (vector != null) {
                totalDimensions += vector.dimension();
            }
        }

        if (totalDimensions == 0) {
            return null;
        }

        var result = new double[totalDimensions];
        var offset = 0;

        for (var vector : vectors) {
            if (vector != null) {
                var values = vector.values();
                System.arraycopy(values, 0, result, offset, values.length);
                offset += values.length;
            }
        }

        return new DenseVector(result);
    }

    /**
     * Apply aggregation strategy to multiple vectors.
     */
    public static DenseVector aggregate(List<DenseVector> vectors, AggregationType type) {
        return switch (type) {
            case MEAN -> aggregateMean(vectors);
            case SUM -> aggregateSum(vectors);
            case MAX -> aggregateMax(vectors);
            case MIN -> aggregateMin(vectors);
            case CONCAT -> concatenate(vectors);
            case WEIGHTED_MEAN -> throw new IllegalArgumentException("WEIGHTED_MEAN requires weights parameter");
        };
    }

    /**
     * Sum multiple vectors element-wise.
     */
    private static DenseVector aggregateSum(List<DenseVector> vectors) {
        if (vectors == null || vectors.isEmpty()) {
            return null;
        }

        var first = vectors.get(0);
        if (first == null) {
            return null;
        }

        var dimensions = first.dimension();
        var result = new double[dimensions];

        for (var vector : vectors) {
            if (vector != null && vector.dimension() == dimensions) {
                var values = vector.values();
                for (int i = 0; i < dimensions; i++) {
                    result[i] += values[i];
                }
            }
        }

        return new DenseVector(result);
    }

    /**
     * Take element-wise maximum of multiple vectors.
     */
    private static DenseVector aggregateMax(List<DenseVector> vectors) {
        if (vectors == null || vectors.isEmpty()) {
            return null;
        }

        var first = vectors.get(0);
        if (first == null) {
            return null;
        }

        var dimensions = first.dimension();
        var result = new double[dimensions];
        Arrays.fill(result, Double.NEGATIVE_INFINITY);

        for (var vector : vectors) {
            if (vector != null && vector.dimension() == dimensions) {
                var values = vector.values();
                for (int i = 0; i < dimensions; i++) {
                    if (values[i] > result[i]) {
                        result[i] = values[i];
                    }
                }
            }
        }

        // Replace any remaining -Infinity with 0
        for (int i = 0; i < dimensions; i++) {
            if (result[i] == Double.NEGATIVE_INFINITY) {
                result[i] = 0.0;
            }
        }

        return new DenseVector(result);
    }

    /**
     * Take element-wise minimum of multiple vectors.
     */
    private static DenseVector aggregateMin(List<DenseVector> vectors) {
        if (vectors == null || vectors.isEmpty()) {
            return null;
        }

        var first = vectors.get(0);
        if (first == null) {
            return null;
        }

        var dimensions = first.dimension();
        var result = new double[dimensions];
        Arrays.fill(result, Double.POSITIVE_INFINITY);

        for (var vector : vectors) {
            if (vector != null && vector.dimension() == dimensions) {
                var values = vector.values();
                for (int i = 0; i < dimensions; i++) {
                    if (values[i] < result[i]) {
                        result[i] = values[i];
                    }
                }
            }
        }

        // Replace any remaining +Infinity with 0
        for (int i = 0; i < dimensions; i++) {
            if (result[i] == Double.POSITIVE_INFINITY) {
                result[i] = 0.0;
            }
        }

        return new DenseVector(result);
    }

    /**
     * Create preprocessing pipeline builder.
     */
    public static PreprocessingPipeline.Builder pipeline() {
        return new PreprocessingPipeline.Builder();
    }

    /**
     * Immutable preprocessing pipeline that applies transformations in sequence.
     */
    public static final class PreprocessingPipeline {
        private final List<Function<DenseVector, DenseVector>> transformations;

        private PreprocessingPipeline(List<Function<DenseVector, DenseVector>> transformations) {
            this.transformations = List.copyOf(transformations);
        }

        /**
         * Apply all transformations in sequence.
         */
        public DenseVector apply(DenseVector input) {
            var current = input;
            for (var transformation : transformations) {
                current = transformation.apply(current);
                if (current == null) {
                    break; // Stop on null result
                }
            }
            return current;
        }

        /**
         * Apply pipeline to multiple vectors.
         */
        public List<DenseVector> apply(List<DenseVector> inputs) {
            return inputs.stream()
                        .map(this::apply)
                        .filter(Objects::nonNull)
                        .toList();
        }

        /**
         * Get number of transformation steps.
         */
        public int getStepCount() {
            return transformations.size();
        }

        /**
         * Builder for preprocessing pipeline.
         */
        public static final class Builder {
            private final List<Function<DenseVector, DenseVector>> transformations = new ArrayList<>();

            /**
             * Add normalization step.
             */
            public Builder normalize(NormalizationType type) {
                transformations.add(vector -> VectorPreprocessor.normalize(vector, type));
                return this;
            }

            /**
             * Add complement coding step.
             */
            public Builder complementCode() {
                transformations.add(VectorPreprocessor::complementCode);
                return this;
            }

            /**
             * Add custom transformation step.
             */
            public Builder transform(Function<DenseVector, DenseVector> transformation) {
                transformations.add(transformation);
                return this;
            }

            /**
             * Build immutable pipeline.
             */
            public PreprocessingPipeline build() {
                if (transformations.isEmpty()) {
                    throw new IllegalStateException("Pipeline must have at least one transformation");
                }
                return new PreprocessingPipeline(transformations);
            }
        }
    }
}