package com.hellblazer.art.nlp.fasttext;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.assertj.core.api.Assertions.*;

import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.nlp.fasttext.VectorPreprocessor.NormalizationType;
import com.hellblazer.art.nlp.fasttext.VectorPreprocessor.AggregationType;

/**
 * Tests for VectorPreprocessor utilities.
 */
@DisplayName("VectorPreprocessor Tests")
class VectorPreprocessorTest {

    @Test
    @DisplayName("Should apply L1 normalization correctly")
    void shouldApplyL1Normalization() {
        var vector = new DenseVector(new double[]{3.0f, 4.0f, -5.0}); // L1 norm = 12
        var normalized = VectorPreprocessor.normalizeL1(vector);
        
        assertThat(normalized).isNotNull();
        var values = normalized.values();
        assertThat(values[0]).isCloseTo(0.25, offset(1e-6));    // 3/12
        assertThat(values[1]).isCloseTo(0.333333, offset(1e-5)); // 4/12
        assertThat(values[2]).isCloseTo(-0.416667, offset(1e-5)); // -5/12
        
        // Verify L1 norm is 1
        var l1Norm = Math.abs(values[0]) + Math.abs(values[1]) + Math.abs(values[2]);
        assertThat(l1Norm).isCloseTo(1.0, offset(1e-6));
    }

    @Test
    @DisplayName("Should apply L2 normalization correctly")
    void shouldApplyL2Normalization() {
        var vector = new DenseVector(new double[]{3.0f, 4.0f, 0.0}); // L2 norm = 5
        var normalized = VectorPreprocessor.normalizeL2(vector);
        
        assertThat(normalized).isNotNull();
        var values = normalized.values();
        assertThat(values[0]).isCloseTo(0.6, offset(1e-6));  // 3/5
        assertThat(values[1]).isCloseTo(0.8, offset(1e-6));  // 4/5
        assertThat(values[2]).isCloseTo(0.0, offset(1e-6));  // 0/5
        
        // Verify L2 norm is 1
        var l2Norm = Math.sqrt(values[0]*values[0] + values[1]*values[1] + values[2]*values[2]);
        assertThat(l2Norm).isCloseTo(1.0, offset(1e-6));
    }

    @Test
    @DisplayName("Should apply Min-Max normalization correctly")
    void shouldApplyMinMaxNormalization() {
        var vector = new DenseVector(new double[]{10.0f, 20.0f, 5.0}); // Min=5, Max=20, Range=15
        var normalized = VectorPreprocessor.normalizeMinMax(vector);
        
        assertThat(normalized).isNotNull();
        var values = normalized.values();
        assertThat(values[0]).isCloseTo(0.333333, offset(1e-5)); // (10-5)/15
        assertThat(values[1]).isCloseTo(1.0, offset(1e-6));      // (20-5)/15
        assertThat(values[2]).isCloseTo(0.0, offset(1e-6));      // (5-5)/15
        
        // Verify all values are in [0,1]
        for (var value : values) {
            assertThat(value).isBetween(0.0, 1.0);
        }
    }

    @Test
    @DisplayName("Should apply Z-score normalization correctly")
    void shouldApplyZScoreNormalization() {
        var vector = new DenseVector(new double[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0}); // Mean=3, Std=√2
        var normalized = VectorPreprocessor.normalizeZScore(vector);
        
        assertThat(normalized).isNotNull();
        var values = normalized.values();
        
        // Check mean is approximately 0
        var mean = (values[0] + values[1] + values[2] + values[3] + values[4]) / 5;
        assertThat(mean).isCloseTo(0.0, offset(1e-5));
        
        // Check standard deviation is approximately 1
        var variance = 0.0f;
        for (var value : values) {
            variance += (value - mean) * (value - mean);
        }
        var std = Math.sqrt(variance / values.length);
        assertThat(std).isCloseTo(1.0, offset(1e-5));
    }

    @Test
    @DisplayName("Should handle zero vectors in normalization")
    void shouldHandleZeroVectorsInNormalization() {
        var zeroVector = new DenseVector(new double[]{0.0f, 0.0f, 0.0});
        
        // L1 and L2 normalization should return original zero vector
        var l1Normalized = VectorPreprocessor.normalizeL1(zeroVector);
        var l2Normalized = VectorPreprocessor.normalizeL2(zeroVector);
        
        assertThat(l1Normalized.values()).containsExactly(0.0f, 0.0f, 0.0);
        assertThat(l2Normalized.values()).containsExactly(0.0f, 0.0f, 0.0);
    }

    @Test
    @DisplayName("Should handle constant vectors in normalization")
    void shouldHandleConstantVectorsInNormalization() {
        var constantVector = new DenseVector(new double[]{5.0f, 5.0f, 5.0});
        
        // Min-Max should return original vector (range = 0)
        var minMaxNormalized = VectorPreprocessor.normalizeMinMax(constantVector);
        assertThat(minMaxNormalized.values()).containsExactly(5.0f, 5.0f, 5.0);
        
        // Z-score should return original vector (std = 0)
        var zScoreNormalized = VectorPreprocessor.normalizeZScore(constantVector);
        assertThat(zScoreNormalized.values()).containsExactly(5.0f, 5.0f, 5.0);
    }

    @Test
    @DisplayName("Should apply complement coding correctly")
    void shouldApplyComplementCoding() {
        var vector = new DenseVector(new double[]{0.2f, 0.8f, 0.5});
        var complementCoded = VectorPreprocessor.complementCode(vector);
        
        assertThat(complementCoded).isNotNull();
        assertThat(complementCoded.dimension()).isEqualTo(6); // Doubled
        
        var values = complementCoded.values();
        // Original values (normalized to [0,1] first, but these are already in range)
        assertThat(values[0]).isCloseTo(0.0, offset(1e-5));   // (0.2 - 0.2) / 0.6
        assertThat(values[1]).isCloseTo(1.0, offset(1e-5));   // (0.8 - 0.2) / 0.6
        assertThat(values[2]).isCloseTo(0.5, offset(1e-5));   // (0.5 - 0.2) / 0.6
        
        // Complement values
        assertThat(values[3]).isCloseTo(1.0, offset(1e-5));   // 1 - 0.0
        assertThat(values[4]).isCloseTo(0.0, offset(1e-5));   // 1 - 1.0
        assertThat(values[5]).isCloseTo(0.5, offset(1e-5));   // 1 - 0.5
    }

    @Test
    @DisplayName("Should aggregate vectors using mean")
    void shouldAggregateVectorsUsingMean() {
        var vectors = List.of(
            new DenseVector(new double[]{1.0f, 2.0f, 3.0}),
            new DenseVector(new double[]{4.0f, 5.0f, 6.0}),
            new DenseVector(new double[]{7.0f, 8.0f, 9.0})
        );
        
        var mean = VectorPreprocessor.aggregateMean(vectors);
        
        assertThat(mean).isNotNull();
        var values = mean.values();
        assertThat(values[0]).isCloseTo(4.0, offset(1e-6)); // (1+4+7)/3
        assertThat(values[1]).isCloseTo(5.0, offset(1e-6)); // (2+5+8)/3
        assertThat(values[2]).isCloseTo(6.0, offset(1e-6)); // (3+6+9)/3
    }

    @Test
    @DisplayName("Should aggregate vectors using weighted mean")
    void shouldAggregateVectorsUsingWeightedMean() {
        var vectors = List.of(
            new DenseVector(new double[]{1.0f, 2.0}),
            new DenseVector(new double[]{3.0f, 4.0})
        );
        var weights = List.of(0.3, 0.7);
        
        var weightedMean = VectorPreprocessor.aggregateWeightedMean(vectors, weights);
        
        assertThat(weightedMean).isNotNull();
        var values = weightedMean.values();
        assertThat(values[0]).isCloseTo(2.4, offset(1e-6)); // (1*0.3 + 3*0.7)
        assertThat(values[1]).isCloseTo(3.4, offset(1e-6)); // (2*0.3 + 4*0.7)
    }

    @Test
    @DisplayName("Should concatenate vectors correctly")
    void shouldConcatenateVectors() {
        var vectors = List.of(
            new DenseVector(new double[]{1.0f, 2.0}),
            new DenseVector(new double[]{3.0f, 4.0f, 5.0}),
            new DenseVector(new double[]{6.0})
        );
        
        var concatenated = VectorPreprocessor.concatenate(vectors);
        
        assertThat(concatenated).isNotNull();
        assertThat(concatenated.dimension()).isEqualTo(6);
        assertThat(concatenated.values()).containsExactly(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    }

    @Test
    @DisplayName("Should handle null vectors in aggregation")
    void shouldHandleNullVectorsInAggregation() {
        var vectors = new ArrayList<DenseVector>();
        vectors.add(new DenseVector(new double[]{1.0, 2.0}));
        vectors.add(null);
        vectors.add(new DenseVector(new double[]{3.0, 4.0}));
        
        var mean = VectorPreprocessor.aggregateMean(vectors);
        
        assertThat(mean).isNotNull();
        var values = mean.values();
        assertThat(values[0]).isCloseTo(2.0, offset(1e-6)); // (1+3)/2
        assertThat(values[1]).isCloseTo(3.0, offset(1e-6)); // (2+4)/2
    }

    @Test
    @DisplayName("Should handle dimension mismatches in aggregation")
    void shouldHandleDimensionMismatchesInAggregation() {
        var vectors = List.of(
            new DenseVector(new double[]{1.0f, 2.0}),
            new DenseVector(new double[]{3.0f, 4.0f, 5.0}), // Wrong dimensions
            new DenseVector(new double[]{6.0f, 7.0})
        );
        
        var mean = VectorPreprocessor.aggregateMean(vectors);
        
        assertThat(mean).isNotNull();
        var values = mean.values();
        // Should average only the vectors with matching dimensions
        assertThat(values[0]).isCloseTo(3.5, offset(1e-6)); // (1+6)/2
        assertThat(values[1]).isCloseTo(4.5, offset(1e-6)); // (2+7)/2
    }

    @Test
    @DisplayName("Should handle empty vector list")
    void shouldHandleEmptyVectorList() {
        var emptyList = List.<DenseVector>of();
        
        assertThat(VectorPreprocessor.aggregateMean(emptyList)).isNull();
        assertThat(VectorPreprocessor.concatenate(emptyList)).isNull();
    }

    @Test
    @DisplayName("Should handle null inputs gracefully")
    void shouldHandleNullInputsGracefully() {
        assertThat(VectorPreprocessor.normalizeL1(null)).isNull();
        assertThat(VectorPreprocessor.normalizeL2(null)).isNull();
        assertThat(VectorPreprocessor.normalizeMinMax(null)).isNull();
        assertThat(VectorPreprocessor.normalizeZScore(null)).isNull();
        assertThat(VectorPreprocessor.complementCode(null)).isNull();
        assertThat(VectorPreprocessor.aggregateMean(null)).isNull();
        assertThat(VectorPreprocessor.concatenate(null)).isNull();
    }

    @Test
    @DisplayName("Should apply normalization by type")
    void shouldApplyNormalizationByType() {
        var vector = new DenseVector(new double[]{3.0f, 4.0f, 0.0});
        
        var none = VectorPreprocessor.normalize(vector, NormalizationType.NONE);
        var l1 = VectorPreprocessor.normalize(vector, NormalizationType.L1);
        var l2 = VectorPreprocessor.normalize(vector, NormalizationType.L2);
        var minMax = VectorPreprocessor.normalize(vector, NormalizationType.MIN_MAX);
        var zScore = VectorPreprocessor.normalize(vector, NormalizationType.Z_SCORE);
        
        assertThat(none).isSameAs(vector); // No change for NONE
        assertThat(l1).isNotNull();
        assertThat(l2).isNotNull();
        assertThat(minMax).isNotNull();
        assertThat(zScore).isNotNull();
        
        // Each should produce different results
        assertThat(l1.values()).isNotEqualTo(l2.values());
        assertThat(l2.values()).isNotEqualTo(minMax.values());
    }

    @Test
    @DisplayName("Should build preprocessing pipeline")
    void shouldBuildPreprocessingPipeline() {
        var pipeline = VectorPreprocessor.pipeline()
            .normalize(NormalizationType.MIN_MAX)
            .complementCode()
            .normalize(NormalizationType.L2)
            .build();
        
        assertThat(pipeline.getStepCount()).isEqualTo(3);
    }

    @Test
    @DisplayName("Should apply preprocessing pipeline")
    void shouldApplyPreprocessingPipeline() {
        var pipeline = VectorPreprocessor.pipeline()
            .normalize(NormalizationType.MIN_MAX)
            .complementCode()
            .build();
        
        var input = new DenseVector(new double[]{10.0f, 20.0f, 5.0});
        var result = pipeline.apply(input);
        
        assertThat(result).isNotNull();
        assertThat(result.dimension()).isEqualTo(6); // Complement coded doubles size
        
        // First three should be min-max normalized, next three should be complements
        var values = result.values();
        assertThat(values[0]).isCloseTo(0.333333, offset(1e-5)); // (10-5)/15
        assertThat(values[1]).isCloseTo(1.0, offset(1e-6));      // (20-5)/15
        assertThat(values[2]).isCloseTo(0.0, offset(1e-6));      // (5-5)/15
        assertThat(values[3]).isCloseTo(0.666667, offset(1e-5)); // 1 - 0.333333
        assertThat(values[4]).isCloseTo(0.0, offset(1e-6));      // 1 - 1.0
        assertThat(values[5]).isCloseTo(1.0, offset(1e-6));      // 1 - 0.0
    }

    @Test
    @DisplayName("Should apply pipeline to multiple vectors")
    void shouldApplyPipelineToMultipleVectors() {
        var pipeline = VectorPreprocessor.pipeline()
            .normalize(NormalizationType.L2)
            .build();
        
        var inputs = new ArrayList<DenseVector>();
        inputs.add(new DenseVector(new double[]{3.0, 4.0}));
        inputs.add(new DenseVector(new double[]{1.0, 1.0}));
        inputs.add(null); // Should be filtered out
        
        var results = pipeline.apply(inputs);
        
        assertThat(results).hasSize(2); // Null filtered out
        
        // Verify L2 normalization
        for (var result : results) {
            var values = result.values();
            var norm = Math.sqrt(values[0]*values[0] + values[1]*values[1]);
            assertThat(norm).isCloseTo(1.0, offset(1e-6));
        }
    }

    @Test
    @DisplayName("Should stop pipeline on null intermediate result")
    void shouldStopPipelineOnNullIntermediateResult() {
        var pipeline = VectorPreprocessor.pipeline()
            .transform(vector -> null) // Returns null
            .normalize(NormalizationType.L2) // Should not be reached
            .build();
        
        var input = new DenseVector(new double[]{1.0f, 2.0});
        var result = pipeline.apply(input);
        
        assertThat(result).isNull();
    }

    @Test
    @DisplayName("Should require at least one transformation in pipeline")
    void shouldRequireAtLeastOneTransformationInPipeline() {
        assertThatThrownBy(() -> VectorPreprocessor.pipeline().build())
            .isInstanceOf(IllegalStateException.class)
            .hasMessageContaining("Pipeline must have at least one transformation");
    }

    @Test
    @DisplayName("Should support custom transformations in pipeline")
    void shouldSupportCustomTransformationsInPipeline() {
        var pipeline = VectorPreprocessor.pipeline()
            .transform(vector -> {
                // Double all values
                var values = vector.values();
                var doubled = new double[values.length];
                for (int i = 0; i < values.length; i++) {
                    doubled[i] = values[i] * 2;
                }
                return new DenseVector(doubled);
            })
            .build();
        
        var input = new DenseVector(new double[]{1.0, 2.0, 3.0});
        var result = pipeline.apply(input);
        
        assertThat(result).isNotNull();
        assertThat(result.values()).containsExactly(2.0f, 4.0f, 6.0);
    }
}