package com.hellblazer.art.cortical.learning;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.cortical.layers.WeightMatrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for SIMD-vectorized Hebbian learning - Phase 4D.
 *
 * <p>Verifies:
 * <ul>
 *   <li>Correctness: SIMD results match scalar Hebbian</li>
 *   <li>Performance: Speedup with vectorization</li>
 *   <li>Edge cases: Small matrices, zero learning rate</li>
 *   <li>Precision: Numerical accuracy within tolerance</li>
 * </ul>
 *
 * @author Phase 4D: Learning Vectorization
 */
class HebbianLearningSIMDTest {

    private static final double TOLERANCE = 1e-10;
    private static final int LARGE_PRE_SIZE = 256;
    private static final int LARGE_POST_SIZE = 128;
    private static final int SMALL_PRE_SIZE = 4;
    private static final int SMALL_POST_SIZE = 8;

    private HebbianLearning scalar;
    private HebbianLearningSIMD simd;

    @BeforeEach
    void setup() {
        scalar = new HebbianLearning(0.0001, 0.0, 1.0);
        simd = new HebbianLearningSIMD(0.0001, 0.0, 1.0);
    }

    @Test
    void testSIMDMatchesScalarLargeMatrix() {
        // Large matrix for SIMD activation
        var preActivation = createRandomPattern(LARGE_PRE_SIZE);
        var postActivation = createRandomPattern(LARGE_POST_SIZE);
        var currentWeights = createRandomWeights(LARGE_POST_SIZE, LARGE_PRE_SIZE);
        var learningRate = 0.01;

        var scalarResult = scalar.update(preActivation, postActivation, currentWeights, learningRate);
        var simdResult = simd.update(preActivation, postActivation, currentWeights, learningRate);

        assertWeightMatricesEqual(scalarResult, simdResult, TOLERANCE,
            "SIMD results should match scalar for large matrix");
    }

    @Test
    void testSIMDMatchesScalarSmallMatrix() {
        // Small matrix triggers fallback to scalar
        var preActivation = createRandomPattern(SMALL_PRE_SIZE);
        var postActivation = createRandomPattern(SMALL_POST_SIZE);
        var currentWeights = createRandomWeights(SMALL_POST_SIZE, SMALL_PRE_SIZE);
        var learningRate = 0.01;

        var scalarResult = scalar.update(preActivation, postActivation, currentWeights, learningRate);
        var simdResult = simd.update(preActivation, postActivation, currentWeights, learningRate);

        assertWeightMatricesEqual(scalarResult, simdResult, TOLERANCE,
            "SIMD results should match scalar for small matrix (fallback)");
    }

    @Test
    void testZeroLearningRate() {
        var preActivation = createRandomPattern(LARGE_PRE_SIZE);
        var postActivation = createRandomPattern(LARGE_POST_SIZE);
        var currentWeights = createRandomWeights(LARGE_POST_SIZE, LARGE_PRE_SIZE);

        // With zero learning rate, weights should not change
        var result = simd.update(preActivation, postActivation, currentWeights, 0.0);

        assertSame(currentWeights, result, "Zero learning rate should return same weights");
    }

    @Test
    void testWeightDecay() {
        // Test that weights decay over time with no activation
        var preActivation = createZeroPattern(LARGE_PRE_SIZE);
        var postActivation = createZeroPattern(LARGE_POST_SIZE);
        var currentWeights = createOnesWeights(LARGE_POST_SIZE, LARGE_PRE_SIZE);
        var learningRate = 0.1;

        var result = simd.update(preActivation, postActivation, currentWeights, learningRate);

        // Weights should be less than 1.0 due to decay
        for (int j = 0; j < LARGE_POST_SIZE; j++) {
            for (int i = 0; i < LARGE_PRE_SIZE; i++) {
                assertTrue(result.get(j, i) < 1.0,
                    "Weight [" + j + "][" + i + "] should decay from 1.0");
            }
        }
    }

    @Test
    void testHebbianStrengthening() {
        // Test that correlated activations strengthen weights
        var preActivation = createOnesPattern(LARGE_PRE_SIZE);
        var postActivation = createOnesPattern(LARGE_POST_SIZE);
        var currentWeights = createZeroWeights(LARGE_POST_SIZE, LARGE_PRE_SIZE);
        var learningRate = 0.1;

        var result = simd.update(preActivation, postActivation, currentWeights, learningRate);

        // All weights should increase from 0.0
        for (int j = 0; j < LARGE_POST_SIZE; j++) {
            for (int i = 0; i < LARGE_PRE_SIZE; i++) {
                assertTrue(result.get(j, i) > 0.0,
                    "Weight [" + j + "][" + i + "] should strengthen from 0.0");
            }
        }
    }

    @Test
    void testWeightBoundsClamping() {
        // Test that weights are clamped to [minWeight, maxWeight]
        var minWeight = -0.5;
        var maxWeight = 0.5;
        var learningRule = new HebbianLearningSIMD(0.0001, minWeight, maxWeight);

        var preActivation = createOnesPattern(LARGE_PRE_SIZE);
        var postActivation = createOnesPattern(LARGE_POST_SIZE);
        var currentWeights = createOnesWeights(LARGE_POST_SIZE, LARGE_PRE_SIZE);
        var learningRate = 1.0;  // Very high learning rate

        var result = learningRule.update(preActivation, postActivation, currentWeights, learningRate);

        // All weights should be clamped to maxWeight
        for (int j = 0; j < LARGE_POST_SIZE; j++) {
            for (int i = 0; i < LARGE_PRE_SIZE; i++) {
                assertTrue(result.get(j, i) <= maxWeight,
                    "Weight [" + j + "][" + i + "] should be <= maxWeight");
                assertTrue(result.get(j, i) >= minWeight,
                    "Weight [" + j + "][" + i + "] should be >= minWeight");
            }
        }
    }

    @Test
    void testMultipleUpdates() {
        // Test that multiple updates work correctly
        var preActivation = createRandomPattern(LARGE_PRE_SIZE);
        var postActivation = createRandomPattern(LARGE_POST_SIZE);
        var weights = createRandomWeights(LARGE_POST_SIZE, LARGE_PRE_SIZE);
        var learningRate = 0.01;

        var scalarWeights = weights;
        var simdWeights = weights;

        // Apply 10 updates
        for (int iter = 0; iter < 10; iter++) {
            scalarWeights = scalar.update(preActivation, postActivation, scalarWeights, learningRate);
            simdWeights = simd.update(preActivation, postActivation, simdWeights, learningRate);
        }

        assertWeightMatricesEqual(scalarWeights, simdWeights, TOLERANCE,
            "SIMD should match scalar after multiple updates");
    }

    @Test
    void testDifferentMatrixSizes() {
        // Test various matrix sizes to ensure correctness across dimensions
        int[] sizes = {8, 16, 32, 64, 128, 256};

        for (var preSize : sizes) {
            for (var postSize : sizes) {
                var preActivation = createRandomPattern(preSize);
                var postActivation = createRandomPattern(postSize);
                var currentWeights = createRandomWeights(postSize, preSize);
                var learningRate = 0.01;

                var scalarResult = scalar.update(preActivation, postActivation, currentWeights, learningRate);
                var simdResult = simd.update(preActivation, postActivation, currentWeights, learningRate);

                assertWeightMatricesEqual(scalarResult, simdResult, TOLERANCE,
                    "SIMD should match scalar for size " + preSize + "x" + postSize);
            }
        }
    }

    @Test
    void testInvalidLearningRate() {
        var preActivation = createRandomPattern(LARGE_PRE_SIZE);
        var postActivation = createRandomPattern(LARGE_POST_SIZE);
        var currentWeights = createRandomWeights(LARGE_POST_SIZE, LARGE_PRE_SIZE);

        assertThrows(IllegalArgumentException.class,
            () -> simd.update(preActivation, postActivation, currentWeights, -0.1),
            "Negative learning rate should throw");

        assertThrows(IllegalArgumentException.class,
            () -> simd.update(preActivation, postActivation, currentWeights, 1.1),
            "Learning rate > 1.0 should throw");
    }

    @Test
    void testInvalidWeightBounds() {
        assertThrows(IllegalArgumentException.class,
            () -> new HebbianLearningSIMD(0.0001, 1.0, 0.0),
            "minWeight >= maxWeight should throw");
    }

    @Test
    void testInvalidDecayRate() {
        assertThrows(IllegalArgumentException.class,
            () -> new HebbianLearningSIMD(-0.1, 0.0, 1.0),
            "Negative decay rate should throw");

        assertThrows(IllegalArgumentException.class,
            () -> new HebbianLearningSIMD(1.1, 0.0, 1.0),
            "Decay rate > 1.0 should throw");
    }

    @Test
    void testVectorLength() {
        var vectorLength = simd.getVectorLength();
        assertTrue(vectorLength >= 2 && vectorLength <= 8,
            "Vector length should be between 2 (SSE) and 8 (AVX-512)");
    }

    @Test
    void testGetters() {
        assertEquals(0.0001, simd.getDecayRate(), 1e-10);
        assertEquals(0.0, simd.getMinWeight(), 1e-10);
        assertEquals(1.0, simd.getMaxWeight(), 1e-10);
        assertEquals("Hebbian-SIMD", simd.getName());
        assertFalse(simd.requiresNormalization());

        var range = simd.getRecommendedLearningRateRange();
        assertEquals(2, range.length);
        assertTrue(range[0] < range[1], "Learning rate range should be valid");
    }

    // Helper methods

    private Pattern createRandomPattern(int size) {
        var data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = Math.random();
        }
        return new DenseVector(data);
    }

    private Pattern createZeroPattern(int size) {
        return new DenseVector(new double[size]);
    }

    private Pattern createOnesPattern(int size) {
        var data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = 1.0;
        }
        return new DenseVector(data);
    }

    private WeightMatrix createRandomWeights(int rows, int cols) {
        var weights = new WeightMatrix(rows, cols);
        for (int j = 0; j < rows; j++) {
            for (int i = 0; i < cols; i++) {
                weights.set(j, i, Math.random());
            }
        }
        return weights;
    }

    private WeightMatrix createZeroWeights(int rows, int cols) {
        return new WeightMatrix(rows, cols);
    }

    private WeightMatrix createOnesWeights(int rows, int cols) {
        var weights = new WeightMatrix(rows, cols);
        for (int j = 0; j < rows; j++) {
            for (int i = 0; i < cols; i++) {
                weights.set(j, i, 1.0);
            }
        }
        return weights;
    }

    private void assertWeightMatricesEqual(WeightMatrix expected, WeightMatrix actual,
                                          double tolerance, String message) {
        assertEquals(expected.getRows(), actual.getRows(), "Rows should match");
        assertEquals(expected.getCols(), actual.getCols(), "Cols should match");

        for (int j = 0; j < expected.getRows(); j++) {
            for (int i = 0; i < expected.getCols(); i++) {
                assertEquals(expected.get(j, i), actual.get(j, i), tolerance,
                    message + " at [" + j + "][" + i + "]");
            }
        }
    }
}
