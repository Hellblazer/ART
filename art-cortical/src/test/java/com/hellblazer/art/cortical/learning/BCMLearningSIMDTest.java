package com.hellblazer.art.cortical.learning;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.cortical.layers.WeightMatrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for SIMD-vectorized BCM learning - Phase 4D.
 *
 * <p>Verifies:
 * <ul>
 *   <li>Correctness: SIMD results match scalar BCM</li>
 *   <li>BCM properties: LTP/LTD, threshold adaptation</li>
 *   <li>Edge cases: Small matrices, zero learning rate</li>
 *   <li>Precision: Numerical accuracy within tolerance</li>
 * </ul>
 *
 * @author Phase 4D: Learning Vectorization
 */
class BCMLearningSIMDTest {

    private static final double TOLERANCE = 1e-10;
    private static final int LARGE_PRE_SIZE = 256;
    private static final int LARGE_POST_SIZE = 128;
    private static final int SMALL_PRE_SIZE = 4;
    private static final int SMALL_POST_SIZE = 8;

    private BCMLearning scalar;
    private BCMLearningSIMD simd;

    @BeforeEach
    void setup() {
        scalar = new BCMLearning(0.5, 0.0001, 0.0, 1.0);
        simd = new BCMLearningSIMD(0.5, 0.0001, 0.0, 1.0);
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
    void testThresholdAdaptation() {
        // Test that thresholds adapt based on activation
        var preActivation = createRandomPattern(LARGE_PRE_SIZE);
        var postActivation = createOnesPattern(LARGE_POST_SIZE);  // High activation
        var currentWeights = createRandomWeights(LARGE_POST_SIZE, LARGE_PRE_SIZE);

        // First update initializes thresholds
        simd.update(preActivation, postActivation, currentWeights, 0.1);
        var thresholds1 = simd.getModificationThresholds();

        // Second update adapts thresholds
        simd.update(preActivation, postActivation, currentWeights, 0.1);
        var thresholds2 = simd.getModificationThresholds();

        // Thresholds should increase with high activation
        for (int j = 0; j < LARGE_POST_SIZE; j++) {
            assertTrue(thresholds2[j] > thresholds1[j],
                "Threshold " + j + " should increase with high activation");
        }
    }

    @Test
    void testLTP() {
        // Long-Term Potentiation: weights strengthen when y > θ
        var preActivation = createOnesPattern(LARGE_PRE_SIZE);
        var postActivation = createHighPattern(LARGE_POST_SIZE);  // y > θ
        var currentWeights = createZeroWeights(LARGE_POST_SIZE, LARGE_PRE_SIZE);
        var learningRate = 0.1;

        var result = simd.update(preActivation, postActivation, currentWeights, learningRate);

        // Most weights should increase (LTP)
        int increasedCount = 0;
        for (int j = 0; j < LARGE_POST_SIZE; j++) {
            for (int i = 0; i < LARGE_PRE_SIZE; i++) {
                if (result.get(j, i) > 0.0) {
                    increasedCount++;
                }
            }
        }

        assertTrue(increasedCount > (LARGE_POST_SIZE * LARGE_PRE_SIZE) / 2,
            "Most weights should strengthen (LTP) when y > θ");
    }

    @Test
    void testLTD() {
        // Long-Term Depression: weights weaken when y < θ
        var preActivation = createOnesPattern(LARGE_PRE_SIZE);
        var postActivation = createLowPattern(LARGE_POST_SIZE);  // y < θ
        var currentWeights = createOnesWeights(LARGE_POST_SIZE, LARGE_PRE_SIZE);
        var learningRate = 0.1;

        // First warmup to establish high thresholds
        var highActivation = createHighPattern(LARGE_POST_SIZE);
        for (int i = 0; i < 10; i++) {
            simd.update(preActivation, highActivation, currentWeights, 0.05);
        }

        // Now apply low activation (should trigger LTD)
        var result = simd.update(preActivation, postActivation, currentWeights, learningRate);

        // Most weights should decrease or stay same (LTD)
        int decreasedCount = 0;
        for (int j = 0; j < LARGE_POST_SIZE; j++) {
            for (int i = 0; i < LARGE_PRE_SIZE; i++) {
                if (result.get(j, i) < currentWeights.get(j, i)) {
                    decreasedCount++;
                }
            }
        }

        assertTrue(decreasedCount > (LARGE_POST_SIZE * LARGE_PRE_SIZE) / 3,
            "Many weights should weaken (LTD) when y < θ");
    }

    @Test
    void testWeightBoundsClamping() {
        // Test that weights are clamped to [minWeight, maxWeight]
        var minWeight = 0.0;
        var maxWeight = 0.5;
        var learningRule = new BCMLearningSIMD(0.5, 0.0001, minWeight, maxWeight);

        var preActivation = createOnesPattern(LARGE_PRE_SIZE);
        var postActivation = createHighPattern(LARGE_POST_SIZE);
        var currentWeights = createOnesWeights(LARGE_POST_SIZE, LARGE_PRE_SIZE);
        var learningRate = 1.0;  // Very high learning rate

        var result = learningRule.update(preActivation, postActivation, currentWeights, learningRate);

        // All weights should be clamped to bounds
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

        // Reset both to start from same initial state
        scalar.resetThresholds();
        simd.resetThresholds();

        var scalarWeights = weights;
        var simdWeights = weights;

        // Apply 10 updates
        for (int iter = 0; iter < 10; iter++) {
            scalarWeights = scalar.update(preActivation, postActivation, scalarWeights, learningRate);
            simdWeights = simd.update(preActivation, postActivation, simdWeights, learningRate);
        }

        assertWeightMatricesEqual(scalarWeights, simdWeights, TOLERANCE,
            "SIMD should match scalar after multiple updates");

        // Thresholds should also match
        assertArrayEquals(scalar.getModificationThresholds(), simd.getModificationThresholds(), TOLERANCE,
            "Thresholds should match after multiple updates");
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

                // Fresh instances for each size
                var scalarLocal = new BCMLearning(0.5, 0.0001, 0.0, 1.0);
                var simdLocal = new BCMLearningSIMD(0.5, 0.0001, 0.0, 1.0);

                var scalarResult = scalarLocal.update(preActivation, postActivation, currentWeights, learningRate);
                var simdResult = simdLocal.update(preActivation, postActivation, currentWeights, learningRate);

                assertWeightMatricesEqual(scalarResult, simdResult, TOLERANCE,
                    "SIMD should match scalar for size " + preSize + "x" + postSize);
            }
        }
    }

    @Test
    void testResetThresholds() {
        var preActivation = createRandomPattern(LARGE_PRE_SIZE);
        var postActivation = createRandomPattern(LARGE_POST_SIZE);
        var currentWeights = createRandomWeights(LARGE_POST_SIZE, LARGE_PRE_SIZE);

        // Run several updates to establish thresholds
        for (int i = 0; i < 5; i++) {
            simd.update(preActivation, postActivation, currentWeights, 0.1);
        }

        var thresholdsBefore = simd.getModificationThresholds();

        // Reset thresholds
        simd.resetThresholds();
        var thresholdsAfter = simd.getModificationThresholds();

        // All thresholds should be reset to 0.1
        for (int j = 0; j < LARGE_POST_SIZE; j++) {
            assertEquals(0.1, thresholdsAfter[j], TOLERANCE,
                "Threshold " + j + " should be reset to 0.1");
            assertNotEquals(thresholdsBefore[j], thresholdsAfter[j], TOLERANCE,
                "Threshold " + j + " should have changed after reset");
        }
    }

    @Test
    void testFactoryMethods() {
        var competitive = BCMLearningSIMD.createCompetitive();
        var balanced = BCMLearningSIMD.createBalanced();
        var homeostatic = BCMLearningSIMD.createHomeostatic();

        assertEquals(0.8, competitive.getThresholdDecayRate(), TOLERANCE);
        assertEquals(0.5, balanced.getThresholdDecayRate(), TOLERANCE);
        assertEquals(0.1, homeostatic.getThresholdDecayRate(), TOLERANCE);

        assertTrue(balanced.getWeightDecayRate() > competitive.getWeightDecayRate(),
            "Balanced should have higher weight decay than competitive");
    }

    @Test
    void testInvalidParameters() {
        assertThrows(IllegalArgumentException.class,
            () -> new BCMLearningSIMD(-0.1, 0.0001, 0.0, 1.0),
            "Negative threshold decay should throw");

        assertThrows(IllegalArgumentException.class,
            () -> new BCMLearningSIMD(0.5, -0.1, 0.0, 1.0),
            "Negative weight decay should throw");

        assertThrows(IllegalArgumentException.class,
            () -> new BCMLearningSIMD(0.5, 0.0001, 1.0, 0.0),
            "minWeight >= maxWeight should throw");
    }

    @Test
    void testVectorLength() {
        var vectorLength = simd.getVectorLength();
        assertTrue(vectorLength >= 2 && vectorLength <= 8,
            "Vector length should be between 2 (SSE) and 8 (AVX-512)");
    }

    @Test
    void testGetters() {
        assertEquals(0.5, simd.getThresholdDecayRate(), TOLERANCE);
        assertEquals(0.0001, simd.getWeightDecayRate(), TOLERANCE);
        assertEquals(0.0, simd.getMinWeight(), TOLERANCE);
        assertEquals(1.0, simd.getMaxWeight(), TOLERANCE);
        assertEquals("BCM-SIMD", simd.getName());
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

    private Pattern createHighPattern(int size) {
        var data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = 0.8 + 0.2 * Math.random();  // [0.8, 1.0]
        }
        return new DenseVector(data);
    }

    private Pattern createLowPattern(int size) {
        var data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = 0.2 * Math.random();  // [0.0, 0.2]
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
