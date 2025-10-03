package com.hellblazer.art.cortical.learning;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.cortical.layers.WeightMatrix;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for pooled Hebbian learning - Phase 4E.
 *
 * <p>Verifies:
 * <ul>
 *   <li>Correctness: Pooled matches standard Hebbian</li>
 *   <li>Memory efficiency: Pool reuse works correctly</li>
 *   <li>Resource management: Proper cleanup</li>
 * </ul>
 *
 * @author Phase 4E: Memory Optimization
 */
class HebbianLearningPooledTest {

    private static final double TOLERANCE = 1e-10;
    private static final int POST_SIZE = 128;
    private static final int PRE_SIZE = 256;

    private HebbianLearning standard;
    private HebbianLearningPooled pooled;

    @BeforeEach
    void setup() {
        standard = new HebbianLearning(0.0001, 0.0, 1.0);
        pooled = new HebbianLearningPooled(0.0001, 0.0, 1.0, POST_SIZE, PRE_SIZE);
    }

    @AfterEach
    void tearDown() {
        if (pooled != null) {
            pooled.close();
        }
    }

    @Test
    void testPooledMatchesStandard() {
        var preActivation = createRandomPattern(PRE_SIZE);
        var postActivation = createRandomPattern(POST_SIZE);
        var currentWeights = createRandomWeights(POST_SIZE, PRE_SIZE);
        var learningRate = 0.01;

        var standardResult = standard.update(preActivation, postActivation, currentWeights, learningRate);
        var pooledResult = pooled.update(preActivation, postActivation, currentWeights, learningRate);

        assertWeightMatricesEqual(standardResult, pooledResult, TOLERANCE,
            "Pooled results should match standard");
    }

    @Test
    void testUpdatePooledReusesMatrix() {
        var preActivation = createRandomPattern(PRE_SIZE);
        var postActivation = createRandomPattern(POST_SIZE);
        var currentWeights = createRandomWeights(POST_SIZE, PRE_SIZE);
        var learningRate = 0.01;

        // First update
        var weights1 = pooled.updatePooled(preActivation, postActivation, currentWeights, learningRate);
        pooled.returnToPool(weights1);

        // Second update should reuse same matrix
        var weights2 = pooled.updatePooled(preActivation, postActivation, currentWeights, learningRate);

        assertSame(weights1, weights2, "Should reuse pooled matrix");

        pooled.returnToPool(weights2);
    }

    @Test
    void testUpdatePooledCorrectness() {
        var preActivation = createRandomPattern(PRE_SIZE);
        var postActivation = createRandomPattern(POST_SIZE);
        var currentWeights = createRandomWeights(POST_SIZE, PRE_SIZE);
        var learningRate = 0.01;

        var standardResult = standard.update(preActivation, postActivation, currentWeights, learningRate);
        var pooledResult = pooled.updatePooled(preActivation, postActivation, currentWeights, learningRate);

        assertWeightMatricesEqual(standardResult, pooledResult, TOLERANCE,
            "Pooled update should match standard");

        pooled.returnToPool(pooledResult);
    }

    @Test
    void testPrewarm() {
        pooled.prewarm(5);

        var pool = pooled.getPool();
        assertEquals(5, pool.getPoolSize(), "Should have 5 prewarmed matrices");
    }

    @Test
    void testAutoCloseablePattern() {
        Pattern preActivation = createRandomPattern(PRE_SIZE);
        Pattern postActivation = createRandomPattern(POST_SIZE);
        WeightMatrix currentWeights = createRandomWeights(POST_SIZE, PRE_SIZE);

        try (var learning = new HebbianLearningPooled(0.0001, 0.0, 1.0, POST_SIZE, PRE_SIZE)) {
            learning.prewarm(3);
            assertEquals(3, learning.getPool().getPoolSize());

            var result = learning.update(preActivation, postActivation, currentWeights, 0.01);
            assertNotNull(result);
        }  // Pool cleared automatically

        // Can't directly verify pool is cleared since it's closed,
        // but no exception should be thrown
    }

    @Test
    void testMultipleUpdatesWithPooling() {
        var preActivation = createRandomPattern(PRE_SIZE);
        var postActivation = createRandomPattern(POST_SIZE);
        var weights = createRandomWeights(POST_SIZE, PRE_SIZE);

        for (int i = 0; i < 10; i++) {
            var updated = pooled.updatePooled(preActivation, postActivation, weights, 0.01);
            weights = updated;  // Use result as input for next iteration
        }

        // Final weights should still be valid
        assertNotNull(weights);

        // Return final weights
        pooled.returnToPool(weights);

        // Pool should have 1 matrix
        assertEquals(1, pooled.getPool().getPoolSize());
    }

    @Test
    void testPoolSizeLimit() {
        var smallPooled = new HebbianLearningPooled(0.0001, 0.0, 1.0, POST_SIZE, PRE_SIZE, 2);

        var preActivation = createRandomPattern(PRE_SIZE);
        var postActivation = createRandomPattern(POST_SIZE);
        var currentWeights = createRandomWeights(POST_SIZE, PRE_SIZE);

        // Create more matrices than pool size
        var w1 = smallPooled.updatePooled(preActivation, postActivation, currentWeights, 0.01);
        var w2 = smallPooled.updatePooled(preActivation, postActivation, currentWeights, 0.01);
        var w3 = smallPooled.updatePooled(preActivation, postActivation, currentWeights, 0.01);

        // Return all
        smallPooled.returnToPool(w1);
        smallPooled.returnToPool(w2);
        smallPooled.returnToPool(w3);

        // Pool should not exceed max size (2)
        assertTrue(smallPooled.getPool().getPoolSize() <= 2);

        smallPooled.close();
    }

    @Test
    void testGetters() {
        assertEquals("Hebbian-Pooled", pooled.getName());
        assertFalse(pooled.requiresNormalization());

        var range = pooled.getRecommendedLearningRateRange();
        assertEquals(2, range.length);
        assertTrue(range[0] < range[1]);
    }

    @Test
    void testPoolStats() {
        pooled.prewarm(3);

        var stats = pooled.getPoolStats();

        assertTrue(stats.contains("128x256"));
        assertTrue(stats.contains("poolSize=3"));
    }

    @Test
    void testToString() {
        var str = pooled.toString();

        assertTrue(str.contains("HebbianLearningPooled"), "Should contain class name");
        assertTrue(str.contains("decay="), "Should contain decay rate");
        assertTrue(str.contains("pool="), "Should contain pool info");
    }

    @Test
    void testExceptionHandlingInUpdatePooled() {
        // Create pooled learning with invalid dimensions to trigger error
        var preActivation = createRandomPattern(PRE_SIZE);
        var postActivation = createRandomPattern(POST_SIZE);
        var wrongWeights = new WeightMatrix(64, 64);  // Wrong size

        assertThrows(IllegalArgumentException.class, () ->
            pooled.updatePooled(preActivation, postActivation, wrongWeights, 0.01)
        );

        // Matrix should be returned to pool on error (pool size increases by 1)
        assertEquals(1, pooled.getPool().getPoolSize(),
            "Matrix should be returned to pool after error");
    }

    // Helper methods

    private Pattern createRandomPattern(int size) {
        var data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = Math.random();
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
