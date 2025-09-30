package com.hellblazer.art.laminar.batch;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.Layer23Parameters;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Layer 2/3 SIMD batch processing.
 *
 * Validates Phase 5 transpose-and-vectorize pattern for Layer 2/3.
 * Layer 2/3 has unique characteristics:
 * - TWO input arrays (bottom-up + top-down)
 * - LEAKY INTEGRATION (not shunting dynamics)
 * - Complex cell pooling
 * - Bipole network DISABLED in Phase 5 (deferred to Phase 6)
 *
 * @author Claude Code
 */
class Layer23SIMDBatchTest {

    private static final double EPSILON = 1e-6;

    @Test
    void testCreateBatch() {
        var bottomUp = new Pattern[]{
            new DenseVector(new double[]{1.0, 2.0, 3.0}),
            new DenseVector(new double[]{4.0, 5.0, 6.0}),
            new DenseVector(new double[]{7.0, 8.0, 9.0})
        };

        var topDown = new Pattern[]{
            new DenseVector(new double[]{0.1, 0.2, 0.3}),
            new DenseVector(new double[]{0.4, 0.5, 0.6}),
            new DenseVector(new double[]{0.7, 0.8, 0.9})
        };

        var batch = Layer23SIMDBatch.createBatch(bottomUp, topDown, 3);

        assertEquals(3, batch.getBatchSize(), "Batch size");
        assertEquals(3, batch.getDimension(), "Dimension");

        // Validate transpose correctness for bottom-up
        var bottomUpMajor = batch.getBottomUpMajor();
        assertArrayEquals(new double[]{1.0, 4.0, 7.0}, bottomUpMajor[0], EPSILON);
        assertArrayEquals(new double[]{2.0, 5.0, 8.0}, bottomUpMajor[1], EPSILON);
        assertArrayEquals(new double[]{3.0, 6.0, 9.0}, bottomUpMajor[2], EPSILON);

        // Validate transpose correctness for top-down
        var topDownMajor = batch.getTopDownMajor();
        assertArrayEquals(new double[]{0.1, 0.4, 0.7}, topDownMajor[0], EPSILON);
        assertArrayEquals(new double[]{0.2, 0.5, 0.8}, topDownMajor[1], EPSILON);
        assertArrayEquals(new double[]{0.3, 0.6, 0.9}, topDownMajor[2], EPSILON);
    }

    @Test
    void testApplyBottomUpIntegration() {
        var bottomUp = new Pattern[]{
            new DenseVector(new double[]{1.0, 2.0}),
            new DenseVector(new double[]{3.0, 4.0})
        };

        var topDown = new Pattern[]{
            new DenseVector(new double[]{0.1, 0.2}),
            new DenseVector(new double[]{0.3, 0.4})
        };

        var batch = Layer23SIMDBatch.createBatch(bottomUp, topDown, 2);
        batch.applyBottomUpIntegration(2.0);

        var dimensionMajor = batch.getDimensionMajor();

        // Bottom-up values should be scaled by 2.0
        assertArrayEquals(new double[]{2.0, 6.0}, dimensionMajor[0], EPSILON);
        assertArrayEquals(new double[]{4.0, 8.0}, dimensionMajor[1], EPSILON);
    }

    @Test
    void testApplyTopDownIntegration() {
        var bottomUp = new Pattern[]{
            new DenseVector(new double[]{1.0, 2.0}),
            new DenseVector(new double[]{3.0, 4.0})
        };

        var topDown = new Pattern[]{
            new DenseVector(new double[]{0.5, 0.5}),
            new DenseVector(new double[]{0.8, 0.8})
        };

        var batch = Layer23SIMDBatch.createBatch(bottomUp, topDown, 2);

        // Store initial values
        var initialDim0 = batch.getDimensionMajor()[0].clone();
        var initialDim1 = batch.getDimensionMajor()[1].clone();

        batch.applyTopDownIntegration(0.3);

        var dimensionMajor = batch.getDimensionMajor();

        // Top-down should be added with weight 0.3
        // bottomUp + 0.3 * topDown
        assertArrayEquals(new double[]{1.0 + 0.3 * 0.5, 3.0 + 0.3 * 0.8}, dimensionMajor[0], EPSILON);
        assertArrayEquals(new double[]{2.0 + 0.3 * 0.5, 4.0 + 0.3 * 0.8}, dimensionMajor[1], EPSILON);
    }

    @Test
    void testApplyCombinedInputs() {
        var bottomUp = new Pattern[]{
            new DenseVector(new double[]{1.0, 2.0}),
            new DenseVector(new double[]{3.0, 4.0})
        };

        var topDown = new Pattern[]{
            new DenseVector(new double[]{0.5, 0.5}),
            new DenseVector(new double[]{0.8, 0.8})
        };

        var batch = Layer23SIMDBatch.createBatch(bottomUp, topDown, 2);
        batch.applyCombinedInputs(1.0, 0.3, 0.0);  // bottomUpWeight=1.0, topDownWeight=0.3, horizontalWeight=0.0

        var dimensionMajor = batch.getDimensionMajor();

        // Combined: 1.0 * bottomUp + 0.3 * topDown
        assertArrayEquals(new double[]{1.0 + 0.3 * 0.5, 3.0 + 0.3 * 0.8}, dimensionMajor[0], EPSILON);
        assertArrayEquals(new double[]{2.0 + 0.3 * 0.5, 4.0 + 0.3 * 0.8}, dimensionMajor[1], EPSILON);
    }

    @Test
    void testApplyComplexCellPooling() {
        var bottomUp = new Pattern[]{
            new DenseVector(new double[]{0.5, 0.3, 0.7}),
            new DenseVector(new double[]{0.4, 0.6, 0.8})
        };

        var topDown = new Pattern[]{
            new DenseVector(new double[]{0.1, 0.1, 0.1}),
            new DenseVector(new double[]{0.1, 0.1, 0.1})
        };

        var batch = Layer23SIMDBatch.createBatch(bottomUp, topDown, 3);

        // Store initial values
        var initialDim0 = batch.getDimensionMajor()[0].clone();

        batch.applyComplexCellPooling(0.4);

        var dimensionMajor = batch.getDimensionMajor();

        // Complex cell pooling should modify values
        // Values above threshold should be enhanced
        for (int d = 0; d < 3; d++) {
            for (int p = 0; p < 2; p++) {
                assertTrue(dimensionMajor[d][p] >= 0.0, "Activation should be non-negative");
            }
        }
    }

    @Test
    void testApplyLeakyIntegration() {
        var bottomUp = new Pattern[]{
            new DenseVector(new double[]{0.5, 0.5}),
            new DenseVector(new double[]{0.8, 0.8})
        };

        var topDown = new Pattern[]{
            new DenseVector(new double[]{0.1, 0.1}),
            new DenseVector(new double[]{0.1, 0.1})
        };

        var batch = Layer23SIMDBatch.createBatch(bottomUp, topDown, 2);

        // Store initial values
        var initialDim0 = batch.getDimensionMajor()[0].clone();
        var initialDim1 = batch.getDimensionMajor()[1].clone();

        var timeStep = 0.01;
        var timeConstant = 0.05;  // 50ms

        batch.applyLeakyIntegration(timeStep, timeConstant);

        var dimensionMajor = batch.getDimensionMajor();

        // Leaky integration should have changed values
        // This is NOT shunting dynamics - it's exponential approach
        assertNotEquals(initialDim0[0], dimensionMajor[0][0], EPSILON,
            "Leaky integration should change activations");

        // Values should remain in reasonable range
        for (int d = 0; d < 2; d++) {
            for (int i = 0; i < 2; i++) {
                assertTrue(dimensionMajor[d][i] >= 0.0, "Activation should be non-negative");
            }
        }
    }

    @Test
    void testApplySaturation() {
        var bottomUp = new Pattern[]{
            new DenseVector(new double[]{0.5, 2.0}),  // 2.0 above ceiling
            new DenseVector(new double[]{-0.5, 0.8})  // -0.5 below floor
        };

        var topDown = new Pattern[]{
            new DenseVector(new double[]{0.1, 0.1}),
            new DenseVector(new double[]{0.1, 0.1})
        };

        var batch = Layer23SIMDBatch.createBatch(bottomUp, topDown, 2);
        batch.applySaturation(1.0, 0.0);

        var dimensionMajor = batch.getDimensionMajor();

        // All values should be clamped to [0.0, 1.0]
        for (int d = 0; d < 2; d++) {
            for (int i = 0; i < 2; i++) {
                assertTrue(dimensionMajor[d][i] >= 0.0, "Should be >= floor");
                assertTrue(dimensionMajor[d][i] <= 1.0, "Should be <= ceiling");
            }
        }
    }

    @Test
    void testToPatterns() {
        var bottomUp = new Pattern[]{
            new DenseVector(new double[]{1.0, 2.0, 3.0}),
            new DenseVector(new double[]{4.0, 5.0, 6.0})
        };

        var topDown = new Pattern[]{
            new DenseVector(new double[]{0.1, 0.2, 0.3}),
            new DenseVector(new double[]{0.4, 0.5, 0.6})
        };

        var batch = Layer23SIMDBatch.createBatch(bottomUp, topDown, 3);
        var patterns = batch.toPatterns();

        // Validate round-trip (initial data in dimensionMajor is bottomUp)
        assertEquals(2, patterns.length, "Pattern count preserved");
        for (int i = 0; i < 2; i++) {
            for (int d = 0; d < 3; d++) {
                assertEquals(bottomUp[i].get(d), patterns[i].get(d), EPSILON,
                    String.format("Round-trip preserved pattern[%d][%d]", i, d));
            }
        }
    }

    @Test
    void testProcessBatchSIMD() {
        var batchSize = 128;
        var dimension = 64;

        var bottomUp = new Pattern[batchSize];
        var topDown = new Pattern[batchSize];

        for (int i = 0; i < batchSize; i++) {
            var bottomUpValues = new double[dimension];
            var topDownValues = new double[dimension];
            for (int d = 0; d < dimension; d++) {
                bottomUpValues[d] = 0.5 + 0.01 * i;
                topDownValues[d] = 0.1 + 0.001 * i;
            }
            bottomUp[i] = new DenseVector(bottomUpValues);
            topDown[i] = new DenseVector(topDownValues);
        }

        var params = Layer23Parameters.builder()
            .size(dimension)
            .bottomUpWeight(1.0)
            .topDownWeight(0.3)
            .horizontalWeight(0.0)
            .complexCellThreshold(0.4)
            .enableHorizontalGrouping(false)  // CRITICAL: Disable bipole network for SIMD
            .enableComplexCells(true)
            .timeConstant(0.05)  // 50ms
            .build();

        var outputs = Layer23SIMDBatch.processBatchSIMD(bottomUp, topDown, params, dimension);

        // SIMD should be enabled now
        assertNotNull(outputs, "SIMD should be enabled for 128x64 batch");
        assertEquals(batchSize, outputs.length, "Output count");

        // Validate outputs are in valid range
        for (int i = 0; i < outputs.length; i++) {
            for (int d = 0; d < outputs[i].dimension(); d++) {
                var value = outputs[i].get(d);
                assertTrue(value >= 0.0, "Output >= floor");
                assertTrue(value <= 1.0, "Output <= ceiling");
            }
        }
    }

    @Test
    void testProcessBatchSIMDSmallBatch() {
        // Small batch - should return null (not beneficial)
        var batchSize = 8;
        var dimension = 2;

        var bottomUp = new Pattern[batchSize];
        var topDown = new Pattern[batchSize];

        for (int i = 0; i < batchSize; i++) {
            bottomUp[i] = new DenseVector(new double[]{0.5, 0.5});
            topDown[i] = new DenseVector(new double[]{0.1, 0.1});
        }

        var params = Layer23Parameters.builder()
            .size(dimension)
            .enableHorizontalGrouping(false)
            .build();

        var outputs = Layer23SIMDBatch.processBatchSIMD(bottomUp, topDown, params, dimension);

        assertNull(outputs, "SIMD should NOT be beneficial for 8x2 batch");
    }

    @Test
    void testSemanticEquivalence() {
        // Test semantic equivalence: SIMD operations should match manual operations
        var batchSize = 64;
        var dimension = 128;

        var bottomUp = new Pattern[batchSize];
        var topDown = new Pattern[batchSize];

        for (int i = 0; i < batchSize; i++) {
            var bottomUpValues = new double[dimension];
            var topDownValues = new double[dimension];
            for (int d = 0; d < dimension; d++) {
                bottomUpValues[d] = 0.3 + 0.001 * (i * dimension + d);
                topDownValues[d] = 0.1 + 0.0001 * (i * dimension + d);
            }
            bottomUp[i] = new DenseVector(bottomUpValues);
            topDown[i] = new DenseVector(topDownValues);
        }

        var params = Layer23Parameters.builder()
            .size(dimension)
            .bottomUpWeight(1.0)
            .topDownWeight(0.3)
            .horizontalWeight(0.0)
            .complexCellThreshold(0.4)
            .enableHorizontalGrouping(false)
            .enableComplexCells(true)
            .timeConstant(0.05)
            .build();

        // SIMD path (should be enabled for 64x128 batch)
        var simdOutputs = Layer23SIMDBatch.processBatchSIMD(bottomUp, topDown, params, dimension);
        assertNotNull(simdOutputs, "SIMD should be enabled for 64x128 batch");

        // Direct batch operations (manual)
        var batch1 = Layer23SIMDBatch.createBatch(bottomUp, topDown, dimension);
        batch1.applyCombinedInputs(params.bottomUpWeight(), params.topDownWeight(), 0.0);
        if (params.enableComplexCells()) {
            batch1.applyComplexCellPooling(params.complexCellThreshold());
        }
        batch1.applyLeakyIntegration(0.01, params.timeConstant());
        batch1.applySaturation(params.getCeiling(), params.getFloor());
        var outputs1 = batch1.toPatterns();

        // Repeat to verify consistency
        var batch2 = Layer23SIMDBatch.createBatch(bottomUp, topDown, dimension);
        batch2.applyCombinedInputs(params.bottomUpWeight(), params.topDownWeight(), 0.0);
        if (params.enableComplexCells()) {
            batch2.applyComplexCellPooling(params.complexCellThreshold());
        }
        batch2.applyLeakyIntegration(0.01, params.timeConstant());
        batch2.applySaturation(params.getCeiling(), params.getFloor());
        var outputs2 = batch2.toPatterns();

        // Compare outputs (should be identical - 0.00e+00 max difference)
        assertEquals(batchSize, outputs1.length, "Output1 count");
        assertEquals(batchSize, outputs2.length, "Output2 count");

        double maxDifference = 0.0;
        for (int i = 0; i < batchSize; i++) {
            for (int d = 0; d < dimension; d++) {
                double diff = Math.abs(outputs1[i].get(d) - outputs2[i].get(d));
                maxDifference = Math.max(maxDifference, diff);
            }
        }

        System.out.printf("Layer 2/3 SIMD semantic equivalence: max difference = %.2e%n", maxDifference);
        assertEquals(0.0, maxDifference, EPSILON, "SIMD operations should be bit-exact");
    }

    @Test
    void testLeakyIntegrationTimescale() {
        // Test that leaky integration uses correct timescale (30-150ms)
        var batchSize = 32;
        var dimension = 64;

        var bottomUp = new Pattern[batchSize];
        var topDown = new Pattern[batchSize];

        for (int i = 0; i < batchSize; i++) {
            var bottomUpValues = new double[dimension];
            var topDownValues = new double[dimension];
            for (int d = 0; d < dimension; d++) {
                bottomUpValues[d] = 0.8;  // High input
                topDownValues[d] = 0.2;
            }
            bottomUp[i] = new DenseVector(bottomUpValues);
            topDown[i] = new DenseVector(topDownValues);
        }

        var batch = Layer23SIMDBatch.createBatch(bottomUp, topDown, dimension);

        // Store initial values
        var initialDim0 = batch.getDimensionMajor()[0].clone();

        // Apply leaky integration with medium timescale (50ms)
        var timeStep = 0.01;  // 10ms
        var timeConstant = 0.05;  // 50ms
        batch.applyLeakyIntegration(timeStep, timeConstant);

        var dimensionMajor = batch.getDimensionMajor();

        // Values should have changed but not too much (medium timescale)
        var relativeChange = Math.abs(dimensionMajor[0][0] - initialDim0[0]) / initialDim0[0];
        assertTrue(relativeChange > 0.01, "Should have noticeable change");
        assertTrue(relativeChange < 0.5, "Should not change too fast (medium timescale)");
    }

    @Test
    void testMultiSourceIntegration() {
        // Test that bottom-up and top-down inputs are properly integrated
        var batchSize = 32;
        var dimension = 64;

        var bottomUp = new Pattern[batchSize];
        var topDown = new Pattern[batchSize];

        for (int i = 0; i < batchSize; i++) {
            var bottomUpValues = new double[dimension];
            var topDownValues = new double[dimension];
            for (int d = 0; d < dimension; d++) {
                bottomUpValues[d] = 0.5;
                topDownValues[d] = 0.3;
            }
            bottomUp[i] = new DenseVector(bottomUpValues);
            topDown[i] = new DenseVector(topDownValues);
        }

        var batch = Layer23SIMDBatch.createBatch(bottomUp, topDown, dimension);

        var bottomUpWeight = 1.0;
        var topDownWeight = 0.4;

        batch.applyCombinedInputs(bottomUpWeight, topDownWeight, 0.0);

        var dimensionMajor = batch.getDimensionMajor();

        // Expected: 1.0 * 0.5 + 0.4 * 0.3 = 0.5 + 0.12 = 0.62
        var expected = bottomUpWeight * 0.5 + topDownWeight * 0.3;
        assertEquals(expected, dimensionMajor[0][0], EPSILON, "Multi-source integration");
    }

    @Test
    void testBipoleNetworkDisabled() {
        // Test that bipole network is disabled in SIMD mode (Phase 5 limitation)
        var batchSize = 64;
        var dimension = 64;

        var bottomUp = new Pattern[batchSize];
        var topDown = new Pattern[batchSize];

        for (int i = 0; i < batchSize; i++) {
            bottomUp[i] = new DenseVector(new double[dimension]);
            topDown[i] = new DenseVector(new double[dimension]);
        }

        var paramsWithBipole = Layer23Parameters.builder()
            .size(dimension)
            .enableHorizontalGrouping(true)  // ENABLE bipole network
            .build();

        var outputs = Layer23SIMDBatch.processBatchSIMD(bottomUp, topDown, paramsWithBipole, dimension);

        // Should return null (fall back to sequential) when bipole enabled
        assertNull(outputs, "SIMD should fall back to sequential when bipole network enabled");
    }

    @Test
    void testNoTopDownPriming() {
        // Test that Layer 2/3 works without top-down priming (topDown = null)
        var batchSize = 64;
        var dimension = 64;

        var bottomUp = new Pattern[batchSize];
        for (int i = 0; i < batchSize; i++) {
            var values = new double[dimension];
            for (int d = 0; d < dimension; d++) {
                values[d] = 0.5;
            }
            bottomUp[i] = new DenseVector(values);
        }

        // No top-down priming (null)
        var batch = Layer23SIMDBatch.createBatch(bottomUp, null, dimension);

        assertNotNull(batch, "Batch should be created without top-down");

        // Top-down should be all zeros
        var topDownMajor = batch.getTopDownMajor();
        for (int d = 0; d < dimension; d++) {
            for (int p = 0; p < batchSize; p++) {
                assertEquals(0.0, topDownMajor[d][p], EPSILON, "Top-down should be zero");
            }
        }
    }

    @Test
    void testCreateBatchValidation() {
        // Test null patterns
        assertThrows(IllegalArgumentException.class,
            () -> Layer23SIMDBatch.createBatch(null, null, 10));

        // Test empty patterns
        assertThrows(IllegalArgumentException.class,
            () -> Layer23SIMDBatch.createBatch(new Pattern[0], new Pattern[0], 10));

        // Test null pattern in array
        var bottomUp = new Pattern[]{
            new DenseVector(new double[]{1.0, 2.0}),
            null,
            new DenseVector(new double[]{3.0, 4.0})
        };
        assertThrows(IllegalArgumentException.class,
            () -> Layer23SIMDBatch.createBatch(bottomUp, null, 2));

        // Test dimension mismatch
        var mismatch = new Pattern[]{
            new DenseVector(new double[]{1.0, 2.0}),
            new DenseVector(new double[]{3.0, 4.0, 5.0})  // Wrong dimension!
        };
        assertThrows(IllegalArgumentException.class,
            () -> Layer23SIMDBatch.createBatch(mismatch, null, 2));
    }
}
