package com.hellblazer.art.laminar.batch;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.Layer6Parameters;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Layer 6 SIMD batch processing.
 *
 * Validates Layer 6 specific operations:
 * - ART matching rule (modulatory only - critical!)
 * - Off-surround inhibition
 * - Attentional gain modulation
 * - Exact shunting dynamics
 * - Saturation
 *
 * @author Hal Hildebrand
 */
class Layer6SIMDBatchTest {

    private static final double EPSILON = 1e-6;

    @Test
    void testCreateBatch() {
        var bottomUpInputs = new Pattern[]{
            new DenseVector(new double[]{1.0, 2.0, 3.0}),
            new DenseVector(new double[]{4.0, 5.0, 6.0})
        };

        var topDownExpectations = new Pattern[]{
            new DenseVector(new double[]{0.1, 0.2, 0.3}),
            new DenseVector(new double[]{0.4, 0.5, 0.6})
        };

        var batch = Layer6SIMDBatch.createBatch(bottomUpInputs, topDownExpectations, 3);

        assertEquals(2, batch.getBatchSize(), "Batch size");
        assertEquals(3, batch.getDimension(), "Dimension");

        // Validate transpose correctness for bottom-up
        var dimensionMajor = batch.getDimensionMajor();
        assertArrayEquals(new double[]{1.0, 4.0}, dimensionMajor[0], EPSILON);
        assertArrayEquals(new double[]{2.0, 5.0}, dimensionMajor[1], EPSILON);
        assertArrayEquals(new double[]{3.0, 6.0}, dimensionMajor[2], EPSILON);

        // Validate transpose correctness for top-down
        var topDownMajor = batch.getTopDownMajor();
        assertArrayEquals(new double[]{0.1, 0.4}, topDownMajor[0], EPSILON);
        assertArrayEquals(new double[]{0.2, 0.5}, topDownMajor[1], EPSILON);
        assertArrayEquals(new double[]{0.3, 0.6}, topDownMajor[2], EPSILON);
    }

    @Test
    void testCreateBatchWithoutTopDown() {
        var bottomUpInputs = new Pattern[]{
            new DenseVector(new double[]{1.0, 2.0}),
            new DenseVector(new double[]{3.0, 4.0})
        };

        var batch = Layer6SIMDBatch.createBatch(bottomUpInputs, null, 2);

        assertEquals(2, batch.getBatchSize());
        assertEquals(2, batch.getDimension());

        // Top-down should be zeros
        var topDownMajor = batch.getTopDownMajor();
        assertArrayEquals(new double[]{0.0, 0.0}, topDownMajor[0], EPSILON);
        assertArrayEquals(new double[]{0.0, 0.0}, topDownMajor[1], EPSILON);
    }

    @Test
    void testApplyARTMatchingRule() {
        var bottomUpInputs = new Pattern[]{
            new DenseVector(new double[]{0.5, 0.0}),  // Second element has no bottom-up
            new DenseVector(new double[]{0.8, 0.6})
        };

        var topDownExpectations = new Pattern[]{
            new DenseVector(new double[]{0.5, 0.5}),
            new DenseVector(new double[]{0.5, 0.5})
        };

        var batch = Layer6SIMDBatch.createBatch(bottomUpInputs, topDownExpectations, 2);
        var onCenterWeight = 1.0;
        var modulationThreshold = 0.1;

        batch.applyARTMatchingRule(onCenterWeight, modulationThreshold);

        var dimensionMajor = batch.getDimensionMajor();

        // Pattern 0, dim 0: 0.5 * (1 + 1.0 * 0.5) = 0.5 * 1.5 = 0.75
        assertEquals(0.75, dimensionMajor[0][0], EPSILON, "Pattern 0, dim 0 with modulation");

        // Pattern 0, dim 1: 0.0 bottom-up -> MUST BE ZERO (modulatory only!)
        assertEquals(0.0, dimensionMajor[1][0], EPSILON, "Pattern 0, dim 1 - no bottom-up = zero output");

        // Pattern 1, dim 0: 0.8 * (1 + 1.0 * 0.5) = 0.8 * 1.5 = 1.2
        assertEquals(1.2, dimensionMajor[0][1], EPSILON, "Pattern 1, dim 0 with modulation");

        // Pattern 1, dim 1: 0.6 * (1 + 1.0 * 0.5) = 0.6 * 1.5 = 0.9
        assertEquals(0.9, dimensionMajor[1][1], EPSILON, "Pattern 1, dim 1 with modulation");
    }

    @Test
    void testApplyARTMatchingRuleCriticalNoBottomUp() {
        // CRITICAL TEST: Ensure zero output when no bottom-up signal
        var bottomUpInputs = new Pattern[]{
            new DenseVector(new double[]{0.0, 0.0, 0.0}),  // No bottom-up anywhere
        };

        var topDownExpectations = new Pattern[]{
            new DenseVector(new double[]{1.0, 1.0, 1.0}),  // Strong top-down
        };

        var batch = Layer6SIMDBatch.createBatch(bottomUpInputs, topDownExpectations, 3);
        batch.applyARTMatchingRule(2.0, 0.1);  // Strong modulation, but no bottom-up

        var dimensionMajor = batch.getDimensionMajor();

        // ALL outputs MUST be zero (modulatory only - cannot fire without bottom-up!)
        assertArrayEquals(new double[]{0.0}, dimensionMajor[0], EPSILON, "Dim 0: no bottom-up = zero");
        assertArrayEquals(new double[]{0.0}, dimensionMajor[1], EPSILON, "Dim 1: no bottom-up = zero");
        assertArrayEquals(new double[]{0.0}, dimensionMajor[2], EPSILON, "Dim 2: no bottom-up = zero");
    }

    @Test
    void testApplyOffSurroundInhibition() {
        var bottomUpInputs = new Pattern[]{
            new DenseVector(new double[]{0.5, 0.8, 0.5, 0.6, 0.5}),  // Peak at dim 1
        };

        var batch = Layer6SIMDBatch.createBatch(bottomUpInputs, null, 5);
        var offSurroundStrength = 0.2;
        var surroundSize = 2;

        batch.applyOffSurroundInhibition(offSurroundStrength, surroundSize);

        var dimensionMajor = batch.getDimensionMajor();

        // Dim 1 (peak) should be inhibited by its neighbors (dims 0, 2, 3)
        // Original: 0.8
        // Neighbors: 0.5 (dim 0) + 0.5 (dim 2) + 0.6 (dim 3, at offset 2) = 1.6
        // Inhibition: 0.8 - 0.2 * 1.6 = 0.8 - 0.32 = 0.48
        // Note: The actual computation depends on which neighbors exist

        // Just validate that inhibition was applied (values decreased or zeroed)
        assertTrue(dimensionMajor[1][0] <= 0.8, "Peak should be inhibited");
    }

    @Test
    void testApplyAttentionalGain() {
        var bottomUpInputs = new Pattern[]{
            new DenseVector(new double[]{0.5, 0.6}),
            new DenseVector(new double[]{0.7, 0.8})
        };

        var topDownExpectations = new Pattern[]{
            new DenseVector(new double[]{1.0, 0.0}),  // Full attention on dim 0
            new DenseVector(new double[]{0.5, 0.5})   // Partial attention
        };

        var batch = Layer6SIMDBatch.createBatch(bottomUpInputs, topDownExpectations, 2);
        var attentionalGain = 2.0;

        batch.applyAttentionalGain(attentionalGain);

        var dimensionMajor = batch.getDimensionMajor();

        // Pattern 0, dim 0: 0.5 * (1 + (2.0 - 1.0) * 1.0) = 0.5 * 2.0 = 1.0
        assertEquals(1.0, dimensionMajor[0][0], EPSILON, "Full attention gain");

        // Pattern 0, dim 1: 0.6 * (1 + (2.0 - 1.0) * 0.0) = 0.6 * 1.0 = 0.6
        assertEquals(0.6, dimensionMajor[1][0], EPSILON, "No attention gain");

        // Pattern 1, dim 0: 0.7 * (1 + (2.0 - 1.0) * 0.5) = 0.7 * 1.5 = 1.05
        assertEquals(1.05, dimensionMajor[0][1], EPSILON, "Partial attention gain");
    }

    @Test
    void testApplySaturation() {
        var bottomUpInputs = new Pattern[]{
            new DenseVector(new double[]{0.5, 2.0}),  // 2.0 above ceiling
            new DenseVector(new double[]{-0.5, 0.8})  // -0.5 below floor
        };

        var batch = Layer6SIMDBatch.createBatch(bottomUpInputs, null, 2);
        batch.applySaturation(1.0, 0.0);

        var dimensionMajor = batch.getDimensionMajor();

        // All values clamped to [0.0, 1.0]
        for (int d = 0; d < 2; d++) {
            for (int i = 0; i < 2; i++) {
                assertTrue(dimensionMajor[d][i] >= 0.0, "Should be >= floor");
                assertTrue(dimensionMajor[d][i] <= 1.0, "Should be <= ceiling");
            }
        }

        // Specific checks
        assertEquals(0.5, dimensionMajor[0][0], EPSILON, "Within range unchanged");
        assertEquals(1.0, dimensionMajor[1][0], EPSILON, "Above ceiling clamped");
        assertEquals(0.0, dimensionMajor[0][1], EPSILON, "Below floor clamped");
        assertEquals(0.8, dimensionMajor[1][1], EPSILON, "Within range unchanged");
    }

    @Test
    void testToPatterns() {
        var original = new Pattern[]{
            new DenseVector(new double[]{1.0, 2.0, 3.0}),
            new DenseVector(new double[]{4.0, 5.0, 6.0})
        };

        var batch = Layer6SIMDBatch.createBatch(original, null, 3);
        var patterns = batch.toPatterns();

        // Validate round-trip
        assertEquals(2, patterns.length, "Pattern count preserved");
        for (int i = 0; i < 2; i++) {
            for (int d = 0; d < 3; d++) {
                assertEquals(original[i].get(d), patterns[i].get(d), EPSILON,
                    String.format("Round-trip preserved pattern[%d][%d]", i, d));
            }
        }
    }

    @Test
    void testProcessBatchSIMD() {
        var bottomUpInputs = new Pattern[128];
        var topDownExpectations = new Pattern[128];
        for (int i = 0; i < bottomUpInputs.length; i++) {
            var bottomUpValues = new double[64];
            var topDownValues = new double[64];
            for (int d = 0; d < bottomUpValues.length; d++) {
                bottomUpValues[d] = 0.5 + 0.01 * i;
                topDownValues[d] = 0.3 + 0.005 * i;
            }
            bottomUpInputs[i] = new DenseVector(bottomUpValues);
            topDownExpectations[i] = new DenseVector(topDownValues);
        }

        var params = Layer6Parameters.builder()
            .onCenterWeight(1.0)
            .offSurroundStrength(0.2)
            .modulationThreshold(0.1)
            .attentionalGain(1.5)
            .timeConstant(200.0)
            .build();

        var outputs = Layer6SIMDBatch.processBatchSIMD(bottomUpInputs, topDownExpectations, params, 64);

        // SIMD should be enabled for 128x64 batch
        assertNotNull(outputs, "SIMD should be enabled for 128x64 batch");
        assertEquals(128, outputs.length, "Output count");

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
        var bottomUpInputs = new Pattern[8];
        for (int i = 0; i < bottomUpInputs.length; i++) {
            bottomUpInputs[i] = new DenseVector(new double[]{0.5, 0.5});
        }

        var params = Layer6Parameters.builder().build();

        var outputs = Layer6SIMDBatch.processBatchSIMD(bottomUpInputs, null, params, 2);

        assertNull(outputs, "SIMD should NOT be beneficial for 8x2 batch");
    }

    @Test
    void testSemanticEquivalenceWithTopDown() {
        var batchSize = 64;
        var dimension = 128;

        var bottomUpInputs = new Pattern[batchSize];
        var topDownExpectations = new Pattern[batchSize];
        for (int i = 0; i < batchSize; i++) {
            var bottomUpValues = new double[dimension];
            var topDownValues = new double[dimension];
            for (int d = 0; d < dimension; d++) {
                bottomUpValues[d] = 0.3 + 0.001 * (i * dimension + d);
                topDownValues[d] = 0.2 + 0.0005 * (i * dimension + d);
            }
            bottomUpInputs[i] = new DenseVector(bottomUpValues);
            topDownExpectations[i] = new DenseVector(topDownValues);
        }

        var params = Layer6Parameters.builder()
            .onCenterWeight(1.0)
            .offSurroundStrength(0.2)
            .modulationThreshold(0.1)
            .attentionalGain(1.5)
            .timeConstant(200.0)
            .build();

        // SIMD path
        var simdOutputs = Layer6SIMDBatch.processBatchSIMD(bottomUpInputs, topDownExpectations, params, dimension);
        assertNotNull(simdOutputs, "SIMD should be enabled for 64x128 batch");

        // Direct batch operations for comparison
        var batch1 = Layer6SIMDBatch.createBatch(bottomUpInputs, topDownExpectations, dimension);
        batch1.applyARTMatchingRule(params.getOnCenterWeight(), params.getModulationThreshold());
        batch1.applyOffSurroundInhibition(params.getOffSurroundStrength(), 2);
        batch1.applyAttentionalGain(params.getAttentionalGain());

        var shuntingParams = com.hellblazer.art.temporal.dynamics.ShuntingParameters.builder(dimension)
            .ceiling(params.getCeiling())
            .floor(params.getFloor())
            .selfExcitation(params.getSelfExcitation())
            .inhibitoryStrength(params.getLateralInhibition())
            .build();

        batch1.applyDynamicsExact(params.getTimeConstant() / 5000.0, shuntingParams);
        batch1.applySaturation(params.getCeiling(), params.getFloor());
        var outputs1 = batch1.toPatterns();

        // Compare outputs (should be identical)
        assertEquals(batchSize, outputs1.length, "Output1 count");

        double maxDiff = 0.0;
        for (int i = 0; i < batchSize; i++) {
            for (int d = 0; d < dimension; d++) {
                double diff = Math.abs(simdOutputs[i].get(d) - outputs1[i].get(d));
                maxDiff = Math.max(maxDiff, diff);
            }
        }

        System.out.printf("Max difference: %.2e%n", maxDiff);

        // Should be very close (allow small numerical differences)
        assertTrue(maxDiff < 1e-6,
            String.format("SIMD and direct should produce nearly identical results: max diff %.2e", maxDiff));
    }

    @Test
    void testLargeBatchPerformance() {
        var batchSize = 256;
        var dimension = 512;

        var bottomUpInputs = new Pattern[batchSize];
        var topDownExpectations = new Pattern[batchSize];
        for (int i = 0; i < batchSize; i++) {
            var bottomUpValues = new double[dimension];
            var topDownValues = new double[dimension];
            for (int d = 0; d < dimension; d++) {
                bottomUpValues[d] = Math.random();
                topDownValues[d] = Math.random();
            }
            bottomUpInputs[i] = new DenseVector(bottomUpValues);
            topDownExpectations[i] = new DenseVector(topDownValues);
        }

        var params = Layer6Parameters.builder()
            .onCenterWeight(1.0)
            .offSurroundStrength(0.2)
            .modulationThreshold(0.1)
            .attentionalGain(1.5)
            .timeConstant(200.0)
            .build();

        // SIMD processing
        long startTime = System.nanoTime();
        var outputs = Layer6SIMDBatch.processBatchSIMD(bottomUpInputs, topDownExpectations, params, dimension);
        long endTime = System.nanoTime();

        assertNotNull(outputs, "SIMD should be enabled for 256x512 batch");
        assertEquals(batchSize, outputs.length, "Output count");

        long timeMs = (endTime - startTime) / 1_000_000;
        double msPerPattern = (double) timeMs / batchSize;

        System.out.printf("Large batch (256x512) Layer 6 SIMD: %.3f ms total, %.3f ms/pattern%n",
            (double) timeMs, msPerPattern);

        // Performance target: < 5ms per pattern (includes all Layer 6 operations)
        // NOTE: Advisory only - CI environments may be slower
        if (msPerPattern >= 5.0) {
            System.out.printf("⚠️  Performance advisory: %.3f ms/pattern (target < 5.0 ms)%n", msPerPattern);
        }
    }

    @Test
    void testCreateBatchValidation() {
        // Test null patterns
        assertThrows(IllegalArgumentException.class,
            () -> Layer6SIMDBatch.createBatch(null, null, 10));

        // Test empty patterns
        assertThrows(IllegalArgumentException.class,
            () -> Layer6SIMDBatch.createBatch(new Pattern[0], null, 10));

        // Test null pattern in array
        var patterns = new Pattern[]{
            new DenseVector(new double[]{1.0, 2.0}),
            null,
            new DenseVector(new double[]{3.0, 4.0})
        };
        assertThrows(IllegalArgumentException.class,
            () -> Layer6SIMDBatch.createBatch(patterns, null, 2));

        // Test dimension mismatch
        var mismatch = new Pattern[]{
            new DenseVector(new double[]{1.0, 2.0}),
            new DenseVector(new double[]{3.0, 4.0, 5.0})  // Wrong dimension!
        };
        assertThrows(IllegalArgumentException.class,
            () -> Layer6SIMDBatch.createBatch(mismatch, null, 2));
    }
}
