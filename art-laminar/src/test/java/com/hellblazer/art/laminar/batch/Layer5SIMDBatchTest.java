package com.hellblazer.art.laminar.batch;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.Layer5Parameters;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Layer 5 SIMD batch processing.
 *
 * Validates Layer 5 specific operations:
 * - Amplification gain
 * - Burst firing detection and amplification
 * - State persistence
 * - Exact shunting dynamics
 * - Output normalization
 *
 * @author Claude Code
 */
class Layer5SIMDBatchTest {

    private static final double EPSILON = 1e-6;

    @Test
    void testCreateBatch() {
        var patterns = new Pattern[]{
            new DenseVector(new double[]{1.0, 2.0, 3.0}),
            new DenseVector(new double[]{4.0, 5.0, 6.0})
        };

        var batch = Layer5SIMDBatch.createBatch(patterns, null, 3);

        assertEquals(2, batch.getBatchSize(), "Batch size");
        assertEquals(3, batch.getDimension(), "Dimension");

        // Validate transpose correctness
        var dimensionMajor = batch.getDimensionMajor();
        assertArrayEquals(new double[]{1.0, 4.0}, dimensionMajor[0], EPSILON);
        assertArrayEquals(new double[]{2.0, 5.0}, dimensionMajor[1], EPSILON);
        assertArrayEquals(new double[]{3.0, 6.0}, dimensionMajor[2], EPSILON);
    }

    @Test
    void testCreateBatchWithPreviousStates() {
        var patterns = new Pattern[]{
            new DenseVector(new double[]{1.0, 2.0}),
            new DenseVector(new double[]{3.0, 4.0})
        };

        var previousStates = new Pattern[]{
            new DenseVector(new double[]{0.5, 0.6}),
            new DenseVector(new double[]{0.7, 0.8})
        };

        var batch = Layer5SIMDBatch.createBatch(patterns, previousStates, 2);

        assertEquals(2, batch.getBatchSize());
        assertEquals(2, batch.getDimension());
    }

    @Test
    void testApplyAmplificationGain() {
        var patterns = new Pattern[]{
            new DenseVector(new double[]{1.0, 2.0}),
            new DenseVector(new double[]{3.0, 4.0})
        };

        var batch = Layer5SIMDBatch.createBatch(patterns, null, 2);
        batch.applyAmplificationGain(2.0);

        var dimensionMajor = batch.getDimensionMajor();

        // All values should be scaled by 2.0
        assertArrayEquals(new double[]{2.0, 6.0}, dimensionMajor[0], EPSILON);
        assertArrayEquals(new double[]{4.0, 8.0}, dimensionMajor[1], EPSILON);
    }

    @Test
    void testApplyBurstFiring() {
        var patterns = new Pattern[]{
            new DenseVector(new double[]{0.5, 0.9}),  // 0.9 > 0.8 threshold
            new DenseVector(new double[]{0.6, 0.85}) // 0.85 > 0.8 threshold
        };

        var batch = Layer5SIMDBatch.createBatch(patterns, null, 2);
        batch.applyBurstFiring(0.8, 2.0);  // Threshold 0.8, amplify by 2.0

        var dimensionMajor = batch.getDimensionMajor();

        // Pattern 0: [0.5, 1.8] (0.9 * 2.0)
        assertArrayEquals(new double[]{0.5, 0.6}, dimensionMajor[0], EPSILON, "Below threshold unchanged");
        assertArrayEquals(new double[]{1.8, 1.7}, dimensionMajor[1], EPSILON, "Above threshold amplified");
    }

    @Test
    void testApplyStatePersistence() {
        var patterns = new Pattern[]{
            new DenseVector(new double[]{1.0, 2.0}),
            new DenseVector(new double[]{3.0, 4.0})
        };

        var previousStates = new Pattern[]{
            new DenseVector(new double[]{0.5, 0.6}),
            new DenseVector(new double[]{0.7, 0.8})
        };

        var batch = Layer5SIMDBatch.createBatch(patterns, previousStates, 2);
        var persistence = 0.9;  // 90% persistence
        batch.applyStatePersistence(persistence);

        var dimensionMajor = batch.getDimensionMajor();

        // Result: current + 0.9 * previous
        // Dim 0: [1.0 + 0.9*0.5, 3.0 + 0.9*0.7] = [1.45, 3.63]
        // Dim 1: [2.0 + 0.9*0.6, 4.0 + 0.9*0.8] = [2.54, 4.72]
        assertArrayEquals(new double[]{1.45, 3.63}, dimensionMajor[0], EPSILON);
        assertArrayEquals(new double[]{2.54, 4.72}, dimensionMajor[1], EPSILON);
    }

    @Test
    void testApplyOutputGain() {
        var patterns = new Pattern[]{
            new DenseVector(new double[]{1.0, 2.0}),
            new DenseVector(new double[]{3.0, 4.0})
        };

        var batch = Layer5SIMDBatch.createBatch(patterns, null, 2);
        batch.applyOutputGain(1.5);

        var dimensionMajor = batch.getDimensionMajor();

        assertArrayEquals(new double[]{1.5, 4.5}, dimensionMajor[0], EPSILON);
        assertArrayEquals(new double[]{3.0, 6.0}, dimensionMajor[1], EPSILON);
    }

    @Test
    void testApplyOutputNormalization() {
        var patterns = new Pattern[]{
            new DenseVector(new double[]{1.0, 2.0}),
            new DenseVector(new double[]{3.0, 4.0})
        };

        var batch = Layer5SIMDBatch.createBatch(patterns, null, 2);
        batch.applyOutputNormalization(0.1);

        var dimensionMajor = batch.getDimensionMajor();

        // Pattern 0 sum: 1.0 + 2.0 = 3.0
        // Normalizer: 1 / (1 + 0.1 * 3.0) = 1 / 1.3 = 0.769
        // Pattern 1 sum: 3.0 + 4.0 = 7.0
        // Normalizer: 1 / (1 + 0.1 * 7.0) = 1 / 1.7 = 0.588

        assertEquals(1.0 * 0.769, dimensionMajor[0][0], 0.01, "Pattern 0, dim 0");
        assertEquals(2.0 * 0.769, dimensionMajor[1][0], 0.01, "Pattern 0, dim 1");
        assertEquals(3.0 * 0.588, dimensionMajor[0][1], 0.01, "Pattern 1, dim 0");
        assertEquals(4.0 * 0.588, dimensionMajor[1][1], 0.01, "Pattern 1, dim 1");
    }

    @Test
    void testApplySaturation() {
        var patterns = new Pattern[]{
            new DenseVector(new double[]{0.5, 2.0}),  // 2.0 above ceiling
            new DenseVector(new double[]{-0.5, 0.8})  // -0.5 below floor
        };

        var batch = Layer5SIMDBatch.createBatch(patterns, null, 2);
        batch.applySaturation(1.0, 0.0);

        var dimensionMajor = batch.getDimensionMajor();

        // All values clamped to [0.0, 1.0]
        for (int d = 0; d < 2; d++) {
            for (int i = 0; i < 2; i++) {
                assertTrue(dimensionMajor[d][i] >= 0.0, "Should be >= floor");
                assertTrue(dimensionMajor[d][i] <= 1.0, "Should be <= ceiling");
            }
        }
    }

    @Test
    void testUpdatePreviousActivation() {
        var patterns = new Pattern[]{
            new DenseVector(new double[]{1.0, 2.0}),
            new DenseVector(new double[]{3.0, 4.0})
        };

        var batch = Layer5SIMDBatch.createBatch(patterns, null, 2);

        // Modify current state
        batch.applyAmplificationGain(2.0);

        // Update previous activation
        batch.updatePreviousActivation();

        // Previous activation should now match current (after amplification)
        var dimensionMajor = batch.getDimensionMajor();
        assertEquals(dimensionMajor[0][0], 2.0, EPSILON);
        assertEquals(dimensionMajor[1][1], 8.0, EPSILON);
    }

    @Test
    void testToPatterns() {
        var original = new Pattern[]{
            new DenseVector(new double[]{1.0, 2.0, 3.0}),
            new DenseVector(new double[]{4.0, 5.0, 6.0})
        };

        var batch = Layer5SIMDBatch.createBatch(original, null, 3);
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
        var patterns = new Pattern[128];
        for (int i = 0; i < patterns.length; i++) {
            var values = new double[64];
            for (int d = 0; d < values.length; d++) {
                values[d] = 0.5 + 0.01 * i;
            }
            patterns[i] = new DenseVector(values);
        }

        var params = Layer5Parameters.builder()
            .amplificationGain(1.5)
            .burstThreshold(0.8)
            .burstAmplification(2.0)
            .outputGain(1.0)
            .outputNormalization(0.01)
            .timeConstant(100.0)
            .build();

        var outputs = Layer5SIMDBatch.processBatchSIMD(patterns, null, params, 64);

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
        var patterns = new Pattern[8];
        for (int i = 0; i < patterns.length; i++) {
            patterns[i] = new DenseVector(new double[]{0.5, 0.5});
        }

        var params = Layer5Parameters.builder().build();

        var outputs = Layer5SIMDBatch.processBatchSIMD(patterns, null, params, 2);

        assertNull(outputs, "SIMD should NOT be beneficial for 8x2 batch");
    }

    @Test
    void testSemanticEquivalenceWithStatePersistence() {
        var batchSize = 64;
        var dimension = 128;

        var patterns = new Pattern[batchSize];
        var previousStates = new Pattern[batchSize];
        for (int i = 0; i < batchSize; i++) {
            var values = new double[dimension];
            var prevValues = new double[dimension];
            for (int d = 0; d < dimension; d++) {
                values[d] = 0.3 + 0.001 * (i * dimension + d);
                prevValues[d] = 0.2 + 0.001 * (i * dimension + d);
            }
            patterns[i] = new DenseVector(values);
            previousStates[i] = new DenseVector(prevValues);
        }

        var params = Layer5Parameters.builder()
            .amplificationGain(1.5)
            .burstThreshold(0.8)
            .burstAmplification(2.0)
            .outputGain(1.0)
            .outputNormalization(0.01)
            .timeConstant(100.0)
            .build();

        // SIMD path
        var simdOutputs = Layer5SIMDBatch.processBatchSIMD(patterns, previousStates, params, dimension);
        assertNotNull(simdOutputs, "SIMD should be enabled for 64x128 batch");

        // Direct batch operations for comparison
        var batch1 = Layer5SIMDBatch.createBatch(patterns, previousStates, dimension);
        batch1.applyAmplificationGain(params.getAmplificationGain());
        batch1.applyBurstFiring(params.getBurstThreshold(), params.getBurstAmplification());
        var persistence = 1.0 - params.getDecayRate() * 0.01;
        batch1.applyStatePersistence(persistence);

        var shuntingParams = com.hellblazer.art.temporal.dynamics.ShuntingParameters.builder(dimension)
            .ceiling(params.getCeiling())
            .floor(params.getFloor())
            .selfExcitation(params.getSelfExcitation())
            .inhibitoryStrength(params.getLateralInhibition())
            .build();

        batch1.applyDynamicsExact(Math.min(params.getTimeConstant() / 10000.0, 0.01), shuntingParams);
        batch1.applyOutputGain(params.getOutputGain());
        batch1.applyOutputNormalization(params.getOutputNormalization());
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

        var patterns = new Pattern[batchSize];
        for (int i = 0; i < batchSize; i++) {
            var values = new double[dimension];
            for (int d = 0; d < dimension; d++) {
                values[d] = Math.random();
            }
            patterns[i] = new DenseVector(values);
        }

        var params = Layer5Parameters.builder()
            .amplificationGain(1.5)
            .burstThreshold(0.8)
            .burstAmplification(2.0)
            .outputGain(1.0)
            .outputNormalization(0.01)
            .timeConstant(100.0)
            .build();

        // SIMD processing
        long startTime = System.nanoTime();
        var outputs = Layer5SIMDBatch.processBatchSIMD(patterns, null, params, dimension);
        long endTime = System.nanoTime();

        assertNotNull(outputs, "SIMD should be enabled for 256x512 batch");
        assertEquals(batchSize, outputs.length, "Output count");

        long timeMs = (endTime - startTime) / 1_000_000;
        double msPerPattern = (double) timeMs / batchSize;

        System.out.printf("Large batch (256x512) Layer 5 SIMD: %.3f ms total, %.3f ms/pattern%n",
            (double) timeMs, msPerPattern);

        // Performance target: < 5ms per pattern (includes all Layer 5 operations)
        assertTrue(msPerPattern < 5.0,
            String.format("Layer 5 SIMD performance should be reasonable: %.3f ms/pattern", msPerPattern));
    }

    @Test
    void testCreateBatchValidation() {
        // Test null patterns
        assertThrows(IllegalArgumentException.class,
            () -> Layer5SIMDBatch.createBatch(null, null, 10));

        // Test empty patterns
        assertThrows(IllegalArgumentException.class,
            () -> Layer5SIMDBatch.createBatch(new Pattern[0], null, 10));

        // Test null pattern in array
        var patterns = new Pattern[]{
            new DenseVector(new double[]{1.0, 2.0}),
            null,
            new DenseVector(new double[]{3.0, 4.0})
        };
        assertThrows(IllegalArgumentException.class,
            () -> Layer5SIMDBatch.createBatch(patterns, null, 2));

        // Test dimension mismatch
        var mismatch = new Pattern[]{
            new DenseVector(new double[]{1.0, 2.0}),
            new DenseVector(new double[]{3.0, 4.0, 5.0})  // Wrong dimension!
        };
        assertThrows(IllegalArgumentException.class,
            () -> Layer5SIMDBatch.createBatch(mismatch, null, 2));
    }
}
