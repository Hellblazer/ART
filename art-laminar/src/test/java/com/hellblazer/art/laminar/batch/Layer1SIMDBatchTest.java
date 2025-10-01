package com.hellblazer.art.laminar.batch;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.Layer1Parameters;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Layer 1 SIMD batch processing.
 *
 * Validates Layer 1 specific operations:
 * - Attention state update with very slow decay (200-1000ms)
 * - Memory trace integration (long-term persistence)
 * - Priming effect calculation (attention + memory combination)
 * - Apical dendrite signal generation for Layer 2/3
 * - Very slow timescale validation
 * - Sustained persistence (effects last after input ends)
 * - Priming-only behavior (does NOT drive responses)
 *
 * @author Hal Hildebrand
 */
class Layer1SIMDBatchTest {

    private static final double EPSILON = 1e-6;

    @Test
    void testCreateBatch() {
        var expectations = new Pattern[]{
            new DenseVector(new double[]{1.0, 2.0, 3.0}),
            new DenseVector(new double[]{4.0, 5.0, 6.0})
        };

        var batch = Layer1SIMDBatch.createBatch(expectations, null, null, null, 3);

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
        var expectations = new Pattern[]{
            new DenseVector(new double[]{1.0, 2.0}),
            new DenseVector(new double[]{3.0, 4.0})
        };

        var prevAttention = new Pattern[]{
            new DenseVector(new double[]{0.5, 0.6}),
            new DenseVector(new double[]{0.7, 0.8})
        };

        var prevPriming = new Pattern[]{
            new DenseVector(new double[]{0.1, 0.2}),
            new DenseVector(new double[]{0.3, 0.4})
        };

        var prevMemory = new Pattern[]{
            new DenseVector(new double[]{0.05, 0.06}),
            new DenseVector(new double[]{0.07, 0.08})
        };

        var batch = Layer1SIMDBatch.createBatch(expectations, prevAttention, prevPriming, prevMemory, 2);

        assertEquals(2, batch.getBatchSize());
        assertEquals(2, batch.getDimension());
    }

    @Test
    void testApplyAttentionStateUpdate() {
        var expectations = new Pattern[]{
            new DenseVector(new double[]{0.8, 0.0}),  // Strong attention at dim 0
            new DenseVector(new double[]{0.0, 0.9})   // Strong attention at dim 1
        };

        var prevAttention = new Pattern[]{
            new DenseVector(new double[]{0.5, 0.0}),
            new DenseVector(new double[]{0.0, 0.5})
        };

        var batch = Layer1SIMDBatch.createBatch(expectations, prevAttention, null, null, 2);

        // Apply very slow decay (0.001 = very slow)
        batch.applyAttentionStateUpdate(0.001, 0.3);

        var attentionMajor = batch.getAttentionMajor();

        // Pattern 0: attention[0] should increase (0.5 decayed + 0.8 * 0.3 shift)
        // Pattern 1: attention[1] should increase (0.5 decayed + 0.9 * 0.3 shift)
        assertTrue(attentionMajor[0][0] > 0.5, "Attention should increase with new input");
        assertTrue(attentionMajor[1][1] > 0.5, "Attention should increase with new input");

        // Attention should decay VERY slowly (almost no change)
        assertTrue(attentionMajor[0][0] < 1.0, "Attention should not exceed 1.0");
    }

    @Test
    void testApplyMemoryTraceIntegration() {
        var expectations = new Pattern[]{
            new DenseVector(new double[]{0.8, 0.0}),
            new DenseVector(new double[]{0.0, 0.9})
        };

        var prevAttention = new Pattern[]{
            new DenseVector(new double[]{0.5, 0.0}),  // Above memory threshold
            new DenseVector(new double[]{0.0, 0.5})   // Above memory threshold
        };

        var prevMemory = new Pattern[]{
            new DenseVector(new double[]{0.1, 0.0}),
            new DenseVector(new double[]{0.0, 0.1})
        };

        var batch = Layer1SIMDBatch.createBatch(expectations, prevAttention, null, prevMemory, 2);

        // First update attention state
        batch.applyAttentionStateUpdate(0.001, 0.3);

        // Then integrate memory trace (attention > 0.3 threshold builds memory)
        batch.applyMemoryTraceIntegration(0.00005, 0.3);

        var memoryMajor = batch.getMemoryMajor();

        // Memory should increase slowly where attention is strong
        assertTrue(memoryMajor[0][0] > 0.1, "Memory should accumulate from sustained attention");
        assertTrue(memoryMajor[1][1] > 0.1, "Memory should accumulate from sustained attention");

        // Memory should cap at 0.8
        assertTrue(memoryMajor[0][0] <= 0.8, "Memory should be capped");
    }

    @Test
    void testApplyPrimingEffect() {
        var expectations = new Pattern[]{
            new DenseVector(new double[]{0.8, 0.0}),
            new DenseVector(new double[]{0.0, 0.9})
        };

        var prevAttention = new Pattern[]{
            new DenseVector(new double[]{0.5, 0.0}),
            new DenseVector(new double[]{0.0, 0.5})
        };

        var prevMemory = new Pattern[]{
            new DenseVector(new double[]{0.2, 0.0}),
            new DenseVector(new double[]{0.0, 0.2})
        };

        var batch = Layer1SIMDBatch.createBatch(expectations, prevAttention, null, prevMemory, 2);

        // Update attention and memory first
        batch.applyAttentionStateUpdate(0.001, 0.3);
        batch.applyMemoryTraceIntegration(0.00005, 0.3);

        // Calculate priming effect (attention + 0.5 * memory) * primingStrength
        batch.applyPrimingEffect(0.3);

        var primingMajor = batch.getPrimingMajor();

        // Priming should be combination of attention + memory
        assertTrue(primingMajor[0][0] > 0.0, "Priming should be present");
        assertTrue(primingMajor[1][1] > 0.0, "Priming should be present");

        // Priming should be capped at 0.5 (cannot drive responses)
        assertTrue(primingMajor[0][0] <= 0.5, "Priming should be capped at 0.5");
        assertTrue(primingMajor[1][1] <= 0.5, "Priming should be capped at 0.5");
    }

    @Test
    void testApplyApicalDendriteSignal() {
        var expectations = new Pattern[]{
            new DenseVector(new double[]{0.8, 0.0}),
            new DenseVector(new double[]{0.0, 0.9})
        };

        var prevAttention = new Pattern[]{
            new DenseVector(new double[]{0.5, 0.0}),
            new DenseVector(new double[]{0.0, 0.5})
        };

        var prevPriming = new Pattern[]{
            new DenseVector(new double[]{0.2, 0.0}),
            new DenseVector(new double[]{0.0, 0.2})
        };

        var batch = Layer1SIMDBatch.createBatch(expectations, prevAttention, prevPriming, null, 2);

        // Update attention first
        batch.applyAttentionStateUpdate(0.001, 0.3);

        // Generate apical dendrite signal = (attention + priming) * integration
        batch.applyApicalDendriteSignal(0.5);

        var dimensionMajor = batch.getDimensionMajor();

        // Apical signal should be present for Layer 2/3 integration
        assertTrue(dimensionMajor[0][0] > 0.0, "Apical signal should be present");
        assertTrue(dimensionMajor[1][1] > 0.0, "Apical signal should be present");
    }

    @Test
    void testApplySaturation() {
        var expectations = new Pattern[]{
            new DenseVector(new double[]{0.5, 2.0}),  // 2.0 above ceiling
            new DenseVector(new double[]{-0.5, 0.8})  // -0.5 below floor
        };

        var batch = Layer1SIMDBatch.createBatch(expectations, null, null, null, 2);
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
    void testStatePreservationAcrossBatches() {
        var expectations1 = new Pattern[]{
            new DenseVector(new double[]{0.8, 0.0}),
            new DenseVector(new double[]{0.0, 0.8})
        };

        var batch1 = Layer1SIMDBatch.createBatch(expectations1, null, null, null, 2);
        batch1.applyAttentionStateUpdate(0.001, 0.3);

        var attentionAfterBatch1 = batch1.getAttentionMajor();
        var attention1_0 = attentionAfterBatch1[0][0];

        // Second batch with zero input - attention should persist
        var expectations2 = new Pattern[]{
            new DenseVector(new double[]{0.0, 0.0}),
            new DenseVector(new double[]{0.0, 0.0})
        };

        // Use previous attention state from batch1
        var prevAttention = new Pattern[]{
            new DenseVector(new double[]{attention1_0, 0.0}),
            new DenseVector(new double[]{0.0, attentionAfterBatch1[1][1]})
        };

        var batch2 = Layer1SIMDBatch.createBatch(expectations2, prevAttention, null, null, 2);
        batch2.applyAttentionStateUpdate(0.001, 0.3);

        var attentionAfterBatch2 = batch2.getAttentionMajor();

        // Attention should persist (very slow decay means minimal change)
        assertTrue(attentionAfterBatch2[0][0] > attention1_0 * 0.95,
            "Attention should persist with very slow decay");
    }

    @Test
    void testMemoryTracePersistence() {
        // Build up memory trace over multiple steps
        var expectations = new Pattern[]{
            new DenseVector(new double[]{0.8, 0.0}),
            new DenseVector(new double[]{0.0, 0.8})
        };

        var prevAttention = new Pattern[]{
            new DenseVector(new double[]{0.5, 0.0}),
            new DenseVector(new double[]{0.0, 0.5})
        };

        var prevMemory = new Pattern[]{
            new DenseVector(new double[]{0.1, 0.0}),
            new DenseVector(new double[]{0.0, 0.1})
        };

        var batch = Layer1SIMDBatch.createBatch(expectations, prevAttention, null, prevMemory, 2);

        // Update attention and memory
        batch.applyAttentionStateUpdate(0.001, 0.3);
        batch.applyMemoryTraceIntegration(0.00005, 0.3);

        var memoryBefore = batch.getMemoryMajor()[0][0];

        // Apply again with zero input
        var expectations2 = new Pattern[]{
            new DenseVector(new double[]{0.0, 0.0}),
            new DenseVector(new double[]{0.0, 0.0})
        };

        var newAttention = new Pattern[]{
            new DenseVector(new double[]{batch.getAttentionMajor()[0][0], 0.0}),
            new DenseVector(new double[]{0.0, batch.getAttentionMajor()[1][1]})
        };

        var newMemory = new Pattern[]{
            new DenseVector(new double[]{memoryBefore, 0.0}),
            new DenseVector(new double[]{0.0, batch.getMemoryMajor()[1][1]})
        };

        var batch2 = Layer1SIMDBatch.createBatch(expectations2, newAttention, null, newMemory, 2);
        batch2.applyMemoryTraceIntegration(0.00005, 0.3);

        var memoryAfter = batch2.getMemoryMajor()[0][0];

        // Memory should persist (very slow decay)
        assertTrue(memoryAfter > memoryBefore * 0.99,
            "Memory should persist for seconds with minimal decay");
    }

    @Test
    void testVerySlowTimescale() {
        // Test that Layer 1 time constants are indeed very slow (200-1000ms)
        var expectations = new Pattern[]{
            new DenseVector(new double[]{0.8, 0.0}),
            new DenseVector(new double[]{0.0, 0.8})
        };

        var prevAttention = new Pattern[]{
            new DenseVector(new double[]{0.5, 0.0}),
            new DenseVector(new double[]{0.0, 0.5})
        };

        var batch = Layer1SIMDBatch.createBatch(expectations, prevAttention, null, null, 2);

        // Very slow decay rate (0.001 corresponds to ~1000ms time constant)
        batch.applyAttentionStateUpdate(0.001, 0.3);

        var attentionAfter = batch.getAttentionMajor()[0][0];
        var attentionBefore = 0.5;

        // With very slow decay, attention should change slowly
        // Decay: 0.5 * (1 - 0.001 * 0.01) + 0.8 * 0.3 = ~0.74
        var expectedApprox = attentionBefore * (1.0 - 0.001 * 0.01) + 0.8 * 0.3;

        assertEquals(expectedApprox, attentionAfter, 0.05,
            "Attention should evolve with very slow timescale");
    }

    @Test
    void testSustainedPersistence() {
        // Test that effects last long after input ends
        var expectations = new Pattern[]{
            new DenseVector(new double[]{0.8, 0.0}),
            new DenseVector(new double[]{0.0, 0.8})
        };

        var batch = Layer1SIMDBatch.createBatch(expectations, null, null, null, 2);

        // Build up attention
        for (int step = 0; step < 10; step++) {
            batch.applyAttentionStateUpdate(0.001, 0.3);
            batch.applyMemoryTraceIntegration(0.00005, 0.3);
        }

        var attentionPeak = batch.getAttentionMajor()[0][0];
        var memoryPeak = batch.getMemoryMajor()[0][0];

        // Now remove input and measure persistence
        var zeroExpectations = new Pattern[]{
            new DenseVector(new double[]{0.0, 0.0}),
            new DenseVector(new double[]{0.0, 0.0})
        };

        var currentAttention = new Pattern[]{
            new DenseVector(new double[]{attentionPeak, 0.0}),
            new DenseVector(new double[]{0.0, batch.getAttentionMajor()[1][1]})
        };

        var currentMemory = new Pattern[]{
            new DenseVector(new double[]{memoryPeak, 0.0}),
            new DenseVector(new double[]{0.0, batch.getMemoryMajor()[1][1]})
        };

        var batch2 = Layer1SIMDBatch.createBatch(zeroExpectations, currentAttention, null, currentMemory, 2);

        // Apply 100 steps with zero input
        for (int step = 0; step < 100; step++) {
            batch2.applyAttentionStateUpdate(0.001, 0.3);
        }

        var attentionAfter = batch2.getAttentionMajor()[0][0];

        // Attention should still be > 30% of peak after 100 steps
        assertTrue(attentionAfter > attentionPeak * 0.3,
            String.format("Attention should persist: peak=%.3f, after100=%.3f", attentionPeak, attentionAfter));
    }

    @Test
    void testPrimingOnlyNoDriving() {
        // Test that Layer 1 provides priming but does NOT drive responses
        var expectations = new Pattern[]{
            new DenseVector(new double[]{1.0, 1.0}),  // Maximum input
            new DenseVector(new double[]{1.0, 1.0})
        };

        var batch = Layer1SIMDBatch.createBatch(expectations, null, null, null, 2);

        // Apply all operations
        batch.applyAttentionStateUpdate(0.001, 0.3);
        batch.applyMemoryTraceIntegration(0.00005, 0.3);
        batch.applyPrimingEffect(0.3);

        var primingMajor = batch.getPrimingMajor();

        // Priming should be CAPPED at 0.5 (cannot drive responses)
        for (int d = 0; d < 2; d++) {
            for (int i = 0; i < 2; i++) {
                assertTrue(primingMajor[d][i] <= 0.5,
                    String.format("Priming MUST be <= 0.5 (cannot drive): priming[%d][%d]=%.3f",
                        d, i, primingMajor[d][i]));
            }
        }
    }

    @Test
    void testSmallBatch() {
        // Small batch - should return null (not beneficial)
        var expectations = new Pattern[8];
        for (int i = 0; i < expectations.length; i++) {
            expectations[i] = new DenseVector(new double[]{0.5, 0.5});
        }

        var params = Layer1Parameters.builder().build();

        var outputs = Layer1SIMDBatch.processTopDownBatchSIMD(expectations, params, 2);

        assertNull(outputs, "SIMD should NOT be beneficial for 8x2 batch");
    }

    @Test
    void testSinglePattern() {
        // Single pattern - should return null
        var expectations = new Pattern[]{
            new DenseVector(new double[]{0.8, 0.0, 0.0})
        };

        var params = Layer1Parameters.builder().build();

        var outputs = Layer1SIMDBatch.processTopDownBatchSIMD(expectations, params, 3);

        assertNull(outputs, "SIMD should NOT be beneficial for single pattern");
    }

    @Test
    void testEmptyExpectation() {
        // Zero input - should still process correctly
        var expectations = new Pattern[]{
            new DenseVector(new double[]{0.0, 0.0}),
            new DenseVector(new double[]{0.0, 0.0})
        };

        var prevAttention = new Pattern[]{
            new DenseVector(new double[]{0.5, 0.0}),
            new DenseVector(new double[]{0.0, 0.5})
        };

        var batch = Layer1SIMDBatch.createBatch(expectations, prevAttention, null, null, 2);
        batch.applyAttentionStateUpdate(0.001, 0.3);

        var attentionAfter = batch.getAttentionMajor();

        // Attention should decay but persist
        assertTrue(attentionAfter[0][0] > 0.0, "Attention should persist even with zero input");
        assertTrue(attentionAfter[0][0] < 0.5, "Attention should decay slightly");
    }

    @Test
    void testSemanticEquivalence() {
        // Test complete pipeline semantic equivalence (0.00e+00 max difference)
        var batchSize = 64;
        var dimension = 128;

        var expectations = new Pattern[batchSize];
        for (int i = 0; i < batchSize; i++) {
            var values = new double[dimension];
            for (int d = 0; d < dimension; d++) {
                values[d] = 0.3 + 0.001 * (i * dimension + d);
            }
            expectations[i] = new DenseVector(values);
        }

        var params = Layer1Parameters.builder()
            .timeConstant(500.0)
            .primingStrength(0.3)
            .sustainedDecayRate(0.001)
            .apicalIntegration(0.5)
            .attentionShiftRate(0.3)
            .build();

        // SIMD path
        var simdOutputs = Layer1SIMDBatch.processTopDownBatchSIMD(expectations, params, dimension);
        assertNotNull(simdOutputs, "SIMD should be enabled for 64x128 batch");

        // Direct batch operations for comparison
        var batch1 = Layer1SIMDBatch.createBatch(expectations, null, null, null, dimension);
        batch1.applyAttentionStateUpdate(params.getSustainedDecayRate(), params.getAttentionShiftRate());
        batch1.applyMemoryTraceIntegration(params.getSustainedDecayRate() * 0.05, 0.3);
        batch1.applyPrimingEffect(params.getPrimingStrength());
        batch1.applyApicalDendriteSignal(params.getApicalIntegration());

        var shuntingParams = com.hellblazer.art.temporal.dynamics.ShuntingParameters.builder(dimension)
            .ceiling(params.getCeiling())
            .floor(params.getFloor())
            .selfExcitation(params.getSelfExcitation())
            .inhibitoryStrength(params.getLateralInhibition())
            .build();

        batch1.applyDynamicsExact(params.getTimeConstant() / 20000.0, shuntingParams);
        batch1.applySaturation(params.getCeiling(), params.getFloor());
        var outputs1 = batch1.toPatterns();

        // Compare outputs (should be bit-exact)
        assertEquals(batchSize, outputs1.length, "Output1 count");

        double maxDiff = 0.0;
        for (int i = 0; i < batchSize; i++) {
            for (int d = 0; d < dimension; d++) {
                double diff = Math.abs(simdOutputs[i].get(d) - outputs1[i].get(d));
                maxDiff = Math.max(maxDiff, diff);
            }
        }

        System.out.printf("Layer 1 SIMD max difference: %.2e%n", maxDiff);

        // Should be bit-exact (0.00e+00 max difference)
        assertTrue(maxDiff < 1e-10,
            String.format("SIMD and direct should produce bit-exact results: max diff %.2e", maxDiff));
    }

    @Test
    void testToPatterns() {
        var original = new Pattern[]{
            new DenseVector(new double[]{1.0, 2.0, 3.0}),
            new DenseVector(new double[]{4.0, 5.0, 6.0})
        };

        var batch = Layer1SIMDBatch.createBatch(original, null, null, null, 3);
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
    void testCreateBatchValidation() {
        // Test null expectations
        assertThrows(IllegalArgumentException.class,
            () -> Layer1SIMDBatch.createBatch(null, null, null, null, 10));

        // Test empty expectations
        assertThrows(IllegalArgumentException.class,
            () -> Layer1SIMDBatch.createBatch(new Pattern[0], null, null, null, 10));

        // Test null pattern in array
        var expectations = new Pattern[]{
            new DenseVector(new double[]{1.0, 2.0}),
            null,
            new DenseVector(new double[]{3.0, 4.0})
        };
        assertThrows(IllegalArgumentException.class,
            () -> Layer1SIMDBatch.createBatch(expectations, null, null, null, 2));

        // Test dimension mismatch
        var mismatch = new Pattern[]{
            new DenseVector(new double[]{1.0, 2.0}),
            new DenseVector(new double[]{3.0, 4.0, 5.0})  // Wrong dimension!
        };
        assertThrows(IllegalArgumentException.class,
            () -> Layer1SIMDBatch.createBatch(mismatch, null, null, null, 2));
    }
}
