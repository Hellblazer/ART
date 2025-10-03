package com.hellblazer.art.cortical.batch;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Validation tests for SIMD batch processing configuration and correctness.
 *
 * <p>This test suite validates:
 * <ul>
 *   <li>SIMDConfiguration creation and validation</li>
 *   <li>Auto-tuning batch size selection</li>
 *   <li>SIMD usage decision logic</li>
 *   <li>Bit-exact equivalence with sequential processing (when SIMD implementations exist)</li>
 * </ul>
 *
 * <h2>Test Strategy</h2>
 * <p>Following Phase 1A test-first approach:
 * <ol>
 *   <li>Configuration validation tests (this class)</li>
 *   <li>Bit-exact equivalence tests (added as SIMD implementations are ported)</li>
 *   <li>Performance benchmarks (SIMDBenchmark.java)</li>
 * </ol>
 *
 * @author Phase 1A: Preparation & Test Framework
 */
@DisplayName("SIMD Configuration Validation")
class SIMDValidationTest {

    private static final double EPSILON = 1e-10;

    @Test
    @DisplayName("Optimal configuration has correct defaults")
    void testOptimalConfiguration() {
        var config = SIMDConfiguration.optimal();

        assertEquals(64, config.miniBatchSize(), "Default mini-batch size should be 64");
        assertTrue(config.autoTuning(), "Auto-tuning should be enabled by default");
        assertEquals(1.05, config.fallbackThreshold(), EPSILON, "Default fallback threshold");
        assertTrue(config.vectorLaneCount() >= 2, "Vector lane count should be at least 2 for doubles");
    }

    @Test
    @DisplayName("Custom batch size configuration")
    void testCustomBatchSize() {
        var config = SIMDConfiguration.withBatchSize(128);

        assertEquals(128, config.miniBatchSize(), "Should use specified batch size");
        assertTrue(config.autoTuning(), "Auto-tuning should still be enabled");
    }

    @Test
    @DisplayName("Invalid batch size throws exception")
    void testInvalidBatchSize() {
        assertThrows(IllegalArgumentException.class, () ->
            SIMDConfiguration.withBatchSize(8),
            "Batch size below minimum should throw"
        );
    }

    @Test
    @DisplayName("Auto-tuning configuration")
    void testAutoTuningConfig() {
        var enabled = SIMDConfiguration.withAutoTuning(true);
        var disabled = SIMDConfiguration.withAutoTuning(false);

        assertTrue(enabled.autoTuning(), "Auto-tuning should be enabled");
        assertFalse(disabled.autoTuning(), "Auto-tuning should be disabled");
    }

    @Test
    @DisplayName("Batch size must be >= vector lane count")
    void testBatchSizeValidation() {
        int laneCount = SIMDConfiguration.hardwareVectorLaneCount();

        assertThrows(IllegalArgumentException.class, () ->
            new SIMDConfiguration(laneCount - 1, laneCount, true, 1.05),
            "Batch size < lane count should throw"
        );

        // Should not throw
        assertDoesNotThrow(() ->
            new SIMDConfiguration(laneCount, laneCount, true, 1.05),
            "Batch size == lane count should be valid"
        );
    }

    @Test
    @DisplayName("Fallback threshold validation")
    void testFallbackThresholdValidation() {
        int laneCount = SIMDConfiguration.hardwareVectorLaneCount();

        assertThrows(IllegalArgumentException.class, () ->
            new SIMDConfiguration(64, laneCount, true, 0.5),
            "Threshold < 1.0 should throw"
        );

        assertThrows(IllegalArgumentException.class, () ->
            new SIMDConfiguration(64, laneCount, true, 15.0),
            "Threshold > 10.0 should throw"
        );

        // Should not throw
        assertDoesNotThrow(() ->
            new SIMDConfiguration(64, laneCount, true, 1.0),
            "Threshold 1.0 should be valid"
        );

        assertDoesNotThrow(() ->
            new SIMDConfiguration(64, laneCount, true, 10.0),
            "Threshold 10.0 should be valid"
        );
    }

    @Test
    @DisplayName("SIMD usage decision for various batch sizes")
    void testShouldUseSIMD() {
        var config = SIMDConfiguration.optimal();  // Auto-tuning enabled, mini-batch size 64

        // Too small for SIMD
        assertFalse(config.shouldUseSIMD(8), "Batch size 8 too small");
        assertFalse(config.shouldUseSIMD(15), "Batch size 15 too small");

        // With auto-tuning, needs at least 2 mini-batches (128 patterns for batch size 64)
        assertFalse(config.shouldUseSIMD(64), "Single mini-batch insufficient with auto-tuning");
        assertFalse(config.shouldUseSIMD(100), "Less than 2 mini-batches");
        assertTrue(config.shouldUseSIMD(128), "Two mini-batches sufficient");
        assertTrue(config.shouldUseSIMD(256), "Large batch should use SIMD");
    }

    @Test
    @DisplayName("SIMD usage without auto-tuning")
    void testShouldUseSIMDWithoutAutoTuning() {
        var config = SIMDConfiguration.withAutoTuning(false);

        // Without auto-tuning, just needs to meet minimum and configured mini-batch size
        assertFalse(config.shouldUseSIMD(8), "Below minimum");
        assertFalse(config.shouldUseSIMD(32), "Below mini-batch size");
        assertTrue(config.shouldUseSIMD(64), "Meets mini-batch size");
        assertTrue(config.shouldUseSIMD(128), "Above mini-batch size");
    }

    @Test
    @DisplayName("Optimal mini-batch size selection")
    void testOptimalMiniBatchSize() {
        var config = SIMDConfiguration.optimal();

        // Very small batches: use full batch (sequential)
        assertEquals(8, config.getOptimalMiniBatchSize(8), "Tiny batch");
        assertEquals(15, config.getOptimalMiniBatchSize(15), "Small batch");

        // Medium batches: use 32-pattern mini-batches
        assertEquals(32, config.getOptimalMiniBatchSize(32), "Medium batch lower bound");
        assertEquals(32, config.getOptimalMiniBatchSize(64), "Medium batch mid");
        assertEquals(32, config.getOptimalMiniBatchSize(127), "Medium batch upper bound");

        // Large batches: use 64-pattern mini-batches
        assertEquals(64, config.getOptimalMiniBatchSize(128), "Large batch lower bound");
        assertEquals(64, config.getOptimalMiniBatchSize(256), "Large batch");
        assertEquals(64, config.getOptimalMiniBatchSize(1024), "Very large batch");
    }

    @Test
    @DisplayName("Optimal mini-batch size without auto-tuning")
    void testOptimalMiniBatchSizeNoAutoTuning() {
        var config = SIMDConfiguration.withAutoTuning(false);

        // Without auto-tuning, always return configured mini-batch size
        assertEquals(64, config.getOptimalMiniBatchSize(8), "Uses configured size");
        assertEquals(64, config.getOptimalMiniBatchSize(32), "Uses configured size");
        assertEquals(64, config.getOptimalMiniBatchSize(128), "Uses configured size");
        assertEquals(64, config.getOptimalMiniBatchSize(1024), "Uses configured size");
    }

    @Test
    @DisplayName("Hardware vector lane count is reasonable")
    void testHardwareVectorLaneCount() {
        int laneCount = SIMDConfiguration.hardwareVectorLaneCount();

        assertTrue(laneCount >= 2, "Lane count should be at least 2 for doubles");
        assertTrue(laneCount <= 16, "Lane count should not exceed 16 for current hardware");
        assertTrue(isPowerOf2(laneCount), "Lane count should be power of 2");

        System.out.printf("Hardware vector lane count: %d (for doubles)%n", laneCount);
    }

    @Test
    @DisplayName("Vector species is available")
    void testVectorSpecies() {
        var species = SIMDConfiguration.vectorSpecies();

        assertNotNull(species, "Vector species should be available");
        assertEquals(SIMDConfiguration.hardwareVectorLaneCount(), species.length(),
            "Species length should match hardware lane count");
    }

    /**
     * Test helper: check if number is power of 2.
     */
    private boolean isPowerOf2(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }

    // ===== Bit-Exact Equivalence Tests =====
    // These will be implemented as SIMD layer implementations are ported

    // @Test
    // @DisplayName("Layer4 SIMD batch-64 bit-exact with sequential")
    // void testLayer4MiniBatch64BitExactEquivalence() {
    //     // TODO: Implement when Layer4SIMDBatch is ported (Phase 1D)
    // }

    // @Test
    // @DisplayName("Adaptive batch sizing maintains bit-exact results")
    // void testAdaptiveBatchSizing() {
    //     // TODO: Implement in Phase 1C
    // }
}
