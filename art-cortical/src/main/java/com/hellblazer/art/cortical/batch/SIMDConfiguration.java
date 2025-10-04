package com.hellblazer.art.cortical.batch;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Configuration for SIMD batch processing in cortical layers.
 *
 * <p>This record encapsulates all parameters needed for optimal SIMD performance:
 * <ul>
 *   <li><b>miniBatchSize</b>: Number of patterns processed together (default: 64, increased from 32)</li>
 *   <li><b>vectorLaneCount</b>: Hardware vector lane width (typically 2, 4, or 8 for doubles)</li>
 *   <li><b>autoTuning</b>: Enable automatic batch size selection based on input size</li>
 *   <li><b>fallbackThreshold</b>: Speedup threshold below which to fall back to sequential (default: 1.05)</li>
 * </ul>
 *
 * <h2>Design Rationale</h2>
 * <p>Based on Phase 1 SIMD enhancement plan:
 * <ul>
 *   <li>64-pattern mini-batch size targets 1.40x-1.50x speedup (up from 1.30x)</li>
 *   <li>Auto-tuning adapts to small batches that don't benefit from SIMD overhead</li>
 *   <li>Fallback threshold prevents performance regression on edge cases</li>
 *   <li>Vector lane count matches hardware capabilities (SPECIES_PREFERRED)</li>
 * </ul>
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * // Optimal configuration for most use cases
 * var config = SIMDConfiguration.optimal();
 *
 * // Custom configuration for large batches
 * var config = SIMDConfiguration.withBatchSize(128);
 *
 * // Disable auto-tuning for predictable performance
 * var config = new SIMDConfiguration(64, SPECIES.length(), false, 1.0);
 * }</pre>
 *
 * @param miniBatchSize Number of patterns in each mini-batch (must be >= vectorLaneCount)
 * @param vectorLaneCount Vector lane width from VectorSpecies.PREFERRED.length()
 * @param autoTuning Enable automatic batch size adaptation
 * @param fallbackThreshold Minimum speedup to use SIMD (1.0 = always use, 1.05 = require 5% improvement)
 *
 * @author Phase 1 SIMD Enhancement (ART_CORTICAL_ENHANCEMENT_PLAN.md)
 */
public record SIMDConfiguration(
    int miniBatchSize,
    int vectorLaneCount,
    boolean autoTuning,
    double fallbackThreshold
) {
    /** Default mini-batch size (increased from 32 to 64 in Phase 1) */
    public static final int DEFAULT_MINI_BATCH_SIZE = 64;

    /** Minimum batch size threshold for SIMD to be worthwhile */
    public static final int MIN_SIMD_BATCH_SIZE = 16;

    /** Default fallback threshold (5% improvement required) */
    public static final double DEFAULT_FALLBACK_THRESHOLD = 1.05;

    /** Preferred vector species for double precision */
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    /**
     * Create optimal configuration with default settings.
     *
     * <p>This is the recommended configuration for most use cases:
     * <ul>
     *   <li>Mini-batch size: 64 patterns</li>
     *   <li>Vector lane count: hardware-specific (SPECIES_PREFERRED)</li>
     *   <li>Auto-tuning: enabled</li>
     *   <li>Fallback threshold: 1.05 (require 5% speedup)</li>
     * </ul>
     *
     * @return Optimal SIMD configuration
     */
    public static SIMDConfiguration optimal() {
        return new SIMDConfiguration(
            DEFAULT_MINI_BATCH_SIZE,
            SPECIES.length(),
            true,
            DEFAULT_FALLBACK_THRESHOLD
        );
    }

    /**
     * Create configuration with specific batch size.
     *
     * <p>Uses default values for all other parameters.
     *
     * @param batchSize Number of patterns per mini-batch
     * @return Configuration with specified batch size
     * @throws IllegalArgumentException if batchSize < MIN_SIMD_BATCH_SIZE
     */
    public static SIMDConfiguration withBatchSize(int batchSize) {
        if (batchSize < MIN_SIMD_BATCH_SIZE) {
            throw new IllegalArgumentException(
                "Batch size %d is too small (minimum: %d)".formatted(batchSize, MIN_SIMD_BATCH_SIZE)
            );
        }
        return new SIMDConfiguration(
            batchSize,
            SPECIES.length(),
            true,
            DEFAULT_FALLBACK_THRESHOLD
        );
    }

    /**
     * Create configuration with auto-tuning explicitly enabled or disabled.
     *
     * @param enableAutoTuning Whether to enable automatic batch size adaptation
     * @return Configuration with specified auto-tuning setting
     */
    public static SIMDConfiguration withAutoTuning(boolean enableAutoTuning) {
        return new SIMDConfiguration(
            DEFAULT_MINI_BATCH_SIZE,
            SPECIES.length(),
            enableAutoTuning,
            DEFAULT_FALLBACK_THRESHOLD
        );
    }

    /**
     * Compact canonical constructor with validation.
     *
     * @throws IllegalArgumentException if configuration is invalid
     */
    public SIMDConfiguration {
        // Validate mini-batch size
        if (miniBatchSize < vectorLaneCount) {
            throw new IllegalArgumentException(
                "miniBatchSize (%d) must be >= vectorLaneCount (%d)"
                .formatted(miniBatchSize, vectorLaneCount)
            );
        }

        // Validate vector lane count
        if (vectorLaneCount < 1 || vectorLaneCount > 16) {
            throw new IllegalArgumentException(
                "vectorLaneCount (%d) must be in range [1, 16]".formatted(vectorLaneCount)
            );
        }

        // Validate fallback threshold
        if (fallbackThreshold < 1.0 || fallbackThreshold > 10.0) {
            throw new IllegalArgumentException(
                "fallbackThreshold (%.2f) must be in range [1.0, 10.0]".formatted(fallbackThreshold)
            );
        }

        // Warn about non-power-of-2 batch sizes (may have alignment issues)
        if (!isPowerOf2(miniBatchSize) && miniBatchSize != 32 && miniBatchSize != 64) {
            // This is advisory only - non-power-of-2 sizes are allowed but may be suboptimal
            // Log at DEBUG level if SLF4J is available
        }
    }

    /**
     * Determine if SIMD should be used for a given batch size.
     *
     * <p>SIMD is recommended when:
     * <ul>
     *   <li>Batch size >= MIN_SIMD_BATCH_SIZE (16 patterns)</li>
     *   <li>Batch size >= miniBatchSize (configured threshold)</li>
     *   <li>Auto-tuning is disabled, or batch is large enough to amortize overhead</li>
     * </ul>
     *
     * @param batchSize Number of patterns to process
     * @return true if SIMD processing is recommended
     */
    public boolean shouldUseSIMD(int batchSize) {
        if (batchSize < MIN_SIMD_BATCH_SIZE) {
            return false;  // Too small for SIMD overhead
        }

        if (!autoTuning) {
            return batchSize >= miniBatchSize;  // Simple threshold when auto-tuning disabled
        }

        // Auto-tuning logic: require at least 2 full mini-batches for worthwhile SIMD
        return batchSize >= (miniBatchSize * 2);
    }

    /**
     * Calculate optimal mini-batch size for a given total batch size.
     *
     * <p>Auto-tuning heuristics:
     * <ul>
     *   <li>Small batches (< 32): use full batch (no mini-batching)</li>
     *   <li>Medium batches (32-127): use 32-pattern mini-batches</li>
     *   <li>Large batches (>= 128): use 64-pattern mini-batches</li>
     * </ul>
     *
     * @param totalBatchSize Total number of patterns to process
     * @return Optimal mini-batch size for this total batch size
     */
    public int getOptimalMiniBatchSize(int totalBatchSize) {
        if (!autoTuning) {
            return miniBatchSize;  // Use configured size when auto-tuning disabled
        }

        if (totalBatchSize < MIN_SIMD_BATCH_SIZE) {
            return totalBatchSize;  // Process entire batch at once (sequential)
        }

        if (totalBatchSize < 32) {
            return totalBatchSize;  // Too small for mini-batching
        }

        if (totalBatchSize < 128) {
            return 32;  // Medium batches: use smaller mini-batches
        }

        return 64;  // Large batches: use full 64-pattern mini-batches
    }

    /**
     * Check if a number is a power of 2.
     *
     * @param n Number to check
     * @return true if n is a power of 2
     */
    private static boolean isPowerOf2(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }

    /**
     * Get current vector lane count from hardware.
     *
     * @return Hardware vector lane count for double precision
     */
    public static int hardwareVectorLaneCount() {
        return SPECIES.length();
    }

    /**
     * Get vector species for SIMD operations.
     *
     * @return Preferred vector species for double precision
     */
    public static VectorSpecies<Double> vectorSpecies() {
        return SPECIES;
    }
}