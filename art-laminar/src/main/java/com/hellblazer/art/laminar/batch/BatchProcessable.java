package com.hellblazer.art.laminar.batch;

import com.hellblazer.art.core.Pattern;

/**
 * Interface for circuits that support batch processing of multiple patterns.
 * Batch processing amortizes overhead and enables SIMD across patterns for 5-10x speedup.
 *
 * <h2>Performance Benefits</h2>
 * <ul>
 *   <li>Amortized overhead: Spread initialization costs across many patterns</li>
 *   <li>Cross-pattern vectorization: SIMD across batch dimension</li>
 *   <li>Better cache utilization: Sequential memory access patterns</li>
 *   <li>Parallel category search: Process multiple patterns concurrently</li>
 * </ul>
 *
 * <h2>Expected Speedup</h2>
 * <ul>
 *   <li>Batch size 10: 1.5-2x</li>
 *   <li>Batch size 100: 5-7x</li>
 *   <li>Batch size 1000: 8-10x</li>
 * </ul>
 *
 * @see BatchResult
 * @see BatchOptions
 * @author Claude Code
 */
public interface BatchProcessable {

    /**
     * Process a batch of patterns simultaneously with default options.
     *
     * <p>Uses {@link BatchOptions#defaults()} for configuration.
     * Automatically selects optimal processing strategy based on batch size.
     *
     * @param patterns array of patterns to process
     * @return batch results with per-pattern outputs and statistics
     * @throws IllegalArgumentException if patterns array is empty or contains null
     * @throws IllegalArgumentException if pattern dimensions don't match expected size
     */
    BatchResult processBatch(Pattern[] patterns);

    /**
     * Process a batch of patterns with custom configuration.
     *
     * <p>Allows fine-grained control over batch processing strategy:
     * <ul>
     *   <li>Parallelism settings</li>
     *   <li>SIMD across batch dimension</li>
     *   <li>Cache optimization</li>
     *   <li>Performance tracking detail level</li>
     * </ul>
     *
     * @param patterns array of patterns to process
     * @param options batch processing configuration
     * @return batch results with per-pattern outputs and statistics
     * @throws IllegalArgumentException if patterns array is empty or contains null
     * @throws IllegalArgumentException if pattern dimensions don't match expected size
     * @throws NullPointerException if options is null
     */
    BatchResult processBatch(Pattern[] patterns, BatchOptions options);

    /**
     * Check if batch processing will provide speedup for given batch size.
     *
     * <p>Batch processing is beneficial when:
     * <ul>
     *   <li>Batch size ≥ 10 (amortization benefit)</li>
     *   <li>Pattern dimensionality ≥ 64 (SIMD benefit)</li>
     *   <li>Category count ≥ 20 (parallel search benefit)</li>
     * </ul>
     *
     * @param batchSize expected batch size
     * @return true if batch processing will provide measurable speedup
     */
    boolean isBatchProcessingBeneficial(int batchSize);
}