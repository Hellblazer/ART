package com.hellblazer.art.laminar.batch;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.LayerParameters;

/**
 * Interface for layers that support batch processing.
 * Enables efficient processing of multiple patterns through a layer simultaneously.
 *
 * <h2>Performance Benefits</h2>
 * <ul>
 *   <li>Reduced per-pattern overhead</li>
 *   <li>Better memory locality (sequential access)</li>
 *   <li>Potential for SIMD across batch dimension</li>
 *   <li>Amortized layer state setup</li>
 * </ul>
 *
 * <h2>Implementation Strategy</h2>
 * <p>Layers should implement batch processing by:
 * <ol>
 *   <li>Validating batch inputs</li>
 *   <li>Pre-allocating output arrays</li>
 *   <li>Processing patterns with shared layer state</li>
 *   <li>Minimizing intermediate allocations</li>
 * </ol>
 *
 * <h2>Semantic Equivalence</h2>
 * <p>Batch processing must produce identical results to sequential single-pattern
 * processing. The only difference should be performance.
 *
 * @author Claude Code
 */
public interface BatchLayer {

    /**
     * Process a batch of patterns through this layer.
     *
     * <p>Processes all patterns with the same layer parameters,
     * producing one output pattern per input pattern.
     *
     * @param inputs batch of input patterns [batchSize][dimension]
     * @param params layer parameters (shared across batch)
     * @return batch of output patterns
     * @throws IllegalArgumentException if inputs array empty or contains null
     * @throws IllegalArgumentException if pattern dimensions don't match layer size
     * @throws NullPointerException if params is null
     */
    Pattern[] processBatchBottomUp(Pattern[] inputs, LayerParameters params);

    /**
     * Process batch with top-down expectations.
     *
     * <p>Only applicable for layers that support top-down processing (e.g., Layer 6).
     * Other layers may throw UnsupportedOperationException.
     *
     * @param expectations batch of top-down expectation patterns
     * @param params layer parameters
     * @return batch of output patterns
     * @throws UnsupportedOperationException if layer doesn't support top-down
     * @throws IllegalArgumentException if inputs invalid
     * @throws NullPointerException if params is null
     */
    default Pattern[] processBatchTopDown(Pattern[] expectations, LayerParameters params) {
        throw new UnsupportedOperationException(
            "Layer " + getClass().getSimpleName() + " does not support batch top-down processing");
    }

    /**
     * Check if batch processing is beneficial for given batch size.
     *
     * <p>For most layers, batch processing is beneficial when:
     * <ul>
     *   <li>Batch size >= 10 (amortization benefit)</li>
     *   <li>Pattern dimension >= 64 (SIMD benefit)</li>
     * </ul>
     *
     * @param batchSize expected batch size
     * @return true if batch processing will provide speedup
     */
    default boolean isBatchProcessingBeneficial(int batchSize) {
        return batchSize >= 10;
    }

    /**
     * Get layer dimension (size).
     *
     * @return number of units in layer
     */
    int getSize();

    /**
     * Get layer name/identifier.
     *
     * @return layer identifier
     */
    String getId();
}