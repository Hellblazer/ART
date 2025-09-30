package com.hellblazer.art.laminar.batch;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.LayerParameters;

/**
 * Interface for layers that support stateful batch processing.
 *
 * <h2>Stateful Batch Processing</h2>
 *
 * Stateful batch processing combines:
 * <ul>
 *   <li><b>Sequential pattern processing</b> - Patterns processed one at a time to preserve state evolution</li>
 *   <li><b>Layer-level SIMD optimization</b> - Internal layer computation uses SIMD where beneficial</li>
 * </ul>
 *
 * <h3>Motivation</h3>
 *
 * <p>ART laminar circuits have state-dependent learning where Pattern N+1 must see
 * the effects of Pattern N (attention state, category learning, activation history, etc.).
 * This requires sequential pattern processing through the circuit.
 *
 * <p>However, within each pattern's processing, individual layers can use SIMD to optimize
 * their internal computations (shunting dynamics, competitive selection, etc.) without
 * breaking semantic equivalence.
 *
 * <h3>Example</h3>
 *
 * <pre>
 * // Circuit-level: Sequential pattern processing
 * for (var pattern : patterns) {
 *     // Layer-level: SIMD-optimized internal computation
 *     var output = layer.processWithStatefulSIMD(pattern, params);
 *     // State updated, ready for next pattern
 * }
 * </pre>
 *
 * <h3>State Evolution</h3>
 *
 * <pre>
 * Pattern 0: [Initial State]   → Process with SIMD → [State After 0] → Output 0
 * Pattern 1: [State After 0]    → Process with SIMD → [State After 1] → Output 1
 * Pattern 2: [State After 1]    → Process with SIMD → [State After 2] → Output 2
 * </pre>
 *
 * Each pattern sees and modifies the layer state, maintaining semantic equivalence
 * with fully sequential processing while gaining SIMD benefits.
 *
 * <h3>Performance</h3>
 *
 * Expected speedup from Phase 5 individual layer SIMD results:
 * <ul>
 *   <li>Layer 4: 1.5x (shunting dynamics)</li>
 *   <li>Layer 5: 1.3x (burst firing)</li>
 *   <li>Layer 6: 1.4x (ART matching)</li>
 *   <li>Overall circuit: 1.4-1.5x throughput improvement</li>
 * </ul>
 *
 * @author Hal Hildebrand
 */
public interface StatefulBatchProcessor {

    /**
     * Process single pattern with SIMD-optimized layer-internal computation.
     *
     * <p>This method:
     * <ol>
     *   <li>Uses SIMD for layer-internal operations (shunting dynamics, competitive selection, etc.)</li>
     *   <li>Updates layer state based on pattern result</li>
     *   <li>Returns processed pattern</li>
     * </ol>
     *
     * <p>State evolution is preserved because patterns are processed sequentially
     * through the circuit. SIMD is used only for the internal computation within
     * each pattern's processing.
     *
     * <h4>Semantic Equivalence</h4>
     *
     * Must produce identical results to sequential scalar processing:
     * <pre>
     * // These must produce identical outputs and state evolution:
     * output1 = layer.processBottomUp(pattern, params);        // Scalar
     * output2 = layer.processWithStatefulSIMD(pattern, params); // SIMD
     * assertEquals(output1, output2, 0.0);  // Bit-exact equivalence
     * </pre>
     *
     * @param input Input pattern to process
     * @param parameters Layer-specific parameters
     * @return Processed pattern with layer state updated
     */
    Pattern processWithStatefulSIMD(Pattern input, LayerParameters parameters);

    /**
     * Check if stateful SIMD is beneficial for this pattern.
     *
     * <p>SIMD optimization has overhead (single-pattern batch creation, state management).
     * For small patterns or simple operations, scalar processing may be faster.
     *
     * <p>Typical heuristics:
     * <ul>
     *   <li>Pattern dimension >= 64: SIMD beneficial</li>
     *   <li>Pattern dimension < 64: Scalar faster</li>
     * </ul>
     *
     * <p>Layers can override with more sophisticated cost models based on:
     * <ul>
     *   <li>Pattern dimension</li>
     *   <li>Layer complexity (number of operations)</li>
     *   <li>Current state size</li>
     * </ul>
     *
     * @param pattern Pattern to process
     * @return true if SIMD provides benefit, false to fall back to scalar
     */
    default boolean isStatefulSIMDBeneficial(Pattern pattern) {
        // Default: SIMD beneficial for dimension >= 64
        return pattern.dimension() >= 64;
    }
}
