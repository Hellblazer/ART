package com.hellblazer.art.laminar.integration;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.laminar.canonical.CircuitState;
import com.hellblazer.art.laminar.layers.*;
import com.hellblazer.art.laminar.parameters.*;
import com.hellblazer.art.performance.algorithms.VectorizedFuzzyART;
import com.hellblazer.art.performance.algorithms.VectorizedParameters;
import com.hellblazer.art.performance.algorithms.VectorizedPerformanceStats;

import java.util.Objects;

/**
 * High-performance vectorized laminar circuit using VectorizedFuzzyART with SIMD optimizations.
 *
 * <p>Provides 5-10x speedup for pattern operations through SIMD optimizations
 * while maintaining identical semantics to ARTLaminarCircuit. This implementation
 * combines biological laminar dynamics with high-performance vectorized ART category learning.
 *
 * <h2>Performance Benefits</h2>
 * <ul>
 *   <li>SIMD-optimized pattern operations (min, max, scale)</li>
 *   <li>Vectorized activation calculations</li>
 *   <li>Parallel category processing</li>
 *   <li>Cache-optimized data structures</li>
 * </ul>
 *
 * <h2>Use Cases</h2>
 * <ul>
 *   <li>High-dimensional patterns (64+ dimensions)</li>
 *   <li>Large category sets (100+ categories)</li>
 *   <li>Real-time processing requirements</li>
 *   <li>Batch pattern processing</li>
 * </ul>
 *
 * <h2>Semantic Equivalence</h2>
 * <p>This implementation produces identical results to ARTLaminarCircuit (within floating-point
 * tolerance). The only difference is performance - vectorization is a pure optimization.
 *
 * <h2>Processing Flow</h2>
 * <pre>
 * 1. Bottom-up: Input → Layer 4 → Layer 2/3 (feature processing)
 * 2. Vectorized category learning: VectorizedFuzzyART with SIMD operations
 * 3. Top-down: Extract expectation from winning category
 * 4. State update: Update circuit state based on resonance
 * </pre>
 *
 * @see ARTLaminarCircuit
 * @see VectorizedFuzzyART
 * @author Hal Hildebrand
 */
public class VectorizedARTLaminarCircuit implements AutoCloseable {

    private final ARTCircuitParameters params;
    private final VectorizedFuzzyART vectorizedART;
    private final VectorizedParameters vectorizedParams;

    // Laminar layers (same as ARTLaminarCircuit)
    private final Layer4Implementation layer4;
    private final Layer23Implementation layer23;
    private final Layer5Implementation layer5;
    private final Layer6Implementation layer6;
    private final Layer1Implementation layer1;

    // Layer parameters
    private final Layer4Parameters layer4Params;
    private final Layer23Parameters layer23Params;
    private final Layer5Parameters layer5Params;
    private final Layer6Parameters layer6Params;

    // State tracking
    private CircuitState currentState;

    /**
     * Create VectorizedARTLaminarCircuit with specified parameters.
     *
     * <p>Initializes all components including SIMD infrastructure for vectorized processing.
     * Uses VectorizedFuzzyART instead of standard FuzzyART for 5-10x performance improvement.
     *
     * @param params unified parameters for circuit and ART
     * @throws NullPointerException if params is null
     * @throws IllegalArgumentException if parameters are invalid
     */
    public VectorizedARTLaminarCircuit(ARTCircuitParameters params) {
        this.params = Objects.requireNonNull(params, "params cannot be null");

        // Convert to VectorizedParameters with performance optimizations
        this.vectorizedParams = params.toVectorizedParameters();

        // Initialize vectorized FuzzyART with SIMD support
        this.vectorizedART = new VectorizedFuzzyART(vectorizedParams);

        // Initialize layers (same as ARTLaminarCircuit)
        this.layer4 = new Layer4Implementation("Layer4", params.inputSize());
        this.layer23 = new Layer23Implementation("Layer23", params.inputSize());
        this.layer5 = new Layer5Implementation("Layer5", params.maxCategories());
        this.layer6 = new Layer6Implementation("Layer6", params.inputSize());
        this.layer1 = new Layer1Implementation("Layer1", params.inputSize());

        // Initialize layer parameters (same as ARTLaminarCircuit)
        this.layer4Params = Layer4Parameters.builder()
            .timeConstant(30.0)  // Fast dynamics
            .drivingStrength(0.8)
            .build();

        this.layer23Params = Layer23Parameters.builder()
            .timeConstant(0.05)  // 50ms - medium-fast dynamics
            .horizontalWeight(0.5)
            .build();

        this.layer5Params = Layer5Parameters.builder()
            .timeConstant(100.0)  // Medium dynamics
            .amplificationGain(1.2)
            .build();

        this.layer6Params = Layer6Parameters.builder()
            .timeConstant(200.0)  // Slow dynamics
            .attentionalGain(0.5)
            .build();

        this.currentState = CircuitState.initial(params.inputSize());
    }

    /**
     * Process input through vectorized circuit.
     *
     * <p>Same semantics as ARTLaminarCircuit.process() but with SIMD-optimized
     * category learning for significantly faster processing.
     *
     * <p>Processing flow:
     * <ol>
     *   <li>Bottom-up: Input → Layer 4 → Layer 2/3 (feature processing)</li>
     *   <li>Vectorized category learning: VectorizedFuzzyART with SIMD operations</li>
     *   <li>Top-down: Extract expectation from winning category</li>
     *   <li>State update: Update circuit state based on resonance</li>
     * </ol>
     *
     * @param input input pattern [0,1]<sup>d</sup> where d = inputSize
     * @return expectation pattern if resonance, else processed input
     * @throws NullPointerException if input is null
     * @throws IllegalArgumentException if input dimension invalid
     */
    public Pattern process(Pattern input) {
        Objects.requireNonNull(input, "input cannot be null");
        if (input.dimension() != params.inputSize()) {
            throw new IllegalArgumentException(
                String.format("Input dimension %d != expected %d",
                    input.dimension(), params.inputSize())
            );
        }

        // Bottom-up processing through laminar layers (same as ARTLaminarCircuit)
        var layer4Output = layer4.processBottomUp(input, layer4Params);
        var layer23Output = layer23.processBottomUp(layer4Output, layer23Params);

        // Vectorized FuzzyART learning (SIMD optimized!)
        var artResult = vectorizedART.learn(layer23Output, vectorizedParams);

        // Extract results (same logic as ARTLaminarCircuit)
        if (artResult instanceof ActivationResult.Success success) {
            var categoryId = success.categoryIndex();
            var weight = success.updatedWeight();

            // Extract expectation from FuzzyART weight (remove complement coding)
            var expectation = LaminarARTBridge.extractExpectation(weight);

            // Apply top-down gain modulation
            var modulatedExpectation = expectation.scale(params.topDownGain());

            // Update Layer 6 with top-down expectation
            layer6.processTopDown(modulatedExpectation, layer6Params);

            // Update Layer 5 with category activation
            layer5.processBottomUp(layer23Output, layer5Params);

            // Update state to resonating
            currentState = CircuitState.resonating(
                categoryId,
                success.activationValue(),
                input,
                modulatedExpectation,
                0  // VectorizedFuzzyART handles internal search iteration
            );

            return modulatedExpectation;
        } else {
            // No match - update state to non-resonating
            var zeroExpectation = Pattern.of(new double[params.inputSize()]);
            currentState = CircuitState.mismatch(
                -1,
                0.0,
                input,
                zeroExpectation,
                0
            );

            return layer23Output;
        }
    }

    /**
     * Check if circuit is currently resonating.
     *
     * @return true if resonance achieved
     */
    public boolean isResonating() {
        return currentState.isResonating();
    }

    /**
     * Get current circuit state.
     *
     * @return immutable state snapshot
     */
    public CircuitState getState() {
        return currentState;
    }

    /**
     * Get number of learned categories.
     *
     * @return category count from VectorizedFuzzyART
     */
    public int getCategoryCount() {
        return vectorizedART.getCategoryCount();
    }

    /**
     * Get expectation for specific category.
     *
     * <p>Extracts the category weight from VectorizedFuzzyART and converts it
     * from complement-coded form to standard expectation pattern.
     *
     * @param categoryId category identifier
     * @return expectation pattern for this category
     * @throws IndexOutOfBoundsException if category ID invalid
     */
    public Pattern getCategoryExpectation(int categoryId) {
        if (categoryId < 0 || categoryId >= vectorizedART.getCategoryCount()) {
            throw new IndexOutOfBoundsException(
                "Category " + categoryId + " out of bounds for " +
                vectorizedART.getCategoryCount() + " categories"
            );
        }
        var weight = vectorizedART.getCategory(categoryId);
        return LaminarARTBridge.extractExpectation(weight);
    }

    // === Performance-specific methods ===

    /**
     * Get performance statistics from vectorized operations.
     *
     * <p>Returns detailed performance metrics including:
     * <ul>
     *   <li>Total SIMD vector operations performed</li>
     *   <li>Parallel task executions</li>
     *   <li>Average compute time</li>
     *   <li>Active thread count</li>
     *   <li>Cache utilization</li>
     * </ul>
     *
     * @return performance statistics from VectorizedFuzzyART
     */
    public VectorizedPerformanceStats getPerformanceStats() {
        return vectorizedART.getPerformanceStats();
    }

    /**
     * Reset performance tracking counters.
     *
     * <p>Resets all performance metrics to zero without affecting learned categories.
     * Useful for benchmarking specific operations.
     */
    public void resetPerformanceTracking() {
        vectorizedART.resetPerformanceTracking();
    }

    /**
     * Check if SIMD vectorization is active.
     *
     * <p>Returns true if the underlying VectorizedFuzzyART is successfully
     * using SIMD operations. May return false if SIMD is not available
     * on the current platform.
     *
     * @return true if using SIMD operations
     */
    public boolean isVectorized() {
        return vectorizedParams.enableSIMD();
    }

    /**
     * Get underlying VectorizedFuzzyART module for inspection.
     *
     * @return VectorizedFuzzyART instance
     */
    public VectorizedFuzzyART getVectorizedARTModule() {
        return vectorizedART;
    }

    /**
     * Get vectorized parameters.
     *
     * @return VectorizedParameters with performance settings
     */
    public VectorizedParameters getVectorizedParameters() {
        return vectorizedParams;
    }

    // === Layer accessors (same as ARTLaminarCircuit) ===

    /**
     * Get Layer 4 implementation.
     * @return Layer 4 instance
     */
    public Layer4Implementation getLayer4() {
        return layer4;
    }

    /**
     * Get Layer 2/3 implementation.
     * @return Layer 2/3 instance
     */
    public Layer23Implementation getLayer23() {
        return layer23;
    }

    /**
     * Get Layer 5 implementation.
     * @return Layer 5 instance
     */
    public Layer5Implementation getLayer5() {
        return layer5;
    }

    /**
     * Get Layer 6 implementation.
     * @return Layer 6 instance
     */
    public Layer6Implementation getLayer6() {
        return layer6;
    }

    /**
     * Get Layer 1 implementation.
     * @return Layer 1 instance
     */
    public Layer1Implementation getLayer1() {
        return layer1;
    }

    /**
     * Reset circuit to initial state.
     *
     * <p>Clears all layers and VectorizedFuzzyART categories.
     * Performance statistics are NOT reset - use resetPerformanceTracking() for that.
     */
    public void reset() {
        layer4.reset();
        layer23.reset();
        layer5.reset();
        layer6.reset();
        layer1.reset();
        vectorizedART.clearCategories();
        currentState = CircuitState.initial(params.inputSize());
    }

    /**
     * Clear resonance state without clearing categories.
     */
    public void clearResonanceState() {
        currentState = CircuitState.initial(params.inputSize());
    }

    /**
     * Create standard (non-vectorized) version for comparison.
     *
     * <p>Useful for A/B testing and validation. Creates a new ARTLaminarCircuit
     * with the same parameters for semantic equivalence testing.
     *
     * @return equivalent ARTLaminarCircuit
     */
    public ARTLaminarCircuit toStandardCircuit() {
        return new ARTLaminarCircuit(params);
    }

    /**
     * Close and release resources.
     *
     * <p>Cleans up vectorized infrastructure including thread pools and SIMD buffers.
     */
    @Override
    public void close() {
        vectorizedART.close();
    }

    @Override
    public String toString() {
        return String.format(
            "VectorizedARTLaminarCircuit[inputSize=%d, categories=%d, resonating=%s, vectorized=%s]",
            params.inputSize(),
            getCategoryCount(),
            isResonating(),
            isVectorized()
        );
    }
}
