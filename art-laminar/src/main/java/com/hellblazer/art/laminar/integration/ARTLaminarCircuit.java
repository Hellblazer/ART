package com.hellblazer.art.laminar.integration;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.algorithms.FuzzyART;
import com.hellblazer.art.core.parameters.FuzzyParameters;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.laminar.batch.*;
import com.hellblazer.art.laminar.canonical.CircuitState;
import com.hellblazer.art.laminar.layers.*;
import com.hellblazer.art.laminar.parameters.*;

import java.util.Objects;

/**
 * Hybrid laminar circuit integrating FuzzyART category learning.
 *
 * <p>Combines biological laminar dynamics (Layers 1-6) with ART category
 * learning mechanism. Uses FuzzyART for template management and category
 * search, replacing manual template management in PredictionGenerator.
 *
 * <h2>Key Features</h2>
 * <ul>
 *   <li>Activation-based category ordering (more sophisticated than sequential)</li>
 *   <li>Complement-coded weight vectors via FuzzyART</li>
 *   <li>Vigilance-based resonance detection</li>
 *   <li>Incremental template learning</li>
 *   <li>Batch processing support for 5-10x speedup</li>
 * </ul>
 *
 * <h2>Integration Points</h2>
 * <ul>
 *   <li>Layer 4 → Layer 2/3 processing feeds FuzzyART</li>
 *   <li>FuzzyART weights become top-down expectations</li>
 *   <li>Resonance drives learning and state updates</li>
 * </ul>
 *
 * <h2>Processing Flow</h2>
 * <pre>
 * 1. Bottom-up: Input → Layer 4 → Layer 2/3 (feature processing)
 * 2. Category learning: FuzzyART learns processed input
 * 3. Top-down: Extract expectation from winning category
 * 4. State update: Update circuit state based on resonance
 * </pre>
 *
 * @see FuzzyART
 * @see FullLaminarCircuitImpl
 * @see BatchProcessable
 * @author Hal Hildebrand
 */
public class ARTLaminarCircuit implements AutoCloseable, BatchProcessable {

    private final ARTCircuitParameters params;
    private final FuzzyART artModule;
    private final FuzzyParameters fuzzyParams;

    // Laminar layers (biological dynamics)
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
     * Create ARTLaminarCircuit with specified parameters.
     *
     * <p>Initializes all 6 laminar layers and FuzzyART instance with
     * appropriate parameter mapping.
     *
     * @param params unified parameters for circuit and ART
     * @throws NullPointerException if params is null
     * @throws IllegalArgumentException if parameters are invalid
     */
    public ARTLaminarCircuit(ARTCircuitParameters params) {
        this.params = Objects.requireNonNull(params, "params cannot be null");
        this.fuzzyParams = params.toFuzzyParameters();
        this.artModule = new FuzzyART();

        // Initialize layers
        this.layer4 = new Layer4Implementation("Layer4", params.inputSize());
        this.layer23 = new Layer23Implementation("Layer23", params.inputSize());
        this.layer5 = new Layer5Implementation("Layer5", params.maxCategories());
        this.layer6 = new Layer6Implementation("Layer6", params.inputSize());
        this.layer1 = new Layer1Implementation("Layer1", params.inputSize());

        // Initialize layer parameters
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
     * Process input through hybrid laminar-ART circuit.
     *
     * <p>Processing flow:
     * <ol>
     *   <li>Bottom-up: Input → Layer 4 → Layer 2/3 (feature processing)</li>
     *   <li>Category learning: FuzzyART learns processed input</li>
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

        // Bottom-up processing through laminar layers
        var layer4Output = layer4.processBottomUp(input, layer4Params);
        var layer23Output = layer23.processBottomUp(layer4Output, layer23Params);

        // FuzzyART category learning (this handles activation, vigilance, and learning)
        var artResult = artModule.learn(layer23Output, fuzzyParams);

        // Extract results and update state
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
            // (In full implementation, Layer 5 would process category-specific dynamics)
            layer5.processBottomUp(layer23Output, layer5Params);

            // Update state to resonating
            currentState = CircuitState.resonating(
                categoryId,
                success.activationValue(),
                input,
                modulatedExpectation,
                0  // FuzzyART handles internal search iteration
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
     * @return category count from FuzzyART
     */
    public int getCategoryCount() {
        return artModule.getCategoryCount();
    }

    /**
     * Get expectation for specific category.
     *
     * <p>Extracts the category weight from FuzzyART and converts it
     * from complement-coded form to standard expectation pattern.
     *
     * @param categoryId category identifier
     * @return expectation pattern for this category
     * @throws IndexOutOfBoundsException if category ID invalid
     */
    public Pattern getCategoryExpectation(int categoryId) {
        if (categoryId < 0 || categoryId >= artModule.getCategoryCount()) {
            throw new IndexOutOfBoundsException(
                "Category " + categoryId + " out of bounds for " +
                artModule.getCategoryCount() + " categories"
            );
        }
        var weight = artModule.getCategory(categoryId);
        return LaminarARTBridge.extractExpectation(weight);
    }

    /**
     * Get underlying FuzzyART module for inspection.
     *
     * @return FuzzyART instance
     */
    public FuzzyART getARTModule() {
        return artModule;
    }

    /**
     * Get ART parameters.
     *
     * @return FuzzyART parameters
     */
    public FuzzyParameters getARTParameters() {
        return fuzzyParams;
    }

    /**
     * Reset circuit to initial state.
     *
     * <p>Clears all layers and FuzzyART categories.
     */
    public void reset() {
        layer4.reset();
        layer23.reset();
        layer5.reset();
        layer6.reset();
        layer1.reset();
        artModule.clear();
        currentState = CircuitState.initial(params.inputSize());
    }

    /**
     * Clear resonance state without clearing categories.
     */
    public void clearResonanceState() {
        currentState = CircuitState.initial(params.inputSize());
    }

    // Layer accessors (for biological validation in tests)

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
     * Close and release resources.
     *
     * @throws Exception if resource cleanup fails
     */
    @Override
    public void close() throws Exception {
        // FuzzyART doesn't implement AutoCloseable, but we provide the interface
        // for future-proofing
        reset();
    }

    @Override
    public String toString() {
        return String.format(
            "ARTLaminarCircuit[inputSize=%d, categories=%d, resonating=%s]",
            params.inputSize(),
            getCategoryCount(),
            isResonating()
        );
    }

    // ==================== Batch Processing Implementation ====================

    /**
     * Process batch of patterns with default options.
     * Phase 1 implementation: Sequential processing with overhead amortization.
     *
     * @param patterns array of patterns to process
     * @return batch results
     * @throws IllegalArgumentException if patterns array invalid
     */
    @Override
    public BatchResult processBatch(Pattern[] patterns) {
        return processBatch(patterns, BatchOptions.defaults());
    }

    /**
     * Process batch of patterns with custom options.
     * Phase 1 implementation: Sequential processing with overhead amortization.
     *
     * @param patterns array of patterns to process
     * @param options batch processing configuration
     * @return batch results with per-pattern outputs and statistics
     * @throws IllegalArgumentException if patterns array invalid
     * @throws NullPointerException if options is null
     */
    @Override
    public BatchResult processBatch(Pattern[] patterns, BatchOptions options) {
        validateBatchInput(patterns);
        Objects.requireNonNull(options, "options cannot be null");

        // Phase 1: Sequential batch processing with amortization
        // Future phases will add parallelism and SIMD across batch
        return processBatchSequential(patterns, options);
    }

    /**
     * Check if batch processing will provide speedup.
     *
     * @param batchSize expected batch size
     * @return true if batch processing beneficial
     */
    @Override
    public boolean isBatchProcessingBeneficial(int batchSize) {
        // Batch processing beneficial for:
        // 1. Large batches (>=10 patterns) - amortization benefit
        // 2. High-dimensional patterns (>=64D) - SIMD benefit
        // 3. Many categories (>=20) - parallel search benefit (future phases)
        return batchSize >= 10 ||
               params.inputSize() >= 64 ||
               getCategoryCount() >= 20;
    }

    /**
     * Validate batch input array.
     *
     * @param patterns patterns to validate
     * @throws IllegalArgumentException if invalid
     */
    private void validateBatchInput(Pattern[] patterns) {
        if (patterns == null) {
            throw new IllegalArgumentException("patterns array cannot be null");
        }
        if (patterns.length == 0) {
            throw new IllegalArgumentException("patterns array cannot be empty");
        }

        // Check dimensions and nulls
        for (int i = 0; i < patterns.length; i++) {
            if (patterns[i] == null) {
                throw new IllegalArgumentException("patterns[" + i + "] is null");
            }
            if (patterns[i].dimension() != params.inputSize()) {
                throw new IllegalArgumentException(
                    String.format("patterns[%d] dimension %d != expected %d",
                        i, patterns[i].dimension(), params.inputSize()));
            }
        }
    }

    /**
     * Process batch using best available strategy based on options.
     * - Phase 1: Sequential with amortization
     * - Phase 6A: Stateful batch with layer-level SIMD
     *
     * @param patterns patterns to process
     * @param options batch options
     * @return batch results
     */
    private BatchResult processBatchSequential(Pattern[] patterns, BatchOptions options) {
        // Phase 6A: Stateful batch processing enabled
        // - Circuit level: Sequential pattern processing (preserves state evolution)
        // - Layer level: SIMD optimization per pattern (computational efficiency)
        // Expected 1.4-1.5x speedup with 0.00e+00 semantic equivalence

        return processBatchPhase6A(patterns, options);
    }

    /**
     * Phase 1: Sequential batch processing with overhead amortization.
     * Processes patterns one by one but with reduced per-pattern overhead.
     *
     * @param patterns patterns to process
     * @param options batch options
     * @return batch results
     */
    private BatchResult processBatchPhase1(Pattern[] patterns, BatchOptions options) {
        var batchSize = patterns.length;
        var outputs = new Pattern[batchSize];
        var categoryIds = new int[batchSize];
        var activationValues = new double[batchSize];
        var resonating = new boolean[batchSize];

        // Track category count before batch
        var initialCategoryCount = getCategoryCount();

        // Timing tracking
        long startTime = System.nanoTime();
        long layer4Time = 0, layer23Time = 0, layer5Time = 0, layer6Time = 0, artTime = 0;

        // Process each pattern with reduced overhead
        for (int i = 0; i < batchSize; i++) {
            var pattern = patterns[i];

            // Layer 4 processing
            long t0 = options.trackDetailedStats() ? System.nanoTime() : 0;
            var layer4Output = layer4.processBottomUp(pattern, layer4Params);
            if (options.trackDetailedStats()) layer4Time += System.nanoTime() - t0;

            // Layer 2/3 processing
            long t1 = options.trackDetailedStats() ? System.nanoTime() : 0;
            var layer23Output = layer23.processBottomUp(layer4Output, layer23Params);
            if (options.trackDetailedStats()) layer23Time += System.nanoTime() - t1;

            // ART category learning
            long t2 = options.trackDetailedStats() ? System.nanoTime() : 0;
            var artResult = artModule.learn(layer23Output, fuzzyParams);
            if (options.trackDetailedStats()) artTime += System.nanoTime() - t2;

            // Process result
            if (artResult instanceof ActivationResult.Success success) {
                var categoryId = success.categoryIndex();
                var weight = success.updatedWeight();
                var expectation = LaminarARTBridge.extractExpectation(weight);
                var modulatedExpectation = expectation.scale(params.topDownGain());

                // Layer 6 top-down
                long t3 = options.trackDetailedStats() ? System.nanoTime() : 0;
                layer6.processTopDown(modulatedExpectation, layer6Params);
                if (options.trackDetailedStats()) layer6Time += System.nanoTime() - t3;

                // Layer 5 processing
                long t4 = options.trackDetailedStats() ? System.nanoTime() : 0;
                layer5.processBottomUp(layer23Output, layer5Params);
                if (options.trackDetailedStats()) layer5Time += System.nanoTime() - t4;

                outputs[i] = modulatedExpectation;
                categoryIds[i] = categoryId;
                activationValues[i] = success.activationValue();
                resonating[i] = true;
            } else {
                outputs[i] = layer23Output;
                categoryIds[i] = -1;
                activationValues[i] = 0.0;
                resonating[i] = false;
            }
        }

        long totalTime = System.nanoTime() - startTime;

        // Calculate statistics
        var categoriesCreated = getCategoryCount() - initialCategoryCount;
        var statistics = new BatchStatistics(
            batchSize,
            totalTime,
            layer4Time,
            layer23Time,
            layer5Time,
            layer6Time,
            artTime,
            categoriesCreated,
            batchSize,  // One search per pattern
            0,          // SIMD ops not tracked in Phase 1
            0,          // No parallelism in Phase 1
            -1.0        // Cache not tracked in Phase 1
        );

        return new BatchResult(outputs, categoryIds, activationValues, resonating, statistics);
    }

    /**
     * Phase 6A: Stateful batch processing with mini-batch SIMD optimization.
     *
     * <p>This approach combines:
     * <ul>
     *   <li>Mini-batch processing through layers (SIMD efficiency)</li>
     *   <li>Sequential ART learning (state evolution)</li>
     * </ul>
     *
     * <p>Patterns are processed in mini-batches of 4 through laminar layers (Layer 4 and 2/3),
     * which gives SIMD real batches to optimize. Then each pattern is processed sequentially
     * through ART learning to maintain category state accumulation.
     *
     * <p>Expected performance: 1.3-1.5x speedup over baseline with 0.00e+00 semantic equivalence.
     *
     * @param patterns patterns to process
     * @param options batch options
     * @return batch results
     */
    private BatchResult processBatchPhase6A(Pattern[] patterns, BatchOptions options) {
        var batchSize = patterns.length;
        var outputs = new Pattern[batchSize];
        var categoryIds = new int[batchSize];
        var activationValues = new double[batchSize];
        var resonating = new boolean[batchSize];

        // Track category count before batch
        var initialCategoryCount = getCategoryCount();

        // Timing tracking
        long startTime = System.nanoTime();
        long layer4Time = 0, layer23Time = 0, layer5Time = 0, layer6Time = 0, artTime = 0;

        // Mini-batch size for SIMD efficiency (32 patterns per mini-batch)
        // Must be >= 32 to pass BatchDataLayout.isTransposeAndVectorizeBeneficial() threshold
        final int miniBatchSize = 32;

        // Process patterns in mini-batches for SIMD efficiency
        for (int miniBatchStart = 0; miniBatchStart < batchSize; miniBatchStart += miniBatchSize) {
            // Determine actual mini-batch size (may be smaller for last batch)
            int actualMiniBatchSize = Math.min(miniBatchSize, batchSize - miniBatchStart);

            // Extract mini-batch
            var miniBatch = new Pattern[actualMiniBatchSize];
            System.arraycopy(patterns, miniBatchStart, miniBatch, 0, actualMiniBatchSize);

            // Layer 4 mini-batch processing (direct SIMD call)
            long t0 = options.trackDetailedStats() ? System.nanoTime() : 0;
            var layer4Outputs = Layer4SIMDBatch.processBatchSIMD(miniBatch, layer4Params, params.inputSize());
            if (layer4Outputs == null) {
                // SIMD not beneficial - fall back to sequential
                layer4Outputs = new Pattern[actualMiniBatchSize];
                for (int i = 0; i < actualMiniBatchSize; i++) {
                    layer4Outputs[i] = layer4.processBottomUp(miniBatch[i], layer4Params);
                }
            }
            if (options.trackDetailedStats()) layer4Time += System.nanoTime() - t0;

            // Layer 2/3 mini-batch processing (direct SIMD call)
            // No top-down priming in mini-batch path for now (Layer 1 processing is sequential)
            long t1 = options.trackDetailedStats() ? System.nanoTime() : 0;
            var layer23Outputs = Layer23SIMDBatch.processBatchSIMD(layer4Outputs, null, layer23Params, params.inputSize());
            if (layer23Outputs == null) {
                // SIMD not beneficial (or bipole network enabled) - fall back to sequential
                layer23Outputs = new Pattern[actualMiniBatchSize];
                for (int i = 0; i < actualMiniBatchSize; i++) {
                    layer23Outputs[i] = layer23.processBottomUp(layer4Outputs[i], layer23Params);
                }
            }
            if (options.trackDetailedStats()) layer23Time += System.nanoTime() - t1;

            // ART learning: Sequential to maintain category state
            for (int i = 0; i < actualMiniBatchSize; i++) {
                int globalIdx = miniBatchStart + i;
                var layer23Output = layer23Outputs[i];

                // ART category learning
                long t2 = options.trackDetailedStats() ? System.nanoTime() : 0;
                var artResult = artModule.learn(layer23Output, fuzzyParams);
                if (options.trackDetailedStats()) artTime += System.nanoTime() - t2;

                // Process result
                if (artResult instanceof ActivationResult.Success success) {
                    var categoryId = success.categoryIndex();
                    var weight = success.updatedWeight();
                    var expectation = LaminarARTBridge.extractExpectation(weight);
                    var modulatedExpectation = expectation.scale(params.topDownGain());

                    // Layer 6 top-down
                    long t3 = options.trackDetailedStats() ? System.nanoTime() : 0;
                    layer6.processTopDown(modulatedExpectation, layer6Params);
                    if (options.trackDetailedStats()) layer6Time += System.nanoTime() - t3;

                    // Layer 5 processing
                    long t4 = options.trackDetailedStats() ? System.nanoTime() : 0;
                    layer5.processBottomUp(layer23Output, layer5Params);
                    if (options.trackDetailedStats()) layer5Time += System.nanoTime() - t4;

                    outputs[globalIdx] = modulatedExpectation;
                    categoryIds[globalIdx] = categoryId;
                    activationValues[globalIdx] = success.activationValue();
                    resonating[globalIdx] = true;
                } else {
                    outputs[globalIdx] = layer23Output;
                    categoryIds[globalIdx] = -1;
                    activationValues[globalIdx] = 0.0;
                    resonating[globalIdx] = false;
                }
            }
        }


        long totalTime = System.nanoTime() - startTime;

        // Calculate statistics
        var categoriesCreated = getCategoryCount() - initialCategoryCount;
        var statistics = new BatchStatistics(
            batchSize,
            totalTime,
            layer4Time,
            layer23Time,
            layer5Time,
            layer6Time,
            artTime,
            categoriesCreated,
            batchSize,  // One search per pattern
            0,          // SIMD ops tracked per layer
            0,          // No parallelism in Phase 6A
            -1.0        // Cache not tracked
        );

        return new BatchResult(outputs, categoryIds, activationValues, resonating, statistics);
    }

    /**
     * Phase 2: Batch processing with layer-level batching.
     * Processes entire batch through each layer at once for better performance.
     *
     * @param patterns patterns to process
     * @param options batch options
     * @return batch results
     */
    @Deprecated
    private BatchResult processBatchWithLayerBatching(Pattern[] patterns, BatchOptions options) {
        var batchSize = patterns.length;
        var outputs = new Pattern[batchSize];
        var categoryIds = new int[batchSize];
        var activationValues = new double[batchSize];
        var resonating = new boolean[batchSize];

        // Track category count before batch
        var initialCategoryCount = getCategoryCount();

        // Timing tracking
        long startTime = System.nanoTime();
        long layer4Time = 0, layer23Time = 0, layer5Time = 0, layer6Time = 0, artTime = 0;

        // Phase 2: Batch layer processing
        var batchLayer = (com.hellblazer.art.laminar.batch.BatchLayer) layer4;

        // Layer 4: Process entire batch at once
        long t0 = options.trackDetailedStats() ? System.nanoTime() : 0;
        var layer4Outputs = batchLayer.processBatchBottomUp(patterns, layer4Params);
        if (options.trackDetailedStats()) layer4Time = System.nanoTime() - t0;

        // Layer 2/3: Process batch (if supports batching)
        long t1 = options.trackDetailedStats() ? System.nanoTime() : 0;
        Pattern[] layer23Outputs;
        if (layer23 instanceof com.hellblazer.art.laminar.batch.BatchLayer batchLayer23) {
            layer23Outputs = batchLayer23.processBatchBottomUp(layer4Outputs, layer23Params);
        } else {
            // Fallback to sequential
            layer23Outputs = new Pattern[batchSize];
            for (int i = 0; i < batchSize; i++) {
                layer23Outputs[i] = layer23.processBottomUp(layer4Outputs[i], layer23Params);
            }
        }
        if (options.trackDetailedStats()) layer23Time = System.nanoTime() - t1;

        // ART processing: Still sequential (no batch ART API yet)
        long t2 = options.trackDetailedStats() ? System.nanoTime() : 0;
        for (int i = 0; i < batchSize; i++) {
            var artResult = artModule.learn(layer23Outputs[i], fuzzyParams);

            if (artResult instanceof ActivationResult.Success success) {
                var categoryId = success.categoryIndex();
                var weight = success.updatedWeight();
                var expectation = LaminarARTBridge.extractExpectation(weight);
                var modulatedExpectation = expectation.scale(params.topDownGain());

                // Layer 6/5 top-down (sequential for now)
                layer6.processTopDown(modulatedExpectation, layer6Params);
                layer5.processBottomUp(layer23Outputs[i], layer5Params);

                outputs[i] = modulatedExpectation;
                categoryIds[i] = categoryId;
                activationValues[i] = success.activationValue();
                resonating[i] = true;
            } else {
                outputs[i] = layer23Outputs[i];
                categoryIds[i] = -1;
                activationValues[i] = 0.0;
                resonating[i] = false;
            }
        }
        if (options.trackDetailedStats()) artTime = System.nanoTime() - t2;

        long totalTime = System.nanoTime() - startTime;

        // Calculate statistics
        var categoriesCreated = getCategoryCount() - initialCategoryCount;
        var statistics = new BatchStatistics(
            batchSize,
            totalTime,
            layer4Time,
            layer23Time,
            layer5Time,
            layer6Time,
            artTime,
            categoriesCreated,
            batchSize,  // One search per pattern
            0,          // SIMD ops tracked in future phases
            0,          // Parallel tasks tracked in future phases
            -1.0        // Cache not tracked yet
        );

        return new BatchResult(outputs, categoryIds, activationValues, resonating, statistics);
    }
}
