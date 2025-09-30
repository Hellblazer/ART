package com.hellblazer.art.laminar.canonical;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.layers.*;
import com.hellblazer.art.laminar.parameters.*;

import java.util.HashSet;
import java.util.Set;

/**
 * Complete implementation of canonical laminar circuit with all 6 layers and resonance dynamics.
 *
 * Implements the full ART processing cycle in the canonical neocortical circuit:
 *
 * <h2>Processing Flow</h2>
 * <pre>
 * 1. Bottom-Up Phase:
 *    Input → Layer 4 → Layer 2/3 → Layer 5 → Layer 6
 *
 * 2. Top-Down Phase:
 *    Layer 6 → PredictionGenerator → Expectation → Layer 4
 *
 * 3. Resonance Check:
 *    IF matchScore >= vigilance:
 *      RESONANCE (stable match, learning occurs)
 *    ELSE:
 *      MISMATCH RESET (suppress category, search next)
 *
 * 4. Category Search:
 *    Try categories in activation order until match or max iterations
 * </pre>
 *
 * <h2>Key ART Dynamics</h2>
 * <ul>
 *   <li><b>Resonance</b>: Stable match between bottom-up and top-down</li>
 *   <li><b>Mismatch Reset</b>: Suppresses category, searches for better match</li>
 *   <li><b>Learning</b>: Only during resonance (prevents catastrophic forgetting)</li>
 *   <li><b>Uncommitted Categories</b>: All-ones templates match any input</li>
 * </ul>
 *
 * <h2>Biological Basis</h2>
 * Based on canonical laminar neocortical circuit (Douglas & Martin, 2004;
 * Raizada & Grossberg, 2003) combined with Adaptive Resonance Theory
 * (Carpenter & Grossberg, 1987).
 *
 * @see "A Canonical Laminar Neocortical Circuit..." Raizada & Grossberg (2003)
 * @see "Adaptive Resonance Theory" Carpenter & Grossberg (1987)
 * @author Claude Code
 */
public class FullLaminarCircuitImpl {

    // 6 laminar layers
    private final Layer4Implementation layer4;
    private final Layer23Implementation layer23;
    private final Layer5Implementation layer5;
    private final Layer6Implementation layer6;
    private final Layer1Implementation layer1;  // For future attention control

    // ART components
    private final PredictionGenerator predictionGenerator;
    private final PredictionErrorProcessor errorProcessor;

    // Parameters and state
    private final CircuitParameters params;
    private CircuitState currentState;
    private final Set<Integer> suppressedCategories;  // Reset mechanism

    // Layer parameters
    private final Layer4Parameters layer4Params;
    private final Layer5Parameters layer5Params;
    private final Layer6Parameters layer6Params;

    /**
     * Create full laminar circuit with specified parameters.
     *
     * Initializes all 6 layers, prediction generator, error processor,
     * and ART dynamics machinery.
     *
     * @param params circuit parameters controlling all components
     */
    public FullLaminarCircuitImpl(CircuitParameters params) {
        this.params = params;
        this.suppressedCategories = new HashSet<>();
        this.currentState = CircuitState.initial(params.inputSize());

        // Initialize layers with appropriate sizes
        this.layer4 = new Layer4Implementation("Layer4", params.inputSize());
        this.layer23 = new Layer23Implementation("Layer23", params.inputSize());
        this.layer5 = new Layer5Implementation("Layer5", params.categorySize());
        this.layer6 = new Layer6Implementation("Layer6", params.inputSize());
        this.layer1 = new Layer1Implementation("Layer1", params.inputSize());

        // Initialize layer parameters
        this.layer4Params = Layer4Parameters.builder()
            .timeConstant(30.0)  // Fast dynamics
            .drivingStrength(0.8)
            .build();

        this.layer5Params = Layer5Parameters.builder()
            .timeConstant(100.0)  // Medium dynamics
            .amplificationGain(1.2)
            .build();

        this.layer6Params = Layer6Parameters.builder()
            .timeConstant(200.0)  // Slow dynamics
            .attentionalGain(0.5)
            .build();

        // Initialize ART components
        var predictionParams = new PredictionParameters(
            params.topDownGain(),
            params.expectationThreshold(),
            100,
            params.learningRate()
        );
        this.predictionGenerator = new PredictionGenerator(
            params.inputSize(),
            predictionParams
        );

        var matchingParams = new MatchingParameters(
            params.vigilance(),
            params.resetThreshold(),
            params.maxSearchIterations(),
            true
        );
        this.errorProcessor = new PredictionErrorProcessor(matchingParams);
    }

    /**
     * Process input through complete laminar circuit.
     *
     * Implements full ART processing cycle:
     * 1. Bottom-up feedforward through layers
     * 2. Category activation in Layer 5
     * 3. Category search with vigilance test
     * 4. Top-down expectation generation
     * 5. Match test and resonance detection
     * 6. Learning if resonance achieved
     *
     * @param input input pattern [0,1]^d where d = inputSize
     * @return output pattern (top-down expectation if resonating)
     */
    public Pattern process(Pattern input) {
        if (input.dimension() != params.inputSize()) {
            throw new IllegalArgumentException(
                String.format("Input dimension %d != expected %d",
                    input.dimension(), params.inputSize())
            );
        }

        // Try to find matching category (may search multiple)
        var matchResult = findMatchingCategory(input);

        if (matchResult.found()) {
            // Update state to resonating
            currentState = CircuitState.resonating(
                matchResult.category(),
                matchResult.matchScore(),
                input,
                matchResult.expectation(),
                matchResult.searchIteration()
            );

            // Learn template during resonance
            learn(input, matchResult.category());

            return matchResult.expectation();
        } else {
            // No match found - update state to non-resonating
            currentState = CircuitState.mismatch(
                -1,
                0.0,
                input,
                Pattern.of(new double[params.inputSize()]),
                matchResult.searchIteration()
            );

            return input;  // Return input if no resonance
        }
    }

    /**
     * Find matching category through search with vigilance test.
     *
     * Implements ART category search:
     * 1. Try uncommitted categories first (they match anything)
     * 2. For each committed category (in order):
     *    a. Generate top-down expectation
     *    b. Test vigilance criterion
     *    c. If match: return category (resonance)
     *    d. If mismatch: suppress and try next
     * 3. If no match: create new uncommitted category
     *
     * @param input input pattern to match
     * @return match result with category, expectation, and match score
     */
    private MatchResult findMatchingCategory(Pattern input) {
        var searchIteration = 0;

        // Try to find matching category
        while (searchIteration < params.maxSearchIterations()) {
            // Find next unsuppressed category to try
            var categoryId = findNextCategory(searchIteration);

            if (categoryId < 0) {
                // No more categories - create new one
                return createNewCategory(input, searchIteration);
            }

            // Generate top-down expectation from this category
            var categoryActivation = createCategoryPattern(categoryId);
            var expectation = predictionGenerator.generateExpectation(categoryActivation);

            // Test vigilance criterion
            var matchStats = errorProcessor.computeStatistics(
                input,
                expectation,
                params.vigilance()
            );

            if (matchStats.resonates()) {
                // Match found - resonance!
                return new MatchResult(
                    true,
                    categoryId,
                    matchStats.matchScore(),
                    expectation,
                    searchIteration
                );
            } else {
                // Mismatch - suppress and continue search
                suppressedCategories.add(categoryId);
                searchIteration++;
            }
        }

        // Max iterations reached - create new category
        return createNewCategory(input, searchIteration);
    }

    /**
     * Process Layer 2/3 with grouping.
     */
    private Pattern processLayer23(Pattern input) {
        layer23.receiveBottomUpInput(input);
        layer23.process(input, params.timeStep());
        return layer23.getActivation();
    }

    /**
     * Find next category to try in search order.
     *
     * Search order:
     * 1. Try categories 0, 1, 2, ... in order
     * 2. Skip suppressed categories
     * 3. Return -1 if all categories exhausted
     */
    private int findNextCategory(int iteration) {
        for (var i = 0; i < params.categorySize(); i++) {
            if (!suppressedCategories.contains(i)) {
                return i;
            }
        }
        return -1;  // All categories suppressed
    }

    /**
     * Create category activation pattern (one-hot encoding).
     */
    private Pattern createCategoryPattern(int categoryId) {
        var pattern = new double[params.categorySize()];
        pattern[categoryId] = 1.0;  // One-hot
        return Pattern.of(pattern);
    }

    /**
     * Create new uncommitted category for novel input.
     */
    private MatchResult createNewCategory(Pattern input, int iteration) {
        // Find first uncommitted category
        var newCategory = predictionGenerator.getCommittedCount();

        if (newCategory >= params.categorySize()) {
            // No categories available - use last category
            newCategory = params.categorySize() - 1;
        }

        // Uncommitted category has all-ones template (matches anything)
        var categoryPattern = createCategoryPattern(newCategory);
        var expectation = predictionGenerator.generateExpectation(categoryPattern);

        // Should match perfectly (uncommitted matches all)
        var matchStats = errorProcessor.computeStatistics(
            input,
            expectation,
            params.vigilance()
        );

        return new MatchResult(
            true,
            newCategory,
            matchStats.matchScore(),
            expectation,
            iteration
        );
    }

    /**
     * Learn template during resonance.
     *
     * Updates category template using incremental learning rule:
     * T(t+1) = T(t) + α * (X - T(t))
     *
     * Only called during resonance - prevents catastrophic forgetting.
     *
     * @param input resonant input pattern
     * @param categoryId category to update
     */
    private void learn(Pattern input, int categoryId) {
        predictionGenerator.updateTemplate(
            categoryId,
            input,
            params.learningRate()
        );
    }

    /**
     * Check if circuit is currently in resonance state.
     *
     * @return true if resonance achieved (stable match)
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
     * Clear resonance state (for testing sequential patterns).
     */
    public void clearResonanceState() {
        suppressedCategories.clear();
        currentState = CircuitState.initial(params.inputSize());
    }

    /**
     * Reset entire circuit to initial state.
     *
     * Clears all layers, templates, and state.
     */
    public void reset() {
        layer4.reset();
        layer23.reset();
        layer5.reset();
        layer6.reset();
        layer1.reset();
        predictionGenerator.reset();
        suppressedCategories.clear();
        currentState = CircuitState.initial(params.inputSize());
    }

    // Accessors for testing

    public Layer4Implementation getLayer4() {
        return layer4;
    }

    public Layer23Implementation getLayer23() {
        return layer23;
    }

    public Layer5Implementation getLayer5() {
        return layer5;
    }

    public Layer6Implementation getLayer6() {
        return layer6;
    }

    public Layer1Implementation getLayer1() {
        return layer1;
    }

    public PredictionGenerator getPredictionGenerator() {
        return predictionGenerator;
    }

    public PredictionErrorProcessor getErrorProcessor() {
        return errorProcessor;
    }

    public CircuitParameters getParameters() {
        return params;
    }

    /**
     * Internal record for category search results.
     */
    private record MatchResult(
        boolean found,
        int category,
        double matchScore,
        Pattern expectation,
        int searchIteration
    ) {}
}