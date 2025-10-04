package com.hellblazer.art.laminar.canonical;

import com.hellblazer.art.core.Pattern;

/**
 * Current state of the full laminar circuit.
 *
 * Captures the dynamic state during ART processing including:
 * - Resonance status (stable match achieved)
 * - Active category (winning category ID)
 * - Match score (vigilance test result)
 * - Search iteration (number of category search attempts)
 * - Last input pattern (for comparison)
 * - Last expectation (top-down prediction)
 *
 * Immutable snapshot of circuit state at a specific time.
 *
 * @param isResonating true if circuit achieved resonance (stable match)
 * @param activeCategory winning category ID, -1 if none active
 * @param matchScore ART match score [0,1]: M = |X ∩ E| / |X|
 * @param searchIteration number of category search attempts (reset count)
 * @param lastInput most recent input pattern processed
 * @param lastExpectation most recent top-down expectation generated
 *
 * @see "Adaptive Resonance Theory" Carpenter & Grossberg (1987)
 * @author Hal Hildebrand
 */
public record CircuitState(
    boolean isResonating,
    int activeCategory,
    double matchScore,
    int searchIteration,
    Pattern lastInput,
    Pattern lastExpectation
) {
    /**
     * Compact validation constructor.
     */
    public CircuitState {
        if (activeCategory < -1) {
            throw new IllegalArgumentException("activeCategory must be >= -1");
        }
        if (matchScore < 0.0 || matchScore > 1.0) {
            throw new IllegalArgumentException("matchScore must be in [0,1]");
        }
        if (searchIteration < 0) {
            throw new IllegalArgumentException("searchIteration must be non-negative");
        }
        if (lastInput == null) {
            throw new IllegalArgumentException("lastInput must not be null");
        }
        if (lastExpectation == null) {
            throw new IllegalArgumentException("lastExpectation must not be null");
        }
    }

    /**
     * Create initial empty state.
     *
     * @param inputSize dimensionality of input patterns
     * @return initial state with no activity
     */
    public static CircuitState initial(int inputSize) {
        return new CircuitState(
            false,  // Not resonating
            -1,     // No active category
            0.0,    // Zero match score
            0,      // No search iterations
            Pattern.of(new double[inputSize]),  // Zero input
            Pattern.of(new double[inputSize])   // Zero expectation
        );
    }

    /**
     * Create state representing resonance.
     *
     * @param category winning category ID
     * @param matchScore match score achieved
     * @param input input pattern
     * @param expectation top-down expectation
     * @param iteration search iteration count
     * @return resonating state
     */
    public static CircuitState resonating(
        int category,
        double matchScore,
        Pattern input,
        Pattern expectation,
        int iteration
    ) {
        return new CircuitState(
            true,
            category,
            matchScore,
            iteration,
            input,
            expectation
        );
    }

    /**
     * Create state representing mismatch (no resonance).
     *
     * @param category attempted category ID
     * @param matchScore match score (below vigilance)
     * @param input input pattern
     * @param expectation top-down expectation
     * @param iteration search iteration count
     * @return non-resonating state
     */
    public static CircuitState mismatch(
        int category,
        double matchScore,
        Pattern input,
        Pattern expectation,
        int iteration
    ) {
        return new CircuitState(
            false,
            category,
            matchScore,
            iteration,
            input,
            expectation
        );
    }

    @Override
    public String toString() {
        return String.format(
            "CircuitState[resonating=%s, category=%d, match=%.3f, iteration=%d]",
            isResonating, activeCategory, matchScore, searchIteration
        );
    }
}
