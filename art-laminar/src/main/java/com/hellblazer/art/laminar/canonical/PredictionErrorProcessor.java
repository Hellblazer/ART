package com.hellblazer.art.laminar.canonical;

import com.hellblazer.art.core.Pattern;

/**
 * Computes prediction errors and implements the ART matching rule.
 *
 * The matching rule determines whether bottom-up input resonates with top-down
 * expectation, or whether mismatch triggers reset and category search. This is
 * the core mechanism preventing catastrophic forgetting in ART networks.
 *
 * <h2>Critical ART Match Formula (ASYMMETRIC!)</h2>
 * <pre>
 * M = |X ∩ E| / |X|
 *   = Σ min(x_i, e_i) / Σ x_i
 *
 * where:
 *   X = input pattern (bottom-up from Layer 4)
 *   E = expectation pattern (top-down from Layer 6)
 *   ∩ = element-wise minimum (intersection)
 *   |·| = L1 norm (sum of elements)
 * </pre>
 *
 * <h2>Why Asymmetric?</h2>
 * The denominator is |X| (input norm) ONLY, not |X + E| or |E|. This asymmetry
 * is critical for ART's stability-plasticity balance:
 * <ul>
 *   <li>Prevents overly general categories (match is strict on input)</li>
 *   <li>Allows subset matching (expectation can be broader than input)</li>
 *   <li>Creates "matching law" described by Grossberg</li>
 * </ul>
 *
 * <h2>Vigilance Test</h2>
 * <pre>
 * if M >= ρ: RESONANCE (accept category, enable learning)
 * else:      RESET (reject category, search for new one)
 * </pre>
 *
 * <h2>Biological Basis</h2>
 * Comparison between Layer 4 bottom-up activation and Layer 6 top-down expectation
 * in canonical neocortical circuit. Mismatch drives attentional reset and category
 * search via Layer 1 gain control.
 *
 * @see "Adaptive Resonance Theory" Carpenter & Grossberg (1987)
 * @see "A Canonical Laminar Neocortical Circuit..." Raizada & Grossberg (2003)
 * @author Claude Code
 */
public class PredictionErrorProcessor {

    private final MatchingParameters params;

    /**
     * Create prediction error processor with specified parameters.
     *
     * @param params matching parameters controlling vigilance and reset behavior
     */
    public PredictionErrorProcessor(MatchingParameters params) {
        if (params == null) {
            throw new IllegalArgumentException("params must not be null");
        }
        this.params = params;
    }

    /**
     * Compute ART match score.
     *
     * Implements the asymmetric ART matching rule:
     * <pre>
     * M = |X ∩ E| / |X| = Σ min(x_i, e_i) / Σ x_i
     * </pre>
     *
     * The intersection (∩) is computed as element-wise minimum. Normalization
     * by |X| (input norm) creates asymmetric matching that prevents overly
     * general categories - a key ART principle.
     *
     * <h3>Edge Cases</h3>
     * <ul>
     *   <li>Zero input: returns 0.0 (avoids division by zero)</li>
     *   <li>Zero expectation: valid, returns 0.0 (no overlap)</li>
     *   <li>Perfect match (X == E): returns 1.0</li>
     * </ul>
     *
     * <h3>Mathematical Properties</h3>
     * <ul>
     *   <li>Range: [0, 1]</li>
     *   <li>Asymmetric: M(X,E) ≠ M(E,X) in general</li>
     *   <li>Monotonic: increasing overlap → increasing match</li>
     * </ul>
     *
     * @param input bottom-up feature pattern from Layer 4
     * @param expectation top-down expected pattern from Layer 6
     * @return match score [0,1] where 1.0 = perfect match
     * @throws IllegalArgumentException if dimensions don't match
     * @throws NullPointerException if input or expectation is null
     */
    public double computeMatchScore(Pattern input, Pattern expectation) {
        if (input == null || expectation == null) {
            throw new NullPointerException("input and expectation must not be null");
        }
        if (input.dimension() != expectation.dimension()) {
            throw new IllegalArgumentException(
                String.format("Dimension mismatch: input=%d, expectation=%d",
                    input.dimension(), expectation.dimension())
            );
        }

        // Compute intersection (element-wise minimum) and input norm
        var intersection = 0.0;
        var inputNorm = 0.0;

        for (var i = 0; i < input.dimension(); i++) {
            var inputVal = input.get(i);
            var expectVal = expectation.get(i);
            intersection += Math.min(inputVal, expectVal);
            inputNorm += inputVal;
        }

        // Handle edge case: zero input (avoid division by zero)
        if (inputNorm < 1e-10) {
            return 0.0;
        }

        // ART match score: |X ∩ E| / |X|
        return intersection / inputNorm;
    }

    /**
     * Compute complete match statistics.
     *
     * Performs full ART matching computation including:
     * <ul>
     *   <li>Match score calculation</li>
     *   <li>Vigilance test</li>
     *   <li>Prediction error computation</li>
     *   <li>Norm calculations</li>
     * </ul>
     *
     * This is the primary method for evaluating whether input resonates with
     * expectation in the laminar circuit.
     *
     * @param input bottom-up pattern from Layer 4
     * @param expectation top-down pattern from Layer 6
     * @param vigilance match threshold [0,1] for resonance
     * @return complete match statistics including resonance decision
     * @throws IllegalArgumentException if dimensions don't match or vigilance invalid
     * @throws NullPointerException if input or expectation is null
     */
    public MatchStatistics computeStatistics(Pattern input, Pattern expectation, double vigilance) {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("vigilance must be in [0,1]");
        }

        // Compute match score
        var matchScore = computeMatchScore(input, expectation);

        // Test vigilance criterion
        var resonates = matchScore >= vigilance;

        // Compute prediction error: error = input - expectation (signed)
        var error = new double[input.dimension()];
        var errorSum = 0.0;
        for (var i = 0; i < input.dimension(); i++) {
            error[i] = input.get(i) - expectation.get(i);
            errorSum += Math.abs(error[i]);
        }

        // Error magnitude: average absolute error
        var errorMagnitude = errorSum / input.dimension();

        // Compute norms for statistics
        var inputNorm = input.l1Norm();
        var expectationNorm = expectation.l1Norm();

        return new MatchStatistics(
            matchScore,
            resonates,
            Pattern.of(error),
            errorMagnitude,
            inputNorm,
            expectationNorm
        );
    }

    /**
     * Test vigilance criterion (simplified interface).
     *
     * Returns true if match score meets or exceeds vigilance parameter,
     * indicating resonance should occur. False indicates mismatch reset.
     *
     * <h3>Vigilance Parameter Effects</h3>
     * <ul>
     *   <li>Low vigilance (0.3-0.5): broad categories, more generalization</li>
     *   <li>Medium vigilance (0.6-0.8): balanced specificity (typical)</li>
     *   <li>High vigilance (0.85-0.95): narrow categories, more specific</li>
     * </ul>
     *
     * @param input bottom-up feature pattern
     * @param expectation top-down expected pattern
     * @param vigilance match threshold [0,1]
     * @return true if resonance (match), false if reset (mismatch)
     * @throws IllegalArgumentException if dimensions don't match or vigilance invalid
     */
    public boolean testVigilance(Pattern input, Pattern expectation, double vigilance) {
        var matchScore = computeMatchScore(input, expectation);
        return matchScore >= vigilance;
    }

    /**
     * Compute prediction error vector.
     *
     * Error = input - expectation (element-wise, signed)
     *
     * <h3>Error Interpretation</h3>
     * <ul>
     *   <li>Positive error: input exceeds expectation (unexpected feature present)</li>
     *   <li>Negative error: expectation exceeds input (expected feature missing)</li>
     *   <li>Zero error: perfect prediction</li>
     * </ul>
     *
     * This error signal can drive learning in adaptive systems, though in standard
     * ART learning only occurs during resonance (not based on error magnitude).
     *
     * @param input actual bottom-up pattern
     * @param expectation predicted pattern from top-down
     * @return error vector (can contain negative values)
     * @throws IllegalArgumentException if dimensions don't match
     * @throws NullPointerException if input or expectation is null
     */
    public Pattern computePredictionError(Pattern input, Pattern expectation) {
        if (input == null || expectation == null) {
            throw new NullPointerException("input and expectation must not be null");
        }
        if (input.dimension() != expectation.dimension()) {
            throw new IllegalArgumentException("Dimension mismatch");
        }

        var error = new double[input.dimension()];
        for (var i = 0; i < input.dimension(); i++) {
            error[i] = input.get(i) - expectation.get(i);
        }

        return Pattern.of(error);
    }

    /**
     * Check if reset needed based on vigilance and error threshold.
     *
     * Reset triggered if:
     * <ul>
     *   <li>Match score below vigilance (failed vigilance test), OR</li>
     *   <li>Error magnitude exceeds reset threshold</li>
     * </ul>
     *
     * This provides dual control over category stability: both match-based
     * (standard ART) and error-based (extended mechanism).
     *
     * @param input bottom-up pattern
     * @param expectation top-down pattern
     * @return true if reset should occur (search for new category)
     */
    public boolean needsReset(Pattern input, Pattern expectation) {
        var matchScore = computeMatchScore(input, expectation);

        // Failed vigilance test
        if (matchScore < params.vigilance()) {
            return true;
        }

        // Check error threshold
        var error = computePredictionError(input, expectation);
        var errorMagnitude = 0.0;
        for (var i = 0; i < error.dimension(); i++) {
            errorMagnitude += Math.abs(error.get(i));
        }
        errorMagnitude /= error.dimension();

        return errorMagnitude > params.resetThreshold();
    }

    /**
     * Get current matching parameters.
     *
     * @return matching parameters (immutable)
     */
    public MatchingParameters getParameters() {
        return params;
    }
}