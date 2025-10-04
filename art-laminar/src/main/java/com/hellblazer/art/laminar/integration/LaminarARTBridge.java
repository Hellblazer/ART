package com.hellblazer.art.laminar.integration;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.core.weights.FuzzyWeight;
import com.hellblazer.art.laminar.canonical.CircuitState;
import com.hellblazer.art.laminar.canonical.CircuitParameters;
import com.hellblazer.art.core.parameters.FuzzyParameters;

import java.util.Objects;

/**
 * Utility class bridging laminar circuit and ART algorithm representations.
 *
 * <p>Provides translation methods for:
 * <ul>
 *   <li>Pattern extraction from complement-coded weights</li>
 *   <li>ActivationResult to CircuitState conversion</li>
 *   <li>Resonance score computation</li>
 *   <li>Parameter consistency validation</li>
 * </ul>
 *
 * <h2>Key Responsibility: Complement Coding Translation</h2>
 * <p>FuzzyART uses complement coding [x, 1-x], while laminar uses raw [x].
 * This bridge extracts the non-complement portion for laminar layer processing.
 *
 * <h3>Example:</h3>
 * <pre>
 * FuzzyWeight: [0.8, 0.2, 0.3, 0.7, 0.2, 0.8, 0.7, 0.3]
 * Expectation: [0.8, 0.2, 0.3, 0.7]  // First half only
 * </pre>
 *
 * @author Hal Hildebrand
 */
public final class LaminarARTBridge {

    /**
     * Private constructor prevents instantiation of utility class.
     */
    private LaminarARTBridge() {
        throw new AssertionError("Utility class cannot be instantiated");
    }

    /**
     * Extract expectation pattern from FuzzyART weight vector.
     *
     * <p>FuzzyART uses complement coding: [x₁, x₂, ..., xₙ, 1-x₁, 1-x₂, ..., 1-xₙ]
     * This method extracts the first half (non-complement portion) as expectation.
     *
     * <p>Supports both FuzzyWeight and VectorizedFuzzyWeight through the WeightVector interface.
     *
     * <h3>Algorithm:</h3>
     * <ol>
     *   <li>Verify dimension is even (complement coding requirement)</li>
     *   <li>Extract first half of weight data</li>
     *   <li>Return as Pattern</li>
     * </ol>
     *
     * @param artWeight weight vector with complement coding (FuzzyWeight or VectorizedFuzzyWeight)
     * @return expectation pattern (non-complement-coded)
     * @throws IllegalStateException if weight dimension is odd
     * @throws NullPointerException if artWeight is null
     */
    public static Pattern extractExpectation(WeightVector artWeight) {
        Objects.requireNonNull(artWeight, "artWeight cannot be null");

        var fullDimension = artWeight.dimension();
        if (fullDimension % 2 != 0) {
            throw new IllegalStateException(
                "Weight dimension must be even (complement coding), got: " + fullDimension
            );
        }

        // Extract first half (non-complement portion)
        var halfSize = fullDimension / 2;
        var expectation = new double[halfSize];
        for (int i = 0; i < halfSize; i++) {
            expectation[i] = artWeight.get(i);
        }

        return Pattern.of(expectation);
    }

    /**
     * Convert ActivationResult to CircuitState.
     *
     * <p>Maps ART resonance outcome to laminar circuit state representation,
     * including category ID, match score, input, and expectation.
     *
     * @param result ART activation result
     * @param input original input pattern
     * @param expectation extracted expectation pattern
     * @return CircuitState representing resonance or mismatch
     * @throws NullPointerException if any parameter is null
     */
    public static CircuitState toCircuitState(
            ActivationResult result,
            Pattern input,
            Pattern expectation) {
        Objects.requireNonNull(result, "result cannot be null");
        Objects.requireNonNull(input, "input cannot be null");
        Objects.requireNonNull(expectation, "expectation cannot be null");

        if (result instanceof ActivationResult.Success success) {
            return CircuitState.resonating(
                success.categoryIndex(),
                success.activationValue(),
                input,
                expectation,
                0  // FuzzyART handles search internally
            );
        } else {
            return CircuitState.mismatch(
                -1,
                0.0,
                input,
                expectation,
                0
            );
        }
    }

    /**
     * Compute resonance score from ActivationResult.
     *
     * <p>Uses activation value as proxy for match quality.
     * Higher activation indicates better match between input and category.
     *
     * @param result ART activation result
     * @return resonance score [0.0-1.0], or 0.0 if no match
     * @throws NullPointerException if result is null
     */
    public static double computeResonanceScore(ActivationResult result) {
        Objects.requireNonNull(result, "result cannot be null");

        if (result instanceof ActivationResult.Success success) {
            return success.activationValue();
        } else {
            return 0.0;
        }
    }

    /**
     * Check if ActivationResult indicates resonance.
     *
     * @param result ART activation result
     * @return true if resonance achieved (Success result)
     * @throws NullPointerException if result is null
     */
    public static boolean isResonance(ActivationResult result) {
        Objects.requireNonNull(result, "result cannot be null");
        return result instanceof ActivationResult.Success;
    }

    /**
     * Extract category ID from ActivationResult.
     *
     * @param result ART activation result
     * @return category ID, or -1 if no match
     * @throws NullPointerException if result is null
     */
    public static int extractCategoryId(ActivationResult result) {
        Objects.requireNonNull(result, "result cannot be null");

        if (result instanceof ActivationResult.Success success) {
            return success.categoryIndex();
        } else {
            return -1;
        }
    }

    /**
     * Validate parameter consistency between CircuitParameters and FuzzyParameters.
     *
     * <p>Checks that shared parameters (vigilance, learning rate) match between
     * laminar circuit and FuzzyART configurations.
     *
     * <h3>Validated Parameters:</h3>
     * <ul>
     *   <li>Vigilance (ρ): Must match within 1e-6 tolerance</li>
     *   <li>Learning rate (β): Must match within 1e-6 tolerance</li>
     * </ul>
     *
     * @param circuitParams circuit parameters
     * @param fuzzyParams FuzzyART parameters
     * @throws IllegalArgumentException if parameters are inconsistent
     * @throws NullPointerException if any parameter is null
     */
    public static void validateParameterConsistency(
            CircuitParameters circuitParams,
            FuzzyParameters fuzzyParams) {
        Objects.requireNonNull(circuitParams, "circuitParams cannot be null");
        Objects.requireNonNull(fuzzyParams, "fuzzyParams cannot be null");

        // Check vigilance consistency
        if (Math.abs(circuitParams.vigilance() - fuzzyParams.vigilance()) > 1e-6) {
            throw new IllegalArgumentException(
                String.format("Vigilance mismatch: circuit=%.4f, fuzzy=%.4f",
                    circuitParams.vigilance(), fuzzyParams.vigilance())
            );
        }

        // Check learning rate consistency
        if (Math.abs(circuitParams.learningRate() - fuzzyParams.beta()) > 1e-6) {
            throw new IllegalArgumentException(
                String.format("Learning rate mismatch: circuit=%.4f, fuzzy=%.4f",
                    circuitParams.learningRate(), fuzzyParams.beta())
            );
        }
    }

    /**
     * Check parameter consistency between CircuitParameters and FuzzyParameters.
     *
     * <p>Non-throwing variant that returns boolean instead of throwing exception.
     * Useful for assertions and conditional logic.
     *
     * @param circuitParams circuit parameters
     * @param fuzzyParams FuzzyART parameters
     * @return true if parameters are consistent, false otherwise
     * @throws NullPointerException if any parameter is null
     */
    public static boolean validateParameterConsistency(
            CircuitParameters circuitParams,
            FuzzyParameters fuzzyParams,
            boolean throwOnMismatch) {
        Objects.requireNonNull(circuitParams, "circuitParams cannot be null");
        Objects.requireNonNull(fuzzyParams, "fuzzyParams cannot be null");

        // Check vigilance consistency
        if (Math.abs(circuitParams.vigilance() - fuzzyParams.vigilance()) > 1e-6) {
            if (throwOnMismatch) {
                throw new IllegalArgumentException(
                    String.format("Vigilance mismatch: circuit=%.4f, fuzzy=%.4f",
                        circuitParams.vigilance(), fuzzyParams.vigilance())
                );
            }
            return false;
        }

        // Check learning rate consistency
        if (Math.abs(circuitParams.learningRate() - fuzzyParams.beta()) > 1e-6) {
            if (throwOnMismatch) {
                throw new IllegalArgumentException(
                    String.format("Learning rate mismatch: circuit=%.4f, fuzzy=%.4f",
                        circuitParams.learningRate(), fuzzyParams.beta())
                );
            }
            return false;
        }

        return true;
    }

    /**
     * Validate ARTCircuitParameters internal consistency.
     *
     * <p>Ensures that the FuzzyParameters and CircuitParameters derived from
     * ARTCircuitParameters are mutually consistent.
     *
     * @param artParams ART circuit parameters to validate
     * @throws IllegalArgumentException if derived parameters are inconsistent
     * @throws NullPointerException if artParams is null
     */
    public static void validateARTCircuitParameters(ARTCircuitParameters artParams) {
        Objects.requireNonNull(artParams, "artParams cannot be null");

        var fuzzyParams = artParams.toFuzzyParameters();
        var circuitParams = artParams.toCircuitParameters();

        validateParameterConsistency(circuitParams, fuzzyParams);
    }
}
