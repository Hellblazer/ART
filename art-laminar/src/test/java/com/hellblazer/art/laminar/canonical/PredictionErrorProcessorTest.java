package com.hellblazer.art.laminar.canonical;

import com.hellblazer.art.core.Pattern;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for PredictionErrorProcessor - implements ART matching rule and prediction error calculation.
 *
 * Implements test-first RED-GREEN-REFACTOR approach for Day 2 of Phase 2 Week 3.
 * These tests verify the ART matching mechanism that compares bottom-up input with
 * top-down expectation, implementing the vigilance test and prediction error calculation.
 *
 * Critical Formula (ASYMMETRIC!):
 * <pre>
 * matchScore = |input ∩ expectation| / |input|
 *            = Σ(min(input[i], expectation[i])) / Σ(input[i])
 *
 * NOT: |input ∩ expectation| / |input + expectation|  ← This is WRONG!
 * </pre>
 *
 * @author Hal Hildebrand
 */
class PredictionErrorProcessorTest {

    private static final double EPSILON = 0.001;
    private PredictionErrorProcessor processor;

    @BeforeEach
    void setUp() {
        processor = new PredictionErrorProcessor(new MatchingParameters());
    }

    /**
     * Test 1: Perfect match score.
     *
     * Given: Input pattern identical to expectation
     * When: Compute match score
     * Then: Match score = 1.0 (perfect match)
     *
     * Validates: M = |X ∩ E| / |X| where X = E
     *           M = |X| / |X| = 1.0
     */
    @Test
    void testPerfectMatchScore() {
        // Identical patterns
        var pattern = Pattern.of(0.8, 0.6, 0.4, 0.2);
        var matchScore = processor.computeMatchScore(pattern, pattern);

        // Perfect match should give score = 1.0
        assertEquals(1.0, matchScore, EPSILON,
            "Perfect match (input == expectation) should give match score 1.0");
    }

    /**
     * Test 2: Partial match score with asymmetric formula.
     *
     * Given: Input and expectation with partial overlap
     * When: Compute match score
     * Then: Match score = |intersection| / |input| (asymmetric!)
     *
     * Critical Test: This validates the ASYMMETRIC ART match formula.
     * The denominator is |input| ONLY, not |input + expectation|!
     *
     * Example:
     *   input       = [0.8, 0.6, 0.4, 0.2]  |input| = 2.0
     *   expectation = [0.9, 0.3, 0.5, 0.1]
     *   intersection = [0.8, 0.3, 0.4, 0.1]  |intersection| = 1.6
     *   matchScore = 1.6 / 2.0 = 0.8
     */
    @Test
    void testPartialMatchScore() {
        var input = Pattern.of(0.8, 0.6, 0.4, 0.2);
        var expectation = Pattern.of(0.9, 0.3, 0.5, 0.1);

        var matchScore = processor.computeMatchScore(input, expectation);

        // Compute expected score manually:
        // intersection = min(0.8,0.9), min(0.6,0.3), min(0.4,0.5), min(0.2,0.1)
        //              = [0.8, 0.3, 0.4, 0.1]
        // |intersection| = 0.8 + 0.3 + 0.4 + 0.1 = 1.6
        // |input| = 0.8 + 0.6 + 0.4 + 0.2 = 2.0
        // matchScore = 1.6 / 2.0 = 0.8
        var expectedScore = 0.8;

        assertEquals(expectedScore, matchScore, EPSILON,
            "Match score should be |intersection| / |input| (asymmetric formula)");
    }

    /**
     * Test 3: Vigilance acceptance (match above threshold).
     *
     * Given: High match score above vigilance parameter
     * When: Test vigilance criterion
     * Then: Returns true (resonance achieved)
     *
     * Validates: if M >= ρ then RESONANCE
     */
    @Test
    void testVigilanceAcceptance() {
        var vigilance = 0.7;

        // Perfect match (score = 1.0) exceeds vigilance (0.7)
        var input = Pattern.of(0.8, 0.6, 0.4, 0.2);
        var expectation = Pattern.of(0.8, 0.6, 0.4, 0.2);

        var matchResult = processor.computeStatistics(input, expectation, vigilance);

        assertTrue(matchResult.resonates(),
            "Perfect match (score=1.0) should resonate with vigilance=0.7");
        assertEquals(1.0, matchResult.matchScore(), EPSILON,
            "Match score should be 1.0 for identical patterns");
    }

    /**
     * Test 4: Vigilance rejection (match below threshold).
     *
     * Given: Low match score below vigilance parameter
     * When: Test vigilance criterion
     * Then: Returns false (mismatch reset)
     *
     * Validates: if M < ρ then RESET
     */
    @Test
    void testVigilanceRejection() {
        var vigilance = 0.9;  // High vigilance (very specific)

        // Poor match: input and expectation have low overlap
        var input = Pattern.of(0.8, 0.6, 0.4, 0.2);
        var expectation = Pattern.of(0.2, 0.1, 0.0, 0.0);

        var matchResult = processor.computeStatistics(input, expectation, vigilance);

        // Compute expected score:
        // intersection = [0.2, 0.1, 0.0, 0.0]
        // |intersection| = 0.3
        // |input| = 2.0
        // matchScore = 0.3 / 2.0 = 0.15 < 0.9
        assertFalse(matchResult.resonates(),
            "Poor match (score < 0.9) should NOT resonate with vigilance=0.9");
        assertTrue(matchResult.matchScore() < vigilance,
            "Match score should be below vigilance threshold");
    }

    /**
     * Test 5: Prediction error calculation (signed difference).
     *
     * Given: Input pattern and expectation pattern
     * When: Compute prediction error
     * Then: Error = input - expectation (element-wise, signed)
     *
     * Validates:
     *   error[i] = input[i] - expectation[i]
     *   Positive error: input exceeds expectation (unexpected feature)
     *   Negative error: expectation exceeds input (missing feature)
     */
    @Test
    void testPredictionErrorCalculation() {
        var input = Pattern.of(0.8, 0.6, 0.4, 0.2);
        var expectation = Pattern.of(0.5, 0.7, 0.3, 0.1);

        var matchResult = processor.computeStatistics(input, expectation, 0.7);

        // Expected error = input - expectation (element-wise)
        // error = [0.8-0.5, 0.6-0.7, 0.4-0.3, 0.2-0.1]
        //       = [0.3, -0.1, 0.1, 0.1]
        var expectedError = Pattern.of(0.3, -0.1, 0.1, 0.1);

        var errorSignal = matchResult.errorSignal();
        assertNotNull(errorSignal, "Error signal should not be null");
        assertEquals(input.dimension(), errorSignal.dimension(),
            "Error dimension should match input dimension");

        for (int i = 0; i < input.dimension(); i++) {
            assertEquals(expectedError.get(i), errorSignal.get(i), EPSILON,
                "Error at index " + i + " should be input[i] - expectation[i]");
        }

        // Error magnitude = Σ|error[i]| / N
        // = (0.3 + 0.1 + 0.1 + 0.1) / 4 = 0.6 / 4 = 0.15
        var expectedMagnitude = 0.15;
        assertEquals(expectedMagnitude, matchResult.errorMagnitude(), EPSILON,
            "Error magnitude should be average absolute error");
    }

    /**
     * Test 6: Zero input handling (edge case).
     *
     * Given: Zero input pattern (all zeros)
     * When: Compute match score
     * Then: Returns 0.0 (avoid division by zero)
     *
     * Edge case handling: Zero input should not cause NaN or infinity.
     * Match score defined as 0.0 for zero input (no information to match).
     */
    @Test
    void testZeroInputHandling() {
        var zeroInput = Pattern.of(0.0, 0.0, 0.0, 0.0);
        var expectation = Pattern.of(0.5, 0.7, 0.3, 0.1);

        var matchScore = processor.computeMatchScore(zeroInput, expectation);

        // Zero input should return match score 0.0 (not NaN or Infinity)
        assertEquals(0.0, matchScore, EPSILON,
            "Zero input should produce match score 0.0 (avoid division by zero)");
        assertFalse(Double.isNaN(matchScore), "Match score should not be NaN");
        assertFalse(Double.isInfinite(matchScore), "Match score should not be infinite");
    }

    /**
     * Additional Test: Verify asymmetry of match formula.
     *
     * This test explicitly verifies that M(X,E) ≠ M(E,X) in general.
     * The ART match formula is asymmetric because denominator is |input| only.
     */
    @Test
    void testMatchFormulaAsymmetry() {
        var pattern1 = Pattern.of(1.0, 0.0, 0.0, 0.0);  // |p1| = 1.0
        var pattern2 = Pattern.of(0.5, 0.5, 0.5, 0.5);  // |p2| = 2.0

        // Forward: M(p1, p2) = |p1 ∩ p2| / |p1|
        var forwardScore = processor.computeMatchScore(pattern1, pattern2);

        // Reverse: M(p2, p1) = |p2 ∩ p1| / |p2|
        var reverseScore = processor.computeMatchScore(pattern2, pattern1);

        // Forward: intersection = [0.5, 0.0, 0.0, 0.0], sum = 0.5, |p1| = 1.0
        // M(p1,p2) = 0.5 / 1.0 = 0.5
        assertEquals(0.5, forwardScore, EPSILON, "Forward match score");

        // Reverse: intersection = [0.5, 0.0, 0.0, 0.0], sum = 0.5, |p2| = 2.0
        // M(p2,p1) = 0.5 / 2.0 = 0.25
        assertEquals(0.25, reverseScore, EPSILON, "Reverse match score");

        // Verify asymmetry
        assertNotEquals(forwardScore, reverseScore, EPSILON,
            "ART match formula should be asymmetric: M(X,E) ≠ M(E,X)");
    }
}
