package com.hellblazer.art.laminar.validation;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.canonical.MatchingParameters;
import com.hellblazer.art.laminar.canonical.PredictionErrorProcessor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive validation tests for ART matching rule.
 * Validates the asymmetric matching formula and vigilance test from Grossberg's ART theory.
 *
 * Key Formula: M = |X ∩ E| / |X| = Σ min(x_i, e_i) / Σ x_i
 *
 * @author Hal Hildebrand
 */
class ARTMatchingRuleValidationTest {

    private PredictionErrorProcessor processor;
    private MatchingParameters params;

    @BeforeEach
    void setUp() {
        params = new MatchingParameters();
        processor = new PredictionErrorProcessor(params);
    }

    /**
     * Test 1: Perfect match should give score = 1.0.
     * When X == E, intersection equals X, so M = |X| / |X| = 1.0
     */
    @Test
    void testPerfectMatch() {
        var pattern = new DenseVector(new double[]{0.8, 0.6, 0.4, 0.2});
        var expectation = new DenseVector(new double[]{0.8, 0.6, 0.4, 0.2});

        double match = processor.computeMatchScore(pattern, expectation);

        assertEquals(1.0, match, 0.001,
            "Perfect match (X == E) should give score 1.0");
    }

    /**
     * Test 2: Zero input should handle gracefully (avoid division by zero).
     */
    @Test
    void testZeroInput() {
        var zeroPattern = new DenseVector(new double[]{0.0, 0.0, 0.0, 0.0});
        var expectation = new DenseVector(new double[]{0.5, 0.5, 0.5, 0.5});

        double match = processor.computeMatchScore(zeroPattern, expectation);

        assertEquals(0.0, match, 0.001,
            "Zero input should return 0.0 (avoid division by zero)");
    }

    /**
     * Test 3: Subset matching - expectation broader than input.
     * E superset of X → M should be high.
     * This is critical for ART's ability to form general categories.
     */
    @Test
    void testSubsetMatching() {
        var input = new DenseVector(new double[]{1.0, 0.0, 0.0, 0.0});
        var broadExpectation = new DenseVector(new double[]{1.0, 1.0, 1.0, 1.0});

        double match = processor.computeMatchScore(input, broadExpectation);

        // M = min(1,1) / 1 = 1.0
        assertEquals(1.0, match, 0.001,
            "Broad expectation (superset) should match narrow input perfectly");
    }

    /**
     * Test 4: Asymmetry - M(X,E) ≠ M(E,X) in general.
     * This is the key property that prevents overly general categories.
     */
    @Test
    void testAsymmetry() {
        var narrow = new DenseVector(new double[]{1.0, 0.0, 0.0, 0.0});
        var broad = new DenseVector(new double[]{1.0, 1.0, 1.0, 1.0});

        double matchNarrowInput = processor.computeMatchScore(narrow, broad);
        double matchBroadInput = processor.computeMatchScore(broad, narrow);

        // narrow as input: M = 1/1 = 1.0
        // broad as input: M = 1/4 = 0.25
        assertEquals(1.0, matchNarrowInput, 0.001);
        assertEquals(0.25, matchBroadInput, 0.001);

        assertNotEquals(matchNarrowInput, matchBroadInput,
            "ART matching is asymmetric: M(X,E) ≠ M(E,X)");
    }

    /**
     * Test 5: Vigilance test - accept or reject category.
     * M >= ρ → Accept (resonance)
     * M < ρ → Reject (reset)
     */
    @Test
    void testVigilanceTest() {
        var input = new DenseVector(new double[]{1.0, 1.0, 0.0, 0.0});
        var goodMatch = new DenseVector(new double[]{1.0, 0.9, 0.1, 0.1}); // M ≈ 0.95
        var poorMatch = new DenseVector(new double[]{0.3, 0.3, 0.3, 0.3}); // M ≈ 0.3

        double goodScore = processor.computeMatchScore(input, goodMatch);
        double poorScore = processor.computeMatchScore(input, poorMatch);

        // High vigilance (0.85) - good match passes, poor match fails
        double highVigilance = 0.85;
        assertTrue(goodScore >= highVigilance,
            "Good match should pass high vigilance test");
        assertFalse(poorScore >= highVigilance,
            "Poor match should fail high vigilance test");

        // Low vigilance (0.25) - both pass
        double lowVigilance = 0.25;
        assertTrue(goodScore >= lowVigilance,
            "Good match should pass low vigilance");
        assertTrue(poorScore >= lowVigilance,
            "Even poor match should pass low vigilance");
    }

    /**
     * Test 6: Partial overlap - typical ART scenario.
     */
    @Test
    void testPartialOverlap() {
        var input = new DenseVector(new double[]{0.8, 0.6, 0.0, 0.0});
        var expectation = new DenseVector(new double[]{0.7, 0.5, 0.3, 0.1});

        double match = processor.computeMatchScore(input, expectation);

        // M = (min(0.8,0.7) + min(0.6,0.5) + 0 + 0) / (0.8 + 0.6)
        // M = (0.7 + 0.5) / 1.4 = 1.2 / 1.4 ≈ 0.857
        assertEquals(0.857, match, 0.01,
            "Partial overlap should give intermediate match score");
    }

    /**
     * Test 7: Monotonicity - increasing overlap increases match.
     */
    @Test
    void testMonotonicity() {
        var input = new DenseVector(new double[]{1.0, 1.0, 1.0, 1.0});

        var lowOverlap = new DenseVector(new double[]{0.3, 0.0, 0.0, 0.0});
        var medOverlap = new DenseVector(new double[]{0.5, 0.5, 0.0, 0.0});
        var highOverlap = new DenseVector(new double[]{0.8, 0.8, 0.8, 0.0});

        double matchLow = processor.computeMatchScore(input, lowOverlap);
        double matchMed = processor.computeMatchScore(input, medOverlap);
        double matchHigh = processor.computeMatchScore(input, highOverlap);

        assertTrue(matchLow < matchMed,
            "More overlap should increase match (low < med)");
        assertTrue(matchMed < matchHigh,
            "More overlap should increase match (med < high)");
    }

    /**
     * Test 8: Dimension mismatch should throw exception.
     */
    @Test
    void testDimensionMismatch() {
        var pattern4D = new DenseVector(new double[]{1.0, 0.5, 0.5, 0.0});
        var pattern8D = new DenseVector(new double[]{1.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.0});

        assertThrows(IllegalArgumentException.class,
            () -> processor.computeMatchScore(pattern4D, pattern8D),
            "Dimension mismatch should throw IllegalArgumentException");
    }

    /**
     * Test 9: Null inputs should throw exception.
     */
    @Test
    void testNullInputs() {
        var validPattern = new DenseVector(new double[]{1.0, 0.5});

        assertThrows(NullPointerException.class,
            () -> processor.computeMatchScore(null, validPattern),
            "Null input should throw NullPointerException");

        assertThrows(NullPointerException.class,
            () -> processor.computeMatchScore(validPattern, null),
            "Null expectation should throw NullPointerException");
    }

    /**
     * Test 10: Match score range - always [0, 1].
     */
    @Test
    void testMatchScoreRange() {
        var input = new DenseVector(new double[]{0.9, 0.7, 0.5, 0.3});

        // Test with various expectations
        for (int trial = 0; trial < 100; trial++) {
            var expectation = createRandomPattern(4);
            double match = processor.computeMatchScore(input, expectation);

            assertTrue(match >= 0.0 && match <= 1.0,
                String.format("Match score %.3f outside valid range [0,1]", match));
        }
    }

    // Helper method
    private Pattern createRandomPattern(int dim) {
        var data = new double[dim];
        for (int i = 0; i < dim; i++) {
            data[i] = Math.random();
        }
        return new DenseVector(data);
    }
}
