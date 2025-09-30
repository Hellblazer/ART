package com.hellblazer.art.laminar.canonical;

import com.hellblazer.art.core.Pattern;

/**
 * Complete statistics from ART matching computation.
 *
 * Provides detailed information about the match between input and expectation,
 * including the match score, vigilance test result, prediction error, and
 * various norms for analysis.
 *
 * Immutable record capturing a single matching event in the ART circuit.
 *
 * @param matchScore ART match score [0,1]: M = |X ∩ E| / |X|
 * @param resonates true if matchScore >= vigilance (resonance achieved)
 * @param errorSignal signed prediction error: X - E (element-wise)
 * @param errorMagnitude average absolute error: Σ|error[i]| / N
 * @param inputNorm L1 norm of input: Σ|input[i]|
 * @param expectationNorm L1 norm of expectation: Σ|expectation[i]|
 *
 * @author Claude Code
 */
public record MatchStatistics(
    double matchScore,
    boolean resonates,
    Pattern errorSignal,
    double errorMagnitude,
    double inputNorm,
    double expectationNorm
) {
    /**
     * Validate statistics values.
     */
    public MatchStatistics {
        if (matchScore < 0.0 || matchScore > 1.0) {
            throw new IllegalArgumentException("matchScore must be in [0,1]");
        }
        if (errorMagnitude < 0.0) {
            throw new IllegalArgumentException("errorMagnitude must be non-negative");
        }
        if (inputNorm < 0.0) {
            throw new IllegalArgumentException("inputNorm must be non-negative");
        }
        if (expectationNorm < 0.0) {
            throw new IllegalArgumentException("expectationNorm must be non-negative");
        }
        if (errorSignal == null) {
            throw new IllegalArgumentException("errorSignal must not be null");
        }
    }

    @Override
    public String toString() {
        return String.format(
            "MatchStatistics[matchScore=%.3f, resonates=%s, errorMag=%.3f, inputNorm=%.3f, expectNorm=%.3f]",
            matchScore, resonates, errorMagnitude, inputNorm, expectationNorm
        );
    }
}