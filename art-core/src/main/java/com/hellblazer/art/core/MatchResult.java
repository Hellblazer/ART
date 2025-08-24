package com.hellblazer.art.core;

/**
 * Sealed interface representing the result of vigilance testing in ART algorithms.
 * Used to determine whether a category match is accepted or rejected based on the vigilance parameter.
 */
public sealed interface MatchResult permits MatchResult.Accepted, MatchResult.Rejected {
    
    /**
     * Match accepted - input matches the category well enough according to vigilance test.
     * @param matchValue the computed match value (typically |input ∧ weight| / |input|)
     * @param vigilanceThreshold the vigilance threshold that was satisfied
     */
    record Accepted(double matchValue, double vigilanceThreshold) implements MatchResult {
        public Accepted {
            if (Double.isNaN(matchValue) || Double.isInfinite(matchValue) || matchValue < 0.0) {
                throw new IllegalArgumentException("Match value must be non-negative and finite, got: " + matchValue);
            }
            if (Double.isNaN(vigilanceThreshold) || Double.isInfinite(vigilanceThreshold) || 
                vigilanceThreshold < 0.0 || vigilanceThreshold > 1.0) {
                throw new IllegalArgumentException("Vigilance threshold must be in range [0, 1], got: " + vigilanceThreshold);
            }
            if (matchValue < vigilanceThreshold - 1e-10) { // Small tolerance for floating point
                throw new IllegalArgumentException("Match value " + matchValue + 
                    " must be >= vigilance threshold " + vigilanceThreshold);
            }
        }
    }
    
    /**
     * Match rejected - input does not match the category well enough according to vigilance test.
     * @param matchValue the computed match value that failed the test
     * @param vigilanceThreshold the vigilance threshold that was not satisfied
     */
    record Rejected(double matchValue, double vigilanceThreshold) implements MatchResult {
        public Rejected {
            if (Double.isNaN(matchValue) || Double.isInfinite(matchValue) || matchValue < 0.0) {
                throw new IllegalArgumentException("Match value must be non-negative and finite, got: " + matchValue);
            }
            if (Double.isNaN(vigilanceThreshold) || Double.isInfinite(vigilanceThreshold) ||
                vigilanceThreshold < 0.0 || vigilanceThreshold > 1.0) {
                throw new IllegalArgumentException("Vigilance threshold must be in range [0, 1], got: " + vigilanceThreshold);
            }
            if (matchValue >= vigilanceThreshold + 1e-10) { // Small tolerance for floating point
                throw new IllegalArgumentException("Match value " + matchValue + 
                    " must be < vigilance threshold " + vigilanceThreshold + " for rejection");
            }
        }
    }
    
    /**
     * Check if this match result represents acceptance.
     * @return true if the match was accepted, false otherwise
     */
    default boolean isAccepted() {
        return this instanceof Accepted;
    }
    
    /**
     * Check if this match result represents rejection.
     * @return true if the match was rejected, false otherwise
     */
    default boolean isRejected() {
        return this instanceof Rejected;
    }
    
    /**
     * Get the match value regardless of acceptance/rejection.
     * @return the computed match value
     */
    default double getMatchValue() {
        return switch (this) {
            case Accepted accepted -> accepted.matchValue();
            case Rejected rejected -> rejected.matchValue();
        };
    }
    
    /**
     * Get the vigilance threshold used in the test.
     * @return the vigilance threshold
     */
    default double getVigilanceThreshold() {
        return switch (this) {
            case Accepted accepted -> accepted.vigilanceThreshold();
            case Rejected rejected -> rejected.vigilanceThreshold();
        };
    }
    
    /**
     * Convenient pattern matching for match results.
     * @param onAccepted handler for accepted matches
     * @param onRejected handler for rejected matches
     * @param <T> the return type
     * @return the result of the appropriate handler
     */
    default <T> T match(java.util.function.Function<Accepted, T> onAccepted,
                       java.util.function.Function<Rejected, T> onRejected) {
        return switch (this) {
            case Accepted accepted -> onAccepted.apply(accepted);
            case Rejected rejected -> onRejected.apply(rejected);
        };
    }
}