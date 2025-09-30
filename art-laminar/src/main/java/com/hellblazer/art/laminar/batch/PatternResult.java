package com.hellblazer.art.laminar.batch;

import com.hellblazer.art.core.Pattern;

/**
 * Result for a single pattern within a batch processing operation.
 *
 * <p>Contains the output pattern and processing metadata for one input pattern.
 * Extracted from {@link BatchResult} for individual pattern inspection.
 *
 * @param output output pattern (expectation if resonating, else processed input)
 * @param categoryId winning category ID (-1 if mismatch/no resonance)
 * @param activationValue category activation value (0.0 if mismatch)
 * @param resonating whether this pattern achieved resonance
 *
 * @see BatchResult#getResult(int)
 * @author Claude Code
 */
public record PatternResult(
    Pattern output,
    int categoryId,
    double activationValue,
    boolean resonating
) {
    /**
     * Validate pattern result fields.
     *
     * @throws NullPointerException if output is null
     * @throws IllegalArgumentException if resonating but categoryId < 0
     * @throws IllegalArgumentException if activationValue < 0 or > 1
     */
    public PatternResult {
        if (output == null) {
            throw new NullPointerException("output cannot be null");
        }
        if (resonating && categoryId < 0) {
            throw new IllegalArgumentException(
                "Resonating pattern must have valid category ID (got " + categoryId + ")");
        }
        if (activationValue < 0.0 || activationValue > 1.0) {
            throw new IllegalArgumentException(
                "Activation value must be in [0,1] (got " + activationValue + ")");
        }
    }

    /**
     * Check if this pattern created a new category.
     * Must be combined with batch-level tracking to determine if truly new.
     *
     * @return true if resonating with non-zero learning
     */
    public boolean mightBeNewCategory() {
        return resonating && categoryId >= 0;
    }

    /**
     * Get resonance quality metric.
     *
     * @return activation value if resonating, else 0.0
     */
    public double getResonanceQuality() {
        return resonating ? activationValue : 0.0;
    }

    @Override
    public String toString() {
        if (resonating) {
            return String.format("PatternResult[category=%d, activation=%.3f, resonating=true]",
                categoryId, activationValue);
        } else {
            return "PatternResult[resonating=false]";
        }
    }
}