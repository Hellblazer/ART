package com.hellblazer.art.laminar.core;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.IResonanceParameters;
import com.hellblazer.art.laminar.events.ResetEvent;
import java.util.List;
import java.util.Set;

/**
 * Interface for controlling resonance and attention in laminar circuits.
 * Implements the matching and reset mechanisms of ART.
 *
 * @author Hal Hildebrand
 */
public interface IResonanceController {

    // === Resonance Detection ===

    /**
     * Check if current circuit state is resonant.
     *
     * @param bottomUp Bottom-up input pattern
     * @param topDown Top-down expectation pattern
     * @param parameters Resonance parameters
     * @return true if resonance achieved
     */
    boolean isResonant(Pattern bottomUp, Pattern topDown,
                       IResonanceParameters parameters);

    /**
     * Calculate match score between patterns.
     *
     * @param bottomUp Bottom-up input
     * @param topDown Top-down expectation
     * @return Match score (0.0 to 1.0)
     */
    double calculateMatch(Pattern bottomUp, Pattern topDown);

    /**
     * Get current vigilance parameter.
     *
     * @return Vigilance value
     */
    double getVigilance();

    /**
     * Set vigilance parameter.
     *
     * @param vigilance New vigilance value (0.0 to 1.0)
     */
    void setVigilance(double vigilance);

    // === Attention Control ===

    /**
     * Focus attention on specific features.
     *
     * @param features Feature indices to attend
     */
    void focusAttention(int[] features);

    /**
     * Get current attention weights.
     *
     * @return Attention weight pattern
     */
    Pattern getAttentionWeights();

    /**
     * Apply attention gain to a pattern.
     *
     * @param input Input pattern
     * @return Attention-modulated pattern
     */
    Pattern applyAttention(Pattern input);

    // === Reset Mechanism ===

    /**
     * Check if reset should occur.
     *
     * @param matchScore Current match score
     * @return true if reset needed
     */
    boolean shouldReset(double matchScore);

    /**
     * Perform reset operation.
     *
     * @param categoryIndex Category to reset
     */
    void reset(int categoryIndex);

    /**
     * Get reset history for debugging.
     *
     * @return List of reset events
     */
    List<ResetEvent> getResetHistory();

    // === Search Control ===

    /**
     * Get next category to try after reset.
     *
     * @param excludedCategories Already tried categories
     * @return Next category index, or -1 if none
     */
    int getNextCategory(Set<Integer> excludedCategories);

    /**
     * Update search order based on success.
     *
     * @param categoryIndex Successful category
     */
    void reinforceSearchOrder(int categoryIndex);
}