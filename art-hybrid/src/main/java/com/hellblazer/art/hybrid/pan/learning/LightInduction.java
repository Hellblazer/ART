package com.hellblazer.art.hybrid.pan.learning;

import java.util.HashMap;
import java.util.Map;

/**
 * Light Induction system for PAN network.
 *
 * According to the PAN paper, light induction (λ) represents the expected
 * influence on learning. It biases the learning process to help distinguish
 * between novel patterns and well-known patterns.
 *
 * Paper notation:
 * - λ_j: Light induction factor for category j
 * - Higher λ for novel patterns (encourages new learning)
 * - Lower λ for well-known patterns (maintains stability)
 */
public class LightInduction {

    // Base lambda value (small positive value as per paper)
    private final double baseLambda;

    // Track historical influence of each category
    private final Map<Integer, Double> categoryInfluence;

    // Track novelty scores for categories
    private final Map<Integer, Double> categoryNovelty;

    // Decay factor for historical influence
    private static final double INFLUENCE_DECAY = 0.9;

    // Learning rate for influence updates
    private static final double INFLUENCE_LEARNING_RATE = 0.1;

    /**
     * Create light induction system with default base lambda.
     */
    public LightInduction() {
        this(0.01); // Default small positive value
    }

    /**
     * Create light induction system with specified base lambda.
     *
     * @param baseLambda The base light induction factor
     */
    public LightInduction(double baseLambda) {
        if (baseLambda <= 0) {
            throw new IllegalArgumentException("Base lambda must be positive");
        }
        this.baseLambda = baseLambda;
        this.categoryInfluence = new HashMap<>();
        this.categoryNovelty = new HashMap<>();
    }

    /**
     * Compute light induction factor for a category.
     *
     * Per paper: λ represents the expected influence on learning.
     * - Higher for novel patterns (encourages exploration)
     * - Lower for well-known patterns (maintains stability)
     *
     * @param categoryId The category index
     * @param stmActivity Short-term memory activity level [0,1]
     * @param ltmConfidence Long-term memory confidence [0,1]
     * @return Light induction factor λ
     */
    public double computeLambda(int categoryId, double stmActivity, double ltmConfidence) {
        // Compute novelty: high when both STM and LTM are low
        double novelty = 1.0 - (0.5 * stmActivity + 0.5 * ltmConfidence);

        // Get historical influence (default 0 for new categories)
        double historicalInfluence = categoryInfluence.getOrDefault(categoryId, 0.0);

        // Get current novelty score
        double currentNovelty = categoryNovelty.getOrDefault(categoryId, novelty);

        // Update novelty tracking with exponential moving average
        categoryNovelty.put(categoryId, 0.7 * currentNovelty + 0.3 * novelty);

        // Compute lambda based on:
        // 1. Base value (always present)
        // 2. Novelty bonus (higher for novel patterns)
        // 3. Influence penalty (lower for frequently used categories)
        double lambda = baseLambda * (1.0 + novelty - 0.5 * historicalInfluence);

        // Ensure lambda remains positive and bounded
        return Math.max(baseLambda * 0.1, Math.min(baseLambda * 10.0, lambda));
    }

    /**
     * Compute lambda for supervised learning with target.
     *
     * @param categoryId The category index
     * @param error The prediction error
     * @param ltmConfidence Long-term memory confidence
     * @return Light induction factor for supervised learning
     */
    public double computeSupervisedLambda(int categoryId, double error, double ltmConfidence) {
        // For supervised learning, lambda is influenced by error magnitude
        double errorInfluence = Math.abs(error);

        // High error and low confidence -> high lambda (need more learning)
        // Low error and high confidence -> low lambda (already well-learned)
        double novelty = errorInfluence * (1.0 - ltmConfidence);

        double historicalInfluence = categoryInfluence.getOrDefault(categoryId, 0.0);

        // Supervised lambda emphasizes error correction
        double lambda = baseLambda * (1.0 + 2.0 * novelty - historicalInfluence);

        return Math.max(baseLambda * 0.1, Math.min(baseLambda * 20.0, lambda));
    }

    /**
     * Update the influence tracking after learning.
     *
     * @param categoryId The category that was updated
     * @param learningOutcome The learning outcome (e.g., error reduction)
     */
    public void updateInfluence(int categoryId, double learningOutcome) {
        // Get current influence
        double currentInfluence = categoryInfluence.getOrDefault(categoryId, 0.0);

        // Update with decay and new learning
        double newInfluence = INFLUENCE_DECAY * currentInfluence +
                            INFLUENCE_LEARNING_RATE * Math.abs(learningOutcome);

        // Normalize to [0,1]
        newInfluence = Math.min(1.0, newInfluence);

        categoryInfluence.put(categoryId, newInfluence);
    }

    /**
     * Compute light induction for experience replay.
     *
     * @param timeSinceExperience Time since the experience (in milliseconds)
     * @param originalLambda The original lambda when experience was recorded
     * @return Adjusted lambda for replay
     */
    public double computeReplayLambda(long timeSinceExperience, double originalLambda) {
        // Decay lambda over time for replay
        // Recent experiences get closer to original lambda
        // Old experiences get reduced lambda (less influence)
        double timeFactor = Math.exp(-timeSinceExperience / 600000.0); // 10 minute half-life
        return originalLambda * timeFactor;
    }

    /**
     * Reset influence for a category (e.g., after significant change).
     *
     * @param categoryId The category to reset
     */
    public void resetCategoryInfluence(int categoryId) {
        categoryInfluence.remove(categoryId);
        categoryNovelty.remove(categoryId);
    }

    /**
     * Clear all tracked influences.
     */
    public void clear() {
        categoryInfluence.clear();
        categoryNovelty.clear();
    }

    /**
     * Get the current base lambda value.
     *
     * @return Base lambda
     */
    public double getBaseLambda() {
        return baseLambda;
    }

    /**
     * Get diagnostics information.
     *
     * @return Map of diagnostic data
     */
    public Map<String, Object> getDiagnostics() {
        Map<String, Object> diagnostics = new HashMap<>();
        diagnostics.put("baseLambda", baseLambda);
        diagnostics.put("numCategories", categoryInfluence.size());
        diagnostics.put("avgInfluence", categoryInfluence.values().stream()
            .mapToDouble(Double::doubleValue)
            .average()
            .orElse(0.0));
        diagnostics.put("avgNovelty", categoryNovelty.values().stream()
            .mapToDouble(Double::doubleValue)
            .average()
            .orElse(0.0));
        return diagnostics;
    }
}