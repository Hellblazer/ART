package com.hellblazer.art.hybrid.pan.decision;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.hybrid.pan.parameters.PANParameters;

/**
 * Dual-Criterion Decision System for PAN.
 *
 * According to the paper, PAN uses both STM (Short-Term Memory) and
 * LTM (Long-Term Memory) criteria to make decisions about pattern matching
 * and learning.
 *
 * Decision types:
 * - RESONATE: Both STM and LTM agree on match
 * - LEARN_NEW: Novel pattern requiring new category
 * - ADAPT_EXISTING: Partial match requiring adaptation
 * - REJECT: Pattern doesn't meet criteria
 */
public class DualCriterionDecisionSystem {

    /**
     * Decision outcomes for the dual-criterion system.
     */
    public enum Decision {
        /**
         * Both STM and LTM agree - strong match.
         * Action: Use existing category with minimal update.
         */
        RESONATE,

        /**
         * Novel pattern detected by both criteria.
         * Action: Create new category.
         */
        LEARN_NEW,

        /**
         * Partial match - one criterion matches, other doesn't.
         * Action: Adapt existing category with stronger learning.
         */
        ADAPT_EXISTING,

        /**
         * Pattern rejected by criteria.
         * Action: No learning, possibly noise.
         */
        REJECT
    }

    /**
     * Decision context containing all relevant information.
     */
    public static class DecisionContext {
        public final int categoryId;
        public final double stmResonance;
        public final double ltmConfidence;
        public final double combinedActivation;
        public final Decision decision;
        public final String reasoning;

        public DecisionContext(int categoryId, double stmResonance, double ltmConfidence,
                              double combinedActivation, Decision decision, String reasoning) {
            this.categoryId = categoryId;
            this.stmResonance = stmResonance;
            this.ltmConfidence = ltmConfidence;
            this.combinedActivation = combinedActivation;
            this.decision = decision;
            this.reasoning = reasoning;
        }
    }

    // Thresholds for decision making
    private final double stmThreshold;
    private final double ltmThreshold;
    private final double combinedThreshold;

    /**
     * Create decision system with default thresholds.
     */
    public DualCriterionDecisionSystem(PANParameters parameters) {
        // STM threshold is typically the vigilance parameter
        this.stmThreshold = parameters.vigilance();

        // LTM threshold is typically slightly higher for stability
        this.ltmThreshold = Math.min(1.0, parameters.vigilance() * 1.1);

        // Combined threshold for final decision
        this.combinedThreshold = parameters.vigilance();
    }

    /**
     * Create decision system with custom thresholds.
     */
    public DualCriterionDecisionSystem(double stmThreshold, double ltmThreshold,
                                      double combinedThreshold) {
        this.stmThreshold = stmThreshold;
        this.ltmThreshold = ltmThreshold;
        this.combinedThreshold = combinedThreshold;
    }

    /**
     * Make a decision based on dual criteria.
     *
     * @param stmResonance STM resonance value [0,1]
     * @param ltmConfidence LTM confidence value [0,1]
     * @return Decision based on both criteria
     */
    public Decision makeDecision(double stmResonance, double ltmConfidence) {
        boolean stmMatch = stmResonance >= stmThreshold;
        boolean ltmMatch = ltmConfidence >= ltmThreshold;

        if (stmMatch && ltmMatch) {
            // Both criteria agree - strong match
            return Decision.RESONATE;
        } else if (!stmMatch && !ltmMatch) {
            // Neither criterion matches - novel pattern
            return Decision.LEARN_NEW;
        } else {
            // One matches, other doesn't - partial match
            return Decision.ADAPT_EXISTING;
        }
    }

    /**
     * Make a decision with detailed context.
     *
     * @param categoryId The category being evaluated
     * @param stmResonance STM resonance value [0,1]
     * @param ltmConfidence LTM confidence value [0,1]
     * @return Detailed decision context
     */
    public DecisionContext makeDetailedDecision(int categoryId, double stmResonance,
                                               double ltmConfidence) {
        // Compute combined activation
        double combinedActivation = 0.7 * stmResonance + 0.3 * ltmConfidence;

        // Make basic decision
        Decision decision = makeDecision(stmResonance, ltmConfidence);

        // Build reasoning string
        String reasoning = buildReasoning(decision, stmResonance, ltmConfidence);

        return new DecisionContext(categoryId, stmResonance, ltmConfidence,
                                 combinedActivation, decision, reasoning);
    }

    /**
     * Make a decision for category creation.
     *
     * @param bestStmResonance Best STM match found
     * @param bestLtmConfidence Best LTM confidence found
     * @param maxCategories Maximum allowed categories
     * @param currentCategories Current number of categories
     * @return Whether to create a new category
     */
    public boolean shouldCreateNewCategory(double bestStmResonance, double bestLtmConfidence,
                                          int maxCategories, int currentCategories) {
        // Check if we have room for new categories
        if (currentCategories >= maxCategories) {
            return false;
        }

        // Check if pattern is sufficiently novel
        Decision decision = makeDecision(bestStmResonance, bestLtmConfidence);
        return decision == Decision.LEARN_NEW;
    }

    /**
     * Determine learning rate based on decision.
     *
     * @param decision The decision made
     * @param baseLearningRate Base learning rate
     * @return Adjusted learning rate
     */
    public double adjustLearningRate(Decision decision, double baseLearningRate) {
        switch (decision) {
            case RESONATE:
                // Strong match - minimal learning
                return baseLearningRate * 0.1;

            case ADAPT_EXISTING:
                // Partial match - moderate learning
                return baseLearningRate * 0.5;

            case LEARN_NEW:
                // Novel pattern - full learning
                return baseLearningRate;

            case REJECT:
                // No learning
                return 0.0;

            default:
                return baseLearningRate;
        }
    }

    /**
     * Check if combined activation passes threshold.
     *
     * @param stmResonance STM resonance
     * @param ltmConfidence LTM confidence
     * @return True if combined activation passes threshold
     */
    public boolean passesCombinedThreshold(double stmResonance, double ltmConfidence) {
        double combined = 0.7 * stmResonance + 0.3 * ltmConfidence;
        return combined >= combinedThreshold;
    }

    /**
     * Evaluate multiple candidates and select best.
     *
     * @param candidates Array of [categoryId, stmResonance, ltmConfidence] triplets
     * @return Best category index or -1 if none suitable
     */
    public int selectBestCategory(double[][] candidates) {
        int bestCategory = -1;
        double bestScore = -1.0;
        Decision bestDecision = Decision.REJECT;

        for (double[] candidate : candidates) {
            int categoryId = (int) candidate[0];
            double stm = candidate[1];
            double ltm = candidate[2];

            Decision decision = makeDecision(stm, ltm);
            double score = 0.7 * stm + 0.3 * ltm;

            // Prefer RESONATE > ADAPT_EXISTING > LEARN_NEW > REJECT
            if (decision == Decision.RESONATE) {
                if (bestDecision != Decision.RESONATE || score > bestScore) {
                    bestCategory = categoryId;
                    bestScore = score;
                    bestDecision = decision;
                }
            } else if (decision == Decision.ADAPT_EXISTING &&
                     bestDecision != Decision.RESONATE) {
                if (bestDecision != Decision.ADAPT_EXISTING || score > bestScore) {
                    bestCategory = categoryId;
                    bestScore = score;
                    bestDecision = decision;
                }
            }
        }

        return bestCategory;
    }

    /**
     * Build reasoning string for a decision.
     */
    private String buildReasoning(Decision decision, double stmResonance, double ltmConfidence) {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("STM=%.3f (%s threshold %.3f), ",
            stmResonance,
            stmResonance >= stmThreshold ? "passes" : "fails",
            stmThreshold));
        sb.append(String.format("LTM=%.3f (%s threshold %.3f) -> ",
            ltmConfidence,
            ltmConfidence >= ltmThreshold ? "passes" : "fails",
            ltmThreshold));

        switch (decision) {
            case RESONATE:
                sb.append("Strong match (both criteria pass)");
                break;
            case LEARN_NEW:
                sb.append("Novel pattern (both criteria fail)");
                break;
            case ADAPT_EXISTING:
                sb.append("Partial match (mixed criteria)");
                break;
            case REJECT:
                sb.append("Rejected (below thresholds)");
                break;
        }

        return sb.toString();
    }

    /**
     * Get diagnostic information.
     *
     * @return Diagnostic map
     */
    public java.util.Map<String, Object> getDiagnostics() {
        java.util.Map<String, Object> diagnostics = new java.util.HashMap<>();
        diagnostics.put("stmThreshold", stmThreshold);
        diagnostics.put("ltmThreshold", ltmThreshold);
        diagnostics.put("combinedThreshold", combinedThreshold);
        return diagnostics;
    }
}