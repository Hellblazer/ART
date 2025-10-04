package com.hellblazer.art.laminar.canonical;

import com.hellblazer.art.core.Pattern;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Generates top-down expectations from category activations.
 *
 * Implements the ART top-down template mechanism where learned
 * category representations generate expected feature patterns.
 *
 * Biological basis: Layer 5 → Layer 6 → Layer 4 feedback pathway
 * in canonical neocortical circuit.
 *
 * @see "A Canonical Laminar Neocortical Circuit..." Raizada & Grossberg (2003)
 * @author Hal Hildebrand
 */
public class PredictionGenerator {

    private final Map<Integer, Pattern> templates;
    private final int featureDimension;
    private final PredictionParameters params;

    /**
     * Create prediction generator for given feature dimension.
     *
     * @param featureDimension number of features in patterns
     * @param params prediction parameters
     */
    public PredictionGenerator(int featureDimension, PredictionParameters params) {
        this.featureDimension = featureDimension;
        this.params = params;
        this.templates = new ConcurrentHashMap<>();
    }

    /**
     * Generate expectation from category activation pattern.
     *
     * Computes weighted sum of learned templates based on category
     * activation strengths. Multiple active categories blend their
     * expectations proportionally.
     *
     * Mathematical formula:
     * <pre>
     * E_i = (Σ_c α_c * T_{c,i}) / (Σ_c α_c) * g
     *
     * where:
     *   E_i     = expectation for feature i
     *   α_c     = activation of category c
     *   T_{c,i} = template weight for category c, feature i
     *   g       = topDownGain (modulatory strength)
     * </pre>
     *
     * Biological interpretation: Layer 5 pyramidal neurons project learned
     * category templates back through Layer 6 to Layer 4, generating expected
     * sensory patterns. This top-down expectation modulates bottom-up processing
     * via the ART matching rule.
     *
     * @param categoryActivation activation strength per category [0,1]^n
     * @return expected feature pattern [0,1]^d where d = featureDimension
     */
    public Pattern generateExpectation(Pattern categoryActivation) {
        var expectation = new double[featureDimension];
        var totalActivation = 0.0;

        // Weighted sum of active category templates
        for (var i = 0; i < categoryActivation.dimension(); i++) {
            var activation = categoryActivation.get(i);
            if (activation > params.expectationThreshold()) {
                var template = getOrCreateTemplate(i);
                for (var j = 0; j < featureDimension; j++) {
                    expectation[j] += activation * template.get(j);
                }
                totalActivation += activation;
            }
        }

        // Normalize by total activation
        if (totalActivation > 0.0) {
            for (var i = 0; i < featureDimension; i++) {
                expectation[i] /= totalActivation;
            }
        }

        // Apply top-down gain modulation
        for (var i = 0; i < featureDimension; i++) {
            expectation[i] *= params.topDownGain();
        }

        return Pattern.of(expectation);
    }

    /**
     * Get learned template for specific category.
     *
     * If category has been committed (learned), returns the learned template.
     * If uncommitted, returns maximally general template (all ones) which
     * matches any input perfectly - this prevents catastrophic forgetting.
     *
     * ART Principle: Uncommitted categories start with all-ones weights,
     * allowing them to match any pattern. First resonant episode commits
     * the category and makes its template specific.
     *
     * @param categoryId category identifier (non-negative integer)
     * @return learned template pattern [0,1]^d or [1,1,...,1] if uncommitted
     */
    public Pattern getCategoryTemplate(int categoryId) {
        return templates.computeIfAbsent(categoryId,
            id -> createUncommittedTemplate());
    }

    /**
     * Update template through learning.
     *
     * Only called during resonance. Implements incremental learning
     * rule: Δw = α(x - w) which gradually adapts template toward
     * resonant pattern.
     *
     * Mathematical formula:
     * <pre>
     * T_{c,i}(t+1) = T_{c,i}(t) + α * (x_i - T_{c,i}(t))
     *              = (1-α) * T_{c,i}(t) + α * x_i
     *
     * where:
     *   T_{c,i}(t) = template for category c, feature i at time t
     *   α          = learningRate ∈ [0,1]
     *   x_i        = resonant feature value
     * </pre>
     *
     * Learning properties:
     * - α = 1.0: immediate learning (one-shot)
     * - α = 0.1: slow learning (10% per iteration, typical)
     * - Convergence after n iterations: T_n ≈ x + (1-α)^n * (T_0 - x)
     *
     * Critical ART principle: Learning only occurs during resonance
     * (match criterion satisfied), preventing catastrophic forgetting.
     *
     * @param categoryId category to update (non-negative integer)
     * @param feature resonant feature pattern [0,1]^d
     * @param learningRate adaptation rate [0,1], typically 0.05-0.2
     */
    public void updateTemplate(int categoryId, Pattern feature, double learningRate) {
        var template = getCategoryTemplate(categoryId);
        var updated = new double[featureDimension];

        for (var i = 0; i < featureDimension; i++) {
            var current = template.get(i);
            var target = feature.get(i);
            updated[i] = current + learningRate * (target - current);
        }

        templates.put(categoryId, Pattern.of(updated));
    }

    /**
     * Check if category has been committed (learned).
     *
     * A category is committed if it has undergone at least one learning
     * episode (resonance). Uncommitted categories retain their initial
     * all-ones weights and will match any input perfectly.
     *
     * @param categoryId category to check (non-negative integer)
     * @return true if category has learned template, false if uncommitted
     */
    public boolean isCommitted(int categoryId) {
        return templates.containsKey(categoryId);
    }

    /**
     * Get number of committed categories.
     *
     * Committed categories have undergone learning and represent
     * distinct learned patterns. The number of committed categories
     * grows as the network encounters novel patterns.
     *
     * @return count of categories with learned templates (≥ 0)
     */
    public int getCommittedCount() {
        return templates.size();
    }

    /**
     * Reset to initial state (clear all templates).
     *
     * Removes all learned templates, returning network to initial
     * uncommitted state. Useful for starting new learning episodes
     * or clearing learned knowledge.
     */
    public void reset() {
        templates.clear();
    }

    /**
     * Create uncommitted template (maximally general).
     *
     * Uncommitted templates initialized to all ones so they
     * match any input perfectly (no constraint). First resonant
     * episode makes template specific.
     *
     * @return uncommitted template [1,1,...,1]
     */
    private Pattern createUncommittedTemplate() {
        var template = new double[featureDimension];
        for (var i = 0; i < featureDimension; i++) {
            template[i] = 1.0;
        }
        return Pattern.of(template);
    }

    /**
     * Get or create template for category.
     */
    private Pattern getOrCreateTemplate(int categoryId) {
        return templates.computeIfAbsent(categoryId,
            id -> createUncommittedTemplate());
    }
}
