package com.hellblazer.art.laminar.impl;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.laminar.core.IResonanceController;
import com.hellblazer.art.laminar.parameters.IResonanceParameters;
import com.hellblazer.art.laminar.events.ResetEvent;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Default implementation of resonance controller for ART matching and reset.
 * Implements vigilance-based matching and category search order management.
 *
 * @author Hal Hildebrand
 */
public class DefaultResonanceController implements IResonanceController {

    private double vigilance;
    private Pattern attentionWeights;
    private final List<ResetEvent> resetHistory;
    private final Map<Integer, Integer> categoryAccessCount;
    private final Map<Integer, Double> categoryMatchScores;
    private int[] focusedFeatures;

    public DefaultResonanceController() {
        this(0.8);
    }

    public DefaultResonanceController(double vigilance) {
        this.vigilance = vigilance;
        this.resetHistory = new CopyOnWriteArrayList<>();
        this.categoryAccessCount = new HashMap<>();
        this.categoryMatchScores = new HashMap<>();
        this.focusedFeatures = null;
    }

    @Override
    public boolean isResonant(Pattern bottomUp, Pattern topDown, IResonanceParameters parameters) {
        var matchScore = calculateMatch(bottomUp, topDown);
        return matchScore >= parameters.getVigilance();
    }

    @Override
    public double calculateMatch(Pattern bottomUp, Pattern topDown) {
        // Apply attention if focused features are set
        var attendedBottomUp = focusedFeatures != null ?
                applyFocusedAttention(bottomUp) : bottomUp;

        // ART match function: |I ∧ F| / |I|
        var intersection = 0.0;
        var inputNorm = 0.0;

        var minDim = Math.min(attendedBottomUp.dimension(), topDown.dimension());

        for (int i = 0; i < minDim; i++) {
            var inputValue = attendedBottomUp.get(i);
            var expectationValue = topDown.get(i);

            // Fuzzy AND operation
            intersection += Math.min(inputValue, expectationValue);
            inputNorm += inputValue;
        }

        // Handle remaining dimensions if sizes differ
        for (int i = minDim; i < attendedBottomUp.dimension(); i++) {
            inputNorm += attendedBottomUp.get(i);
        }

        // Prevent division by zero
        if (inputNorm == 0.0) {
            return 0.0;
        }

        return intersection / inputNorm;
    }

    @Override
    public double getVigilance() {
        return vigilance;
    }

    @Override
    public void setVigilance(double vigilance) {
        this.vigilance = Math.max(0.0, Math.min(1.0, vigilance));
    }

    @Override
    public void focusAttention(int[] features) {
        this.focusedFeatures = features != null ? features.clone() : null;
        updateAttentionWeights();
    }

    @Override
    public Pattern getAttentionWeights() {
        if (attentionWeights == null) {
            updateAttentionWeights();
        }
        return attentionWeights;
    }

    @Override
    public Pattern applyAttention(Pattern input) {
        if (attentionWeights == null) {
            return input;
        }

        var result = new double[input.dimension()];
        var attentionDim = attentionWeights.dimension();

        for (int i = 0; i < input.dimension(); i++) {
            var weight = i < attentionDim ? attentionWeights.get(i) : 1.0;
            result[i] = input.get(i) * weight;
        }

        return new DenseVector(result);
    }

    @Override
    public boolean shouldReset(double matchScore) {
        return matchScore < vigilance;
    }

    @Override
    public void reset(int categoryIndex) {
        var event = new ResetEvent(categoryIndex, "Vigilance failure",
                                  categoryMatchScores.getOrDefault(categoryIndex, 0.0));
        resetHistory.add(event);

        // Track reset for search order
        categoryAccessCount.merge(categoryIndex, 1, Integer::sum);
    }

    @Override
    public List<ResetEvent> getResetHistory() {
        return new ArrayList<>(resetHistory);
    }

    @Override
    public int getNextCategory(Set<Integer> excludedCategories) {
        // Find category with highest match score not in excluded set
        var bestCategory = -1;
        var bestScore = -1.0;

        for (var entry : categoryMatchScores.entrySet()) {
            var category = entry.getKey();
            var score = entry.getValue();

            if (!excludedCategories.contains(category) && score > bestScore) {
                bestScore = score;
                bestCategory = category;
            }
        }

        return bestCategory;
    }

    @Override
    public void reinforceSearchOrder(int categoryIndex) {
        // Successful match - boost this category's priority
        categoryMatchScores.merge(categoryIndex, 0.1,
                                  (old, boost) -> Math.min(1.0, old + boost));
    }

    /**
     * Update match score for a category.
     */
    public void updateMatchScore(int categoryIndex, double score) {
        categoryMatchScores.put(categoryIndex, score);
    }

    /**
     * Clear all match scores.
     */
    public void clearMatchScores() {
        categoryMatchScores.clear();
    }

    /**
     * Clear reset history.
     */
    public void clearHistory() {
        resetHistory.clear();
        categoryAccessCount.clear();
    }

    /**
     * Get categories sorted by match score.
     */
    public List<Integer> getCategoriesByMatchScore() {
        return categoryMatchScores.entrySet().stream()
                .sorted(Map.Entry.<Integer, Double>comparingByValue().reversed())
                .map(Map.Entry::getKey)
                .toList();
    }

    /**
     * Apply focused attention to specific features.
     */
    private Pattern applyFocusedAttention(Pattern input) {
        if (focusedFeatures == null || focusedFeatures.length == 0) {
            return input;
        }

        var result = new double[input.dimension()];
        var focusSet = new HashSet<Integer>();
        for (int f : focusedFeatures) {
            if (f >= 0 && f < input.dimension()) {
                focusSet.add(f);
            }
        }

        for (int i = 0; i < input.dimension(); i++) {
            if (focusSet.contains(i)) {
                result[i] = input.get(i);
            } else {
                result[i] = input.get(i) * 0.1; // Suppress unfocused features
            }
        }

        return new DenseVector(result);
    }

    /**
     * Update attention weights based on focused features.
     */
    private void updateAttentionWeights() {
        if (focusedFeatures == null || focusedFeatures.length == 0) {
            attentionWeights = null;
            return;
        }

        // Find max index to determine weight array size
        var maxIndex = 0;
        for (int f : focusedFeatures) {
            maxIndex = Math.max(maxIndex, f);
        }

        var weights = new double[maxIndex + 1];
        Arrays.fill(weights, 0.1); // Background attention

        // Set focused features to full attention
        for (int f : focusedFeatures) {
            if (f >= 0 && f < weights.length) {
                weights[f] = 1.0;
            }
        }

        attentionWeights = new DenseVector(weights);
    }

    /**
     * Get total number of resets.
     */
    public int getResetCount() {
        return resetHistory.size();
    }

    /**
     * Get reset count for specific category.
     */
    public int getResetCount(int categoryIndex) {
        return (int) resetHistory.stream()
                .filter(e -> e.getCategoryIndex() == categoryIndex)
                .count();
    }
}