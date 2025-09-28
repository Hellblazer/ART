package com.hellblazer.art.temporal.learning;

import java.util.Arrays;

/**
 * Competitive Instar Learning implementation
 * Self-normalizing adaptive weight updates for masking field learning
 * Based on Equation 12 from Kazerounian & Grossberg (2014)
 */
public class CompetitiveInstarLearning {

    private final float learningRate;
    private final float minWeight;
    private final float maxWeight;
    private final boolean selfNormalizing;

    // Learning state
    private float[][] weights;
    private int numCategories;
    private int inputDimension;
    private float[] categoryActivations;

    public CompetitiveInstarLearning(float learningRate, int numCategories, int inputDimension) {
        this(learningRate, numCategories, inputDimension, true);
    }

    public CompetitiveInstarLearning(float learningRate, int numCategories, int inputDimension,
                                    boolean selfNormalizing) {
        this.learningRate = learningRate;
        this.numCategories = numCategories;
        this.inputDimension = inputDimension;
        this.selfNormalizing = selfNormalizing;
        this.minWeight = 0.0f;
        this.maxWeight = 1.0f;

        initializeWeights();
    }

    /**
     * Update weights using competitive instar learning rule
     * dWij/dt = αf(cj)[(1-Wij)xi - Wij∑xk]
     *
     * @param input Input pattern
     * @param categoryActivations Activation of each category
     * @param winnerIndex Index of winning category
     */
    public void updateWeights(float[] input, float[] categoryActivations, int winnerIndex) {
        this.categoryActivations = categoryActivations;

        if (winnerIndex < 0 || winnerIndex >= numCategories) {
            return;  // No winner, no learning
        }

        var winnerActivation = categoryActivations[winnerIndex];
        if (winnerActivation <= 0) {
            return;  // No significant activation
        }

        // Compute sum of inputs for normalization
        var inputSum = selfNormalizing ? sumArray(input) : 1.0f;

        // Update weights for winning category
        updateCategoryWeights(winnerIndex, input, winnerActivation, inputSum);

        // Optional: Update other categories with smaller learning (soft competition)
        if (isSoftCompetition()) {
            updateNonWinnerWeights(input, categoryActivations, winnerIndex, inputSum);
        }
    }

    /**
     * Update weights for a specific category
     */
    private void updateCategoryWeights(int categoryIndex, float[] input,
                                      float activation, float inputSum) {
        var categoryWeights = weights[categoryIndex];

        for (int i = 0; i < inputDimension; i++) {
            // Competitive instar learning rule
            // dW/dt = α * f(c) * [(1-W)x - W*Σx]
            var currentWeight = categoryWeights[i];
            var inputValue = i < input.length ? input[i] : 0;

            var learningSignal = threshold(activation);
            var excitatory = (1 - currentWeight) * inputValue;
            var inhibitory = currentWeight * inputSum;

            var deltaWeight = learningRate * learningSignal * (excitatory - inhibitory);

            // Update weight
            categoryWeights[i] = currentWeight + deltaWeight;

            // Bound weights
            categoryWeights[i] = Math.max(minWeight, Math.min(maxWeight, categoryWeights[i]));
        }

        // Normalize weights if required
        if (selfNormalizing) {
            normalizeWeights(categoryIndex);
        }
    }

    /**
     * Optional soft competition: update non-winners with reduced learning
     */
    private void updateNonWinnerWeights(float[] input, float[] activations,
                                       int winnerIndex, float inputSum) {
        var softLearningRate = learningRate * 0.1f;  // 10% of winner's learning rate

        for (int cat = 0; cat < numCategories; cat++) {
            if (cat != winnerIndex && activations[cat] > 0) {
                var activation = activations[cat];
                var categoryWeights = weights[cat];

                for (int i = 0; i < inputDimension; i++) {
                    var currentWeight = categoryWeights[i];
                    var inputValue = i < input.length ? input[i] : 0;

                    var learningSignal = threshold(activation);
                    var excitatory = (1 - currentWeight) * inputValue;
                    var inhibitory = currentWeight * inputSum;

                    var deltaWeight = softLearningRate * learningSignal * (excitatory - inhibitory);

                    categoryWeights[i] = currentWeight + deltaWeight;
                    categoryWeights[i] = Math.max(minWeight, Math.min(maxWeight, categoryWeights[i]));
                }

                if (selfNormalizing) {
                    normalizeWeights(cat);
                }
            }
        }
    }

    /**
     * Compute activations for all categories given input
     */
    public float[] computeActivations(float[] input) {
        var activations = new float[numCategories];

        for (int cat = 0; cat < numCategories; cat++) {
            activations[cat] = computeCategoryActivation(input, cat);
        }

        return activations;
    }

    /**
     * Compute activation for a specific category
     */
    public float computeCategoryActivation(float[] input, int categoryIndex) {
        var categoryWeights = weights[categoryIndex];
        var activation = 0.0f;

        for (int i = 0; i < Math.min(input.length, inputDimension); i++) {
            activation += input[i] * categoryWeights[i];
        }

        return activation;
    }

    /**
     * Get learned weights for a category
     */
    public float[] getCategoryWeights(int categoryIndex) {
        return Arrays.copyOf(weights[categoryIndex], inputDimension);
    }

    /**
     * Set weights for a category (for initialization or loading)
     */
    public void setCategoryWeights(int categoryIndex, float[] newWeights) {
        System.arraycopy(newWeights, 0, weights[categoryIndex], 0,
                        Math.min(newWeights.length, inputDimension));
        if (selfNormalizing) {
            normalizeWeights(categoryIndex);
        }
    }

    /**
     * Reset all weights to initial random values
     */
    public void resetWeights() {
        initializeWeights();
    }

    /**
     * Reset weights for a specific category
     */
    public void resetCategoryWeights(int categoryIndex) {
        for (int i = 0; i < inputDimension; i++) {
            weights[categoryIndex][i] = (float) (Math.random() * 0.1);
        }
        if (selfNormalizing) {
            normalizeWeights(categoryIndex);
        }
    }

    // Helper methods

    private void initializeWeights() {
        weights = new float[numCategories][inputDimension];
        categoryActivations = new float[numCategories];

        // Initialize with small random values
        for (int cat = 0; cat < numCategories; cat++) {
            for (int i = 0; i < inputDimension; i++) {
                weights[cat][i] = (float) (Math.random() * 0.1);
            }
            if (selfNormalizing) {
                normalizeWeights(cat);
            }
        }
    }

    private void normalizeWeights(int categoryIndex) {
        var sum = sumArray(weights[categoryIndex]);
        if (sum > 0) {
            for (int i = 0; i < inputDimension; i++) {
                weights[categoryIndex][i] /= sum;
            }
        }
    }

    private float sumArray(float[] array) {
        var sum = 0.0f;
        for (var val : array) {
            sum += Math.abs(val);
        }
        return sum;
    }

    private float threshold(float x) {
        // Sigmoid threshold function for learning signal
        return x > 0 ? (float) (1.0 / (1.0 + Math.exp(-x))) : 0;
    }

    private boolean isSoftCompetition() {
        // Could be made configurable
        return false;  // Default to hard competition (winner-take-all)
    }

    // Getters for monitoring

    public float getLearningRate() {
        return learningRate;
    }

    public int getNumCategories() {
        return numCategories;
    }

    public int getInputDimension() {
        return inputDimension;
    }

    public float[][] getAllWeights() {
        var copy = new float[numCategories][inputDimension];
        for (int i = 0; i < numCategories; i++) {
            System.arraycopy(weights[i], 0, copy[i], 0, inputDimension);
        }
        return copy;
    }
}