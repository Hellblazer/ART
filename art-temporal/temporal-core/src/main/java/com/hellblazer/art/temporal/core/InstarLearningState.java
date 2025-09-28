package com.hellblazer.art.temporal.core;

/**
 * State representation for competitive instar learning.
 * Based on Equation 12 from Kazerounian & Grossberg (2014):
 * dW_ij/dt = L * Y_j * (X_i * Z_i - W_ij)
 */
public class InstarLearningState extends State {
    private final double[][] weights;          // W_ij: adaptive filter weights
    private final double[] categoryActivations; // Y_j: masking field activations
    private final boolean[] categoryUsage;      // Track which categories have been used
    private final double[] normalizationFactors; // For weight normalization

    public InstarLearningState(double[][] weights) {
        this.weights = deepClone(weights);
        this.categoryActivations = new double[weights.length];
        this.categoryUsage = new boolean[weights.length];
        this.normalizationFactors = computeNormalizationFactors();
    }

    public InstarLearningState(int numCategories, int inputDimension) {
        this.weights = new double[numCategories][inputDimension];
        this.categoryActivations = new double[numCategories];
        this.categoryUsage = new boolean[numCategories];
        this.normalizationFactors = new double[numCategories];

        // Initialize weights with small random values
        initializeWeights();
    }

    private void initializeWeights() {
        var random = new java.util.Random(42);
        for (int j = 0; j < weights.length; j++) {
            double sum = 0.0;
            for (int i = 0; i < weights[j].length; i++) {
                weights[j][i] = 0.1 + random.nextDouble() * 0.1;
                sum += weights[j][i];
            }
            // Normalize to sum to 1
            for (int i = 0; i < weights[j].length; i++) {
                weights[j][i] /= sum;
            }
            normalizationFactors[j] = 1.0;
        }
    }

    private double[][] deepClone(double[][] array) {
        var result = new double[array.length][];
        for (int i = 0; i < array.length; i++) {
            result[i] = array[i].clone();
        }
        return result;
    }

    private double[] computeNormalizationFactors() {
        var factors = new double[weights.length];
        for (int j = 0; j < weights.length; j++) {
            double sum = 0.0;
            for (double w : weights[j]) {
                sum += w;
            }
            factors[j] = (sum > 0) ? 1.0 / sum : 1.0;
        }
        return factors;
    }

    public double[][] getWeights() {
        return deepClone(weights);
    }

    public double[] getCategoryActivations() {
        return categoryActivations.clone();
    }

    public boolean[] getCategoryUsage() {
        return categoryUsage.clone();
    }

    /**
     * Get weight vector for specific category.
     */
    public double[] getCategoryWeights(int category) {
        return weights[category].clone();
    }

    /**
     * Set activation for a specific category.
     */
    public void setCategoryActivation(int category, double activation) {
        categoryActivations[category] = activation;
        if (activation > 0) {
            categoryUsage[category] = true;
        }
    }

    /**
     * Compute match between input and category weights.
     */
    public double computeMatch(double[] input, int category) {
        if (input.length != weights[category].length) {
            throw new IllegalArgumentException("Input dimension must match weight dimension");
        }

        double match = 0.0;
        for (int i = 0; i < input.length; i++) {
            match += Math.min(input[i], weights[category][i]);
        }
        return match;
    }

    /**
     * Find best matching category for input.
     */
    public int findBestMatch(double[] input) {
        int bestCategory = -1;
        double bestMatch = -1.0;

        for (int j = 0; j < weights.length; j++) {
            double match = computeMatch(input, j);
            if (match > bestMatch) {
                bestMatch = match;
                bestCategory = j;
            }
        }

        return bestCategory;
    }

    /**
     * Apply weight normalization constraint (weights sum to 1).
     */
    public void normalizeWeights() {
        for (int j = 0; j < weights.length; j++) {
            normalizeCategory(j);
        }
    }

    private void normalizeCategory(int category) {
        double sum = 0.0;
        for (double w : weights[category]) {
            sum += w;
        }

        if (sum > 0) {
            for (int i = 0; i < weights[category].length; i++) {
                weights[category][i] /= sum;
            }
            normalizationFactors[category] = 1.0;
        }
    }

    /**
     * Get number of categories that have been used (activated at least once).
     */
    public int getUsedCategoryCount() {
        int count = 0;
        for (boolean used : categoryUsage) {
            if (used) count++;
        }
        return count;
    }

    /**
     * Find first unused category for new learning.
     */
    public int findUnusedCategory() {
        for (int j = 0; j < categoryUsage.length; j++) {
            if (!categoryUsage[j]) {
                return j;
            }
        }
        return -1; // All categories used
    }

    /**
     * Reset a category to initial state.
     */
    public void resetCategory(int category) {
        var random = new java.util.Random();
        for (int i = 0; i < weights[category].length; i++) {
            weights[category][i] = 0.1 + random.nextDouble() * 0.1;
        }
        normalizeCategory(category);
        categoryActivations[category] = 0.0;
        categoryUsage[category] = false;
    }

    @Override
    public State add(State other) {
        if (!(other instanceof InstarLearningState s)) {
            throw new IllegalArgumentException("Can only add InstarLearningState to InstarLearningState");
        }

        var result = new double[weights.length][];
        for (int j = 0; j < weights.length; j++) {
            result[j] = vectorizedOperation(weights[j], s.weights[j], (a, b) -> a.add(b));
        }

        return new InstarLearningState(result);
    }

    @Override
    public State scale(double scalar) {
        var result = new double[weights.length][];
        for (int j = 0; j < weights.length; j++) {
            result[j] = new double[weights[j].length];
            for (int i = 0; i < weights[j].length; i++) {
                result[j][i] = weights[j][i] * scalar;
            }
        }
        var newState = new InstarLearningState(result);
        newState.normalizeWeights(); // Maintain normalization constraint
        return newState;
    }

    @Override
    public double distance(State other) {
        if (!(other instanceof InstarLearningState s)) {
            throw new IllegalArgumentException("Can only compute distance to InstarLearningState");
        }

        double sum = 0.0;
        for (int j = 0; j < weights.length; j++) {
            for (int i = 0; i < weights[j].length; i++) {
                double diff = weights[j][i] - s.weights[j][i];
                sum += diff * diff;
            }
        }
        return Math.sqrt(sum);
    }

    @Override
    public int dimension() {
        return weights.length * weights[0].length;
    }

    @Override
    public State copy() {
        var copy = new InstarLearningState(weights);
        System.arraycopy(categoryActivations, 0, copy.categoryActivations, 0, categoryActivations.length);
        System.arraycopy(categoryUsage, 0, copy.categoryUsage, 0, categoryUsage.length);
        return copy;
    }

    @Override
    public double[] toArray() {
        // Flatten weight matrix
        var result = new double[dimension()];
        int index = 0;
        for (double[] categoryWeights : weights) {
            System.arraycopy(categoryWeights, 0, result, index, categoryWeights.length);
            index += categoryWeights.length;
        }
        return result;
    }

    @Override
    public State fromArray(double[] values) {
        // Reconstruct weight matrix from flat array
        var numCategories = weights.length;
        var inputDim = weights[0].length;
        var result = new double[numCategories][inputDim];

        int index = 0;
        for (int j = 0; j < numCategories; j++) {
            System.arraycopy(values, index, result[j], 0, inputDim);
            index += inputDim;
        }

        return new InstarLearningState(result);
    }

    /**
     * Compute learning statistics.
     */
    public LearningStatistics computeStatistics() {
        double avgWeight = 0.0;
        double minWeight = Double.MAX_VALUE;
        double maxWeight = Double.MIN_VALUE;

        for (double[] categoryWeights : weights) {
            for (double w : categoryWeights) {
                avgWeight += w;
                minWeight = Math.min(minWeight, w);
                maxWeight = Math.max(maxWeight, w);
            }
        }
        avgWeight /= dimension();

        return new LearningStatistics(avgWeight, minWeight, maxWeight, getUsedCategoryCount());
    }

    public record LearningStatistics(double averageWeight, double minWeight,
                                     double maxWeight, int usedCategories) {}
}