/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 *
 * This file is part of Java ART Neural Networks.
 *
 * Java ART Neural Networks is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Java ART Neural Networks is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with Java ART Neural Networks. If not, see <https://www.gnu.org/licenses/>.
 */
package com.hellblazer.art.temporal.vectorized;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Vectorized implementation of Competitive Instar Learning using Java Vector API for SIMD optimization.
 *
 * This implementation provides significant performance improvements for:
 * - SIMD-optimized weight updates using vectorized arithmetic
 * - Parallel activation computations across multiple categories
 * - Batch processing of competitive learning operations
 * - Vectorized normalization and thresholding functions
 *
 * Target speedup: 15-50x for weight update operations depending on input dimension
 * and category count. Peak performance achieved with input dimensions that are
 * multiples of vector species length.
 *
 * Mathematical Foundation:
 * Vectorized competitive instar learning: dW_ij/dt = α*f(c_j)*[(1-W_ij)*x_i - W_ij*∑x_k]
 * Implemented using SIMD operations for element-wise arithmetic across weight dimensions.
 *
 * Vectorized activation computation: a_j = ∑(x_i * W_ij) for all i
 * Computed in parallel using vector dot product operations.
 *
 * @author Hal Hildebrand
 */
public class VectorizedCompetitiveInstar {

    private static final Logger log = LoggerFactory.getLogger(VectorizedCompetitiveInstar.class);

    // SIMD Configuration
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int VECTOR_LENGTH = SPECIES.length();

    // Learning Configuration
    private final float learningRate;
    private final float minWeight;
    private final float maxWeight;
    private final boolean selfNormalizing;
    private final boolean softCompetition;
    private final float softLearningFactor; // Factor for non-winner learning

    // Network Structure
    private final int numCategories;
    private final int inputDimension;
    private final int paddedInputDimension; // Padded to vector boundary

    // Vectorized State Arrays
    private float[][] weights;              // [category][paddedInputDimension]
    private float[] categoryActivations;    // [category]
    private float[] inputBuffer;            // [paddedInputDimension] for current input
    private float[] normalizationFactors;   // [category] for weight normalization

    // Performance Tracking
    private final AtomicLong vectorOperations = new AtomicLong(0);
    private final AtomicLong weightUpdates = new AtomicLong(0);
    private final AtomicLong activationComputations = new AtomicLong(0);
    private final AtomicLong computationTimeNanos = new AtomicLong(0);

    /**
     * Create vectorized competitive instar learning with specified parameters.
     *
     * @param learningRate learning rate for weight updates
     * @param numCategories number of competitive categories
     * @param inputDimension dimensionality of input patterns
     * @param selfNormalizing whether to normalize weights after updates
     * @param softCompetition whether to use soft competition (partial learning for non-winners)
     */
    public VectorizedCompetitiveInstar(float learningRate, int numCategories, int inputDimension,
                                     boolean selfNormalizing, boolean softCompetition) {
        this.learningRate = learningRate;
        this.numCategories = numCategories;
        this.inputDimension = inputDimension;
        this.selfNormalizing = selfNormalizing;
        this.softCompetition = softCompetition;
        this.softLearningFactor = 0.1f; // 10% of winner's learning rate

        // Pad input dimension to vector boundary for optimal SIMD performance
        this.paddedInputDimension = ((inputDimension + VECTOR_LENGTH - 1) / VECTOR_LENGTH) * VECTOR_LENGTH;

        this.minWeight = 0.0f;
        this.maxWeight = 1.0f;

        log.debug("Initializing VectorizedCompetitiveInstar: categories={}, inputDim={}, paddedDim={}, vectorLen={}",
                 numCategories, inputDimension, paddedInputDimension, VECTOR_LENGTH);

        initializeArrays();
        initializeWeights();
    }

    /**
     * Convenience constructor with default settings.
     */
    public VectorizedCompetitiveInstar(float learningRate, int numCategories, int inputDimension) {
        this(learningRate, numCategories, inputDimension, true, false);
    }

    /**
     * Update weights using vectorized competitive instar learning rule.
     * dW_ij/dt = α*f(c_j)*[(1-W_ij)*x_i - W_ij*∑x_k]
     *
     * @param input input pattern
     * @param categoryActivations activation levels for each category
     * @param winnerIndex index of winning category (-1 if no winner)
     */
    public void updateWeights(float[] input, float[] categoryActivations, int winnerIndex) {
        var startTime = System.nanoTime();
        try {
            // Copy and pad input for vectorized operations
            prepareInputBuffer(input);

            // Store activations
            System.arraycopy(categoryActivations, 0, this.categoryActivations, 0,
                           Math.min(categoryActivations.length, numCategories));

            if (winnerIndex >= 0 && winnerIndex < numCategories) {
                // Update winner with full learning rate
                updateCategoryWeightsVectorized(winnerIndex, this.categoryActivations[winnerIndex], learningRate);

                // Optional soft competition for non-winners
                if (softCompetition) {
                    updateNonWinnerWeightsVectorized(winnerIndex);
                }
            }

            weightUpdates.incrementAndGet();

        } finally {
            computationTimeNanos.addAndGet(System.nanoTime() - startTime);
        }
    }

    /**
     * Compute activations for all categories using vectorized operations.
     *
     * @param input input pattern
     * @return activation values for each category
     */
    public float[] computeActivations(float[] input) {
        var startTime = System.nanoTime();
        try {
            prepareInputBuffer(input);
            var activations = new float[numCategories];

            // Compute activations in parallel using SIMD
            for (int category = 0; category < numCategories; category++) {
                activations[category] = computeCategoryActivationVectorized(category);
            }

            activationComputations.incrementAndGet();
            return activations;

        } finally {
            computationTimeNanos.addAndGet(System.nanoTime() - startTime);
        }
    }

    /**
     * Compute activation for a specific category using vectorized dot product.
     *
     * @param input input pattern
     * @param categoryIndex index of category
     * @return activation value for the category
     */
    public float computeCategoryActivation(float[] input, int categoryIndex) {
        if (categoryIndex < 0 || categoryIndex >= numCategories) {
            return 0.0f;
        }

        prepareInputBuffer(input);
        return computeCategoryActivationVectorized(categoryIndex);
    }

    /**
     * Get learned weights for a category.
     *
     * @param categoryIndex index of category
     * @return copy of weights for the category (non-padded)
     */
    public float[] getCategoryWeights(int categoryIndex) {
        if (categoryIndex < 0 || categoryIndex >= numCategories) {
            return new float[inputDimension];
        }

        return Arrays.copyOf(weights[categoryIndex], inputDimension);
    }

    /**
     * Set weights for a category.
     *
     * @param categoryIndex index of category
     * @param newWeights new weight values
     */
    public void setCategoryWeights(int categoryIndex, float[] newWeights) {
        if (categoryIndex < 0 || categoryIndex >= numCategories) {
            return;
        }

        var categoryWeights = weights[categoryIndex];

        // Copy weights
        var copyLength = Math.min(newWeights.length, inputDimension);
        System.arraycopy(newWeights, 0, categoryWeights, 0, copyLength);

        // Zero-pad remaining elements
        for (int i = inputDimension; i < paddedInputDimension; i++) {
            categoryWeights[i] = 0.0f;
        }

        if (selfNormalizing) {
            normalizeWeightsVectorized(categoryIndex);
        }
    }

    /**
     * Reset all weights to initial random values.
     */
    public void resetWeights() {
        initializeWeights();
    }

    /**
     * Reset weights for a specific category.
     *
     * @param categoryIndex index of category to reset
     */
    public void resetCategoryWeights(int categoryIndex) {
        if (categoryIndex < 0 || categoryIndex >= numCategories) {
            return;
        }

        var categoryWeights = weights[categoryIndex];

        // Initialize with small random values
        for (int i = 0; i < inputDimension; i++) {
            categoryWeights[i] = (float) (Math.random() * 0.1);
        }

        // Zero-pad
        for (int i = inputDimension; i < paddedInputDimension; i++) {
            categoryWeights[i] = 0.0f;
        }

        if (selfNormalizing) {
            normalizeWeightsVectorized(categoryIndex);
        }
    }

    /**
     * Get all weights for monitoring and debugging.
     *
     * @return copy of all weight matrices (non-padded)
     */
    public float[][] getAllWeights() {
        var copy = new float[numCategories][inputDimension];
        for (int i = 0; i < numCategories; i++) {
            System.arraycopy(weights[i], 0, copy[i], 0, inputDimension);
        }
        return copy;
    }

    /**
     * Get performance metrics for vectorized operations.
     *
     * @return performance statistics
     */
    public VectorizedInstarPerformanceMetrics getPerformanceMetrics() {
        return new VectorizedInstarPerformanceMetrics();
    }

    /**
     * Reset performance tracking counters.
     */
    public void resetPerformanceTracking() {
        vectorOperations.set(0);
        weightUpdates.set(0);
        activationComputations.set(0);
        computationTimeNanos.set(0);
    }

    // === Getters ===

    public float getLearningRate() {
        return learningRate;
    }

    public int getNumCategories() {
        return numCategories;
    }

    public int getInputDimension() {
        return inputDimension;
    }

    public boolean isSelfNormalizing() {
        return selfNormalizing;
    }

    public boolean isSoftCompetition() {
        return softCompetition;
    }

    // === Private Vectorized Implementation Methods ===

    private void initializeArrays() {
        weights = new float[numCategories][paddedInputDimension];
        categoryActivations = new float[numCategories];
        inputBuffer = new float[paddedInputDimension];
        normalizationFactors = new float[numCategories];
    }

    private void initializeWeights() {
        // Initialize with small random values
        for (int category = 0; category < numCategories; category++) {
            var categoryWeights = weights[category];

            for (int i = 0; i < inputDimension; i++) {
                categoryWeights[i] = (float) (Math.random() * 0.1);
            }

            // Zero-pad
            for (int i = inputDimension; i < paddedInputDimension; i++) {
                categoryWeights[i] = 0.0f;
            }

            if (selfNormalizing) {
                normalizeWeightsVectorized(category);
            }
        }
    }

    private void prepareInputBuffer(float[] input) {
        // Copy input to padded buffer
        var copyLength = Math.min(input.length, inputDimension);
        System.arraycopy(input, 0, inputBuffer, 0, copyLength);

        // Zero-pad remaining elements
        for (int i = copyLength; i < paddedInputDimension; i++) {
            inputBuffer[i] = 0.0f;
        }
    }

    private void updateCategoryWeightsVectorized(int categoryIndex, float activation, float effectiveLearningRate) {
        var categoryWeights = weights[categoryIndex];
        var learningSignal = thresholdFunction(activation);

        if (learningSignal <= 0) return;

        // Compute input sum for normalization (if self-normalizing)
        var inputSum = selfNormalizing ? computeInputSumVectorized() : 1.0f;

        // Vectorized weight update: dW = α * f(c) * [(1-W)*x - W*Σx]
        var learningRateVector = FloatVector.broadcast(SPECIES, effectiveLearningRate * learningSignal);
        var inputSumVector = FloatVector.broadcast(SPECIES, inputSum);
        var oneVector = FloatVector.broadcast(SPECIES, 1.0f);
        var minWeightVector = FloatVector.broadcast(SPECIES, minWeight);
        var maxWeightVector = FloatVector.broadcast(SPECIES, maxWeight);

        for (int i = 0; i < paddedInputDimension; i += VECTOR_LENGTH) {
            var weights_vec = FloatVector.fromArray(SPECIES, categoryWeights, i);
            var input_vec = FloatVector.fromArray(SPECIES, inputBuffer, i);

            // (1-W)*x
            var excitatory = oneVector.sub(weights_vec).mul(input_vec);

            // W*Σx
            var inhibitory = weights_vec.mul(inputSumVector);

            // dW = α * f(c) * [(1-W)*x - W*Σx]
            var deltaWeight = learningRateVector.mul(excitatory.sub(inhibitory));

            // Update weights: W_new = W + dW
            var newWeights = weights_vec.add(deltaWeight);

            // Bound to [minWeight, maxWeight]
            newWeights = newWeights.max(minWeightVector).min(maxWeightVector);

            newWeights.intoArray(categoryWeights, i);
        }

        // Normalize if required
        if (selfNormalizing) {
            normalizeWeightsVectorized(categoryIndex);
        }

        vectorOperations.incrementAndGet();
    }

    private void updateNonWinnerWeightsVectorized(int winnerIndex) {
        var softRate = learningRate * softLearningFactor;

        for (int category = 0; category < numCategories; category++) {
            if (category != winnerIndex && categoryActivations[category] > 0) {
                updateCategoryWeightsVectorized(category, categoryActivations[category], softRate);
            }
        }
    }

    private float computeCategoryActivationVectorized(int categoryIndex) {
        var categoryWeights = weights[categoryIndex];
        var sum = FloatVector.zero(SPECIES);

        // Vectorized dot product
        for (int i = 0; i < inputDimension; i += VECTOR_LENGTH) {
            var remaining = Math.min(VECTOR_LENGTH, inputDimension - i);

            if (remaining == VECTOR_LENGTH) {
                var weights_vec = FloatVector.fromArray(SPECIES, categoryWeights, i);
                var input_vec = FloatVector.fromArray(SPECIES, inputBuffer, i);
                sum = sum.add(weights_vec.mul(input_vec));
            } else {
                // Handle partial vector at end
                for (int j = 0; j < remaining; j++) {
                    var product = categoryWeights[i + j] * inputBuffer[i + j];
                    sum = sum.add(FloatVector.broadcast(SPECIES, product));
                }
            }
        }

        vectorOperations.incrementAndGet();
        return sum.reduceLanes(VectorOperators.ADD);
    }

    private float computeInputSumVectorized() {
        var sum = FloatVector.zero(SPECIES);

        for (int i = 0; i < inputDimension; i += VECTOR_LENGTH) {
            var remaining = Math.min(VECTOR_LENGTH, inputDimension - i);

            if (remaining == VECTOR_LENGTH) {
                var input_vec = FloatVector.fromArray(SPECIES, inputBuffer, i);
                // Use absolute values for normalization
                sum = sum.add(input_vec.abs());
            } else {
                // Handle partial vector at end
                for (int j = 0; j < remaining; j++) {
                    sum = sum.add(FloatVector.broadcast(SPECIES, Math.abs(inputBuffer[i + j])));
                }
            }
        }

        return sum.reduceLanes(VectorOperators.ADD);
    }

    private void normalizeWeightsVectorized(int categoryIndex) {
        var categoryWeights = weights[categoryIndex];

        // Compute sum of weights
        var sum = FloatVector.zero(SPECIES);

        for (int i = 0; i < inputDimension; i += VECTOR_LENGTH) {
            var remaining = Math.min(VECTOR_LENGTH, inputDimension - i);

            if (remaining == VECTOR_LENGTH) {
                var weights_vec = FloatVector.fromArray(SPECIES, categoryWeights, i);
                sum = sum.add(weights_vec.abs());
            } else {
                // Handle partial vector at end
                for (int j = 0; j < remaining; j++) {
                    sum = sum.add(FloatVector.broadcast(SPECIES, Math.abs(categoryWeights[i + j])));
                }
            }
        }

        var totalSum = sum.reduceLanes(VectorOperators.ADD);

        if (totalSum > 0) {
            // Normalize weights
            var normVector = FloatVector.broadcast(SPECIES, totalSum);

            for (int i = 0; i < paddedInputDimension; i += VECTOR_LENGTH) {
                var weights_vec = FloatVector.fromArray(SPECIES, categoryWeights, i);
                var normalized = weights_vec.div(normVector);
                normalized.intoArray(categoryWeights, i);
            }
        }

        normalizationFactors[categoryIndex] = totalSum;
        vectorOperations.incrementAndGet();
    }

    private float thresholdFunction(float activation) {
        // Sigmoid threshold function for learning signal
        return activation > 0 ? (float) (1.0 / (1.0 + Math.exp(-activation))) : 0.0f;
    }

    // === Performance Metrics Implementation ===

    public class VectorizedInstarPerformanceMetrics {
        public long getVectorOperations() {
            return vectorOperations.get();
        }

        public long getWeightUpdates() {
            return weightUpdates.get();
        }

        public long getActivationComputations() {
            return activationComputations.get();
        }

        public long getComputationTimeNanos() {
            return computationTimeNanos.get();
        }

        public double getAverageUpdateTime() {
            var updates = weightUpdates.get();
            return updates > 0 ? (double) computationTimeNanos.get() / updates : 0.0;
        }

        public double getAverageActivationTime() {
            var computations = activationComputations.get();
            return computations > 0 ? (double) computationTimeNanos.get() / computations : 0.0;
        }

        public int getVectorSpeciesLength() {
            return VECTOR_LENGTH;
        }

        public double getVectorizationEfficiency() {
            return (double) inputDimension / paddedInputDimension;
        }

        public long getMemoryUsage() {
            return (long) numCategories * paddedInputDimension * Float.BYTES +
                   numCategories * Float.BYTES * 2 +
                   paddedInputDimension * Float.BYTES;
        }

        public double getThroughputOpsPerSecond() {
            var timeSeconds = computationTimeNanos.get() / 1e9;
            var totalOps = weightUpdates.get() + activationComputations.get();
            return timeSeconds > 0 ? totalOps / timeSeconds : 0.0;
        }

        @Override
        public String toString() {
            return String.format(
                "VectorizedInstarPerformanceMetrics{" +
                "vectorOps=%d, weightUpdates=%d, activationComps=%d, " +
                "avgUpdateTime=%.2fμs, avgActivationTime=%.2fμs, " +
                "vectorLen=%d, efficiency=%.2f%%, throughput=%.1f ops/sec}",
                getVectorOperations(), getWeightUpdates(), getActivationComputations(),
                getAverageUpdateTime() / 1000.0, getAverageActivationTime() / 1000.0,
                getVectorSpeciesLength(), getVectorizationEfficiency() * 100.0,
                getThroughputOpsPerSecond()
            );
        }
    }
}