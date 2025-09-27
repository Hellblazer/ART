package com.hellblazer.art.hybrid.pan;

import com.hellblazer.art.core.ARTAlgorithm;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.hybrid.pan.memory.DualMemoryManager;
import com.hellblazer.art.hybrid.pan.memory.ExperienceReplayBuffer;
import com.hellblazer.art.hybrid.pan.parameters.PANParameters;
import com.hellblazer.art.hybrid.pan.preprocessing.CNNPreprocessor;
import com.hellblazer.art.hybrid.pan.weight.BPARTWeight;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

/**
 * PAN (Pretrained Adaptive Resonance Network) implementation.
 *
 * Based on "Pretrained back propagation based adaptive resonance theory network
 * for adaptive learning" by Zhang et al. (2023).
 *
 * Key innovations:
 * - CNN feature extraction (with optional pretraining)
 * - BPART nodes with backpropagation and negative weights
 * - STM/LTM dual memory system
 * - Experience replay for continual learning
 * - Light induction bias factor
 *
 * This implementation directly implements ARTAlgorithm rather than extending BaseART
 * to have full control over the learning process as required by the paper's architecture.
 */
public final class PAN implements ARTAlgorithm<PANParameters>, AutoCloseable {

    // Core components
    private final CNNPreprocessor cnnPreprocessor;
    private final List<WeightVector> categories;
    private final DualMemoryManager memoryManager;
    private final ExperienceReplayBuffer replayBuffer;

    // For supervised learning - maps category indices to class labels
    private final Map<Integer, Integer> categoryToLabel;

    // Performance tracking
    private long totalSamples = 0;
    private long correctPredictions = 0;
    private double averageLoss = 0.0;
    private long trainingStartTime = System.currentTimeMillis();

    // Thread safety
    private final Object lock = new Object();
    private volatile boolean closed = false;

    /**
     * Create a new PAN instance with specified parameters.
     */
    public PAN(PANParameters parameters) {
        this.cnnPreprocessor = new CNNPreprocessor(
            parameters.cnnConfig(),
            parameters.enableCNNPretraining()
        );
        this.categories = new ArrayList<>();
        this.memoryManager = new DualMemoryManager(
            parameters.stmDecayRate(),
            parameters.ltmConsolidationThreshold()
        );
        this.replayBuffer = new ExperienceReplayBuffer(
            parameters.replayBufferSize(),
            parameters.replayBatchSize()
        );
        this.categoryToLabel = new HashMap<>();
    }

    /**
     * Learn a pattern (unsupervised mode).
     * In PAN, this performs CNN feature extraction followed by BPART processing.
     */
    @Override
    public ActivationResult learn(Pattern input, PANParameters parameters) {
        ensureNotClosed();
        long startTime = System.currentTimeMillis();

        synchronized (lock) {
            // Step 1: CNN feature extraction
            Pattern features = cnnPreprocessor.extractFeatures(input);

            // Step 2: Find best matching category
            int bestCategory = -1;
            double bestActivation = -Double.MAX_VALUE;
            BPARTWeight bestWeight = null;

            for (int i = 0; i < categories.size(); i++) {
                var weight = (BPARTWeight) categories.get(i);
                double activation = weight.calculateActivation(features);

                // Check basic vigilance first
                if (activation >= parameters.vigilance()) {
                    // Then check enhanced vigilance with STM/LTM
                    if (memoryManager.checkEnhancedVigilance(features, weight, parameters.vigilance())) {
                        if (activation > bestActivation) {
                            bestActivation = activation;
                            bestCategory = i;
                            bestWeight = weight;
                        }
                    }
                }
            }

            // Step 3: Create new category or update existing
            if (bestCategory == -1) {
                // No match - create new category
                if (categories.size() < parameters.maxCategories()) {
                    bestWeight = BPARTWeight.createFromPattern(features, parameters);
                    categories.add(bestWeight);
                    bestCategory = categories.size() - 1;
                    bestActivation = bestWeight.calculateActivation(features); // Calculate actual activation

                    // Register in memory systems
                    memoryManager.registerNewCategory(bestCategory, features);
                } else {
                    // Max categories reached - return no match
                    return ActivationResult.NoMatch.instance();
                }
            } else {
                // Update existing category with backpropagation
                WeightVector updatedWeight = performBackpropagation(
                    bestWeight, features, null, parameters
                );
                categories.set(bestCategory, updatedWeight);
                bestWeight = (BPARTWeight) updatedWeight;

                // Update memory systems
                memoryManager.updateCategory(bestCategory, features);
            }

            // Step 4: Add to experience replay buffer
            replayBuffer.addExperience(features, null, bestWeight, 0.0);

            // Step 5: Perform experience replay with probability
            if (ThreadLocalRandom.current().nextDouble() < parameters.replayFrequency()) {
                performExperienceReplay(parameters);
            }

            // Update tracking
            totalSamples++;
            long elapsedTime = System.currentTimeMillis() - startTime;

            return new ActivationResult.Success(bestCategory, bestActivation, bestWeight);
        }
    }

    /**
     * Supervised learning with target labels.
     *
     * @param input The input pattern (e.g., image)
     * @param target The target pattern (e.g., one-hot encoded label)
     * @param parameters The PAN parameters
     * @return The activation result including category assignment
     */
    public ActivationResult learnSupervised(Pattern input, Pattern target, PANParameters parameters) {
        ensureNotClosed();

        synchronized (lock) {
            // Step 1: CNN feature extraction
            Pattern features = cnnPreprocessor.extractFeatures(input);

            // Step 2: Find or create category for this input-target pair
            int targetLabel = extractLabelFromPattern(target);

            // Check if we have a category for this label
            Integer existingCategory = findBestCategoryForLabel(targetLabel, features, parameters);

            BPARTWeight weight;
            int category;

            if (existingCategory != null) {
                // Update existing category
                category = existingCategory;
                weight = (BPARTWeight) categories.get(category);

                // Backpropagation with target
                WeightVector updatedWeight = performBackpropagation(
                    weight, features, target, parameters
                );
                categories.set(category, updatedWeight);
                weight = (BPARTWeight) updatedWeight;
                memoryManager.updateCategory(category, features);
            } else {
                // Create new category for this label
                if (categories.size() < parameters.maxCategories()) {
                    weight = BPARTWeight.createFromPatternWithTarget(features, target, parameters);
                    categories.add(weight);
                    category = categories.size() - 1;
                    categoryToLabel.put(category, targetLabel);
                    memoryManager.registerNewCategory(category, features);
                } else {
                    // Max categories reached - find best category to update
                    category = findClosestCategoryForLabel(targetLabel, features);
                    if (category == -1) {
                        // No category with this label, reassign closest one
                        category = 0;
                        double bestSim = -1;
                        for (int i = 0; i < categories.size(); i++) {
                            var w = (BPARTWeight) categories.get(i);
                            double sim = w.calculateActivation(features);
                            if (sim > bestSim) {
                                bestSim = sim;
                                category = i;
                            }
                        }
                        categoryToLabel.put(category, targetLabel); // Reassign label
                    }
                    weight = (BPARTWeight) categories.get(category);
                    WeightVector updatedWeight = performBackpropagation(
                        weight, features, target, parameters
                    );
                    categories.set(category, updatedWeight);
                    weight = (BPARTWeight) updatedWeight;
                }
            }

            // Update tracking - recalculate activation after any updates
            double activation = weight.calculateActivation(features);
            // Check if the category is correctly mapped to the target label
            Integer mappedLabel = categoryToLabel.get(category);
            boolean correct = (mappedLabel != null && mappedLabel == targetLabel);
            if (correct) correctPredictions++;
            totalSamples++;

            // Add to replay buffer with reward signal
            double reward = correct ? 1.0 : -1.0;
            replayBuffer.addExperience(features, target, weight, reward);

            return new ActivationResult.Success(category, activation, weight);
        }
    }

    /**
     * Predict the category for an input pattern without learning.
     */
    @Override
    public ActivationResult predict(Pattern input, PANParameters parameters) {
        ensureNotClosed();

        synchronized (lock) {
            // CNN feature extraction
            Pattern features = cnnPreprocessor.extractFeatures(input);

            // Find best matching category
            int bestCategory = -1;
            double bestActivation = -Double.MAX_VALUE;

            for (int i = 0; i < categories.size(); i++) {
                var weight = (BPARTWeight) categories.get(i);
                double activation = weight.calculateActivation(features);

                if (activation > bestActivation) {
                    bestActivation = activation;
                    bestCategory = i;
                }
            }

            if (bestCategory == -1 || bestActivation < parameters.vigilance()) {
                return ActivationResult.NoMatch.instance();
            }
            return new ActivationResult.Success(
                bestCategory,
                bestActivation,
                categories.get(bestCategory)
            );
        }
    }

    /**
     * Predict and return the class label for supervised learning.
     */
    public int predictLabel(Pattern input, PANParameters parameters) {
        var result = predict(input, parameters);
        if (result instanceof ActivationResult.Success success) {
            Integer label = categoryToLabel.get(success.categoryIndex());
            return label != null ? label : -1;
        }
        return -1;
    }

    /**
     * Batch learning for efficiency.
     */
    @Override
    public List<ActivationResult> learnBatch(List<Pattern> inputs, PANParameters parameters) {
        ensureNotClosed();

        // Process batch in parallel using virtual threads
        return inputs.parallelStream()
            .map(input -> learn(input, parameters))
            .toList();
    }

    /**
     * Batch prediction for efficiency.
     */
    @Override
    public List<ActivationResult> predictBatch(List<Pattern> inputs, PANParameters parameters) {
        ensureNotClosed();

        return inputs.parallelStream()
            .map(input -> predict(input, parameters))
            .toList();
    }

    /**
     * Get the current number of categories.
     */
    @Override
    public int getCategoryCount() {
        synchronized (lock) {
            return categories.size();
        }
    }

    /**
     * Get all categories.
     */
    @Override
    public List<WeightVector> getCategories() {
        synchronized (lock) {
            return new ArrayList<>(categories);
        }
    }

    /**
     * Get a specific category by index.
     */
    @Override
    public WeightVector getCategory(int index) {
        synchronized (lock) {
            if (index >= 0 && index < categories.size()) {
                return categories.get(index);
            }
            return null;
        }
    }

    /**
     * Clear all learned categories and reset state.
     */
    @Override
    public void clear() {
        synchronized (lock) {
            categories.clear();
            categoryToLabel.clear();
            memoryManager.clear();
            replayBuffer.clear();
            totalSamples = 0;
            correctPredictions = 0;
            averageLoss = 0.0;
            trainingStartTime = System.currentTimeMillis();
        }
    }

    /**
     * Get performance statistics.
     */
    public Map<String, Object> getPerformanceStats() {
        synchronized (lock) {
            Map<String, Object> stats = new HashMap<>();
            stats.put("totalSamples", totalSamples);
            stats.put("correctPredictions", correctPredictions);
            stats.put("accuracy", totalSamples > 0 ? (double) correctPredictions / totalSamples : 0.0);
            stats.put("averageLoss", averageLoss);
            stats.put("trainingTimeMs", System.currentTimeMillis() - trainingStartTime);
            stats.put("categoryCount", categories.size());
            stats.put("memoryUsageBytes", estimateMemoryUsage());
            return stats;
        }
    }

    /**
     * Reset performance tracking.
     */
    public void resetPerformanceTracking() {
        synchronized (lock) {
            totalSamples = 0;
            correctPredictions = 0;
            averageLoss = 0.0;
            trainingStartTime = System.currentTimeMillis();
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            cnnPreprocessor.close();
            memoryManager.close();
            replayBuffer.clear();
            categories.clear();
            categoryToLabel.clear();
        }
    }

    // Private helper methods

    private void ensureNotClosed() {
        if (closed) {
            throw new IllegalStateException("PAN is closed");
        }
    }

    private WeightVector performBackpropagation(BPARTWeight weight, Pattern features,
                                                Pattern target, PANParameters parameters) {
        // Perform local backpropagation within BPART node
        double[] gradients = weight.computeGradients(features, target);

        // Create gradient pattern for weight update
        var gradientPattern = new DenseVector(gradients);

        // Update weights using gradient descent with momentum
        return weight.update(gradientPattern, parameters);
    }

    private void performExperienceReplay(PANParameters parameters) {
        var batch = replayBuffer.sampleBatch();

        for (var experience : batch) {
            if (experience != null) {
                var weight = experience.nodeState();
                var updatedWeight = performBackpropagation(
                    weight,
                    experience.features(),
                    experience.target(),
                    parameters
                );

                // Update if this weight is still in use
                for (int i = 0; i < categories.size(); i++) {
                    if (categories.get(i).equals(weight)) {
                        categories.set(i, updatedWeight);
                        break;
                    }
                }
            }
        }
    }

    private int extractLabelFromPattern(Pattern target) {
        // Extract label from one-hot encoded pattern or other encoding
        // For one-hot: find the index with value 1.0
        for (int i = 0; i < target.dimension(); i++) {
            if (target.get(i) > 0.5) {
                return i;
            }
        }
        return -1;
    }

    private Integer findBestCategoryForLabel(int label, Pattern features, PANParameters parameters) {
        // Find the best matching category that already has this label
        Integer bestCategory = null;
        double bestActivation = -Double.MAX_VALUE;

        for (var entry : categoryToLabel.entrySet()) {
            if (entry.getValue() == label) {
                int categoryIdx = entry.getKey();
                var weight = (BPARTWeight) categories.get(categoryIdx);
                double activation = weight.calculateActivation(features);

                // Check if this category is a good match for the features
                if (activation >= parameters.vigilance() * 0.8) { // Slightly relaxed for supervised
                    if (activation > bestActivation) {
                        bestActivation = activation;
                        bestCategory = categoryIdx;
                    }
                }
            }
        }

        return bestCategory;
    }

    private int findClosestCategoryForLabel(int label, Pattern features) {
        // Find closest category with the specified label
        int bestCategory = -1;
        double bestActivation = -Double.MAX_VALUE;

        for (var entry : categoryToLabel.entrySet()) {
            if (entry.getValue() == label) {
                int categoryIdx = entry.getKey();
                var weight = (BPARTWeight) categories.get(categoryIdx);
                double activation = weight.calculateActivation(features);
                if (activation > bestActivation) {
                    bestActivation = activation;
                    bestCategory = categoryIdx;
                }
            }
        }

        return bestCategory;
    }

    private long estimateMemoryUsage() {
        long total = 0;
        total += cnnPreprocessor.estimateMemoryUsage();
        total += memoryManager.estimateMemoryUsage();
        total += replayBuffer.estimateMemoryUsage();

        // Estimate category memory (approximate)
        total += categories.size() * 8192L; // Estimate 8KB per category

        return total;
    }
}