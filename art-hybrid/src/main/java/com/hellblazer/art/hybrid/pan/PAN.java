package com.hellblazer.art.hybrid.pan;

import com.hellblazer.art.core.ARTAlgorithm;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.hybrid.pan.decision.DualCriterionDecisionSystem;
import com.hellblazer.art.hybrid.pan.learning.BackpropagationUpdater;
import com.hellblazer.art.hybrid.pan.learning.LightInduction;
import com.hellblazer.art.hybrid.pan.memory.DualMemoryManager;
import com.hellblazer.art.hybrid.pan.memory.ExperienceReplayBuffer;
import com.hellblazer.art.hybrid.pan.parameters.PANParameters;
import com.hellblazer.art.hybrid.pan.performance.PANProfiler;
import com.hellblazer.art.hybrid.pan.preprocessing.CNNPreprocessor;
import com.hellblazer.art.hybrid.pan.serialization.PANSerializer;
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

    // New components from Phases 2-4
    private final LightInduction lightInduction;
    private final BackpropagationUpdater backpropUpdater;
    private final DualCriterionDecisionSystem decisionSystem;

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

        // Initialize new components
        this.lightInduction = new LightInduction(parameters.biasFactor());
        this.backpropUpdater = new BackpropagationUpdater(parameters.momentum());
        this.decisionSystem = new DualCriterionDecisionSystem(parameters);

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
        var profiler = PANProfiler.getInstance();

        synchronized (lock) {
            // Step 1: CNN feature extraction
            Pattern rawFeatures;
            try (var timer = profiler.startTimer("cnn_extraction")) {
                rawFeatures = cnnPreprocessor.extractFeatures(input);
            }

            // Step 1.5: Conditionally normalize CNN features based on parameters
            Pattern normalizedFeatures;
            try (var timer = profiler.startTimer("normalization")) {
                if (parameters.enableFeatureNormalization()) {
                    normalizedFeatures = normalizeWithGlobalBounds(rawFeatures, parameters);
                } else {
                    normalizedFeatures = rawFeatures; // Skip normalization to preserve distinctiveness
                }
            }

            // Step 1.6: Apply complement coding for proper FuzzyART compatibility
            Pattern features;
            try (var timer = profiler.startTimer("complement_coding")) {
                features = applyComplementCoding(normalizedFeatures);
            }

            // Step 2: Find best matching category using dual-criterion system
            int bestCategory = -1;
            double bestActivation = -Double.MAX_VALUE;
            BPARTWeight bestWeight = null;
            DualCriterionDecisionSystem.Decision decision = DualCriterionDecisionSystem.Decision.LEARN_NEW;

            try (var timer = profiler.startTimer("category_search")) {
                for (int i = 0; i < categories.size(); i++) {
                    var weight = (BPARTWeight) categories.get(i);

                    // Compute STM resonance and LTM confidence using hardcoded similarity
                    double stmResonance = weight.calculateResonanceIntensity(features);
                    double ltmConfidence = memoryManager.computeLTMConfidence(i, features);
                    double combinedActivation = weight.calculateActivation(features);

                    if (combinedActivation > bestActivation) {
                        bestActivation = combinedActivation;
                        bestCategory = i;
                        bestWeight = weight;
                        // Make decision based on dual criteria
                        decision = decisionSystem.makeDecision(stmResonance, ltmConfidence);
                    }
                }
            }

            // Step 3: Apply dual-criterion decision with more sophisticated learning logic
            // Instead of just checking threshold, also check if pattern is sufficiently different
            boolean shouldCreateNewCategory = false;

            if (bestCategory == -1) {
                // No existing categories, create first one
                shouldCreateNewCategory = true;
            } else {
                // Use vigilance parameter correctly - higher vigilance means stricter categorization
                // In ART, similarity must be >= vigilance to accept an existing category
                var existingWeight = (BPARTWeight) categories.get(bestCategory);
                double similarity = computeFuzzyARTSimilarity(features, existingWeight.forwardWeights());

                // Debug learning decision
                if (System.getProperty("pan.debug") != null) {
                    System.out.printf("  Learning decision: bestCategory=%d, similarity=%.3f, vigilance=%.3f\n",
                        bestCategory, similarity, parameters.vigilance());
                }

                // Standard ART vigilance test: create new category if similarity < vigilance
                if (similarity < parameters.vigilance()) {
                    shouldCreateNewCategory = true;
                }

            }

            if (shouldCreateNewCategory) {
                // No match or insufficient match - create new category
                if (categories.size() < parameters.maxCategories()) {
                    bestWeight = BPARTWeight.createFromPattern(features, parameters);
                    // For unsupervised learning, skip backpropagation to maintain category stability
                    // This preserves the original ART behavior where categories are stable once created
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
                // For unsupervised learning, skip weight updates to maintain category stability
                // Only update memory systems for tracking
                memoryManager.updateCategory(bestCategory, features);
            }

            // Step 4: Add to experience replay buffer
            replayBuffer.addExperience(features, null, bestWeight, 0.0);

            // Step 5: Skip experience replay for unsupervised learning to maintain category stability
            // Experience replay is designed for supervised learning where targets guide updates
            // In unsupervised learning, categories should remain stable once created
            // (Experience replay will still be used in learnSupervised() method)

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
            Pattern rawFeatures = cnnPreprocessor.extractFeatures(input);

            // Step 1.5: Conditionally normalize CNN features based on parameters
            Pattern normalizedFeatures;
            if (parameters.enableFeatureNormalization()) {
                normalizedFeatures = normalizeWithGlobalBounds(rawFeatures, parameters);
            } else {
                normalizedFeatures = rawFeatures; // Skip normalization to preserve distinctiveness
            }

            // Step 1.6: Apply complement coding for proper FuzzyART compatibility
            Pattern features = applyComplementCoding(normalizedFeatures);

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
                    // Perform initial backprop to strengthen the new category
                    weight = (BPARTWeight) performBackpropagation(weight, features, target, parameters);
                    categories.add(weight);
                    category = categories.size() - 1;
                    categoryToLabel.put(category, targetLabel);
                    memoryManager.registerNewCategory(category, features);
                } else {
                    // Max categories reached - find best category to update
                    category = findClosestCategoryForLabel(targetLabel, features, parameters);
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
            Pattern rawFeatures = cnnPreprocessor.extractFeatures(input);

            // Conditionally normalize CNN features based on parameters
            Pattern normalizedFeatures;
            if (parameters.enableFeatureNormalization()) {
                normalizedFeatures = normalizeWithGlobalBounds(rawFeatures, parameters);
            } else {
                normalizedFeatures = rawFeatures; // Skip normalization to preserve distinctiveness
            }

            // Apply complement coding for proper FuzzyART compatibility
            Pattern features = applyComplementCoding(normalizedFeatures);

            // Use IDENTICAL sequential search logic as learn() method
            int bestCategory = -1;
            double bestActivation = -Double.MAX_VALUE;

            // Sequential search with vigilance test - EXACTLY like learn() method
            for (int i = 0; i < categories.size(); i++) {
                var weight = (BPARTWeight) categories.get(i);

                // Calculate activation and similarity
                double combinedActivation = weight.calculateActivation(features);
                double similarity = computeFuzzyARTSimilarity(features, weight.forwardWeights());

                // Debug output
                if (System.getProperty("pan.debug") != null) {
                    double resonance = weight.calculateResonanceIntensity(features);
                    double confidence = weight.calculateLocationConfidence(features);
                    System.out.printf("  Prediction: Category %d: activation=%.6f, similarity=%.6f, vigilance=%.6f (resonance=%.6f, confidence=%.6f)\n",
                        i, combinedActivation, similarity, parameters.vigilance(), resonance, confidence);
                }

                // ART sequential search: first category that passes vigilance with highest activation wins
                if (similarity >= parameters.vigilance() && combinedActivation > bestActivation) {
                    bestActivation = combinedActivation;
                    bestCategory = i;
                }
            }

            if (bestCategory == -1) {
                return ActivationResult.NoMatch.instance();
            }

            // Vigilance test already applied in loop above - bestCategory is guaranteed to pass vigilance
            var bestWeight = (BPARTWeight) categories.get(bestCategory);

            // Debug prediction decision
            if (System.getProperty("pan.debug") != null) {
                double similarity = computeFuzzyARTSimilarity(features, bestWeight.forwardWeights());
                System.out.printf("  Prediction decision: bestCategory=%d, similarity=%.3f, vigilance=%.3f\n",
                    bestCategory, similarity, parameters.vigilance());
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
     * Create PAN from saved model.
     */
    public static PAN fromSavedModel(PANSerializer.SavedPANModel model) {
        var pan = new PAN(model.parameters());

        // Restore categories
        pan.categories.clear();
        pan.categories.addAll(model.categories());

        // Restore category-label mapping
        pan.categoryToLabel.clear();
        pan.categoryToLabel.putAll(model.categoryToLabel());

        // Restore statistics
        pan.totalSamples = model.totalSamples();
        pan.correctPredictions = model.correctPredictions();

        // Restore CNN weights
        if (model.cnnConv1Weights() != null && model.cnnConv1Weights().length > 0) {
            pan.cnnPreprocessor.setConv1Weights(model.cnnConv1Weights());
        }
        if (model.cnnConv2Weights() != null && model.cnnConv2Weights().length > 0) {
            pan.cnnPreprocessor.setConv2Weights(model.cnnConv2Weights());
        }

        return pan;
    }

    /**
     * Get category to label mapping.
     */
    public Map<Integer, Integer> getCategoryToLabel() {
        synchronized (lock) {
            return new HashMap<>(categoryToLabel);
        }
    }

    /**
     * Get total samples processed.
     */
    public long getTotalSamples() {
        return totalSamples;
    }

    /**
     * Get correct predictions count.
     */
    public long getCorrectPredictions() {
        return correctPredictions;
    }

    /**
     * Get CNN preprocessor for pretraining.
     */
    public CNNPreprocessor getCNNPreprocessor() {
        return cnnPreprocessor;
    }

    /**
     * Get a specific category by index.
     */
    @Override
    public WeightVector getCategory(int index) {
        synchronized (lock) {
            if (index < 0 || index >= categories.size()) {
                throw new IndexOutOfBoundsException(
                    "Category index " + index + " out of bounds [0, " + categories.size() + ")"
                );
            }
            return categories.get(index);
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
            lightInduction.clear();
            backpropUpdater.resetMomentum();
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
        // Get current activation for the weight using hardcoded similarity
        double output = weight.calculateActivation(features);

        // Compute light induction factor
        int categoryId = categories.indexOf(weight);
        double stmActivity = weight.calculateResonanceIntensity(features);
        double ltmConfidence = memoryManager.computeLTMConfidence(categoryId, features);
        double lambda = lightInduction.computeLambda(categoryId, stmActivity, ltmConfidence);

        // Apply backpropagation with proper equation (no double negative)
        BPARTWeight updatedWeight;
        if (target != null) {
            // Supervised learning
            updatedWeight = backpropUpdater.applySupervisedBackpropagation(
                weight, features, target, lambda, parameters.learningRate()
            );
        } else {
            // Unsupervised learning
            updatedWeight = backpropUpdater.applyBackpropagation(
                weight, features, output, lambda, parameters.learningRate()
            );
        }

        // Update light induction tracking
        double learningOutcome = updatedWeight.calculateActivation(features) - output;
        lightInduction.updateInfluence(categoryId, learningOutcome);

        return updatedWeight;
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

                // Be more lenient for supervised learning to encourage category reuse
                // Use a much lower threshold to prefer existing categories with same label
                if (activation >= parameters.vigilance() * 0.2) { // Even more lenient for supervised learning
                    if (activation > bestActivation) {
                        bestActivation = activation;
                        bestCategory = categoryIdx;
                    }
                }
            }
        }

        return bestCategory;
    }

    private int findClosestCategoryForLabel(int label, Pattern features, PANParameters parameters) {
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

    /**
     * Calculate pattern dissimilarity to determine if a new category should be created
     * even when activation is above threshold. This helps with distinct pattern separation.
     */
    private double calculatePatternDissimilarity(Pattern newPattern, BPARTWeight existingWeight) {
        // Use Euclidean distance as a dissimilarity measure
        // This is more sensitive to pattern differences than Fuzzy ART similarity
        double sumSquaredDiff = 0.0;
        int size = Math.min(newPattern.dimension(), existingWeight.forwardWeights().length);

        for (int i = 0; i < size; i++) {
            double diff = newPattern.get(i) - existingWeight.forwardWeights()[i];
            sumSquaredDiff += diff * diff;
        }

        double distance = Math.sqrt(sumSquaredDiff);

        // Normalize by pattern magnitude to get a relative dissimilarity measure
        double patternMagnitude = 0.0;
        for (int i = 0; i < newPattern.dimension(); i++) {
            patternMagnitude += newPattern.get(i) * newPattern.get(i);
        }
        patternMagnitude = Math.sqrt(patternMagnitude);

        if (patternMagnitude == 0.0) {
            return 0.0;
        }

        // Return normalized distance as dissimilarity [0, infinity)
        // Values > 0.5 indicate highly dissimilar patterns
        return distance / patternMagnitude;
    }

    /**
     * Compute Fuzzy ART similarity between pattern and weights.
     * This replaces the configurable similarity measures with the original hardcoded logic.
     */
    private double computeFuzzyARTSimilarity(Pattern input, double[] weights) {
        double minSum = 0.0;
        double inputSum = 0.0;
        int size = Math.min(input.dimension(), weights.length);

        for (int i = 0; i < size; i++) {
            double inputVal = Math.abs(input.get(i));
            double weightVal = Math.abs(weights[i]);
            minSum += Math.min(inputVal, weightVal);
            inputSum += inputVal;
        }

        // Avoid division by zero
        if (inputSum == 0.0) {
            return 0.0;
        }

        return minSum / inputSum; // Always in [0,1]
    }

    /**
     * Apply complement coding to a pattern for proper FuzzyART compatibility.
     * Transforms [x] to [x, 1-x] as required by FuzzyART theory.
     *
     * @param input the original pattern
     * @return complement-coded pattern with doubled dimension
     */
    private Pattern applyComplementCoding(Pattern input) {
        int originalDim = input.dimension();
        double[] complementCoded = new double[originalDim * 2];

        // First half: original values
        for (int i = 0; i < originalDim; i++) {
            complementCoded[i] = input.get(i);
        }

        // Second half: complement values (1 - x)
        for (int i = 0; i < originalDim; i++) {
            complementCoded[originalDim + i] = 1.0 - complementCoded[i];
        }

        return new DenseVector(complementCoded);
    }

    /**
     * FIXED: Normalize pattern using global bounds to preserve inter-pattern relationships.
     * This fixes the clustering issue where per-pattern normalization made distinct
     * patterns identical after normalization.
     *
     * @param input the input pattern
     * @param parameters PAN parameters containing global bounds
     * @return normalized pattern with values in [0,1] using global scale
     */
    private Pattern normalizeWithGlobalBounds(Pattern input, PANParameters parameters) {
        int dim = input.dimension();
        double[] normalized = new double[dim];

        double globalMin = parameters.globalMinBound();
        double globalMax = parameters.globalMaxBound();
        double range = globalMax - globalMin;

        if (range == 0.0) {
            // If global bounds are identical, set all values to 0.5
            Arrays.fill(normalized, 0.5);
        } else {
            for (int i = 0; i < dim; i++) {
                // Clamp to global bounds and normalize
                double val = Math.max(globalMin, Math.min(globalMax, input.get(i)));
                normalized[i] = (val - globalMin) / range;
            }
        }

        return new DenseVector(normalized);
    }

    /**
     * DEPRECATED: Per-pattern normalization that caused clustering issues.
     * This method destroys inter-pattern relationships by normalizing each pattern
     * individually, making distinct patterns identical after normalization.
     *
     * @deprecated Use normalizeWithGlobalBounds() instead
     */
    @Deprecated
    private Pattern normalizeToUnitRange(Pattern input) {
        int dim = input.dimension();
        double[] normalized = new double[dim];

        // Find min and max values (PER-PATTERN - this is the problem!)
        double min = Double.MAX_VALUE;
        double max = -Double.MAX_VALUE;
        for (int i = 0; i < dim; i++) {
            double val = input.get(i);
            min = Math.min(min, val);
            max = Math.max(max, val);
        }

        // Normalize to [0,1] range
        double range = max - min;
        if (range == 0.0) {
            // All values are the same, set to 0.5
            Arrays.fill(normalized, 0.5);
        } else {
            for (int i = 0; i < dim; i++) {
                normalized[i] = (input.get(i) - min) / range;
            }
        }

        return new DenseVector(normalized);
    }

    /**
     * Helper methods for debugging pattern analysis.
     */
    private double getPatternMin(Pattern pattern) {
        double min = Double.MAX_VALUE;
        for (int i = 0; i < pattern.dimension(); i++) {
            min = Math.min(min, pattern.get(i));
        }
        return min;
    }

    private double getPatternMax(Pattern pattern) {
        double max = -Double.MAX_VALUE;
        for (int i = 0; i < pattern.dimension(); i++) {
            max = Math.max(max, pattern.get(i));
        }
        return max;
    }
}