package com.hellblazer.art.temporal.integration;

import com.hellblazer.art.temporal.memory.WorkingMemory;
import com.hellblazer.art.temporal.memory.WorkingMemoryParameters;
import com.hellblazer.art.temporal.masking.MaskingField;
import com.hellblazer.art.temporal.masking.MaskingFieldParameters;
import com.hellblazer.art.core.BaseART;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.results.ActivationResult;

import java.util.ArrayList;
import java.util.List;

/**
 * TemporalART: Integration of working memory and masking field for temporal sequence learning.
 * Based on Kazerounian & Grossberg (2014) neural dynamics of speech and language coding.
 */
public class TemporalART extends BaseART<TemporalARTParameters> {

    private final TemporalARTParameters parameters;
    private final WorkingMemory workingMemory;
    private final MaskingField maskingField;
    private final List<TemporalCategory> categories;
    private double currentTime;
    private TemporalARTState currentState;
    private boolean learningEnabled;

    public TemporalART(TemporalARTParameters parameters) {
        this.parameters = parameters;
        this.workingMemory = new WorkingMemory(parameters.getWorkingMemoryParameters());
        this.maskingField = new MaskingField(
            parameters.getMaskingFieldParameters(),
            workingMemory
        );
        this.categories = new ArrayList<>();
        this.currentTime = 0.0;
        this.learningEnabled = true;
        this.currentState = createInitialState();
    }

    /**
     * Learn from double array (backward compatibility).
     */
    public void learn(double[] pattern) {
        var patternInput = Pattern.of(pattern);
        learn(patternInput, parameters);
    }

    /**
     * Predict from double array (backward compatibility).
     */
    public int predict(double[] pattern) {
        var patternInput = Pattern.of(pattern);
        var result = predict(patternInput, parameters);
        return result instanceof ActivationResult.Success success ? success.categoryIndex() : -1;
    }


    // Track the last processed input to avoid duplicate temporal processing
    private Pattern lastProcessedInput;

    /**
     * Internal method for temporal processing with duplicate protection.
     */
    private void processTemporalInput(Pattern input) {
        // Avoid duplicate processing for the same input within a single learning step
        if (input == lastProcessedInput) {
            return;
        }
        lastProcessedInput = input;
        var inputData = extractPatternData(input);

        // Store pattern in working memory
        workingMemory.storeItem(inputData, currentTime);

        // Process through masking field
        var temporalPattern = workingMemory.getTemporalPattern();
        // Convert WorkingMemory.TemporalPattern to TemporalPattern
        var convertedPattern = convertTemporalPattern(temporalPattern);
        maskingField.processTemporalPattern(convertedPattern);

        // Get winning chunks from masking field
        var chunks = maskingField.getListChunks();
        var winners = maskingField.getState().getWinningNodes();

        if (learningEnabled && !winners.isEmpty()) {
            // Create or update temporal categories based on chunks
            processTemporalCategories(chunks, temporalPattern);
        }

        // Update time
        currentTime += parameters.getTimeStep();

        // Update state
        updateState();
    }

    // Abstract methods from BaseART

    @Override
    protected double calculateActivation(Pattern input, WeightVector weight, TemporalARTParameters params) {
        // Calculate activation based on temporal pattern similarity
        var inputArray = extractPatternData(input);
        var weightArray = weight instanceof TemporalWeight tw ? tw.getData() : extractWeightData(weight);

        double dotProduct = 0.0;
        double inputNorm = 0.0;
        double weightNorm = 0.0;

        for (int i = 0; i < Math.min(inputArray.length, weightArray.length); i++) {
            dotProduct += inputArray[i] * weightArray[i];
            inputNorm += inputArray[i] * inputArray[i];
            weightNorm += weightArray[i] * weightArray[i];
        }

        if (inputNorm > 0 && weightNorm > 0) {
            return dotProduct / (Math.sqrt(inputNorm) * Math.sqrt(weightNorm));
        }
        return 0.0;
    }

    @Override
    protected com.hellblazer.art.core.results.MatchResult checkVigilance(Pattern input, WeightVector weight, TemporalARTParameters params) {
        double match = calculateActivation(input, weight, params);
        boolean passes = match >= params.getVigilance();
        return passes ? new com.hellblazer.art.core.results.MatchResult.Accepted(match, params.getVigilance()) :
                       new com.hellblazer.art.core.results.MatchResult.Rejected(match, params.getVigilance());
    }

    @Override
    protected WeightVector updateWeights(Pattern input, WeightVector currentWeight, TemporalARTParameters params) {
        // Ensure temporal processing happens for weight updates
        processTemporalInput(input);

        var inputData = extractPatternData(input);
        var weightData = currentWeight instanceof TemporalWeight tw ? tw.getData() : extractWeightData(currentWeight);
        var learningRate = params.getLearningRate();

        double[] newWeights = new double[weightData.length];
        for (int i = 0; i < weightData.length; i++) {
            if (i < inputData.length) {
                newWeights[i] = (1 - learningRate) * weightData[i] + learningRate * inputData[i];
            } else {
                newWeights[i] = weightData[i];
            }
        }

        return new TemporalWeight(newWeights);
    }

    @Override
    protected WeightVector createInitialWeight(Pattern input, TemporalARTParameters params) {
        // Ensure temporal processing happens for new category creation
        processTemporalInput(input);

        var inputData = extractPatternData(input);
        return new TemporalWeight(inputData.clone());
    }

    // Original TemporalART methods

    public TemporalARTState getState() {
        return currentState;
    }

    public void reset() {
        super.clear();
        categories.clear(); // Clear temporal categories
        workingMemory.reset();
        maskingField.reset();
        currentTime = 0.0;
        learningEnabled = true;
        lastProcessedInput = null; // Reset temporal processing guard
        updateState(); // Update state after all components are reset
    }

    /**
     * Process temporal sequences to form or update categories.
     */
    private void processTemporalCategories(
        List<com.hellblazer.art.temporal.masking.ListChunk> chunks,
        com.hellblazer.art.temporal.memory.WorkingMemory.TemporalPattern temporalPattern
    ) {

        // Process each chunk as a potential category
        for (var chunk : chunks) {
            var chunkPattern = chunk.getChunkPattern();
            if (chunkPattern.length == 0) continue;

            // Check for resonance with existing categories
            boolean found = false;
            for (var category : categories) {
                double match = category.computeMatch(chunkPattern);
                if (match >= parameters.getVigilance()) {
                    // Update existing category
                    if (learningEnabled) {
                        category.update(chunkPattern, parameters.getLearningRate());
                    }
                    found = true;
                    break;
                }
            }

            // Create new category if no match found
            if (!found && learningEnabled && categories.size() < parameters.getMaxCategories()) {
                categories.add(new TemporalCategory(
                    chunkPattern,
                    chunk.size(),
                    chunk.getTemporalSpan(),
                    currentTime
                ));
            }
        }

        // Also process the full temporal pattern as a category
        var combinedPattern = temporalPattern.getCombinedPattern();
        if (combinedPattern.length > 0) {
            processPatternAsCategory(combinedPattern, temporalPattern.sequenceLength());
        }
    }

    /**
     * Process a pattern as a potential category.
     */
    private void processPatternAsCategory(double[] pattern, int sequenceLength) {
        boolean found = false;
        for (var category : categories) {
            double match = category.computeMatch(pattern);
            if (match >= parameters.getVigilance()) {
                if (learningEnabled) {
                    category.update(pattern, parameters.getLearningRate());
                }
                found = true;
                break;
            }
        }

        if (!found && learningEnabled && categories.size() < parameters.getMaxCategories()) {
            categories.add(new TemporalCategory(
                pattern,
                sequenceLength,
                sequenceLength,
                currentTime
            ));
        }
    }

    /**
     * Find best matching category for a temporal pattern.
     */
    private int findBestMatchingCategory(
        com.hellblazer.art.temporal.memory.WorkingMemory.TemporalPattern pattern
    ) {
        var combinedPattern = pattern.getCombinedPattern();
        if (combinedPattern.length == 0 || categories.isEmpty()) {
            return -1;
        }

        double bestMatch = 0.0;
        int bestIndex = -1;

        for (int i = 0; i < categories.size(); i++) {
            double match = categories.get(i).computeMatch(combinedPattern);
            if (match > bestMatch && match >= parameters.getVigilance()) {
                bestMatch = match;
                bestIndex = i;
            }
        }

        return bestIndex;
    }

    /**
     * Create initial state.
     */
    private TemporalARTState createInitialState() {
        return new TemporalARTState(
            workingMemory.getState(),
            maskingField.getState(),
            new ArrayList<>(categories),
            currentTime,
            learningEnabled
        );
    }

    /**
     * Update current state.
     */
    private void updateState() {
        currentState = new TemporalARTState(
            workingMemory.getState(),
            maskingField.getState(),
            new ArrayList<>(categories),
            currentTime,
            learningEnabled
        );
    }

    /**
     * Enable or disable learning.
     */
    public void setLearningEnabled(boolean enabled) {
        this.learningEnabled = enabled;
    }

    /**
     * Get current simulation time.
     */
    public double getCurrentTime() {
        return currentTime;
    }

    /**
     * Get temporal categories.
     */
    public List<TemporalCategory> getTemporalCategories() {
        return new ArrayList<>(categories);
    }

    /**
     * Get the masking field (for debugging).
     */
    public MaskingField getMaskingField() {
        return maskingField;
    }

    /**
     * Process a complete sequence.
     */
    public void processSequence(List<double[]> sequence) {
        for (var pattern : sequence) {
            learn(pattern);
        }
    }

    /**
     * Predict sequence category.
     */
    public int predictSequence(List<double[]> sequence) {
        // Clear working memory for fresh prediction
        workingMemory.reset();

        // Store all patterns and get final prediction
        int lastPrediction = -1;
        for (var pattern : sequence) {
            lastPrediction = predict(pattern);
        }

        return lastPrediction;
    }

    /**
     * Get performance statistics.
     */
    public TemporalARTStatistics getStatistics() {
        return new TemporalARTStatistics(
            categories.size(),
            workingMemory.getState().getItemCount(),
            maskingField.getListChunks().size(),
            maskingField.getItemNodes().size(),
            calculateAverageChunkSize(),
            calculateCompressionRatio()
        );
    }

    private double calculateAverageChunkSize() {
        var chunks = maskingField.getListChunks();
        if (chunks.isEmpty()) return 0.0;

        int totalSize = chunks.stream()
            .mapToInt(c -> c.size())
            .sum();
        return (double) totalSize / chunks.size();
    }

    private double calculateCompressionRatio() {
        int totalItems = workingMemory.getState().getItemCount();
        int numChunks = maskingField.getListChunks().size();
        if (numChunks == 0) return 1.0;
        return (double) totalItems / numChunks;
    }

    /**
     * Extract weight data from a WeightVector.
     */
    private double[] extractWeightData(WeightVector weight) {
        var data = new double[weight.dimension()];
        for (int i = 0; i < weight.dimension(); i++) {
            data[i] = weight.get(i);
        }
        return data;
    }

    /**
     * Extract pattern data from a Pattern.
     */
    private double[] extractPatternData(Pattern pattern) {
        var data = new double[pattern.dimension()];
        for (int i = 0; i < pattern.dimension(); i++) {
            data[i] = pattern.get(i);
        }
        return data;
    }

    /**
     * Convert WorkingMemory.TemporalPattern to TemporalPattern.
     */
    private com.hellblazer.art.temporal.memory.TemporalPattern convertTemporalPattern(
        com.hellblazer.art.temporal.memory.WorkingMemory.TemporalPattern wmPattern) {
        // Extract data from the inner class and create the standalone class
        var combinedPattern = wmPattern.getCombinedPattern();
        var patterns = new java.util.ArrayList<double[]>();
        var weights = new java.util.ArrayList<Double>();

        // For simplicity, treat the combined pattern as a single pattern with weight 1.0
        patterns.add(combinedPattern);
        weights.add(1.0);

        return new com.hellblazer.art.temporal.memory.TemporalPattern(patterns, weights, parameters.getWorkingMemoryParameters().getPrimacyGradient());
    }

    /**
     * Close resources (required by AutoCloseable).
     */
    @Override
    public void close() {
        // Clean up any resources if needed
        reset();
    }
}