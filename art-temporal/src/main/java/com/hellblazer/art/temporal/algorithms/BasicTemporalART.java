package com.hellblazer.art.temporal.algorithms;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.temporal.*;
import com.hellblazer.art.temporal.dynamics.*;
import com.hellblazer.art.temporal.learning.*;
import com.hellblazer.art.temporal.masking.*;
import com.hellblazer.art.temporal.memory.*;
import com.hellblazer.art.temporal.parameters.*;
import com.hellblazer.art.temporal.results.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

/**
 * Basic Temporal ART implementation combining working memory and masking fields
 * Implements real-time sequence chunking and learning
 * Based on Kazerounian & Grossberg (2014)
 */
public class BasicTemporalART implements TemporalARTAlgorithm<TemporalParameters> {

    // Core components
    private final ItemOrderWorkingMemory workingMemory;
    private final MaskingFieldNetwork maskingField;
    private final CompetitiveInstarLearning learning;
    private final HabituativeTransmitterGates transmitterGates;
    private final ShuntingDynamics shuntingDynamics;

    // Parameters
    private final WorkingMemoryParameters wmParameters;
    private final MaskingParameters mfParameters;
    private final float vigilance;
    private final float learningRate;

    // State
    private List<TemporalCategory> categories;
    private int maxCategories;
    private boolean supervised;
    private TemporalParameters temporalParams;

    // Performance metrics
    private long processingTime;
    private int learningCycles;

    public BasicTemporalART(WorkingMemoryParameters wmParameters,
                           MaskingParameters mfParameters,
                           float vigilance,
                           float learningRate,
                           int maxCategories) {
        this.wmParameters = wmParameters;
        this.mfParameters = mfParameters;
        this.vigilance = vigilance;
        this.learningRate = learningRate;
        this.maxCategories = maxCategories;

        // Initialize components
        var maxSeqLength = wmParameters.capacity();
        var inputDim = mfParameters.fieldSize();

        this.workingMemory = new ItemOrderWorkingMemoryImpl(wmParameters, maxSeqLength, inputDim);
        this.maskingField = new MaskingFieldNetworkImpl(mfParameters, 5, 10);  // 5 scales, 10 chunks each
        this.learning = new CompetitiveInstarLearning(learningRate, maxCategories, inputDim);

        var numGates = maxSeqLength * inputDim;
        this.transmitterGates = new HabituativeTransmitterGates(
            (float)mfParameters.transmitterRecoveryRate(),
            (float)mfParameters.transmitterDepletionRate(),
            0.1f,  // Quadratic rate - enables proper habituation/depletion
            numGates,
            (float)mfParameters.timeStep()
        );

        this.shuntingDynamics = new ShuntingDynamics(
            (float)wmParameters.decayRate(),
            (float)wmParameters.maxActivation(),  // use max activation as upper bound
            0.0f,  // lower bound not in parameters
            (float)wmParameters.temporalResolution()
        );

        this.categories = new ArrayList<>();
        this.supervised = false;
    }

    // Process a sequence of patterns
    public TemporalResult processSequence(List<Pattern> sequence) {
        var startTime = System.nanoTime();

        // Store sequence in working memory
        var temporalPattern = new TemporalPatternImpl(sequence);
        workingMemory.storeSequence(temporalPattern);

        // Update working memory dynamics
        for (int t = 0; t < 10; t++) {  // 10 time steps
            workingMemory.updateDynamics(wmParameters.temporalResolution());
        }

        // Get current working memory state
        var currentState = workingMemory.getCurrentContents();

        // Process through masking field
        var maskingResult = maskingField.process(currentState);

        // Apply transmitter gates
        var features = extractFeatures(currentState);
        var gatedPattern = transmitterGates.applyGates(features);

        // Find best matching category
        var matchResult = findBestMatch(gatedPattern);

        // Learn if appropriate
        if (shouldLearn(matchResult)) {
            performLearning(temporalPattern, matchResult);
            learningCycles++;
        }

        // Update transmitter gates based on gated pattern activations
        // The gates should be updated with the actual pattern that was processed
        // This creates habituation to repeated patterns
        transmitterGates.updateGates(gatedPattern);

        // Create result
        processingTime = System.nanoTime() - startTime;
        return createTemporalResult(currentState, maskingResult, matchResult);
    }

    // Learn from a single pattern
    public void learn(Pattern pattern, boolean supervised) {
        this.supervised = supervised;

        // Convert single pattern to sequence
        var sequence = List.of(pattern);
        var result = processSequence(sequence);

        // processSequence already handles learning and category creation via performLearning
        // We only need to update weights if temporal resonance was achieved
        if (result.hasTemporalResonance()) {
            // Update category weights if we have a winning category
            var activationResult = result.getActivationResult();
            if (activationResult instanceof ActivationResult.Success success) {
                var categoryIdx = success.categoryIndex();
                if (categoryIdx >= 0 && categoryIdx < categories.size()) {
                    var features = extractPatternFeatures(pattern);
                    categories.get(categoryIdx).updateWeights(features);
                }
            }
        }
        // Don't create new category here - performLearning in processSequence already handles it
    }

    // Predict category for a pattern
    public ActivationResult predict(Pattern pattern) {
        // Process as single-item sequence
        var sequence = List.of(pattern);
        var result = processSequence(sequence);

        // Return the activation result from temporal result
        return result.getActivationResult();
    }

    @Override
    public int getCategoryCount() {
        return categories.size();
    }

    public void setParameters(TemporalParameters parameters) {
        this.temporalParams = parameters;
    }

    // Get temporal parameters
    public TemporalParameters getParameters() {
        return temporalParams;
    }

    // TemporalARTAlgorithm interface methods

    @Override
    public TemporalResult learnTemporal(TemporalPattern temporalPattern) {
        workingMemory.storeSequence(temporalPattern);
        return processSequence(temporalPattern.getSequence());
    }

    @Override
    public TemporalResult predictTemporal(TemporalPattern temporalPattern) {
        workingMemory.storeSequence(temporalPattern);
        return processSequence(temporalPattern.getSequence());
    }

    // Learn from a batch of patterns
    public void learnBatch(List<Pattern> patterns) {
        for (var pattern : patterns) {
            learn(pattern, supervised);
        }
    }

    // Add missing ARTAlgorithm interface methods
    public WeightVector getCategory(int index) {
        if (index >= 0 && index < categories.size()) {
            var category = categories.get(index);
            var categoryWeights = category.getWeights();
            return new WeightVector() {
                @Override
                public double get(int index) {
                    return categoryWeights[index];
                }

                @Override
                public int dimension() {
                    return categoryWeights.length;
                }

                @Override
                public double l1Norm() {
                    double sum = 0;
                    for (var w : categoryWeights) {
                        sum += Math.abs(w);
                    }
                    return sum;
                }

                @Override
                public WeightVector update(Pattern input, Object params) {
                    category.updateWeights(extractPatternFeatures(input));
                    return this;
                }

            };
        }
        return null;
    }

    // Get all categories
    public List<WeightVector> getCategories() {
        var result = new ArrayList<WeightVector>();
        for (int i = 0; i < categories.size(); i++) {
            result.add(getCategory(i));
        }
        return result;
    }

    // Helper method to convert double[] to float[]
    private float[] convertPrimacyValues(double[] primacyValues) {
        var result = new float[primacyValues.length];
        for (int i = 0; i < primacyValues.length; i++) {
            result[i] = (float)primacyValues[i];
        }
        return result;
    }

    @Override
    public TemporalResult processSequenceItem(Pattern item) {
        workingMemory.storeItem(item, System.currentTimeMillis());
        return processSequence(List.of(item));
    }

    @Override
    public void resetTemporalState() {
        workingMemory.clear();
        maskingField.reset();
        transmitterGates.resetAllGates();
    }

    @Override
    public List<TemporalPattern> getTemporalChunks() {
        // Get chunks from the masking field's current state
        var boundaries = maskingField.detectChunkBoundaries();
        var chunks = new ArrayList<TemporalPattern>();
        var currentState = workingMemory.getCurrentContents();

        if (boundaries.length > 0 && !currentState.isEmpty()) {
            var sequence = currentState.getSequence();
            int start = 0;
            for (int boundary : boundaries) {
                if (boundary > start && boundary <= sequence.size()) {
                    chunks.add(currentState.getSubsequence(start, boundary));
                    start = boundary;
                }
            }
            // Add final chunk if any remaining
            if (start < sequence.size()) {
                chunks.add(currentState.getSubsequence(start, sequence.size()));
            }
        }
        return chunks;
    }

    @Override
    public boolean wouldCreateNewChunk(TemporalPattern temporalPattern) {
        var result = predictTemporal(temporalPattern);
        return result.hasNewChunks();
    }

    @Override
    public TemporalPattern getWorkingMemoryContents() {
        return workingMemory.getCurrentContents();
    }

    @Override
    public int getWorkingMemoryCapacity() {
        return wmParameters.capacity();
    }

    @Override
    public double[][] getMaskingFieldActivations() {
        return maskingField.getAllActivations();
    }

    @Override
    public TemporalParameters getTemporalParameters() {
        return temporalParams;
    }

    @Override
    public void setTemporalParameters(TemporalParameters parameters) {
        this.temporalParams = parameters;
    }

    public void reset() {
        resetTemporalState();
        categories.clear();
        learningCycles = 0;
        processingTime = 0;
    }

    // Add missing clear method from ARTAlgorithm interface
    public void clear() {
        reset();
    }

    // AutoCloseable interface method
    @Override
    public void close() throws Exception {
        // Clean up resources if needed
        if (workingMemory != null) {
            workingMemory.close();
        }
        if (maskingField != null) {
            maskingField.close();
        }
    }

    // Helper methods

    // Extract features from a temporal pattern
    private float[] extractFeatures(TemporalPattern pattern) {
        if (pattern.isEmpty()) {
            return new float[0];
        }

        var sequences = pattern.getSequence();
        if (sequences.isEmpty()) {
            return new float[0];
        }

        // Get the first pattern's dimension for feature extraction
        var firstPattern = sequences.get(0);
        var dimension = firstPattern.dimension();

        // Create feature vector by averaging across sequence
        var features = new float[dimension];
        for (var p : sequences) {
            for (int i = 0; i < dimension; i++) {
                features[i] += (float)p.get(i);
            }
        }

        // Normalize by sequence length
        var length = sequences.size();
        for (int i = 0; i < dimension; i++) {
            features[i] /= length;
        }

        return features;
    }

    // Extract features from a single pattern
    private float[] extractPatternFeatures(Pattern pattern) {
        var dimension = pattern.dimension();
        var features = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            features[i] = (float)pattern.get(i);
        }
        return features;
    }

    private MatchResult findBestMatch(float[] pattern) {
        var bestMatch = -1;
        var bestActivation = 0.0f;
        var bestSimilarity = 0.0f;

        for (int i = 0; i < categories.size(); i++) {
            var category = categories.get(i);
            var activation = category.computeActivation(pattern);

            if (activation > bestActivation) {
                bestActivation = activation;
                bestMatch = i;
                bestSimilarity = category.computeSimilarity(pattern);
            }
        }

        return new MatchResult(bestMatch, bestActivation, bestSimilarity);
    }

    private boolean shouldLearn(MatchResult match) {
        // Learn if no match or match below vigilance
        return match.categoryIndex < 0 ||
               match.similarity < vigilance ||
               (supervised && match.similarity < vigilance * 1.2f);
    }

    private void performLearning(TemporalPattern pattern, MatchResult match) {
        if (match.categoryIndex >= 0) {
            // Update existing category
            var category = categories.get(match.categoryIndex);
            category.updateWeights(extractFeatures(pattern));
            var allActivations = maskingField.getAllActivations();
            if (allActivations.length > 0) {
                var floatActivations = new float[allActivations[0].length];
                for (int i = 0; i < allActivations[0].length; i++) {
                    floatActivations[i] = (float)allActivations[0][i];
                }
                learning.updateWeights(extractFeatures(pattern),
                                      floatActivations,
                                      match.categoryIndex);
            }
        } else if (categories.size() < maxCategories) {
            // Create new category
            createNewCategory(pattern);
        }

        // Process pattern through masking field for adaptation
        maskingField.process(pattern);
    }

    private void createNewCategory(Pattern pattern) {
        createNewCategory(new TemporalPatternWrapper(pattern));
    }

    private void createNewCategory(TemporalPattern pattern) {
        var newCategory = new TemporalCategory(
            categories.size(),
            extractFeatures(pattern),
            pattern.getSequence().size()
        );
        categories.add(newCategory);
    }

    private TemporalResult createTemporalResult(TemporalPattern pattern,
                                               MaskingResult maskingResult,
                                               MatchResult matchResult) {
        return new BasicTemporalResult(
            matchResult.categoryIndex,
            matchResult.activation,
            matchResult.similarity >= vigilance,
            pattern,
            maskingResult,
            convertPrimacyValues(workingMemory.getPrimacyValues()),
            transmitterGates.getTransmitterLevels()
        );
    }

    // Inner classes

    private static class TemporalPatternImpl implements TemporalPattern {
        private final List<Pattern> sequence;

        TemporalPatternImpl(List<Pattern> sequence) {
            this.sequence = new ArrayList<>(sequence);
        }

        @Override
        public List<Pattern> getSequence() {
            return sequence;
        }

        @Override
        public TemporalPattern getSubsequence(int startTime, int endTime) {
            if (startTime < 0 || endTime > sequence.size() || startTime >= endTime) {
                throw new IndexOutOfBoundsException("Invalid subsequence bounds");
            }
            return new TemporalPatternImpl(sequence.subList(startTime, endTime));
        }

        @Override
        public boolean isEmpty() {
            return sequence.isEmpty();
        }
    }

    private static class MatchResult {
        final int categoryIndex;
        final float activation;
        final float similarity;

        MatchResult(int categoryIndex, float activation, float similarity) {
            this.categoryIndex = categoryIndex;
            this.activation = activation;
            this.similarity = similarity;
        }
    }

    private static class TemporalCategory {
        private final int index;
        private float[] weights;
        private int sequenceLength;
        private int updateCount;

        TemporalCategory(int index, float[] initialWeights, int sequenceLength) {
            this.index = index;
            this.weights = Arrays.copyOf(initialWeights, initialWeights.length);
            this.sequenceLength = sequenceLength;
            this.updateCount = 0;
        }

        float computeActivation(float[] input) {
            var activation = 0.0f;
            for (int i = 0; i < Math.min(input.length, weights.length); i++) {
                activation += input[i] * weights[i];
            }
            return activation;
        }

        float computeSimilarity(float[] input) {
            var dotProduct = computeActivation(input);
            var inputNorm = norm(input);
            var weightNorm = norm(weights);

            if (inputNorm > 0 && weightNorm > 0) {
                return dotProduct / (inputNorm * weightNorm);
            }
            return 0;
        }

        void updateWeights(float[] input) {
            // Simple averaging update
            for (int i = 0; i < Math.min(input.length, weights.length); i++) {
                weights[i] = (weights[i] * updateCount + input[i]) / (updateCount + 1);
            }
            updateCount++;
        }

        private float norm(float[] vector) {
            var sum = 0.0f;
            for (var val : vector) {
                sum += val * val;
            }
            return (float) Math.sqrt(sum);
        }

        float[] getWeights() {
            return Arrays.copyOf(weights, weights.length);
        }
    }

    private static class TemporalPatternWrapper implements TemporalPattern {
        private final Pattern pattern;

        TemporalPatternWrapper(Pattern pattern) {
            this.pattern = pattern;
        }

        @Override
        public List<Pattern> getSequence() {
            return List.of(pattern);
        }

        @Override
        public TemporalPattern getSubsequence(int startTime, int endTime) {
            if (startTime == 0 && endTime == 1) return this;
            throw new IndexOutOfBoundsException("Single-item pattern only has one element");
        }

        @Override
        public boolean isEmpty() {
            return false;
        }
    }

    private static class BasicTemporalResult implements TemporalResult {
        private final int winningCategory;
        private final float maxActivation;
        private final boolean resonant;
        private final TemporalPattern temporalPattern;
        private final MaskingResult maskingResult;
        private final float[] primacyGradient;
        private final float[] transmitterStates;

        BasicTemporalResult(int winningCategory, float maxActivation, boolean resonant,
                          TemporalPattern temporalPattern, MaskingResult maskingResult,
                          float[] primacyGradient, float[] transmitterStates) {
            this.winningCategory = winningCategory;
            this.maxActivation = maxActivation;
            this.resonant = resonant;
            this.temporalPattern = temporalPattern;
            this.maskingResult = maskingResult;
            this.primacyGradient = primacyGradient;
            this.transmitterStates = transmitterStates;
        }

        @Override
        public ActivationResult getActivationResult() {
            if (winningCategory >= 0) {
                // Create a dummy weight vector since Success requires it
                return new ActivationResult.Success(winningCategory, (double)maxActivation, new WeightVector() {
                    @Override
                    public double get(int index) { return 0; }
                    @Override
                    public int dimension() { return 0; }
                    @Override
                    public double l1Norm() { return 0; }
                    @Override
                    public WeightVector update(Pattern input, Object params) { return this; }
                });
            } else {
                return ActivationResult.NoMatch.INSTANCE;
            }
        }

        @Override
        public List<TemporalPattern> getIdentifiedChunks() {
            // For now, return empty list as MaskingResult doesn't have getChunks()
            return new ArrayList<>();
        }

        @Override
        public Optional<TemporalPattern> getPrimaryChunk() {
            var chunks = getIdentifiedChunks();
            return chunks.isEmpty() ? Optional.empty() : Optional.of(chunks.get(0));
        }

        @Override
        public boolean hasNewChunks() {
            // Check if new chunks were created based on activation threshold
            return maskingResult != null && maskingResult.getMaxActivation() > 0.5;
        }

        @Override
        public TemporalPattern getWorkingMemoryState() {
            return temporalPattern;
        }

        @Override
        public double[][] getMaskingFieldActivations() {
            return maskingResult != null ? maskingResult.getActivations() : new double[0][0];
        }

        @Override
        public double[][] getTransmitterGateValues() {
            // Convert float[] to double[][]
            if (transmitterStates == null) return new double[0][0];
            var result = new double[1][transmitterStates.length];
            for (int i = 0; i < transmitterStates.length; i++) {
                result[0][i] = transmitterStates[i];
            }
            return result;
        }

        @Override
        public double getProcessingTime() {
            return 0.0;  // Would need to store this
        }

        @Override
        public boolean hasTemporalResonance() {
            return resonant;
        }

        @Override
        public double getResonanceQuality() {
            return resonant ? maxActivation : 0.0;
        }

        @Override
        public boolean requiredChunking() {
            // Check if chunking was required based on boundaries
            return maskingResult != null && maskingResult.getChunkBoundaries().length > 0;
        }

        @Override
        public List<Integer> getChunkBoundaries() {
            var list = new ArrayList<Integer>();
            if (maskingResult != null) {
                var boundaries = maskingResult.getChunkBoundaries();
                for (int boundary : boundaries) {
                    list.add(boundary);
                }
            }
            return list;
        }

        @Override
        public Optional<TemporalPrediction> getPrediction() {
            return Optional.empty();  // Not implemented for basic version
        }

        @Override
        public TemporalPerformanceMetrics getPerformanceMetrics() {
            return new BasicPerformanceMetrics();
        }
    }

    private static class BasicPerformanceMetrics implements TemporalResult.TemporalPerformanceMetrics {
        @Override
        public double getWorkingMemoryTime() { return 0.0; }

        @Override
        public double getMaskingFieldTime() { return 0.0; }

        @Override
        public double getChunkingTime() { return 0.0; }

        @Override
        public long getMemoryUsage() { return 0L; }

        @Override
        public long getSIMDOperationCount() { return 0L; }
    }

}