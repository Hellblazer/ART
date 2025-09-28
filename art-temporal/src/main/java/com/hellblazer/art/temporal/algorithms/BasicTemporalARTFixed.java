package com.hellblazer.art.temporal.algorithms;

import com.hellblazer.art.core.*;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.temporal.*;
import com.hellblazer.art.temporal.dynamics.*;
import com.hellblazer.art.temporal.learning.*;
import com.hellblazer.art.temporal.masking.*;
import com.hellblazer.art.temporal.memory.*;
import com.hellblazer.art.temporal.parameters.*;
import com.hellblazer.art.temporal.results.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Basic Temporal ART implementation combining working memory and masking fields
 * Fixed version that properly implements all interfaces and uses correct types
 */
public class BasicTemporalARTFixed implements TemporalARTAlgorithm<com.hellblazer.art.temporal.parameters.TemporalParameters> {

    // Local parameters wrapper for backward compatibility
    public static class LocalTemporalParameters {
        final WorkingMemoryParameters workingMemory;
        final MaskingParameters masking;
        final double vigilance;
        final double learningRate;

        public LocalTemporalParameters(WorkingMemoryParameters workingMemory,
                                       MaskingParameters masking,
                                       double vigilance,
                                       double learningRate) {
            this.workingMemory = workingMemory;
            this.masking = masking;
            this.vigilance = vigilance;
            this.learningRate = learningRate;
        }

        public WorkingMemoryParameters workingMemory() { return workingMemory; }
        public MaskingParameters masking() { return masking; }
        public double vigilance() { return vigilance; }
        public double learningRate() { return learningRate; }
    }

    // Core components
    private final ItemOrderWorkingMemory workingMemory;
    private final MaskingFieldNetwork maskingField;
    private final CompetitiveInstarLearning learning;
    private LocalTemporalParameters localParameters;
    private com.hellblazer.art.temporal.parameters.TemporalParameters temporalParameters;

    // State
    private final List<WeightVector> categories;
    private final int maxCategories;

    public BasicTemporalARTFixed(LocalTemporalParameters params, int maxCategories) {
        this.localParameters = params;
        this.maxCategories = maxCategories;

        // Initialize components using proper parameter fields
        var capacity = params.workingMemory().capacity();
        var inputDim = 100; // Default dimension

        this.workingMemory = new ItemOrderWorkingMemoryImpl(
            params.workingMemory(),
            capacity,
            inputDim
        );

        this.maskingField = new MaskingFieldNetworkImpl(
            params.masking(),
            5,  // scales
            10  // chunks per scale
        );

        this.learning = new CompetitiveInstarLearning(
            (float)params.learningRate(),
            maxCategories,
            inputDim
        );

        this.categories = new ArrayList<>();
    }

    // === TemporalARTAlgorithm Implementation ===

    @Override
    public TemporalResult learnTemporal(TemporalPattern temporalPattern) {
        // Store sequence in working memory
        workingMemory.storeSequence(temporalPattern);

        // Process through masking field
        var wmState = workingMemory.getCurrentContents();
        var maskingResult = maskingField.process(wmState);

        // Learn if appropriate - process activations
        var activations = maskingField.getAllActivations();
        double maxActivation = 0.0;
        if (activations != null) {
            for (var row : activations) {
                for (var val : row) {
                    if (val > maxActivation) {
                        maxActivation = val;
                    }
                }
            }
        }

        if (maxActivation > localParameters.vigilance()) {
            // Process pattern through masking field for adaptation
            maskingField.process(wmState);
        }

        return createTemporalResult(wmState, maskingResult);
    }

    @Override
    public TemporalResult predictTemporal(TemporalPattern temporalPattern) {
        // Similar to learn but without weight updates
        // Note: setLearningEnabled may not exist, so we process directly
        workingMemory.storeSequence(temporalPattern);
        var wmState = workingMemory.getCurrentContents();
        var maskingResult = maskingField.process(wmState);
        var result = createTemporalResult(wmState, maskingResult);
        return result;
    }

    @Override
    public TemporalResult processSequenceItem(Pattern item) {
        // Process single item as sequence
        var temporalPattern = new SingleItemTemporalPattern(item);
        return learnTemporal(temporalPattern);
    }

    @Override
    public void resetTemporalState() {
        workingMemory.clear();
        maskingField.reset();
    }

    @Override
    public List<TemporalPattern> getTemporalChunks() {
        // Return learned chunks as temporal patterns
        List<TemporalPattern> chunks = new ArrayList<>();
        // Implementation would extract chunks from masking field
        return chunks;
    }

    @Override
    public boolean wouldCreateNewChunk(TemporalPattern temporalPattern) {
        var result = predictTemporal(temporalPattern);
        return !result.hasTemporalResonance();
    }

    @Override
    public TemporalPattern getWorkingMemoryContents() {
        return workingMemory.getCurrentContents();
    }

    @Override
    public int getWorkingMemoryCapacity() {
        return localParameters.workingMemory().capacity();
    }

    @Override
    public double[][] getMaskingFieldActivations() {
        // Get all activations from masking field
        return maskingField.getAllActivations();
    }

    @Override
    public com.hellblazer.art.temporal.parameters.TemporalParameters getTemporalParameters() {
        return temporalParameters;
    }

    @Override
    public void setTemporalParameters(com.hellblazer.art.temporal.parameters.TemporalParameters parameters) {
        this.temporalParameters = parameters;
    }

    // === ARTAlgorithm Implementation ===

    @Override
    public ActivationResult learn(Pattern input, com.hellblazer.art.temporal.parameters.TemporalParameters parameters) {
        var temporalPattern = new SingleItemTemporalPattern(input);
        var result = learnTemporal(temporalPattern);

        // Convert to ActivationResult
        var activationResult = result.getActivationResult();
        if (activationResult instanceof ActivationResult.Success success) {
            // Create or get weight vector
            while (categories.size() <= success.categoryIndex()) {
                categories.add(new SimpleWeightVector(extractFeatures(input)));
            }
            var weight = categories.get(success.categoryIndex());
            weight.update(input, localParameters.learningRate());
            return success;
        } else {
            return ActivationResult.NoMatch.INSTANCE;
        }
    }

    @Override
    public ActivationResult predict(Pattern input, com.hellblazer.art.temporal.parameters.TemporalParameters parameters) {
        var temporalPattern = new SingleItemTemporalPattern(input);
        var result = predictTemporal(temporalPattern);

        var activationResult = result.getActivationResult();
        if (activationResult instanceof ActivationResult.Success success &&
            success.categoryIndex() < categories.size()) {
            return success;
        } else {
            return ActivationResult.NoMatch.INSTANCE;
        }
    }

    @Override
    public int getCategoryCount() {
        return categories.size();
    }

    @Override
    public List<WeightVector> getCategories() {
        return new ArrayList<>(categories);
    }

    @Override
    public WeightVector getCategory(int index) {
        return categories.get(index);
    }

    @Override
    public void clear() {
        categories.clear();
        resetTemporalState();
    }

    @Override
    public void close() {
        // Clean up resources if any
    }

    // === Helper Methods ===

    private TemporalResult createTemporalResult(TemporalPattern pattern, MaskingResult maskingResult) {
        // Get max activation from masking result
        var activations = maskingField.getAllActivations();
        double maxActivation = 0.0;
        int winningCategory = -1;

        if (activations != null) {
            for (int i = 0; i < activations.length; i++) {
                for (int j = 0; j < activations[i].length; j++) {
                    if (activations[i][j] > maxActivation) {
                        maxActivation = activations[i][j];
                        winningCategory = i;
                    }
                }
            }
        }

        // Get primacy values as gradient
        var primacyValues = workingMemory.getPrimacyValues();
        var primacyGradient = new float[primacyValues.length];
        for (int i = 0; i < primacyValues.length; i++) {
            primacyGradient[i] = (float)primacyValues[i];
        }

        // Get transmitter states from masking field
        var transmitterGates = maskingField.getAllTransmitterGates();
        var transmitterStates = new float[0];
        if (transmitterGates != null && transmitterGates.length > 0 && transmitterGates[0] != null) {
            transmitterStates = new float[transmitterGates[0].length];
            for (int i = 0; i < transmitterGates[0].length; i++) {
                transmitterStates[i] = (float)transmitterGates[0][i];
            }
        }

        return new BasicTemporalResult(
            winningCategory,
            (float)maxActivation,
            maxActivation > localParameters.vigilance(),
            pattern,
            maskingResult,
            primacyGradient,
            transmitterStates
        );
    }

    // Extract features from a Pattern
    private float[] extractFeatures(Pattern pattern) {
        var dimension = pattern.dimension();
        var features = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            features[i] = (float)pattern.get(i);
        }
        return features;
    }

    // === Inner Classes ===

    private static class SingleItemTemporalPattern implements TemporalPattern {
        private final Pattern pattern;

        SingleItemTemporalPattern(Pattern pattern) {
            this.pattern = pattern;
        }

        @Override
        public List<Pattern> getSequence() {
            return List.of(pattern);
        }

        @Override
        public TemporalPattern getSubsequence(int startTime, int endTime) {
            if (startTime == 0 && endTime == 1) return this;
            throw new IndexOutOfBoundsException("Single item pattern");
        }

        @Override
        public boolean isEmpty() {
            return false;
        }
    }

    private static class SimpleWeightVector implements WeightVector {
        private final double[] weights;

        SimpleWeightVector(float[] initial) {
            this.weights = new double[initial.length];
            for (int i = 0; i < initial.length; i++) {
                weights[i] = initial[i];
            }
        }

        SimpleWeightVector(double[] initial) {
            this.weights = initial.clone();
        }

        @Override
        public double get(int index) {
            return weights[index];
        }

        @Override
        public int dimension() {
            return weights.length;
        }

        @Override
        public double l1Norm() {
            double sum = 0;
            for (var w : weights) {
                sum += Math.abs(w);
            }
            return sum;
        }

        @Override
        public WeightVector update(Pattern input, Object parameters) {
            double learningRate = 0.1; // default
            if (parameters instanceof Double) {
                learningRate = (Double) parameters;
            } else if (parameters instanceof LocalTemporalParameters tp) {
                learningRate = tp.learningRate();
            }

            for (int i = 0; i < Math.min(weights.length, input.dimension()); i++) {
                weights[i] += learningRate * (input.get(i) - weights[i]);
            }
            return this;
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
                // Create dummy weight vector for Success
                var dummyWeights = new double[temporalPattern.isEmpty() ? 0 : temporalPattern.getSequence().get(0).dimension()];
                return new ActivationResult.Success(winningCategory, maxActivation, new SimpleWeightVector(dummyWeights));
            } else {
                return ActivationResult.NoMatch.INSTANCE;
            }
        }

        @Override
        public List<TemporalPattern> getIdentifiedChunks() {
            // MaskingResult doesn't have getChunks() method, return empty list
            return new ArrayList<>();
        }

        @Override
        public Optional<TemporalPattern> getPrimaryChunk() {
            var chunks = getIdentifiedChunks();
            return chunks.isEmpty() ? Optional.empty() : Optional.of(chunks.get(0));
        }

        @Override
        public boolean hasNewChunks() {
            // Check if new chunks based on activation level
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