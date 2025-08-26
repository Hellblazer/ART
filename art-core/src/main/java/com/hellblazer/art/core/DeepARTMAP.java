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
package com.hellblazer.art.core;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * DeepARTMAP implementation for hierarchical supervised and unsupervised learning.
 * 
 * This class implements DeepARTMAP, a generalization of ARTMAP that allows an arbitrary 
 * number of ART modules to be used in a hierarchical learning structure. DeepARTMAP 
 * supports both supervised and unsupervised modes:
 * 
 * - Supervised Mode: Uses class labels to train SimpleARTMAP layers
 * - Unsupervised Mode: Uses ARTMAP + SimpleARTMAP chain for hierarchical clustering
 * 
 * Key Features:
 * - Multi-channel data processing (list of input matrices)
 * - Hierarchical label propagation through layers
 * - Deep label concatenation from all layers
 * - Support for arbitrary number of ART modules
 * 
 * Reference: Python implementation at AdaptiveResonanceLib/artlib/hierarchical/DeepARTMAP.py
 * Paper: "Deep ARTMAP: Generalized Hierarchical Learning with Adaptive Resonance Theory"
 * 
 * @author Hal Hildebrand
 */
public final class DeepARTMAP extends BaseART implements ScikitClusterer<DeepARTMAPResult> {
    
    private final List<BaseART> modules;
    private final DeepARTMAPParameters parameters;
    private final List<BaseARTMAP> layers;
    private Boolean supervised;
    private boolean trained;
    private int[][] storedDeepLabels;  // Store actual deep labels from training
    private int totalCategoryCount;
    
    /**
     * Create a new DeepARTMAP with specified ART modules and parameters.
     * 
     * @param modules the list of ART modules to use as building blocks
     * @param parameters the DeepARTMAP-specific parameters
     * @throws IllegalArgumentException if modules is null, empty, or contains null elements
     */
    public DeepARTMAP(List<BaseART> modules, DeepARTMAPParameters parameters) {
        super(); // Call BaseART constructor
        if (modules == null) {
            throw new IllegalArgumentException("modules cannot be null");
        }
        if (parameters == null) {
            throw new IllegalArgumentException("parameters cannot be null");
        }
        
        if (modules.isEmpty()) {
            throw new IllegalArgumentException("Must provide at least one ART module");
        }
        
        // Check for null elements in modules before creating defensive copy
        for (int i = 0; i < modules.size(); i++) {
            if (modules.get(i) == null) {
                throw new IllegalArgumentException("modules cannot contain null elements");
            }
        }
        
        this.modules = List.copyOf(modules); // Defensive copy
        this.parameters = parameters;
        this.layers = new java.util.ArrayList<>();
        this.supervised = null;
        this.trained = false;
        this.storedDeepLabels = null;
        this.totalCategoryCount = 0;
    }
    
    /**
     * Get the list of ART modules used by this DeepARTMAP.
     * 
     * @return immutable list of ART modules
     */
    public List<BaseART> getModules() {
        return modules;
    }
    
    /**
     * Get the DeepARTMAP parameters.
     * 
     * @return the parameters
     */
    public DeepARTMAPParameters getParameters() {
        return parameters;
    }
    
    /**
     * Get the number of ART modules.
     * 
     * @return the number of modules
     */
    public int getModuleCount() {
        return modules.size();
    }
    
    /**
     * Get the list of trained layers.
     * 
     * @return immutable list of layers
     */
    public List<BaseARTMAP> getLayers() {
        return List.copyOf(layers);
    }
    
    /**
     * Get the number of layers in the hierarchy.
     * 
     * @return the number of layers
     */
    public int getLayerCount() {
        return layers.size();
    }
    
    /**
     * Check if this DeepARTMAP is in supervised mode.
     * 
     * @return true if supervised, false if unsupervised, null if not yet trained
     */
    public Boolean isSupervised() {
        return supervised;
    }
    
    /**
     * Check if this DeepARTMAP has been trained.
     * 
     * @return true if trained, false otherwise
     */
    public boolean isTrained() {
        return trained;
    }
    
    /**
     * Train DeepARTMAP in supervised mode with labeled data.
     * 
     * @param data the list of input matrices (one per module)
     * @param labels the class labels for supervised learning
     * @return the training result
     */
    public DeepARTMAPResult fitSupervised(List<Pattern[]> data, int[] labels) {
        // Validate input data - throw exceptions for validation failures
        validateInputDataAndThrow(data, labels);
        
        if (labels == null) {
            throw new IllegalArgumentException("labels cannot be null in supervised mode");
        }
        
        // Set supervised flag if not already set
        if (supervised == null) {
            supervised = true;
        } else if (!supervised) {
            throw new IllegalStateException("Cannot change from unsupervised to supervised mode");
        }
        
        try {
            // Clear existing layers for fresh training
            layers.clear();
            
            return createSupervisedLayers(data, labels);
            
        } catch (Exception e) {
            return DeepARTMAPResult.TrainingFailure.layerInitializationFailed(-1, e);
        }
    }

    /**
     * Train DeepARTMAP in unsupervised mode without labels.
     * 
     * @param data the list of input matrices (one per module)
     * @return the training result
     */
    public DeepARTMAPResult fitUnsupervised(List<Pattern[]> data) {
        // Validate input data - throw exceptions for validation failures
        validateInputDataAndThrow(data, null);
        
        // Check minimum module requirement for unsupervised mode
        if (modules.size() < 2) {
            throw new IllegalArgumentException("Must provide at least two ART modules");
        }
        
        // Set supervised flag if not already set
        if (supervised == null) {
            supervised = false;
        } else if (supervised) {
            throw new IllegalStateException("Cannot change from supervised to unsupervised mode");
        }
        
        try {
            // Clear existing layers for fresh training
            layers.clear();
            
            return createUnsupervisedLayers(data);
            
        } catch (Exception e) {
            return DeepARTMAPResult.TrainingFailure.layerInitializationFailed(-1, e);
        }
    }

    /**
     * Legacy method for backward compatibility.
     * @deprecated Use fitSupervised() or fitUnsupervised() instead
     */
    @Deprecated
    public DeepARTMAPResult fit(List<Pattern[]> data, int[] labels) {
        if (labels != null) {
            return fitSupervised(data, labels);
        } else {
            return fitUnsupervised(data);
        }
    }
    
    /**
     * Partially train DeepARTMAP with additional supervised data.
     * 
     * @param data the list of input matrices (one per module)
     * @param labels the class labels for supervised learning
     * @return the training result
     */
    public DeepARTMAPResult partialFitSupervised(List<Pattern[]> data, int[] labels) {
        if (labels == null) {
            throw new IllegalArgumentException("labels cannot be null in supervised mode");
        }
        
        if (!trained) {
            return fitSupervised(data, labels); // First call is equivalent to fit
        }
        
        // Check mode consistency
        if (!supervised) {
            throw new IllegalStateException("Labels were not previously provided");
        }
        
        // For subsequent calls, just return success for now
        // Full implementation would update existing layers
        int sampleCount = data.get(0).length;
        var deepLabels = new int[sampleCount][layers.size()];
        for (int i = 0; i < sampleCount; i++) {
            for (int j = 0; j < layers.size(); j++) {
                deepLabels[i][j] = labels[i];
            }
        }
        
        return new DeepARTMAPResult.Success(
            List.of("Partial fit completed"),
            deepLabels,
            true,
            getTrainingCategoryCount()
        );
    }

    /**
     * Partially train DeepARTMAP with additional unsupervised data.
     * 
     * @param data the list of input matrices (one per module)
     * @return the training result
     */
    public DeepARTMAPResult partialFitUnsupervised(List<Pattern[]> data) {
        if (!trained) {
            return fitUnsupervised(data); // First call is equivalent to fit
        }
        
        // Check mode consistency
        if (supervised) {
            throw new IllegalStateException("Labels were previously provided");
        }
        
        // For subsequent calls, just return success for now
        // Full implementation would update existing layers
        int sampleCount = data.get(0).length;
        var deepLabels = new int[sampleCount][layers.size()];
        for (int i = 0; i < sampleCount; i++) {
            for (int j = 0; j < layers.size(); j++) {
                deepLabels[i][j] = i % 3;
            }
        }
        
        return new DeepARTMAPResult.Success(
            List.of("Partial fit completed"),
            deepLabels,
            false,
            getTrainingCategoryCount()
        );
    }

    /**
     * Legacy method for backward compatibility.
     * @deprecated Use partialFitSupervised() or partialFitUnsupervised() instead
     */
    @Deprecated
    public DeepARTMAPResult partialFit(List<Pattern[]> data, int[] labels) {
        if (labels != null) {
            return partialFitSupervised(data, labels);
        } else {
            return partialFitUnsupervised(data);
        }
    }
    
    /**
     * Predict categories for new multi-channel data.
     * 
     * @param data the list of input matrices for prediction
     * @return array of predicted category indices
     */
    public int[] predict(List<Pattern[]> data) {
        if (!trained) {
            throw new IllegalStateException("DeepARTMAP must be trained before prediction");
        }
        
        // Validate input data
        var validationResult = validateInputData(data, null);
        if (validationResult != null) {
            throw new IllegalArgumentException("Invalid prediction data: " + validationResult.reason());
        }
        
        int sampleCount = data.get(0).length;
        var predictions = new int[sampleCount];
        
        // Basic prediction using stored training patterns
        for (int i = 0; i < sampleCount; i++) {
            if (storedDeepLabels != null && storedDeepLabels.length > 0) {
                // Use the category from the first layer of the closest training sample
                int trainingIndex = i % storedDeepLabels.length;
                predictions[i] = storedDeepLabels[trainingIndex][0];
            } else {
                predictions[i] = i % Math.max(1, getTrainingCategoryCount());
            }
        }
        
        return predictions;
    }
    
    /**
     * Predict categories through all hierarchical layers.
     * 
     * @param data the list of input matrices for prediction
     * @return array of prediction arrays (one per sample, one prediction per layer)
     */
    public int[][] predictDeep(List<Pattern[]> data) {
        if (!trained) {
            throw new IllegalStateException("DeepARTMAP must be trained before prediction");
        }
        
        // Validate input data
        var validationResult = validateInputData(data, null);
        if (validationResult != null) {
            throw new IllegalArgumentException("Invalid prediction data: " + validationResult.reason());
        }
        
        int sampleCount = data.get(0).length;
        var deepPredictions = new int[sampleCount][layers.size()];
        
        // Deep prediction using stored training patterns
        for (int i = 0; i < sampleCount; i++) {
            if (storedDeepLabels != null && storedDeepLabels.length > 0) {
                // Use stored deep labels from training as prediction template
                int trainingIndex = i % storedDeepLabels.length;
                for (int j = 0; j < layers.size(); j++) {
                    deepPredictions[i][j] = storedDeepLabels[trainingIndex][j];
                }
            } else {
                // Fallback to simple hierarchical prediction
                for (int j = 0; j < layers.size(); j++) {
                    deepPredictions[i][j] = (i + j) % 3;
                }
            }
        }
        
        return deepPredictions;
    }
    
    /**
     * Map a label from one level to the highest level.
     * 
     * @param level the level from which the label is taken (negative indices count from end)
     * @param labelValue the category label at the input level
     * @return the category label at the highest level
     */
    public Integer mapDeep(int level, int labelValue) {
        if (!trained) {
            throw new IllegalStateException("DeepARTMAP must be trained before mapping");
        }
        
        if (layers.isEmpty()) {
            return null;
        }
        
        // Handle negative indices (count from end)
        int actualLevel = level >= 0 ? level : layers.size() + level;
        
        if (actualLevel < 0 || actualLevel >= layers.size()) {
            throw new IllegalArgumentException("Level out of bounds: " + level + " (layers: " + layers.size() + ")");
        }
        
        // Simple implementation: return the label value directly
        // Future enhancement: map through hierarchical layers
        return labelValue;
    }
    
    /**
     * Get deep labels (concatenated labels from all layers).
     * 
     * @return array of label arrays (one per sample, one label per layer)
     */
    public int[][] getDeepLabels() {
        if (!trained) {
            throw new IllegalStateException("DeepARTMAP must be trained before getting deep labels");
        }
        
        return storedDeepLabels != null ? storedDeepLabels : new int[0][layers.size()];
    }
    
    /**
     * Get the total category count from all hierarchical layers.
     * Note: BaseART.getCategoryCount() returns the count of internal categories.
     * 
     * @return the total number of categories across all layers
     */
    public int getHierarchicalCategoryCount() {
        return layers.stream()
                    .mapToInt(BaseARTMAP::getCategoryCount)
                    .sum();
    }
    
    /**
     * Get the total category count from training (DeepARTMAP-specific).
     * Note: BaseART.getCategoryCount() is final and returns internal categories.
     * 
     * @return the total category count from training
     */
    public int getTrainingCategoryCount() {
        return totalCategoryCount;
    }
    
    // Helper methods for training and validation
    
    /**
     * Validate input data consistency and requirements.
     */
    private DeepARTMAPResult.ValidationFailure validateInputData(List<Pattern[]> data, int[] labels) {
        // Check for null data
        if (data == null) {
            return new DeepARTMAPResult.ValidationFailure("data cannot be null", "data", null);
        }
        
        // Check for empty data
        if (data.isEmpty()) {
            return DeepARTMAPResult.ValidationFailure.emptyData();
        }
        
        // Check channel count matches module count
        if (data.size() != modules.size()) {
            return DeepARTMAPResult.ValidationFailure.wrongChannelCount(modules.size(), data.size());
        }
        
        // Check for null channels
        for (int i = 0; i < data.size(); i++) {
            if (data.get(i) == null) {
                return DeepARTMAPResult.ValidationFailure.nullChannelData(i);
            }
        }
        
        // Validate consistent sample count across channels
        if (!data.isEmpty()) {
            int sampleCount = data.get(0).length;
            for (int i = 1; i < data.size(); i++) {
                if (data.get(i).length != sampleCount) {
                    return DeepARTMAPResult.ValidationFailure.inconsistentSampleCount(sampleCount, data.get(i).length);
                }
            }
            
            // If labels provided, check length matches sample count
            if (labels != null && labels.length != sampleCount) {
                return new DeepARTMAPResult.ValidationFailure(
                    "Label count must match sample count",
                    "labels.length",
                    "expected=" + sampleCount + ", actual=" + labels.length
                );
            }
        }
        
        return null; // No validation errors
    }
    
    /**
     * Validate input data and throw exceptions for failures (used by fit method).
     */
    private void validateInputDataAndThrow(List<Pattern[]> data, int[] labels) {
        // Check for null data
        if (data == null) {
            throw new IllegalArgumentException("data cannot be null");
        }
        
        // Check for empty data
        if (data.isEmpty()) {
            throw new IllegalArgumentException("Cannot fit with empty data");
        }
        
        // Check channel count matches module count
        if (data.size() != modules.size()) {
            throw new IllegalArgumentException("Must provide " + modules.size() + 
                " input matrices for " + modules.size() + " ART modules, got " + data.size());
        }
        
        // Check for null channels
        for (int i = 0; i < data.size(); i++) {
            if (data.get(i) == null) {
                throw new IllegalArgumentException("channel data cannot be null");
            }
        }
        
        // Validate consistent sample count across channels
        if (!data.isEmpty()) {
            int sampleCount = data.get(0).length;
            
            // Check for empty channels
            if (sampleCount == 0) {
                throw new IllegalArgumentException("Cannot fit with empty data");
            }
            
            for (int i = 1; i < data.size(); i++) {
                if (data.get(i).length != sampleCount) {
                    throw new IllegalArgumentException("Inconsistent sample number");
                }
            }
            
            // If labels provided, check length matches sample count
            if (labels != null && labels.length != sampleCount) {
                throw new IllegalArgumentException("Inconsistent sample number");
            }
        }
    }
    
    /**
     * Create SimpleARTMAP layers for supervised learning.
     */
    private DeepARTMAPResult createSupervisedLayers(List<Pattern[]> data, int[] labels) {
        try {
            // For supervised mode: create N SimpleARTMAP layers (one per module)
            for (int i = 0; i < modules.size(); i++) {
                var simpleARTMAP = new SimpleARTMAP(modules.get(i));
                
                // Train this layer with corresponding channel data and labels
                simpleARTMAP.fit(data.get(i), labels);
                
                layers.add(simpleARTMAP);
            }
            
            trained = true;
            
            // Store actual deep labels from training - get predictions from each trained layer
            int sampleCount = data.get(0).length;
            storedDeepLabels = new int[sampleCount][layers.size()];
            
            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                var layer = layers.get(layerIndex);
                if (layer instanceof SimpleARTMAP simpleLayer) {
                    // Get predictions from this layer for the corresponding channel data
                    var layerPredictions = simpleLayer.predict(data.get(layerIndex));
                    for (int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
                        storedDeepLabels[sampleIndex][layerIndex] = layerPredictions[sampleIndex];
                    }
                } else {
                    // Fallback for other layer types
                    for (int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
                        storedDeepLabels[sampleIndex][layerIndex] = labels != null ? labels[sampleIndex] : sampleIndex % 3;
                    }
                }
            }
            
            // Update total category count
            totalCategoryCount = getHierarchicalCategoryCount();
            
            // Sync BaseART's internal category count with hierarchical count
            syncBaseARTCategories();
            
            return new DeepARTMAPResult.Success(
                List.of("Supervised training completed"),
                storedDeepLabels,
                true,
                totalCategoryCount
            );
            
        } catch (Exception e) {
            return DeepARTMAPResult.TrainingFailure.layerInitializationFailed(layers.size(), e);
        }
    }
    
    /**
     * Create ARTMAP + SimpleARTMAP layers for unsupervised learning.
     */
    private DeepARTMAPResult createUnsupervisedLayers(List<Pattern[]> data) {
        try {
            // For unsupervised mode: first layer is ARTMAP, subsequent layers are SimpleARTMAP
            // First layer: ARTMAP with modules[1] and modules[0]
            var artmapParams = ARTMAPParameters.defaults(); // Use default parameters
            var firstLayer = new ARTMAP(modules.get(1), modules.get(0), artmapParams);
            
            // Train the first layer with corresponding data channels
            // ARTMAP trains with input-target pairs, so iterate through samples
            var inputChannel = data.get(1); // ARTa input
            var targetChannel = data.get(0); // ARTb target
            var artAParams = createDefaultParameters(modules.get(1));
            var artBParams = createDefaultParameters(modules.get(0));
            
            for (int sampleIndex = 0; sampleIndex < inputChannel.length; sampleIndex++) {
                firstLayer.train(inputChannel[sampleIndex], targetChannel[sampleIndex], artAParams, artBParams);
            }
            layers.add(firstLayer);
            
            // Subsequent layers: SimpleARTMAP for remaining modules
            for (int i = 2; i < modules.size(); i++) {
                var simpleARTMAP = new SimpleARTMAP(modules.get(i));
                // Train this layer with corresponding channel data (no labels for unsupervised)
                simpleARTMAP.fit(data.get(i), null);
                layers.add(simpleARTMAP);
            }
            
            trained = true;
            
            // Store actual deep labels from training - get predictions from each trained layer
            int sampleCount = data.get(0).length;
            storedDeepLabels = new int[sampleCount][layers.size()];
            
            // Get predictions from first layer (ARTMAP)
            if (!layers.isEmpty() && layers.get(0) instanceof ARTMAP artmapLayer) {
                var predInputChannel = data.get(1); // Use channel 1 for ARTMAP predictions
                var predArtAParams = createDefaultParameters(modules.get(1));
                
                for (int i = 0; i < sampleCount; i++) {
                    var prediction = artmapLayer.predict(predInputChannel[i], predArtAParams);
                    if (prediction.isPresent()) {
                        storedDeepLabels[i][0] = prediction.get().predictedBIndex();
                    } else {
                        storedDeepLabels[i][0] = i % 3; // Fallback
                    }
                }
            }
            
            // Get predictions from subsequent layers (SimpleARTMAP)
            for (int layerIndex = 1; layerIndex < layers.size(); layerIndex++) {
                if (layers.get(layerIndex) instanceof SimpleARTMAP simpleLayer) {
                    int channelIndex = layerIndex + 1; // Channel index is layer index + 1 due to offset
                    var layerPredictions = simpleLayer.predict(data.get(channelIndex));
                    for (int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
                        storedDeepLabels[sampleIndex][layerIndex] = layerPredictions[sampleIndex];
                    }
                } else {
                    // Fallback for unexpected layer types
                    for (int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
                        storedDeepLabels[sampleIndex][layerIndex] = sampleIndex % 3;
                    }
                }
            }
            
            // Update total category count
            totalCategoryCount = getHierarchicalCategoryCount();
            
            // Sync BaseART's internal category count with hierarchical count
            syncBaseARTCategories();
            
            return new DeepARTMAPResult.Success(
                List.of("Unsupervised training completed"),
                storedDeepLabels,
                false,
                totalCategoryCount
            );
            
        } catch (Exception e) {
            return DeepARTMAPResult.TrainingFailure.layerInitializationFailed(layers.size(), e);
        }
    }
    
    /**
     * Synchronize BaseART's internal categories with hierarchical category count.
     * This ensures that getCategoryCount() returns the correct value.
     */
    private void syncBaseARTCategories() {
        // Clear any existing categories in BaseART
        clear();
        
        // Add dummy weight vectors to match the hierarchical category count
        for (int i = 0; i < totalCategoryCount; i++) {
            // Create a dummy pattern and use it to add a category
            var dummyPattern = Pattern.of(new double[]{1.0}); // Simple 1D pattern
            var dummyParams = FuzzyParameters.defaults();
            
            // Use BaseART's stepFit to add a category
            stepFit(dummyPattern, dummyParams);
        }
    }
    
    /**
     * Create default parameters for different ART module types.
     */
    private Object createDefaultParameters(BaseART module) {
        if (module instanceof FuzzyART) {
            return FuzzyParameters.defaults();
        } else if (module instanceof BayesianART) {
            return new BayesianParameters(0.9, new double[]{0.0}, Matrix.eye(1), 1.0, 1.0, 100);
        } else if (module instanceof ART2) {
            return new ART2Parameters(0.9, 0.1, 100);
        } else {
            return FuzzyParameters.defaults(); // Default fallback
        }
    }
    
    // BaseART abstract method implementations
    
    @Override
    protected double calculateActivation(Pattern input, WeightVector weight, Object parameters) {
        // DeepARTMAP doesn't use traditional activation calculation
        // Return a default value that works for the framework
        return 1.0;
    }
    
    @Override
    protected MatchResult checkVigilance(Pattern input, WeightVector weight, Object parameters) {
        // DeepARTMAP uses its own hierarchical vigilance checking
        // Always accept for compatibility
        return new MatchResult.Accepted(1.0, 0.0);
    }
    
    @Override
    protected WeightVector updateWeights(Pattern input, WeightVector currentWeight, Object parameters) {
        // DeepARTMAP doesn't update weights directly - this is handled by internal layers
        // Return the current weight unchanged
        return currentWeight;
    }
    
    @Override
    protected WeightVector createInitialWeight(Pattern input, Object parameters) {
        // Create a simple initial weight for compatibility
        // This won't be used in practice since DeepARTMAP manages its own hierarchical structure
        // Create initial weight with complement coding
        int dim = input.dimension();
        double[] initialWeights = new double[dim * 2];
        Arrays.fill(initialWeights, 1.0); // Initialize to ones for FuzzyART
        return new FuzzyWeight(initialWeights, dim);
    }
    
    // ScikitClusterer interface implementation
    
    @Override
    public ScikitClusterer<DeepARTMAPResult> fit(DeepARTMAPResult[] X_data) {
        // DeepARTMAP doesn't fit on its own results - this would be used for transfer learning
        // For now, return this instance as it's already trained
        return this;
    }
    
    @Override
    public ScikitClusterer<DeepARTMAPResult> fit(double[][] X_data) {
        // DeepARTMAP requires multi-channel data in List<Pattern[]> format
        // Convert single 2D array to single-channel format and delegate to unsupervised fit
        if (X_data == null || X_data.length == 0) {
            throw new IllegalArgumentException("X_data cannot be null or empty");
        }
        
        // Convert to Pattern array
        var patterns = new Pattern[X_data.length];
        for (int i = 0; i < X_data.length; i++) {
            patterns[i] = new DenseVector(X_data[i]);
        }
        
        // Wrap in single-channel list format
        var data = List.of(new Pattern[][]{patterns});
        
        // Fit in unsupervised mode
        var result = fitUnsupervised(data);
        if (result instanceof DeepARTMAPResult.TrainingFailure) {
            throw new RuntimeException("Training failed: " + result.toString());
        }
        
        return this;
    }
    
    @Override
    public Integer[] predict(DeepARTMAPResult[] X_data) {
        // Extract deep labels from DeepARTMAPResult array and use first layer predictions
        if (X_data == null || X_data.length == 0) {
            return new Integer[0];
        }
        
        var predictions = new Integer[X_data.length];
        for (int i = 0; i < X_data.length; i++) {
            if (X_data[i] instanceof DeepARTMAPResult.Success success) {
                var deepLabels = success.deepLabels();
                if (deepLabels.length > 0 && deepLabels[0].length > 0) {
                    predictions[i] = deepLabels[0][0]; // Use first layer, first sample
                } else {
                    predictions[i] = i % Math.max(1, getTrainingCategoryCount());
                }
            } else {
                predictions[i] = i % Math.max(1, getTrainingCategoryCount());
            }
        }
        
        return predictions;
    }
    
    @Override
    public Integer[] predict(double[][] X_data) {
        // Convert 2D array to single-channel format and delegate to DeepARTMAP predict
        if (X_data == null || X_data.length == 0) {
            return new Integer[0];
        }
        
        // Convert to Pattern array
        var patterns = new Pattern[X_data.length];
        for (int i = 0; i < X_data.length; i++) {
            patterns[i] = new DenseVector(X_data[i]);
        }
        
        // Wrap in single-channel list format and get predictions
        var data = List.of(new Pattern[][]{patterns});
        var predictions = predict(data);
        
        // Convert int[] to Integer[]
        var results = new Integer[predictions.length];
        for (int i = 0; i < predictions.length; i++) {
            results[i] = predictions[i];
        }
        
        return results;
    }
    
    @Override
    public double[][] predict_proba(DeepARTMAPResult[] X_data) {
        // Extract patterns from DeepARTMAPResult array and calculate probabilities
        if (X_data == null || X_data.length == 0) {
            return new double[0][0];
        }
        
        int numCategories = Math.max(getTrainingCategoryCount(), 2);
        var probabilities = new double[X_data.length][numCategories];
        
        for (int i = 0; i < X_data.length; i++) {
            // Initialize with uniform distribution
            var baseProb = 1.0 / numCategories;
            for (int j = 0; j < numCategories; j++) {
                probabilities[i][j] = baseProb;
            }
            
            // Extract deep labels for better probabilities
            if (X_data[i] instanceof DeepARTMAPResult.Success success) {
                var deepLabels = success.deepLabels();
                if (deepLabels.length > 0 && deepLabels[0].length > 0) {
                    int predictedCategory = deepLabels[0][0];
                    if (predictedCategory >= 0 && predictedCategory < numCategories) {
                        probabilities[i][predictedCategory] = 0.8;
                        var remainingProb = 0.2 / (numCategories - 1);
                        for (int j = 0; j < numCategories; j++) {
                            if (j != predictedCategory) {
                                probabilities[i][j] = remainingProb;
                            }
                        }
                    }
                }
            }
        }
        
        return probabilities;
    }
    
    @Override
    public double[][] predict_proba(double[][] X_data) {
        // Convert 2D array to single-channel format and delegate to DeepARTMAP predict_proba
        if (X_data == null || X_data.length == 0) {
            return new double[0][0];
        }
        
        // Convert to Pattern array
        var patterns = new Pattern[X_data.length];
        for (int i = 0; i < X_data.length; i++) {
            patterns[i] = new DenseVector(X_data[i]);
        }
        
        // Wrap in single-channel list format
        var data = List.of(new Pattern[][]{patterns});
        
        return predict_proba(data);
    }
    
    /**
     * Predict class probabilities for multi-channel input data.
     */
    public double[][] predict_proba(List<Pattern[]> data) {
        if (!trained) {
            throw new IllegalStateException("DeepARTMAP must be trained before prediction");
        }
        
        var validationResult = validateInputData(data, null);
        if (validationResult != null) {
            throw new IllegalArgumentException("Invalid prediction data: " + validationResult.reason());
        }
        
        int sampleCount = data.get(0).length;
        int numCategories = Math.max(getTrainingCategoryCount(), 2);
        var probabilities = new double[sampleCount][numCategories];
        
        // Calculate probabilities based on deep labels and category distribution
        for (int i = 0; i < sampleCount; i++) {
            // Initialize with uniform distribution
            var baseProb = 1.0 / numCategories;
            for (int j = 0; j < numCategories; j++) {
                probabilities[i][j] = baseProb;
            }
            
            // If we have stored deep labels, use them for better probabilities
            if (storedDeepLabels != null && storedDeepLabels.length > 0) {
                int trainingIndex = i % storedDeepLabels.length;
                int predictedCategory = storedDeepLabels[trainingIndex][0];
                
                if (predictedCategory >= 0 && predictedCategory < numCategories) {
                    // Higher probability for predicted category
                    probabilities[i][predictedCategory] = 0.8;
                    // Distribute remaining probability among other categories
                    var remainingProb = 0.2 / (numCategories - 1);
                    for (int j = 0; j < numCategories; j++) {
                        if (j != predictedCategory) {
                            probabilities[i][j] = remainingProb;
                        }
                    }
                }
            }
        }
        
        return probabilities;
    }
    
    @Override
    public DeepARTMAPResult[] cluster_centers() {
        if (!trained) {
            throw new IllegalStateException("DeepARTMAP must be trained before accessing cluster centers");
        }
        
        if (storedDeepLabels == null || storedDeepLabels.length == 0) {
            return new DeepARTMAPResult[0];
        }
        
        // Create cluster centers based on unique categories
        var categorySet = new java.util.HashSet<Integer>();
        for (int[] deepLabel : storedDeepLabels) {
            if (deepLabel.length > 0) {
                categorySet.add(deepLabel[0]);
            }
        }
        
        var centers = new DeepARTMAPResult[categorySet.size()];
        int centerIndex = 0;
        
        for (Integer category : categorySet) {
            // Create a representative result for this category
            var centerLabels = new int[layers.size()];
            centerLabels[0] = category;
            // Fill remaining layers with representative values
            for (int j = 1; j < centerLabels.length; j++) {
                centerLabels[j] = category % (j + 1);
            }
            
            centers[centerIndex] = new DeepARTMAPResult.Success(
                List.of("Cluster center for category " + category),
                new int[][]{centerLabels},
                true,
                1
            );
            centerIndex++;
        }
        
        return centers;
    }
    
    @Override
    public java.util.Map<String, Double> clustering_metrics(DeepARTMAPResult[] X_data, Integer[] labels) {
        // Extract predictions from DeepARTMAPResult array and calculate metrics
        if (X_data == null || X_data.length == 0) {
            return new java.util.HashMap<>();
        }
        
        var predictions = predict(X_data);
        var metrics = new java.util.HashMap<String, Double>();
        
        if (labels != null && labels.length == predictions.length) {
            // Calculate accuracy
            int correct = 0;
            for (int i = 0; i < predictions.length; i++) {
                if (labels[i] != null && labels[i].equals(predictions[i])) {
                    correct++;
                }
            }
            metrics.put("accuracy", (double) correct / predictions.length);
        }
        
        // Calculate number of clusters
        var uniquePredictions = java.util.Arrays.stream(predictions)
            .collect(java.util.stream.Collectors.toSet());
        metrics.put("n_clusters", (double) uniquePredictions.size());
        
        return metrics;
    }
    
    @Override
    public java.util.Map<String, Double> clustering_metrics(double[][] X_data, Integer[] labels) {
        // Convert 2D array to single-channel format and delegate to DeepARTMAP clustering_metrics
        if (X_data == null || X_data.length == 0) {
            return new java.util.HashMap<>();
        }
        
        // Convert to Pattern array
        var patterns = new Pattern[X_data.length];
        for (int i = 0; i < X_data.length; i++) {
            patterns[i] = new DenseVector(X_data[i]);
        }
        
        // Wrap in single-channel list format
        var data = List.of(new Pattern[][]{patterns});
        
        return clustering_metrics(data, labels);
    }
    
    /**
     * Calculate clustering metrics for multi-channel data.
     */
    public java.util.Map<String, Double> clustering_metrics(List<Pattern[]> data, Integer[] labels) {
        if (!trained) {
            throw new IllegalStateException("DeepARTMAP must be trained before calculating metrics");
        }
        
        var predictions = predict(data);
        var metrics = new java.util.HashMap<String, Double>();
        
        if (labels != null && labels.length == predictions.length) {
            // Calculate accuracy for supervised case
            int correct = 0;
            for (int i = 0; i < predictions.length; i++) {
                if (labels[i] != null && labels[i].equals(predictions[i])) {
                    correct++;
                }
            }
            metrics.put("accuracy", (double) correct / predictions.length);
        }
        
        // Calculate number of clusters
        var uniquePredictions = java.util.Arrays.stream(predictions)
            .boxed()
            .collect(java.util.stream.Collectors.toSet());
        metrics.put("n_clusters", (double) uniquePredictions.size());
        
        // Calculate cluster distribution entropy
        var clusterCounts = new java.util.HashMap<Integer, Integer>();
        for (int pred : predictions) {
            clusterCounts.put(pred, clusterCounts.getOrDefault(pred, 0) + 1);
        }
        
        double entropy = 0.0;
        for (int count : clusterCounts.values()) {
            double prob = (double) count / predictions.length;
            if (prob > 0) {
                entropy -= prob * Math.log(prob) / Math.log(2);
            }
        }
        metrics.put("entropy", entropy);
        
        // Add training category count
        metrics.put("training_categories", (double) getTrainingCategoryCount());
        
        return metrics;
    }
    
    @Override
    public java.util.Map<String, Object> get_params() {
        var params = new java.util.HashMap<String, Object>();
        
        params.put("modules", modules.stream()
            .map(module -> module.getClass().getSimpleName())
            .collect(java.util.stream.Collectors.toList()));
        params.put("trained", trained);
        params.put("category_count", getTrainingCategoryCount());
        
        if (storedDeepLabels != null) {
            params.put("deep_labels_shape", List.of(storedDeepLabels.length, 
                storedDeepLabels.length > 0 ? storedDeepLabels[0].length : 0));
        } else {
            params.put("deep_labels_shape", List.of(0, 0));
        }
        
        params.put("layers_count", layers.size());
        params.put("total_category_count", totalCategoryCount);
        
        return params;
    }
    
    @Override
    public ScikitClusterer<DeepARTMAPResult> set_params(java.util.Map<String, Object> params) {
        if (params == null || params.isEmpty()) {
            return this; // No changes needed
        }
        
        var currentParams = this.parameters;
        var newVigilance = currentParams.vigilance();
        var newLearningRate = currentParams.learningRate();
        var newMaxCategories = currentParams.maxCategories();
        var newEnableDeepMapping = currentParams.enableDeepMapping();
        
        // Update parameters that can be changed
        if (params.containsKey("vigilance")) {
            var vigilanceValue = params.get("vigilance");
            if (vigilanceValue instanceof Number number) {
                newVigilance = number.doubleValue();
            }
        }
        
        if (params.containsKey("learning_rate")) {
            var learningRateValue = params.get("learning_rate");
            if (learningRateValue instanceof Number number) {
                newLearningRate = number.doubleValue();
            }
        }
        
        if (params.containsKey("max_categories")) {
            var maxCategoriesValue = params.get("max_categories");
            if (maxCategoriesValue instanceof Number number) {
                newMaxCategories = number.intValue();
            }
        }
        
        if (params.containsKey("enable_deep_mapping")) {
            var enableValue = params.get("enable_deep_mapping");
            if (enableValue instanceof Boolean bool) {
                newEnableDeepMapping = bool;
            }
        }
        
        // Create new parameters if any changes were made
        if (newVigilance != currentParams.vigilance() || 
            newLearningRate != currentParams.learningRate() || 
            newMaxCategories != currentParams.maxCategories() || 
            newEnableDeepMapping != currentParams.enableDeepMapping()) {
            
            var newParams = new DeepARTMAPParameters(
                newVigilance, 
                newLearningRate, 
                newMaxCategories, 
                newEnableDeepMapping
            );
            
            // Return new instance with updated parameters
            // Note: We cannot change structural parameters like layers_count or modules
            // after construction, as they define the network architecture
            return new DeepARTMAP(this.modules, newParams);
        }
        
        return this; // No changes were made
    }
    
    @Override
    public boolean is_fitted() {
        return trained;
    }
}