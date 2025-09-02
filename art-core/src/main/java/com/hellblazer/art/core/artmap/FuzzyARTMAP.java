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
package com.hellblazer.art.core.artmap;

import com.hellblazer.art.core.BaseARTMAP;
import com.hellblazer.art.core.MatchTrackingMode;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.algorithms.FuzzyART;
import com.hellblazer.art.core.parameters.FuzzyARTMAPParameters;
import com.hellblazer.art.core.parameters.FuzzyParameters;
import com.hellblazer.art.core.results.ActivationResult;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * FuzzyARTMAP implementation for supervised classification.
 * 
 * FuzzyARTMAP is an optimized supervised learning variant that combines:
 * - FuzzyART clustering with complement coding
 * - Map field for cluster-to-label associations
 * - Match tracking for handling label conflicts
 * 
 * This implementation provides compatibility with the Python AdaptiveResonanceLib
 * FuzzyARTMAP while leveraging Java's performance advantages.
 * 
 * Key features:
 * - Complement coding for input preprocessing
 * - Fast convergence with single-pass learning
 * - Incremental learning support (partial_fit)
 * - Dual prediction modes (B-labels only or A+B labels)
 * 
 * @author Hal Hildebrand
 */
public class FuzzyARTMAP implements BaseARTMAP {
    
    private final FuzzyARTMAPParameters parameters;
    private final FuzzyART fuzzyART;
    private final Map<Integer, Integer> mapField;  // category -> label mapping
    private boolean trained;
    private int[] trainingLabels;  // Store training labels for consistency
    
    /**
     * Create a new FuzzyARTMAP with specified parameters.
     * 
     * @param parameters the FuzzyARTMAP parameters
     */
    public FuzzyARTMAP(FuzzyARTMAPParameters parameters) {
        this.parameters = Objects.requireNonNull(parameters, "parameters cannot be null");
        this.fuzzyART = new FuzzyART();
        this.mapField = new HashMap<>();
        this.trained = false;
    }
    
    /**
     * Create a new FuzzyARTMAP with default parameters.
     */
    public FuzzyARTMAP() {
        this(FuzzyARTMAPParameters.defaultParameters());
    }
    
    /**
     * Create a new FuzzyARTMAP with FuzzyParameters and match tracking settings.
     * Constructor for compatibility with tests.
     * 
     * @param fuzzyParams the FuzzyART parameters
     * @param matchTrackingMode the match tracking mode (unused, always MT_PLUS)
     * @param epsilon the epsilon value for match tracking
     */
    public FuzzyARTMAP(FuzzyParameters fuzzyParams, MatchTrackingMode matchTrackingMode, double epsilon) {
        this(new FuzzyARTMAPParameters(fuzzyParams.vigilance(), fuzzyParams.alpha(), fuzzyParams.beta(), epsilon));
    }
    
    /**
     * Result of training on a single sample.
     */
    public record TrainResult(
        int category,
        int label,
        boolean matchTrackingOccurred,
        double adjustedVigilance
    ) {}
    
    /**
     * Result of dual prediction (A-side and B-side labels).
     */
    public record PredictABResult(
        int[] aLabels,
        int[] bLabels
    ) {}
    
    /**
     * Train on a single pattern with its label.
     * 
     * @param input the input pattern (should be complement coded)
     * @param label the class label
     * @return the training result
     */
    public TrainResult trainSingle(Pattern input, int label) {
        Objects.requireNonNull(input, "input cannot be null");
        
        // Use MutableFuzzyParameters for match tracking to work
        var mutableParams = new com.hellblazer.art.core.parameters.MutableFuzzyParameters(parameters.toFuzzyParameters());
        final boolean[] matchTrackingOccurred = {false};
        final int[] conflictCount = {0};
        double adjustedVigilance = parameters.rho();
        
        // Use match reset function to handle label conflicts
        com.hellblazer.art.core.MatchResetFunction matchReset = (Pattern in, com.hellblazer.art.core.WeightVector weight, 
                         int categoryIndex, Object params, java.util.Optional<Object> cache) -> {
            if (mapField.containsKey(categoryIndex)) {
                int existingLabel = mapField.get(categoryIndex);
                if (existingLabel != label) {
                    // Conflict - reject this category, will trigger match tracking
                    conflictCount[0]++;
                    matchTrackingOccurred[0] = true;
                    // Conflict - will trigger match tracking
                    return false;
                }
            }
            return true;
        };
        
        // Train with match tracking using mutable parameters
        var result = fuzzyART.stepFit(
            input,
            mutableParams,
            matchReset,
            MatchTrackingMode.MT_PLUS,
            parameters.epsilon()
        );
        
        if (result instanceof ActivationResult.Success success) {
            int category = success.categoryIndex();
            
            // Update map field
            if (!mapField.containsKey(category)) {
                mapField.put(category, label);
            }
            
            // Get the adjusted vigilance from mutable parameters after match tracking
            adjustedVigilance = mutableParams.vigilance();
            
            trained = true;
            return new TrainResult(category, label, matchTrackingOccurred[0], adjustedVigilance);
        } else {
            throw new IllegalStateException("FuzzyART failed to create category");
        }
    }
    
    /**
     * Fit the model to training data.
     * 
     * @param data array of input patterns (should be complement coded)
     * @param labels array of class labels
     */
    public void fit(Pattern[] data, int[] labels) {
        Objects.requireNonNull(data, "data cannot be null");
        Objects.requireNonNull(labels, "labels cannot be null");
        
        if (data.length == 0) {
            throw new IllegalArgumentException("Cannot fit with empty data");
        }
        
        if (data.length != labels.length) {
            throw new IllegalArgumentException(
                "Data and labels must have same length: " + 
                data.length + " != " + labels.length
            );
        }
        
        // Clear existing state
        clear();
        
        // Store training labels
        this.trainingLabels = labels.clone();
        
        // Train on each sample
        for (int i = 0; i < data.length; i++) {
            trainSingle(data[i], labels[i]);
        }
        
        trained = true;
    }
    
    /**
     * Incrementally fit the model with new data.
     * 
     * @param data array of new input patterns
     * @param labels array of class labels for new data
     */
    public void partialFit(Pattern[] data, int[] labels) {
        Objects.requireNonNull(data, "data cannot be null");
        Objects.requireNonNull(labels, "labels cannot be null");
        
        if (data.length != labels.length) {
            throw new IllegalArgumentException(
                "Data and labels must have same length: " + 
                data.length + " != " + labels.length
            );
        }
        
        // If not yet trained, this is equivalent to fit
        if (!trained) {
            fit(data, labels);
            return;
        }
        
        // Extend training labels array
        var oldLength = trainingLabels != null ? trainingLabels.length : 0;
        var newLabels = new int[oldLength + labels.length];
        if (trainingLabels != null) {
            System.arraycopy(trainingLabels, 0, newLabels, 0, oldLength);
        }
        System.arraycopy(labels, 0, newLabels, oldLength, labels.length);
        trainingLabels = newLabels;
        
        // Train on new samples
        for (int i = 0; i < data.length; i++) {
            trainSingle(data[i], labels[i]);
        }
    }
    
    /**
     * Predict class labels for input patterns.
     * 
     * @param data array of input patterns (should be complement coded)
     * @return array of predicted class labels
     */
    public int[] predict(Pattern[] data) {
        Objects.requireNonNull(data, "data cannot be null");
        
        if (!trained) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        
        var predictions = new int[data.length];
        var fuzzyParams = parameters.toFuzzyParameters();
        
        for (int i = 0; i < data.length; i++) {
            // Find best matching category (no learning)
            var result = fuzzyART.stepPredict(data[i], fuzzyParams);
            
            if (result instanceof ActivationResult.Success success) {
                int category = success.categoryIndex();
                
                // Look up label mapping
                if (mapField.containsKey(category)) {
                    predictions[i] = mapField.get(category);
                } else {
                    // Unknown category - should not happen after training
                    predictions[i] = -1;
                }
            } else {
                // No match found
                predictions[i] = -1;
            }
        }
        
        return predictions;
    }
    
    /**
     * Predict both A-side (category) and B-side (class) labels.
     * 
     * @param data array of input patterns
     * @return result containing both A and B labels
     */
    public PredictABResult predictAB(Pattern[] data) {
        Objects.requireNonNull(data, "data cannot be null");
        
        if (!trained) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        
        var aLabels = new int[data.length];
        var bLabels = new int[data.length];
        var fuzzyParams = parameters.toFuzzyParameters();
        
        for (int i = 0; i < data.length; i++) {
            // Find best matching category
            var result = fuzzyART.stepPredict(data[i], fuzzyParams);
            
            if (result instanceof ActivationResult.Success success) {
                int category = success.categoryIndex();
                aLabels[i] = category;
                
                // Look up B-side label
                if (mapField.containsKey(category)) {
                    bLabels[i] = mapField.get(category);
                } else {
                    bLabels[i] = -1;
                }
            } else {
                aLabels[i] = -1;
                bLabels[i] = -1;
            }
        }
        
        return new PredictABResult(aLabels, bLabels);
    }
    
    /**
     * Get the underlying FuzzyART module.
     * 
     * @return the FuzzyART instance
     */
    public FuzzyART getModuleA() {
        return fuzzyART;
    }
    
    /**
     * Get the underlying FuzzyART module (alias for getModuleA).
     * 
     * @return the FuzzyART instance
     */
    public FuzzyART getArtModule() {
        return fuzzyART;
    }
    
    /**
     * Get the map field (category to label mappings).
     * 
     * @return unmodifiable view of the map field
     */
    public Map<Integer, Integer> getMapField() {
        return Map.copyOf(mapField);
    }
    
    // BaseARTMAP interface implementation
    
    @Override
    public boolean isTrained() {
        return trained;
    }
    
    @Override
    public int getCategoryCount() {
        return fuzzyART.getCategoryCount();
    }
    
    @Override
    public void clear() {
        fuzzyART.clear();
        mapField.clear();
        trained = false;
        trainingLabels = null;
    }
}