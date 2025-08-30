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

import com.hellblazer.art.core.BaseART;
import com.hellblazer.art.core.BaseARTMAP;
import com.hellblazer.art.core.MatchResetFunction;
import com.hellblazer.art.core.MatchTrackingMethod;
import com.hellblazer.art.core.MatchTrackingMode;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.parameters.SimpleARTMAPParameters;
import com.hellblazer.art.core.results.ActivationResult;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * SimpleARTMAP implementation for supervised classification.
 * 
 * SimpleARTMAP is a simplified version of ARTMAP that uses:
 * - A single ART module (module_a) for clustering input patterns
 * - A map field that maintains many-to-one mappings from clusters to labels
 * - Match tracking to handle label conflicts by adjusting vigilance
 * 
 * When a pattern activates a cluster that maps to a different label,
 * match tracking increases the vigilance to force the search for a new cluster.
 * 
 * @author Hal Hildebrand
 */
public class SimpleARTMAP implements BaseARTMAP {
    
    private final BaseART moduleA;
    private final SimpleARTMAPParameters mapParameters;
    private final Map<Integer, Integer> mapField;  // cluster_id -> class_label
    private boolean matchTrackingOccurred;
    private double adjustedVigilance;
    private boolean trained = false;
    
    /**
     * Create a new SimpleARTMAP with the specified ART module and parameters.
     * 
     * @param moduleA the ART module to use for clustering
     * @param mapParameters the SimpleARTMAP parameters
     */
    public SimpleARTMAP(BaseART moduleA, SimpleARTMAPParameters mapParameters) {
        this.moduleA = Objects.requireNonNull(moduleA, "moduleA cannot be null");
        this.mapParameters = Objects.requireNonNull(mapParameters, "mapParameters cannot be null");
        this.mapField = new HashMap<>();
        this.matchTrackingOccurred = false;
        this.adjustedVigilance = 0.0;
    }
    
    /**
     * Get the underlying ART module A.
     * 
     * @return the ART module
     */
    public BaseART getModuleA() {
        return moduleA;
    }
    
    /**
     * Get the size of the map field.
     * 
     * @return number of cluster-to-label mappings
     */
    public int getMapFieldSize() {
        return mapField.size();
    }
    
    // BaseARTMAP interface implementation
    
    @Override
    public boolean isTrained() {
        return trained;
    }
    
    @Override
    public int getCategoryCount() {
        return moduleA.getCategoryCount();
    }
    
    @Override
    public void clear() {
        mapField.clear();
        moduleA.clear();
        trained = false;
        matchTrackingOccurred = false;
        adjustedVigilance = 0.0;
    }
    
    /**
     * Result of a training step.
     */
    public record TrainResult(
        int categoryA,
        int predictedLabel,
        boolean matchTrackingOccurred,
        double adjustedVigilance
    ) {}
    
    /**
     * Train on a single input pattern with its label.
     * 
     * @param input the input pattern
     * @param label the class label
     * @param artParams parameters for the ART module
     * @return the training result
     */
    public TrainResult train(Pattern input, int label, Object artParams) {
        Objects.requireNonNull(input, "input cannot be null");
        Objects.requireNonNull(artParams, "artParams cannot be null");
        
        matchTrackingOccurred = false;
        adjustedVigilance = 0.0;
        
        // Use match reset function to handle label conflicts
        MatchResetFunction matchResetFunc = (inputPattern, weight, categoryIndex, params, cache) -> {
            // Check if this category has a conflicting label
            if (mapField.containsKey(categoryIndex)) {
                int existingLabel = mapField.get(categoryIndex);
                if (existingLabel != label) {
                    // Reject this category - it has a different label
                    matchTrackingOccurred = true;
                    return false;
                }
            }
            // Accept this category
            return true;
        };
        
        // Use stepFit with match reset function and match tracking
        var result = moduleA.stepFit(
            input, 
            artParams, 
            matchResetFunc,
            MatchTrackingMode.MT_PLUS,
            mapParameters.epsilon()
        );
        
        if (result instanceof ActivationResult.Success success) {
            int categoryA = success.categoryIndex();
            
            // Create or verify the mapping
            if (!mapField.containsKey(categoryA)) {
                mapField.put(categoryA, label);
            } else {
                // Verify the mapping is correct (should always be true due to match reset func)
                assert mapField.get(categoryA) == label : "Label mismatch after match tracking";
            }
            
            trained = true;
            
            // If match tracking occurred, adjust the vigilance
            if (matchTrackingOccurred) {
                adjustedVigilance = mapParameters.mapFieldVigilance() + mapParameters.epsilon();
            }
            
            return new TrainResult(categoryA, label, matchTrackingOccurred, adjustedVigilance);
        } else {
            // This shouldn't happen with FuzzyART - it should always create a category
            throw new IllegalStateException("Unable to create category for input");
        }
    }
    
    /**
     * Fit the model to training data.
     * 
     * @param data array of input patterns
     * @param labels array of class labels
     * @param artParams parameters for the ART module
     */
    public void fit(Pattern[] data, int[] labels, Object artParams) {
        Objects.requireNonNull(data, "data cannot be null");
        Objects.requireNonNull(labels, "labels cannot be null");
        Objects.requireNonNull(artParams, "artParams cannot be null");
        
        if (data.length != labels.length) {
            throw new IllegalArgumentException("data and labels must have same length");
        }
        
        for (int i = 0; i < data.length; i++) {
            train(data[i], labels[i], artParams);
        }
        trained = true;
    }
    
    /**
     * Predict the class label for a single input pattern.
     * 
     * @param input the input pattern
     * @param artParams parameters for the ART module
     * @return the predicted class label, or -1 if no match
     */
    public int predict(Pattern input, Object artParams) {
        Objects.requireNonNull(input, "input cannot be null");
        Objects.requireNonNull(artParams, "artParams cannot be null");
        
        // Use stepFit for non-learning prediction (no actual learning occurs)
        var result = moduleA.stepFit(input, artParams);
        
        if (result instanceof ActivationResult.Success success) {
            int categoryA = success.categoryIndex();
            
            // Look up the label mapping
            if (mapField.containsKey(categoryA)) {
                return mapField.get(categoryA);
            } else {
                // No mapping found - unknown pattern
                return -1;
            }
        } else {
            // No category activated
            return -1;
        }
    }
    
    /**
     * Predict class labels for multiple input patterns.
     * 
     * @param data array of input patterns
     * @param artParams parameters for the ART module
     * @return array of predicted class labels
     */
    public int[] predict(Pattern[] data, Object artParams) {
        Objects.requireNonNull(data, "data cannot be null");
        Objects.requireNonNull(artParams, "artParams cannot be null");
        
        var predictions = new int[data.length];
        for (int i = 0; i < data.length; i++) {
            predictions[i] = predict(data[i], artParams);
        }
        return predictions;
    }
}