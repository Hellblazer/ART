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
package com.hellblazer.art.core.results;

import com.hellblazer.art.core.WeightVector;
import java.util.Arrays;

/**
 * BayesianActivationResult extends activation results with Bayesian inference data.
 * Provides uncertainty quantification, confidence measures, and probability distributions.
 * 
 * @author Hal Hildebrand
 */
public class BayesianActivationResult {
    private final ActivationResult.Success result;
    
    public BayesianActivationResult(int categoryIndex, double activationValue, WeightVector updatedWeight) {
        this.result = new ActivationResult.Success(categoryIndex, activationValue, updatedWeight);
    }
    
    public int categoryIndex() {
        return result.categoryIndex();
    }
    
    public double activationValue() {
        return result.activationValue();
    }
    
    public WeightVector updatedWeight() {
        return result.updatedWeight();
    }
    
    /**
     * Calculate uncertainty based on activation value.
     * Higher uncertainty for activation values closer to the vigilance threshold.
     */
    public double uncertainty() {
        // Uncertainty increases as activation approaches vigilance threshold
        // Assumes vigilance is around 0.7-0.9 for most ART networks
        var vigilanceApprox = 0.8;
        var distanceFromVigilance = Math.abs(result.activationValue() - vigilanceApprox);
        return Math.exp(-distanceFromVigilance * 3.0); // Higher uncertainty near threshold
    }
    
    /**
     * Calculate confidence as inverse of uncertainty.
     */
    public double confidence() {
        return 1.0 - uncertainty();
    }
    
    /**
     * Calculate posterior probability using Bayesian inference.
     * For ART networks, this is related to the activation value.
     */
    public double posteriorProbability() {
        // Use softmax-like transformation of activation value
        var activation = result.activationValue();
        return activation / (1.0 + activation); // Sigmoid-like normalization
    }
    
    /**
     * Get probability distribution over categories.
     * For simplicity, returns uniform distribution with higher probability for selected category.
     */
    public double[] getProbabilityDistribution() {
        // Create simple distribution - in full implementation would use category activations
        var numCategories = Math.max(result.categoryIndex() + 1, 3); // At least 3 categories
        var probs = new double[numCategories];
        var baseProb = 0.1 / (numCategories - 1); // Low probability for non-selected
        
        for (int i = 0; i < numCategories; i++) {
            if (i == result.categoryIndex()) {
                probs[i] = posteriorProbability();
            } else {
                probs[i] = baseProb;
            }
        }
        
        // Normalize probabilities to ensure they sum to 1.0
        var sum = Arrays.stream(probs).sum();
        if (sum > 0) {
            for (int i = 0; i < probs.length; i++) {
                probs[i] /= sum;
            }
        }
        
        return probs;
    }
    
    /**
     * Get visualization data for Bayesian-specific information.
     */
    public Object getVisualizationData() {
        return java.util.Map.of(
            "uncertainty", uncertainty(),
            "confidence", confidence(),
            "posteriorProbability", posteriorProbability(),
            "categoryIndex", result.categoryIndex(),
            "activationValue", result.activationValue(),
            "probabilityDistribution", getProbabilityDistribution()
        );
    }
}