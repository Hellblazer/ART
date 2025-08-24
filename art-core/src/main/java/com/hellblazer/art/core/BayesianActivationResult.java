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

/**
 * BayesianActivationResult wraps ActivationResult.Success with Bayesian-specific data - MINIMAL STUB FOR TEST COMPILATION
 * This is a minimal implementation to allow tests to compile.
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
    
    public double uncertainty() {
        throw new UnsupportedOperationException("Not implemented yet");
    }
    
    public double confidence() {
        throw new UnsupportedOperationException("Not implemented yet");
    }
    
    public double posteriorProbability() {
        throw new UnsupportedOperationException("Not implemented yet");
    }
    
    public double[] getProbabilityDistribution() {
        throw new UnsupportedOperationException("Not implemented yet");
    }
    
    public Object getVisualizationData() {
        throw new UnsupportedOperationException("Not implemented yet");
    }
}