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
package com.hellblazer.art.core.parameters;

/**
 * Parameters for the FuzzyARTMAP supervised learning algorithm.
 * 
 * FuzzyARTMAP combines FuzzyART clustering with supervised learning through
 * a map field that associates clusters with class labels. The parameters
 * control both the FuzzyART clustering behavior and the match tracking
 * mechanism for handling label conflicts.
 * 
 * @param rho     Vigilance parameter in [0,1] - controls category specificity
 * @param alpha   Choice parameter > 0 - biases category selection
 * @param beta    Learning rate in [0,1] - controls weight update speed
 * @param epsilon Match tracking increment > 0 - vigilance adjustment for conflicts
 * 
 * @author Hal Hildebrand
 */
public record FuzzyARTMAPParameters(
    double rho,
    double alpha,
    double beta,
    double epsilon
) {
    
    /**
     * Validate and create FuzzyARTMAP parameters.
     * 
     * @throws IllegalArgumentException if any parameter is out of valid range
     */
    public FuzzyARTMAPParameters {
        if (rho < 0.0 || rho > 1.0) {
            throw new IllegalArgumentException(
                "Vigilance parameter rho must be in [0,1], got: " + rho
            );
        }
        if (alpha < 0.0) {
            throw new IllegalArgumentException(
                "Choice parameter alpha must be positive, got: " + alpha
            );
        }
        if (beta < 0.0 || beta > 1.0) {
            throw new IllegalArgumentException(
                "Learning rate beta must be in [0,1], got: " + beta
            );
        }
        if (epsilon < 0.0) {
            throw new IllegalArgumentException(
                "Match tracking epsilon must be positive, got: " + epsilon
            );
        }
    }
    
    /**
     * Create FuzzyParameters for the internal FuzzyART module.
     * 
     * @return FuzzyParameters with rho, alpha, and beta values
     */
    public FuzzyParameters toFuzzyParameters() {
        return new FuzzyParameters(rho, alpha, beta);
    }
    
    /**
     * Get default FuzzyARTMAP parameters matching Python implementation.
     * 
     * @return default parameters (rho=0.8, alpha=1e-10, beta=1.0, epsilon=1e-10)
     */
    public static FuzzyARTMAPParameters defaultParameters() {
        return new FuzzyARTMAPParameters(0.8, 1e-10, 1.0, 1e-10);
    }
}