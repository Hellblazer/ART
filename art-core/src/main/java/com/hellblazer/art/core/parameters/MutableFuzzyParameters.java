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
 * Mutable wrapper for FuzzyParameters to support match tracking.
 * 
 * This class is used internally by FuzzyART and FuzzyARTMAP to allow
 * vigilance parameter adjustment during match tracking while maintaining
 * the immutability of the public FuzzyParameters API.
 * 
 * @author Hal Hildebrand
 */
public class MutableFuzzyParameters {
    private double vigilance;
    private final double alpha;
    private final double beta;
    
    /**
     * Create mutable parameters from immutable FuzzyParameters.
     * 
     * @param params the immutable parameters to copy
     */
    public MutableFuzzyParameters(FuzzyParameters params) {
        this.vigilance = params.vigilance();
        this.alpha = params.alpha();
        this.beta = params.beta();
    }
    
    /**
     * Get the current vigilance value.
     * 
     * @return the vigilance parameter
     */
    public double vigilance() {
        return vigilance;
    }
    
    /**
     * Set the vigilance parameter.
     * 
     * @param vigilance the new vigilance value
     */
    public void setVigilance(double vigilance) {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in range [0, 1], got: " + vigilance);
        }
        this.vigilance = vigilance;
    }
    
    /**
     * Get the choice parameter alpha.
     * 
     * @return the alpha parameter
     */
    public double alpha() {
        return alpha;
    }
    
    /**
     * Get the learning rate beta.
     * 
     * @return the beta parameter
     */
    public double beta() {
        return beta;
    }
    
    /**
     * Convert back to immutable FuzzyParameters.
     * 
     * @return immutable FuzzyParameters with current values
     */
    public FuzzyParameters toImmutable() {
        return new FuzzyParameters(vigilance, alpha, beta);
    }
    
    @Override
    public String toString() {
        return String.format("MutableFuzzyParameters{ρ=%.3f, α=%.3f, β=%.3f}", 
                           vigilance, alpha, beta);
    }
}