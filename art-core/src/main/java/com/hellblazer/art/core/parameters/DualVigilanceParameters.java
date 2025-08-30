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

import java.util.Objects;

/**
 * Immutable parameters for DualVigilanceART algorithm.
 * 
 * DualVigilanceART uses two vigilance thresholds to improve noise handling:
 * - Lower vigilance (rho_lb): Defines boundary nodes for noise tolerance
 * - Upper vigilance (rho): Standard matching criterion for category assignment
 * 
 * @param rhoLb lower vigilance parameter (ρ_lb) in range [0, 1], must be < rho
 * @param rho upper vigilance parameter (ρ) in range [0, 1], must be > rhoLb
 * @param beta learning rate (β) in range (0, 1]
 * @param alpha choice parameter (α) for activation calculation, typically small positive value
 * @param maxCategories maximum number of categories allowed
 */
public record DualVigilanceParameters(
    double rhoLb, 
    double rho, 
    double beta,
    double alpha,
    int maxCategories
) {
    
    /**
     * Constructor with validation.
     */
    public DualVigilanceParameters {
        // Validate lower vigilance
        if (rhoLb < 0.0 || rhoLb > 1.0) {
            throw new IllegalArgumentException(
                "Lower vigilance must be in range [0, 1], got: " + rhoLb);
        }
        
        // Validate upper vigilance
        if (rho < 0.0 || rho > 1.0) {
            throw new IllegalArgumentException(
                "Upper vigilance must be in range [0, 1], got: " + rho);
        }
        
        // Validate relationship between vigilance parameters
        if (rhoLb >= rho) {
            throw new IllegalArgumentException(
                "Lower vigilance must be less than upper vigilance. Got rhoLb=" + 
                rhoLb + ", rho=" + rho);
        }
        
        // Validate learning rate
        if (beta <= 0.0 || beta > 1.0) {
            throw new IllegalArgumentException(
                "Learning rate must be in range (0, 1], got: " + beta);
        }
        
        // Validate alpha
        if (alpha < 0.0) {
            throw new IllegalArgumentException(
                "Alpha must be non-negative, got: " + alpha);
        }
        
        // Validate max categories
        if (maxCategories <= 0) {
            throw new IllegalArgumentException(
                "Max categories must be positive, got: " + maxCategories);
        }
        
        // Check for NaN values
        if (Double.isNaN(rhoLb) || Double.isNaN(rho) || Double.isNaN(beta) || Double.isNaN(alpha)) {
            throw new IllegalArgumentException("Parameters cannot be NaN");
        }
        
        // Check for infinite values
        if (Double.isInfinite(rhoLb) || Double.isInfinite(rho) || Double.isInfinite(beta) || Double.isInfinite(alpha)) {
            throw new IllegalArgumentException("Parameters cannot be infinite");
        }
    }
    
    /**
     * Create DualVigilanceParameters with specified values.
     * 
     * @param rhoLb lower vigilance parameter ρ_lb ∈ [0, 1]
     * @param rho upper vigilance parameter ρ ∈ [0, 1], ρ > ρ_lb
     * @param beta learning rate β ∈ (0, 1]
     * @param maxCategories maximum categories allowed
     * @return new DualVigilanceParameters instance
     */
    public static DualVigilanceParameters of(
            double rhoLb, double rho, double beta, double alpha, int maxCategories) {
        return new DualVigilanceParameters(rhoLb, rho, beta, alpha, maxCategories);
    }
    
    /**
     * Create DualVigilanceParameters with default values.
     * Default: rhoLb=0.4, rho=0.7, beta=0.1, maxCategories=1000
     * 
     * @return default DualVigilanceParameters
     */
    public static DualVigilanceParameters defaults() {
        return new DualVigilanceParameters(0.4, 0.7, 0.1, 0.001, 1000);
    }
    
    /**
     * Get the vigilance gap (difference between upper and lower).
     * 
     * @return rho - rhoLb
     */
    public double vigilanceGap() {
        return rho - rhoLb;
    }
    
    /**
     * Check if a match value passes the upper vigilance test.
     * Uses epsilon tolerance to handle floating point precision issues.
     * 
     * @param matchValue the match value to test
     * @return true if matchValue >= rho (within epsilon tolerance)
     */
    public boolean passesUpperVigilance(double matchValue) {
        return matchValue >= (rho - 1e-12);
    }
    
    /**
     * Check if a match value passes the lower vigilance test.
     * Uses epsilon tolerance to handle floating point precision issues.
     * 
     * @param matchValue the match value to test
     * @return true if matchValue >= rhoLb (within epsilon tolerance)
     */
    public boolean passesLowerVigilance(double matchValue) {
        return matchValue >= (rhoLb - 1e-10);
    }
    
    /**
     * Check if a match value is in the boundary zone.
     * A value is in the boundary zone if it passes lower but fails upper vigilance.
     * 
     * @param matchValue the match value to test
     * @return true if rhoLb <= matchValue < rho
     */
    public boolean isInBoundaryZone(double matchValue) {
        return passesLowerVigilance(matchValue) && !passesUpperVigilance(matchValue);
    }
    
    /**
     * Create a new DualVigilanceParameters with different lower vigilance.
     * 
     * @param newRhoLb the new lower vigilance value
     * @return new DualVigilanceParameters instance
     */
    public DualVigilanceParameters withLowerVigilance(double newRhoLb) {
        return new DualVigilanceParameters(newRhoLb, rho, beta, alpha, maxCategories);
    }
    
    /**
     * Create a new DualVigilanceParameters with different upper vigilance.
     * 
     * @param newRho the new upper vigilance value
     * @return new DualVigilanceParameters instance
     */
    public DualVigilanceParameters withUpperVigilance(double newRho) {
        return new DualVigilanceParameters(rhoLb, newRho, beta, alpha, maxCategories);
    }
    
    /**
     * Create a new DualVigilanceParameters with different learning rate.
     * 
     * @param newBeta the new learning rate
     * @return new DualVigilanceParameters instance
     */
    public DualVigilanceParameters withLearningRate(double newBeta) {
        return new DualVigilanceParameters(rhoLb, rho, newBeta, alpha, maxCategories);
    }
    
    /**
     * Create a new DualVigilanceParameters with different max categories.
     * 
     * @param newMaxCategories the new maximum categories
     * @return new DualVigilanceParameters instance
     */
    public DualVigilanceParameters withMaxCategories(int newMaxCategories) {
        return new DualVigilanceParameters(rhoLb, rho, beta, alpha, newMaxCategories);
    }
    
    /**
     * Create a builder for DualVigilanceParameters.
     * 
     * @return new Builder instance
     */
    public static Builder builder() {
        return new Builder();
    }
    
    /**
     * Builder class for DualVigilanceParameters.
     */
    public static class Builder {
        private double rhoLb = 0.4;
        private double rho = 0.7;
        private double beta = 0.1;
        private double alpha = 0.001;
        private int maxCategories = 1000;
        
        /**
         * Set the lower vigilance parameter.
         * 
         * @param rhoLb the lower vigilance ρ_lb ∈ [0, 1]
         * @return this builder
         */
        public Builder lowerVigilance(double rhoLb) {
            this.rhoLb = rhoLb;
            return this;
        }
        
        /**
         * Set the upper vigilance parameter.
         * 
         * @param rho the upper vigilance ρ ∈ [0, 1]
         * @return this builder
         */
        public Builder upperVigilance(double rho) {
            this.rho = rho;
            return this;
        }
        
        /**
         * Set the learning rate.
         * 
         * @param beta the learning rate β ∈ (0, 1]
         * @return this builder
         */
        public Builder learningRate(double beta) {
            this.beta = beta;
            return this;
        }
        
        /**
         * Set the alpha parameter.
         * 
         * @param alpha the alpha parameter for activation calculation
         * @return this builder
         */
        public Builder alpha(double alpha) {
            this.alpha = alpha;
            return this;
        }
        
        /**
         * Set the maximum categories.
         * 
         * @param maxCategories the maximum number of categories
         * @return this builder
         */
        public Builder maxCategories(int maxCategories) {
            this.maxCategories = maxCategories;
            return this;
        }
        
        /**
         * Build the DualVigilanceParameters instance.
         * 
         * @return new DualVigilanceParameters with specified values
         */
        public DualVigilanceParameters build() {
            return new DualVigilanceParameters(rhoLb, rho, beta, alpha, maxCategories);
        }
    }
    
    @Override
    public String toString() {
        return String.format(
            "DualVigilanceParameters{ρ_lb=%.3f, ρ=%.3f, β=%.3f, α=%.3f, maxCat=%d, gap=%.3f}", 
            rhoLb, rho, beta, alpha, maxCategories, vigilanceGap());
    }
}