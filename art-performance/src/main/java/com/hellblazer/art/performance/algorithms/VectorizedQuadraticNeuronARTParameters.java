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
package com.hellblazer.art.performance.algorithms;

/**
 * Parameters for VectorizedQuadraticNeuronART algorithm.
 * 
 * QuadraticNeuronART clusters data in hyper-ellipsoids by utilizing a quadratic
 * neural network for activation and resonance. This vectorized version provides
 * enhanced performance through SIMD operations and optimized matrix computations.
 * 
 * Key features:
 * - Quadratic neuron activation: T = exp(-s^2 * ||W*x - b||^2)
 * - Adaptive learning rates for centroid, weights, and quadratic term
 * - SIMD-optimized matrix-vector operations
 * - Performance tracking for ellipsoidal clustering metrics
 */
public class VectorizedQuadraticNeuronARTParameters {
    
    private final double vigilance;
    private final double sInit;
    private final double learningRateB;
    private final double learningRateW;
    private final double learningRateS;
    private final VectorizedParameters baseParameters;
    private final boolean enableAdaptiveS;
    private final double minS;
    private final double maxS;
    private final int matrixDimension;
    private final double regularizationFactor;
    private final boolean enableMatrixRegularization;
    
    public VectorizedQuadraticNeuronARTParameters(
            double vigilance,
            double sInit,
            double learningRateB,
            double learningRateW,
            double learningRateS,
            VectorizedParameters baseParameters,
            boolean enableAdaptiveS,
            double minS,
            double maxS,
            int matrixDimension,
            double regularizationFactor,
            boolean enableMatrixRegularization) {
        
        validateInputs(vigilance, sInit, learningRateB, learningRateW, learningRateS,
                       minS, maxS, matrixDimension, regularizationFactor);
        
        this.vigilance = vigilance;
        this.sInit = sInit;
        this.learningRateB = learningRateB;
        this.learningRateW = learningRateW;
        this.learningRateS = learningRateS;
        this.baseParameters = baseParameters;
        this.enableAdaptiveS = enableAdaptiveS;
        this.minS = minS;
        this.maxS = maxS;
        this.matrixDimension = matrixDimension;
        this.regularizationFactor = regularizationFactor;
        this.enableMatrixRegularization = enableMatrixRegularization;
    }
    
    public static VectorizedQuadraticNeuronARTParameters defaults() {
        var defaultBaseParams = VectorizedParameters.createDefault();
        
        return new VectorizedQuadraticNeuronARTParameters(
            0.75,    // vigilance
            0.5,     // sInit
            0.1,     // learningRateB
            0.1,     // learningRateW
            0.05,    // learningRateS
            defaultBaseParams,
            true,    // enableAdaptiveS
            0.01,    // minS
            10.0,    // maxS
            100,     // matrixDimension (default input dimension)
            0.001,   // regularizationFactor
            true     // enableMatrixRegularization
        );
    }
    
    public static VectorizedQuadraticNeuronARTParameters forDimension(int inputDimension) {
        var baseParams = VectorizedParameters.createDefault();
        
        return new VectorizedQuadraticNeuronARTParameters(
            0.75,    // vigilance
            0.5,     // sInit
            0.1,     // learningRateB
            0.1,     // learningRateW
            0.05,    // learningRateS
            baseParams,
            true,    // enableAdaptiveS
            0.01,    // minS
            10.0,    // maxS
            inputDimension, // matrixDimension
            0.001,   // regularizationFactor
            true     // enableMatrixRegularization
        );
    }
    
    private static void validateInputs(double vigilance, double sInit, double learningRateB,
                                       double learningRateW, double learningRateS,
                                       double minS, double maxS, int matrixDimension,
                                       double regularizationFactor) {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in [0, 1], got: " + vigilance);
        }
        
        if (sInit <= 0.0) {
            throw new IllegalArgumentException("Initial quadratic term must be positive, got: " + sInit);
        }
        
        if (learningRateB <= 0.0 || learningRateB > 1.0) {
            throw new IllegalArgumentException("Learning rate for bias must be in (0, 1], got: " + learningRateB);
        }
        
        if (learningRateW < 0.0 || learningRateW > 1.0) {
            throw new IllegalArgumentException("Learning rate for weights must be in [0, 1], got: " + learningRateW);
        }
        
        if (learningRateS < 0.0 || learningRateS > 1.0) {
            throw new IllegalArgumentException("Learning rate for quadratic term must be in [0, 1], got: " + learningRateS);
        }
        
        if (minS <= 0.0 || maxS <= minS) {
            throw new IllegalArgumentException("Must have 0 < minS < maxS, got minS=" + minS + ", maxS=" + maxS);
        }
        
        if (matrixDimension <= 0) {
            throw new IllegalArgumentException("Matrix dimension must be positive, got: " + matrixDimension);
        }
        
        if (regularizationFactor < 0.0) {
            throw new IllegalArgumentException("Regularization factor must be non-negative, got: " + regularizationFactor);
        }
    }
    
    // Getters
    
    public double vigilanceThreshold() {
        return vigilance;
    }
    
    public double getVigilance() {
        return vigilance;
    }
    
    public double getSInit() {
        return sInit;
    }
    
    public double getLearningRateB() {
        return learningRateB;
    }
    
    public double getLearningRateW() {
        return learningRateW;
    }
    
    public double getLearningRateS() {
        return learningRateS;
    }
    
    public VectorizedParameters getBaseParameters() {
        return baseParameters;
    }
    
    public boolean isAdaptiveSEnabled() {
        return enableAdaptiveS;
    }
    
    public double getMinS() {
        return minS;
    }
    
    public double getMaxS() {
        return maxS;
    }
    
    public int getMatrixDimension() {
        return matrixDimension;
    }
    
    public double getRegularizationFactor() {
        return regularizationFactor;
    }
    
    public boolean isMatrixRegularizationEnabled() {
        return enableMatrixRegularization;
    }
    
    /**
     * Clamp the quadratic term s to valid range.
     */
    public double clampS(double s) {
        if (!enableAdaptiveS) {
            return sInit; // Use fixed initial value
        }
        return Math.max(minS, Math.min(maxS, s));
    }
    
    /**
     * Check if the given dimension matches the expected matrix dimension.
     */
    public boolean isValidDimension(int dimension) {
        return dimension == matrixDimension;
    }
    
    /**
     * Calculate the total number of matrix elements.
     */
    public int getMatrixSize() {
        return matrixDimension * matrixDimension;
    }
    
    /**
     * Get the expected number of parameters for the quadratic neuron.
     * This includes the matrix elements, centroid vector, and quadratic term.
     */
    public int getTotalParameterCount() {
        return getMatrixSize() + matrixDimension + 1; // Matrix + centroid + s
    }
    
    /**
     * Create a builder for more complex parameter configurations.
     */
    public static Builder builder() {
        return new Builder();
    }
    
    public static class Builder {
        private double vigilance = 0.75;
        private double sInit = 0.5;
        private double learningRateB = 0.1;
        private double learningRateW = 0.1;
        private double learningRateS = 0.05;
        private VectorizedParameters baseParameters;
        private boolean enableAdaptiveS = true;
        private double minS = 0.01;
        private double maxS = 10.0;
        private int matrixDimension = 100;
        private double regularizationFactor = 0.001;
        private boolean enableMatrixRegularization = true;
        
        public Builder vigilance(double vigilance) {
            this.vigilance = vigilance;
            return this;
        }
        
        public Builder sInit(double sInit) {
            this.sInit = sInit;
            return this;
        }
        
        public Builder learningRateB(double learningRateB) {
            this.learningRateB = learningRateB;
            return this;
        }
        
        public Builder learningRateW(double learningRateW) {
            this.learningRateW = learningRateW;
            return this;
        }
        
        public Builder learningRateS(double learningRateS) {
            this.learningRateS = learningRateS;
            return this;
        }
        
        public Builder baseParameters(VectorizedParameters baseParameters) {
            this.baseParameters = baseParameters;
            return this;
        }
        
        public Builder enableAdaptiveS(boolean enableAdaptiveS) {
            this.enableAdaptiveS = enableAdaptiveS;
            return this;
        }
        
        public Builder minS(double minS) {
            this.minS = minS;
            return this;
        }
        
        public Builder maxS(double maxS) {
            this.maxS = maxS;
            return this;
        }
        
        public Builder matrixDimension(int matrixDimension) {
            this.matrixDimension = matrixDimension;
            return this;
        }
        
        public Builder regularizationFactor(double regularizationFactor) {
            this.regularizationFactor = regularizationFactor;
            return this;
        }
        
        public Builder enableMatrixRegularization(boolean enableMatrixRegularization) {
            this.enableMatrixRegularization = enableMatrixRegularization;
            return this;
        }
        
        public VectorizedQuadraticNeuronARTParameters build() {
            if (baseParameters == null) {
                baseParameters = VectorizedParameters.createDefault();
            }
            
            return new VectorizedQuadraticNeuronARTParameters(
                vigilance, sInit, learningRateB, learningRateW, learningRateS,
                baseParameters, enableAdaptiveS, minS, maxS, matrixDimension,
                regularizationFactor, enableMatrixRegularization
            );
        }
    }
}