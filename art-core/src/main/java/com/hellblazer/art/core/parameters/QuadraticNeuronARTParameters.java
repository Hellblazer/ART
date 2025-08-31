/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 * 
 * This file is part of Java ART Neural Networks.
 */
package com.hellblazer.art.core.parameters;

/**
 * Parameters for QuadraticNeuronART algorithm.
 * 
 * QuadraticNeuronART clusters data in hyper-ellipsoids by utilizing a quadratic
 * neural network for activation and resonance.
 * 
 * @param vigilance the vigilance parameter (ρ) in range [0, 1]
 * @param sInit the initial quadratic term s_init
 * @param learningRateB the learning rate for cluster mean (bias) lr_b in range (0, 1]
 * @param learningRateW the learning rate for cluster weights lr_w in range [0, 1]
 * @param learningRateS the learning rate for the quadratic term lr_s in range [0, 1]
 */
public record QuadraticNeuronARTParameters(
    double vigilance,
    double sInit,
    double learningRateB,
    double learningRateW,
    double learningRateS
) {
    
    /**
     * Constructor with validation.
     */
    public QuadraticNeuronARTParameters {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in range [0, 1], got: " + vigilance);
        }
        if (learningRateB <= 0.0 || learningRateB > 1.0) {
            throw new IllegalArgumentException("Learning rate for bias (learningRateB) must be in range (0, 1], got: " + learningRateB);
        }
        if (learningRateW < 0.0 || learningRateW > 1.0) {
            throw new IllegalArgumentException("Learning rate for weights (learningRateW) must be in range [0, 1], got: " + learningRateW);
        }
        if (learningRateS < 0.0 || learningRateS > 1.0) {
            throw new IllegalArgumentException("Learning rate for quadratic term (learningRateS) must be in range [0, 1], got: " + learningRateS);
        }
    }
    
    /**
     * Create parameters with specified values.
     */
    public static QuadraticNeuronARTParameters of(double vigilance, double sInit, 
                                                  double learningRateB, double learningRateW, 
                                                  double learningRateS) {
        return new QuadraticNeuronARTParameters(vigilance, sInit, learningRateB, learningRateW, learningRateS);
    }
    
    /**
     * Create parameters with default values.
     * Default: vigilance=0.7, sInit=0.5, learningRateB=0.1, learningRateW=0.1, learningRateS=0.05
     */
    public static QuadraticNeuronARTParameters defaults() {
        return new QuadraticNeuronARTParameters(0.7, 0.5, 0.1, 0.1, 0.05);
    }
    
    /**
     * Get the vigilance parameter (alias for rho)
     */
    public double getRho() {
        return vigilance;
    }
    
    /**
     * Get the initial quadratic term
     */
    public double getSInit() {
        return sInit;
    }
    
    /**
     * Get the learning rate for bias
     */
    public double getLearningRateB() {
        return learningRateB;
    }
    
    /**
     * Get the learning rate for weights
     */
    public double getLearningRateW() {
        return learningRateW;
    }
    
    /**
     * Get the learning rate for quadratic term
     */
    public double getLearningRateS() {
        return learningRateS;
    }
    
    /**
     * Validate parameters
     */
    public void validate() {
        // Validation already done in constructor
    }
    
    /**
     * Builder for QuadraticNeuronARTParameters.
     */
    public static class Builder {
        private double rho = 0.7;
        private double sInit = 0.5;
        private double learningRateB = 0.1;
        private double learningRateW = 0.1;
        private double learningRateS = 0.05;

        public Builder rho(double rho) {
            this.rho = rho;
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

        public QuadraticNeuronARTParameters build() {
            return new QuadraticNeuronARTParameters(rho, sInit, learningRateB, learningRateW, learningRateS);
        }
    }

    public static Builder builder() {
        return new Builder();
    }
}