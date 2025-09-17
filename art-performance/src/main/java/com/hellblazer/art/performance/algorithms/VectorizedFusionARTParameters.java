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

import java.util.Arrays;
import java.util.List;

/**
 * Parameters for VectorizedFusionART algorithm.
 * Extends base vectorized parameters with multi-channel fusion specific settings.
 */
public class VectorizedFusionARTParameters {
    
    private final double vigilance;
    private final double learningRate;
    private final double[] gammaValues;
    private final int[] channelDimensions;
    private final double[] channelVigilance;
    private final double[] channelLearningRates;
    private final VectorizedParameters baseParameters;
    private final boolean enableChannelSkipping;
    private final double activationThreshold;
    private final int maxSearchAttempts;
    
    public VectorizedFusionARTParameters(
            double vigilance,
            double learningRate,
            double[] gammaValues,
            int[] channelDimensions,
            double[] channelVigilance,
            double[] channelLearningRates,
            VectorizedParameters baseParameters,
            boolean enableChannelSkipping,
            double activationThreshold,
            int maxSearchAttempts) {
        
        validateInputs(vigilance, learningRate, gammaValues, channelDimensions, 
                       channelVigilance, channelLearningRates, activationThreshold, maxSearchAttempts);
        
        this.vigilance = vigilance;
        this.learningRate = learningRate;
        this.gammaValues = Arrays.copyOf(gammaValues, gammaValues.length);
        this.channelDimensions = Arrays.copyOf(channelDimensions, channelDimensions.length);
        this.channelVigilance = channelVigilance != null ? Arrays.copyOf(channelVigilance, channelVigilance.length) : null;
        this.channelLearningRates = channelLearningRates != null ? Arrays.copyOf(channelLearningRates, channelLearningRates.length) : null;
        this.baseParameters = baseParameters;
        this.enableChannelSkipping = enableChannelSkipping;
        this.activationThreshold = activationThreshold;
        this.maxSearchAttempts = maxSearchAttempts;
    }
    
    public static VectorizedFusionARTParameters defaults() {
        // Create default 2-channel fusion with equal weights
        var defaultGamma = new double[]{0.5, 0.5};
        var defaultChannelDims = new int[]{50, 50}; // 50 dimensions per channel
        var defaultBaseParams = VectorizedParameters.createDefault();
        
        return new VectorizedFusionARTParameters(
            0.75,    // vigilance
            0.01,    // learningRate
            defaultGamma,
            defaultChannelDims,
            null,    // channelVigilance (use global)
            null,    // channelLearningRates (use global)
            defaultBaseParams,
            false,   // enableChannelSkipping
            0.001,   // activationThreshold
            50       // maxSearchAttempts
        );
    }
    
    public static VectorizedFusionARTParameters createMultiChannel(int numChannels, int dimensionsPerChannel) {
        // Create equal-weighted multi-channel parameters
        var gamma = new double[numChannels];
        var channelDims = new int[numChannels];
        double weight = 1.0 / numChannels;
        
        Arrays.fill(gamma, weight);
        Arrays.fill(channelDims, dimensionsPerChannel);
        
        var baseParams = VectorizedParameters.createDefault();
        
        return new VectorizedFusionARTParameters(
            0.75,    // vigilance
            0.01,    // learningRate
            gamma,
            channelDims,
            null,    // channelVigilance (use global)
            null,    // channelLearningRates (use global)
            baseParams,
            false,   // enableChannelSkipping
            0.001,   // activationThreshold
            50       // maxSearchAttempts
        );
    }
    
    private static void validateInputs(double vigilance, double learningRate, double[] gammaValues,
                                       int[] channelDimensions, double[] channelVigilance,
                                       double[] channelLearningRates, double activationThreshold,
                                       int maxSearchAttempts) {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in [0, 1], got: " + vigilance);
        }
        
        if (learningRate < 0.0 || learningRate > 1.0) {
            throw new IllegalArgumentException("Learning rate must be in [0, 1], got: " + learningRate);
        }
        
        if (gammaValues == null || gammaValues.length < 2) {
            throw new IllegalArgumentException("FusionART requires at least 2 channels");
        }
        
        if (channelDimensions == null || channelDimensions.length != gammaValues.length) {
            throw new IllegalArgumentException("Channel dimensions must match gamma values length");
        }
        
        // Validate gamma values sum to 1.0
        double gammaSum = Arrays.stream(gammaValues).sum();
        if (Math.abs(gammaSum - 1.0) > 1e-6) {
            throw new IllegalArgumentException("Gamma values must sum to 1.0, got: " + gammaSum);
        }
        
        // Validate gamma values are in [0, 1]
        for (int i = 0; i < gammaValues.length; i++) {
            if (gammaValues[i] < 0.0 || gammaValues[i] > 1.0) {
                throw new IllegalArgumentException("Gamma values must be in [0, 1], got: " + gammaValues[i] + " at index " + i);
            }
        }
        
        // Validate channel dimensions are positive
        for (int i = 0; i < channelDimensions.length; i++) {
            if (channelDimensions[i] <= 0) {
                throw new IllegalArgumentException("Channel dimensions must be positive, got: " + channelDimensions[i] + " at index " + i);
            }
        }
        
        // Validate channel vigilance if provided
        if (channelVigilance != null) {
            if (channelVigilance.length != gammaValues.length) {
                throw new IllegalArgumentException("Channel vigilance length must match number of channels");
            }
            for (int i = 0; i < channelVigilance.length; i++) {
                if (channelVigilance[i] < 0.0 || channelVigilance[i] > 1.0) {
                    throw new IllegalArgumentException("Channel vigilance must be in [0, 1], got: " + channelVigilance[i] + " at index " + i);
                }
            }
        }
        
        // Validate channel learning rates if provided
        if (channelLearningRates != null) {
            if (channelLearningRates.length != gammaValues.length) {
                throw new IllegalArgumentException("Channel learning rates length must match number of channels");
            }
            for (int i = 0; i < channelLearningRates.length; i++) {
                if (channelLearningRates[i] < 0.0 || channelLearningRates[i] > 1.0) {
                    throw new IllegalArgumentException("Channel learning rates must be in [0, 1], got: " + channelLearningRates[i] + " at index " + i);
                }
            }
        }
        
        if (activationThreshold < 0.0) {
            throw new IllegalArgumentException("Activation threshold must be non-negative, got: " + activationThreshold);
        }
        
        if (maxSearchAttempts <= 0) {
            throw new IllegalArgumentException("Max search attempts must be positive, got: " + maxSearchAttempts);
        }
    }
    
    // Getters
    
    public double vigilanceThreshold() {
        return vigilance;
    }
    
    public double getLearningRate() {
        return learningRate;
    }
    
    public double[] getGammaValues() {
        return Arrays.copyOf(gammaValues, gammaValues.length);
    }
    
    public int[] getChannelDimensions() {
        return Arrays.copyOf(channelDimensions, channelDimensions.length);
    }
    
    public int getNumChannels() {
        return gammaValues.length;
    }
    
    public int getTotalDimension() {
        return Arrays.stream(channelDimensions).sum();
    }
    
    public double getChannelVigilance(int channel) {
        if (channelVigilance != null && channel < channelVigilance.length) {
            return channelVigilance[channel];
        }
        return vigilance; // Use global vigilance as default
    }
    
    public double getChannelLearningRate(int channel) {
        if (channelLearningRates != null && channel < channelLearningRates.length) {
            return channelLearningRates[channel];
        }
        return learningRate; // Use global learning rate as default
    }
    
    public VectorizedParameters getBaseParameters() {
        return baseParameters;
    }
    
    public boolean isChannelSkippingEnabled() {
        return enableChannelSkipping;
    }
    
    public double getActivationThreshold() {
        return activationThreshold;
    }
    
    public int getMaxSearchAttempts() {
        return maxSearchAttempts;
    }
    
    /**
     * Check if pattern has the expected total dimension.
     */
    public boolean isValidPatternDimension(int dimension) {
        return dimension == getTotalDimension();
    }
    
    /**
     * Get the start and end indices for a specific channel in the combined pattern.
     */
    public int[] getChannelIndices(int channel) {
        if (channel < 0 || channel >= getNumChannels()) {
            throw new IllegalArgumentException("Channel index out of bounds: " + channel);
        }
        
        int start = 0;
        for (int i = 0; i < channel; i++) {
            start += channelDimensions[i];
        }
        int end = start + channelDimensions[channel];
        
        return new int[]{start, end};
    }
    
    /**
     * Create a builder for more complex parameter configurations.
     */
    public static Builder builder() {
        return new Builder();
    }
    
    public static class Builder {
        private double vigilance = 0.75;
        private double learningRate = 0.01;
        private double[] gammaValues;
        private int[] channelDimensions;
        private double[] channelVigilance;
        private double[] channelLearningRates;
        private VectorizedParameters baseParameters;
        private boolean enableChannelSkipping = false;
        private double activationThreshold = 0.001;
        private int maxSearchAttempts = 50;
        
        public Builder vigilance(double vigilance) {
            this.vigilance = vigilance;
            return this;
        }
        
        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }
        
        public Builder gammaValues(double[] gammaValues) {
            this.gammaValues = gammaValues != null ? Arrays.copyOf(gammaValues, gammaValues.length) : null;
            return this;
        }
        
        public Builder channelDimensions(int[] channelDimensions) {
            this.channelDimensions = channelDimensions != null ? Arrays.copyOf(channelDimensions, channelDimensions.length) : null;
            return this;
        }
        
        public Builder channelVigilance(double[] channelVigilance) {
            this.channelVigilance = channelVigilance != null ? Arrays.copyOf(channelVigilance, channelVigilance.length) : null;
            return this;
        }
        
        public Builder channelLearningRates(double[] channelLearningRates) {
            this.channelLearningRates = channelLearningRates != null ? Arrays.copyOf(channelLearningRates, channelLearningRates.length) : null;
            return this;
        }
        
        public Builder baseParameters(VectorizedParameters baseParameters) {
            this.baseParameters = baseParameters;
            return this;
        }
        
        public Builder enableChannelSkipping(boolean enableChannelSkipping) {
            this.enableChannelSkipping = enableChannelSkipping;
            return this;
        }
        
        public Builder activationThreshold(double activationThreshold) {
            this.activationThreshold = activationThreshold;
            return this;
        }
        
        public Builder maxSearchAttempts(int maxSearchAttempts) {
            this.maxSearchAttempts = maxSearchAttempts;
            return this;
        }
        
        public VectorizedFusionARTParameters build() {
            if (baseParameters == null) {
                baseParameters = VectorizedParameters.createDefault();
            }
            if (gammaValues == null || channelDimensions == null) {
                throw new IllegalArgumentException("Gamma values and channel dimensions are required");
            }
            
            return new VectorizedFusionARTParameters(
                vigilance, learningRate, gammaValues, channelDimensions,
                channelVigilance, channelLearningRates, baseParameters,
                enableChannelSkipping, activationThreshold, maxSearchAttempts
            );
        }
    }
}