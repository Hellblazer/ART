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
 * Parameters for VectorizedFusionART algorithm.
 * Extends VectorizedParameters with fusion-specific settings for multi-channel processing.
 */
public record VectorizedFusionARTParameters(
    double vigilance,
    double learningRate,
    double[] gamma,
    int[] channelDimensions,
    double[] channelVigilance,
    double[] channelWeights,
    VectorizedParameters baseParameters,
    boolean enableChannelSkipping,
    double activationThreshold,
    int maxSearchAttempts
) {
    
    public VectorizedFusionARTParameters {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be between 0.0 and 1.0");
        }
        if (learningRate < 0.0 || learningRate > 1.0) {
            throw new IllegalArgumentException("Learning rate must be between 0.0 and 1.0");
        }
        if (gamma == null || gamma.length == 0) {
            throw new IllegalArgumentException("Gamma values cannot be null or empty");
        }
        if (channelDimensions == null || channelDimensions.length == 0) {
            throw new IllegalArgumentException("Channel dimensions cannot be null or empty");
        }
        if (channelVigilance == null || channelVigilance.length != gamma.length) {
            throw new IllegalArgumentException("Channel vigilance array must match gamma array length");
        }
        if (channelWeights == null || channelWeights.length != gamma.length) {
            throw new IllegalArgumentException("Channel weights array must match gamma array length");
        }
        if (baseParameters == null) {
            throw new IllegalArgumentException("Base parameters cannot be null");
        }
        if (activationThreshold < 0.0 || activationThreshold > 1.0) {
            throw new IllegalArgumentException("Activation threshold must be between 0.0 and 1.0");
        }
        if (maxSearchAttempts < 1) {
            throw new IllegalArgumentException("Max search attempts must be at least 1");
        }
    }
    
    public static VectorizedFusionARTParameters createDefault() {
        var baseParams = VectorizedParameters.createDefault();
        return new VectorizedFusionARTParameters(
            0.7,                                                    // vigilance
            0.01,                                                   // learningRate
            new double[]{1.0, 1.0, 1.0},                           // gamma
            new int[]{4, 4, 4},                                    // channelDimensions
            new double[]{0.7, 0.7, 0.7},                          // channelVigilance
            new double[]{0.6, 0.3, 0.1},                          // channelWeights
            baseParams,                                             // baseParameters
            false,                                                  // enableChannelSkipping
            0.5,                                                    // activationThreshold
            10                                                      // maxSearchAttempts
        );
    }
    
    public static VectorizedFusionARTParameters createWithVigilance(double vigilance) {
        var defaultParams = createDefault();
        return new VectorizedFusionARTParameters(
            vigilance,
            defaultParams.learningRate(),
            defaultParams.gamma(),
            defaultParams.channelDimensions(),
            new double[]{vigilance, vigilance, vigilance},
            defaultParams.channelWeights(),
            defaultParams.baseParameters(),
            defaultParams.enableChannelSkipping(),
            defaultParams.activationThreshold(),
            defaultParams.maxSearchAttempts()
        );
    }
    
    public int getNumChannels() {
        return gamma.length;
    }
    
    public int getTotalDimensions() {
        int total = 0;
        for (int dim : channelDimensions) {
            total += dim;
        }
        return total;
    }
    
    // Legacy method names for compatibility
    public int getTotalDimension() {
        return getTotalDimensions();
    }
    
    public double[] getGammaValues() {
        return gamma.clone();
    }
    
    public int[] getChannelDimensions() {
        return channelDimensions.clone();
    }
    
    public double vigilanceThreshold() {
        return vigilance;
    }
    
    public double getLearningRate() {
        return learningRate;
    }
    
    public boolean isValidPatternDimension(int dimension) {
        return dimension == getTotalDimensions();
    }
    
    public double getChannelVigilance(int channel) {
        if (channel >= 0 && channel < channelVigilance.length) {
            return channelVigilance[channel];
        }
        return vigilance;
    }
    
    public double getChannelWeight(int channel) {
        if (channel >= 0 && channel < channelWeights.length) {
            return channelWeights[channel];
        }
        return 1.0 / getNumChannels(); // Equal weights as fallback
    }
    
    public double getGamma(int channel) {
        if (channel >= 0 && channel < gamma.length) {
            return gamma[channel];
        }
        return 1.0; // Default gamma
    }
    
    public int getChannelDimension(int channel) {
        if (channel >= 0 && channel < channelDimensions.length) {
            return channelDimensions[channel];
        }
        return 0;
    }
}