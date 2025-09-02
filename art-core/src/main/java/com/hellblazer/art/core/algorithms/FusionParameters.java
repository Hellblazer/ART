package com.hellblazer.art.core.algorithms;

/**
 * Parameters for FusionART algorithm.
 * Contains settings for multi-channel fusion learning.
 */
public class FusionParameters {
    private final double vigilance;
    private final double learningRate;
    private final double[] channelVigilance;
    private final double[] channelLearningRates;
    
    private FusionParameters(Builder builder) {
        this.vigilance = builder.vigilance;
        this.learningRate = builder.learningRate;
        this.channelVigilance = builder.channelVigilance;
        this.channelLearningRates = builder.channelLearningRates;
    }
    
    public double getVigilance() {
        return vigilance;
    }
    
    public double getLearningRate() {
        return learningRate;
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
    
    public static Builder builder() {
        return new Builder();
    }
    
    public static class Builder {
        private double vigilance = 0.5;
        private double learningRate = 0.01;
        private double[] channelVigilance;
        private double[] channelLearningRates;
        
        public Builder vigilance(double vigilance) {
            if (vigilance < 0.0 || vigilance > 1.0) {
                throw new IllegalArgumentException("Vigilance must be in [0, 1]");
            }
            this.vigilance = vigilance;
            return this;
        }
        
        public Builder learningRate(double learningRate) {
            if (learningRate < 0.0 || learningRate > 1.0) {
                throw new IllegalArgumentException("Learning rate must be in [0, 1]");
            }
            this.learningRate = learningRate;
            return this;
        }
        
        public Builder channelVigilance(double[] channelVigilance) {
            this.channelVigilance = channelVigilance != null ? channelVigilance.clone() : null;
            return this;
        }
        
        public Builder channelLearningRates(double[] channelLearningRates) {
            this.channelLearningRates = channelLearningRates != null ? channelLearningRates.clone() : null;
            return this;
        }
        
        public FusionParameters build() {
            return new FusionParameters(this);
        }
    }
}