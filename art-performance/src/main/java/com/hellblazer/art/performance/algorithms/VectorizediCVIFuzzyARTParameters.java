package com.hellblazer.art.performance.algorithms;

import java.util.Objects;

/**
 * Parameters for VectorizediCVIFuzzyART (incremental CVI FuzzyART).
 * 
 * iCVIFuzzyART combines FuzzyART with incremental Cluster Validity Index
 * monitoring for automatic vigilance adaptation and cluster quality optimization.
 */
public class VectorizediCVIFuzzyARTParameters {
    
    // Core FuzzyART parameters
    private final double vigilance;
    private final double alpha;
    private final double learningRate;
    private final boolean useComplementCoding;
    
    // CVI parameters
    private final int cviUpdateFrequency;
    private final boolean adaptiveVigilance;
    private final double minVigilance;
    private final double maxVigilance;
    private final double vigilanceStep;
    
    // Memory management
    private final int maxMemoryPatterns;
    private final int bufferSize;
    
    // Update coordination
    private final UpdateCoordination updateCoordination;
    private final boolean forceNonIncremental;
    
    // Base parameters
    private final VectorizedParameters baseParameters;
    
    public enum UpdateCoordination {
        INDEPENDENT,
        SYNCHRONIZED,
        BATCH
    }
    
    public VectorizediCVIFuzzyARTParameters(
            double vigilance,
            double alpha,
            double learningRate,
            boolean useComplementCoding,
            int cviUpdateFrequency,
            boolean adaptiveVigilance,
            double minVigilance,
            double maxVigilance,
            double vigilanceStep,
            int maxMemoryPatterns,
            int bufferSize,
            UpdateCoordination updateCoordination,
            boolean forceNonIncremental,
            VectorizedParameters baseParameters) {
        
        validateParameters(vigilance, alpha, learningRate, cviUpdateFrequency,
                         minVigilance, maxVigilance, vigilanceStep,
                         maxMemoryPatterns, bufferSize);
        
        this.vigilance = vigilance;
        this.alpha = alpha;
        this.learningRate = learningRate;
        this.useComplementCoding = useComplementCoding;
        this.cviUpdateFrequency = cviUpdateFrequency;
        this.adaptiveVigilance = adaptiveVigilance;
        this.minVigilance = minVigilance;
        this.maxVigilance = maxVigilance;
        this.vigilanceStep = vigilanceStep;
        this.maxMemoryPatterns = maxMemoryPatterns;
        this.bufferSize = bufferSize;
        this.updateCoordination = Objects.requireNonNull(updateCoordination, "Update coordination cannot be null");
        this.forceNonIncremental = forceNonIncremental;
        this.baseParameters = Objects.requireNonNull(baseParameters, "Base parameters cannot be null");
    }
    
    /**
     * Create default iCVIFuzzyART parameters.
     */
    public VectorizediCVIFuzzyARTParameters() {
        this(0.75, 0.001, 1.0, true, 10, true, 0.3, 0.95, 0.05,
             10000, 1000, UpdateCoordination.INDEPENDENT, false,
             VectorizedParameters.createDefault());
    }
    
    /**
     * Create parameters optimized for streaming data.
     */
    public static VectorizediCVIFuzzyARTParameters forStreaming(int bufferSize) {
        return new VectorizediCVIFuzzyARTParameters(
            0.7, 0.001, 0.8, true, 5, true, 0.4, 0.9, 0.05,
            bufferSize * 10, bufferSize, UpdateCoordination.BATCH, false,
            VectorizedParameters.createDefault()
        );
    }
    
    private static void validateParameters(double vigilance, double alpha,
                                          double learningRate, int cviUpdateFrequency,
                                          double minVigilance, double maxVigilance,
                                          double vigilanceStep, int maxMemoryPatterns,
                                          int bufferSize) {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in [0, 1], got: " + vigilance);
        }
        if (alpha < 0.0) {
            throw new IllegalArgumentException("Alpha must be non-negative, got: " + alpha);
        }
        if (learningRate < 0.0 || learningRate > 1.0) {
            throw new IllegalArgumentException("Learning rate must be in [0, 1], got: " + learningRate);
        }
        if (cviUpdateFrequency < 1) {
            throw new IllegalArgumentException("CVI update frequency must be >= 1, got: " + cviUpdateFrequency);
        }
        if (minVigilance < 0.0 || minVigilance > 1.0) {
            throw new IllegalArgumentException("Min vigilance must be in [0, 1], got: " + minVigilance);
        }
        if (maxVigilance < 0.0 || maxVigilance > 1.0) {
            throw new IllegalArgumentException("Max vigilance must be in [0, 1], got: " + maxVigilance);
        }
        if (minVigilance > maxVigilance) {
            throw new IllegalArgumentException("Min vigilance must be <= max vigilance");
        }
        if (vigilanceStep < 0.0 || vigilanceStep > 1.0) {
            throw new IllegalArgumentException("Vigilance step must be in [0, 1], got: " + vigilanceStep);
        }
        if (maxMemoryPatterns < 1) {
            throw new IllegalArgumentException("Max memory patterns must be >= 1, got: " + maxMemoryPatterns);
        }
        if (bufferSize < 1) {
            throw new IllegalArgumentException("Buffer size must be >= 1, got: " + bufferSize);
        }
    }
    
    // Getters
    
    public double getVigilance() {
        return vigilance;
    }
    
    public double vigilanceThreshold() {
        return vigilance;
    }
    
    public double getAlpha() {
        return alpha;
    }
    
    public double getLearningRate() {
        return learningRate;
    }
    
    public boolean isUseComplementCoding() {
        return useComplementCoding;
    }
    
    public int getCviUpdateFrequency() {
        return cviUpdateFrequency;
    }
    
    public boolean isAdaptiveVigilance() {
        return adaptiveVigilance;
    }
    
    public double getMinVigilance() {
        return minVigilance;
    }
    
    public double getMaxVigilance() {
        return maxVigilance;
    }
    
    public double getVigilanceStep() {
        return vigilanceStep;
    }
    
    public int getMaxMemoryPatterns() {
        return maxMemoryPatterns;
    }
    
    public int getBufferSize() {
        return bufferSize;
    }
    
    public UpdateCoordination getUpdateCoordination() {
        return updateCoordination;
    }
    
    public boolean isForceNonIncremental() {
        return forceNonIncremental;
    }
    
    public VectorizedParameters getBaseParameters() {
        return baseParameters;
    }
    
    /**
     * Calculate adapted vigilance based on CVI feedback.
     */
    public double calculateAdaptedVigilance(double cviScore, double currentVigilance) {
        if (!adaptiveVigilance) {
            return currentVigilance;
        }
        
        // Higher CVI scores suggest better clustering - can lower vigilance
        // Lower CVI scores suggest poor clustering - should raise vigilance
        double targetVigilance = currentVigilance;
        
        if (cviScore < 0.5) {
            // Poor clustering - increase vigilance
            targetVigilance = Math.min(maxVigilance, currentVigilance + vigilanceStep);
        } else if (cviScore > 0.8) {
            // Good clustering - can decrease vigilance
            targetVigilance = Math.max(minVigilance, currentVigilance - vigilanceStep);
        }
        
        return targetVigilance;
    }
    
    /**
     * Check if CVI should be updated based on pattern count.
     */
    public boolean shouldUpdateCVI(int patternsSinceLastUpdate) {
        return patternsSinceLastUpdate >= cviUpdateFrequency;
    }
    
    @Override
    public String toString() {
        return String.format("VectorizediCVIFuzzyARTParameters{vigilance=%.3f, alpha=%.4f, " +
                           "learningRate=%.3f, complement=%b, cviFreq=%d, adaptive=%b, " +
                           "maxPatterns=%d}",
                           vigilance, alpha, learningRate, useComplementCoding,
                           cviUpdateFrequency, adaptiveVigilance, maxMemoryPatterns);
    }
}