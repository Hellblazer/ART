package com.hellblazer.art.performance.algorithms;

import java.util.Objects;

/**
 * Parameters for VectorizedARTSTAR (ART with STability and Adaptability Regulation).
 * 
 * ARTSTAR provides automatic regulation of stability vs adaptability through:
 * - Dynamic vigilance adjustment
 * - Category strength tracking with decay
 * - Automatic pruning of weak categories  
 * - Stability/adaptability balance monitoring
 * - Regulated learning with performance tracking
 */
public class VectorizedARTSTARParameters {
    
    // Core ART parameters
    private final double baseVigilance;
    private final double alpha;
    private final double learningRate;
    
    // Stability/Adaptability regulation
    private final double stabilityBias;
    private final double adaptabilityBias;
    private final double regulationRate;
    private final double decayRate;
    
    // Category management
    private final double pruningThreshold;
    private final int maxCategories;
    private final int minCategoryAge;
    
    // Dynamic vigilance range
    private final double minVigilance;
    private final double maxVigilance;
    private final double vigilanceAdjustmentRate;
    
    // Performance monitoring
    private final int performanceWindowSize;
    private final double targetSuccessRate;
    
    // Base parameters
    private final VectorizedParameters baseParameters;
    
    public VectorizedARTSTARParameters(
            double baseVigilance,
            double alpha,
            double learningRate,
            double stabilityBias,
            double adaptabilityBias,
            double regulationRate,
            double decayRate,
            double pruningThreshold,
            int maxCategories,
            int minCategoryAge,
            double minVigilance,
            double maxVigilance,
            double vigilanceAdjustmentRate,
            int performanceWindowSize,
            double targetSuccessRate,
            VectorizedParameters baseParameters) {
        
        validateParameters(baseVigilance, alpha, learningRate, stabilityBias,
                         adaptabilityBias, regulationRate, decayRate,
                         pruningThreshold, maxCategories, minCategoryAge,
                         minVigilance, maxVigilance, vigilanceAdjustmentRate,
                         performanceWindowSize, targetSuccessRate);
        
        this.baseVigilance = baseVigilance;
        this.alpha = alpha;
        this.learningRate = learningRate;
        this.stabilityBias = stabilityBias;
        this.adaptabilityBias = adaptabilityBias;
        this.regulationRate = regulationRate;
        this.decayRate = decayRate;
        this.pruningThreshold = pruningThreshold;
        this.maxCategories = maxCategories;
        this.minCategoryAge = minCategoryAge;
        this.minVigilance = minVigilance;
        this.maxVigilance = maxVigilance;
        this.vigilanceAdjustmentRate = vigilanceAdjustmentRate;
        this.performanceWindowSize = performanceWindowSize;
        this.targetSuccessRate = targetSuccessRate;
        this.baseParameters = Objects.requireNonNull(baseParameters, "Base parameters cannot be null");
    }
    
    /**
     * Create default ARTSTAR parameters.
     */
    public VectorizedARTSTARParameters() {
        this(0.7, 0.001, 0.5, 0.5, 0.5, 0.1, 0.01,
             0.1, 100, 10, 0.3, 0.95, 0.05,
             50, 0.8, VectorizedParameters.createDefault());
    }
    
    /**
     * Create ARTSTAR parameters with specific base vigilance.
     */
    public static VectorizedARTSTARParameters withVigilance(double vigilance) {
        return new VectorizedARTSTARParameters(
            vigilance, 0.001, 0.5, 0.5, 0.5, 0.1, 0.01,
            0.1, 100, 10, Math.max(0.0, vigilance - 0.4),
            Math.min(1.0, vigilance + 0.25), 0.05,
            50, 0.8, VectorizedParameters.createDefault()
        );
    }
    
    private static void validateParameters(double baseVigilance, double alpha, 
                                          double learningRate, double stabilityBias,
                                          double adaptabilityBias, double regulationRate,
                                          double decayRate, double pruningThreshold,
                                          int maxCategories, int minCategoryAge,
                                          double minVigilance, double maxVigilance,
                                          double vigilanceAdjustmentRate,
                                          int performanceWindowSize, double targetSuccessRate) {
        if (baseVigilance < 0.0 || baseVigilance > 1.0) {
            throw new IllegalArgumentException("Base vigilance must be in [0, 1], got: " + baseVigilance);
        }
        if (alpha < 0.0) {
            throw new IllegalArgumentException("Alpha must be non-negative, got: " + alpha);
        }
        if (learningRate < 0.0 || learningRate > 1.0) {
            throw new IllegalArgumentException("Learning rate must be in [0, 1], got: " + learningRate);
        }
        if (stabilityBias < 0.0 || stabilityBias > 1.0) {
            throw new IllegalArgumentException("Stability bias must be in [0, 1], got: " + stabilityBias);
        }
        if (adaptabilityBias < 0.0 || adaptabilityBias > 1.0) {
            throw new IllegalArgumentException("Adaptability bias must be in [0, 1], got: " + adaptabilityBias);
        }
        if (regulationRate < 0.0 || regulationRate > 1.0) {
            throw new IllegalArgumentException("Regulation rate must be in [0, 1], got: " + regulationRate);
        }
        if (decayRate < 0.0 || decayRate > 1.0) {
            throw new IllegalArgumentException("Decay rate must be in [0, 1], got: " + decayRate);
        }
        if (pruningThreshold < 0.0 || pruningThreshold > 1.0) {
            throw new IllegalArgumentException("Pruning threshold must be in [0, 1], got: " + pruningThreshold);
        }
        if (maxCategories < 1) {
            throw new IllegalArgumentException("Max categories must be >= 1, got: " + maxCategories);
        }
        if (minCategoryAge < 0) {
            throw new IllegalArgumentException("Min category age must be >= 0, got: " + minCategoryAge);
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
        if (vigilanceAdjustmentRate < 0.0 || vigilanceAdjustmentRate > 1.0) {
            throw new IllegalArgumentException("Vigilance adjustment rate must be in [0, 1], got: " + vigilanceAdjustmentRate);
        }
        if (performanceWindowSize < 1) {
            throw new IllegalArgumentException("Performance window size must be >= 1, got: " + performanceWindowSize);
        }
        if (targetSuccessRate < 0.0 || targetSuccessRate > 1.0) {
            throw new IllegalArgumentException("Target success rate must be in [0, 1], got: " + targetSuccessRate);
        }
    }
    
    // Getters
    
    public double getBaseVigilance() {
        return baseVigilance;
    }
    
    public double vigilanceThreshold() {
        return baseVigilance;
    }
    
    public double getAlpha() {
        return alpha;
    }
    
    public double getLearningRate() {
        return learningRate;
    }
    
    public double getStabilityBias() {
        return stabilityBias;
    }
    
    public double getAdaptabilityBias() {
        return adaptabilityBias;
    }
    
    public double getRegulationRate() {
        return regulationRate;
    }
    
    public double getDecayRate() {
        return decayRate;
    }
    
    public double getPruningThreshold() {
        return pruningThreshold;
    }
    
    public int getMaxCategories() {
        return maxCategories;
    }
    
    public int getMinCategoryAge() {
        return minCategoryAge;
    }
    
    public double getMinVigilance() {
        return minVigilance;
    }
    
    public double getMaxVigilance() {
        return maxVigilance;
    }
    
    public double getVigilanceAdjustmentRate() {
        return vigilanceAdjustmentRate;
    }
    
    public int getPerformanceWindowSize() {
        return performanceWindowSize;
    }
    
    public double getTargetSuccessRate() {
        return targetSuccessRate;
    }
    
    public VectorizedParameters getBaseParameters() {
        return baseParameters;
    }
    
    /**
     * Calculate dynamic vigilance based on network state.
     */
    public double calculateDynamicVigilance(double stabilityLevel, double adaptabilityNeed) {
        double vigilanceAdjustment = (stabilityLevel - adaptabilityNeed) * vigilanceAdjustmentRate;
        double dynamicVigilance = baseVigilance + vigilanceAdjustment;
        return Math.max(minVigilance, Math.min(maxVigilance, dynamicVigilance));
    }
    
    /**
     * Calculate regulated learning rate based on stability/adaptability.
     */
    public double calculateRegulatedLearningRate(double stability, double adaptability) {
        double regulationFactor = 1.0 + (adaptability - stability) * regulationRate;
        return Math.max(0.0, Math.min(1.0, learningRate * regulationFactor));
    }
    
    /**
     * Check if category should be pruned.
     */
    public boolean shouldPruneCategory(double categoryStrength, int categoryAge) {
        return categoryStrength < pruningThreshold && categoryAge >= minCategoryAge;
    }
    
    /**
     * Convert to core ARTSTARParameters.
     */
    public com.hellblazer.art.core.parameters.ARTSTARParameters toParameters() {
        // Map our extended parameters to core ARTSTAR parameters (10 parameters)
        return new com.hellblazer.art.core.parameters.ARTSTARParameters(
            baseVigilance,                // vigilance
            alpha,                         // alpha (choice parameter)
            learningRate,                  // beta (learning rate)
            stabilityBias,                 // stabilityFactor
            adaptabilityBias,              // adaptabilityFactor
            regulationRate,                // regulationRate
            decayRate,                     // categoryDecayRate
            vigilanceAdjustmentRate,       // vigilanceRange
            pruningThreshold,              // minCategoryStrength
            maxCategories                  // maxCategories
        );
    }
    
    @Override
    public String toString() {
        return String.format("VectorizedARTSTARParameters{vigilance=%.3f, alpha=%.4f, " +
                           "learningRate=%.3f, stability=%.3f, adaptability=%.3f, " +
                           "regulation=%.3f, decay=%.3f, maxCategories=%d}",
                           baseVigilance, alpha, learningRate, stabilityBias,
                           adaptabilityBias, regulationRate, decayRate, maxCategories);
    }
}