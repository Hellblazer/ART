package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.parameters.FuzzyParameters;
import java.util.Objects;

/**
 * Immutable parameters for VectorizedSalienceART with performance optimization settings.
 * Combines salience-aware learning parameters with vectorization configuration.
 */
public record VectorizedSalienceParameters(
    double vigilance,
    double learningRate,
    double alpha,
    boolean enableSIMD,
    boolean useSparseMode,
    double sparsityThreshold,
    double salienceUpdateRate,
    SalienceCalculationType calculationType,
    boolean adaptiveSalience,
    double minimumSalience,
    double maximumSalience,
    int simdThreshold
) {
    
    /**
     * Types of salience calculation strategies
     */
    public enum SalienceCalculationType {
        FREQUENCY,          // Frequency-based salience
        STATISTICAL,        // Combined statistical measures
        INFORMATION_GAIN,   // Information-theoretic approach
        VARIANCE_BASED      // Variance reduction approach
    }
    
    /**
     * Constructor with validation
     */
    public VectorizedSalienceParameters {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in [0,1], got: " + vigilance);
        }
        if (learningRate < 0.0 || learningRate > 1.0) {
            throw new IllegalArgumentException("Learning rate must be in [0,1], got: " + learningRate);
        }
        if (alpha < 0.0) {
            throw new IllegalArgumentException("Alpha must be non-negative, got: " + alpha);
        }
        if (sparsityThreshold < 0.0 || sparsityThreshold > 1.0) {
            throw new IllegalArgumentException("Sparsity threshold must be in [0,1], got: " + sparsityThreshold);
        }
        if (salienceUpdateRate < 0.0 || salienceUpdateRate > 1.0) {
            throw new IllegalArgumentException("Salience update rate must be in [0,1], got: " + salienceUpdateRate);
        }
        if (minimumSalience < 0.0 || minimumSalience > 1.0) {
            throw new IllegalArgumentException("Minimum salience must be in [0,1], got: " + minimumSalience);
        }
        if (maximumSalience < 0.0 || maximumSalience > 1.0) {
            throw new IllegalArgumentException("Maximum salience must be in [0,1], got: " + maximumSalience);
        }
        if (minimumSalience > maximumSalience) {
            throw new IllegalArgumentException("Minimum salience cannot exceed maximum salience");
        }
        if (simdThreshold < 0) {
            throw new IllegalArgumentException("SIMD threshold must be non-negative, got: " + simdThreshold);
        }
        
        Objects.requireNonNull(calculationType, "Calculation type cannot be null");
    }
    
    /**
     * Create default parameters optimized for general use
     */
    public static VectorizedSalienceParameters createDefault() {
        return new VectorizedSalienceParameters(
            0.75,                           // vigilance
            0.1,                            // learningRate
            0.001,                          // alpha
            true,                           // enableSIMD
            true,                           // useSparseMode
            0.01,                           // sparsityThreshold
            0.01,                           // salienceUpdateRate
            SalienceCalculationType.STATISTICAL, // calculationType
            false,                          // adaptiveSalience
            0.0,                            // minimumSalience
            1.0,                            // maximumSalience
            100                             // simdThreshold
        );
    }
    
    /**
     * Create high-performance parameters for large-scale processing
     */
    public static VectorizedSalienceParameters createHighPerformance() {
        return new VectorizedSalienceParameters(
            0.8,                            // Higher vigilance for better discrimination
            0.05,                           // Lower learning rate for stability
            0.001,                          // alpha
            true,                           // enableSIMD
            true,                           // useSparseMode
            0.005,                          // Lower sparsity threshold
            0.005,                          // Faster salience updates
            SalienceCalculationType.INFORMATION_GAIN, // More sophisticated calculation
            true,                           // adaptiveSalience
            0.01,                           // minimumSalience
            0.99,                           // maximumSalience
            50                              // Lower SIMD threshold
        );
    }
    
    /**
     * Create memory-optimized parameters for resource-constrained environments
     */
    public static VectorizedSalienceParameters createMemoryOptimized() {
        return new VectorizedSalienceParameters(
            0.7,                            // Standard vigilance
            0.15,                           // Higher learning rate for faster convergence
            0.001,                          // alpha
            false,                          // Disable SIMD to save memory
            true,                           // Use sparse mode
            0.02,                           // More aggressive sparsity
            0.02,                           // Standard salience update
            SalienceCalculationType.FREQUENCY, // Simpler calculation
            false,                          // No adaptive salience
            0.0,                            // minimumSalience
            1.0,                            // maximumSalience
            200                             // Higher SIMD threshold
        );
    }
    
    /**
     * Convert to base parameters for SalienceAwareART
     */
    public FuzzyParameters toBaseParameters() {
        return new FuzzyParameters(vigilance, alpha, learningRate);
    }
    
    // Builder pattern
    public static Builder builder() {
        return new Builder();
    }
    
    public static class Builder {
        private double vigilance = 0.75;
        private double learningRate = 0.1;
        private double alpha = 0.001;
        private boolean enableSIMD = true;
        private boolean useSparseMode = true;
        private double sparsityThreshold = 0.01;
        private double salienceUpdateRate = 0.01;
        private SalienceCalculationType calculationType = SalienceCalculationType.STATISTICAL;
        private boolean adaptiveSalience = false;
        private double minimumSalience = 0.0;
        private double maximumSalience = 1.0;
        private int simdThreshold = 100;
        
        public Builder vigilance(double vigilance) {
            this.vigilance = vigilance;
            return this;
        }
        
        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }
        
        public Builder alpha(double alpha) {
            this.alpha = alpha;
            return this;
        }
        
        public Builder enableSIMD(boolean enableSIMD) {
            this.enableSIMD = enableSIMD;
            return this;
        }
        
        public Builder useSparseMode(boolean useSparseMode) {
            this.useSparseMode = useSparseMode;
            return this;
        }
        
        public Builder sparsityThreshold(double sparsityThreshold) {
            this.sparsityThreshold = sparsityThreshold;
            return this;
        }
        
        public Builder salienceUpdateRate(double salienceUpdateRate) {
            this.salienceUpdateRate = salienceUpdateRate;
            return this;
        }
        
        public Builder calculationType(SalienceCalculationType calculationType) {
            this.calculationType = calculationType;
            return this;
        }
        
        public Builder adaptiveSalience(boolean adaptiveSalience) {
            this.adaptiveSalience = adaptiveSalience;
            return this;
        }
        
        public Builder minimumSalience(double minimumSalience) {
            this.minimumSalience = minimumSalience;
            return this;
        }
        
        public Builder maximumSalience(double maximumSalience) {
            this.maximumSalience = maximumSalience;
            return this;
        }
        
        public Builder simdThreshold(int simdThreshold) {
            this.simdThreshold = simdThreshold;
            return this;
        }
        
        public VectorizedSalienceParameters build() {
            return new VectorizedSalienceParameters(
                vigilance, learningRate, alpha, enableSIMD, useSparseMode,
                sparsityThreshold, salienceUpdateRate, calculationType,
                adaptiveSalience, minimumSalience, maximumSalience, simdThreshold
            );
        }
    }
    
    // With methods for immutable updates
    public VectorizedSalienceParameters withVigilance(double newVigilance) {
        return new VectorizedSalienceParameters(
            newVigilance, learningRate, alpha, enableSIMD, useSparseMode,
            sparsityThreshold, salienceUpdateRate, calculationType,
            adaptiveSalience, minimumSalience, maximumSalience, simdThreshold
        );
    }
    
    public VectorizedSalienceParameters withLearningRate(double newLearningRate) {
        return new VectorizedSalienceParameters(
            vigilance, newLearningRate, alpha, enableSIMD, useSparseMode,
            sparsityThreshold, salienceUpdateRate, calculationType,
            adaptiveSalience, minimumSalience, maximumSalience, simdThreshold
        );
    }
    
    public VectorizedSalienceParameters withSalienceUpdateRate(double newRate) {
        return new VectorizedSalienceParameters(
            vigilance, learningRate, alpha, enableSIMD, useSparseMode,
            sparsityThreshold, newRate, calculationType,
            adaptiveSalience, minimumSalience, maximumSalience, simdThreshold
        );
    }
    
    public VectorizedSalienceParameters withUseSparseMode(boolean newUseSparseMode) {
        return new VectorizedSalienceParameters(
            vigilance, learningRate, alpha, enableSIMD, newUseSparseMode,
            sparsityThreshold, salienceUpdateRate, calculationType,
            adaptiveSalience, minimumSalience, maximumSalience, simdThreshold
        );
    }
    
    public VectorizedSalienceParameters withCalculationType(SalienceCalculationType newType) {
        return new VectorizedSalienceParameters(
            vigilance, learningRate, alpha, enableSIMD, useSparseMode,
            sparsityThreshold, salienceUpdateRate, newType,
            adaptiveSalience, minimumSalience, maximumSalience, simdThreshold
        );
    }
    
    public VectorizedSalienceParameters withEnableSIMD(boolean newEnableSIMD) {
        return new VectorizedSalienceParameters(
            vigilance, learningRate, alpha, newEnableSIMD, useSparseMode,
            sparsityThreshold, salienceUpdateRate, calculationType,
            adaptiveSalience, minimumSalience, maximumSalience, simdThreshold
        );
    }
    
    public VectorizedSalienceParameters withSparsityThreshold(double newSparsityThreshold) {
        return new VectorizedSalienceParameters(
            vigilance, learningRate, alpha, enableSIMD, useSparseMode,
            newSparsityThreshold, salienceUpdateRate, calculationType,
            adaptiveSalience, minimumSalience, maximumSalience, simdThreshold
        );
    }
}