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
package com.hellblazer.art.algorithms;

import com.hellblazer.art.core.parameters.HypersphereParameters;

/**
 * Extended parameters for VectorizedHypersphereART with SIMD optimization settings.
 * 
 * Extends standard HypersphereParameters with additional configuration
 * for vectorization, parallelization, and performance optimization.
 * 
 * Key SIMD Parameters:
 * - simdThreshold: Minimum dimension for SIMD operations
 * - parallelThreshold: Minimum categories for parallel processing  
 * - parallelismLevel: Thread pool size for parallel operations
 * - enableVectorization: Master switch for SIMD operations
 * - cacheSize: Input pattern cache size for performance
 * 
 * Standard HypersphereART Parameters:
 * - vigilance: Similarity threshold for category acceptance
 * - defaultRadius: Initial radius for new hyperspheres
 * - adaptiveRadius: Whether to use adaptive radius adjustment
 * - maxCategories: Maximum number of categories allowed
 */
public record VectorizedHypersphereParameters(
    // Standard HypersphereART parameters
    double vigilance,
    double learningRate,
    int inputDimensions,
    int maxCategories,
    
    // SIMD optimization parameters
    boolean enableSIMD,
    int simdThreshold,
    int parallelismLevel,
    boolean enableCaching,
    int cacheSize,
    double expansionFactor,
    int useSIMD  // 1 for true, 0 for false to match compilation expectation
) {
    
    /**
     * Constructor with validation.
     */
    public VectorizedHypersphereParameters {
        // Validate standard parameters
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in [0,1], got: " + vigilance);
        }
        if (learningRate < 0.0 || learningRate > 1.0) {
            throw new IllegalArgumentException("Learning rate must be in [0,1], got: " + learningRate);
        }
        if (inputDimensions <= 0) {
            throw new IllegalArgumentException("Input dimensions must be positive, got: " + inputDimensions);
        }
        if (maxCategories <= 0) {
            throw new IllegalArgumentException("Max categories must be positive, got: " + maxCategories);
        }
        
        // Validate SIMD parameters
        if (simdThreshold < 1) {
            throw new IllegalArgumentException("SIMD threshold must be positive, got: " + simdThreshold);
        }
        if (parallelismLevel < 1) {
            throw new IllegalArgumentException("Parallelism level must be positive, got: " + parallelismLevel);
        }
        if (cacheSize < 0) {
            throw new IllegalArgumentException("Cache size must be non-negative, got: " + cacheSize);
        }
        if (expansionFactor <= 1.0) {
            throw new IllegalArgumentException("Expansion factor must be > 1.0, got: " + expansionFactor);
        }
    }
    
    /**
     * Create with high-performance settings optimized for large datasets.
     */
    public static VectorizedHypersphereParameters highPerformance(int inputDimensions) {
        return new VectorizedHypersphereParameters(
            0.7, // vigilance - lower for faster learning
            0.8, // learningRate - higher for faster adaptation
            inputDimensions,
            1000, // maxCategories
            true, // enableSIMD
            4,    // simdThreshold - aggressive SIMD usage
            Math.max(4, Runtime.getRuntime().availableProcessors() * 2), // parallelismLevel
            true, // enableCaching
            5000, // cacheSize - larger cache
            1.2,  // expansionFactor
            1     // useSIMD = true
        );
    }
    
    /**
     * Create with conservative settings for compatibility testing.
     */
    public static VectorizedHypersphereParameters conservative(int inputDimensions) {
        return new VectorizedHypersphereParameters(
            0.9, // vigilance - higher for more precise categories
            0.1, // learningRate - lower for stable learning
            inputDimensions,
            500, // maxCategories
            true, // enableSIMD
            16,  // simdThreshold - conservative SIMD usage
            2,   // parallelismLevel - minimal threading
            true, // enableCaching
            100, // cacheSize - small cache
            1.1, // expansionFactor
            inputDimensions >= 16 ? 1 : 0 // useSIMD based on threshold
        );
    }
    
    /**
     * Create with SIMD disabled (scalar-only) for comparison testing.
     */
    public static VectorizedHypersphereParameters scalarOnly(int inputDimensions) {
        return new VectorizedHypersphereParameters(
            0.8, // vigilance
            0.5, // learningRate
            inputDimensions,
            1000, // maxCategories
            false, // enableSIMD - disabled
            Integer.MAX_VALUE, // simdThreshold - effectively disable SIMD
            1,    // parallelismLevel - single thread
            false, // enableCaching
            0,    // cacheSize - no caching
            1.1, // expansionFactor
            0    // useSIMD = false
        );
    }
    
    /**
     * Create a builder for custom parameter configuration.
     */
    public static Builder builder() {
        return new Builder();
    }
    
    /**
     * Builder class for flexible parameter construction.
     */
    public static class Builder {
        private double vigilance = 0.8;
        private double learningRate = 0.5;
        private int inputDimensions = -1; // Must be set
        private int maxCategories = 1000;
        private boolean enableSIMD = true;
        private int simdThreshold = 8;
        private int parallelismLevel = Runtime.getRuntime().availableProcessors();
        private boolean enableCaching = true;
        private int cacheSize = 1000;
        private double expansionFactor = 1.1;
        
        public Builder vigilance(double vigilance) {
            this.vigilance = vigilance;
            return this;
        }
        
        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }
        
        public Builder inputDimensions(int inputDimensions) {
            this.inputDimensions = inputDimensions;
            return this;
        }
        
        public Builder maxCategories(int maxCategories) {
            this.maxCategories = maxCategories;
            return this;
        }
        
        public Builder enableSIMD(boolean enableSIMD) {
            this.enableSIMD = enableSIMD;
            return this;
        }
        
        public Builder simdThreshold(int simdThreshold) {
            this.simdThreshold = simdThreshold;
            return this;
        }
        
        public Builder parallelismLevel(int parallelismLevel) {
            this.parallelismLevel = parallelismLevel;
            return this;
        }
        
        public Builder enableCaching(boolean enableCaching) {
            this.enableCaching = enableCaching;
            return this;
        }
        
        public Builder cacheSize(int cacheSize) {
            this.cacheSize = cacheSize;
            return this;
        }
        
        public Builder expansionFactor(double expansionFactor) {
            this.expansionFactor = expansionFactor;
            return this;
        }
        
        public VectorizedHypersphereParameters build() {
            if (inputDimensions <= 0) {
                throw new IllegalArgumentException("Input dimensions must be positive, got: " + inputDimensions);
            }
            return new VectorizedHypersphereParameters(
                vigilance, learningRate, inputDimensions, maxCategories,
                enableSIMD, simdThreshold, parallelismLevel, enableCaching,
                cacheSize, expansionFactor, 
                (enableSIMD && inputDimensions >= simdThreshold) ? 1 : 0
            );
        }
        
        public Builder toBuilder() {
            return new Builder()
                .vigilance(vigilance)
                .learningRate(learningRate)
                .inputDimensions(inputDimensions)
                .maxCategories(maxCategories)
                .enableSIMD(enableSIMD)
                .simdThreshold(simdThreshold)
                .parallelismLevel(parallelismLevel)
                .enableCaching(enableCaching)
                .cacheSize(cacheSize)
                .expansionFactor(expansionFactor);
        }
    }
    
    /**
     * Create a copy with modified parameters.
     */
    public Builder toBuilder() {
        return new Builder()
            .vigilance(vigilance)
            .learningRate(learningRate)
            .inputDimensions(inputDimensions)
            .maxCategories(maxCategories)
            .enableSIMD(enableSIMD)
            .simdThreshold(simdThreshold)
            .parallelismLevel(parallelismLevel)
            .enableCaching(enableCaching)
            .cacheSize(cacheSize)
            .expansionFactor(expansionFactor);
    }
    
    /**
     * Check if SIMD should be used for given input dimension.
     */
    public boolean shouldUseSIMD() {
        return enableSIMD && inputDimensions >= simdThreshold;
    }
    
    /**
     * Convert to standard HypersphereParameters for compatibility.
     */
    public HypersphereParameters toStandardParameters() {
        return new HypersphereParameters(vigilance, 1.0, true);
    }
    
    @Override
    public String toString() {
        return String.format(
            "VectorizedHypersphereParameters{vigilance=%.3f, learningRate=%.3f, inputDim=%d, " +
            "maxCategories=%d, SIMD=%s(threshold=%d), parallelism=%d, cache=%d}",
            vigilance, learningRate, inputDimensions,
            maxCategories, enableSIMD, simdThreshold, parallelismLevel, cacheSize
        );
    }
}