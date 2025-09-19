package com.hellblazer.art.performance.algorithms;

/**
 * Parameters for VectorizedART1 binary pattern recognition algorithm.
 * 
 * VectorizedART1 extends the classic ART1 algorithm with performance optimizations
 * including SIMD vectorization and parallel processing while maintaining full
 * compatibility with binary pattern recognition semantics.
 * 
 * Parameter validation ensures algorithm correctness and prevents common configuration errors.
 */
public record VectorizedART1Parameters(
    double vigilance,           // Vigilance parameter [0, 1] - controls category selectivity
    double L,                   // Uncommitted node bias >= 1.0 - affects choice function
    int parallelismLevel,       // Number of threads for parallel processing >= 1
    int parallelThreshold,      // Minimum categories to trigger parallel processing >= 0
    int maxCacheSize,          // Maximum input cache size >= 0
    boolean enableSIMD         // Enable SIMD vectorization optimizations
) {
    
    /**
     * Constructor with comprehensive parameter validation.
     * 
     * @param vigilance Vigilance parameter - must be in range [0, 1]
     * @param L Uncommitted node bias - must be >= 1.0
     * @param parallelismLevel Number of parallel threads - must be >= 1
     * @param parallelThreshold Categories threshold for parallel processing - must be >= 0
     * @param maxCacheSize Maximum cache size - must be >= 0
     * @param enableSIMD Whether to enable SIMD optimizations
     * 
     * @throws IllegalArgumentException if any parameter is outside valid range
     */
    public VectorizedART1Parameters {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException(
                String.format("Vigilance must be in range [0, 1], got: %.3f", vigilance));
        }
        
        if (L < 1.0) {
            throw new IllegalArgumentException(
                String.format("L parameter must be >= 1.0 for ART1 algorithm, got: %.3f", L));
        }
        
        if (parallelismLevel < 1) {
            throw new IllegalArgumentException(
                String.format("Parallelism level must be >= 1, got: %d", parallelismLevel));
        }
        
        if (parallelThreshold < 0) {
            throw new IllegalArgumentException(
                String.format("Parallel threshold must be >= 0, got: %d", parallelThreshold));
        }
        
        if (maxCacheSize < 0) {
            throw new IllegalArgumentException(
                String.format("Max cache size must be >= 0, got: %d", maxCacheSize));
        }
    }
    
    /**
     * Create default parameters suitable for most binary pattern recognition tasks.
     * Alias for defaultParameters() to match test expectations.
     * 
     * @return VectorizedART1Parameters with default values
     */
    public static VectorizedART1Parameters createDefault() {
        return defaultParameters();
    }
    
    /**
     * Create parameters with a specific vigilance value using default settings for other parameters.
     * 
     * @param vigilance Vigilance parameter [0, 1]
     * @return VectorizedART1Parameters with specified vigilance and default other values
     */
    public static VectorizedART1Parameters createWithVigilance(double vigilance) {
        return new VectorizedART1Parameters(
            vigilance,                                   // custom vigilance
            2.0,                                         // default L
            Runtime.getRuntime().availableProcessors(), // default parallelismLevel
            100,                                         // default parallelThreshold
            1000,                                        // default maxCacheSize
            true                                         // default enableSIMD
        );
    }
    
    /**
     * Create default parameters suitable for most binary pattern recognition tasks.
     * 
     * Default configuration:
     * - vigilance: 0.75 (moderate selectivity)
     * - L: 2.0 (standard ART1 choice parameter)
     * - parallelismLevel: available processors
     * - parallelThreshold: 100 categories
     * - maxCacheSize: 1000 patterns
     * - enableSIMD: true
     * 
     * @return VectorizedART1Parameters with default values
     */
    public static VectorizedART1Parameters defaultParameters() {
        return new VectorizedART1Parameters(
            0.75,                                    // vigilance
            2.0,                                     // L
            Runtime.getRuntime().availableProcessors(), // parallelismLevel
            100,                                     // parallelThreshold
            1000,                                    // maxCacheSize
            true                                     // enableSIMD
        );
    }
    
    /**
     * Create parameters optimized for high performance on large datasets.
     * 
     * Performance configuration:
     * - Lower parallel threshold for earlier parallel activation
     * - Larger cache for better reuse
     * - Maximum available parallelism
     * - SIMD enabled
     * 
     * @param vigilance Vigilance parameter [0, 1]
     * @param L Choice parameter >= 1.0
     * @return VectorizedART1Parameters optimized for performance
     */
    public static VectorizedART1Parameters highPerformance(double vigilance, double L) {
        return new VectorizedART1Parameters(
            vigilance,
            L,
            Math.max(4, Runtime.getRuntime().availableProcessors()), // At least 4 threads
            10,                                      // Early parallel activation
            5000,                                    // Large cache
            true                                     // SIMD enabled
        );
    }
    
    /**
     * Create parameters optimized for memory efficiency.
     * 
     * Memory-efficient configuration:
     * - Smaller cache to reduce memory usage
     * - Higher parallel threshold to avoid thread overhead
     * - Minimal parallelism for small datasets
     * 
     * @param vigilance Vigilance parameter [0, 1]
     * @param L Choice parameter >= 1.0
     * @return VectorizedART1Parameters optimized for memory efficiency
     */
    public static VectorizedART1Parameters memoryEfficient(double vigilance, double L) {
        return new VectorizedART1Parameters(
            vigilance,
            L,
            2,                                       // Minimal parallelism
            500,                                     // Higher threshold
            100,                                     // Small cache
            true                                     // SIMD still beneficial
        );
    }
    
    /**
     * Create parameters for exact compatibility with original ART1.
     * 
     * Compatibility configuration:
     * - Single-threaded execution
     * - SIMD disabled for exact numerical matching
     * - Minimal caching
     * 
     * @param vigilance Vigilance parameter [0, 1]
     * @param L Choice parameter >= 1.0
     * @return VectorizedART1Parameters for ART1 compatibility
     */
    public static VectorizedART1Parameters compatibilityMode(double vigilance, double L) {
        return new VectorizedART1Parameters(
            vigilance,
            L,
            1,                                       // Single-threaded
            Integer.MAX_VALUE,                       // Disable parallel processing
            0,                                       // No caching
            false                                    // No SIMD
        );
    }
    
    /**
     * Create a builder for customized parameter construction.
     * 
     * @return VectorizedART1ParametersBuilder for fluent configuration
     */
    public static Builder builder() {
        return new Builder();
    }
    
    /**
     * Get a performance summary string describing the configuration.
     * 
     * @return human-readable parameter summary
     */
    public String getConfigurationSummary() {
        return String.format(
            "VectorizedART1[vigilance=%.3f, L=%.1f, threads=%d, parallel@%d, cache=%d, SIMD=%s]",
            vigilance, L, parallelismLevel, parallelThreshold, maxCacheSize, enableSIMD
        );
    }
    
    /**
     * Check if this configuration will trigger parallel processing for a given category count.
     * 
     * @param categoryCount Number of categories in the network
     * @return true if parallel processing will be used
     */
    public boolean willUseParallelProcessing(int categoryCount) {
        return categoryCount >= parallelThreshold && parallelismLevel > 1;
    }
    
    /**
     * Check if SIMD optimizations are enabled and will be used.
     * 
     * @return true if SIMD vectorization is enabled
     */
    public boolean willUseSIMD() {
        return enableSIMD;
    }
    
    /**
     * Create a copy of these parameters with a different vigilance value.
     * 
     * @param newVigilance New vigilance parameter [0, 1]
     * @return VectorizedART1Parameters with updated vigilance
     * @throws IllegalArgumentException if vigilance is out of range
     */
    public VectorizedART1Parameters withVigilance(double newVigilance) {
        return new VectorizedART1Parameters(newVigilance, L, parallelismLevel, 
                                           parallelThreshold, maxCacheSize, enableSIMD);
    }
    
    /**
     * Create a copy of these parameters with a different L value.
     * 
     * @param newL New L parameter >= 1.0
     * @return VectorizedART1Parameters with updated L
     * @throws IllegalArgumentException if L is < 1.0
     */
    public VectorizedART1Parameters withL(double newL) {
        return new VectorizedART1Parameters(vigilance, newL, parallelismLevel,
                                           parallelThreshold, maxCacheSize, enableSIMD);
    }
    
    /**
     * Create a copy of these parameters with SIMD enabled or disabled.
     * 
     * @param simdEnabled Whether to enable SIMD optimizations
     * @return VectorizedART1Parameters with updated SIMD setting
     */
    public VectorizedART1Parameters withSIMD(boolean simdEnabled) {
        return new VectorizedART1Parameters(vigilance, L, parallelismLevel,
                                           parallelThreshold, maxCacheSize, simdEnabled);
    }
    
    /**
     * Builder pattern for VectorizedART1Parameters construction.
     * Provides fluent API for parameter configuration with validation.
     */
    public static class Builder {
        private double vigilance = 0.75;
        private double L = 2.0;
        private int parallelismLevel = Runtime.getRuntime().availableProcessors();
        private int parallelThreshold = 100;
        private int maxCacheSize = 1000;
        private boolean enableSIMD = true;
        
        /**
         * Set the vigilance parameter.
         * 
         * @param vigilance Vigilance value [0, 1]
         * @return this builder
         */
        public Builder vigilance(double vigilance) {
            this.vigilance = vigilance;
            return this;
        }
        
        /**
         * Set the L parameter (uncommitted node bias).
         * 
         * @param L L value >= 1.0
         * @return this builder
         */
        public Builder L(double L) {
            this.L = L;
            return this;
        }
        
        /**
         * Set the number of parallel threads.
         * 
         * @param parallelismLevel Number of threads >= 1
         * @return this builder
         */
        public Builder parallelismLevel(int parallelismLevel) {
            this.parallelismLevel = parallelismLevel;
            return this;
        }
        
        /**
         * Set the threshold for enabling parallel processing.
         * 
         * @param parallelThreshold Minimum categories for parallel processing >= 0
         * @return this builder
         */
        public Builder parallelThreshold(int parallelThreshold) {
            this.parallelThreshold = parallelThreshold;
            return this;
        }
        
        /**
         * Set the maximum cache size.
         * 
         * @param maxCacheSize Maximum cached patterns >= 0
         * @return this builder
         */
        public Builder maxCacheSize(int maxCacheSize) {
            this.maxCacheSize = maxCacheSize;
            return this;
        }
        
        /**
         * Enable or disable SIMD optimizations.
         * 
         * @param enableSIMD Whether to use SIMD vectorization
         * @return this builder
         */
        public Builder enableSIMD(boolean enableSIMD) {
            this.enableSIMD = enableSIMD;
            return this;
        }
        
        /**
         * Build the VectorizedART1Parameters with current configuration.
         * 
         * @return Validated VectorizedART1Parameters instance
         * @throws IllegalArgumentException if any parameter is invalid
         */
        public VectorizedART1Parameters build() {
            return new VectorizedART1Parameters(vigilance, L, parallelismLevel,
                                               parallelThreshold, maxCacheSize, enableSIMD);
        }
    }
    
    @Override
    public String toString() {
        return getConfigurationSummary();
    }
}