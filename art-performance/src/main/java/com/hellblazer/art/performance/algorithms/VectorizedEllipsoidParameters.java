package com.hellblazer.art.performance.algorithms;

/**
 * Parameters for VectorizedEllipsoidART with ellipsoidal category representation.
 * 
 * EllipsoidART uses ellipsoidal geometry for category boundaries rather than
 * hyperrectangles (FuzzyART) or hyperspheres (HypersphereART). This provides
 * more flexible category shapes that can adapt to data distributions.
 * 
 * Key Parameters:
 * - vigilance: Controls category granularity (higher = more categories)
 * - mu: Shape parameter controlling ellipsoid orientation (0 = elongated, 1 = circular)
 * - baseRadius: Base radius for ellipsoidal boundaries
 * 
 * Performance Parameters:
 * - parallelismLevel: Number of parallel threads for large category sets
 * - parallelThreshold: Minimum categories before enabling parallelism
 * - maxCacheSize: Maximum cached input patterns for SIMD optimization
 * - enableSIMD: Enable SIMD vectorization for distance calculations
 */
public record VectorizedEllipsoidParameters(
    double vigilance,
    double mu,
    double baseRadius,
    int parallelismLevel,
    int parallelThreshold,
    int maxCacheSize,
    boolean enableSIMD
) {
    
    /**
     * Create VectorizedEllipsoidParameters with validation.
     */
    public VectorizedEllipsoidParameters {
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in [0,1], got: " + vigilance);
        }
        if (mu < 0.0 || mu > 1.0) {
            throw new IllegalArgumentException("Mu must be in [0,1], got: " + mu);
        }
        if (baseRadius <= 0.0) {
            throw new IllegalArgumentException("Base radius must be > 0, got: " + baseRadius);
        }
        if (parallelismLevel < 1) {
            throw new IllegalArgumentException("Parallelism level must be >= 1, got: " + parallelismLevel);
        }
        if (parallelThreshold < 1) {
            throw new IllegalArgumentException("Parallel threshold must be >= 1, got: " + parallelThreshold);
        }
        if (maxCacheSize < 0) {
            throw new IllegalArgumentException("Max cache size must be >= 0, got: " + maxCacheSize);
        }
    }
    
    /**
     * Create default parameters optimized for typical ellipsoidal clustering.
     */
    public static VectorizedEllipsoidParameters createDefault() {
        int processors = Runtime.getRuntime().availableProcessors();
        return new VectorizedEllipsoidParameters(
            0.75,                      // vigilance - moderate granularity
            0.9,                       // mu - nearly circular ellipsoids
            1.0,                       // baseRadius - unit radius
            Math.max(2, processors/2), // parallelismLevel
            50,                        // parallelThreshold
            1000,                      // maxCacheSize
            true                       // enableSIMD
        );
    }
    
    /**
     * Create parameters for high-precision clustering (more categories).
     */
    public static VectorizedEllipsoidParameters createHighPrecision() {
        int processors = Runtime.getRuntime().availableProcessors();
        return new VectorizedEllipsoidParameters(
            0.9,                       // High vigilance for fine clustering
            0.95,                      // Nearly circular for precision
            0.8,                       // Smaller radius for tighter clusters
            processors,                // Use all cores
            20,                        // Lower threshold for parallelism
            2000,                      // Larger cache
            true                       // enableSIMD
        );
    }
    
    /**
     * Create parameters for flexible clustering (elongated ellipsoids).
     */
    public static VectorizedEllipsoidParameters createFlexibleClustering() {
        return new VectorizedEllipsoidParameters(
            0.6,                       // Lower vigilance for fewer categories
            0.3,                       // Low mu for elongated ellipsoids
            1.5,                       // Larger radius for flexibility
            4,                         // Moderate parallelism
            100,                       // Higher threshold
            500,                       // Moderate cache
            true                       // enableSIMD
        );
    }
    
    /**
     * Create memory-optimized parameters for resource-constrained environments.
     */
    public static VectorizedEllipsoidParameters createMemoryOptimized() {
        return new VectorizedEllipsoidParameters(
            0.7,                       // Standard vigilance
            0.8,                       // Moderate mu
            1.0,                       // Standard radius
            2,                         // Minimal parallelism
            200,                       // High threshold for parallelism
            100,                       // Small cache
            false                      // Disable SIMD to save memory
        );
    }
    
    /**
     * Create real-time parameters optimized for low-latency processing.
     */
    public static VectorizedEllipsoidParameters createRealTime() {
        return new VectorizedEllipsoidParameters(
            0.75,                      // Standard vigilance
            0.9,                       // Nearly circular for speed
            1.0,                       // Standard radius
            4,                         // Moderate parallelism
            10,                        // Very low threshold
            500,                       // Moderate cache
            true                       // enableSIMD for speed
        );
    }
    
    /**
     * Create a new VectorizedEllipsoidParameters with modified vigilance.
     */
    public VectorizedEllipsoidParameters withVigilance(double newVigilance) {
        return new VectorizedEllipsoidParameters(
            newVigilance, mu, baseRadius, parallelismLevel, 
            parallelThreshold, maxCacheSize, enableSIMD
        );
    }
    
    /**
     * Create a new VectorizedEllipsoidParameters with modified mu (shape parameter).
     */
    public VectorizedEllipsoidParameters withMu(double newMu) {
        return new VectorizedEllipsoidParameters(
            vigilance, newMu, baseRadius, parallelismLevel,
            parallelThreshold, maxCacheSize, enableSIMD
        );
    }
    
    /**
     * Create a new VectorizedEllipsoidParameters with modified base radius.
     */
    public VectorizedEllipsoidParameters withBaseRadius(double newBaseRadius) {
        return new VectorizedEllipsoidParameters(
            vigilance, mu, newBaseRadius, parallelismLevel,
            parallelThreshold, maxCacheSize, enableSIMD
        );
    }
    
    /**
     * Create a new VectorizedEllipsoidParameters with modified parallelism settings.
     */
    public VectorizedEllipsoidParameters withParallelism(int newParallelismLevel, int newParallelThreshold) {
        return new VectorizedEllipsoidParameters(
            vigilance, mu, baseRadius, newParallelismLevel,
            newParallelThreshold, maxCacheSize, enableSIMD
        );
    }
    
    /**
     * Create a new VectorizedEllipsoidParameters with modified cache settings.
     */
    public VectorizedEllipsoidParameters withCacheSettings(int newMaxCacheSize, boolean newEnableSIMD) {
        return new VectorizedEllipsoidParameters(
            vigilance, mu, baseRadius, parallelismLevel,
            parallelThreshold, newMaxCacheSize, newEnableSIMD
        );
    }
    
    /**
     * Get the effective ellipse eccentricity based on mu parameter.
     * Higher mu means more circular (lower eccentricity).
     */
    public double getEccentricity() {
        return Math.sqrt(1.0 - mu * mu);
    }
    
    /**
     * Get the semi-major axis length.
     */
    public double getSemiMajorAxis() {
        return baseRadius / Math.sqrt(mu);
    }
    
    /**
     * Get the semi-minor axis length.
     */
    public double getSemiMinorAxis() {
        return baseRadius * Math.sqrt(mu);
    }
    
    /**
     * Check if ellipsoids are nearly circular (mu > 0.9).
     */
    public boolean isNearlyCircular() {
        return mu > 0.9;
    }
    
    /**
     * Check if ellipsoids are highly elongated (mu < 0.3).
     */
    public boolean isHighlyElongated() {
        return mu < 0.3;
    }
    
    /**
     * Get estimated memory usage for given problem size.
     */
    public long getEstimatedMemoryUsage(int expectedCategories, int inputDimension) {
        long baseUsage = 1000L;
        // Each category stores center + covariance matrix
        long categoryUsage = expectedCategories * (inputDimension + inputDimension * inputDimension) * 8L;
        long cacheUsage = maxCacheSize * inputDimension * 4L; // float arrays for SIMD
        return baseUsage + categoryUsage + cacheUsage;
    }
    
    /**
     * Check if configuration is suitable for given problem size.
     */
    public boolean isSuitableFor(int expectedCategories, int inputDimension) {
        long memoryUsage = getEstimatedMemoryUsage(expectedCategories, inputDimension);
        long availableMemory = Runtime.getRuntime().maxMemory();
        return memoryUsage < (availableMemory * 0.8); // Use 80% threshold
    }
    
    /**
     * Get recommended configuration for problem characteristics.
     */
    public static VectorizedEllipsoidParameters recommend(int expectedCategories, int inputDimension, String priority) {
        return switch (priority.toLowerCase()) {
            case "precision", "accuracy" -> {
                var params = createHighPrecision();
                if (!params.isSuitableFor(expectedCategories, inputDimension)) {
                    yield params.withCacheSettings(params.maxCacheSize / 2, true);
                }
                yield params;
            }
            case "flexibility", "adaptive" -> createFlexibleClustering();
            case "memory" -> createMemoryOptimized();
            case "speed", "realtime" -> {
                var params = createRealTime();
                if (!params.isSuitableFor(expectedCategories, inputDimension)) {
                    yield params.withCacheSettings(100, false);
                }
                yield params;
            }
            default -> createDefault();
        };
    }
    
    @Override
    public String toString() {
        return String.format("VectorizedEllipsoidParameters{vigilance=%.3f, mu=%.3f, baseRadius=%.3f, " +
                           "parallel=%d/%d, cache=%d, simd=%s, eccentricity=%.3f}",
                           vigilance, mu, baseRadius, parallelismLevel, parallelThreshold, 
                           maxCacheSize, enableSIMD, getEccentricity());
    }
}