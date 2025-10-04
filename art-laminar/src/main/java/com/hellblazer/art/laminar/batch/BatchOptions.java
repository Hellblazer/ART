package com.hellblazer.art.laminar.batch;

/**
 * Configuration options for batch processing operations.
 *
 * <p>Controls various optimization strategies for batch processing including:
 * <ul>
 *   <li>Parallel processing across batch</li>
 *   <li>SIMD vectorization across batch dimension</li>
 *   <li>Cache optimization strategies</li>
 *   <li>Performance tracking detail level</li>
 * </ul>
 *
 * <h2>Preset Configurations</h2>
 * <ul>
 *   <li>{@link #defaults()}: Balanced performance and overhead</li>
 *   <li>{@link #throughput()}: Maximum throughput for large batches</li>
 *   <li>{@link #profiling()}: Detailed tracking for debugging</li>
 * </ul>
 *
 * @param enableParallelism enable parallel processing across batch patterns
 * @param maxParallelism maximum concurrent threads (0 = auto-detect based on CPU cores)
 * @param enableSIMDAcrossBatch enable SIMD vectorization across batch dimension
 * @param cacheOptimize optimize memory layout for cache locality
 * @param trackDetailedStats collect detailed per-layer performance metrics
 *
 * @author Hal Hildebrand
 */
public record BatchOptions(
    boolean enableParallelism,
    int maxParallelism,
    boolean enableSIMDAcrossBatch,
    boolean cacheOptimize,
    boolean trackDetailedStats
) {
    /**
     * Default options optimized for balanced performance.
     *
     * <p>Configuration:
     * <ul>
     *   <li>Parallelism: enabled (auto-detect cores)</li>
     *   <li>SIMD across batch: enabled</li>
     *   <li>Cache optimization: enabled</li>
     *   <li>Detailed stats: disabled (minimal overhead)</li>
     * </ul>
     *
     * @return default batch options
     */
    public static BatchOptions defaults() {
        return new BatchOptions(
            true,   // Enable parallelism
            0,      // Auto-detect parallelism
            true,   // Enable SIMD across batch
            true,   // Cache optimize
            false   // No detailed stats by default
        );
    }

    /**
     * Options optimized for maximum throughput with large batches.
     *
     * <p>Configuration:
     * <ul>
     *   <li>Parallelism: enabled (auto-detect cores)</li>
     *   <li>SIMD across batch: enabled</li>
     *   <li>Cache optimization: enabled</li>
     *   <li>Detailed stats: disabled (zero overhead)</li>
     * </ul>
     *
     * <p>Use for batch sizes >100 where throughput is critical.
     *
     * @return throughput-optimized options
     */
    public static BatchOptions throughput() {
        return new BatchOptions(
            true,   // Enable parallelism
            0,      // Auto-detect
            true,   // Enable SIMD
            true,   // Cache optimize
            false   // No overhead from stats
        );
    }

    /**
     * Options optimized for debugging and profiling.
     *
     * <p>Configuration:
     * <ul>
     *   <li>Parallelism: disabled (reproducible timing)</li>
     *   <li>SIMD across batch: disabled (clearer profiling)</li>
     *   <li>Cache optimization: disabled (explicit behavior)</li>
     *   <li>Detailed stats: enabled (full metrics)</li>
     * </ul>
     *
     * <p>Use for benchmarking, debugging, or understanding performance characteristics.
     *
     * @return profiling-optimized options
     */
    public static BatchOptions profiling() {
        return new BatchOptions(
            false,  // Disable parallelism for reproducibility
            1,      // Single-threaded
            false,  // Disable SIMD for clarity
            false,  // Don't optimize for cache
            true    // Track everything
        );
    }

    /**
     * Create custom options with specific parallelism level.
     *
     * @param parallelism number of concurrent threads to use
     * @return options with specified parallelism
     * @throws IllegalArgumentException if parallelism < 1
     */
    public static BatchOptions withParallelism(int parallelism) {
        if (parallelism < 1) {
            throw new IllegalArgumentException("Parallelism must be >= 1");
        }
        return new BatchOptions(
            true,
            parallelism,
            true,
            true,
            false
        );
    }

    /**
     * Validate options are consistent.
     *
     * @throws IllegalArgumentException if options are invalid
     */
    public BatchOptions {
        if (maxParallelism < 0) {
            throw new IllegalArgumentException("maxParallelism must be >= 0");
        }
        if (enableParallelism && maxParallelism == 0) {
            // Auto-detect: use available processors
            maxParallelism = Runtime.getRuntime().availableProcessors();
        }
        if (!enableParallelism && maxParallelism > 1) {
            throw new IllegalArgumentException(
                "Cannot set maxParallelism > 1 when parallelism disabled");
        }
    }

    /**
     * Get effective parallelism level.
     *
     * @return actual number of threads that will be used
     */
    public int getEffectiveParallelism() {
        return enableParallelism ? maxParallelism : 1;
    }

    /**
     * Check if any optimization is enabled.
     *
     * @return true if at least one optimization is active
     */
    public boolean hasOptimizations() {
        return enableParallelism || enableSIMDAcrossBatch || cacheOptimize;
    }
}
