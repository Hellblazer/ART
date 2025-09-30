package com.hellblazer.art.laminar.benchmarks;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.integration.ARTCircuitParameters;
import com.hellblazer.art.laminar.integration.ARTLaminarCircuit;
import com.hellblazer.art.laminar.integration.VectorizedARTLaminarCircuit;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Optimized performance test targeting SIMD speedup with proper parameter tuning.
 *
 * Key optimizations:
 * - Low vigilance (0.3) to create many categories (50-100+)
 * - Diverse patterns (20 clusters) to maximize category diversity
 * - High dimensions (256D, 512D) where SIMD excels
 */
public class OptimizedPerformanceTest {

    private static final int WARMUP_ITERATIONS = 200;
    private static final int MEASUREMENT_ITERATIONS = 2000;

    public static void main(String[] args) {
        System.out.println("=== OPTIMIZED ART Laminar Circuit Performance ===");
        System.out.println("Target: >5x vectorization speedup\n");

        // Test with optimized parameters
        testOptimizedConfiguration();
    }

    private static void testOptimizedConfiguration() {
        int[] dimensions = {100, 256, 512};
        double[] vigilances = {0.3, 0.5};  // Low vigilance = more categories
        int clusters = 20;  // Diverse data

        for (int dim : dimensions) {
            for (double vigilance : vigilances) {
                System.out.printf("Dimension: %dD, Vigilance: %.1f, Clusters: %d%n", dim, vigilance, clusters);
                System.out.println("─".repeat(80));

                runOptimizedBenchmark(dim, vigilance, clusters);
                System.out.println();
            }
        }
    }

    private static void runOptimizedBenchmark(int inputSize, double vigilance, int clusters) {
        // Generate diverse patterns
        var patterns = generateDiversePatterns(inputSize, MEASUREMENT_ITERATIONS, clusters);

        // Create circuits with optimized parameters
        var params = ARTCircuitParameters.builder(inputSize)
            .vigilance(vigilance)           // LOW vigilance = MORE categories
            .learningRate(0.8)              // Fast convergence
            .choiceParameter(0.001)         // Standard
            .maxCategories(200)             // Allow many categories
            .build();

        var standardCircuit = new ARTLaminarCircuit(params);
        var vectorizedCircuit = new VectorizedARTLaminarCircuit(params);

        // === WARMUP PHASE ===
        System.out.print("Warming up... ");
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            standardCircuit.process(patterns.get(i % patterns.size()));
            vectorizedCircuit.process(patterns.get(i % patterns.size()));
        }
        System.out.println("done");

        standardCircuit.reset();
        vectorizedCircuit.reset();

        // === MEASUREMENT PHASE - STANDARD ===
        System.out.print("Measuring standard circuit... ");
        long standardStart = System.nanoTime();
        for (var pattern : patterns) {
            standardCircuit.process(pattern);
        }
        long standardTime = (System.nanoTime() - standardStart) / 1_000_000; // ms
        int standardCategories = standardCircuit.getCategoryCount();
        System.out.printf("done (%d ms, %d categories)%n", standardTime, standardCategories);

        // === MEASUREMENT PHASE - VECTORIZED ===
        System.out.print("Measuring vectorized circuit... ");
        vectorizedCircuit.resetPerformanceTracking();
        long vectorizedStart = System.nanoTime();
        for (var pattern : patterns) {
            vectorizedCircuit.process(pattern);
        }
        long vectorizedTime = (System.nanoTime() - vectorizedStart) / 1_000_000; // ms
        int vectorizedCategories = vectorizedCircuit.getCategoryCount();
        var stats = vectorizedCircuit.getPerformanceStats();
        System.out.printf("done (%d ms, %d categories)%n", vectorizedTime, vectorizedCategories);

        // === ANALYSIS ===
        double speedup = (double) standardTime / vectorizedTime;
        double throughputStd = MEASUREMENT_ITERATIONS / (standardTime / 1000.0);
        double throughputVec = MEASUREMENT_ITERATIONS / (vectorizedTime / 1000.0);

        System.out.println("\n📊 RESULTS:");
        System.out.printf("  Standard:   %,6d ms | %.1f μs/pattern | %,6.0f patterns/sec | %d categories%n",
                         standardTime,
                         standardTime * 1000.0 / MEASUREMENT_ITERATIONS,
                         throughputStd,
                         standardCategories);
        System.out.printf("  Vectorized: %,6d ms | %.1f μs/pattern | %,6.0f patterns/sec | %d categories%n",
                         vectorizedTime,
                         vectorizedTime * 1000.0 / MEASUREMENT_ITERATIONS,
                         throughputVec,
                         vectorizedCategories);
        System.out.println();
        System.out.printf("  ⚡ SPEEDUP: %.2fx %s%n",
                         speedup,
                         speedup > 2.0 ? "🚀 EXCELLENT!" :
                         speedup > 1.5 ? "✅ GOOD" :
                         speedup > 1.2 ? "⚠️  MODEST" : "❌ NO BENEFIT");
        System.out.println();
        System.out.printf("  🔬 SIMD Stats:%n");
        System.out.printf("     Vector Operations: %,d%n", stats.totalVectorOperations());
        System.out.printf("     Parallel Tasks:    %,d%n", stats.totalParallelTasks());
        System.out.printf("     Active Threads:    %d%n", stats.activeThreads());
        System.out.printf("     Parallel Efficiency: %.1f%%%n", stats.getParallelEfficiency() * 100);

        // Validate category counts are reasonable
        if (standardCategories < 10) {
            System.out.printf("  ⚠️  WARNING: Only %d categories (need >50 for best speedup)%n", standardCategories);
            System.out.println("     Recommendation: Lower vigilance or increase cluster diversity");
        } else if (standardCategories > 50) {
            System.out.println("  ✅ Category count optimal for parallel processing");
        }
    }

    /**
     * Generate diverse patterns with multiple clusters.
     * More clusters = more categories = better parallelism.
     *
     * Uses VERY DIFFERENT patterns to force category creation.
     */
    private static List<Pattern> generateDiversePatterns(int inputSize, int count, int numClusters) {
        var patterns = new ArrayList<Pattern>(count);
        var random = new Random(42); // Deterministic for reproducibility

        for (int i = 0; i < count; i++) {
            int cluster = i % numClusters;
            double[] data = new double[inputSize];

            // Create VERY DIFFERENT cluster centers (evenly spaced)
            double clusterCenter = cluster / (double) (numClusters - 1); // 0.0 to 1.0

            for (int j = 0; j < inputSize; j++) {
                // For each cluster, create a distinct pattern
                // Use sine wave with different phases per cluster
                double phase = (cluster * Math.PI * 2) / numClusters;
                double wave = (Math.sin(j * 0.1 + phase) + 1.0) / 2.0; // [0, 1]

                // Mix cluster center with wave pattern
                double base = 0.7 * clusterCenter + 0.3 * wave;

                // Add SMALL noise (don't blur clusters together)
                double noise = random.nextGaussian() * 0.05;

                // Combine and clamp to [0, 1]
                data[j] = Math.max(0.0, Math.min(1.0, base + noise));
            }

            patterns.add(new DenseVector(data));
        }

        return patterns;
    }
}