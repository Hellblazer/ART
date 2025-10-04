package com.hellblazer.art.laminar.benchmarks;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.canonical.CircuitParameters;
import com.hellblazer.art.laminar.canonical.FullLaminarCircuitImpl;
import com.hellblazer.art.laminar.integration.ARTCircuitParameters;
import com.hellblazer.art.laminar.integration.ARTLaminarCircuit;
import com.hellblazer.art.laminar.integration.VectorizedARTLaminarCircuit;

import java.util.ArrayList;
import java.util.List;

/**
 * Simple performance analysis tool to measure and compare circuit implementations.
 * Provides quantitative measurements without JMH infrastructure complexity.
 */
public class SimplePerformanceAnalysis {

    private static final int WARMUP_ITERATIONS = 100;
    private static final int MEASUREMENT_ITERATIONS = 1000;

    public static void main(String[] args) {
        System.out.println("=== ART Laminar Circuit Performance Analysis ===\n");

        // Test multiple input sizes
        int[] inputSizes = {50, 100, 200};

        for (int inputSize : inputSizes) {
            System.out.println("Input Size: " + inputSize + "D");
            System.out.println("─".repeat(60));

            analyzePerformance(inputSize);
            System.out.println();
        }

        System.out.println("\n=== High-Dimensional Vectorization Analysis ===\n");
        analyzeVectorization();
    }

    private static void analyzePerformance(int inputSize) {
        // Generate test patterns
        var patterns = generatePatterns(inputSize, MEASUREMENT_ITERATIONS);

        // 1. Baseline: FullLaminarCircuitImpl (manual templates)
        var manualParams = CircuitParameters.builder()
            .inputSize(inputSize)
            .categorySize(inputSize / 2)
            .vigilance(0.75)
            .learningRate(0.5)
            .timeStep(0.01)
            .build();
        var manualCircuit = new FullLaminarCircuitImpl(manualParams);

        long manualTime = measureCircuit(manualCircuit, patterns, "Manual");

        // 2. ART Integration: ARTLaminarCircuit (FuzzyART)
        var artParams = ARTCircuitParameters.builder(inputSize)
            .vigilance(0.75)
            .learningRate(0.5)
            .maxCategories(inputSize / 2)
            .build();
        var artCircuit = new ARTLaminarCircuit(artParams);

        long artTime = measureCircuit(artCircuit, patterns, "ART");

        // 3. Vectorized: VectorizedARTLaminarCircuit (SIMD)
        var vectorizedCircuit = new VectorizedARTLaminarCircuit(artParams);

        long vectorizedTime = measureVectorizedCircuit(vectorizedCircuit, patterns, "Vectorized");

        // Calculate speedups
        double artSpeedup = (double) manualTime / artTime;
        double vectorizedSpeedup = (double) manualTime / vectorizedTime;
        double simdSpeedup = (double) artTime / vectorizedTime;

        System.out.printf("Manual Circuit:     %6d ms (%.2f μs/pattern)%n",
                         manualTime, manualTime * 1000.0 / MEASUREMENT_ITERATIONS);
        System.out.printf("ART Circuit:        %6d ms (%.2f μs/pattern) [%.2fx vs manual]%n",
                         artTime, artTime * 1000.0 / MEASUREMENT_ITERATIONS, artSpeedup);
        System.out.printf("Vectorized Circuit: %6d ms (%.2f μs/pattern) [%.2fx vs manual, %.2fx vs ART]%n",
                         vectorizedTime, vectorizedTime * 1000.0 / MEASUREMENT_ITERATIONS,
                         vectorizedSpeedup, simdSpeedup);

        // Category statistics
        System.out.printf("Categories: ART=%d, Vectorized=%d%n",
                         artCircuit.getCategoryCount(),
                         vectorizedCircuit.getCategoryCount());
    }

    private static void analyzeVectorization() {
        int[] dimensions = {50, 100, 256, 512};

        System.out.println("Dimension | Standard (ms) | Vectorized (ms) | Speedup | SIMD Ops");
        System.out.println("─".repeat(70));

        for (int dim : dimensions) {
            var patterns = generatePatterns(dim, MEASUREMENT_ITERATIONS);

            var params = ARTCircuitParameters.builder(dim)
                .vigilance(0.75)
                .learningRate(0.5)
                .build();

            var standardCircuit = new ARTLaminarCircuit(params);
            var vectorizedCircuit = new VectorizedARTLaminarCircuit(params);

            // Warmup
            for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                standardCircuit.process(patterns.get(i % patterns.size()));
                vectorizedCircuit.process(patterns.get(i % patterns.size()));
            }

            standardCircuit.reset();
            vectorizedCircuit.reset();

            // Measure
            long standardTime = System.nanoTime();
            for (var pattern : patterns) {
                standardCircuit.process(pattern);
            }
            standardTime = (System.nanoTime() - standardTime) / 1_000_000; // ms

            long vectorizedTime = System.nanoTime();
            for (var pattern : patterns) {
                vectorizedCircuit.process(pattern);
            }
            vectorizedTime = (System.nanoTime() - vectorizedTime) / 1_000_000; // ms

            double speedup = (double) standardTime / vectorizedTime;
            var stats = vectorizedCircuit.getPerformanceStats();

            System.out.printf("%6dD   | %10d    | %12d    | %.2fx   | %d%n",
                             dim, standardTime, vectorizedTime, speedup, stats.totalVectorOperations());
        }
    }

    private static long measureCircuit(FullLaminarCircuitImpl circuit,
                                      List<Pattern> patterns,
                                      String name) {
        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            circuit.process(patterns.get(i % patterns.size()));
        }

        circuit.reset();

        // Measure
        long startTime = System.nanoTime();
        for (var pattern : patterns) {
            circuit.process(pattern);
        }
        long elapsedTime = System.nanoTime() - startTime;

        return elapsedTime / 1_000_000; // Convert to milliseconds
    }

    private static long measureCircuit(ARTLaminarCircuit circuit,
                                      List<Pattern> patterns,
                                      String name) {
        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            circuit.process(patterns.get(i % patterns.size()));
        }

        circuit.reset();

        // Measure
        long startTime = System.nanoTime();
        for (var pattern : patterns) {
            circuit.process(pattern);
        }
        long elapsedTime = System.nanoTime() - startTime;

        return elapsedTime / 1_000_000; // Convert to milliseconds
    }

    private static long measureVectorizedCircuit(VectorizedARTLaminarCircuit circuit,
                                                 List<Pattern> patterns,
                                                 String name) {
        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            circuit.process(patterns.get(i % patterns.size()));
        }

        circuit.reset();
        circuit.resetPerformanceTracking();

        // Measure
        long startTime = System.nanoTime();
        for (var pattern : patterns) {
            circuit.process(pattern);
        }
        long elapsedTime = System.nanoTime() - startTime;

        return elapsedTime / 1_000_000; // Convert to milliseconds
    }

    private static List<Pattern> generatePatterns(int inputSize, int count) {
        var patterns = new ArrayList<Pattern>(count);

        // Create clustered patterns for realistic testing
        int clustersPerSize = 5;
        for (int i = 0; i < count; i++) {
            int cluster = i % clustersPerSize;
            double[] data = new double[inputSize];

            for (int j = 0; j < inputSize; j++) {
                // Base cluster value with noise
                double base = (cluster / (double) clustersPerSize) + 0.1;
                double noise = (Math.random() - 0.5) * 0.2;
                data[j] = Math.max(0.0, Math.min(1.0, base + noise));
            }

            patterns.add(new DenseVector(data));
        }

        return patterns;
    }
}