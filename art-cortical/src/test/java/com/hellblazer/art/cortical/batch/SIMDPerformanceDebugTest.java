package com.hellblazer.art.cortical.batch;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import org.junit.jupiter.api.Test;

import java.util.Random;

/**
 * Debug test to investigate SIMD performance issue.
 * Adds detailed timing instrumentation to understand where time is spent.
 */
public class SIMDPerformanceDebugTest {

    @Test
    public void debugSIMDPerformance() {
        System.out.println("\n=== SIMD Performance Debug Investigation ===\n");

        // Test with batch size 64, dimension 64
        int batchSize = 64;
        int dimension = 64;
        var random = new Random(42);
        var patterns = generatePatterns(batchSize, dimension, random);

        // Warmup
        for (int i = 0; i < 10; i++) {
            processSIMDInstrumented(patterns, dimension, 0.1, false); // with lateral
            processSIMDInstrumented(patterns, dimension, 0.0, false); // without lateral
        }

        System.out.println("Configuration: Batch=" + batchSize + ", Dimension=" + dimension);
        System.out.println("----------------------------------------");

        // Test 1: With lateral inhibition (current test configuration)
        System.out.println("\n1. WITH Lateral Inhibition (0.1) - Current Test Config:");
        var stats1 = processSIMDInstrumented(patterns, dimension, 0.1, true);
        printStats(stats1);

        // Test 2: Without lateral inhibition (force SIMD path)
        System.out.println("\n2. WITHOUT Lateral Inhibition (0.0) - Force SIMD Path:");
        var stats2 = processSIMDInstrumented(patterns, dimension, 0.0, true);
        printStats(stats2);

        // Test 3: Sequential baseline
        System.out.println("\n3. Sequential Baseline (for comparison):");
        long seqStart = System.nanoTime();
        processSequential(patterns, dimension);
        long seqTime = System.nanoTime() - seqStart;
        System.out.printf("   Total time: %,d ns%n", seqTime);

        // Analysis
        System.out.println("\n=== ANALYSIS ===");
        System.out.printf("Speedup with lateral (current):    %.2fx (%.1fx SLOWER!)%n",
            (double)seqTime / stats1.totalTime,
            (double)stats1.totalTime / seqTime);
        System.out.printf("Speedup without lateral (SIMD):    %.2fx%n",
            (double)seqTime / stats2.totalTime);
        System.out.printf("Lateral path overhead:              %.1fx slower than SIMD path%n",
            (double)stats1.totalTime / stats2.totalTime);

        // Hypothesis verification
        System.out.println("\n=== HYPOTHESIS VERIFICATION ===");
        if (stats1.usedSIMDPath) {
            System.out.println("❌ Hypothesis WRONG: WITH lateral still used SIMD path");
        } else {
            System.out.println("✅ Hypothesis CONFIRMED: WITH lateral used SLOW sequential path");
        }
        if (!stats2.usedSIMDPath) {
            System.out.println("❌ Unexpected: WITHOUT lateral did NOT use SIMD path");
        } else {
            System.out.println("✅ As expected: WITHOUT lateral used SIMD path");
        }
    }

    private InstrumentedStats processSIMDInstrumented(Pattern[] inputs, int size,
                                                       double lateralInhibition,
                                                       boolean verbose) {
        var stats = new InstrumentedStats();

        // Check if beneficial
        long checkStart = System.nanoTime();
        boolean beneficial = BatchDataLayout.isTransposeAndVectorizeBeneficial(
            inputs.length, size, 10);
        stats.beneficialCheckTime = System.nanoTime() - checkStart;

        if (!beneficial) {
            stats.totalTime = stats.beneficialCheckTime;
            if (verbose) System.out.println("   Not beneficial - would fall back to sequential");
            return stats;
        }

        // Transpose to dimension-major (included in createBatch)
        long createStart = System.nanoTime();
        var batch = Layer4SIMDBatch.createBatch(inputs, size);
        stats.transpose1Time = System.nanoTime() - createStart;  // includes transpose
        stats.createBatchTime = 0;  // combined with transpose above

        // Apply driving strength
        long drivingStart = System.nanoTime();
        batch.applyDrivingStrength(2.0);
        stats.drivingStrengthTime = System.nanoTime() - drivingStart;

        // Apply dynamics (THIS IS THE KEY PART)
        long dynamicsStart = System.nanoTime();
        var shuntingParams = com.hellblazer.art.cortical.dynamics.ShuntingParameters.builder(size)
            .ceiling(1.0)
            .floor(0.0)
            .selfExcitation(0.3)
            .inhibitoryStrength(lateralInhibition)  // KEY PARAMETER!
            .build();

        // Check if lateral interactions are enabled
        stats.usedSIMDPath = !(lateralInhibition > 0.0);

        batch.applyDynamicsExact(0.01, shuntingParams);
        stats.dynamicsTime = System.nanoTime() - dynamicsStart;

        // Apply saturation
        long saturationStart = System.nanoTime();
        batch.applySaturation(1.0, 0.0);
        stats.saturationTime = System.nanoTime() - saturationStart;

        // Transpose back to pattern-major
        long transpose2Start = System.nanoTime();
        var result = batch.toPatterns();
        stats.transpose2Time = System.nanoTime() - transpose2Start;

        stats.totalTime = System.nanoTime() - checkStart;

        return stats;
    }

    private Pattern[] processSequential(Pattern[] inputs, int size) {
        var results = new Pattern[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            results[i] = processPattern(inputs[i], size);
        }
        return results;
    }

    private Pattern processPattern(Pattern input, int size) {
        double[] data = new double[size];
        for (int d = 0; d < size; d++) {
            double x = input.get(d);

            // Driving strength
            x *= 2.0;

            // Shunting dynamics (simplified - no lateral)
            double A = 0.3;
            double B = 1.0;
            double E = x;
            double derivative = -A * x + (B - x) * E;
            x = x + 0.01 * derivative;

            // Saturation
            if (x > 0) {
                x = B * x / (1.0 + x);
            }
            x = Math.max(0.0, Math.min(B, x));

            data[d] = x;
        }
        return new DenseVector(data);
    }

    private Pattern[] generatePatterns(int count, int dimension, Random random) {
        var patterns = new Pattern[count];
        for (int i = 0; i < count; i++) {
            double[] data = new double[dimension];
            for (int d = 0; d < dimension; d++) {
                data[d] = random.nextDouble();
            }
            patterns[i] = new DenseVector(data);
        }
        return patterns;
    }

    private void printStats(InstrumentedStats stats) {
        System.out.printf("   Beneficial check:  %,8d ns (%.1f%%)%n",
            stats.beneficialCheckTime, 100.0 * stats.beneficialCheckTime / stats.totalTime);
        System.out.printf("   Transpose 1:       %,8d ns (%.1f%%)%n",
            stats.transpose1Time, 100.0 * stats.transpose1Time / stats.totalTime);
        System.out.printf("   Create batch:      %,8d ns (%.1f%%)%n",
            stats.createBatchTime, 100.0 * stats.createBatchTime / stats.totalTime);
        System.out.printf("   Driving strength:  %,8d ns (%.1f%%)%n",
            stats.drivingStrengthTime, 100.0 * stats.drivingStrengthTime / stats.totalTime);
        System.out.printf("   Dynamics:          %,8d ns (%.1f%%) %s%n",
            stats.dynamicsTime, 100.0 * stats.dynamicsTime / stats.totalTime,
            stats.usedSIMDPath ? "[SIMD PATH]" : "[SEQUENTIAL PATH - SLOW!]");
        System.out.printf("   Saturation:        %,8d ns (%.1f%%)%n",
            stats.saturationTime, 100.0 * stats.saturationTime / stats.totalTime);
        System.out.printf("   Transpose 2:       %,8d ns (%.1f%%)%n",
            stats.transpose2Time, 100.0 * stats.transpose2Time / stats.totalTime);
        System.out.printf("   TOTAL:             %,8d ns%n", stats.totalTime);
    }

    private static class InstrumentedStats {
        long beneficialCheckTime;
        long transpose1Time;
        long createBatchTime;
        long drivingStrengthTime;
        long dynamicsTime;
        long saturationTime;
        long transpose2Time;
        long totalTime;
        boolean usedSIMDPath;
    }

}