package com.hellblazer.art.cortical.batch;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;
import org.junit.jupiter.api.Test;

import java.util.Random;

/**
 * Fair comparison test between sequential and SIMD implementations.
 * Both paths perform exactly the same operations for accurate comparison.
 */
public class SIMDFairComparisonTest {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    @Test
    public void fairComparisonTest() {
        System.out.println("\n=== SIMD Fair Comparison Test ===\n");

        // Print system info
        System.out.println("System Information:");
        System.out.println("  Vector Species: " + SPECIES);
        System.out.println("  Vector Lane Size: " + SPECIES.length());
        System.out.println("  Vector Bit Size: " + SPECIES.vectorBitSize());
        System.out.println("  Architecture: " + System.getProperty("os.arch"));
        System.out.println();

        // Test parameters
        int batchSize = 64;
        int dimension = 64;
        var random = new Random(42);
        var patterns = generatePatterns(batchSize, dimension, random);

        // Warmup
        for (int i = 0; i < 100; i++) {
            processSequentialExact(patterns, dimension);
            processSIMDManual(patterns, dimension);
            processSIMDWithFramework(patterns, dimension);
        }

        // Measure sequential (exact same operations as SIMD)
        long seqStart = System.nanoTime();
        for (int i = 0; i < 100; i++) {
            processSequentialExact(patterns, dimension);
        }
        long seqTime = (System.nanoTime() - seqStart) / 100;

        // Measure manual SIMD (no framework overhead)
        long simdManualStart = System.nanoTime();
        for (int i = 0; i < 100; i++) {
            processSIMDManual(patterns, dimension);
        }
        long simdManualTime = (System.nanoTime() - simdManualStart) / 100;

        // Measure framework SIMD (current implementation)
        long simdFrameworkStart = System.nanoTime();
        for (int i = 0; i < 100; i++) {
            processSIMDWithFramework(patterns, dimension);
        }
        long simdFrameworkTime = (System.nanoTime() - simdFrameworkStart) / 100;

        // Results
        System.out.println("Performance Results (batch=" + batchSize + ", dim=" + dimension + "):");
        System.out.println("----------------------------------------");
        System.out.printf("Sequential (exact ops):     %,10d ns (1.00x baseline)%n", seqTime);
        System.out.printf("Manual SIMD (no framework): %,10d ns (%.2fx speedup)%n",
            simdManualTime, (double)seqTime / simdManualTime);
        System.out.printf("Framework SIMD (current):   %,10d ns (%.2fx speedup)%n",
            simdFrameworkTime, (double)seqTime / simdFrameworkTime);
        System.out.println();

        // Analysis
        System.out.println("Analysis:");
        double manualSpeedup = (double)seqTime / simdManualTime;
        double frameworkSpeedup = (double)seqTime / simdFrameworkTime;
        double frameworkOverhead = (double)simdFrameworkTime / simdManualTime;

        System.out.printf("  Manual SIMD efficiency: %.1f%% (theoretical max: %.1fx)%n",
            100.0 * manualSpeedup / SPECIES.length(), (double)SPECIES.length());
        System.out.printf("  Framework overhead: %.1fx slower than manual SIMD%n", frameworkOverhead);

        if (manualSpeedup < 1.2) {
            System.out.println("  ⚠️  Manual SIMD not achieving expected speedup!");
            System.out.println("      Possible causes:");
            System.out.println("      - Vector lanes too narrow (only " + SPECIES.length() + " lanes)");
            System.out.println("      - Memory bandwidth limited");
            System.out.println("      - JVM not optimizing vector operations");
        }

        if (frameworkOverhead > 2.0) {
            System.out.println("  ⚠️  Framework has excessive overhead!");
            System.out.println("      Likely causes:");
            System.out.println("      - Unnecessary allocations");
            System.out.println("      - Transpose operations");
            System.out.println("      - Complex parameter handling");
        }
    }

    private Pattern[] processSequentialExact(Pattern[] inputs, int dimension) {
        var results = new Pattern[inputs.length];

        // Parameters (matching SIMD path)
        double decay = 0.3;
        double ceiling = 1.0;
        double floor = 0.0;
        double selfExc = 0.3;
        double deltaT = 0.01;
        double drivingStrength = 2.0;

        for (int i = 0; i < inputs.length; i++) {
            double[] data = new double[dimension];
            for (int d = 0; d < dimension; d++) {
                double x = inputs[i].get(d);

                // 1. Apply driving strength
                x *= drivingStrength;

                // 2. Shunting dynamics (exact same as SIMD)
                double excitation = Math.max(0, selfExc * x + x); // self + external
                double derivative = -decay * x + (ceiling - x) * excitation;
                x = x + deltaT * derivative;
                x = Math.max(floor, Math.min(ceiling, x)); // clamp

                // 3. Saturation
                if (x > 0) {
                    x = ceiling * x / (1.0 + x);
                }
                x = Math.max(floor, Math.min(ceiling, x));

                data[d] = x;
            }
            results[i] = new DenseVector(data);
        }
        return results;
    }

    private Pattern[] processSIMDManual(Pattern[] inputs, int dimension) {
        int batchSize = inputs.length;
        int laneSize = SPECIES.length();

        // Transpose to dimension-major
        double[][] dimMajor = new double[dimension][batchSize];
        for (int d = 0; d < dimension; d++) {
            for (int b = 0; b < batchSize; b++) {
                dimMajor[d][b] = inputs[b].get(d);
            }
        }

        // Parameters
        double decay = 0.3;
        double ceiling = 1.0;
        double floor = 0.0;
        double selfExc = 0.3;
        double deltaT = 0.01;
        double drivingStrength = 2.0;

        var decayVec = DoubleVector.broadcast(SPECIES, decay);
        var ceilingVec = DoubleVector.broadcast(SPECIES, ceiling);
        var floorVec = DoubleVector.broadcast(SPECIES, floor);
        var selfExcVec = DoubleVector.broadcast(SPECIES, selfExc);
        var deltaVec = DoubleVector.broadcast(SPECIES, deltaT);
        var drivingVec = DoubleVector.broadcast(SPECIES, drivingStrength);
        var oneVec = DoubleVector.broadcast(SPECIES, 1.0);
        var zeroVec = DoubleVector.zero(SPECIES);

        // Process dimension-major with SIMD
        for (int d = 0; d < dimension; d++) {
            var row = dimMajor[d];
            int b = 0;

            // SIMD loop
            for (; b < SPECIES.loopBound(batchSize); b += laneSize) {
                // Load
                var x = DoubleVector.fromArray(SPECIES, row, b);

                // 1. Driving strength
                x = x.mul(drivingVec);

                // 2. Shunting dynamics
                var excitation = selfExcVec.mul(x).add(x).max(zeroVec);
                var derivative = decayVec.mul(x).neg().add(
                    ceilingVec.sub(x).mul(excitation)
                );
                x = x.add(derivative.mul(deltaVec));
                x = x.max(floorVec).min(ceilingVec);

                // 3. Saturation (sigmoid)
                var saturated = ceilingVec.mul(x).div(oneVec.add(x));
                x = saturated.max(floorVec).min(ceilingVec);

                // Store
                x.intoArray(row, b);
            }

            // Scalar tail
            for (; b < batchSize; b++) {
                double x = row[b];
                x *= drivingStrength;
                double excitation = Math.max(0, selfExc * x + x);
                double derivative = -decay * x + (ceiling - x) * excitation;
                x = x + deltaT * derivative;
                x = Math.max(floor, Math.min(ceiling, x));
                if (x > 0) {
                    x = ceiling * x / (1.0 + x);
                }
                row[b] = Math.max(floor, Math.min(ceiling, x));
            }
        }

        // Transpose back to pattern-major
        var results = new Pattern[batchSize];
        for (int b = 0; b < batchSize; b++) {
            double[] data = new double[dimension];
            for (int d = 0; d < dimension; d++) {
                data[d] = dimMajor[d][b];
            }
            results[b] = new DenseVector(data);
        }
        return results;
    }

    private Pattern[] processSIMDWithFramework(Pattern[] inputs, int size) {
        // Use the actual framework (with lateralInhibition=0.0 to force SIMD)
        var result = Layer4SIMDBatch.processBatchSIMD(
            inputs,
            2.0,      // drivingStrength
            10.0,     // timeConstant (will become 0.01 after /1000)
            1.0,      // ceiling
            0.0,      // floor
            0.3,      // selfExcitation
            0.0,      // lateralInhibition (ZERO for SIMD path!)
            size
        );

        if (result == null) {
            // Fallback if not beneficial
            return processSequentialExact(inputs, size);
        }
        return result;
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
}