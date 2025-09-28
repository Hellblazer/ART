package com.hellblazer.art.hybrid.pan;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.hybrid.pan.parameters.PANParameters;
import com.hellblazer.art.hybrid.pan.weight.BPARTWeight;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive diagnostic test to analyze why PAN is clustering all distinct patterns
 * into a single category. This test will trace through the normalization, complement
 * coding, and similarity calculations step by step to identify the root cause.
 */
class PANDiagnosticTest {

    private PAN pan;
    private PANParameters parameters;
    private final DecimalFormat df = new DecimalFormat("#.####");

    @BeforeEach
    void setUp() {
        // Enable debug output
        System.setProperty("pan.debug", "true");

        // Use moderate vigilance that should create multiple categories for distinct patterns
        parameters = new PANParameters(
            0.7,  // vigilance - should allow multiple categories for distinct patterns
            PANParameters.defaultParameters().maxCategories(),
            PANParameters.defaultParameters().cnnConfig(),
            PANParameters.defaultParameters().enableCNNPretraining(),
            PANParameters.defaultParameters().learningRate(),
            PANParameters.defaultParameters().momentum(),
            PANParameters.defaultParameters().weightDecay(),
            PANParameters.defaultParameters().allowNegativeWeights(),
            PANParameters.defaultParameters().hiddenUnits(),
            PANParameters.defaultParameters().stmDecayRate(),
            PANParameters.defaultParameters().ltmConsolidationThreshold(),
            PANParameters.defaultParameters().replayBufferSize(),
            PANParameters.defaultParameters().replayBatchSize(),
            PANParameters.defaultParameters().replayFrequency(),
            PANParameters.defaultParameters().biasFactor(),
            false,  // DISABLE normalization to test the fix
            0.0,    // globalMinBound (not used when disabled)
            1.0     // globalMaxBound (not used when disabled)
        );
        pan = new PAN(parameters);
    }

    @AfterEach
    void tearDown() {
        System.clearProperty("pan.debug");
        if (pan != null) {
            pan.close();
        }
    }

    @Test
    void testPatternClusteringDiagnostic() {
        System.out.println("\n=== PAN Pattern Clustering Diagnostic Test ===");
        System.out.println("Vigilance: " + parameters.vigilance());

        // Create three VERY distinct patterns in different value ranges
        // These should definitely form separate categories
        List<Pattern> distinctPatterns = createDistinctTestPatterns();

        System.out.println("\n--- Learning Phase ---");
        int[] categories = new int[distinctPatterns.size()];

        for (int i = 0; i < distinctPatterns.size(); i++) {
            Pattern pattern = distinctPatterns.get(i);
            System.out.println("\nLearning Pattern " + i + ":");

            // Analyze pattern before any processing
            analyzeRawPattern(pattern, i);

            // Learn the pattern
            var result = pan.learn(pattern, parameters);
            assertInstanceOf(ActivationResult.Success.class, result);
            var success = (ActivationResult.Success) result;
            categories[i] = success.categoryIndex();

            System.out.println("Result: Category " + categories[i] + ", Activation: " + df.format(success.activationValue()));
            System.out.println("Total categories after learning: " + pan.getCategoryCount());
        }

        System.out.println("\n--- Final Analysis ---");
        System.out.println("Total patterns learned: " + distinctPatterns.size());
        System.out.println("Total categories created: " + pan.getCategoryCount());
        System.out.println("Categories assigned: " + java.util.Arrays.toString(categories));

        // Test prediction consistency
        System.out.println("\n--- Prediction Consistency Test ---");
        for (int i = 0; i < distinctPatterns.size(); i++) {
            var predictResult = pan.predict(distinctPatterns.get(i), parameters);
            assertInstanceOf(ActivationResult.Success.class, predictResult);
            var predictSuccess = (ActivationResult.Success) predictResult;

            assertEquals(categories[i], predictSuccess.categoryIndex(),
                "Pattern " + i + " should predict the same category it was learned with");
            System.out.println("Pattern " + i + " prediction: Category " + predictSuccess.categoryIndex() + " (consistent)");
        }

        // The diagnostic - this should reveal why we're getting only 1 category
        System.out.println("\n--- DIAGNOSTIC ANALYSIS ---");
        if (pan.getCategoryCount() == 1) {
            System.out.println("❌ PROBLEM CONFIRMED: All distinct patterns clustered into single category!");
            analyzeWhySingleCategory(distinctPatterns);
        } else {
            System.out.println("✅ Good: Multiple categories created as expected");
        }

        // For this diagnostic test, we expect the problem to manifest
        // The test passes regardless - it's a diagnostic tool
        assertTrue(pan.getCategoryCount() >= 1, "At least one category should be created");
    }

    @Test
    void testComplementCodingEffect() {
        System.out.println("\n=== Complement Coding Effect Analysis ===");

        // Test how complement coding affects pattern similarity
        Pattern pattern1 = new DenseVector(new double[]{0.1, 0.2, 0.3});
        Pattern pattern2 = new DenseVector(new double[]{0.7, 0.8, 0.9});

        System.out.println("Original Pattern 1: " + formatPattern(pattern1));
        System.out.println("Original Pattern 2: " + formatPattern(pattern2));

        // Simulate the same transformations that PAN.learn() applies
        Pattern norm1 = simulateNormalization(pattern1);
        Pattern norm2 = simulateNormalization(pattern2);

        System.out.println("Normalized Pattern 1: " + formatPattern(norm1));
        System.out.println("Normalized Pattern 2: " + formatPattern(norm2));

        Pattern comp1 = simulateComplementCoding(norm1);
        Pattern comp2 = simulateComplementCoding(norm2);

        System.out.println("Complement-coded Pattern 1: " + formatPattern(comp1));
        System.out.println("Complement-coded Pattern 2: " + formatPattern(comp2));

        // Calculate similarities at each stage
        double origSimilarity = calculateFuzzyARTSimilarity(pattern1, patternToArray(pattern2));
        double normSimilarity = calculateFuzzyARTSimilarity(norm1, patternToArray(norm2));
        double compSimilarity = calculateFuzzyARTSimilarity(comp1, patternToArray(comp2));

        System.out.println("\nSimilarity Analysis:");
        System.out.println("Original patterns similarity: " + df.format(origSimilarity));
        System.out.println("Normalized patterns similarity: " + df.format(normSimilarity));
        System.out.println("Complement-coded patterns similarity: " + df.format(compSimilarity));

        System.out.println("Vigilance threshold: " + df.format(parameters.vigilance()));
        System.out.println("Should create new category? " + (compSimilarity < parameters.vigilance() ? "YES" : "NO"));
    }

    @Test
    void testNormalizationEffect() {
        System.out.println("\n=== Normalization Effect Analysis ===");

        // Test patterns in very different ranges
        Pattern lowRange = new DenseVector(new double[]{0.1, 0.15, 0.2});      // Low values
        Pattern midRange = new DenseVector(new double[]{0.45, 0.5, 0.55});     // Mid values
        Pattern highRange = new DenseVector(new double[]{0.8, 0.85, 0.9});     // High values

        System.out.println("Before normalization:");
        System.out.println("Low range: " + formatPattern(lowRange));
        System.out.println("Mid range: " + formatPattern(midRange));
        System.out.println("High range: " + formatPattern(highRange));

        Pattern normLow = simulateNormalization(lowRange);
        Pattern normMid = simulateNormalization(midRange);
        Pattern normHigh = simulateNormalization(highRange);

        System.out.println("\nAfter normalization:");
        System.out.println("Low range: " + formatPattern(normLow));
        System.out.println("Mid range: " + formatPattern(normMid));
        System.out.println("High range: " + formatPattern(normHigh));

        // Check if normalization is causing convergence
        double lowMidSim = calculateFuzzyARTSimilarity(normLow, patternToArray(normMid));
        double lowHighSim = calculateFuzzyARTSimilarity(normLow, patternToArray(normHigh));
        double midHighSim = calculateFuzzyARTSimilarity(normMid, patternToArray(normHigh));

        System.out.println("\nSimilarities after normalization:");
        System.out.println("Low-Mid: " + df.format(lowMidSim));
        System.out.println("Low-High: " + df.format(lowHighSim));
        System.out.println("Mid-High: " + df.format(midHighSim));
    }

    private List<Pattern> createDistinctTestPatterns() {
        List<Pattern> patterns = new ArrayList<>();
        Random rand = new Random(42); // Fixed seed for reproducibility

        // Pattern 1: Very low values [0.0, 0.2]
        double[] data1 = new double[100];
        for (int i = 0; i < 100; i++) {
            data1[i] = rand.nextDouble() * 0.2;
        }
        patterns.add(new DenseVector(data1));

        // Pattern 2: Mid values [0.4, 0.6]
        double[] data2 = new double[100];
        for (int i = 0; i < 100; i++) {
            data2[i] = 0.4 + rand.nextDouble() * 0.2;
        }
        patterns.add(new DenseVector(data2));

        // Pattern 3: High values [0.8, 1.0]
        double[] data3 = new double[100];
        for (int i = 0; i < 100; i++) {
            data3[i] = 0.8 + rand.nextDouble() * 0.2;
        }
        patterns.add(new DenseVector(data3));

        return patterns;
    }

    private void analyzeRawPattern(Pattern pattern, int index) {
        double min = Double.MAX_VALUE;
        double max = -Double.MAX_VALUE;
        double sum = 0;

        for (int i = 0; i < pattern.dimension(); i++) {
            double val = pattern.get(i);
            min = Math.min(min, val);
            max = Math.max(max, val);
            sum += val;
        }

        double mean = sum / pattern.dimension();

        System.out.println("  Raw pattern stats: min=" + df.format(min) +
                          ", max=" + df.format(max) +
                          ", mean=" + df.format(mean) +
                          ", range=" + df.format(max - min));
    }

    private void analyzeWhySingleCategory(List<Pattern> patterns) {
        System.out.println("\nAnalyzing why all patterns ended up in single category...");

        // Get the single category weight
        var categories = pan.getCategories();
        if (categories.isEmpty()) {
            System.out.println("No categories found!");
            return;
        }

        var categoryWeight = (BPARTWeight) categories.get(0);

        System.out.println("\nTesting each pattern against the single category:");
        for (int i = 0; i < patterns.size(); i++) {
            Pattern original = patterns.get(i);

            // Apply same transformations as PAN
            Pattern normalized = simulateNormalization(original);
            Pattern complemented = simulateComplementCoding(normalized);

            // Calculate similarity like PAN does
            double similarity = calculateFuzzyARTSimilarity(complemented, categoryWeight.forwardWeights());

            System.out.println("Pattern " + i + ":");
            System.out.println("  After transformations: " + formatPattern(complemented, 5));
            System.out.println("  Similarity to category: " + df.format(similarity));
            System.out.println("  Vigilance threshold: " + df.format(parameters.vigilance()));
            System.out.println("  Would create new category? " + (similarity < parameters.vigilance() ? "YES" : "NO"));
        }
    }

    // Utility methods to simulate PAN's internal transformations

    private Pattern simulateNormalization(Pattern input) {
        int dim = input.dimension();
        double[] normalized = new double[dim];

        // Find min and max values (same logic as PAN.normalizeToUnitRange)
        double min = Double.MAX_VALUE;
        double max = -Double.MAX_VALUE;
        for (int i = 0; i < dim; i++) {
            double val = input.get(i);
            min = Math.min(min, val);
            max = Math.max(max, val);
        }

        // Normalize to [0,1] range
        double range = max - min;
        if (range == 0.0) {
            java.util.Arrays.fill(normalized, 0.5);
        } else {
            for (int i = 0; i < dim; i++) {
                normalized[i] = (input.get(i) - min) / range;
            }
        }

        return new DenseVector(normalized);
    }

    private Pattern simulateComplementCoding(Pattern input) {
        int originalDim = input.dimension();
        double[] complementCoded = new double[originalDim * 2];

        // First half: original values
        for (int i = 0; i < originalDim; i++) {
            complementCoded[i] = input.get(i);
        }

        // Second half: complement values (1 - x)
        for (int i = 0; i < originalDim; i++) {
            complementCoded[originalDim + i] = 1.0 - complementCoded[i];
        }

        return new DenseVector(complementCoded);
    }

    private double calculateFuzzyARTSimilarity(Pattern input, double[] weights) {
        double minSum = 0.0;
        double inputSum = 0.0;
        int size = Math.min(input.dimension(), weights.length);

        for (int i = 0; i < size; i++) {
            double inputVal = Math.abs(input.get(i));
            double weightVal = Math.abs(weights[i]);
            minSum += Math.min(inputVal, weightVal);
            inputSum += inputVal;
        }

        // Avoid division by zero
        if (inputSum == 0.0) {
            return 0.0;
        }

        return minSum / inputSum;
    }

    private String formatPattern(Pattern pattern) {
        return formatPattern(pattern, Math.min(10, pattern.dimension()));
    }

    private String formatPattern(Pattern pattern, int maxElements) {
        StringBuilder sb = new StringBuilder("[");
        int limit = Math.min(maxElements, pattern.dimension());
        for (int i = 0; i < limit; i++) {
            if (i > 0) sb.append(", ");
            sb.append(df.format(pattern.get(i)));
        }
        if (pattern.dimension() > maxElements) {
            sb.append(", ...");
        }
        sb.append("]");
        return sb.toString();
    }

    private double[] patternToArray(Pattern pattern) {
        double[] array = new double[pattern.dimension()];
        for (int i = 0; i < pattern.dimension(); i++) {
            array[i] = pattern.get(i);
        }
        return array;
    }
}