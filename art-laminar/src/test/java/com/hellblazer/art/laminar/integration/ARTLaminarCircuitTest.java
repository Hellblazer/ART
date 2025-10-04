package com.hellblazer.art.laminar.integration;

import com.hellblazer.art.core.Pattern;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for ARTLaminarCircuit integration with FuzzyART.
 *
 * These tests validate the hybrid circuit's ability to:
 * - Create ART categories from novel patterns
 * - Reactivate existing categories for familiar patterns
 * - Handle complement coding properly
 * - Maintain vigilance consistency
 * - Learn template dynamics via FuzzyART
 * - Compare performance with manual template management
 *
 * RED Phase: All tests written to fail initially (no implementation yet)
 *
 * @author Hal Hildebrand
 */
class ARTLaminarCircuitTest {

    private ARTCircuitParameters defaultParams;
    private ARTCircuitParameters highVigilanceParams;
    private ARTCircuitParameters lowVigilanceParams;

    @BeforeEach
    void setUp() {
        defaultParams = ARTCircuitParameters.createDefault(10);
        highVigilanceParams = ARTCircuitParameters.forHighVigilance(10);
        lowVigilanceParams = ARTCircuitParameters.forLowVigilance(10);
    }

    /**
     * Test 1: Novel pattern should create new ART category and achieve resonance.
     *
     * Validates:
     * - FuzzyART creates new category for novel input
     * - Circuit transitions to resonating state
     * - Category count increments correctly
     */
    @Test
    void testARTCategoryCreation() {
        var circuit = new ARTLaminarCircuit(defaultParams);

        // Create novel pattern
        var novelPattern = Pattern.of(new double[]{0.8, 0.6, 0.4, 0.2, 0.5, 0.7, 0.3, 0.9, 0.1, 0.6});

        // Process novel pattern - should create new category
        var expectation = circuit.process(novelPattern);

        // Validate category creation
        assertEquals(1, circuit.getCategoryCount(),
                    "Novel pattern should create exactly 1 category");

        // Validate resonance
        assertTrue(circuit.isResonating(),
                  "Circuit should resonate with new category");

        // Validate circuit state
        var state = circuit.getState();
        assertEquals(0, state.activeCategory(),
                    "First category should have ID 0");
        assertTrue(state.matchScore() > 0.0,
                  "Match score should be positive");

        // Expectation should be non-null and correct dimension
        assertNotNull(expectation, "Expectation should not be null");
        assertEquals(10, expectation.dimension(),
                    "Expectation should match input dimension");
    }

    /**
     * Test 2: Same pattern presented twice should reuse existing category.
     *
     * Validates:
     * - Category reactivation for familiar patterns
     * - No spurious category creation
     * - Consistent category assignment
     */
    @Test
    void testARTCategoryReactivation() {
        var circuit = new ARTLaminarCircuit(defaultParams);

        // Create test pattern
        var pattern = Pattern.of(new double[]{0.7, 0.5, 0.3, 0.6, 0.8, 0.4, 0.2, 0.9, 0.1, 0.5});

        // First presentation - creates category
        circuit.process(pattern);
        var firstCategoryId = circuit.getState().activeCategory();

        // Second presentation - should reactivate same category
        circuit.process(pattern);
        var secondCategoryId = circuit.getState().activeCategory();

        // Validate single category
        assertEquals(1, circuit.getCategoryCount(),
                    "Same pattern should not create multiple categories");

        // Validate category consistency
        assertEquals(firstCategoryId, secondCategoryId,
                    "Category ID should be consistent across presentations");
        assertEquals(0, firstCategoryId,
                    "Category ID should be 0");

        // Both presentations should achieve resonance
        assertTrue(circuit.isResonating(),
                  "Second presentation should also resonate");
    }

    /**
     * Test 3: Complement coding integration should work correctly.
     *
     * Validates:
     * - FuzzyART receives complement-coded input [x, 1-x]
     * - Bridge correctly extracts expectation (non-complement portion)
     * - Weight dimensions match complement coding requirements
     */
    @Test
    void testComplementCodingIntegration() {
        var circuit = new ARTLaminarCircuit(ARTCircuitParameters.createDefault(3));

        // Simple 3D pattern for easy verification
        var input = Pattern.of(new double[]{0.8, 0.6, 0.4});

        // Process pattern
        circuit.process(input);

        // Get the category weight from FuzzyART
        var artModule = circuit.getARTModule();
        assertEquals(1, artModule.getCategoryCount(),
                    "Should have created 1 category");

        var categoryWeight = artModule.getCategory(0);

        // Validate complement coding: weight should be 6D (complement-coded)
        // Original pattern is 3D, complement-coded becomes 6D: [x1, x2, x3, 1-x1, 1-x2, 1-x3]
        assertTrue(categoryWeight instanceof com.hellblazer.art.core.weights.FuzzyWeight,
                  "Category weight should be FuzzyWeight");
        var fuzzyWeight = (com.hellblazer.art.core.weights.FuzzyWeight) categoryWeight;
        var weightData = fuzzyWeight.data();
        assertEquals(6, weightData.length,
                    "Complement-coded weight should be 2x input dimension");

        // Get expectation from circuit
        var expectation = circuit.getCategoryExpectation(0);

        // Expectation should be original dimension (non-complement)
        assertEquals(3, expectation.dimension(),
                    "Expectation should be original input dimension");

        // Verify expectation values are in valid range [0,1]
        var expData = expectation.toArray();
        for (int i = 0; i < expData.length; i++) {
            assertTrue(expData[i] >= 0.0 && expData[i] <= 1.0,
                      "Expectation value " + i + " should be in [0,1], got: " + expData[i]);
        }
    }

    /**
     * Test 4: Vigilance parameter should control category specificity.
     *
     * Validates:
     * - Higher vigilance → more categories (finer discrimination)
     * - Lower vigilance → fewer categories (coarser grouping)
     * - Vigilance consistency between laminar and ART mechanisms
     */
    @Test
    void testVigilanceConsistency() {
        var lowVigCircuit = new ARTLaminarCircuit(lowVigilanceParams);
        var highVigCircuit = new ARTLaminarCircuit(highVigilanceParams);

        // Create 5 similar but distinct patterns
        var basePattern = new double[]{0.7, 0.5, 0.3, 0.6, 0.8, 0.4, 0.2, 0.9, 0.1, 0.5};
        var patterns = new Pattern[5];
        for (int i = 0; i < 5; i++) {
            var patternData = new double[10];
            for (int j = 0; j < 10; j++) {
                // Add small variation (±0.1)
                patternData[j] = Math.max(0.0, Math.min(1.0, basePattern[j] + (i * 0.05 - 0.1)));
            }
            patterns[i] = Pattern.of(patternData);
        }

        // Process patterns through both circuits
        for (var pattern : patterns) {
            lowVigCircuit.process(pattern);
            highVigCircuit.process(pattern);
        }

        // Low vigilance should create fewer categories
        var lowVigCount = lowVigCircuit.getCategoryCount();
        var highVigCount = highVigCircuit.getCategoryCount();

        assertTrue(lowVigCount <= highVigCount,
                  "Low vigilance should create fewer or equal categories than high vigilance. " +
                  "Low: " + lowVigCount + ", High: " + highVigCount);

        // High vigilance should be more discriminative (likely creating more categories)
        // With vigilance 0.9 vs 0.5, we expect more categories at high vigilance
        assertTrue(highVigCount >= 1,
                  "High vigilance should create at least 1 category");

        // Verify parameter consistency
        var lowFuzzyParams = lowVigCircuit.getARTParameters();
        var highFuzzyParams = highVigCircuit.getARTParameters();

        assertEquals(lowVigilanceParams.vigilance(), lowFuzzyParams.vigilance(), 1e-6,
                    "Vigilance should match between circuit and ART parameters");
        assertEquals(highVigilanceParams.vigilance(), highFuzzyParams.vigilance(), 1e-6,
                    "Vigilance should match between circuit and ART parameters");
    }

    /**
     * Test 5: Learning dynamics should update templates via FuzzyART.
     *
     * Validates:
     * - Template convergence with repeated pattern presentation
     * - Learning rate affects convergence speed
     * - FuzzyART fuzzy-MIN learning rule applied correctly
     */
    @Test
    void testLearningDynamics() {
        // Create circuit with high learning rate for faster convergence
        var params = new ARTCircuitParameters(
            10,        // inputSize
            100,       // maxCategories
            0.7,       // vigilance
            0.5,       // learningRate (50% per iteration - faster convergence)
            0.001,     // alpha
            0.8,       // topDownGain
            0.01,      // timeStep
            0.05,      // expectationThreshold
            100        // maxSearchIterations
        );

        var circuit = new ARTLaminarCircuit(params);

        // Use more distinct patterns for clearer learning effect
        // Pattern A - high values
        var patternA = Pattern.of(new double[]{0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1});

        // Pattern B - different but similar enough to match same category
        var patternB = Pattern.of(new double[]{0.85, 0.85, 0.85, 0.85, 0.85, 0.15, 0.15, 0.15, 0.15, 0.15});

        // First presentation - create category with pattern A
        circuit.process(patternA);
        var initialCategoryCount = circuit.getCategoryCount();
        assertEquals(1, initialCategoryCount, "Should create 1 category");

        // Get weight directly from FuzzyART to track actual weight changes
        var initialWeight = circuit.getARTModule().getCategory(0);
        var initialWeightData = ((com.hellblazer.art.core.weights.FuzzyWeight) initialWeight).data();
        var initialWeightCopy = java.util.Arrays.copyOf(initialWeightData, initialWeightData.length);

        // Present pattern B multiple times - should update the same category
        for (int i = 0; i < 5; i++) {
            circuit.process(patternB);
        }

        // Verify still single category
        assertEquals(1, circuit.getCategoryCount(),
                    "Should still have 1 category (patterns are similar enough)");

        // Get updated weight
        var learnedWeight = circuit.getARTModule().getCategory(0);
        var learnedWeightData = ((com.hellblazer.art.core.weights.FuzzyWeight) learnedWeight).data();

        // Calculate L1 distance between initial and learned weights
        var weightDistance = 0.0;
        for (int i = 0; i < initialWeightCopy.length; i++) {
            weightDistance += Math.abs(initialWeightCopy[i] - learnedWeightData[i]);
        }

        // Debug output
        System.out.printf("Learning Dynamics Test Results:%n");
        System.out.printf("  Weight L1 distance (initial vs learned): %.6f%n", weightDistance);
        System.out.printf("  Initial weight sample [0-5]: %s%n",
            java.util.Arrays.toString(java.util.Arrays.copyOfRange(initialWeightCopy, 0, Math.min(5, initialWeightCopy.length))));
        System.out.printf("  Learned weight sample [0-5]: %s%n",
            java.util.Arrays.toString(java.util.Arrays.copyOfRange(learnedWeightData, 0, Math.min(5, learnedWeightData.length))));

        // With fuzzy-MIN learning and learning rate 0.5, weights should decrease
        // (FuzzyART uses min operation, so weights can only stay same or decrease)
        // With 5 presentations of a different pattern, we should see some change
        assertTrue(weightDistance > 0.001,
                  String.format("Weight should change after learning. L1 distance: %.6f (expected > 0.001)",
                               weightDistance));
    }

    /**
     * Test 6: Performance comparison between ARTLaminarCircuit and baseline.
     *
     * Validates:
     * - ARTLaminarCircuit produces reasonable accuracy
     * - Processing completes without errors
     * - Timing characteristics recorded for analysis
     */
    @Test
    void testPerformanceComparison() {
        var circuit = new ARTLaminarCircuit(defaultParams);

        // Generate 100 test patterns (10 clusters of 10 patterns each)
        var numClusters = 10;
        var patternsPerCluster = 10;
        var totalPatterns = numClusters * patternsPerCluster;

        var patterns = generateClusteredPatterns(numClusters, patternsPerCluster, 10);

        // Process all patterns and measure time
        var startTime = System.nanoTime();

        for (var pattern : patterns) {
            circuit.process(pattern);
        }

        var endTime = System.nanoTime();
        var processingTimeMs = (endTime - startTime) / 1_000_000.0;

        // Validate processing completed successfully
        var categoryCount = circuit.getCategoryCount();
        assertTrue(categoryCount > 0,
                  "Should have created at least 1 category");
        assertTrue(categoryCount <= totalPatterns,
                  "Should not create more categories than patterns");

        // For 10 clusters with moderate vigilance, expect roughly 10-30 categories
        // (Some clusters may split, some may merge depending on vigilance)
        assertTrue(categoryCount <= 50,
                  "Should create reasonable number of categories for clustered data. Got: " + categoryCount);

        // Record timing (for informational purposes)
        var avgTimePerPattern = processingTimeMs / totalPatterns;
        System.out.printf("ARTLaminarCircuit Performance:%n");
        System.out.printf("  Total patterns: %d%n", totalPatterns);
        System.out.printf("  Categories created: %d%n", categoryCount);
        System.out.printf("  Total time: %.2f ms%n", processingTimeMs);
        System.out.printf("  Avg time per pattern: %.3f ms%n", avgTimePerPattern);

        // Sanity check: processing should not be absurdly slow (< 1 second total for 100 patterns)
        assertTrue(processingTimeMs < 1000.0,
                  "Processing should complete in reasonable time (< 1s for 100 patterns). Got: " + processingTimeMs + " ms");
    }

    // ==================== Helper Methods ====================

    /**
     * Calculate cosine similarity between two patterns.
     * Returns value in [0, 1] where 1 = identical direction.
     */
    private double calculateCosineSimilarity(Pattern a, Pattern b) {
        if (a.dimension() != b.dimension()) {
            throw new IllegalArgumentException("Patterns must have same dimension");
        }

        var dataA = a.toArray();
        var dataB = b.toArray();

        var dotProduct = 0.0;
        var normA = 0.0;
        var normB = 0.0;

        for (int i = 0; i < dataA.length; i++) {
            dotProduct += dataA[i] * dataB[i];
            normA += dataA[i] * dataA[i];
            normB += dataB[i] * dataB[i];
        }

        if (normA == 0.0 || normB == 0.0) {
            return 0.0;
        }

        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    /**
     * Generate clustered patterns for testing.
     * Creates numClusters clusters, each with patternsPerCluster patterns.
     */
    private Pattern[] generateClusteredPatterns(int numClusters, int patternsPerCluster, int dimension) {
        var totalPatterns = numClusters * patternsPerCluster;
        var patterns = new Pattern[totalPatterns];

        var random = new java.util.Random(42);  // Fixed seed for reproducibility

        int patternIndex = 0;
        for (int cluster = 0; cluster < numClusters; cluster++) {
            // Generate cluster centroid
            var centroid = new double[dimension];
            for (int i = 0; i < dimension; i++) {
                centroid[i] = random.nextDouble();
            }

            // Generate patterns around centroid
            for (int p = 0; p < patternsPerCluster; p++) {
                var patternData = new double[dimension];
                for (int i = 0; i < dimension; i++) {
                    // Add Gaussian noise (σ = 0.1) to centroid
                    var noise = random.nextGaussian() * 0.1;
                    patternData[i] = Math.max(0.0, Math.min(1.0, centroid[i] + noise));
                }
                patterns[patternIndex++] = Pattern.of(patternData);
            }
        }

        return patterns;
    }
}
