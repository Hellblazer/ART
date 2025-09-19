package com.hellblazer.art.performance;

import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import com.hellblazer.art.core.*;
import com.hellblazer.art.core.algorithms.*;
import com.hellblazer.art.core.parameters.*;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.performance.algorithms.*;

/**
 * Comprehensive cross-validation tests between vanilla ART algorithms and their vectorized counterparts.
 * Tests verify that vectorized implementations produce statistically equivalent results to vanilla versions.
 */
@DisplayName("Cross-Validation Tests: Vanilla vs Vectorized ART")
class CrossValidationTest {

    private static final double TRAINING_AGREEMENT_THRESHOLD = 0.80; // 80% training agreement
    private static final double VALIDATION_AGREEMENT_THRESHOLD = 0.75; // 75% validation agreement
    private static final int NUM_PATTERNS = 100;
    private static final int TRAINING_SIZE = 80;
    private static final int VALIDATION_SIZE = 20;
    private static final long SEED = 42L;

    @Test
    @DisplayName("FuzzyART vs VectorizedFuzzyART Cross-Validation")
    void testFuzzyARTCrossValidation() throws Exception {
        var patterns = generateTestPatterns(NUM_PATTERNS, 4, SEED);
        var trainingPatterns = patterns.subList(0, TRAINING_SIZE);
        var validationPatterns = patterns.subList(TRAINING_SIZE, NUM_PATTERNS);

        // Core FuzzyART
        var coreFuzzyART = new FuzzyART();
        var coreParams = new FuzzyParameters(0.7, 0.1, 0.9);

        // Vectorized FuzzyART
        var vectorizedParams = new VectorizedParameters(
            0.7,    // vigilanceThreshold
            0.9,    // learningRate  
            0.1,    // alpha
            1,      // parallelismLevel (single thread for fair comparison)
            1000,   // parallelThreshold
            1000,   // maxCacheSize
            true,   // enableSIMD
            true,   // enableJOML
            0.8     // memoryOptimizationThreshold
        );
        var vectorizedFuzzyART = new VectorizedFuzzyART(vectorizedParams);

        try {
            // Training phase - learn on both algorithms
            var coreTrainingResults = new ArrayList<Integer>();
            var vectorizedTrainingResults = new ArrayList<Integer>();

            for (var pattern : trainingPatterns) {
                var coreResult = coreFuzzyART.learn(pattern, coreParams);
                if (coreResult instanceof ActivationResult.Success coreSuccess) {
                    coreTrainingResults.add(coreSuccess.categoryIndex());
                } else {
                    coreTrainingResults.add(-1);
                }

                var vectorizedResult = vectorizedFuzzyART.learn(pattern, vectorizedParams);
                if (vectorizedResult instanceof ActivationResult.Success vectorizedSuccess) {
                    vectorizedTrainingResults.add(vectorizedSuccess.categoryIndex());
                } else {
                    vectorizedTrainingResults.add(-1);
                }
            }

            // Validation phase - predict on both algorithms
            var coreValidationResults = new ArrayList<Integer>();
            var vectorizedValidationResults = new ArrayList<Integer>();

            for (var pattern : validationPatterns) {
                var coreResult = coreFuzzyART.predict(pattern, coreParams);
                if (coreResult instanceof ActivationResult.Success coreSuccess) {
                    coreValidationResults.add(coreSuccess.categoryIndex());
                } else {
                    coreValidationResults.add(-1); // No match
                }

                var vectorizedResult = vectorizedFuzzyART.predict(pattern, vectorizedParams);
                assertTrue(vectorizedResult instanceof ActivationResult.Success);
                var vectorizedSuccess = (ActivationResult.Success) vectorizedResult;
                vectorizedValidationResults.add(vectorizedSuccess.categoryIndex());
            }

            // Compare results
            var trainingAgreement = calculateAgreement(coreTrainingResults, vectorizedTrainingResults);
            var validationAgreement = calculateAgreement(coreValidationResults, vectorizedValidationResults);

            System.out.printf("FuzzyART - Training agreement: %.2f%%, Validation agreement: %.2f%%%n",
                             trainingAgreement * 100, validationAgreement * 100);
            System.out.printf("Core categories: %d, Vectorized categories: %d%n",
                             coreFuzzyART.getCategoryCount(), vectorizedFuzzyART.getCategoryCount());

            assertTrue(trainingAgreement >= TRAINING_AGREEMENT_THRESHOLD,
                      String.format("Training agreement %.2f%% below threshold %.2f%%",
                                   trainingAgreement * 100, TRAINING_AGREEMENT_THRESHOLD * 100));
            assertTrue(validationAgreement >= VALIDATION_AGREEMENT_THRESHOLD,
                      String.format("Validation agreement %.2f%% below threshold %.2f%%",
                                   validationAgreement * 100, VALIDATION_AGREEMENT_THRESHOLD * 100));

        } finally {
            vectorizedFuzzyART.close();
        }
    }

    @Test
    @DisplayName("GaussianART vs VectorizedGaussianART Cross-Validation")
    void testGaussianARTCrossValidation() throws Exception {
        var patterns = generateTestPatterns(NUM_PATTERNS, 2, SEED + 1);
        var trainingPatterns = patterns.subList(0, TRAINING_SIZE);
        var validationPatterns = patterns.subList(TRAINING_SIZE, NUM_PATTERNS);

        // Core GaussianART
        var coreGaussianART = new GaussianART();
        var sigmaInit = new double[]{0.5, 0.5}; // 2D
        var coreParams = new GaussianParameters(0.8, sigmaInit);

        // Vectorized GaussianART
        var vectorizedParams = new VectorizedGaussianParameters(
            0.8,    // vigilance
            0.1,    // gamma (learning rate)
            1.0,    // rho_a (variance adjustment factor)
            0.5,    // rho_b (minimum variance)
            1,      // parallelismLevel
            true    // enableSIMD
        );
        var vectorizedGaussianART = new VectorizedGaussianART(vectorizedParams);

        try {
            // Training phase
            var coreTrainingResults = new ArrayList<Integer>();
            var vectorizedTrainingResults = new ArrayList<Integer>();

            for (var pattern : trainingPatterns) {
                var coreResult = coreGaussianART.learn(pattern, coreParams);
                coreTrainingResults.add(((ActivationResult.Success)coreResult).categoryIndex());

                var vectorizedResult = vectorizedGaussianART.learn(pattern, vectorizedParams);
                assertTrue(vectorizedResult instanceof ActivationResult.Success);
                var success = (ActivationResult.Success) vectorizedResult;
                vectorizedTrainingResults.add(success.categoryIndex());
            }

            // Validation phase
            var coreValidationResults = new ArrayList<Integer>();
            var vectorizedValidationResults = new ArrayList<Integer>();

            for (var pattern : validationPatterns) {
                var coreResult = coreGaussianART.predict(pattern, coreParams);
                if (coreResult instanceof ActivationResult.Success coreSuccess) {
                    coreValidationResults.add(coreSuccess.categoryIndex());
                } else {
                    coreValidationResults.add(-1); // No match
                }

                var vectorizedResult = vectorizedGaussianART.predict(pattern, vectorizedParams);
                assertTrue(vectorizedResult instanceof ActivationResult.Success);
                var vectorizedSuccess = (ActivationResult.Success) vectorizedResult;
                vectorizedValidationResults.add(vectorizedSuccess.categoryIndex());
            }

            // Compare results
            var trainingAgreement = calculateAgreement(coreTrainingResults, vectorizedTrainingResults);
            var validationAgreement = calculateAgreement(coreValidationResults, vectorizedValidationResults);

            System.out.printf("GaussianART - Training agreement: %.2f%%, Validation agreement: %.2f%%%n",
                             trainingAgreement * 100, validationAgreement * 100);
            System.out.printf("Core categories: %d, Vectorized categories: %d%n",
                             coreGaussianART.getCategoryCount(), vectorizedGaussianART.getCategoryCount());

            assertTrue(trainingAgreement >= TRAINING_AGREEMENT_THRESHOLD,
                      String.format("Training agreement %.2f%% below threshold %.2f%%",
                                   trainingAgreement * 100, TRAINING_AGREEMENT_THRESHOLD * 100));
            assertTrue(validationAgreement >= VALIDATION_AGREEMENT_THRESHOLD,
                      String.format("Validation agreement %.2f%% below threshold %.2f%%",
                                   validationAgreement * 100, VALIDATION_AGREEMENT_THRESHOLD * 100));

        } finally {
            vectorizedGaussianART.close();
        }
    }

    /**
     * Generate test patterns for cross-validation.
     */
    private List<Pattern> generateTestPatterns(int numPatterns, int dimensions, long seed) {
        var random = new Random(seed);
        var patterns = new ArrayList<Pattern>(numPatterns);
        
        for (int i = 0; i < numPatterns; i++) {
            var values = new double[dimensions];
            for (int j = 0; j < dimensions; j++) {
                values[j] = random.nextDouble(); // Random values between 0 and 1
            }
            patterns.add(Pattern.of(values));
        }
        
        return patterns;
    }

    /**
     * Calculate the agreement percentage between two result lists.
     */
    private double calculateAgreement(List<Integer> results1, List<Integer> results2) {
        if (results1.size() != results2.size()) {
            throw new IllegalArgumentException("Result lists must have the same size");
        }
        
        int agreements = 0;
        for (int i = 0; i < results1.size(); i++) {
            if (results1.get(i).equals(results2.get(i))) {
                agreements++;
            }
        }
        
        return (double) agreements / results1.size();
    }
}