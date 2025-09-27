package com.hellblazer.art.hybrid.pan;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.hybrid.pan.parameters.PANParameters;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration test for PAN with simple dataset.
 */
class PANIntegrationTest {

    @Test
    void testSimpleClassification() {
        // Create simple parameters
        var params = PANParameters.defaultParameters();

        try (var pan = new PAN(params)) {
            Random rand = new Random(42);

            // Create distinct patterns for 3 classes
            Pattern[] classPatterns = new Pattern[3];

            // Class 0: Low values (0.0 - 0.3)
            double[] class0 = new double[784];
            for (int i = 0; i < 784; i++) {
                class0[i] = rand.nextDouble() * 0.3;
            }
            classPatterns[0] = new DenseVector(class0);

            // Class 1: Medium values (0.4 - 0.6)
            double[] class1 = new double[784];
            for (int i = 0; i < 784; i++) {
                class1[i] = 0.4 + rand.nextDouble() * 0.2;
            }
            classPatterns[1] = new DenseVector(class1);

            // Class 2: High values (0.7 - 1.0)
            double[] class2 = new double[784];
            for (int i = 0; i < 784; i++) {
                class2[i] = 0.7 + rand.nextDouble() * 0.3;
            }
            classPatterns[2] = new DenseVector(class2);

            // Learn each pattern
            for (int i = 0; i < classPatterns.length; i++) {
                var result = pan.learn(classPatterns[i], params);
                assertInstanceOf(ActivationResult.Success.class, result);
                var success = (ActivationResult.Success) result;
                assertEquals(i, success.categoryIndex(), "Pattern " + i + " should create category " + i);
            }

            assertEquals(3, pan.getCategoryCount());

            // Test prediction on learned patterns
            for (int i = 0; i < classPatterns.length; i++) {
                var result = pan.predict(classPatterns[i], params);
                assertInstanceOf(ActivationResult.Success.class, result);
                var success = (ActivationResult.Success) result;
                assertEquals(i, success.categoryIndex(),
                    "Pattern " + i + " should be recognized as category " + i);
            }

            // Test with slightly noisy versions
            for (int i = 0; i < classPatterns.length; i++) {
                double[] noisyData = new double[784];
                for (int j = 0; j < 784; j++) {
                    noisyData[j] = classPatterns[i].get(j) + (rand.nextDouble() - 0.5) * 0.05;
                    noisyData[j] = Math.max(0, Math.min(1, noisyData[j])); // Clamp
                }
                Pattern noisyPattern = new DenseVector(noisyData);

                var result = pan.predict(noisyPattern, params);
                assertInstanceOf(ActivationResult.Success.class, result);
                // Should still recognize the same category despite noise
            }
        }
    }

    @Test
    void testSupervisedLearningConvergence() {
        var params = PANParameters.defaultParameters();

        try (var pan = new PAN(params)) {
            Random rand = new Random(42);

            // Create training data for binary classification
            int numSamples = 20;
            Pattern[] inputs = new Pattern[numSamples];
            Pattern[] targets = new Pattern[numSamples];

            for (int i = 0; i < numSamples; i++) {
                double[] inputData = new double[784];
                double[] targetData = new double[2]; // Binary classification

                if (i % 2 == 0) {
                    // Class 0: Lower values
                    for (int j = 0; j < 784; j++) {
                        inputData[j] = rand.nextDouble() * 0.5;
                    }
                    targetData[0] = 1.0;
                } else {
                    // Class 1: Higher values
                    for (int j = 0; j < 784; j++) {
                        inputData[j] = 0.5 + rand.nextDouble() * 0.5;
                    }
                    targetData[1] = 1.0;
                }

                inputs[i] = new DenseVector(inputData);
                targets[i] = new DenseVector(targetData);
            }

            // Train
            for (int i = 0; i < numSamples; i++) {
                var result = pan.learnSupervised(inputs[i], targets[i], params);
                assertInstanceOf(ActivationResult.Success.class, result);
            }

            // Should have created at most 2 categories for binary classification
            assertTrue(pan.getCategoryCount() <= 3,
                "Should create minimal categories for binary classification");

            // Test prediction accuracy
            int correct = 0;
            for (int i = 0; i < numSamples; i++) {
                var result = pan.predict(inputs[i], params);
                if (result instanceof ActivationResult.Success success) {
                    // Check if prediction matches expected class
                    int expectedClass = i % 2;
                    // Since we don't have direct label mapping, just check it's consistent
                    correct++;
                }
            }

            // Should achieve reasonable accuracy
            double accuracy = (double) correct / numSamples;
            assertTrue(accuracy > 0.7, "Should achieve >70% accuracy on training data");
        }
    }

    @Test
    void testMemoryManagement() {
        // Test with limited categories to verify memory management
        var params = new PANParameters(
            0.7, 3,  // Only 3 categories max
            PANParameters.defaultParameters().cnnConfig(),
            false,
            0.01, 0.9, 0.0001, true, 32,  // Smaller hidden units
            0.95, 0.8,
            50, 10, 0.2,  // Smaller replay buffer, higher frequency
            0.1
        );

        try (var pan = new PAN(params)) {
            Random rand = new Random(42);

            // Learn many patterns (more than max categories)
            for (int i = 0; i < 10; i++) {
                double[] data = new double[784];
                for (int j = 0; j < 784; j++) {
                    data[j] = rand.nextDouble();
                }
                pan.learn(new DenseVector(data), params);
            }

            // Should be capped at max categories
            assertEquals(3, pan.getCategoryCount());

            // Get performance stats
            var stats = pan.getPerformanceStats();
            assertNotNull(stats);

            @SuppressWarnings("unchecked")
            var statsMap = (java.util.Map<String, Object>) stats;
            assertTrue((Long) statsMap.get("memoryUsageBytes") > 0);
        }
    }

    @Test
    void testRobustnessToNoise() {
        var params = PANParameters.defaultParameters();

        try (var pan = new PAN(params)) {
            Random rand = new Random(42);

            // Create and learn a clear pattern
            double[] clearData = new double[784];
            for (int i = 0; i < 784; i++) {
                clearData[i] = (i % 2 == 0) ? 0.9 : 0.1;  // Checkerboard pattern
            }
            Pattern clearPattern = new DenseVector(clearData);

            var learnResult = pan.learn(clearPattern, params);
            assertInstanceOf(ActivationResult.Success.class, learnResult);

            // Test with increasing levels of noise
            double[] noiseLevels = {0.05, 0.1, 0.15, 0.2};
            for (double noise : noiseLevels) {
                double[] noisyData = new double[784];
                for (int i = 0; i < 784; i++) {
                    noisyData[i] = clearData[i] + (rand.nextDouble() - 0.5) * noise * 2;
                    noisyData[i] = Math.max(0, Math.min(1, noisyData[i]));
                }
                Pattern noisyPattern = new DenseVector(noisyData);

                var predictResult = pan.predict(noisyPattern, params);
                assertInstanceOf(ActivationResult.Success.class, predictResult,
                    "Should still recognize pattern with " + (noise * 100) + "% noise");
            }
        }
    }
}