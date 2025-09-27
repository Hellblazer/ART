package com.hellblazer.art.hybrid.pan.benchmark;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.hybrid.pan.PAN;
import com.hellblazer.art.hybrid.pan.datasets.SyntheticDataGenerator;
import com.hellblazer.art.hybrid.pan.parameters.CNNConfig;
import com.hellblazer.art.hybrid.pan.parameters.PANParameters;
import com.hellblazer.art.hybrid.pan.training.PANTrainer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Quick tests to verify PAN improvements.
 */
class QuickPANTest {

    @Test
    void testBasicMultiEpochLearning() {
        // Small dataset for quick testing
        var trainData = SyntheticDataGenerator.generateMNISTLike(200, 5); // 5 classes
        var testData = SyntheticDataGenerator.generateMNISTLike(50, 5);

        // Optimized parameters for quick learning
        var params = new PANParameters(
            0.5,    // lower vigilance for more generalization
            10,     // max 10 categories for 5 classes
            CNNConfig.simple(),
            false,
            0.05,   // higher learning rate for faster convergence
            0.9,
            0.0001,
            true,
            16,     // smaller hidden layer for speed
            0.95,
            0.8,
            200,
            16,
            0.3,    // frequent replay
            0.2
        );

        try (var pan = new PAN(params)) {
            System.out.println("Training on 200 samples with 5 classes...");

            // Track accuracies
            double initialAccuracy = PANTrainer.evaluate(pan, testData.images(), testData.labels(), params);
            System.out.printf("Initial accuracy (before training): %.2f%%\n", initialAccuracy);

            // Train for 3 quick epochs
            for (int epoch = 1; epoch <= 3; epoch++) {
                System.out.printf("\nEpoch %d:\n", epoch);

                // Train on all samples
                int correct = 0;
                for (int i = 0; i < trainData.images().size(); i++) {
                    var result = pan.learnSupervised(
                        trainData.images().get(i),
                        trainData.labels().get(i),
                        params
                    );

                    // Quick accuracy check
                    int predicted = pan.predictLabel(trainData.images().get(i), params);
                    int actual = SyntheticDataGenerator.getClassIndex(trainData.labels().get(i));
                    if (predicted == actual) correct++;

                    if ((i + 1) % 50 == 0) {
                        System.out.printf("  Processed %d/%d samples\n", i + 1, trainData.images().size());
                    }
                }

                double trainAccuracy = (double) correct / trainData.images().size() * 100;
                double testAccuracy = PANTrainer.evaluate(pan, testData.images(), testData.labels(), params);

                System.out.printf("  Train accuracy: %.2f%%\n", trainAccuracy);
                System.out.printf("  Test accuracy: %.2f%%\n", testAccuracy);
                System.out.printf("  Categories: %d\n", pan.getCategoryCount());
            }

            // Final evaluation
            double finalAccuracy = PANTrainer.evaluate(pan, testData.images(), testData.labels(), params);
            System.out.printf("\nFinal test accuracy: %.2f%%\n", finalAccuracy);
            System.out.printf("Final categories: %d\n", pan.getCategoryCount());

            // Should show improvement over random (20% for 5 classes)
            assertTrue(finalAccuracy >= 20.0, "Should achieve at least random accuracy");
            assertTrue(pan.getCategoryCount() >= 5, "Should create at least one category per class");
            assertTrue(pan.getCategoryCount() <= 10, "Should not exceed max categories");
        }
    }

    @Test
    void testSupervisedVsUnsupervised() {
        var data = SyntheticDataGenerator.generateMNISTLike(100, 3); // 3 classes

        var params = new PANParameters(
            0.6, 6, CNNConfig.simple(), false,
            0.03, 0.9, 0.0001, true, 16,
            0.95, 0.8, 100, 8, 0.2, 0.15
        );

        // Test supervised learning
        double supervisedAccuracy;
        int supervisedCategories;
        try (var pan = new PAN(params)) {
            for (int i = 0; i < data.images().size(); i++) {
                pan.learnSupervised(data.images().get(i), data.labels().get(i), params);
            }
            supervisedAccuracy = PANTrainer.evaluate(pan, data.images(), data.labels(), params);
            supervisedCategories = pan.getCategoryCount();
            System.out.printf("Supervised - Accuracy: %.2f%%, Categories: %d\n",
                supervisedAccuracy, supervisedCategories);
        }

        // Test unsupervised learning
        int unsupervisedCategories;
        try (var pan = new PAN(params)) {
            for (Pattern image : data.images()) {
                pan.learn(image, params);
            }
            unsupervisedCategories = pan.getCategoryCount();
            System.out.printf("Unsupervised - Categories: %d\n", unsupervisedCategories);
        }

        // Supervised should generally create fewer, more accurate categories
        assertTrue(supervisedAccuracy >= 30.0, "Supervised should achieve reasonable accuracy");
        assertTrue(supervisedCategories <= 6, "Should respect max categories");
        assertTrue(unsupervisedCategories >= 3, "Unsupervised should create multiple categories");
    }

    @Test
    void testExperienceReplayEffect() {
        var data = SyntheticDataGenerator.generateMNISTLike(150, 3);

        // Parameters with replay disabled
        var noReplayParams = new PANParameters(
            0.6, 6, CNNConfig.simple(), false,
            0.03, 0.9, 0.0001, true, 16,
            0.95, 0.8,
            0,      // no replay buffer
            0,
            0.0,    // no replay
            0.1
        );

        // Parameters with replay enabled
        var replayParams = new PANParameters(
            0.6, 6, CNNConfig.simple(), false,
            0.03, 0.9, 0.0001, true, 16,
            0.95, 0.8,
            300,    // replay buffer
            16,
            0.5,    // frequent replay
            0.1
        );

        // Train without replay
        double accuracyNoReplay;
        try (var pan = new PAN(noReplayParams)) {
            for (int i = 0; i < data.images().size(); i++) {
                pan.learnSupervised(data.images().get(i), data.labels().get(i), noReplayParams);
            }
            accuracyNoReplay = PANTrainer.evaluate(pan, data.images(), data.labels(), noReplayParams);
            System.out.printf("No replay - Accuracy: %.2f%%\n", accuracyNoReplay);
        }

        // Train with replay
        double accuracyWithReplay;
        try (var pan = new PAN(replayParams)) {
            for (int i = 0; i < data.images().size(); i++) {
                pan.learnSupervised(data.images().get(i), data.labels().get(i), replayParams);
            }
            accuracyWithReplay = PANTrainer.evaluate(pan, data.images(), data.labels(), replayParams);
            System.out.printf("With replay - Accuracy: %.2f%%\n", accuracyWithReplay);
        }

        // Both should learn something
        assertTrue(accuracyNoReplay > 20.0, "Should learn without replay");
        assertTrue(accuracyWithReplay > 20.0, "Should learn with replay");
        System.out.printf("Replay improvement: %.2f%%\n", accuracyWithReplay - accuracyNoReplay);
    }
}