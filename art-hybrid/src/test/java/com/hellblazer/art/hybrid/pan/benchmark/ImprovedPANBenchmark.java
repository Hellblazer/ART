package com.hellblazer.art.hybrid.pan.benchmark;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.hybrid.pan.PAN;
import com.hellblazer.art.hybrid.pan.datasets.SyntheticDataGenerator;
import com.hellblazer.art.hybrid.pan.parameters.CNNConfig;
import com.hellblazer.art.hybrid.pan.parameters.PANParameters;
import com.hellblazer.art.hybrid.pan.training.PANTrainer;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Improved benchmarks using multi-epoch training.
 */
class ImprovedPANBenchmark {

    @Test
    void testMultiEpochTraining() {
        // Generate synthetic MNIST-like data
        var trainData = SyntheticDataGenerator.generateMNISTLike(1000, 10);
        var testData = SyntheticDataGenerator.generateMNISTLike(200, 10);

        // Create optimized parameters
        var params = new PANParameters(
            0.6,    // vigilance
            15,     // maxCategories
            CNNConfig.simple(),
            false,  // no pretraining
            0.02,   // higher learning rate
            0.9,    // momentum
            0.0001, // weight decay
            true,   // allow negative weights
            32,     // smaller hidden layer for faster training
            0.95,   // STM decay
            0.8,    // LTM consolidation
            500,    // replay buffer
            16,     // replay batch
            0.2,    // higher replay frequency
            0.15    // bias factor
        );

        try (var pan = new PAN(params)) {
            // Train with epochs
            var result = PANTrainer.trainWithEpochs(
                pan,
                trainData.images(),
                trainData.labels(),
                testData.images(),
                testData.labels(),
                params,
                10,     // max epochs
                50.0,   // early stopping at 50%
                true    // verbose
            );

            System.out.println("\n=== Training Complete ===");
            System.out.printf("Epochs: %d\n", result.epochsCompleted());
            System.out.printf("Best accuracy: %.2f%% (epoch %d)\n",
                result.bestAccuracy(), result.bestEpoch());
            System.out.printf("Final categories: %d\n", result.finalCategories());
            System.out.printf("Training time: %.2f seconds\n",
                result.trainingTimeMs() / 1000.0);

            // Should achieve better accuracy with multi-epoch training
            assertTrue(result.bestAccuracy() >= 15.0,
                "Should achieve at least 15% with multi-epoch training");
            assertTrue(result.finalCategories() <= 15,
                "Should not exceed max categories");
        }
    }

    @Test
    void testHyperparameterTuning() {
        // Smaller dataset for faster testing
        var trainData = SyntheticDataGenerator.generateMNISTLike(500, 10);
        var valData = SyntheticDataGenerator.generateMNISTLike(100, 10);

        // Create parameter configurations to test
        List<PANParameters> configs = new ArrayList<>();

        // Config 1: Low vigilance, high learning rate
        configs.add(new PANParameters(
            0.5, 20, CNNConfig.simple(), false,
            0.05, 0.9, 0.0001, true, 32,
            0.95, 0.8, 200, 16, 0.1, 0.1
        ));

        // Config 2: Medium vigilance, medium learning rate
        configs.add(new PANParameters(
            0.7, 15, CNNConfig.simple(), false,
            0.02, 0.9, 0.0001, true, 64,
            0.95, 0.8, 500, 32, 0.2, 0.15
        ));

        // Config 3: High vigilance, low learning rate
        configs.add(new PANParameters(
            0.85, 10, CNNConfig.simple(), false,
            0.01, 0.95, 0.0001, true, 32,
            0.95, 0.8, 300, 16, 0.15, 0.2
        ));

        // Search for best parameters
        var searchResult = PANTrainer.hyperparameterSearch(
            trainData.images(),
            trainData.labels(),
            valData.images(),
            valData.labels(),
            configs,
            5,      // max epochs per config
            true    // verbose
        );

        System.out.println("\n=== Hyperparameter Search Complete ===");
        System.out.printf("Best accuracy: %.2f%%\n", searchResult.bestAccuracy());
        System.out.printf("Best config: vigilance=%.2f, lr=%.4f\n",
            searchResult.bestParameters().vigilance(),
            searchResult.bestParameters().learningRate());

        // Print all results
        System.out.println("\nAll configurations:");
        searchResult.allResults().forEach((params, acc) ->
            System.out.printf("  v=%.2f, lr=%.4f -> %.2f%%\n",
                params.vigilance(), params.learningRate(), acc)
        );

        assertNotNull(searchResult.bestParameters());
        assertTrue(searchResult.bestAccuracy() > 10.0,
            "Should find configuration better than random");
    }

    @Test
    void testCrossValidation() {
        // Generate data
        var data = SyntheticDataGenerator.generateMNISTLike(500, 10);

        // Parameters for CV
        var params = new PANParameters(
            0.65, 15, CNNConfig.simple(), false,
            0.02, 0.9, 0.0001, true, 32,
            0.95, 0.8, 300, 16, 0.15, 0.15
        );

        // Perform 3-fold cross-validation
        var cvResult = PANTrainer.crossValidate(
            data.images(),
            data.labels(),
            params,
            3,      // k-folds
            5,      // max epochs
            false   // not verbose for faster execution
        );

        System.out.println("\n=== Cross-Validation Results ===");
        System.out.println(cvResult);
        System.out.println("Fold accuracies: " + cvResult.foldAccuracies());

        assertTrue(cvResult.meanAccuracy() > 10.0,
            "Mean CV accuracy should be better than random");
        assertTrue(cvResult.stdAccuracy() < 20.0,
            "Standard deviation should be reasonable");
    }

    @Test
    void testIncrementalAccuracyImprovement() {
        var data = SyntheticDataGenerator.generateMNISTLike(300, 10);

        var params = new PANParameters(
            0.6, 12, CNNConfig.simple(), false,
            0.03, 0.9, 0.0001, true, 32,
            0.95, 0.8, 200, 16, 0.25, 0.1
        );

        try (var pan = new PAN(params)) {
            List<Double> accuracies = new ArrayList<>();

            // Track accuracy over incremental training
            for (int i = 50; i <= 300; i += 50) {
                // Train on subset
                for (int j = i - 50; j < i; j++) {
                    pan.learnSupervised(
                        data.images().get(j),
                        data.labels().get(j),
                        params
                    );
                }

                // Evaluate on all data seen so far
                double accuracy = PANTrainer.evaluate(
                    pan,
                    data.images().subList(0, i),
                    data.labels().subList(0, i),
                    params
                );

                accuracies.add(accuracy);
                System.out.printf("After %d samples: %.2f%% accuracy, %d categories\n",
                    i, accuracy, pan.getCategoryCount());
            }

            // Check that we're learning something
            double firstAccuracy = accuracies.get(0);
            double lastAccuracy = accuracies.get(accuracies.size() - 1);
            System.out.printf("\nFirst accuracy: %.2f%%, Last accuracy: %.2f%%\n",
                firstAccuracy, lastAccuracy);

            // At least one accuracy should be significantly better than random (10%)
            boolean hasLearning = accuracies.stream().anyMatch(acc -> acc > 20.0);
            assertTrue(hasLearning, "Should show learning with at least 20% accuracy");

            // Should create reasonable number of categories
            assertTrue(pan.getCategoryCount() >= 5,
                "Should create multiple categories");
            assertTrue(pan.getCategoryCount() <= params.maxCategories(),
                "Should respect max categories limit");
        }
    }

    @Test
    void testContinualLearningWithReplay() {
        // Test learning classes sequentially with experience replay
        var data = SyntheticDataGenerator.generateMNISTLike(500, 5); // 5 classes

        // Parameters with strong experience replay
        var params = new PANParameters(
            0.65, 10, CNNConfig.simple(), false,
            0.02, 0.9, 0.0001, true, 32,
            0.95, 0.8,
            1000,   // large replay buffer
            32,     // larger replay batch
            0.3,    // frequent replay
            0.1
        );

        try (var pan = new PAN(params)) {
            // Group samples by class
            List<List<Integer>> classSamples = new ArrayList<>();
            for (int c = 0; c < 5; c++) {
                classSamples.add(new ArrayList<>());
            }

            for (int i = 0; i < data.images().size(); i++) {
                int classIdx = SyntheticDataGenerator.getClassIndex(data.labels().get(i));
                if (classIdx >= 0 && classIdx < 5) {
                    classSamples.get(classIdx).add(i);
                }
            }

            // Learn classes one by one
            List<Double> accuraciesAfterEachClass = new ArrayList<>();

            for (int c = 0; c < 5; c++) {
                System.out.printf("\nLearning class %d...\n", c);

                // Learn this class
                for (int idx : classSamples.get(c)) {
                    pan.learnSupervised(
                        data.images().get(idx),
                        data.labels().get(idx),
                        params
                    );
                }

                // Test on all classes learned so far
                List<Pattern> testInputs = new ArrayList<>();
                List<Pattern> testLabels = new ArrayList<>();
                for (int tc = 0; tc <= c; tc++) {
                    for (int idx : classSamples.get(tc).subList(0,
                            Math.min(20, classSamples.get(tc).size()))) {
                        testInputs.add(data.images().get(idx));
                        testLabels.add(data.labels().get(idx));
                    }
                }

                double accuracy = PANTrainer.evaluate(pan, testInputs, testLabels, params);
                accuraciesAfterEachClass.add(accuracy);
                System.out.printf("Accuracy on classes 0-%d: %.2f%%\n", c, accuracy);
            }

            // With experience replay, should maintain reasonable accuracy
            double finalAccuracy = accuraciesAfterEachClass.get(4);
            System.out.printf("\nFinal accuracy on all classes: %.2f%%\n", finalAccuracy);

            assertTrue(finalAccuracy > 15.0,
                "Should maintain knowledge with experience replay");
        }
    }
}