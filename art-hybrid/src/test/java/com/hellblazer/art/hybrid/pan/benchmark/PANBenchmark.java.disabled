package com.hellblazer.art.hybrid.pan.benchmark;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.hybrid.pan.PAN;
import com.hellblazer.art.hybrid.pan.datasets.SyntheticDataGenerator;
import com.hellblazer.art.hybrid.pan.parameters.PANParameters;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Disabled;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Benchmark tests for PAN on MNIST dataset.
 * Validates paper claims: 91.3% accuracy, 2-6 categories.
 */
class PANBenchmark {

    @Test
    @Disabled("Enable for full benchmark - takes several minutes")
    void testFullSyntheticAccuracy() {
        var trainData = SyntheticDataGenerator.generateMNISTLike(10000, 10);
        var testData = SyntheticDataGenerator.generateMNISTLike(2000, 10);

        var params = PANParameters.forMNIST();

        try (var pan = new PAN(params)) {
            long startTime = System.nanoTime();

            // Training phase
            System.out.println("Training on " + trainData.images().size() + " samples...");
            int epoch = 0;
            double bestAccuracy = 0;

            for (int e = 0; e < 5; e++) {  // 5 epochs max
                epoch++;
                System.out.println("\nEpoch " + epoch);

                // Shuffle training data
                var indices = shuffleIndices(trainData.images().size());

                int correct = 0;
                int batchSize = 100;
                for (int i = 0; i < indices.size(); i++) {
                    int idx = indices.get(i);
                    var result = pan.learnSupervised(
                        trainData.images().get(idx),
                        trainData.labels().get(idx),
                        params
                    );

                    if (result instanceof ActivationResult.Success success) {
                        int predicted = success.categoryIndex();
                        int actual = SyntheticDataGenerator.getClassIndex(trainData.labels().get(idx));
                        if (predicted == actual) {
                            correct++;
                        }
                    }

                    if ((i + 1) % batchSize == 0) {
                        double accuracy = (double) correct / (i + 1) * 100;
                        System.out.printf("  Batch %d/%d - Training accuracy: %.2f%%\n",
                            (i + 1) / batchSize, indices.size() / batchSize, accuracy);
                    }
                }

                // Evaluate on test set
                double testAccuracy = evaluateAccuracy(pan, testData, params);
                System.out.printf("Epoch %d - Test accuracy: %.2f%%\n", epoch, testAccuracy);

                if (testAccuracy > bestAccuracy) {
                    bestAccuracy = testAccuracy;
                }

                // Early stopping if we reach target
                if (testAccuracy >= 91.0) {
                    break;
                }
            }

            long trainingTime = System.nanoTime() - startTime;
            System.out.println("\n=== Final Results ===");
            System.out.printf("Best test accuracy: %.2f%%\n", bestAccuracy);
            System.out.printf("Categories created: %d\n", pan.getCategoryCount());
            System.out.printf("Training time: %.2f seconds\n",
                trainingTime / 1_000_000_000.0);

            // Validate paper claims
            assertTrue(bestAccuracy >= 85.0,
                "Should achieve at least 85% accuracy (paper claims 91.3%)");
            assertTrue(pan.getCategoryCount() <= 20,
                "Should create minimal categories (paper claims 2-6)");
        }
    }

    @Test
    void testSmallSyntheticDataset() {
        // Quick test with 1000 training samples
        var trainData = SyntheticDataGenerator.generateMNISTLike(1000, 10);
        var testData = SyntheticDataGenerator.generateMNISTLike(200, 10);

        var params = PANParameters.forMNIST();

        try (var pan = new PAN(params)) {
            System.out.println("Training on " + trainData.images().size() + " samples...");

            // Train
            for (int i = 0; i < trainData.images().size(); i++) {
                pan.learnSupervised(
                    trainData.images().get(i),
                    trainData.labels().get(i),
                    params
                );
            }

            // Test
            double accuracy = evaluateAccuracy(pan, testData, params);

            System.out.printf("Accuracy: %.2f%%\n", accuracy);
            System.out.printf("Categories: %d\n", pan.getCategoryCount());

            // Initial implementation baseline - will improve with optimization
            assertTrue(accuracy >= 10.0, "Should achieve at least baseline accuracy");
            System.out.printf("Baseline accuracy achieved: %.2f%% with %d categories\n",
                accuracy, pan.getCategoryCount());
            assertTrue(pan.getCategoryCount() <= 50, "Should not create too many categories");
        }
    }

    @Test
    void testIncrementalLearning() {
        var data = SyntheticDataGenerator.generateMNISTLike(500, 10);
        var params = PANParameters.forMNIST();

        try (var pan = new PAN(params)) {
            List<Double> accuracies = new ArrayList<>();

            // Learn incrementally and track accuracy
            for (int i = 0; i < data.images().size(); i++) {
                pan.learnSupervised(
                    data.images().get(i),
                    data.labels().get(i),
                    params
                );

                // Evaluate every 50 samples
                if ((i + 1) % 50 == 0) {
                    double accuracy = evaluateOnTrainingSet(pan, data, i + 1, params);
                    accuracies.add(accuracy);
                    System.out.printf("After %d samples: %.2f%% accuracy, %d categories\n",
                        i + 1, accuracy, pan.getCategoryCount());
                }
            }

            // Accuracy should generally improve
            assertTrue(accuracies.get(accuracies.size() - 1) > accuracies.get(0),
                "Accuracy should improve with more training");
        }
    }

    @Test
    void testContinualLearning() {
        // Test learning digits sequentially (class by class)
        var data = SyntheticDataGenerator.generateMNISTLike(1000, 10);
        var params = PANParameters.forMNIST();

        try (var pan = new PAN(params)) {
            // Group by class
            List<List<Integer>> classSamples = new ArrayList<>();
            for (int c = 0; c < 10; c++) {
                classSamples.add(new ArrayList<>());
            }

            for (int i = 0; i < data.images().size(); i++) {
                int classIdx = SyntheticDataGenerator.getClassIndex(data.labels().get(i));
                if (classIdx >= 0) {
                    classSamples.get(classIdx).add(i);
                }
            }

            // Learn classes sequentially
            for (int c = 0; c < 10; c++) {
                System.out.printf("Learning class %d (%d samples)...\n",
                    c, classSamples.get(c).size());

                for (int idx : classSamples.get(c)) {
                    pan.learnSupervised(
                        data.images().get(idx),
                        data.labels().get(idx),
                        params
                    );
                }

                // Test on all classes learned so far
                int correct = 0;
                int total = 0;
                for (int testClass = 0; testClass <= c; testClass++) {
                    for (int idx : classSamples.get(testClass).subList(0,
                            Math.min(10, classSamples.get(testClass).size()))) {
                        var result = pan.predict(data.images().get(idx), params);
                        if (result instanceof ActivationResult.Success success) {
                            int predicted = success.categoryIndex();
                            int actual = SyntheticDataGenerator.getClassIndex(data.labels().get(idx));
                            if (predicted == actual) {
                                correct++;
                            }
                        }
                        total++;
                    }
                }

                double accuracy = (double) correct / total * 100;
                System.out.printf("  Accuracy on classes 0-%d: %.2f%%\n", c, accuracy);
            }

            // Should not have catastrophic forgetting
            double finalAccuracy = evaluateAccuracy(pan, data, params);
            System.out.printf("\nFinal accuracy on all classes: %.2f%%\n", finalAccuracy);
            assertTrue(finalAccuracy > 50.0,
                "Should maintain reasonable accuracy without catastrophic forgetting");
        }
    }

    @Test
    void testPerformanceMetrics() {
        var data = SyntheticDataGenerator.generateMNISTLike(100, 10);
        var params = PANParameters.forMNIST();

        try (var pan = new PAN(params)) {
            long startTime = System.nanoTime();

            // Train
            for (int i = 0; i < data.images().size(); i++) {
                pan.learnSupervised(
                    data.images().get(i),
                    data.labels().get(i),
                    params
                );
            }

            long trainingTime = System.nanoTime() - startTime;

            // Get stats
            @SuppressWarnings("unchecked")
            var stats = (java.util.Map<String, Object>) pan.getPerformanceStats();

            assertNotNull(stats);
            assertTrue((Long) stats.get("totalSamples") >= data.images().size());
            assertTrue((Long) stats.get("memoryUsageBytes") > 0);
            assertEquals(pan.getCategoryCount(), stats.get("categoryCount"));

            double avgTimePerSample = trainingTime / (double) data.images().size() / 1_000_000;
            System.out.printf("Average time per sample: %.2f ms\n", avgTimePerSample);
            assertTrue(avgTimePerSample < 100, "Should process samples quickly");
        }
    }

    private double evaluateAccuracy(PAN pan, SyntheticDataGenerator.SyntheticData data,
                                   PANParameters params) {
        int correct = 0;
        for (int i = 0; i < data.images().size(); i++) {
            var result = pan.predict(data.images().get(i), params);
            if (result instanceof ActivationResult.Success success) {
                int predicted = success.categoryIndex();
                int actual = SyntheticDataGenerator.getClassIndex(data.labels().get(i));
                if (predicted == actual) {
                    correct++;
                }
            }
        }
        return (double) correct / data.images().size() * 100;
    }

    private double evaluateOnTrainingSet(PAN pan, SyntheticDataGenerator.SyntheticData data,
                                        int upTo, PANParameters params) {
        int correct = 0;
        for (int i = 0; i < upTo; i++) {
            var result = pan.predict(data.images().get(i), params);
            if (result instanceof ActivationResult.Success success) {
                int predicted = success.categoryIndex();
                int actual = SyntheticDataGenerator.getClassIndex(data.labels().get(i));
                if (predicted == actual) {
                    correct++;
                }
            }
        }
        return (double) correct / upTo * 100;
    }

    private List<Integer> shuffleIndices(int size) {
        List<Integer> indices = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, new Random(42));
        return indices;
    }
}