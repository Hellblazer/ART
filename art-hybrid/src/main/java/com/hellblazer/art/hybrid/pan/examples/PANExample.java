package com.hellblazer.art.hybrid.pan.examples;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.hybrid.pan.PAN;
import com.hellblazer.art.hybrid.pan.datasets.SyntheticDataGenerator;
import com.hellblazer.art.hybrid.pan.parameters.CNNConfig;
import com.hellblazer.art.hybrid.pan.parameters.PANParameters;
import com.hellblazer.art.hybrid.pan.serialization.PANSerializer;
import com.hellblazer.art.hybrid.pan.training.PANTrainer;
import com.hellblazer.art.hybrid.pan.visualization.PANVisualizer;
import com.hellblazer.art.hybrid.pan.weight.BPARTWeight;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * Complete example demonstrating PAN usage.
 *
 * This example shows:
 * - Data generation
 * - Model training with multiple epochs
 * - Evaluation and visualization
 * - Model serialization
 * - Hyperparameter tuning
 */
public class PANExample {

    public static void main(String[] args) {
        System.out.println("=== PAN (Pretrained Adaptive Resonance Network) Example ===\n");

        // Run different examples
        try {
            // 1. Basic training example
            System.out.println("1. BASIC TRAINING EXAMPLE");
            System.out.println("-".repeat(40));
            runBasicTrainingExample();

            // 2. Hyperparameter tuning example
            System.out.println("\n2. HYPERPARAMETER TUNING EXAMPLE");
            System.out.println("-".repeat(40));
            runHyperparameterTuningExample();

            // 3. Model serialization example
            System.out.println("\n3. MODEL SERIALIZATION EXAMPLE");
            System.out.println("-".repeat(40));
            runSerializationExample();

            // 4. Visualization example
            System.out.println("\n4. VISUALIZATION EXAMPLE");
            System.out.println("-".repeat(40));
            runVisualizationExample();

        } catch (Exception e) {
            System.err.println("Error in example: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Basic training example with synthetic MNIST-like data.
     */
    private static void runBasicTrainingExample() {
        System.out.println("Generating synthetic MNIST-like data...");
        var trainData = SyntheticDataGenerator.generateMNISTLike(500, 5);
        var testData = SyntheticDataGenerator.generateMNISTLike(100, 5);

        System.out.println("Creating PAN with default parameters...");
        var params = PANParameters.forMNIST();

        try (var pan = new PAN(params)) {
            System.out.println("Training PAN for 3 epochs...\n");

            List<Double> epochAccuracies = new ArrayList<>();

            for (int epoch = 1; epoch <= 3; epoch++) {
                System.out.printf("Epoch %d:\n", epoch);

                // Training
                int correct = 0;
                long startTime = System.currentTimeMillis();

                for (int i = 0; i < trainData.images().size(); i++) {
                    var result = pan.learnSupervised(
                        trainData.images().get(i),
                        trainData.labels().get(i),
                        params
                    );

                    // Check accuracy
                    int predicted = pan.predictLabel(trainData.images().get(i), params);
                    int actual = SyntheticDataGenerator.getClassIndex(trainData.labels().get(i));
                    if (predicted == actual) correct++;

                    // Progress
                    if ((i + 1) % 100 == 0) {
                        System.out.printf("  Processed %d/%d samples\r", i + 1, trainData.images().size());
                    }
                }

                long epochTime = System.currentTimeMillis() - startTime;

                // Evaluation
                double trainAcc = (double) correct / trainData.images().size() * 100;
                double testAcc = PANTrainer.evaluate(pan, testData.images(), testData.labels(), params);
                epochAccuracies.add(testAcc);

                System.out.printf("\n  Train accuracy: %.2f%%\n", trainAcc);
                System.out.printf("  Test accuracy: %.2f%%\n", testAcc);
                System.out.printf("  Categories: %d\n", pan.getCategoryCount());
                System.out.printf("  Epoch time: %.2f seconds\n", epochTime / 1000.0);
            }

            // Final statistics
            @SuppressWarnings("unchecked")
            var stats = (Map<String, Object>) pan.getPerformanceStats();
            System.out.println("\nFinal Model Statistics:");
            System.out.printf("  Total samples: %d\n", stats.get("totalSamples"));
            System.out.printf("  Accuracy: %.2f%%\n", (Double) stats.get("accuracy") * 100);
            System.out.printf("  Categories: %d\n", stats.get("categoryCount"));
            System.out.printf("  Memory usage: %.2f MB\n", (Long) stats.get("memoryUsageBytes") / 1024.0 / 1024.0);
        }
    }

    /**
     * Hyperparameter tuning example.
     */
    private static void runHyperparameterTuningExample() {
        System.out.println("Generating data for hyperparameter tuning...");
        var trainData = SyntheticDataGenerator.generateMNISTLike(300, 3);
        var valData = SyntheticDataGenerator.generateMNISTLike(100, 3);

        System.out.println("Testing different parameter configurations...\n");

        List<PANParameters> configs = Arrays.asList(
            // Conservative: High vigilance, low learning rate
            new PANParameters(0.8, 6, CNNConfig.simple(), false,
                0.01, 0.95, 0.0001, true, 32, 0.95, 0.8, 200, 16, 0.1, 0.1,
                false, 0.0, 1.0), // Disable normalization

            // Balanced: Medium settings
            new PANParameters(0.6, 9, CNNConfig.simple(), false,
                0.03, 0.9, 0.0001, true, 32, 0.95, 0.8, 300, 16, 0.2, 0.15,
                false, 0.0, 1.0), // Disable normalization

            // Aggressive: Low vigilance, high learning rate
            new PANParameters(0.4, 12, CNNConfig.simple(), false,
                0.05, 0.85, 0.0001, true, 32, 0.95, 0.8, 400, 16, 0.3, 0.2,
                false, 0.0, 1.0) // Disable normalization
        );

        var searchResult = PANTrainer.hyperparameterSearch(
            trainData.images(), trainData.labels(),
            valData.images(), valData.labels(),
            configs, 2, false
        );

        System.out.println("\nHyperparameter Search Results:");
        System.out.println("-".repeat(40));

        int configNum = 1;
        for (var entry : searchResult.allResults().entrySet()) {
            var params = entry.getKey();
            var accuracy = entry.getValue();
            System.out.printf("Config %d: vigilance=%.1f, lr=%.3f -> %.2f%% accuracy\n",
                configNum++, params.vigilance(), params.learningRate(), accuracy);
        }

        System.out.printf("\nBest configuration achieved: %.2f%% accuracy\n",
            searchResult.bestAccuracy());
    }

    /**
     * Model serialization example.
     */
    private static void runSerializationExample() throws IOException {
        System.out.println("Training a model to save...");

        var data = SyntheticDataGenerator.generateMNISTLike(200, 3);
        var params = PANParameters.defaultParameters();

        List<WeightVector> categories;
        Map<Integer, Integer> categoryLabels;
        long totalSamples;
        long correctPredictions;

        try (var pan = new PAN(params)) {
            // Train
            for (int i = 0; i < data.images().size(); i++) {
                pan.learnSupervised(data.images().get(i), data.labels().get(i), params);
            }

            // Get model state
            categories = pan.getCategories();
            categoryLabels = new HashMap<>(); // Would need getter in PAN
            @SuppressWarnings("unchecked")
            var stats = (Map<String, Object>) pan.getPerformanceStats();
            totalSamples = (Long) stats.get("totalSamples");
            correctPredictions = (Long) stats.get("correctPredictions");
        }

        // Create saved model (simplified - would need actual CNN weights)
        var savedModel = new PANSerializer.SavedPANModel(
            params, categories, categoryLabels,
            totalSamples, correctPredictions, 0.0, 1000L,
            System.currentTimeMillis(),
            new float[0], new float[0]  // Placeholder CNN weights
        );

        // Save to file
        Path modelPath = Paths.get("pan_model.bin");
        System.out.printf("Saving model to %s...\n", modelPath);
        PANSerializer.saveModel(modelPath, savedModel, true);

        // Load model back
        System.out.println("Loading model from file...");
        var loadedModel = PANSerializer.loadModel(modelPath);

        System.out.println("Loaded model summary:");
        System.out.println(loadedModel.getSummary());
        System.out.printf("  Categories: %d\n", loadedModel.categories().size());
        System.out.printf("  Accuracy: %.2f%%\n", loadedModel.getAccuracy());

        // Clean up
        modelPath.toFile().delete();
    }

    /**
     * Visualization example.
     */
    private static void runVisualizationExample() {
        System.out.println("Creating visualization examples...\n");

        // 1. Training progress chart
        List<Double> accuracies = Arrays.asList(
            10.0, 15.0, 22.0, 28.0, 32.0, 35.0, 33.0, 36.0, 38.0, 37.0
        );
        System.out.println(PANVisualizer.generateProgressChart(accuracies, 40, 10));

        // 2. Confusion matrix
        int[][] confMatrix = {
            {45, 3, 2},
            {5, 40, 5},
            {1, 4, 45}
        };
        List<String> labels = Arrays.asList("Cat0", "Cat1", "Cat2");
        System.out.println(PANVisualizer.generateConfusionMatrix(confMatrix, labels));

        // 3. Category distribution
        Map<Integer, Integer> categoryToLabel = new HashMap<>();
        Map<Integer, Integer> categoryCounts = new HashMap<>();
        for (int i = 0; i < 6; i++) {
            categoryToLabel.put(i, i / 2); // 2 categories per label
            categoryCounts.put(i, 20 + i * 5);
        }
        System.out.println(PANVisualizer.generateCategoryDistribution(categoryToLabel, categoryCounts));

        // 4. Sample pattern visualization
        var data = SyntheticDataGenerator.generateMNISTLike(1, 1);
        if (!data.images().isEmpty()) {
            System.out.println(PANVisualizer.visualizePattern(data.images().get(0), 28, 28));
        }

        // 5. Training summary
        Map<String, Object> additionalStats = new HashMap<>();
        additionalStats.put("Learning rate", 0.02);
        additionalStats.put("Vigilance", 0.6);
        additionalStats.put("Hidden units", 32);

        System.out.println(PANVisualizer.generateTrainingSummary(
            10, 37.0, 38.0, 15, 45000L, additionalStats
        ));
    }
}