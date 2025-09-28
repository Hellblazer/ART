package com.hellblazer.art.hybrid.pan.training;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.hybrid.pan.PAN;
import com.hellblazer.art.hybrid.pan.parameters.PANParameters;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Training utilities for PAN with multi-epoch support.
 */
public class PANTrainer {

    /**
     * Training result with metrics.
     */
    public record TrainingResult(
        int epochsCompleted,
        double finalAccuracy,
        double bestAccuracy,
        int bestEpoch,
        long trainingTimeMs,
        List<Double> epochAccuracies,
        int finalCategories
    ) {}

    /**
     * Train PAN with multiple epochs and early stopping.
     */
    public static TrainingResult trainWithEpochs(
            PAN pan,
            List<Pattern> trainInputs,
            List<Pattern> trainTargets,
            List<Pattern> valInputs,
            List<Pattern> valTargets,
            PANParameters parameters,
            int maxEpochs,
            double earlyStoppingThreshold,
            boolean verbose) {

        if (trainInputs.size() != trainTargets.size()) {
            throw new IllegalArgumentException("Training inputs and targets must have same size");
        }

        long startTime = System.currentTimeMillis();
        List<Double> epochAccuracies = new ArrayList<>();
        double bestAccuracy = 0.0;
        int bestEpoch = 0;
        int epochsWithoutImprovement = 0;

        for (int epoch = 1; epoch <= maxEpochs; epoch++) {
            if (verbose) {
                System.out.printf("\nEpoch %d/%d\n", epoch, maxEpochs);
            }

            // Shuffle training data
            List<Integer> indices = createShuffledIndices(trainInputs.size());

            // Training phase
            int correct = 0;
            int batchSize = Math.min(100, trainInputs.size() / 10);

            for (int i = 0; i < indices.size(); i++) {
                int idx = indices.get(i);

                var result = pan.learnSupervised(
                    trainInputs.get(idx),
                    trainTargets.get(idx),
                    parameters
                );

                if (result instanceof ActivationResult.Success success) {
                    // Check if prediction matches target
                    int predictedLabel = pan.predictLabel(trainInputs.get(idx), parameters);
                    int actualLabel = extractLabel(trainTargets.get(idx));
                    if (predictedLabel == actualLabel) {
                        correct++;
                    }
                }

                // Progress reporting
                if (verbose && (i + 1) % batchSize == 0) {
                    double trainAccuracy = (double) correct / (i + 1) * 100;
                    System.out.printf("  Batch %d/%d - Training accuracy: %.2f%%\r",
                        (i + 1) / batchSize, (indices.size() + batchSize - 1) / batchSize, trainAccuracy);
                }
            }

            // Validation phase
            double valAccuracy = evaluate(pan, valInputs, valTargets, parameters);
            epochAccuracies.add(valAccuracy);

            if (verbose) {
                double trainAccuracy = (double) correct / trainInputs.size() * 100;
                System.out.printf("\nEpoch %d - Train: %.2f%%, Val: %.2f%%, Categories: %d\n",
                    epoch, trainAccuracy, valAccuracy, pan.getCategoryCount());
            }

            // Early stopping check
            if (valAccuracy > bestAccuracy) {
                bestAccuracy = valAccuracy;
                bestEpoch = epoch;
                epochsWithoutImprovement = 0;
            } else {
                epochsWithoutImprovement++;
                if (epochsWithoutImprovement >= 3) {
                    if (verbose) {
                        System.out.println("Early stopping triggered");
                    }
                    break;
                }
            }

            // Check if we've reached the target
            if (valAccuracy >= earlyStoppingThreshold) {
                if (verbose) {
                    System.out.printf("Target accuracy %.2f%% reached\n", earlyStoppingThreshold);
                }
                break;
            }
        }

        long trainingTime = System.currentTimeMillis() - startTime;

        return new TrainingResult(
            epochAccuracies.size(),
            epochAccuracies.isEmpty() ? 0.0 : epochAccuracies.get(epochAccuracies.size() - 1),
            bestAccuracy,
            bestEpoch,
            trainingTime,
            epochAccuracies,
            pan.getCategoryCount()
        );
    }

    /**
     * Evaluate PAN on a dataset without learning.
     */
    public static double evaluate(PAN pan, List<Pattern> inputs, List<Pattern> targets,
                                 PANParameters parameters) {
        if (inputs.size() != targets.size()) {
            throw new IllegalArgumentException("Inputs and targets must have same size");
        }

        int correct = 0;
        for (int i = 0; i < inputs.size(); i++) {
            int predicted = pan.predictLabel(inputs.get(i), parameters);
            int actual = extractLabel(targets.get(i));
            if (predicted == actual) {
                correct++;
            }
        }

        return (double) correct / inputs.size() * 100.0;
    }

    /**
     * Perform k-fold cross-validation.
     */
    public static CrossValidationResult crossValidate(
            List<Pattern> inputs,
            List<Pattern> targets,
            PANParameters parameters,
            int kFolds,
            int maxEpochs,
            boolean verbose) {

        if (inputs.size() != targets.size()) {
            throw new IllegalArgumentException("Inputs and targets must have same size");
        }

        List<Double> foldAccuracies = new ArrayList<>();
        List<Integer> foldCategories = new ArrayList<>();

        int foldSize = inputs.size() / kFolds;

        for (int fold = 0; fold < kFolds; fold++) {
            if (verbose) {
                System.out.printf("\n=== Fold %d/%d ===\n", fold + 1, kFolds);
            }

            // Split data
            int testStart = fold * foldSize;
            int testEnd = (fold == kFolds - 1) ? inputs.size() : (fold + 1) * foldSize;

            List<Pattern> trainInputs = new ArrayList<>();
            List<Pattern> trainTargets = new ArrayList<>();
            List<Pattern> testInputs = new ArrayList<>();
            List<Pattern> testTargets = new ArrayList<>();

            for (int i = 0; i < inputs.size(); i++) {
                if (i >= testStart && i < testEnd) {
                    testInputs.add(inputs.get(i));
                    testTargets.add(targets.get(i));
                } else {
                    trainInputs.add(inputs.get(i));
                    trainTargets.add(targets.get(i));
                }
            }

            // Train on this fold
            try (PAN pan = new PAN(parameters)) {
                var result = trainWithEpochs(
                    pan, trainInputs, trainTargets,
                    testInputs, testTargets,
                    parameters, maxEpochs, 95.0, verbose
                );

                foldAccuracies.add(result.bestAccuracy());
                foldCategories.add(result.finalCategories());
            }
        }

        // Calculate statistics
        double meanAccuracy = foldAccuracies.stream()
            .mapToDouble(Double::doubleValue)
            .average()
            .orElse(0.0);

        double stdAccuracy = calculateStandardDeviation(foldAccuracies, meanAccuracy);

        double meanCategories = foldCategories.stream()
            .mapToInt(Integer::intValue)
            .average()
            .orElse(0.0);

        return new CrossValidationResult(
            kFolds, meanAccuracy, stdAccuracy,
            meanCategories, foldAccuracies
        );
    }

    /**
     * Cross-validation results.
     */
    public record CrossValidationResult(
        int kFolds,
        double meanAccuracy,
        double stdAccuracy,
        double meanCategories,
        List<Double> foldAccuracies
    ) {
        @Override
        public String toString() {
            return String.format("CV Result: %.2f%% ± %.2f%% (%.1f categories)",
                meanAccuracy, stdAccuracy, meanCategories);
        }
    }

    /**
     * Train with different hyperparameter configurations.
     */
    public static HyperparameterSearchResult hyperparameterSearch(
            List<Pattern> trainInputs,
            List<Pattern> trainTargets,
            List<Pattern> valInputs,
            List<Pattern> valTargets,
            List<PANParameters> parameterConfigs,
            int maxEpochs,
            boolean verbose) {

        PANParameters bestParams = null;
        double bestAccuracy = 0.0;
        Map<PANParameters, Double> results = new HashMap<>();

        for (int i = 0; i < parameterConfigs.size(); i++) {
            var params = parameterConfigs.get(i);

            if (verbose) {
                System.out.printf("\n=== Testing configuration %d/%d ===\n",
                    i + 1, parameterConfigs.size());
                System.out.printf("Vigilance: %.2f, LR: %.4f, Hidden: %d\n",
                    params.vigilance(), params.learningRate(), params.hiddenUnits());
            }

            try (PAN pan = new PAN(params)) {
                var result = trainWithEpochs(
                    pan, trainInputs, trainTargets,
                    valInputs, valTargets,
                    params, maxEpochs, 95.0, verbose
                );

                results.put(params, result.bestAccuracy());

                if (result.bestAccuracy() > bestAccuracy) {
                    bestAccuracy = result.bestAccuracy();
                    bestParams = params;
                }
            }
        }

        return new HyperparameterSearchResult(bestParams, bestAccuracy, results);
    }

    /**
     * Hyperparameter search results.
     */
    public record HyperparameterSearchResult(
        PANParameters bestParameters,
        double bestAccuracy,
        Map<PANParameters, Double> allResults
    ) {}

    // Helper methods

    private static List<Integer> createShuffledIndices(int size) {
        List<Integer> indices = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, ThreadLocalRandom.current());
        return indices;
    }

    private static int extractLabel(Pattern target) {
        // Extract from one-hot encoding
        for (int i = 0; i < target.dimension(); i++) {
            if (target.get(i) > 0.5) {
                return i;
            }
        }
        return -1;
    }

    private static double calculateStandardDeviation(List<Double> values, double mean) {
        double sumSquaredDiff = 0.0;
        for (double value : values) {
            double diff = value - mean;
            sumSquaredDiff += diff * diff;
        }
        return Math.sqrt(sumSquaredDiff / values.size());
    }
}