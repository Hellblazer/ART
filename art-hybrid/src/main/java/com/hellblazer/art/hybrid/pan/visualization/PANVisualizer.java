package com.hellblazer.art.hybrid.pan.visualization;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.hybrid.pan.weight.BPARTWeight;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.List;
import java.util.Map;

/**
 * Visualization utilities for PAN training and analysis.
 * Generates text-based visualizations for console output.
 */
public class PANVisualizer {

    /**
     * Generate ASCII art confusion matrix.
     */
    public static String generateConfusionMatrix(int[][] matrix, List<String> classLabels) {
        StringWriter sw = new StringWriter();
        PrintWriter pw = new PrintWriter(sw);

        int numClasses = matrix.length;
        int cellWidth = 6;

        // Header
        pw.println("\n=== Confusion Matrix ===");
        pw.print("      ");
        for (int i = 0; i < numClasses; i++) {
            String label = classLabels != null && i < classLabels.size() ?
                classLabels.get(i) : String.valueOf(i);
            pw.printf("%6s", label.length() > 5 ? label.substring(0, 5) : label);
        }
        pw.println("\n      " + "-".repeat(numClasses * cellWidth));

        // Rows
        for (int i = 0; i < numClasses; i++) {
            String label = classLabels != null && i < classLabels.size() ?
                classLabels.get(i) : String.valueOf(i);
            pw.printf("%-5s|", label.length() > 5 ? label.substring(0, 5) : label);

            for (int j = 0; j < numClasses; j++) {
                pw.printf("%6d", matrix[i][j]);
            }

            // Row accuracy
            int rowSum = 0;
            for (int j = 0; j < numClasses; j++) {
                rowSum += matrix[i][j];
            }
            double accuracy = rowSum > 0 ? (double) matrix[i][i] / rowSum * 100 : 0;
            pw.printf("  | %.1f%%", accuracy);
            pw.println();
        }

        // Overall accuracy
        int correct = 0, total = 0;
        for (int i = 0; i < numClasses; i++) {
            correct += matrix[i][i];
            for (int j = 0; j < numClasses; j++) {
                total += matrix[i][j];
            }
        }
        double overallAccuracy = total > 0 ? (double) correct / total * 100 : 0;
        pw.printf("\nOverall Accuracy: %.2f%% (%d/%d)\n", overallAccuracy, correct, total);

        return sw.toString();
    }

    /**
     * Generate training progress chart.
     */
    public static String generateProgressChart(List<Double> accuracies, int width, int height) {
        if (accuracies.isEmpty()) {
            return "No data to display";
        }

        StringWriter sw = new StringWriter();
        PrintWriter pw = new PrintWriter(sw);

        double maxAcc = accuracies.stream().mapToDouble(Double::doubleValue).max().orElse(100);
        double minAcc = accuracies.stream().mapToDouble(Double::doubleValue).min().orElse(0);
        double range = maxAcc - minAcc;
        if (range == 0) range = 1;

        pw.println("\n=== Training Progress ===");
        pw.println("Accuracy (%)");

        // Create chart
        char[][] chart = new char[height][width];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                chart[i][j] = ' ';
            }
        }

        // Plot points
        for (int i = 0; i < accuracies.size() && i < width; i++) {
            double acc = accuracies.get(i * accuracies.size() / width);
            int y = (int) ((acc - minAcc) / range * (height - 1));
            y = height - 1 - y; // Flip Y axis
            if (y >= 0 && y < height) {
                chart[y][i] = '*';
            }
        }

        // Add Y axis labels
        for (int i = 0; i < height; i++) {
            double value = maxAcc - (i * range / (height - 1));
            pw.printf("%6.1f |", value);
            for (int j = 0; j < width; j++) {
                pw.print(chart[i][j]);
            }
            pw.println();
        }

        // X axis
        pw.print("       +");
        for (int i = 0; i < width; i++) {
            pw.print("-");
        }
        pw.println();
        pw.print("        ");
        for (int i = 0; i < width; i += 10) {
            pw.printf("%-10d", i * accuracies.size() / width);
        }
        pw.println("\n        Epochs");

        // Statistics
        pw.println("\nStatistics:");
        pw.printf("  Max: %.2f%%\n", maxAcc);
        pw.printf("  Min: %.2f%%\n", minAcc);
        pw.printf("  Final: %.2f%%\n", accuracies.get(accuracies.size() - 1));

        return sw.toString();
    }

    /**
     * Generate category distribution visualization.
     */
    public static String generateCategoryDistribution(Map<Integer, Integer> categoryToLabel,
                                                     Map<Integer, Integer> categoryCounts) {
        StringWriter sw = new StringWriter();
        PrintWriter pw = new PrintWriter(sw);

        pw.println("\n=== Category Distribution ===");
        pw.println("Cat | Label | Count | Bar");
        pw.println("----|-------|-------|" + "-".repeat(30));

        int maxCount = categoryCounts.values().stream()
            .mapToInt(Integer::intValue)
            .max()
            .orElse(1);

        for (var entry : categoryToLabel.entrySet()) {
            int category = entry.getKey();
            int label = entry.getValue();
            int count = categoryCounts.getOrDefault(category, 0);
            int barLength = (count * 30) / maxCount;
            String bar = "#".repeat(barLength);

            pw.printf("%3d | %5d | %5d | %s\n", category, label, count, bar);
        }

        // Summary
        int totalCategories = categoryToLabel.size();
        int uniqueLabels = (int) categoryToLabel.values().stream().distinct().count();
        pw.printf("\nTotal categories: %d\n", totalCategories);
        pw.printf("Unique labels: %d\n", uniqueLabels);
        pw.printf("Categories per label: %.2f\n",
            (double) totalCategories / Math.max(1, uniqueLabels));

        return sw.toString();
    }

    /**
     * Generate weight statistics summary.
     */
    public static String generateWeightStatistics(List<BPARTWeight> weights) {
        StringWriter sw = new StringWriter();
        PrintWriter pw = new PrintWriter(sw);

        pw.println("\n=== Weight Statistics ===");

        if (weights.isEmpty()) {
            pw.println("No weights to analyze");
            return sw.toString();
        }

        // Calculate statistics
        double[] meanForward = new double[weights.get(0).forwardWeights().length];
        double[] meanBackward = new double[weights.get(0).backwardWeights().length];
        double minForward = Double.MAX_VALUE, maxForward = -Double.MAX_VALUE;
        double minBackward = Double.MAX_VALUE, maxBackward = -Double.MAX_VALUE;
        long totalUpdates = 0;

        for (var weight : weights) {
            // Forward weights
            for (int i = 0; i < weight.forwardWeights().length; i++) {
                double val = weight.forwardWeights()[i];
                meanForward[i] += val;
                minForward = Math.min(minForward, val);
                maxForward = Math.max(maxForward, val);
            }

            // Backward weights
            for (int i = 0; i < weight.backwardWeights().length; i++) {
                double val = weight.backwardWeights()[i];
                meanBackward[i] += val;
                minBackward = Math.min(minBackward, val);
                maxBackward = Math.max(maxBackward, val);
            }

            totalUpdates += weight.updateCount();
        }

        // Compute means
        for (int i = 0; i < meanForward.length; i++) {
            meanForward[i] /= weights.size();
        }
        for (int i = 0; i < meanBackward.length; i++) {
            meanBackward[i] /= weights.size();
        }

        // Display statistics
        pw.printf("Number of weight vectors: %d\n", weights.size());
        pw.println("\nForward weights:");
        pw.printf("  Dimensions: %d\n", meanForward.length);
        pw.printf("  Range: [%.4f, %.4f]\n", minForward, maxForward);
        pw.printf("  Mean magnitude: %.4f\n", computeMagnitude(meanForward));

        pw.println("\nBackward weights:");
        pw.printf("  Dimensions: %d\n", meanBackward.length);
        pw.printf("  Range: [%.4f, %.4f]\n", minBackward, maxBackward);
        pw.printf("  Mean magnitude: %.4f\n", computeMagnitude(meanBackward));

        pw.printf("\nAverage updates per weight: %.1f\n",
            (double) totalUpdates / weights.size());

        // Check for negative weights
        boolean hasNegative = minForward < 0 || minBackward < 0;
        pw.printf("Contains negative weights: %s\n", hasNegative ? "Yes" : "No");

        return sw.toString();
    }

    /**
     * Generate pattern visualization (for 28x28 images).
     */
    public static String visualizePattern(Pattern pattern, int width, int height) {
        StringWriter sw = new StringWriter();
        PrintWriter pw = new PrintWriter(sw);

        if (pattern.dimension() != width * height) {
            pw.printf("Pattern dimension %d doesn't match %dx%d\n",
                pattern.dimension(), width, height);
            return sw.toString();
        }

        pw.println("\n=== Pattern Visualization ===");

        // ASCII art representation
        String chars = " .:-=+*#%@";
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                double val = pattern.get(idx);
                int charIdx = (int) (val * (chars.length() - 1));
                charIdx = Math.max(0, Math.min(chars.length() - 1, charIdx));
                pw.print(chars.charAt(charIdx));
            }
            pw.println();
        }

        // Statistics
        double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
        double sum = 0;
        for (int i = 0; i < pattern.dimension(); i++) {
            double val = pattern.get(i);
            min = Math.min(min, val);
            max = Math.max(max, val);
            sum += val;
        }
        double mean = sum / pattern.dimension();

        pw.printf("\nStatistics:\n");
        pw.printf("  Range: [%.3f, %.3f]\n", min, max);
        pw.printf("  Mean: %.3f\n", mean);

        return sw.toString();
    }

    /**
     * Generate training summary report.
     */
    public static String generateTrainingSummary(
            int epochs,
            double finalAccuracy,
            double bestAccuracy,
            int finalCategories,
            long trainingTimeMs,
            Map<String, Object> additionalStats) {

        StringWriter sw = new StringWriter();
        PrintWriter pw = new PrintWriter(sw);

        pw.println("\n" + "=".repeat(50));
        pw.println("         PAN TRAINING SUMMARY REPORT");
        pw.println("=".repeat(50));

        pw.println("\nTraining Configuration:");
        pw.printf("  Epochs completed: %d\n", epochs);
        pw.printf("  Training time: %.2f seconds\n", trainingTimeMs / 1000.0);
        pw.printf("  Time per epoch: %.2f seconds\n", trainingTimeMs / 1000.0 / epochs);

        pw.println("\nAccuracy Metrics:");
        pw.printf("  Final accuracy: %.2f%%\n", finalAccuracy);
        pw.printf("  Best accuracy: %.2f%%\n", bestAccuracy);
        pw.printf("  Improvement: %+.2f%%\n", bestAccuracy - finalAccuracy);

        pw.println("\nModel Statistics:");
        pw.printf("  Categories created: %d\n", finalCategories);

        if (additionalStats != null && !additionalStats.isEmpty()) {
            pw.println("\nAdditional Metrics:");
            for (var entry : additionalStats.entrySet()) {
                pw.printf("  %s: %s\n", entry.getKey(), entry.getValue());
            }
        }

        pw.println("\n" + "=".repeat(50));

        return sw.toString();
    }

    // Helper methods

    private static double computeMagnitude(double[] vector) {
        double sum = 0;
        for (double val : vector) {
            sum += val * val;
        }
        return Math.sqrt(sum / vector.length);
    }
}