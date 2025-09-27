package com.hellblazer.art.hybrid.pan.datasets;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Generates synthetic datasets for PAN testing.
 * Creates MNIST-like patterns without external dependencies.
 */
public class SyntheticDataGenerator {

    public record SyntheticData(
        List<Pattern> images,
        List<Pattern> labels,
        int numClasses
    ) {}

    /**
     * Generate MNIST-like synthetic data.
     * Creates patterns that simulate handwritten digits.
     */
    public static SyntheticData generateMNISTLike(int numSamples, int numClasses) {
        Random rand = new Random(42);
        List<Pattern> images = new ArrayList<>(numSamples);
        List<Pattern> labels = new ArrayList<>(numSamples);

        int samplesPerClass = numSamples / numClasses;

        for (int classIdx = 0; classIdx < numClasses; classIdx++) {
            for (int i = 0; i < samplesPerClass; i++) {
                // Create 28x28 pattern
                double[] pixels = generateDigitPattern(classIdx, rand);
                images.add(new DenseVector(pixels));

                // Create one-hot label
                double[] label = new double[numClasses];
                label[classIdx] = 1.0;
                labels.add(new DenseVector(label));
            }
        }

        return new SyntheticData(images, labels, numClasses);
    }

    /**
     * Generate pattern for a specific digit class.
     */
    private static double[] generateDigitPattern(int digit, Random rand) {
        double[] pattern = new double[784]; // 28x28

        // Create base pattern for each digit
        switch (digit) {
            case 0 -> generateZero(pattern, rand);
            case 1 -> generateOne(pattern, rand);
            case 2 -> generateTwo(pattern, rand);
            case 3 -> generateThree(pattern, rand);
            case 4 -> generateFour(pattern, rand);
            case 5 -> generateFive(pattern, rand);
            case 6 -> generateSix(pattern, rand);
            case 7 -> generateSeven(pattern, rand);
            case 8 -> generateEight(pattern, rand);
            case 9 -> generateNine(pattern, rand);
            default -> generateRandom(pattern, rand);
        }

        // Add noise
        addNoise(pattern, rand, 0.1);

        return pattern;
    }

    private static void generateZero(double[] pattern, Random rand) {
        // Draw oval shape
        drawOval(pattern, 14, 14, 8, 10, 0.9);
    }

    private static void generateOne(double[] pattern, Random rand) {
        // Draw vertical line
        drawLine(pattern, 14, 5, 14, 23, 0.9);
        drawLine(pattern, 13, 5, 13, 23, 0.7);
    }

    private static void generateTwo(double[] pattern, Random rand) {
        // Draw curved top and diagonal
        drawArc(pattern, 14, 10, 6, 0, Math.PI, 0.9);
        drawLine(pattern, 20, 10, 8, 23, 0.9);
        drawLine(pattern, 8, 23, 20, 23, 0.9);
    }

    private static void generateThree(double[] pattern, Random rand) {
        // Draw two curves
        drawArc(pattern, 14, 10, 6, -Math.PI/2, Math.PI/2, 0.9);
        drawArc(pattern, 14, 18, 6, -Math.PI/2, Math.PI/2, 0.9);
    }

    private static void generateFour(double[] pattern, Random rand) {
        // Draw angled lines
        drawLine(pattern, 8, 5, 8, 15, 0.9);
        drawLine(pattern, 8, 15, 20, 15, 0.9);
        drawLine(pattern, 18, 5, 18, 23, 0.9);
    }

    private static void generateFive(double[] pattern, Random rand) {
        // Draw S shape
        drawLine(pattern, 20, 5, 8, 5, 0.9);
        drawLine(pattern, 8, 5, 8, 14, 0.9);
        drawArc(pattern, 14, 17, 6, -Math.PI/2, Math.PI/2, 0.9);
    }

    private static void generateSix(double[] pattern, Random rand) {
        // Draw circle with curve
        drawOval(pattern, 14, 18, 6, 6, 0.9);
        drawArc(pattern, 14, 10, 8, Math.PI/2, Math.PI, 0.9);
    }

    private static void generateSeven(double[] pattern, Random rand) {
        // Draw horizontal and diagonal
        drawLine(pattern, 8, 5, 20, 5, 0.9);
        drawLine(pattern, 20, 5, 12, 23, 0.9);
    }

    private static void generateEight(double[] pattern, Random rand) {
        // Draw two circles
        drawOval(pattern, 14, 10, 5, 5, 0.9);
        drawOval(pattern, 14, 18, 6, 5, 0.9);
    }

    private static void generateNine(double[] pattern, Random rand) {
        // Draw circle with tail
        drawOval(pattern, 14, 10, 6, 6, 0.9);
        drawLine(pattern, 20, 10, 20, 23, 0.9);
    }

    private static void generateRandom(double[] pattern, Random rand) {
        for (int i = 0; i < pattern.length; i++) {
            pattern[i] = rand.nextDouble() * 0.3;
        }
    }

    private static void drawLine(double[] pattern, int x1, int y1, int x2, int y2, double intensity) {
        int steps = Math.max(Math.abs(x2 - x1), Math.abs(y2 - y1));
        for (int i = 0; i <= steps; i++) {
            int x = x1 + (x2 - x1) * i / steps;
            int y = y1 + (y2 - y1) * i / steps;
            setPixel(pattern, x, y, intensity);
        }
    }

    private static void drawOval(double[] pattern, int cx, int cy, int rx, int ry, double intensity) {
        for (int angle = 0; angle < 360; angle += 5) {
            double rad = Math.toRadians(angle);
            int x = (int)(cx + rx * Math.cos(rad));
            int y = (int)(cy + ry * Math.sin(rad));
            setPixel(pattern, x, y, intensity);

            // Fill slightly
            if (rx > 2 && ry > 2) {
                setPixel(pattern, x - 1, y, intensity * 0.7);
                setPixel(pattern, x + 1, y, intensity * 0.7);
                setPixel(pattern, x, y - 1, intensity * 0.7);
                setPixel(pattern, x, y + 1, intensity * 0.7);
            }
        }
    }

    private static void drawArc(double[] pattern, int cx, int cy, int r,
                               double startAngle, double endAngle, double intensity) {
        for (double angle = startAngle; angle <= endAngle; angle += 0.1) {
            int x = (int)(cx + r * Math.cos(angle));
            int y = (int)(cy + r * Math.sin(angle));
            setPixel(pattern, x, y, intensity);
        }
    }

    private static void setPixel(double[] pattern, int x, int y, double value) {
        if (x >= 0 && x < 28 && y >= 0 && y < 28) {
            int idx = y * 28 + x;
            pattern[idx] = Math.max(pattern[idx], value);
        }
    }

    private static void addNoise(double[] pattern, Random rand, double noiseLevel) {
        for (int i = 0; i < pattern.length; i++) {
            double noise = (rand.nextDouble() - 0.5) * noiseLevel;
            pattern[i] = Math.max(0, Math.min(1, pattern[i] + noise));
        }
    }

    /**
     * Get class index from one-hot label.
     */
    public static int getClassIndex(Pattern label) {
        for (int i = 0; i < label.dimension(); i++) {
            if (label.get(i) > 0.5) {
                return i;
            }
        }
        return -1;
    }
}