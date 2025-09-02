package com.hellblazer.art.core.test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Generates various types of test data for ART algorithm testing
 */
public class TestDataGenerator {
    
    private final Random random;
    
    public TestDataGenerator(long seed) {
        this.random = new Random(seed);
    }
    
    public TestDataGenerator() {
        this(System.currentTimeMillis());
    }
    
    /**
     * Generate binary patterns for ART1 and binary algorithms
     */
    public double[][] generateBinaryPatterns(int samples, int features, double sparsity) {
        var data = new double[samples][features];
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < features; j++) {
                data[i][j] = random.nextDouble() > sparsity ? 1.0 : 0.0;
            }
        }
        return data;
    }
    
    /**
     * Generate orthogonal binary patterns
     */
    public double[][] generateOrthogonalBinaryPatterns(int count, int features) {
        if (count > features) {
            throw new IllegalArgumentException("Cannot generate more orthogonal patterns than features");
        }
        
        var data = new double[count][features];
        for (int i = 0; i < count; i++) {
            data[i][i] = 1.0;
        }
        return data;
    }
    
    /**
     * Generate Gaussian blob clusters similar to sklearn's make_blobs
     */
    public BlobData generateBlobs(int samples, int centers, int features, double clusterStd) {
        var samplesPerCenter = samples / centers;
        var data = new double[samples][features];
        var labels = new int[samples];
        
        // Generate random centers
        var centerPoints = new double[centers][features];
        for (int c = 0; c < centers; c++) {
            for (int f = 0; f < features; f++) {
                centerPoints[c][f] = random.nextGaussian() * 10; // Spread centers out
            }
        }
        
        // Generate samples around each center
        int sampleIdx = 0;
        for (int c = 0; c < centers; c++) {
            int samplesToGenerate = (c == centers - 1) ? 
                samples - sampleIdx : samplesPerCenter;
                
            for (int s = 0; s < samplesToGenerate; s++) {
                for (int f = 0; f < features; f++) {
                    data[sampleIdx][f] = centerPoints[c][f] + random.nextGaussian() * clusterStd;
                }
                labels[sampleIdx] = c;
                sampleIdx++;
            }
        }
        
        // Shuffle the data
        shuffleData(data, labels);
        
        return new BlobData(data, labels, centerPoints);
    }
    
    /**
     * Generate concentric circles (2D only)
     */
    public double[][] generateConcentricCircles(int samplesPerCircle, int numCircles, double noise) {
        if (numCircles < 1) {
            throw new IllegalArgumentException("Must have at least 1 circle");
        }
        
        var totalSamples = samplesPerCircle * numCircles;
        var data = new double[totalSamples][2];
        
        int idx = 0;
        for (int circle = 0; circle < numCircles; circle++) {
            double radius = (circle + 1) * 2.0;
            
            for (int s = 0; s < samplesPerCircle; s++) {
                double angle = 2 * Math.PI * s / samplesPerCircle;
                data[idx][0] = radius * Math.cos(angle) + random.nextGaussian() * noise;
                data[idx][1] = radius * Math.sin(angle) + random.nextGaussian() * noise;
                idx++;
            }
        }
        
        return data;
    }
    
    /**
     * Generate spiral pattern (2D only)
     */
    public double[][] generateSpiral(int samples, double noise) {
        var data = new double[samples][2];
        
        for (int i = 0; i < samples; i++) {
            double t = (double) i / samples * 4 * Math.PI;
            double radius = t;
            data[i][0] = radius * Math.cos(t) + random.nextGaussian() * noise;
            data[i][1] = radius * Math.sin(t) + random.nextGaussian() * noise;
        }
        
        return data;
    }
    
    /**
     * Generate time series data with seasonal patterns
     */
    public double[][] generateTimeSeriesData(int length, int features, double trendStrength) {
        var data = new double[length][features];
        
        for (int f = 0; f < features; f++) {
            double frequency = 0.1 + random.nextDouble() * 0.3; // Random frequency
            double phase = random.nextDouble() * 2 * Math.PI; // Random phase
            double amplitude = 1 + random.nextDouble() * 2; // Random amplitude
            
            for (int t = 0; t < length; t++) {
                // Trend component
                double trend = trendStrength * t / length;
                
                // Seasonal component
                double seasonal = amplitude * Math.sin(2 * Math.PI * frequency * t + phase);
                
                // Noise component
                double noise = random.nextGaussian() * 0.1;
                
                data[t][f] = trend + seasonal + noise;
            }
        }
        
        return data;
    }
    
    /**
     * Generate XOR pattern for testing non-linear separability
     */
    public XORData generateXORPattern(int samplesPerQuadrant) {
        var totalSamples = samplesPerQuadrant * 4;
        var data = new double[totalSamples][2];
        var labels = new int[totalSamples];
        
        int idx = 0;
        
        // Quadrant 1: (0, 0) - Class 0
        for (int i = 0; i < samplesPerQuadrant; i++) {
            data[idx][0] = random.nextDouble() * 0.4;
            data[idx][1] = random.nextDouble() * 0.4;
            labels[idx] = 0;
            idx++;
        }
        
        // Quadrant 2: (0, 1) - Class 1
        for (int i = 0; i < samplesPerQuadrant; i++) {
            data[idx][0] = random.nextDouble() * 0.4;
            data[idx][1] = 0.6 + random.nextDouble() * 0.4;
            labels[idx] = 1;
            idx++;
        }
        
        // Quadrant 3: (1, 0) - Class 1
        for (int i = 0; i < samplesPerQuadrant; i++) {
            data[idx][0] = 0.6 + random.nextDouble() * 0.4;
            data[idx][1] = random.nextDouble() * 0.4;
            labels[idx] = 1;
            idx++;
        }
        
        // Quadrant 4: (1, 1) - Class 0
        for (int i = 0; i < samplesPerQuadrant; i++) {
            data[idx][0] = 0.6 + random.nextDouble() * 0.4;
            data[idx][1] = 0.6 + random.nextDouble() * 0.4;
            labels[idx] = 0;
            idx++;
        }
        
        shuffleData(data, labels);
        return new XORData(data, labels);
    }
    
    /**
     * Generate edge case data for stress testing
     */
    public double[][] generateEdgeCaseData(EdgeCaseType type, int samples, int features) {
        return switch (type) {
            case ALL_ZEROS -> new double[samples][features];
            case ALL_ONES -> {
                var data = new double[samples][features];
                for (int i = 0; i < samples; i++) {
                    for (int j = 0; j < features; j++) {
                        data[i][j] = 1.0;
                    }
                }
                yield data;
            }
            case SINGLE_FEATURE -> {
                var data = new double[samples][features];
                int activeFeature = random.nextInt(features);
                for (int i = 0; i < samples; i++) {
                    data[i][activeFeature] = random.nextDouble();
                }
                yield data;
            }
            case ALTERNATING -> {
                var data = new double[samples][features];
                for (int i = 0; i < samples; i++) {
                    for (int j = 0; j < features; j++) {
                        data[i][j] = (i + j) % 2;
                    }
                }
                yield data;
            }
            case EXTREME_VALUES -> {
                var data = new double[samples][features];
                for (int i = 0; i < samples; i++) {
                    for (int j = 0; j < features; j++) {
                        data[i][j] = random.nextBoolean() ? 1e-10 : 1e10;
                    }
                }
                yield data;
            }
            case IDENTICAL_SAMPLES -> {
                var data = new double[samples][features];
                var template = new double[features];
                for (int j = 0; j < features; j++) {
                    template[j] = random.nextDouble();
                }
                for (int i = 0; i < samples; i++) {
                    System.arraycopy(template, 0, data[i], 0, features);
                }
                yield data;
            }
            case LINEARLY_DEPENDENT -> {
                var data = new double[samples][features];
                // First column is random
                for (int i = 0; i < samples; i++) {
                    data[i][0] = random.nextDouble();
                }
                // Other columns are linear combinations
                for (int j = 1; j < features; j++) {
                    double factor = random.nextDouble() * 2 - 1;
                    for (int i = 0; i < samples; i++) {
                        data[i][j] = data[i][0] * factor;
                    }
                }
                yield data;
            }
            case SPARSE -> {
                var data = new double[samples][features];
                int nonZeroPerSample = Math.max(1, features / 10);
                for (int i = 0; i < samples; i++) {
                    for (int k = 0; k < nonZeroPerSample; k++) {
                        int j = random.nextInt(features);
                        data[i][j] = random.nextDouble();
                    }
                }
                yield data;
            }
        };
    }
    
    /**
     * Generate noisy moons pattern (2D only)
     */
    public double[][] generateMoons(int samples, double noise) {
        var data = new double[samples][2];
        int halfSamples = samples / 2;
        
        // First moon
        for (int i = 0; i < halfSamples; i++) {
            double angle = Math.PI * i / halfSamples;
            data[i][0] = Math.cos(angle) + random.nextGaussian() * noise;
            data[i][1] = Math.sin(angle) + random.nextGaussian() * noise;
        }
        
        // Second moon (shifted and flipped)
        for (int i = halfSamples; i < samples; i++) {
            double angle = Math.PI * (i - halfSamples) / (samples - halfSamples);
            data[i][0] = 1 - Math.cos(angle) + random.nextGaussian() * noise;
            data[i][1] = 1 - Math.sin(angle) - 0.5 + random.nextGaussian() * noise;
        }
        
        return data;
    }
    
    /**
     * Generate iris-like dataset (simplified version)
     */
    public IrisData generateIrisLikeData() {
        int samplesPerClass = 50;
        int features = 4;
        int classes = 3;
        
        var data = new double[samplesPerClass * classes][features];
        var labels = new int[samplesPerClass * classes];
        var labelNames = new String[]{"Setosa", "Versicolor", "Virginica"};
        
        // Define mean and std for each class and feature (simplified iris characteristics)
        double[][][] params = {
            // Setosa: small flowers
            {{5.0, 0.35}, {3.4, 0.38}, {1.5, 0.17}, {0.2, 0.10}},
            // Versicolor: medium flowers  
            {{5.9, 0.52}, {2.8, 0.31}, {4.3, 0.47}, {1.3, 0.20}},
            // Virginica: large flowers
            {{6.6, 0.64}, {3.0, 0.32}, {5.6, 0.55}, {2.0, 0.27}}
        };
        
        int idx = 0;
        for (int c = 0; c < classes; c++) {
            for (int s = 0; s < samplesPerClass; s++) {
                for (int f = 0; f < features; f++) {
                    double mean = params[c][f][0];
                    double std = params[c][f][1];
                    data[idx][f] = mean + random.nextGaussian() * std;
                    // Ensure non-negative values
                    data[idx][f] = Math.max(0, data[idx][f]);
                }
                labels[idx] = c;
                idx++;
            }
        }
        
        shuffleData(data, labels);
        return new IrisData(data, labels, labelNames);
    }
    
    /**
     * Add noise to existing data
     */
    public double[][] addNoise(double[][] data, double noiseLevel) {
        var noisyData = new double[data.length][data[0].length];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                noisyData[i][j] = data[i][j] + random.nextGaussian() * noiseLevel;
            }
        }
        return noisyData;
    }
    
    /**
     * Add missing values to data
     */
    public double[][] addMissingValues(double[][] data, double missingRate) {
        var dataWithMissing = new double[data.length][data[0].length];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                if (random.nextDouble() < missingRate) {
                    dataWithMissing[i][j] = Double.NaN;
                } else {
                    dataWithMissing[i][j] = data[i][j];
                }
            }
        }
        return dataWithMissing;
    }
    
    private void shuffleData(double[][] data, int[] labels) {
        for (int i = data.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            
            // Swap data
            var tempData = data[i];
            data[i] = data[j];
            data[j] = tempData;
            
            // Swap labels
            int tempLabel = labels[i];
            labels[i] = labels[j];
            labels[j] = tempLabel;
        }
    }
    
    // Data structures for returning complex datasets
    public record BlobData(double[][] data, int[] labels, double[][] centers) {}
    public record XORData(double[][] data, int[] labels) {}
    public record IrisData(double[][] data, int[] labels, String[] labelNames) {}
    
    public enum EdgeCaseType {
        ALL_ZEROS,
        ALL_ONES,
        SINGLE_FEATURE,
        ALTERNATING,
        EXTREME_VALUES,
        IDENTICAL_SAMPLES,
        LINEARLY_DEPENDENT,
        SPARSE
    }
}