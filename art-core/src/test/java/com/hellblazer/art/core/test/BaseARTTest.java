package com.hellblazer.art.core.test;

import com.hellblazer.art.core.BaseART;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.results.ActivationResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.TestInfo;

import java.util.*;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Base class for all ART algorithm tests providing common test utilities
 */
public abstract class BaseARTTest {
    
    protected static final double EPSILON = 1e-10;
    protected static final double LOOSE_EPSILON = 1e-6;
    protected Random random;
    protected long seed;
    
    @BeforeEach
    void setupBase(TestInfo testInfo) {
        seed = testInfo.getDisplayName().hashCode();
        random = new Random(seed);
    }
    
    /**
     * Helper method to train an ART algorithm on data
     */
    protected List<Integer> trainAndPredict(BaseART alg, double[][] data, Object parameters) {
        var labels = new ArrayList<Integer>();
        for (var sample : data) {
            var pattern = Pattern.of(sample);
            var result = alg.stepFit(pattern, parameters);
            if (result instanceof ActivationResult.Success success) {
                labels.add(success.categoryIndex());
            } else if (result instanceof ActivationResult.NoMatch) {
                labels.add(alg.getCategoryCount() - 1); // Last created category
            } else {
                labels.add(-1);
            }
        }
        return labels;
    }
    
    /**
     * Helper method to predict labels for data
     */
    protected List<Integer> predict(BaseART alg, double[][] data, Object parameters) {
        var labels = new ArrayList<Integer>();
        for (var sample : data) {
            var pattern = Pattern.of(sample);
            var result = alg.stepPredict(pattern, parameters);
            if (result instanceof ActivationResult.Success success) {
                labels.add(success.categoryIndex());
            } else if (result instanceof ActivationResult.NoMatch) {
                labels.add(-1);
            } else {
                labels.add(-1);
            }
        }
        return labels;
    }
    
    /**
     * Assert that clustering results are consistent
     */
    protected void assertClustering(BaseART alg, double[][] data, Object parameters, int[] expectedLabels) {
        assertNotNull(alg, "Algorithm cannot be null");
        assertNotNull(data, "Data cannot be null");
        assertNotNull(expectedLabels, "Expected labels cannot be null");
        assertEquals(data.length, expectedLabels.length, "Data and labels length mismatch");
        
        var labels = trainAndPredict(alg, data, parameters);
        assertNotNull(labels, "Predicted labels cannot be null");
        assertEquals(expectedLabels.length, labels.size(), "Predicted labels length mismatch");
        
        // Convert to array for NMI calculation
        var labelArray = labels.stream().mapToInt(Integer::intValue).toArray();
        
        // Check clustering quality using normalized mutual information
        double nmi = calculateNMI(expectedLabels, labelArray);
        assertTrue(nmi > 0.7, "Clustering quality too low: NMI = " + nmi);
    }
    
    /**
     * Assert that algorithm converges within reasonable iterations
     */
    protected void assertConvergence(BaseART alg, double[][] data, Object parameters, int maxIterations) {
        var previousCategories = 0;
        var stableCount = 0;
        var requiredStableIterations = 3;
        
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            // Train on all data
            for (var sample : data) {
                var pattern = Pattern.of(sample);
                alg.stepFit(pattern, parameters);
            }
            
            var currentCategories = alg.getCategoryCount();
            
            if (currentCategories == previousCategories) {
                stableCount++;
                if (stableCount >= requiredStableIterations) {
                    return; // Converged
                }
            } else {
                stableCount = 0;
            }
            previousCategories = currentCategories;
        }
        
        fail("Algorithm did not converge within " + maxIterations + " iterations");
    }
    
    /**
     * Assert that results are reproducible with same seed
     */
    protected void assertReproducible(BaseART alg1, BaseART alg2, double[][] data, Object parameters) {
        var labels1 = trainAndPredict(alg1, data, parameters);
        var labels2 = trainAndPredict(alg2, data, parameters);
        
        assertEquals(labels1.size(), labels2.size(), "Label counts differ");
        for (int i = 0; i < labels1.size(); i++) {
            assertEquals(labels1.get(i), labels2.get(i), 
                        "Results not reproducible at index " + i);
        }
        assertEquals(alg1.getCategoryCount(), alg2.getCategoryCount(), 
                    "Category counts not reproducible");
    }
    
    /**
     * Assert arrays are equal within epsilon tolerance
     */
    protected void assertArrayEquals(double[] expected, double[] actual, double epsilon) {
        assertNotNull(expected, "Expected array cannot be null");
        assertNotNull(actual, "Actual array cannot be null");
        assertEquals(expected.length, actual.length, "Array lengths differ");
        
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], epsilon,
                        "Arrays differ at index " + i);
        }
    }
    
    /**
     * Assert 2D arrays are equal within epsilon tolerance
     */
    protected void assertMatrixEquals(double[][] expected, double[][] actual, double epsilon) {
        assertNotNull(expected, "Expected matrix cannot be null");
        assertNotNull(actual, "Actual matrix cannot be null");
        assertEquals(expected.length, actual.length, "Matrix row counts differ");
        
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(expected[i], actual[i], epsilon);
        }
    }
    
    /**
     * Calculate Normalized Mutual Information between two labelings
     */
    protected double calculateNMI(int[] labels1, int[] labels2) {
        if (labels1.length != labels2.length) {
            throw new IllegalArgumentException("Label arrays must have same length");
        }
        
        int n = labels1.length;
        
        // Calculate contingency table
        Map<Integer, Map<Integer, Integer>> contingency = new HashMap<>();
        Map<Integer, Integer> count1 = new HashMap<>();
        Map<Integer, Integer> count2 = new HashMap<>();
        
        for (int i = 0; i < n; i++) {
            count1.merge(labels1[i], 1, Integer::sum);
            count2.merge(labels2[i], 1, Integer::sum);
            contingency.computeIfAbsent(labels1[i], k -> new HashMap<>())
                      .merge(labels2[i], 1, Integer::sum);
        }
        
        // Calculate mutual information
        double mi = 0.0;
        for (var entry1 : contingency.entrySet()) {
            for (var entry2 : entry1.getValue().entrySet()) {
                int nij = entry2.getValue();
                if (nij > 0) {
                    double pij = (double) nij / n;
                    double pi = (double) count1.get(entry1.getKey()) / n;
                    double pj = (double) count2.get(entry2.getKey()) / n;
                    mi += pij * Math.log(pij / (pi * pj));
                }
            }
        }
        
        // Calculate entropies
        double h1 = calculateEntropy(count1.values(), n);
        double h2 = calculateEntropy(count2.values(), n);
        
        // Return normalized mutual information
        if (h1 == 0 || h2 == 0) {
            return 0.0;
        }
        return 2.0 * mi / (h1 + h2);
    }
    
    private double calculateEntropy(Collection<Integer> counts, int total) {
        double entropy = 0.0;
        for (int count : counts) {
            if (count > 0) {
                double p = (double) count / total;
                entropy -= p * Math.log(p);
            }
        }
        return entropy;
    }
    
    /**
     * Calculate Adjusted Rand Index between two labelings
     */
    protected double calculateARI(int[] labels1, int[] labels2) {
        if (labels1.length != labels2.length) {
            throw new IllegalArgumentException("Label arrays must have same length");
        }
        
        int n = labels1.length;
        
        // Build contingency table
        Map<Integer, Map<Integer, Integer>> contingency = new HashMap<>();
        Map<Integer, Integer> sum1 = new HashMap<>();
        Map<Integer, Integer> sum2 = new HashMap<>();
        
        for (int i = 0; i < n; i++) {
            sum1.merge(labels1[i], 1, Integer::sum);
            sum2.merge(labels2[i], 1, Integer::sum);
            contingency.computeIfAbsent(labels1[i], k -> new HashMap<>())
                      .merge(labels2[i], 1, Integer::sum);
        }
        
        // Calculate index
        double index = 0;
        for (var entry : contingency.entrySet()) {
            for (var count : entry.getValue().values()) {
                index += nChoose2(count);
            }
        }
        
        double sum1Choose2 = sum1.values().stream().mapToDouble(this::nChoose2).sum();
        double sum2Choose2 = sum2.values().stream().mapToDouble(this::nChoose2).sum();
        double expectedIndex = sum1Choose2 * sum2Choose2 / nChoose2(n);
        double maxIndex = 0.5 * (sum1Choose2 + sum2Choose2);
        
        if (maxIndex == expectedIndex) {
            return 0.0;
        }
        return (index - expectedIndex) / (maxIndex - expectedIndex);
    }
    
    private double nChoose2(int n) {
        return n * (n - 1) / 2.0;
    }
    
    /**
     * Assert that parameters are within valid ranges
     */
    protected void assertParametersValid(Map<String, Object> params) {
        assertNotNull(params, "Parameters cannot be null");
        assertFalse(params.isEmpty(), "Parameters cannot be empty");
        
        // Check common ART parameters
        if (params.containsKey("rho")) {
            var rho = ((Number) params.get("rho")).doubleValue();
            assertTrue(rho >= 0 && rho <= 1, "Vigilance rho must be in [0, 1]");
        }
        
        if (params.containsKey("alpha")) {
            var alpha = ((Number) params.get("alpha")).doubleValue();
            assertTrue(alpha >= 0, "Learning rate alpha must be non-negative");
        }
        
        if (params.containsKey("beta")) {
            var beta = ((Number) params.get("beta")).doubleValue();
            assertTrue(beta > 0, "Beta must be positive");
        }
    }
    
    /**
     * Generate random data within range
     */
    protected double[][] generateRandomData(int samples, int features, double min, double max) {
        var data = new double[samples][features];
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < features; j++) {
                data[i][j] = min + (max - min) * random.nextDouble();
            }
        }
        return data;
    }
    
    /**
     * Generate binary data
     */
    protected double[][] generateBinaryData(int samples, int features, double sparsity) {
        var data = new double[samples][features];
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < features; j++) {
                data[i][j] = random.nextDouble() > sparsity ? 1.0 : 0.0;
            }
        }
        return data;
    }
    
    /**
     * Shuffle array using Fisher-Yates algorithm
     */
    protected void shuffle(double[][] array) {
        for (int i = array.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            var temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
    
    /**
     * Split data into train and test sets
     */
    protected DataSplit splitData(double[][] data, double trainRatio) {
        int trainSize = (int) (data.length * trainRatio);
        var indices = IntStream.range(0, data.length).toArray();
        
        // Shuffle indices
        for (int i = indices.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        
        var train = new double[trainSize][];
        var test = new double[data.length - trainSize][];
        
        for (int i = 0; i < trainSize; i++) {
            train[i] = data[indices[i]];
        }
        for (int i = 0; i < test.length; i++) {
            test[i] = data[indices[trainSize + i]];
        }
        
        return new DataSplit(train, test);
    }
    
    public record DataSplit(double[][] train, double[][] test) {}
}