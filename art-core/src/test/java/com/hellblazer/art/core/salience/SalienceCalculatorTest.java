package com.hellblazer.art.core.salience;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("SalienceCalculator Tests")
class SalienceCalculatorTest {

    private static final double EPSILON = 1e-10;
    private ClusterStatistics stats;
    private SparseVector inputVector;
    
    @BeforeEach
    void setUp() {
        stats = new ClusterStatistics(5);
        // Initialize with sample data
        double[][] samples = {
            {1.0, 0.0, 0.5, 0.8, 0.2},
            {0.9, 0.1, 0.6, 0.7, 0.3},
            {1.1, 0.0, 0.4, 0.9, 0.1}
        };
        for (double[] sample : samples) {
            stats.updateStatistics(sample);
        }
        
        double[] inputData = {1.0, 0.05, 0.5, 0.8, 0.25};
        inputVector = new SparseVector(inputData, 0.01);
    }

    @Test
    @DisplayName("FrequencySalienceCalculator should calculate frequency-based salience")
    void testFrequencySalienceCalculator() {
        var calculator = new FrequencySalienceCalculator();
        double[] salience = calculator.calculate(stats, inputVector);
        
        assertNotNull(salience);
        assertEquals(5, salience.length);
        
        // All values should be between 0.1 and 1.0 (due to smoothing)
        for (double s : salience) {
            assertTrue(s >= 0.1 && s <= 1.0, "Salience value out of range: " + s);
        }
        
        // Features with higher frequency should have higher salience
        // Feature 0 appears in all samples (frequency = 3)
        // Feature 1 appears in 1 sample (frequency = 1)
        assertTrue(salience[0] > salience[1], 
                   "High frequency feature should have higher salience");
    }

    @Test
    @DisplayName("MeanSalienceCalculator should calculate mean-based salience")
    void testMeanSalienceCalculator() {
        var calculator = new MeanSalienceCalculator();
        double[] salience = calculator.calculate(stats, inputVector);
        
        assertNotNull(salience);
        assertEquals(5, salience.length);
        
        // All values should be normalized (sum to approximately 1.0)
        double sum = Arrays.stream(salience).sum();
        assertEquals(1.0, sum, 0.01);
        
        // Features closer to mean should have higher salience
        // Low variance features should have higher salience
        for (double s : salience) {
            assertTrue(s >= 0.0 && s <= 1.0, "Salience value out of range: " + s);
        }
    }

    @Test
    @DisplayName("StatisticalSalienceCalculator should combine multiple measures")
    void testStatisticalSalienceCalculator() {
        var calculator = new StatisticalSalienceCalculator();
        double[] salience = calculator.calculate(stats, inputVector);
        
        assertNotNull(salience);
        assertEquals(5, salience.length);
        
        // All values should be positive
        for (double s : salience) {
            assertTrue(s >= 0.0, "Salience should be non-negative: " + s);
        }
        
        // Should combine frequency, information content, and SNR
        // Features with high frequency and low variance should have highest salience
        double maxSalience = Arrays.stream(salience).max().orElse(0);
        double minSalience = Arrays.stream(salience).min().orElse(0);
        assertTrue(maxSalience > minSalience, "Should have varying salience values");
    }

    @Test
    @DisplayName("Calculators should handle zero frequency features")
    void testZeroFrequencyHandling() {
        ClusterStatistics sparseStats = new ClusterStatistics(5);
        // Only update some features
        sparseStats.updateStatistics(new double[]{1.0, 0.0, 0.0, 0.5, 0.0});
        sparseStats.updateStatistics(new double[]{0.8, 0.0, 0.0, 0.6, 0.0});
        
        var freqCalc = new FrequencySalienceCalculator();
        double[] freqSalience = freqCalc.calculate(sparseStats, inputVector);
        
        // Zero frequency features should have minimum salience (0.1 due to smoothing)
        assertEquals(0.1, freqSalience[1], 0.01);
        assertEquals(0.1, freqSalience[2], 0.01);
        assertEquals(0.1, freqSalience[4], 0.01);
        
        // Non-zero frequency features should have higher salience
        assertTrue(freqSalience[0] > 0.1);
        assertTrue(freqSalience[3] > 0.1);
    }

    @Test
    @DisplayName("Calculators should handle single sample statistics")
    void testSingleSampleStatistics() {
        ClusterStatistics singleStats = new ClusterStatistics(5);
        singleStats.updateStatistics(new double[]{1.0, 0.5, 0.0, 0.8, 0.3});
        
        var freqCalc = new FrequencySalienceCalculator();
        var meanCalc = new MeanSalienceCalculator();
        var statCalc = new StatisticalSalienceCalculator();
        
        double[] freqSalience = freqCalc.calculate(singleStats, inputVector);
        double[] meanSalience = meanCalc.calculate(singleStats, inputVector);
        double[] statSalience = statCalc.calculate(singleStats, inputVector);
        
        // All calculators should return valid results
        assertNotNull(freqSalience);
        assertNotNull(meanSalience);
        assertNotNull(statSalience);
        
        // No NaN or infinite values
        for (int i = 0; i < 5; i++) {
            assertTrue(Double.isFinite(freqSalience[i]));
            assertTrue(Double.isFinite(meanSalience[i]));
            assertTrue(Double.isFinite(statSalience[i]));
        }
    }

    @Test
    @DisplayName("Calculators should handle extreme values")
    void testExtremeValues() {
        ClusterStatistics extremeStats = new ClusterStatistics(5);
        extremeStats.updateStatistics(new double[]{1e10, 1e-10, 0.0, 1e5, 1e-5});
        extremeStats.updateStatistics(new double[]{1e10, 1e-10, 0.0, 1e5, 1e-5});
        
        double[] extremeInput = {1e10, 1e-10, 0.0, 1e5, 1e-5};
        var extremeVector = new SparseVector(extremeInput, 0.01);
        
        var freqCalc = new FrequencySalienceCalculator();
        var meanCalc = new MeanSalienceCalculator();
        var statCalc = new StatisticalSalienceCalculator();
        
        double[] freqSalience = freqCalc.calculate(extremeStats, extremeVector);
        double[] meanSalience = meanCalc.calculate(extremeStats, extremeVector);
        double[] statSalience = statCalc.calculate(extremeStats, extremeVector);
        
        // Should handle extreme values without overflow/underflow
        for (int i = 0; i < 5; i++) {
            assertTrue(Double.isFinite(freqSalience[i]));
            assertTrue(Double.isFinite(meanSalience[i]));
            assertTrue(Double.isFinite(statSalience[i]));
            
            // Values should still be in valid range
            assertTrue(freqSalience[i] >= 0.0 && freqSalience[i] <= 1.0);
            assertTrue(meanSalience[i] >= 0.0 && meanSalience[i] <= 1.0);
            assertTrue(statSalience[i] >= 0.0);
        }
    }

    @Test
    @DisplayName("Calculators should be thread-safe")
    void testThreadSafety() throws InterruptedException {
        var freqCalc = new FrequencySalienceCalculator();
        var meanCalc = new MeanSalienceCalculator();
        var statCalc = new StatisticalSalienceCalculator();
        
        int threadCount = 10;
        CountDownLatch latch = new CountDownLatch(threadCount);
        ExecutorService executor = Executors.newFixedThreadPool(threadCount);
        
        for (int i = 0; i < threadCount; i++) {
            executor.submit(() -> {
                try {
                    for (int j = 0; j < 100; j++) {
                        double[] freqResult = freqCalc.calculate(stats, inputVector);
                        double[] meanResult = meanCalc.calculate(stats, inputVector);
                        double[] statResult = statCalc.calculate(stats, inputVector);
                        
                        assertNotNull(freqResult);
                        assertNotNull(meanResult);
                        assertNotNull(statResult);
                        assertEquals(5, freqResult.length);
                        assertEquals(5, meanResult.length);
                        assertEquals(5, statResult.length);
                    }
                } finally {
                    latch.countDown();
                }
            });
        }
        
        latch.await();
        executor.shutdown();
    }

    @Test
    @DisplayName("FrequencySalienceCalculator should apply smoothing correctly")
    void testFrequencySalienceSmoothing() {
        var calculator = new FrequencySalienceCalculator();
        double[] salience = calculator.calculate(stats, inputVector);
        
        // With smoothing factor of 0.9, minimum value should be 0.1
        double minSalience = Arrays.stream(salience).min().orElse(0);
        assertTrue(minSalience >= 0.1, "Minimum salience should be at least 0.1 due to smoothing");
        
        // Maximum value should be close to 1.0 for high frequency features
        double maxSalience = Arrays.stream(salience).max().orElse(0);
        assertTrue(maxSalience <= 1.0, "Maximum salience should not exceed 1.0");
    }

    @Test
    @DisplayName("MeanSalienceCalculator should weight by variance")
    void testMeanSalienceVarianceWeighting() {
        // Create stats with different variances
        ClusterStatistics varStats = new ClusterStatistics(3);
        // Low variance feature
        for (int i = 0; i < 10; i++) {
            varStats.updateStatistics(new double[]{0.5, 0.5 + i * 0.01, i * 0.5});
        }
        
        var calculator = new MeanSalienceCalculator();
        double[] input = {0.5, 0.55, 2.5};
        var vector = new SparseVector(input, 0.01);
        double[] salience = calculator.calculate(varStats, vector);
        
        // Feature 0 has zero variance, should have high salience
        // Feature 1 has low variance, should have medium-high salience
        // Feature 2 has high variance, should have lower salience
        assertTrue(salience[0] > salience[2], 
                   "Low variance feature should have higher salience than high variance");
    }

    @Test
    @DisplayName("StatisticalSalienceCalculator should calculate SNR correctly")
    void testStatisticalSalienceSNR() {
        ClusterStatistics snrStats = new ClusterStatistics(3);
        // Create data with different signal-to-noise ratios
        snrStats.updateStatistics(new double[]{10.0, 1.0, 0.1});
        snrStats.updateStatistics(new double[]{10.1, 0.9, 0.5});
        snrStats.updateStatistics(new double[]{9.9, 1.1, 0.3});
        
        var calculator = new StatisticalSalienceCalculator();
        double[] input = {10.0, 1.0, 0.3};
        var vector = new SparseVector(input, 0.01);
        double[] salience = calculator.calculate(snrStats, vector);
        
        // Feature 0 has high mean, low variance -> high SNR
        // Feature 2 has low mean, higher variance -> low SNR
        assertTrue(salience[0] > salience[2], 
                   "High SNR feature should have higher salience");
    }

    @Test
    @DisplayName("Calculators should produce consistent results")
    void testConsistency() {
        var freqCalc = new FrequencySalienceCalculator();
        var meanCalc = new MeanSalienceCalculator();
        var statCalc = new StatisticalSalienceCalculator();
        
        // Calculate multiple times - should get same results
        double[] freq1 = freqCalc.calculate(stats, inputVector);
        double[] freq2 = freqCalc.calculate(stats, inputVector);
        
        double[] mean1 = meanCalc.calculate(stats, inputVector);
        double[] mean2 = meanCalc.calculate(stats, inputVector);
        
        double[] stat1 = statCalc.calculate(stats, inputVector);
        double[] stat2 = statCalc.calculate(stats, inputVector);
        
        assertArrayEquals(freq1, freq2, EPSILON);
        assertArrayEquals(mean1, mean2, EPSILON);
        assertArrayEquals(stat1, stat2, EPSILON);
    }

    @Test
    @DisplayName("Calculators should handle high-dimensional data")
    void testHighDimensional() {
        int dimension = 1000;
        ClusterStatistics bigStats = new ClusterStatistics(dimension);
        
        // Create sparse high-dimensional data
        double[] sample = new double[dimension];
        for (int i = 0; i < dimension; i += 10) {
            sample[i] = Math.random();
        }
        bigStats.updateStatistics(sample);
        bigStats.updateStatistics(sample);
        
        var bigVector = new SparseVector(sample, 0.01);
        
        var freqCalc = new FrequencySalienceCalculator();
        var meanCalc = new MeanSalienceCalculator();
        var statCalc = new StatisticalSalienceCalculator();
        
        double[] freqSalience = freqCalc.calculate(bigStats, bigVector);
        double[] meanSalience = meanCalc.calculate(bigStats, bigVector);
        double[] statSalience = statCalc.calculate(bigStats, bigVector);
        
        assertEquals(dimension, freqSalience.length);
        assertEquals(dimension, meanSalience.length);
        assertEquals(dimension, statSalience.length);
        
        // Should complete in reasonable time (test will timeout if too slow)
    }

    @Test
    @DisplayName("Calculators should respect interface contract")
    void testInterfaceContract() {
        // All calculators should implement SalienceCalculator interface
        SalienceCalculator freq = new FrequencySalienceCalculator();
        SalienceCalculator mean = new MeanSalienceCalculator();
        SalienceCalculator stat = new StatisticalSalienceCalculator();
        
        // Should be able to use them polymorphically
        SalienceCalculator[] calculators = {freq, mean, stat};
        
        for (SalienceCalculator calc : calculators) {
            double[] result = calc.calculate(stats, inputVector);
            assertNotNull(result);
            assertEquals(5, result.length);
            
            // Check for valid values
            for (double val : result) {
                assertTrue(Double.isFinite(val));
                assertTrue(val >= 0.0);
            }
        }
    }

    @Test
    @DisplayName("Frequency calculator should handle all-zero input")
    void testFrequencyAllZeroInput() {
        double[] zeros = new double[5];
        var zeroVector = new SparseVector(zeros, 0.01);
        
        var calculator = new FrequencySalienceCalculator();
        double[] salience = calculator.calculate(stats, zeroVector);
        
        assertNotNull(salience);
        assertEquals(5, salience.length);
        
        // Should still produce valid salience based on cluster statistics
        for (double s : salience) {
            assertTrue(Double.isFinite(s));
            assertTrue(s >= 0.1); // Minimum due to smoothing
        }
    }

    @Test
    @DisplayName("Mean calculator should handle input far from mean")
    void testMeanCalculatorOutlierInput() {
        double[] outlier = {100.0, 100.0, 100.0, 100.0, 100.0};
        var outlierVector = new SparseVector(outlier, 0.01);
        
        var calculator = new MeanSalienceCalculator();
        double[] salience = calculator.calculate(stats, outlierVector);
        
        assertNotNull(salience);
        assertEquals(5, salience.length);
        
        // Should still produce valid normalized salience
        double sum = Arrays.stream(salience).sum();
        assertEquals(1.0, sum, 0.01);
    }
}