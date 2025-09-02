package com.hellblazer.art.core.salience;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("ClusterStatistics Tests")
class ClusterStatisticsTest {

    private static final double EPSILON = 1e-10;
    private ClusterStatistics stats;
    
    @BeforeEach
    void setUp() {
        stats = new ClusterStatistics(5);
    }

    @Test
    @DisplayName("Should initialize with correct dimension")
    void testInitialization() {
        assertEquals(5, stats.getDimension());
        assertEquals(0, stats.getSampleCount());
        
        // All statistics should be zero initially
        for (int i = 0; i < 5; i++) {
            assertEquals(0.0, stats.getFeatureMean(i), EPSILON);
            assertEquals(0.0, stats.getFeatureFrequency(i), EPSILON);
            assertEquals(0.0, stats.getFeatureVariance(i), EPSILON);
        }
    }

    @Test
    @DisplayName("Should update statistics with single sample")
    void testSingleSampleUpdate() {
        double[] sample = {1.0, 2.0, 3.0, 4.0, 5.0};
        stats.updateStatistics(sample);
        
        assertEquals(1, stats.getSampleCount());
        
        // Mean should equal the sample values
        for (int i = 0; i < 5; i++) {
            assertEquals(sample[i], stats.getFeatureMean(i), EPSILON);
            assertEquals(1.0, stats.getFeatureFrequency(i), EPSILON);
            // Variance should be 0 for single sample
            assertEquals(0.0, stats.getFeatureVariance(i), EPSILON);
        }
    }

    @Test
    @DisplayName("Should calculate incremental mean correctly")
    void testIncrementalMean() {
        double[][] samples = {
            {1.0, 2.0, 3.0, 4.0, 5.0},
            {2.0, 3.0, 4.0, 5.0, 6.0},
            {3.0, 4.0, 5.0, 6.0, 7.0}
        };
        
        for (double[] sample : samples) {
            stats.updateStatistics(sample);
        }
        
        assertEquals(3, stats.getSampleCount());
        
        // Check means
        double[] expectedMeans = {2.0, 3.0, 4.0, 5.0, 6.0};
        for (int i = 0; i < 5; i++) {
            assertEquals(expectedMeans[i], stats.getFeatureMean(i), EPSILON);
        }
    }

    @Test
    @DisplayName("Should calculate incremental variance correctly using Welford's algorithm")
    void testIncrementalVariance() {
        // Test with known variance
        double[][] samples = {
            {1.0, 0.0, 0.0, 0.0, 0.0},
            {2.0, 0.0, 0.0, 0.0, 0.0},
            {3.0, 0.0, 0.0, 0.0, 0.0},
            {4.0, 0.0, 0.0, 0.0, 0.0},
            {5.0, 0.0, 0.0, 0.0, 0.0}
        };
        
        for (double[] sample : samples) {
            stats.updateStatistics(sample);
        }
        
        // Mean of [1,2,3,4,5] = 3
        assertEquals(3.0, stats.getFeatureMean(0), EPSILON);
        
        // Sample variance = sum((x - mean)²) / n
        // = (4 + 1 + 0 + 1 + 4) / 5 = 2.0
        double expectedVariance = 2.0;
        assertEquals(expectedVariance, stats.getFeatureVariance(0), 0.01);
        
        // Other features should have zero variance
        for (int i = 1; i < 5; i++) {
            assertEquals(0.0, stats.getFeatureVariance(i), EPSILON);
        }
    }

    @Test
    @DisplayName("Should track feature frequency correctly")
    void testFeatureFrequency() {
        double[][] samples = {
            {1.0, 0.0, 0.5, 0.0, 0.8},
            {0.9, 0.0, 0.0, 0.0, 0.7},
            {1.1, 0.2, 0.6, 0.0, 0.9}
        };
        
        for (double[] sample : samples) {
            stats.updateStatistics(sample);
        }
        
        // Feature 0: appears in all 3 samples
        assertEquals(3.0, stats.getFeatureFrequency(0), EPSILON);
        // Feature 1: appears in 1 sample
        assertEquals(1.0, stats.getFeatureFrequency(1), EPSILON);
        // Feature 2: appears in 2 samples
        assertEquals(2.0, stats.getFeatureFrequency(2), EPSILON);
        // Feature 3: never appears
        assertEquals(0.0, stats.getFeatureFrequency(3), EPSILON);
        // Feature 4: appears in all 3 samples
        assertEquals(3.0, stats.getFeatureFrequency(4), EPSILON);
    }

    @Test
    @DisplayName("Should handle zero values correctly")
    void testZeroValueHandling() {
        double[] sample = {0.0, 0.0, 1.0, 0.0, 0.0};
        stats.updateStatistics(sample);
        
        assertEquals(1, stats.getSampleCount());
        
        // Only feature 2 should have frequency
        assertEquals(0.0, stats.getFeatureFrequency(0), EPSILON);
        assertEquals(0.0, stats.getFeatureFrequency(1), EPSILON);
        assertEquals(1.0, stats.getFeatureFrequency(2), EPSILON);
        assertEquals(0.0, stats.getFeatureFrequency(3), EPSILON);
        assertEquals(0.0, stats.getFeatureFrequency(4), EPSILON);
    }

    @Test
    @DisplayName("Should handle small values near zero")
    void testSmallValueHandling() {
        double[] sample = {1e-11, 1e-9, 1e-10, 1e-12, 0.1};
        stats.updateStatistics(sample);
        
        // Values below threshold (1e-10) should not count as non-zero
        assertEquals(0.0, stats.getFeatureFrequency(0), EPSILON); // 1e-11 < threshold
        assertEquals(0.0, stats.getFeatureFrequency(2), EPSILON); // 1e-10 = threshold
        assertEquals(0.0, stats.getFeatureFrequency(3), EPSILON); // 1e-12 < threshold
        
        // Values above threshold should count
        assertEquals(1.0, stats.getFeatureFrequency(1), EPSILON); // 1e-9 > threshold
        assertEquals(1.0, stats.getFeatureFrequency(4), EPSILON); // 0.1 > threshold
    }

    @Test
    @DisplayName("Should calculate standard deviation correctly")
    void testStandardDeviation() {
        double[][] samples = {
            {2.0, 4.0, 4.0, 4.0, 5.0},
            {5.0, 7.0, 8.0, 9.0, 10.0}
        };
        
        for (double[] sample : samples) {
            stats.updateStatistics(sample);
        }
        
        // For feature 0: mean = 3.5, values = [2, 5]
        // Variance = ((2-3.5)² + (5-3.5)²) / 2 = (2.25 + 2.25) / 2 = 2.25
        // StdDev = sqrt(2.25) = 1.5
        double stdDev0 = stats.getFeatureStandardDeviation(0);
        assertEquals(1.5, stdDev0, 0.01);
    }

    @Test
    @DisplayName("Should calculate coefficient of variation correctly")
    void testCoefficientOfVariation() {
        double[][] samples = {
            {10.0, 20.0, 30.0, 40.0, 50.0},
            {12.0, 18.0, 32.0, 38.0, 52.0}
        };
        
        for (double[] sample : samples) {
            stats.updateStatistics(sample);
        }
        
        // CV = stdDev / mean
        double cv0 = stats.getCoefficientOfVariation(0);
        assertTrue(cv0 > 0);
        
        // For constant feature (if we had one), CV should be 0
        ClusterStatistics constStats = new ClusterStatistics(1);
        constStats.updateStatistics(new double[]{5.0});
        constStats.updateStatistics(new double[]{5.0});
        assertEquals(0.0, constStats.getCoefficientOfVariation(0), EPSILON);
    }

    @Test
    @DisplayName("Should calculate frequency ratio correctly")
    void testFrequencyRatio() {
        double[][] samples = {
            {1.0, 0.0, 0.5, 0.0, 0.8},
            {0.9, 0.0, 0.0, 0.0, 0.7},
            {1.1, 0.2, 0.6, 0.0, 0.9}
        };
        
        for (double[] sample : samples) {
            stats.updateStatistics(sample);
        }
        
        // Feature 0: 3/3 = 1.0
        assertEquals(1.0, stats.getFrequencyRatio(0), EPSILON);
        // Feature 1: 1/3 = 0.333...
        assertEquals(1.0/3.0, stats.getFrequencyRatio(1), 0.01);
        // Feature 3: 0/3 = 0.0
        assertEquals(0.0, stats.getFrequencyRatio(3), EPSILON);
    }

    @Test
    @DisplayName("Should calculate information content correctly")
    void testInformationContent() {
        double[][] samples = {
            {1.0, 0.0, 0.5, 0.0, 0.8},
            {0.9, 0.0, 0.5, 0.0, 0.7},
            {1.1, 0.0, 0.5, 0.0, 0.9}
        };
        
        for (double[] sample : samples) {
            stats.updateStatistics(sample);
        }
        
        // Feature 2 has constant value (no variance), high frequency
        // Should have high information content
        double ic2 = stats.getInformationContent(2);
        
        // Feature 3 has zero frequency
        // Should have zero information content
        double ic3 = stats.getInformationContent(3);
        
        assertTrue(ic2 > ic3);
        assertEquals(0.0, ic3, EPSILON);
    }

    @Test
    @DisplayName("Should handle batch updates correctly")
    void testBatchUpdate() {
        double[][] batch = {
            {1.0, 2.0, 3.0, 4.0, 5.0},
            {2.0, 3.0, 4.0, 5.0, 6.0},
            {3.0, 4.0, 5.0, 6.0, 7.0}
        };
        
        stats.updateBatch(batch);
        
        assertEquals(3, stats.getSampleCount());
        
        // Verify means
        double[] expectedMeans = {2.0, 3.0, 4.0, 5.0, 6.0};
        for (int i = 0; i < 5; i++) {
            assertEquals(expectedMeans[i], stats.getFeatureMean(i), EPSILON);
        }
    }

    @Test
    @DisplayName("Should reset statistics correctly")
    void testReset() {
        // Add some data
        stats.updateStatistics(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});
        stats.updateStatistics(new double[]{2.0, 3.0, 4.0, 5.0, 6.0});
        
        assertEquals(2, stats.getSampleCount());
        
        // Reset
        stats.reset();
        
        assertEquals(0, stats.getSampleCount());
        for (int i = 0; i < 5; i++) {
            assertEquals(0.0, stats.getFeatureMean(i), EPSILON);
            assertEquals(0.0, stats.getFeatureFrequency(i), EPSILON);
            assertEquals(0.0, stats.getFeatureVariance(i), EPSILON);
        }
    }

    @Test
    @DisplayName("Should copy statistics correctly")
    void testCopy() {
        stats.updateStatistics(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});
        stats.updateStatistics(new double[]{2.0, 3.0, 4.0, 5.0, 6.0});
        
        ClusterStatistics copy = stats.copy();
        
        assertEquals(stats.getSampleCount(), copy.getSampleCount());
        assertEquals(stats.getDimension(), copy.getDimension());
        
        for (int i = 0; i < 5; i++) {
            assertEquals(stats.getFeatureMean(i), copy.getFeatureMean(i), EPSILON);
            assertEquals(stats.getFeatureFrequency(i), copy.getFeatureFrequency(i), EPSILON);
            assertEquals(stats.getFeatureVariance(i), copy.getFeatureVariance(i), EPSILON);
        }
        
        // Modifying copy should not affect original
        copy.updateStatistics(new double[]{10.0, 10.0, 10.0, 10.0, 10.0});
        assertNotEquals(stats.getSampleCount(), copy.getSampleCount());
    }

    @Test
    @DisplayName("Should merge statistics correctly")
    void testMerge() {
        ClusterStatistics stats1 = new ClusterStatistics(3);
        stats1.updateStatistics(new double[]{1.0, 2.0, 3.0});
        stats1.updateStatistics(new double[]{2.0, 3.0, 4.0});
        
        ClusterStatistics stats2 = new ClusterStatistics(3);
        stats2.updateStatistics(new double[]{3.0, 4.0, 5.0});
        stats2.updateStatistics(new double[]{4.0, 5.0, 6.0});
        
        ClusterStatistics merged = ClusterStatistics.merge(stats1, stats2);
        
        assertEquals(4, merged.getSampleCount());
        
        // Mean should be average of all samples
        assertEquals(2.5, merged.getFeatureMean(0), EPSILON); // (1+2+3+4)/4
        assertEquals(3.5, merged.getFeatureMean(1), EPSILON); // (2+3+4+5)/4
        assertEquals(4.5, merged.getFeatureMean(2), EPSILON); // (3+4+5+6)/4
    }

    @Test
    @DisplayName("Should handle large dataset accurately")
    void testLargeDataset() {
        Random random = new Random(42);
        int sampleCount = 10000;
        
        // Generate samples from normal distribution with known parameters
        double trueMean = 5.0;
        double trueStdDev = 2.0;
        
        stats = new ClusterStatistics(1); // Create once for this test
        for (int i = 0; i < sampleCount; i++) {
            double[] sample = new double[1];
            sample[0] = random.nextGaussian() * trueStdDev + trueMean;
            stats.updateStatistics(sample);
        }
        
        // With large sample, statistics should converge to true values
        // Allow some tolerance due to randomness
        assertEquals(trueMean, stats.getFeatureMean(0), 0.5); // Increased tolerance for random data
        assertEquals(trueStdDev, stats.getFeatureStandardDeviation(0), 0.5);
    }

    @Test
    @DisplayName("Should maintain numerical stability with extreme values")
    void testNumericalStability() {
        double[][] extremeSamples = {
            {1e10, 1e-10, 0.0, 1e5, 1e-5},
            {1e10 + 1, 1e-10, 0.0, 1e5, 1e-5},
            {1e10 - 1, 1e-10, 0.0, 1e5, 1e-5}
        };
        
        for (double[] sample : extremeSamples) {
            stats.updateStatistics(sample);
        }
        
        // Should handle extreme values without overflow/underflow
        assertTrue(Double.isFinite(stats.getFeatureMean(0)));
        assertTrue(Double.isFinite(stats.getFeatureVariance(0)));
        assertTrue(Double.isFinite(stats.getFeatureMean(1)));
        assertTrue(Double.isFinite(stats.getFeatureVariance(1)));
    }

    @Test
    @DisplayName("Should be thread-safe for concurrent updates")
    void testThreadSafety() throws InterruptedException {
        int threadCount = 10;
        int samplesPerThread = 100;
        CountDownLatch latch = new CountDownLatch(threadCount);
        ExecutorService executor = Executors.newFixedThreadPool(threadCount);
        
        ClusterStatistics threadSafeStats = new ClusterStatistics(3);
        AtomicInteger totalSamples = new AtomicInteger(0);
        
        for (int i = 0; i < threadCount; i++) {
            executor.submit(() -> {
                try {
                    for (int j = 0; j < samplesPerThread; j++) {
                        double[] sample = {1.0, 2.0, 3.0};
                        threadSafeStats.updateStatistics(sample);
                        totalSamples.incrementAndGet();
                    }
                } finally {
                    latch.countDown();
                }
            });
        }
        
        latch.await();
        executor.shutdown();
        
        // All updates should be accounted for
        assertEquals(threadCount * samplesPerThread, threadSafeStats.getSampleCount());
        
        // With identical samples, mean should be exact
        assertEquals(1.0, threadSafeStats.getFeatureMean(0), EPSILON);
        assertEquals(2.0, threadSafeStats.getFeatureMean(1), EPSILON);
        assertEquals(3.0, threadSafeStats.getFeatureMean(2), EPSILON);
        
        // Variance should be zero for identical samples
        assertEquals(0.0, threadSafeStats.getFeatureVariance(0), EPSILON);
        assertEquals(0.0, threadSafeStats.getFeatureVariance(1), EPSILON);
        assertEquals(0.0, threadSafeStats.getFeatureVariance(2), EPSILON);
    }

    @Test
    @DisplayName("Should serialize and deserialize correctly")
    void testSerialization() {
        stats.updateStatistics(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});
        stats.updateStatistics(new double[]{2.0, 3.0, 4.0, 5.0, 6.0});
        
        // Serialize to string representation
        String serialized = stats.serialize();
        assertNotNull(serialized);
        
        // Deserialize
        ClusterStatistics deserialized = ClusterStatistics.deserialize(serialized);
        
        assertEquals(stats.getSampleCount(), deserialized.getSampleCount());
        assertEquals(stats.getDimension(), deserialized.getDimension());
        
        for (int i = 0; i < 5; i++) {
            assertEquals(stats.getFeatureMean(i), deserialized.getFeatureMean(i), EPSILON);
            assertEquals(stats.getFeatureFrequency(i), deserialized.getFeatureFrequency(i), EPSILON);
            assertEquals(stats.getFeatureVariance(i), deserialized.getFeatureVariance(i), EPSILON);
        }
    }

    @Test
    @DisplayName("Should handle dimension mismatch gracefully")
    void testDimensionMismatch() {
        assertThrows(IllegalArgumentException.class, () -> {
            stats.updateStatistics(new double[]{1.0, 2.0, 3.0}); // Wrong dimension
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            stats.updateStatistics(new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}); // Too many
        });
    }

    @Test
    @DisplayName("Should provide meaningful toString output")
    void testToString() {
        stats.updateStatistics(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});
        
        String str = stats.toString();
        assertNotNull(str);
        assertTrue(str.contains("ClusterStatistics"));
        assertTrue(str.contains("dimension=5"));
        assertTrue(str.contains("sampleCount=1"));
    }
}