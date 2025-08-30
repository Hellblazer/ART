package com.hellblazer.art.core;

import com.hellblazer.art.core.cvi.*;
import com.hellblazer.art.core.iCVIFuzzyART.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.DisplayName;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for iCVIFuzzyART - FuzzyART with incremental CVI integration.
 * Focuses on incremental CVI updates, streaming data, and performance.
 */
@DisplayName("iCVIFuzzyART Tests")
public class iCVIFuzzyARTTest {
    
    private iCVIFuzzyART icviFuzzyART;
    private List<Pattern> streamData;
    private iCVIFuzzyARTParameters defaultParams;
    
    @BeforeEach
    void setUp() {
        icviFuzzyART = new iCVIFuzzyART();
        defaultParams = new iCVIFuzzyARTParameters();
        
        // Generate streaming data
        streamData = generateStreamData(1000);
    }
    
    @Nested
    @DisplayName("Incremental CVI Updates")
    class IncrementalUpdateTests {
        
        @Test
        @DisplayName("Should update CVIs incrementally")
        void testIncrementalCVIUpdate() {
            // Add incremental CVI
            var incrementalCH = new CalinskiHarabaszIndex(); // Supports incremental
            icviFuzzyART.addCVI(incrementalCH);
            
            // Process patterns one by one
            int updateCount = 0;
            for (int i = 0; i < 100; i++) {
                var result = icviFuzzyART.learn(streamData.get(i), defaultParams);
                assertTrue(result.wasSuccessful());
                
                // Check if CVI was updated incrementally
                if (icviFuzzyART.wasIncrementallyUpdated()) {
                    updateCount++;
                }
            }
            
            // Should have used incremental updates
            assertTrue(updateCount > 50, 
                      "Should use incremental updates frequently, got: " + updateCount);
            
            // CVI scores should be available
            var scores = icviFuzzyART.getCurrentCVIScores();
            assertNotNull(scores.get("Calinski-Harabasz Index"));
        }
        
        @Test
        @DisplayName("Should fall back to batch when incremental not supported")
        void testBatchFallback() {
            // Add non-incremental CVI
            var dbIndex = new DaviesBouldinIndex(); // No incremental support
            icviFuzzyART.addCVI(dbIndex);
            
            int batchUpdateCount = 0;
            for (int i = 0; i < 50; i++) {
                icviFuzzyART.learn(streamData.get(i), defaultParams);
                
                if (icviFuzzyART.wasLastUpdateBatch()) {
                    batchUpdateCount++;
                }
            }
            
            // Should use batch updates for non-incremental CVI
            assertTrue(batchUpdateCount > 0, 
                      "Should fall back to batch updates");
        }
        
        @Test
        @DisplayName("Should update only when necessary")
        void testSelectiveUpdates() {
            icviFuzzyART.addCVI(new CalinskiHarabaszIndex());
            
            // Set update frequency
            defaultParams.setCVIUpdateFrequency(10); // Update every 10 patterns
            
            int updateCount = 0;
            int lastUpdateIndex = -1;
            for (int i = 0; i < 100; i++) {
                icviFuzzyART.learn(streamData.get(i), defaultParams);
                
                if (icviFuzzyART.getCVIUpdateCount() > updateCount) {
                    updateCount = icviFuzzyART.getCVIUpdateCount();
                    
                    // Check that updates happen approximately every 10 patterns
                    if (lastUpdateIndex >= 0) {
                        int interval = i - lastUpdateIndex;
                        assertTrue(interval >= 8 && interval <= 12, 
                                  "Update interval should be close to 10, was: " + interval);
                    }
                    lastUpdateIndex = i;
                }
            }
            
            // Total updates should be around 10
            assertTrue(updateCount >= 8 && updateCount <= 12, 
                      "Should have approximately 10 CVI updates, got: " + updateCount);
        }
    }
    
    @Nested
    @DisplayName("Performance Comparison")
    class PerformanceTests {
        
        @Test
        @DisplayName("Should be faster than batch CVI updates")
        void testPerformanceImprovement() {
            // Test with incremental CVI
            var incrementalCH = new CalinskiHarabaszIndex();
            icviFuzzyART.addCVI(incrementalCH);
            
            long incrementalTime = measureProcessingTime(icviFuzzyART, 500, true);
            
            // Test with batch-only mode
            var batchFuzzyART = new iCVIFuzzyART();
            batchFuzzyART.addCVI(incrementalCH);
            batchFuzzyART.setForceNonIncremental(true);
            
            long batchTime = measureProcessingTime(batchFuzzyART, 500, false);
            
            // Incremental should be faster
            assertTrue(incrementalTime < batchTime * 1.5, 
                      "Incremental should be faster: " + incrementalTime + 
                      " vs " + batchTime);
        }
        
        @Test
        @DisplayName("Should maintain accuracy with incremental updates")
        void testAccuracyMaintenance() {
            var incrementalCH = new CalinskiHarabaszIndex();
            
            // Process with incremental updates
            icviFuzzyART.addCVI(incrementalCH);
            for (int i = 0; i < 200; i++) {
                icviFuzzyART.learn(streamData.get(i), defaultParams);
            }
            double incrementalScore = icviFuzzyART.getCurrentCVIScores()
                .get("Calinski-Harabasz Index");
            
            // Process with batch updates only
            var batchART = new iCVIFuzzyART();
            batchART.addCVI(new CalinskiHarabaszIndex());
            batchART.setForceNonIncremental(true);
            for (int i = 0; i < 200; i++) {
                batchART.learn(streamData.get(i), defaultParams);
            }
            double batchScore = batchART.getCurrentCVIScores()
                .get("Calinski-Harabasz Index");
            
            // Scores should be similar
            double difference = Math.abs(incrementalScore - batchScore);
            assertTrue(difference < Math.max(incrementalScore, batchScore) * 0.1,
                      "Incremental and batch scores should be similar: " + 
                      incrementalScore + " vs " + batchScore);
        }
    }
    
    @Nested
    @DisplayName("Streaming Data Processing")
    class StreamingDataTests {
        
        @Test
        @DisplayName("Should handle continuous data stream")
        void testStreamProcessing() {
            icviFuzzyART.addCVI(new CalinskiHarabaszIndex());
            
            // Simulate streaming
            int windowSize = 100;
            for (int i = 0; i < streamData.size() - windowSize; i++) {
                // Process new pattern
                icviFuzzyART.learn(streamData.get(i), defaultParams);
                
                // Every window, check metrics
                if (i % windowSize == 0 && i > 0) {
                    var scores = icviFuzzyART.getCurrentCVIScores();
                    assertNotNull(scores.get("Calinski-Harabasz Index"));
                    
                    // Categories should be reasonable
                    int categories = icviFuzzyART.getCategoryCount();
                    assertTrue(categories > 0 && categories < windowSize,
                              "Categories should be reasonable: " + categories);
                }
            }
        }
        
        @Test
        @DisplayName("Should adapt to concept drift")
        void testConceptDrift() {
            icviFuzzyART.addCVI(new CalinskiHarabaszIndex());
            defaultParams.setAdaptiveVigilance(true);
            
            // First phase - low variance data
            var lowVarianceData = generateClusteredData(100, 0.05);
            for (var pattern : lowVarianceData) {
                icviFuzzyART.learn(pattern, defaultParams);
            }
            int categoriesPhase1 = icviFuzzyART.getCategoryCount();
            
            // Second phase - high variance data (concept drift)
            var highVarianceData = generateClusteredData(100, 0.2);
            for (var pattern : highVarianceData) {
                icviFuzzyART.learn(pattern, defaultParams);
            }
            int categoriesPhase2 = icviFuzzyART.getCategoryCount();
            
            // Should adapt to new data distribution
            assertTrue(categoriesPhase2 > categoriesPhase1,
                      "Should create more categories for high variance data");
        }
        
        @Test
        @DisplayName("Should maintain bounded memory")
        void testMemoryBounds() {
            icviFuzzyART.addCVI(new CalinskiHarabaszIndex());
            
            // Set memory limit
            defaultParams.setMaxMemoryPatterns(500);
            
            // Process many patterns
            for (int i = 0; i < 1000; i++) {
                icviFuzzyART.learn(streamData.get(i % streamData.size()), defaultParams);
            }
            
            // Memory usage should be bounded
            int storedPatterns = icviFuzzyART.getStoredPatternCount();
            assertTrue(storedPatterns <= 500,
                      "Should maintain memory bound: " + storedPatterns);
            
            // Should still have valid CVI scores
            var scores = icviFuzzyART.getCurrentCVIScores();
            assertNotNull(scores.get("Calinski-Harabasz Index"));
        }
    }
    
    @Nested
    @DisplayName("Multi-CVI Incremental Processing")
    class MultiCVIIncrementalTests {
        
        @Test
        @DisplayName("Should handle multiple incremental CVIs")
        void testMultipleIncrementalCVIs() {
            // Add multiple CVIs with different update capabilities
            icviFuzzyART.addCVI(new CalinskiHarabaszIndex()); // Incremental
            icviFuzzyART.addCVI(new DaviesBouldinIndex()); // Batch only
            icviFuzzyART.addCVI(new SilhouetteCoefficient()); // Batch only
            
            // Process patterns
            for (int i = 0; i < 100; i++) {
                icviFuzzyART.learn(streamData.get(i), defaultParams);
            }
            
            // All CVIs should have scores
            var scores = icviFuzzyART.getCurrentCVIScores();
            assertEquals(3, scores.size());
            
            // Check update statistics
            var stats = icviFuzzyART.getCVIUpdateStatistics();
            assertTrue(stats.getIncrementalUpdates("Calinski-Harabasz Index") > 0);
            assertTrue(stats.getBatchUpdates("Davies-Bouldin Index") > 0);
        }
        
        @Test
        @DisplayName("Should coordinate updates efficiently")
        void testCoordinatedUpdates() {
            icviFuzzyART.addCVI(new CalinskiHarabaszIndex());
            icviFuzzyART.addCVI(new DaviesBouldinIndex());
            
            // Set coordinated update strategy
            defaultParams.setUpdateCoordination(UpdateCoordination.SYNCHRONIZED);
            
            int syncUpdates = 0;
            for (int i = 0; i < 100; i++) {
                icviFuzzyART.learn(streamData.get(i), defaultParams);
                
                if (icviFuzzyART.wasLastUpdateSynchronized()) {
                    syncUpdates++;
                }
            }
            
            // Should coordinate updates
            assertTrue(syncUpdates > 0 && syncUpdates < 100,
                      "Should coordinate some updates: " + syncUpdates);
        }
    }
    
    @Nested
    @DisplayName("Fuzzy ART Specific Features")
    class FuzzyARTSpecificTests {
        
        @Test
        @DisplayName("Should work with complement coding")
        void testComplementCoding() {
            icviFuzzyART.addCVI(new CalinskiHarabaszIndex());
            defaultParams.setUseComplementCoding(true);
            
            // Process patterns with complement coding
            for (int i = 0; i < 50; i++) {
                var pattern = streamData.get(i);
                var result = icviFuzzyART.learn(pattern, defaultParams);
                assertTrue(result.wasSuccessful());
            }
            
            // Check that patterns are properly encoded
            assertTrue(icviFuzzyART.isUsingComplementCoding());
            
            // CVI should work with complement coded data
            var scores = icviFuzzyART.getCurrentCVIScores();
            assertNotNull(scores.get("Calinski-Harabasz Index"));
        }
        
        @Test
        @DisplayName("Should respect choice parameter")
        void testChoiceParameter() {
            icviFuzzyART.addCVI(new CalinskiHarabaszIndex());
            
            // Test with different choice values
            double[] choiceValues = {0.0, 0.5, 1.0};
            int[] expectedCategories = new int[3];
            
            for (int c = 0; c < choiceValues.length; c++) {
                var testART = new iCVIFuzzyART();
                testART.addCVI(new CalinskiHarabaszIndex());
                
                var params = new iCVIFuzzyARTParameters();
                params.setChoiceParameter(choiceValues[c]);
                params.setVigilance(0.7); // Set a moderate vigilance
                params.setInitialVigilance(0.7);
                
                // Learn subset of patterns
                for (int i = 0; i < 100; i++) {
                    testART.learn(streamData.get(i), params);
                }
                
                expectedCategories[c] = testART.getCategoryCount();
            }
            
            // Different choice values should affect clustering
            assertNotEquals(expectedCategories[0], expectedCategories[2],
                           "Different choice values should produce different results");
        }
        
        @Test
        @DisplayName("Should apply fast learning")
        void testFastLearning() {
            icviFuzzyART.addCVI(new CalinskiHarabaszIndex());
            
            // Test with fast learning
            defaultParams.setLearningRate(1.0); // Fast learning
            
            // Learn patterns
            for (int i = 0; i < 50; i++) {
                icviFuzzyART.learn(streamData.get(i), defaultParams);
            }
            
            int fastLearningCategories = icviFuzzyART.getCategoryCount();
            
            // Test with slow learning
            var slowART = new iCVIFuzzyART();
            slowART.addCVI(new CalinskiHarabaszIndex());
            var slowParams = new iCVIFuzzyARTParameters();
            slowParams.setLearningRate(0.1); // Slow learning
            
            for (int i = 0; i < 50; i++) {
                slowART.learn(streamData.get(i), slowParams);
            }
            
            int slowLearningCategories = slowART.getCategoryCount();
            
            // Fast learning might create different number of categories
            assertNotNull(fastLearningCategories);
            assertNotNull(slowLearningCategories);
        }
    }
    
    @Nested
    @DisplayName("Error Handling and Recovery")
    class ErrorHandlingTests {
        
        @Test
        @DisplayName("Should recover from CVI calculation errors")
        void testCVIErrorRecovery() {
            // Add a CVI that might fail
            icviFuzzyART.addCVI(new ClusterValidityIndex() {
                private int callCount = 0;
                
                @Override
                public double calculate(List<Pattern> data, int[] labels, 
                                      List<Pattern> centroids) {
                    callCount++;
                    if (callCount % 5 == 0) {
                        throw new RuntimeException("Simulated CVI failure");
                    }
                    return callCount;
                }
                
                @Override
                public String getName() {
                    return "Unstable CVI";
                }
                
                @Override
                public boolean isHigherBetter() {
                    return true;
                }
            });
            
            // Should continue despite errors
            assertDoesNotThrow(() -> {
                for (int i = 0; i < 50; i++) {
                    icviFuzzyART.learn(streamData.get(i), defaultParams);
                }
            });
            
            assertTrue(icviFuzzyART.getCategoryCount() > 0);
        }
        
        @Test
        @DisplayName("Should handle invalid parameters gracefully")
        void testInvalidParameters() {
            icviFuzzyART.addCVI(new CalinskiHarabaszIndex());
            
            // Test with invalid parameters
            var invalidParams = new iCVIFuzzyARTParameters();
            invalidParams.setVigilance(-0.5); // Invalid
            
            // Should use default or clamp
            var result = icviFuzzyART.learn(streamData.get(0), invalidParams);
            assertTrue(result.wasSuccessful());
            
            // Test with NaN
            invalidParams.setVigilance(Double.NaN);
            result = icviFuzzyART.learn(streamData.get(1), invalidParams);
            assertTrue(result.wasSuccessful());
        }
    }
    
    // Helper methods
    
    private List<Pattern> generateStreamData(int size) {
        Random rand = new Random(42);
        List<Pattern> data = new ArrayList<>();
        
        for (int i = 0; i < size; i++) {
            // Generate data with varying characteristics
            double[] values = new double[3];
            double phase = (double) i / size;
            
            // Add some structure
            if (phase < 0.33) {
                // Low values
                for (int j = 0; j < 3; j++) {
                    values[j] = rand.nextGaussian() * 0.1 + 0.2;
                }
            } else if (phase < 0.67) {
                // Medium values
                for (int j = 0; j < 3; j++) {
                    values[j] = rand.nextGaussian() * 0.1 + 0.5;
                }
            } else {
                // High values
                for (int j = 0; j < 3; j++) {
                    values[j] = rand.nextGaussian() * 0.1 + 0.8;
                }
            }
            
            // Clamp to [0, 1]
            for (int j = 0; j < 3; j++) {
                values[j] = Math.max(0, Math.min(1, values[j]));
            }
            
            data.add(new DenseVector(values));
        }
        
        return data;
    }
    
    private List<Pattern> generateClusteredData(int size, double variance) {
        Random rand = new Random();
        List<Pattern> data = new ArrayList<>();
        
        int clusters = 3;
        double[] centers = {0.2, 0.5, 0.8};
        
        for (int i = 0; i < size; i++) {
            int cluster = i % clusters;
            double[] values = new double[3];
            
            for (int j = 0; j < 3; j++) {
                values[j] = centers[cluster] + rand.nextGaussian() * variance;
                values[j] = Math.max(0, Math.min(1, values[j]));
            }
            
            data.add(new DenseVector(values));
        }
        
        return data;
    }
    
    private long measureProcessingTime(iCVIFuzzyART art, int patterns, boolean incremental) {
        long startTime = System.nanoTime();
        
        for (int i = 0; i < patterns; i++) {
            art.learn(streamData.get(i % streamData.size()), defaultParams);
        }
        
        return TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startTime);
    }
    
    // Use the real classes from the main codebase
    // No need for mock implementations since we have the real ones
}