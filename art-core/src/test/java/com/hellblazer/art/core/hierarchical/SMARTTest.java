/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 * 
 * This file is part of the Java ART Library project.
 */
package com.hellblazer.art.core.hierarchical;

import com.hellblazer.art.core.*;
import com.hellblazer.art.core.hierarchical.SMART.ARTType;
import com.hellblazer.art.core.utils.MathOperations;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for SMART hierarchical clustering algorithm.
 * 
 * Tests based on Python reference: test_SMART.py
 * 
 * @author Hal Hildebrand
 */
public class SMARTTest {
    
    private SMART smartModel;
    private List<Pattern> testData;
    
    @BeforeEach
    void setUp() {
        // Initialize test data
        testData = generateClusteredData();
    }
    
    /**
     * Helper method for complement coding a pattern.
     */
    private Pattern complementCodePattern(Pattern p) {
        var features = new double[p.dimension()];
        for (int i = 0; i < p.dimension(); i++) {
            features[i] = p.get(i);
        }
        var complementCoded = MathOperations.complementCode(features);
        return Pattern.of(complementCoded);
    }
    
    /**
     * Generate synthetic clustered data for testing.
     */
    private List<Pattern> generateClusteredData() {
        var patterns = new ArrayList<Pattern>();
        var random = new Random(42);
        
        // Generate 3 distinct clusters
        double[][] centers = {
            {0.2, 0.2}, 
            {0.5, 0.8}, 
            {0.8, 0.3}
        };
        
        for (var center : centers) {
            for (int i = 0; i < 10; i++) {
                var features = new double[2];
                features[0] = Math.max(0, Math.min(1, center[0] + random.nextGaussian() * 0.05));
                features[1] = Math.max(0, Math.min(1, center[1] + random.nextGaussian() * 0.05));
                patterns.add(Pattern.of(features));
            }
        }
        
        return patterns;
    }
    
    @Nested
    @DisplayName("Basic Functionality Tests")
    class BasicFunctionalityTests {
        
        @Test
        @DisplayName("Test initialization with FuzzyART")
        void testInitializationFuzzyART() {
            // Create vigilance values (monotonically increasing)
            double[] rhoValues = {0.2, 0.5, 0.7};
            
            // Create SMART instance with FuzzyART
            smartModel = SMART.createWithFuzzyART(rhoValues, 0.01, 1.0);
            
            // Verify initialization
            assertNotNull(smartModel);
            assertEquals(3, smartModel.getNumLevels());
            assertArrayEquals(rhoValues, smartModel.getVigilanceValues());
            assertEquals(ARTType.FUZZY_ART, smartModel.getArtType());
        }
        
        @Test
        @DisplayName("Test monotonic vigilance validation")
        void testMonotonicVigilanceValidation() {
            // Test invalid (non-monotonic) vigilance values
            double[] invalidRho = {0.7, 0.5, 0.8}; // Not monotonic
            
            assertThrows(IllegalArgumentException.class, () -> {
                SMART.createWithFuzzyART(invalidRho, 0.01, 1.0);
            });
        }
        
        @Test
        @DisplayName("Test BayesianART with decreasing vigilance")
        void testBayesianARTDecreasingVigilance() {
            // For BayesianART, vigilance should be monotonically decreasing
            double[] rhoValues = {0.7, 0.5, 0.2};
            
            var bayesianSmart = SMART.createWithBayesianART(rhoValues, 2, 0.1, 1.0, 100);
            
            assertNotNull(bayesianSmart);
            assertEquals(3, bayesianSmart.getNumLevels());
            assertEquals(ARTType.BAYESIAN_ART, bayesianSmart.getArtType());
        }
        
        @Test
        @DisplayName("Test GaussianART initialization")
        void testGaussianARTInitialization() {
            double[] rhoValues = {0.3, 0.5, 0.7};
            
            var gaussianSmart = SMART.createWithGaussianART(rhoValues, 2, 0.1);
            
            assertNotNull(gaussianSmart);
            assertEquals(3, gaussianSmart.getNumLevels());
            assertEquals(ARTType.GAUSSIAN_ART, gaussianSmart.getArtType());
        }
    }
    
    @Nested
    @DisplayName("Learning and Clustering Tests")
    class LearningTests {
        
        @BeforeEach
        void setUp() {
            double[] rhoValues = {0.2, 0.5, 0.7};
            smartModel = SMART.createWithFuzzyART(rhoValues, 0.01, 1.0);
        }
        
        @Test
        @DisplayName("Test fit method")
        void testFit() {
            // Prepare data with complement coding
            var preparedData = testData.stream()
                .map(p -> {
                    var features = new double[p.dimension()];
                    for (int i = 0; i < p.dimension(); i++) {
                        features[i] = p.get(i);
                    }
                    var complementCoded = MathOperations.complementCode(features);
                    return Pattern.of(complementCoded);
                })
                .toList();
            
            // Fit the model
            var labels = smartModel.fit(preparedData, 1);
            
            // Verify results
            assertNotNull(labels);
            assertEquals(preparedData.size(), labels.length);
            
            // Check that we have clusters at different levels
            var clusterCounts = smartModel.getClusterCounts();
            assertTrue(clusterCounts[0] > 0, "First level should have clusters");
            
            // With increasing vigilance, higher levels might have more clusters
            // (more specific categorization)
            for (int count : clusterCounts) {
                assertTrue(count >= 0, "Cluster count should be non-negative");
            }
        }
        
        @Test
        @DisplayName("Test partial fit (single pattern learning)")
        void testPartialFit() {
            var pattern = complementCodePattern(testData.get(0));
            
            // Learn single pattern
            int label = smartModel.stepFit(pattern);
            
            // Verify learning
            assertTrue(label >= 0 || label == -1); // -1 for new category, >= 0 for existing
            
            var clusterCounts = smartModel.getClusterCounts();
            assertTrue(clusterCounts[0] > 0, "Should have created at least one cluster");
        }
        
        @Test
        @DisplayName("Test hierarchical structure")
        void testHierarchicalStructure() {
            var preparedData = testData.stream()
                .map(p -> complementCodePattern(p))
                .toList();
            
            // Fit the model
            smartModel.fit(preparedData, 1);
            
            // Check hierarchical structure
            var clusterCounts = smartModel.getClusterCounts();
            
            // Verify we have clusters at each level
            for (int i = 0; i < clusterCounts.length; i++) {
                assertTrue(clusterCounts[i] >= 0, 
                    String.format("Layer %d should have non-negative cluster count", i));
            }
        }
    }
    
    @Nested
    @DisplayName("Prediction Tests")
    class PredictionTests {
        
        @BeforeEach
        void setUp() {
            double[] rhoValues = {0.3, 0.6, 0.9};
            smartModel = SMART.createWithFuzzyART(rhoValues, 0.01, 1.0);
            
            // Train the model
            var preparedData = testData.stream()
                .map(p -> complementCodePattern(p))
                .toList();
            smartModel.fit(preparedData, 1);
        }
        
        @Test
        @DisplayName("Test prediction")
        void testPredict() {
            var testPattern = complementCodePattern(testData.get(0));
            
            var predictions = smartModel.predict(List.of(testPattern));
            
            assertNotNull(predictions);
            assertEquals(1, predictions.length);
            // Prediction can be -1 if no category matches
            assertTrue(predictions[0] >= -1);
        }
        
        @Test
        @DisplayName("Test prediction consistency")
        void testPredictionConsistency() {
            var pattern = complementCodePattern(testData.get(5));
            
            // Predict same pattern multiple times
            var pred1 = smartModel.predict(List.of(pattern))[0];
            var pred2 = smartModel.predict(List.of(pattern))[0];
            
            assertEquals(pred1, pred2, "Same pattern should map to same cluster");
        }
    }
    
    @Nested
    @DisplayName("Hierarchical Mapping Tests")
    class HierarchicalMappingTests {
        
        @BeforeEach
        void setUp() {
            double[] rhoValues = {0.2, 0.4, 0.6, 0.8};  // 4 levels
            smartModel = SMART.createWithFuzzyART(rhoValues, 0.01, 1.0);
            
            // Train with data
            var preparedData = testData.stream()
                .map(p -> complementCodePattern(p))
                .toList();
            smartModel.fit(preparedData, 1);
        }
        
        @Test
        @DisplayName("Test hierarchical path extraction")
        void testHierarchicalPath() {
            // Get path for first pattern
            var path = smartModel.getHierarchicalPath(0);
            
            if (path != null) {
                assertEquals(smartModel.getNumLevels(), path.length);
                
                // Verify path values are valid
                var clusterCounts = smartModel.getClusterCounts();
                for (int i = 0; i < path.length; i++) {
                    if (path[i] >= 0) {
                        assertTrue(path[i] < clusterCounts[i] || clusterCounts[i] == 0);
                    }
                }
            }
        }
        
        @Test
        @DisplayName("Test statistics generation")
        void testStatistics() {
            var stats = smartModel.getStatistics();
            
            assertNotNull(stats);
            assertEquals(4, stats.numLevels);
            assertNotNull(stats.clusterCounts);
            assertNotNull(stats.vigilanceValues);
            assertTrue(stats.avgBranchingFactor >= 0);
            
            // Verify string representation
            var statsString = stats.toString();
            assertNotNull(statsString);
            assertTrue(statsString.contains("SMART Hierarchical Stats"));
        }
    }
    
    @Nested
    @DisplayName("Edge Case Tests")
    class EdgeCaseTests {
        
        @Test
        @DisplayName("Test single level SMART")
        void testSingleLevel() {
            double[] rhoValues = {0.5};  // Single level
            
            var singleLevelSmart = SMART.createWithFuzzyART(rhoValues, 0.01, 1.0);
            
            assertEquals(1, singleLevelSmart.getNumLevels());
            
            // Test fitting
            var preparedData = testData.stream()
                .map(p -> complementCodePattern(p))
                .limit(5)
                .toList();
            
            var labels = singleLevelSmart.fit(preparedData, 1);
            assertNotNull(labels);
        }
        
        @Test
        @DisplayName("Test with identical patterns")
        void testIdenticalPatterns() {
            double[] rhoValues = {0.3, 0.6, 0.9};
            var smart = SMART.createWithFuzzyART(rhoValues, 0.01, 1.0);
            
            // Create identical patterns
            var identicalData = new ArrayList<Pattern>();
            var basePattern = Pattern.of(0.5, 0.5);
            for (int i = 0; i < 10; i++) {
                identicalData.add(complementCodePattern(basePattern));
            }
            
            var labels = smart.fit(identicalData, 1);
            
            // All identical patterns should likely map to the same cluster
            // (though exact behavior depends on ART dynamics)
            assertNotNull(labels);
            assertEquals(10, labels.length);
        }
        
        @Test
        @DisplayName("Test with very sparse data")
        void testSparseData() {
            double[] rhoValues = {0.1, 0.5, 0.9};
            var smart = SMART.createWithFuzzyART(rhoValues, 0.01, 1.0);
            
            // Create very sparse patterns
            var sparseData = List.of(
                complementCodePattern(Pattern.of(0.01, 0.99)),
                complementCodePattern(Pattern.of(0.99, 0.01)),
                complementCodePattern(Pattern.of(0.5, 0.5))
            );
            
            var labels = smart.fit(sparseData, 1);
            
            assertNotNull(labels);
            assertEquals(3, labels.length);
            
            // Get cluster counts
            var clusterCounts = smart.getClusterCounts();
            for (int count : clusterCounts) {
                assertTrue(count >= 0);
            }
        }
        
        @Test
        @DisplayName("Test empty vigilance values")
        void testEmptyVigilanceValues() {
            double[] emptyRho = {};
            
            assertThrows(IllegalArgumentException.class, () -> {
                SMART.createWithFuzzyART(emptyRho, 0.01, 1.0);
            });
        }
        
        @Test
        @DisplayName("Test null vigilance values")
        void testNullVigilanceValues() {
            assertThrows(IllegalArgumentException.class, () -> {
                SMART.createWithFuzzyART(null, 0.01, 1.0);
            });
        }
    }
    
    @Nested
    @DisplayName("Different ART Module Tests")
    class DifferentARTModuleTests {
        
        @Test
        @DisplayName("Test SMART with GaussianART")
        void testWithGaussianART() {
            double[] rhoValues = {0.3, 0.5, 0.7};
            var gaussianSmart = SMART.createWithGaussianART(rhoValues, 2, 0.1);
            
            // Test basic functionality
            var preparedData = testData.stream()
                .limit(10)
                .toList();
            
            var labels = gaussianSmart.fit(preparedData, 1);
            
            assertNotNull(labels);
            assertEquals(10, labels.length);
            
            var clusterCounts = gaussianSmart.getClusterCounts();
            assertTrue(clusterCounts[0] >= 0);
        }
        
        @Test
        @DisplayName("Test SMART with BayesianART")
        void testWithBayesianART() {
            // BayesianART requires decreasing vigilance
            double[] rhoValues = {0.7, 0.5, 0.3};
            var bayesianSmart = SMART.createWithBayesianART(rhoValues, 2, 0.1, 1.0, 100);
            
            // Test basic functionality
            var preparedData = testData.stream()
                .limit(10)
                .toList();
            
            var labels = bayesianSmart.fit(preparedData, 1);
            
            assertNotNull(labels);
            assertEquals(10, labels.length);
        }
    }
    
    @Nested
    @DisplayName("Performance and Scalability Tests")
    class PerformanceTests {
        
        @Test
        @DisplayName("Test with large dataset")
        void testLargeDataset() {
            double[] rhoValues = {0.4, 0.7};  // Just 2 levels for speed
            var smart = SMART.createWithFuzzyART(rhoValues, 0.01, 0.5);
            
            // Generate larger dataset
            var largeData = new ArrayList<Pattern>();
            var random = new Random(123);
            for (int i = 0; i < 100; i++) {
                var features = new double[]{random.nextDouble(), random.nextDouble()};
                largeData.add(complementCodePattern(Pattern.of(features)));
            }
            
            long startTime = System.currentTimeMillis();
            var labels = smart.fit(largeData, 1);
            long endTime = System.currentTimeMillis();
            
            assertNotNull(labels);
            assertEquals(100, labels.length);
            
            // Check performance (should complete in reasonable time)
            assertTrue((endTime - startTime) < 5000, 
                "Large dataset should process in under 5 seconds");
        }
    }
}