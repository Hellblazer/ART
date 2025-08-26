/*
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 */

package com.hellblazer.art.algorithms;

import com.hellblazer.art.core.*;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.core.parameters.HypersphereParameters;
import com.hellblazer.art.core.algorithms.HypersphereART;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import com.hellblazer.art.core.Pattern;

/**
 * Comprehensive test suite for VectorizedHypersphereART
 * Tests SIMD-optimized distance calculations, learning, and performance
 */
@DisplayName("VectorizedHypersphereART Tests")
public class VectorizedHypersphereARTTest {

    private VectorizedHypersphereART network;
    private VectorizedHypersphereParameters parameters;

    @BeforeEach
    void setUp() {
        parameters = VectorizedHypersphereParameters.builder()
                .vigilance(0.8)
                .learningRate(0.5)
                .inputDimensions(4)
                .maxCategories(10)
                .enableSIMD(true)
                .build();
        
        network = new VectorizedHypersphereART(parameters);
    }

    @Nested
    @DisplayName("Initialization Tests")
    class InitializationTests {

        @Test
        @DisplayName("Network initializes with correct parameters")
        void testInitialization() {
            assertEquals(0.8, network.getVigilance());
            assertEquals(0.5, network.getLearningRate());
            assertEquals(4, network.getInputDimensions());
            assertEquals(10, network.getMaxCategories());
            assertEquals(0, network.getCategoryCount());
            assertTrue(network.isSIMDEnabled());
        }

        @Test
        @DisplayName("Conservative parameters create valid network")
        void testConservativeParameters() {
            var conservativeParams = VectorizedHypersphereParameters.conservative(4);
            var conservativeNetwork = new VectorizedHypersphereART(conservativeParams);
            
            assertEquals(0.9, conservativeNetwork.getVigilance());
            assertEquals(0.1, conservativeNetwork.getLearningRate());
            assertTrue(conservativeNetwork.isSIMDEnabled());
        }

        @Test
        @DisplayName("High performance parameters create valid network")
        void testHighPerformanceParameters() {
            var highPerfParams = VectorizedHypersphereParameters.highPerformance(8);
            var highPerfNetwork = new VectorizedHypersphereART(highPerfParams);
            
            assertEquals(0.7, highPerfNetwork.getVigilance());
            assertEquals(0.8, highPerfNetwork.getLearningRate());
            assertTrue(highPerfNetwork.isSIMDEnabled());
        }
    }

    @Nested
    @DisplayName("Basic Learning Tests")  
    class BasicLearningTests {

        @Test
        @DisplayName("First pattern creates new category")
        void testFirstPatternLearning() {
            var pattern = createPattern(1.0, 2.0, 3.0, 4.0);
            
            int categoryIndex = network.learn(pattern);
            
            assertEquals(0, categoryIndex);
            assertEquals(1, network.getCategoryCount());
        }

        @Test
        @DisplayName("Similar patterns merge into same category")
        void testSimilarPatternMerging() {
            var pattern1 = createPattern(1.0, 2.0, 3.0, 4.0);
            var pattern2 = createPattern(1.1, 2.1, 3.1, 4.1); // Very similar
            
            int category1 = network.learn(pattern1);
            int category2 = network.learn(pattern2);
            
            assertEquals(category1, category2);
            assertEquals(1, network.getCategoryCount());
        }

        @Test
        @DisplayName("Dissimilar patterns create separate categories")
        void testDissimilarPatternSeparation() {
            var pattern1 = createPattern(1.0, 1.0, 1.0, 1.0);
            var pattern2 = createPattern(10.0, 10.0, 10.0, 10.0); // Very different
            
            int category1 = network.learn(pattern1);
            int category2 = network.learn(pattern2);
            
            assertNotEquals(category1, category2);
            assertEquals(2, network.getCategoryCount());
        }
    }

    @Nested
    @DisplayName("Classification Tests")
    class ClassificationTests {

        @Test
        @DisplayName("Classify returns correct category for learned pattern")
        void testClassifyLearnedPattern() {
            var pattern = createPattern(1.0, 2.0, 3.0, 4.0);
            int learnedCategory = network.learn(pattern);
            
            int classifiedCategory = network.classify(pattern);
            
            assertEquals(learnedCategory, classifiedCategory);
        }

        @Test
        @DisplayName("Classify returns -1 for unrecognized pattern")
        void testClassifyUnrecognizedPattern() {
            var learnedPattern = createPattern(1.0, 1.0, 1.0, 1.0);
            var testPattern = createPattern(10.0, 10.0, 10.0, 10.0);
            
            network.learn(learnedPattern);
            int result = network.classify(testPattern);
            
            assertEquals(-1, result);
        }

        @Test
        @DisplayName("Classify finds best matching category within vigilance")
        void testClassifyBestMatch() {
            var pattern1 = createPattern(1.0, 1.0, 1.0, 1.0);
            var pattern2 = createPattern(5.0, 5.0, 5.0, 5.0);
            var testPattern = createPattern(1.2, 1.2, 1.2, 1.2); // Closer to pattern1
            
            int category1 = network.learn(pattern1);
            int category2 = network.learn(pattern2);
            int classifiedCategory = network.classify(testPattern);
            
            assertEquals(category1, classifiedCategory);
        }
    }

    @Nested
    @DisplayName("Distance Calculation Tests")
    class DistanceCalculationTests {

        @Test
        @DisplayName("Distance calculation is accurate for known vectors")
        void testDistanceAccuracy() {
            var pattern1 = createPattern(0.0, 0.0, 0.0, 0.0);
            var pattern2 = createPattern(3.0, 4.0, 0.0, 0.0); // Distance should be 5.0
            
            // Learn first pattern to create a category
            network.learn(pattern1);
            
            // Use reflection or package-private access to test distance calculation
            // For now, we'll test indirectly through classification behavior
            int category1 = network.classify(pattern1);
            int category2 = network.classify(pattern2);
            
            assertEquals(0, category1); // Should match learned category
            assertEquals(-1, category2); // Should not match due to distance
        }

        @Test
        @DisplayName("SIMD and scalar distance calculations produce same results")
        void testSIMDVsScalarConsistency() {
            var simdParams = VectorizedHypersphereParameters.highPerformance(8);
            var scalarParams = VectorizedHypersphereParameters.scalarOnly(8);
            
            var simdNetwork = new VectorizedHypersphereART(simdParams);
            var scalarNetwork = new VectorizedHypersphereART(scalarParams);
            
            var pattern = createPattern(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
            
            int simdCategory = simdNetwork.learn(pattern);
            int scalarCategory = scalarNetwork.learn(pattern);
            
            assertEquals(simdCategory, scalarCategory);
            
            // Test classification consistency
            int simdClassify = simdNetwork.classify(pattern);
            int scalarClassify = scalarNetwork.classify(pattern);
            
            assertEquals(simdClassify, scalarClassify);
        }
    }

    @Nested
    @DisplayName("Vigilance Parameter Tests")
    class VigilanceTests {

        @Test
        @DisplayName("High vigilance creates more categories")
        void testHighVigilance() {
            var highVigilanceParams = parameters.toBuilder().vigilance(0.95).build();
            var highVigilanceNetwork = new VectorizedHypersphereART(highVigilanceParams);
            
            var patterns = Arrays.asList(
                createPattern(1.0, 1.0, 1.0, 1.0),
                createPattern(1.5, 1.5, 1.5, 1.5),
                createPattern(2.0, 2.0, 2.0, 2.0)
            );
            
            patterns.forEach(highVigilanceNetwork::learn);
            
            // High vigilance should create more categories
            assertTrue(highVigilanceNetwork.getCategoryCount() >= 2);
        }

        @Test
        @DisplayName("Low vigilance creates fewer categories")
        void testLowVigilance() {
            var lowVigilanceParams = parameters.toBuilder().vigilance(0.1).build(); // Very low vigilance
            var lowVigilanceNetwork = new VectorizedHypersphereART(lowVigilanceParams);
            
            var patterns = Arrays.asList(
                createPattern(1.0, 1.0, 1.0, 1.0),
                createPattern(1.2, 1.2, 1.2, 1.2), // Close to first
                createPattern(1.4, 1.4, 1.4, 1.4)  // Close to first two
            );
            
            patterns.forEach(lowVigilanceNetwork::learn);
            
            // Low vigilance should merge similar patterns into fewer categories
            assertTrue(lowVigilanceNetwork.getCategoryCount() <= 2, 
                "Expected <= 2 categories but got " + lowVigilanceNetwork.getCategoryCount());
        }
    }

    @Nested
    @DisplayName("Edge Cases and Error Handling")
    class EdgeCaseTests {

        @Test
        @DisplayName("Handles zero vectors correctly")
        void testZeroVector() {
            var zeroPattern = createPattern(0.0, 0.0, 0.0, 0.0);
            
            assertDoesNotThrow(() -> {
                int category = network.learn(zeroPattern);
                assertEquals(0, category);
            });
        }

        @Test
        @DisplayName("Handles maximum categories limit")
        void testMaxCategoriesLimit() {
            var limitedParams = parameters.toBuilder()
                .maxCategories(2)
                .vigilance(0.99) // Very high vigilance to force category creation
                .build();
            var limitedNetwork = new VectorizedHypersphereART(limitedParams);
            
            var patterns = Arrays.asList(
                createPattern(1.0, 0.0, 0.0, 0.0),
                createPattern(0.0, 1.0, 0.0, 0.0),
                createPattern(0.0, 0.0, 1.0, 0.0), // Should cause exception
                createPattern(0.0, 0.0, 0.0, 1.0)  // Won't be reached
            );
            
            // Learn first two patterns
            limitedNetwork.learn(patterns.get(0));
            limitedNetwork.learn(patterns.get(1));
            assertEquals(2, limitedNetwork.getCategoryCount());
            
            // Third pattern should cause exception due to max categories limit
            assertThrows(IllegalStateException.class, () -> {
                limitedNetwork.learn(patterns.get(2));
            });
        }

        @Test
        @DisplayName("Handles negative values in patterns")
        void testNegativeValues() {
            var negativePattern = createPattern(-1.0, -2.0, -3.0, -4.0);
            
            assertDoesNotThrow(() -> {
                int category = network.learn(negativePattern);
                assertEquals(0, category);
                
                int classified = network.classify(negativePattern);
                assertEquals(category, classified);
            });
        }

        @Test
        @DisplayName("Throws exception for wrong pattern dimension")
        void testWrongPatternDimension() {
            var wrongSizePattern = createPattern(1.0, 2.0); // Should be 4 dimensions
            
            assertThrows(IllegalArgumentException.class, () -> {
                network.learn(wrongSizePattern);
            });
        }
    }

    @Nested
    @DisplayName("Performance and Stress Tests")
    class PerformanceTests {

        @Test
        @DisplayName("Handles large number of patterns efficiently")
        void testLargePatternSet() {
            // Use network with much higher category limit for this test
            var highCapacityParams = parameters.toBuilder().maxCategories(600).build();
            var highCapacityNetwork = new VectorizedHypersphereART(highCapacityParams);
            
            var patterns = generateRandomPatterns(500, 4); // CI-friendly size
            
            long startTime = System.currentTimeMillis();
            
            patterns.forEach(highCapacityNetwork::learn);
            
            long endTime = System.currentTimeMillis();
            long duration = endTime - startTime;
            
            // Should complete within reasonable time (adjust threshold as needed)
            assertTrue(duration < 5000, "Learning should complete within 5 seconds");
            assertTrue(highCapacityNetwork.getCategoryCount() > 0, "Should create at least one category");
            assertTrue(highCapacityNetwork.getCategoryCount() <= patterns.size(), "Should not exceed input pattern count");
        }

        @Test
        @DisplayName("SIMD performance comparison with scalar")
        void testSIMDPerformance() {
            var simdParams = VectorizedHypersphereParameters.highPerformance(16);
            var scalarParams = VectorizedHypersphereParameters.scalarOnly(16);
            
            var simdNetwork = new VectorizedHypersphereART(simdParams);
            var scalarNetwork = new VectorizedHypersphereART(scalarParams);
            
            var patterns = generateRandomPatterns(200, 16); // CI-friendly size
            
            // Test SIMD performance
            long simdStart = System.nanoTime();
            patterns.forEach(simdNetwork::learn);
            long simdTime = System.nanoTime() - simdStart;
            
            // Test scalar performance  
            long scalarStart = System.nanoTime();
            patterns.forEach(scalarNetwork::learn);
            long scalarTime = System.nanoTime() - scalarStart;
            
            // Log performance for informational purposes
            System.out.printf("SIMD time: %d ns, Scalar time: %d ns, Ratio: %.2f%n", 
                simdTime, scalarTime, (double) simdTime / scalarTime);
            
            // For small datasets, SIMD may have overhead - just verify functionality
            // Both should produce same number of categories (functional correctness)
            assertEquals(scalarNetwork.getCategoryCount(), simdNetwork.getCategoryCount());
            
            // Verify both produce meaningful results
            assertTrue(simdNetwork.getCategoryCount() > 0, "SIMD network should learn categories");
            assertTrue(scalarNetwork.getCategoryCount() > 0, "Scalar network should learn categories");
        }

        @Test
        @DisplayName("Memory usage remains reasonable with many categories")
        void testMemoryUsage() {
            // Use network with higher capacity for this test
            var highCapacityParams = parameters.toBuilder().maxCategories(100).build();
            var highCapacityNetwork = new VectorizedHypersphereART(highCapacityParams);
            
            var patterns = generateRandomPatterns(100, 4); // Match network dimension
            
            long initialMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
            
            patterns.forEach(highCapacityNetwork::learn);
            
            System.gc(); // Encourage garbage collection
            
            long finalMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
            long memoryIncrease = finalMemory - initialMemory;
            
            // Memory increase should be reasonable (adjust threshold as needed)
            assertTrue(memoryIncrease < 50_000_000, // 50MB threshold
                "Memory usage should remain reasonable");
        }
    }

    // Helper methods

    private Pattern createPattern(double... values) {
        return Pattern.of(values);
    }

    private List<Pattern> generateRandomPatterns(int count, int dimensions) {
        var random = ThreadLocalRandom.current();
        return IntStream.range(0, count)
            .mapToObj(i -> {
                var values = new double[dimensions];
                for (int j = 0; j < dimensions; j++) {
                    values[j] = random.nextGaussian() * 5.0; // Random values with some spread
                }
                return Pattern.of(values);
            })
            .toList();
    }
}