/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 * 
 * This file is part of Java ART Neural Networks.
 * 
 * Java ART Neural Networks is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Java ART Neural Networks is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with Java ART Neural Networks. If not, see <https://www.gnu.org/licenses/>.
 */
package com.hellblazer.art.core;

import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.concurrent.TimeUnit;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;
import static org.assertj.core.api.Assertions.*;

/**
 * Comprehensive test suite for ART-2 implementation.
 * Written BEFORE any ART-2 implementation exists following test-first methodology.
 * These tests define the complete specification and expected behavior.
 * 
 * ART-2 Key Features:
 * - Continuous input processing (not binary like ART-1)
 * - Input normalization preprocessing layer
 * - Dot product activation function
 * - Distance-based vigilance criterion
 * - Convex combination learning rule
 * 
 * @author Hal Hildebrand
 */
class ART2Test {
    
    private static final double TOLERANCE = 1e-6;
    private static final int DEFAULT_MAX_CATEGORIES = 1000;
    
    protected ART2 createART() {
        var params = new ART2Parameters(0.7, 0.5, DEFAULT_MAX_CATEGORIES);
        return new ART2(params);
    }
    
    @Nested
    @DisplayName("Mathematical Correctness Tests")
    class MathematicalCorrectness {
        
        @Test
        @DisplayName("Should normalize input vectors to unit length")
        void shouldNormalizeInputVectors() {
            var art = createART();
            var rawInput = new DenseVector(new double[]{3.0, 4.0}); // ||v|| = 5
            
            // When: Normalizing input
            var normalized = art.normalizeInput(rawInput);
            
            // Then: Should have unit length
            // Cast Pattern to DenseVector to access values()
            if (!(normalized instanceof DenseVector)) {
                org.junit.jupiter.api.Assertions.fail("normalizeInput should return DenseVector");
            }
            var denseNormalized = (DenseVector) normalized;
            var values = denseNormalized.values();
            var norm = Math.sqrt(values[0]*values[0] + values[1]*values[1]);
            assertThat(norm).isCloseTo(1.0, within(TOLERANCE));
            
            // Should maintain direction
            var expectedX = 3.0 / 5.0; // 0.6
            var expectedY = 4.0 / 5.0; // 0.8
            assertThat(values[0]).isCloseTo(expectedX, within(TOLERANCE));
            assertThat(values[1]).isCloseTo(expectedY, within(TOLERANCE));
        }
        
        @Test
        @DisplayName("Should handle zero vectors in normalization")
        void shouldHandleZeroVectorsInNormalization() {
            var art = createART();
            var zeroInput = new DenseVector(new double[]{0.0, 0.0});
            
            // When: Normalizing zero vector
            // Then: Should handle gracefully (return small random vector or throw exception)
            assertThatCode(() -> art.normalizeInput(zeroInput))
                .doesNotThrowAnyException();
        }
        
        @Test
        @DisplayName("Should calculate dot product activation correctly")
        void shouldCalculateDotProductActivation() {
            var art = createART();
            var input = new DenseVector(new double[]{0.6, 0.8}); // Unit vector
            var weight = new ART2Weight(new DenseVector(new double[]{0.8, 0.6})); // Unit vector
            var params = new ART2Parameters(0.7, 0.5, 100);
            
            // When: Calculating activation
            var activation = art.calculateActivation(Pattern.of(input.values()), weight, params);
            
            // Then: Should be dot product = 0.6*0.8 + 0.8*0.6 = 0.96
            var expected = 0.6 * 0.8 + 0.8 * 0.6;
            assertThat(activation).isCloseTo(expected, within(TOLERANCE));
        }
        
        @Test
        @DisplayName("Should perform distance-based vigilance test")
        void shouldPerformDistanceBasedVigilanceTest() {
            var art = createART();
            var input = new DenseVector(new double[]{0.6, 0.8}); // Unit vector  
            var closeWeight = new ART2Weight(new DenseVector(new double[]{0.65, 0.76})); // Close
            var farWeight = new ART2Weight(new DenseVector(new double[]{-0.6, -0.8})); // Opposite
            var params = new ART2Parameters(0.8, 0.5, 100); // High vigilance
            
            // When: Testing vigilance
            var matchClose = art.checkVigilance(Pattern.of(input.values()), closeWeight, params);
            var matchFar = art.checkVigilance(Pattern.of(input.values()), farWeight, params);
            
            // Then: Close pattern should pass, far should fail
            assertThat(matchClose).isInstanceOf(MatchResult.Accepted.class);
            assertThat(matchFar).isInstanceOf(MatchResult.Rejected.class);
        }
        
        @Test
        @DisplayName("Should update weights using convex combination")
        void shouldUpdateWeightsUsingConvexCombination() {
            var art = createART();
            var input = new DenseVector(new double[]{0.6, 0.8}); // Unit vector
            var oldWeight = new ART2Weight(new DenseVector(new double[]{1.0, 0.0})); // Unit vector
            var params = new ART2Parameters(0.7, 0.3, 100); // Learning rate β = 0.3
            
            // When: Updating weights
            var newWeight = art.updateWeights(Pattern.of(input.values()), oldWeight, params);
            
            // Then: Should be normalized convex combination: (1-β)*old + β*input, then normalized
            var rawCombination = new double[]{
                0.7 * 1.0 + 0.3 * 0.6, // 0.7 + 0.18 = 0.88
                0.7 * 0.0 + 0.3 * 0.8  // 0.0 + 0.24 = 0.24
            };
            // Normalize: norm = √(0.88² + 0.24²) = √0.8320 = 0.9122
            var norm = Math.sqrt(rawCombination[0] * rawCombination[0] + rawCombination[1] * rawCombination[1]);
            var expected = new double[]{
                rawCombination[0] / norm, // 0.88 / 0.9122 = 0.9647
                rawCombination[1] / norm  // 0.24 / 0.9122 = 0.2630
            };
            
            assertThat(newWeight).isInstanceOf(ART2Weight.class);
            var art2Weight = (ART2Weight) newWeight;
            assertThat(art2Weight.vector().values()[0]).isCloseTo(expected[0], within(TOLERANCE));
            assertThat(art2Weight.vector().values()[1]).isCloseTo(expected[1], within(TOLERANCE));
        }
        
        @Test
        @DisplayName("Should maintain weight normalization after updates")
        void shouldMaintainWeightNormalizationAfterUpdates() {
            var art = createART();
            var input = new DenseVector(new double[]{0.6, 0.8}); // Unit vector
            var oldWeight = new ART2Weight(new DenseVector(new double[]{1.0, 0.0})); // Unit vector
            var params = new ART2Parameters(0.7, 0.5, 100);
            
            // When: Updating weights
            var newWeight = art.updateWeights(Pattern.of(input.values()), oldWeight, params);
            
            // Then: Updated weight should be normalized to unit length
            var art2Weight = (ART2Weight) newWeight;
            var values = art2Weight.vector().values();
            var norm = Math.sqrt(values[0]*values[0] + values[1]*values[1]);
            assertThat(norm).isCloseTo(1.0, within(TOLERANCE));
        }
    }
    
    @Nested
    @DisplayName("ART-2 Specific Functionality Tests")
    class ART2SpecificTests {
        
        @Test
        @DisplayName("Should handle continuous input patterns")
        void shouldHandleContinuousInputPatterns() {
            var art = createART();
            var continuousData = new Pattern[]{
                Pattern.of(new double[]{1.5, 2.7, 0.8}),
                Pattern.of(new double[]{1.6, 2.8, 0.9}),
                Pattern.of(new double[]{4.1, 1.2, 3.3})
            };
            
            // When: Training on continuous data
            art.fit(continuousData);
            
            // Then: Should create appropriate categories
            assertTrue(art.is_fitted(), "Should be fitted after training");
            assertTrue(art.getCategoryCount() > 0, "Should have created categories");
            
            // Should handle prediction on continuous test data
            var testPattern = Pattern.of(new double[]{1.55, 2.75, 0.85});
            var result = art.predict(testPattern);
            
            assertThat(result).isInstanceOf(ActivationResult.Success.class);
            var success = (ActivationResult.Success) result;
            assertThat(success.categoryIndex()).isNotNegative();
            assertThat(success.activationValue()).isBetween(0.0, 1.0);
        }
        
        @Test
        @DisplayName("Should distinguish between similar and dissimilar patterns")
        void shouldDistinguishBetweenSimilarAndDissimilarPatterns() {
            var art = createART();
            var params = new ART2Parameters(0.9, 0.5, 100); // High vigilance
            art = new ART2(params);
            
            // Train on two distinct clusters
            var cluster1 = new Pattern[]{
                Pattern.of(new double[]{1.0, 0.1}),
                Pattern.of(new double[]{1.1, 0.0}),
                Pattern.of(new double[]{0.9, 0.2})
            };
            var cluster2 = new Pattern[]{
                Pattern.of(new double[]{0.1, 1.0}),
                Pattern.of(new double[]{0.0, 1.1}),
                Pattern.of(new double[]{0.2, 0.9})
            };
            
            art.fit(cluster1);
            art.fit(cluster2);
            
            // Test predictions
            var testCluster1 = Pattern.of(new double[]{1.05, 0.05});
            var testCluster2 = Pattern.of(new double[]{0.05, 1.05});
            
            var result1 = art.predict(testCluster1);
            var result2 = art.predict(testCluster2);
            
            // Should classify to different categories
            assertThat(result1).isInstanceOf(ActivationResult.Success.class);
            assertThat(result2).isInstanceOf(ActivationResult.Success.class);
            var success1 = (ActivationResult.Success) result1;
            var success2 = (ActivationResult.Success) result2;
            assertThat(success1.categoryIndex()).isNotEqualTo(success2.categoryIndex());
        }
        
        @Test
        @DisplayName("Should handle different learning rates appropriately")
        void shouldHandleDifferentLearningRates() {
            var lowLearningRate = new ART2Parameters(0.7, 0.1, 100); // Slow learning
            var highLearningRate = new ART2Parameters(0.7, 0.9, 100); // Fast learning
            
            var slowArt = new ART2(lowLearningRate);
            var fastArt = new ART2(highLearningRate);
            
            var trainingData = new Pattern[]{
                Pattern.of(new double[]{1.0, 0.0}),
                Pattern.of(new double[]{0.0, 1.0})
            };
            
            slowArt.fit(trainingData);
            fastArt.fit(trainingData);
            
            // Both should be functional but potentially have different behaviors
            assertTrue(slowArt.is_fitted());
            assertTrue(fastArt.is_fitted());
            
            var testInput = Pattern.of(new double[]{0.7, 0.7});
            assertThatCode(() -> {
                slowArt.predict(testInput);
                fastArt.predict(testInput);
            }).doesNotThrowAnyException();
        }
    }
    
    @Nested
    @DisplayName("Scikit-learn Compatibility Tests")
    class ScikitLearnCompatibility {
        
        @Test
        @DisplayName("Should implement ScikitClusterer interface")
        void shouldImplementScikitClustererInterface() {
            var art = createART();
            
            assertThat(art).isInstanceOf(ScikitClusterer.class);
            assertFalse(art.is_fitted());
            
            // Should provide parameter access
            var params = art.get_params();
            assertNotNull(params);
            assertTrue(params.containsKey("vigilance"));
            assertTrue(params.containsKey("learning_rate"));
        }
        
        @Test
        @DisplayName("Should support fit and predict workflow")
        void shouldSupportFitAndPredictWorkflow() {
            var art = createART();
            var X_train = new double[][]{
                {1.0, 2.0}, {1.1, 2.1}, {1.2, 1.9},  // Cluster 1
                {3.0, 1.0}, {3.1, 0.9}, {2.9, 1.1}   // Cluster 2
            };
            
            // Fit
            var fitted = art.fit(X_train);
            assertSame(art, fitted, "fit should return self");
            assertTrue(art.is_fitted());
            
            // Predict
            var X_test = new double[][]{{1.05, 2.05}, {3.05, 0.95}};
            var predictions = art.predict(X_test);
            
            assertEquals(2, predictions.length);
            for (var pred : predictions) {
                assertTrue(pred >= 0, "Predictions should be non-negative");
            }
        }
        
        @Test
        @DisplayName("Should provide cluster centers")
        void shouldProvideClusterCenters() {
            var art = createART();
            var X_train = new double[][]{
                {1.0, 2.0}, {1.1, 2.1},
                {3.0, 1.0}, {3.1, 0.9}
            };
            
            art.fit(X_train);
            
            var centers = art.cluster_centers();
            assertNotNull(centers);
            assertTrue(centers.length > 0);
            
            // Each center should be a valid coordinate
            for (var center : centers) {
                // Cast Pattern to DenseVector to access values()
                if (!(center instanceof DenseVector)) {
                    org.junit.jupiter.api.Assertions.fail("Center should be DenseVector");
                }
                var denseCenter = (DenseVector) center;
                var values = denseCenter.values();
                assertEquals(2, values.length, "Centers should have same dimensionality as input");
                for (var coord : values) {
                    assertTrue(Double.isFinite(coord), "Center coordinates should be finite");
                }
            }
        }
        
        @Test
        @DisplayName("Should provide clustering metrics")
        void shouldProvideClusteringMetrics() {
            var art = createART();
            var X = generateTestData(100, 2);
            
            art.fit(X);
            var labels = art.predict(X);
            var metrics = art.clustering_metrics(X, labels);
            
            // Standard clustering metrics
            assertThat(metrics).containsKeys(
                "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score",
                "inertia", "n_clusters"
            );
            
            // Validate metric ranges
            assertThat((Double) metrics.get("silhouette_score")).isBetween(-1.0, 1.0);
            assertThat(((Number) metrics.get("n_clusters")).intValue()).isGreaterThan(0);
        }
    }
    
    @Nested
    @DisplayName("Performance and Scalability Tests")
    class PerformanceTests {
        
        @Test
        @DisplayName("Should handle large datasets efficiently")
        @Timeout(value = 10, unit = TimeUnit.SECONDS)
        void shouldHandleLargeDatasets() {
            var art = createART();
            var largeDataset = generateTestData(2000, 5); // 2K points, 5D
            
            // Should complete within timeout
            assertThatCode(() -> art.fit(largeDataset)).doesNotThrowAnyException();
            
            // Should maintain reasonable performance for predictions
            var testData = generateTestData(500, 5);
            assertThatCode(() -> art.predict(testData)).doesNotThrowAnyException();
        }
        
        @Test
        @DisplayName("Should maintain performance with increasing dimensionality")
        void shouldMaintainPerformanceWithDimensionality() {
            var dimensions = List.of(2, 5, 10, 20);
            var results = new HashMap<Integer, Long>();
            
            for (var d : dimensions) {
                var art = createART();
                var data = generateTestData(200, d);
                
                var startTime = System.nanoTime();
                art.fit(data);
                var endTime = System.nanoTime();
                
                results.put(d, endTime - startTime);
            }
            
            // Performance should scale reasonably (not exponentially)
            var time2D = results.get(2);
            var time20D = results.get(20);
            var scalingFactor = (double) time20D / time2D;
            
            // Should not exceed O(d^2) scaling
            assertThat(scalingFactor).isLessThan(100.0); // 20^2 / 2^2 = 100
        }
        
        @Test
        @DisplayName("Should handle memory efficiently")
        void shouldHandleMemoryEfficiently() {
            var art = createART();
            var runtime = Runtime.getRuntime();
            
            // Measure memory before
            System.gc();
            var memoryBefore = runtime.totalMemory() - runtime.freeMemory();
            
            // Train on moderately sized dataset
            var data = generateTestData(1000, 10);
            art.fit(data);
            
            // Measure memory after
            System.gc();
            var memoryAfter = runtime.totalMemory() - runtime.freeMemory();
            
            var memoryGrowth = memoryAfter - memoryBefore;
            assertThat(memoryGrowth).isLessThan(50_000_000); // Less than 50MB
        }
    }
    
    @Nested
    @DisplayName("Error Handling and Edge Cases")
    class ErrorHandlingTests {
        
        @ParameterizedTest
        @ValueSource(doubles = {-0.1, 1.1, Double.NaN, Double.POSITIVE_INFINITY})
        @DisplayName("Should reject invalid vigilance parameters")
        void shouldRejectInvalidVigilanceParameters(double invalidVigilance) {
            assertThatThrownBy(() -> {
                new ART2Parameters(invalidVigilance, 0.5, 100);
            }).isInstanceOf(IllegalArgumentException.class)
              .hasMessageContaining("vigilance");
        }
        
        @ParameterizedTest
        @ValueSource(doubles = {-0.1, 1.1, Double.NaN, Double.POSITIVE_INFINITY})
        @DisplayName("Should reject invalid learning rate parameters")
        void shouldRejectInvalidLearningRateParameters(double invalidLearningRate) {
            assertThatThrownBy(() -> {
                new ART2Parameters(0.7, invalidLearningRate, 100);
            }).isInstanceOf(IllegalArgumentException.class)
              .hasMessageContaining("learning");
        }
        
        @Test
        @DisplayName("Should handle empty training data gracefully")
        void shouldHandleEmptyTrainingData() {
            var art = createART();
            var emptyData = new Pattern[0];
            
            assertThatThrownBy(() -> art.fit(emptyData))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("empty");
        }
        
        @Test
        @DisplayName("Should handle prediction before fitting")
        void shouldHandlePredictionBeforeFitting() {
            var art = createART();
            var testInput = Pattern.of(new double[]{0.5, 0.5});
            
            assertThatThrownBy(() -> art.predict(testInput))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("not fitted");
        }
        
        @Test
        @DisplayName("Should handle inconsistent input dimensions")
        void shouldHandleInconsistentInputDimensions() {
            var art = createART();
            var trainingData = new Pattern[]{Pattern.of(new double[]{0.1, 0.2})};
            art.fit(trainingData);
            
            var wrongDimInput = Pattern.of(new double[]{0.1, 0.2, 0.3}); // 3D instead of 2D
            
            assertThatThrownBy(() -> art.predict(wrongDimInput))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("dimension");
        }
        
        @Test
        @DisplayName("Should handle very small input vectors")
        void shouldHandleVerySmallInputVectors() {
            var art = createART();
            var tinyVector = Pattern.of(new double[]{1e-10, 1e-10});
            
            // Should handle gracefully during normalization
            assertThatCode(() -> art.normalizeInput(tinyVector)).doesNotThrowAnyException();
        }
    }
    
    @Nested
    @DisplayName("Integration Tests")
    class IntegrationTests {
        
        @Test
        @DisplayName("Should integrate with BaseART framework")
        void shouldIntegrateWithBaseARTFramework() {
            var art = createART();
            
            // Should implement all required interfaces
            assertThat(art).isInstanceOf(BaseART.class);
            assertThat(art).isInstanceOf(ScikitClusterer.class);
            
            // Should work with BaseART template methods
            var testData = generateTestData(50, 3);
            art.fit(testData);
            
            assertTrue(art.is_fitted());
            assertTrue(art.getCategoryCount() > 0);
        }
        
        @Test
        @DisplayName("Should work with existing test infrastructure")
        void shouldWorkWithExistingTestInfrastructure() {
            var art = createART();
            
            // Should extend BaseARTTest properly
            assertNotNull(art);
            
            // Should work with generateTestData utility
            var data = generateTestData(100, 4);
            assertEquals(100, data.length);
            assertThat(data[0]).hasSize(4);
        }
        
        @Test
        @DisplayName("Should handle concurrent access safely")
        void shouldHandleConcurrentAccessSafely() {
            var art = createART();
            var data = generateTestData(100, 2);
            art.fit(data);
            
            // Test concurrent predictions
            var testInputs = generateTestData(50, 2);
            
            var results = Arrays.stream(testInputs)
                .parallel()
                .mapToInt(input -> {
                    var result = art.predict(Pattern.of(input));
                    return result instanceof ActivationResult.Success success ? success.categoryIndex() : -1;
                })
                .toArray();
            
            // All results should be valid category indices
            for (var result : results) {
                assertTrue(result >= 0, "All predictions should be non-negative");
            }
        }
    }
    
    // Utility methods for test data generation
    private double[][] generateTestData(int numPoints, int dimensions) {
        var random = new Random(42); // Fixed seed for reproducibility
        var data = new double[numPoints][dimensions];
        
        // Create natural clusters
        var numClusters = Math.min(3, Math.max(1, dimensions / 2));
        for (int i = 0; i < numPoints; i++) {
            int cluster = i % numClusters;
            for (int j = 0; j < dimensions; j++) {
                data[i][j] = cluster * 2.0 + random.nextGaussian() * 0.5;
            }
        }
        return data;
    }
}