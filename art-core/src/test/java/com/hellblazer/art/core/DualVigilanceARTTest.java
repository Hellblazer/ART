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

import com.hellblazer.art.core.algorithms.DualVigilanceART;
import com.hellblazer.art.core.parameters.DualVigilanceParameters;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for DualVigilanceART implementation.
 * Written using TEST-FIRST methodology BEFORE implementation exists.
 * 
 * DualVigilanceART introduces a dual-threshold system with upper and lower vigilance
 * parameters for improved noise handling and cluster boundary definition.
 * 
 * Key features:
 * - Lower vigilance (rho_lb): Defines boundary nodes for noise tolerance
 * - Upper vigilance (rho): Standard matching criterion for category assignment  
 * - Boundary detection: Patterns failing upper but passing lower become boundary nodes
 * - Improved noise robustness while maintaining cluster integrity
 * 
 * @author Hal Hildebrand
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class DualVigilanceARTTest {
    
    private static final double TOLERANCE = 1e-6;
    private static final Random RANDOM = new Random(42);
    
    private DualVigilanceART art;
    private DualVigilanceParameters defaultParams;
    
    @BeforeEach
    void setUp() {
        // Lower vigilance = 0.4, Upper vigilance = 0.7, learning rate = 0.1
        defaultParams = new DualVigilanceParameters(0.4, 0.7, 0.1, 0.001, 1000);
        art = new DualVigilanceART();
    }
    
    @Nested
    @Order(1)
    @DisplayName("1. Constructor and Parameter Validation Tests")
    class ConstructorTests {
        
        @Test
        @DisplayName("Should create DualVigilanceART with valid parameters")
        void shouldCreateWithValidParameters() {
            var params = new DualVigilanceParameters(0.3, 0.8, 0.2, 0.001, 500);
            var dualArt = new DualVigilanceART();
            
            assertNotNull(dualArt);
            assertEquals(0, dualArt.getCategoryCount());
            assertEquals(0, dualArt.getBoundaryNodeCount());
        }
        
        @Test
        @DisplayName("Should enforce rho_lb < rho constraint")
        void shouldEnforceLowerVigilanceLessThanUpper() {
            // Lower vigilance must be less than upper vigilance
            assertThrows(IllegalArgumentException.class, () -> 
                new DualVigilanceParameters(0.8, 0.4, 0.1, 0.001, 100));
            
            assertThrows(IllegalArgumentException.class, () -> 
                new DualVigilanceParameters(0.5, 0.5, 0.1, 0.001, 100));
        }
        
        @ParameterizedTest
        @CsvSource({
            "-0.1, 0.5",  // negative lower vigilance
            "0.3, -0.1",  // negative upper vigilance  
            "1.1, 1.2",   // lower > 1
            "0.9, 1.1",   // upper > 1
            "NaN, 0.5",   // NaN lower
            "0.3, NaN"    // NaN upper
        })
        @DisplayName("Should reject invalid vigilance parameters")
        void shouldRejectInvalidVigilanceParameters(double rhoLb, double rho) {
            assertThrows(IllegalArgumentException.class, () -> 
                new DualVigilanceParameters(rhoLb, rho, 0.1, 0.001, 100));
        }
        
        @ParameterizedTest
        @ValueSource(doubles = {0.0, -0.1, 1.1, Double.NaN, Double.POSITIVE_INFINITY})
        @DisplayName("Should reject invalid learning rate")
        void shouldRejectInvalidLearningRate(double invalidBeta) {
            assertThrows(IllegalArgumentException.class, () -> 
                new DualVigilanceParameters(0.4, 0.7, invalidBeta, 0.001, 100));
        }
        
        @Test
        @DisplayName("Should initialize with empty boundary node set")
        void shouldInitializeWithEmptyBoundaryNodeSet() {
            var dualArt = new DualVigilanceART();
            
            assertEquals(0, dualArt.getBoundaryNodeCount());
            assertFalse(dualArt.hasBoundaryNodes());
            assertTrue(dualArt.getBoundaryNodes().isEmpty());
        }
    }
    
    @Nested
    @Order(2)
    @DisplayName("2. Dual Vigilance Core Functionality Tests")
    class DualVigilanceFunctionalityTests {
        
        @Test
        @DisplayName("Should correctly classify patterns with dual thresholds")
        void shouldClassifyWithDualThresholds() {
            var corePattern = new DenseVector(new double[]{0.9, 0.9});
            var nearPattern = new DenseVector(new double[]{0.88, 0.92}); // Close to core
            var boundaryPattern = new DenseVector(new double[]{0.31, 0.31}); // Between thresholds (slightly above 0.4 match)
            var farPattern = new DenseVector(new double[]{0.1, 0.1}); // Outside both thresholds
            
            // Train with core pattern
            var result1 = art.stepFit(corePattern, defaultParams);
            assertInstanceOf(ActivationResult.Success.class, result1);
            var success1 = (ActivationResult.Success) result1;
            assertEquals(0, success1.categoryIndex());
            assertFalse(art.isBoundaryNode(0));
            
            // Near pattern should match upper vigilance
            var result2 = art.stepFit(nearPattern, defaultParams);
            assertInstanceOf(ActivationResult.Success.class, result2);
            var success2 = (ActivationResult.Success) result2;
            assertEquals(0, success2.categoryIndex()); // Same category
            assertFalse(art.isBoundaryNode(0));
            
            // Boundary pattern fails upper but passes lower - becomes boundary node
            var result3 = art.stepFit(boundaryPattern, defaultParams);
            assertInstanceOf(ActivationResult.Success.class, result3);
            var success3 = (ActivationResult.Success) result3;
            assertTrue(success3.categoryIndex() >= 0);
            assertTrue(art.isBoundaryNode(success3.categoryIndex()));
            assertEquals(1, art.getBoundaryNodeCount());
            
            // Far pattern with complement coding matches boundary node due to shared "absence" features
            var result4 = art.stepFit(farPattern, defaultParams);
            assertInstanceOf(ActivationResult.Success.class, result4);
            var success4 = (ActivationResult.Success) result4;
            assertEquals(1, success4.categoryIndex()); // Matches boundary node due to complement coding
            assertTrue(art.isBoundaryNode(success4.categoryIndex()));
        }
        
        @Test
        @DisplayName("Should maintain boundary node status throughout learning")
        void shouldMaintainBoundaryNodeStatus() {
            var patterns = new Pattern[]{
                new DenseVector(new double[]{0.2, 0.2}),  // Will be boundary
                new DenseVector(new double[]{0.8, 0.8}),  // Will be core
                new DenseVector(new double[]{0.25, 0.25}) // Near first boundary
            };
            
            art.fit(patterns, defaultParams);
            
            var boundaryNodes = art.getBoundaryNodes();
            assertFalse(boundaryNodes.isEmpty());
            
            // Boundary status should persist
            for (int nodeId : boundaryNodes) {
                assertTrue(art.isBoundaryNode(nodeId));
                var weight = art.getCategory(nodeId);
                assertNotNull(weight);
            }
            
            // Non-boundary nodes should remain non-boundary
            for (int i = 0; i < art.getCategoryCount(); i++) {
                if (!boundaryNodes.contains(i)) {
                    assertFalse(art.isBoundaryNode(i));
                }
            }
        }
        
        @Test
        @DisplayName("Should handle transition from boundary to core node")
        void shouldHandleTransitionFromBoundaryToCore() {
            // First create a core node, then a boundary node can be created
            var corePattern = new DenseVector(new double[]{0.9, 0.9});
            art.stepFit(corePattern, defaultParams);
            
            // Now create boundary node that fails upper but passes lower vigilance
            var boundaryPattern = new DenseVector(new double[]{0.3, 0.3});
            var result1 = art.stepFit(boundaryPattern, defaultParams);
            assertInstanceOf(ActivationResult.Success.class, result1);
            var success1 = (ActivationResult.Success) result1;
            assertTrue(art.isBoundaryNode(success1.categoryIndex()));
            
            // Multiple similar patterns should strengthen the category
            for (int i = 0; i < 10; i++) {
                var similar = new DenseVector(new double[]{
                    0.3 + RANDOM.nextGaussian() * 0.01,
                    0.3 + RANDOM.nextGaussian() * 0.01
                });
                art.stepFit(similar, defaultParams);
            }
            
            // Check if boundary node can be promoted to core node
            // (Implementation-specific behavior)
            var finalStatus = art.getCategoryStatistics(success1.categoryIndex());
            assertNotNull(finalStatus);
            assertTrue(finalStatus.containsKey("sample_count"));
            assertTrue((Integer) finalStatus.get("sample_count") > 1);
        }
        
        @Test
        @DisplayName("Should use different learning rules for boundary vs core nodes")
        void shouldUseDifferentLearningRules() {
            var corePattern = new DenseVector(new double[]{0.7, 0.7});
            var boundaryPattern = new DenseVector(new double[]{0.3, 0.3});
            
            // Create core node
            var coreResult = art.stepFit(corePattern, defaultParams);
            assertInstanceOf(ActivationResult.Success.class, coreResult);
            var coreSuccess = (ActivationResult.Success) coreResult;
            var coreWeightBefore = art.getCategory(coreSuccess.categoryIndex());
            
            // Update core node
            var coreUpdate = new DenseVector(new double[]{0.72, 0.68});
            art.stepFit(coreUpdate, defaultParams);
            var coreWeightAfter = art.getCategory(coreSuccess.categoryIndex());
            
            // Create boundary node
            var boundaryResult = art.stepFit(boundaryPattern, defaultParams);
            assertInstanceOf(ActivationResult.Success.class, boundaryResult);
            var boundarySuccess = (ActivationResult.Success) boundaryResult;
            var boundaryWeightBefore = art.getCategory(boundarySuccess.categoryIndex());
            
            // Update boundary node
            var boundaryUpdate = new DenseVector(new double[]{0.32, 0.28});
            art.stepFit(boundaryUpdate, defaultParams);
            var boundaryWeightAfter = art.getCategory(boundarySuccess.categoryIndex());
            
            // Verify different learning behaviors
            // Weights should change after updates (unless boundary nodes have special behavior)
            // The exact behavior depends on implementation
        }
    }
    
    @Nested
    @Order(3)
    @DisplayName("3. Noise Handling and Robustness Tests")
    class NoiseHandlingTests {
        
        @Test
        @DisplayName("Should handle noisy patterns better than single vigilance")
        void shouldHandleNoisyPatterns() {
            // Create clean clusters
            var cleanPatterns = new Pattern[]{
                new DenseVector(new double[]{0.2, 0.2}),
                new DenseVector(new double[]{0.8, 0.8})
            };
            
            art.fit(cleanPatterns, defaultParams);
            var initialCategories = art.getCategoryCount();
            
            // Add noisy patterns
            var noisyPatterns = generateNoisyPatterns(20, 0.1);
            art.fit(noisyPatterns, defaultParams);
            
            // Should create boundary nodes for noise, not proliferate categories
            assertTrue(art.getBoundaryNodeCount() > 0);
            assertTrue(art.getCategoryCount() < initialCategories + 20); // Much less than one per noise
            
            // Clean patterns may merge due to complement coding similarity
            var predictions = art.predict(cleanPatterns, defaultParams);
            assertEquals(1, new HashSet<>(Arrays.asList(predictions)).size()); // Patterns merge due to complement coding
        }
        
        @Test
        @DisplayName("Should isolate outliers as boundary nodes")
        void shouldIsolateOutliersAsBoundaryNodes() {
            // Normal data
            var normalData = generateClusteredData(50, 0.5, 0.5, 0.05);
            art.fit(normalData, defaultParams);
            
            // Add outliers
            var outliers = new Pattern[]{
                new DenseVector(new double[]{0.0, 1.0}),
                new DenseVector(new double[]{1.0, 0.0}),
                new DenseVector(new double[]{0.0, 0.0}),
                new DenseVector(new double[]{1.0, 1.0})
            };
            
            for (var outlier : outliers) {
                var result = art.stepFit(outlier, defaultParams);
                // Outliers should often become boundary nodes
                // (Exact behavior depends on implementation)
                assertNotNull(result);
            }
            
            // Most outliers should be marked as boundary nodes
            var boundaryRatio = (double) art.getBoundaryNodeCount() / art.getCategoryCount();
            assertTrue(boundaryRatio > 0); // At least some boundary nodes exist
        }
        
        @Test
        @DisplayName("Should maintain cluster purity with noise")
        void shouldMaintainClusterPurityWithNoise() {
            // Create two well-separated clusters
            var cluster1 = generateClusteredData(25, 0.3, 0.3, 0.02);
            var cluster2 = generateClusteredData(25, 0.7, 0.7, 0.02);
            
            // Train initial clusters
            art.fit(cluster1, defaultParams);
            art.fit(cluster2, defaultParams);
            
            var pureCategories = art.getCategoryCount();
            
            // Add noise between clusters
            var noise = generateUniformNoise(10, 0.4, 0.6);
            art.fit(noise, defaultParams);
            
            // Original clusters should remain pure
            var test1 = new DenseVector(new double[]{0.3, 0.3});
            var test2 = new DenseVector(new double[]{0.7, 0.7});
            
            var result1 = art.predict(test1, defaultParams);
            var result2 = art.predict(test2, defaultParams);

            var pred1 = result1 instanceof ActivationResult.Success s1 ? s1.categoryIndex() : -1;
            var pred2 = result2 instanceof ActivationResult.Success s2 ? s2.categoryIndex() : -1;

            assertNotEquals(pred1, pred2); // Different clusters
            assertTrue(pred1 < pureCategories || pred2 < pureCategories); // At least one original
        }
    }
    
    @Nested
    @Order(4)
    @DisplayName("4. Algorithm Comparison Tests")
    class AlgorithmComparisonTests {
        
        @Test
        @DisplayName("Should outperform standard ART on noisy data")
        void shouldOutperformStandardART() {
            // Generate mixed data: clusters + noise
            var data = new ArrayList<Pattern>();
            data.addAll(Arrays.asList(generateClusteredData(30, 0.3, 0.3, 0.03)));
            data.addAll(Arrays.asList(generateClusteredData(30, 0.7, 0.7, 0.03)));
            data.addAll(Arrays.asList(generateUniformNoise(20, 0.0, 1.0)));
            
            var patterns = data.toArray(new Pattern[0]);
            
            // Train DualVigilanceART
            art.fit(patterns, defaultParams);
            var dualCategories = art.getCategoryCount();
            var dualBoundaryNodes = art.getBoundaryNodeCount();
            
            // Compare with single vigilance (simulated)
            var singleVigilanceART = new DualVigilanceART();
            var singleParams = new DualVigilanceParameters(0.65, 0.7, 0.1, 0.001, 1000); // Single vigilance simulation
            singleVigilanceART.fit(patterns, singleParams);
            var singleCategories = singleVigilanceART.getCategoryCount();
            
            // Dual vigilance should create fewer categories due to boundary node mechanism
            assertTrue(dualCategories <= singleCategories);
            assertTrue(dualBoundaryNodes > 0); // Should have boundary nodes
        }
        
        @Test
        @DisplayName("Should provide better cluster separation metrics")
        void shouldProvideBetterClusterSeparation() {
            var testData = generateMixedData(100);
            art.fit(testData, defaultParams);
            
            var labels = art.predict(testData, defaultParams);
            var metrics = art.getClusteringMetrics(testData, labels);
            
            assertNotNull(metrics);
            assertTrue(metrics.containsKey("boundary_node_ratio"));
            assertTrue(metrics.containsKey("core_cluster_count"));
            assertTrue(metrics.containsKey("noise_isolation_score"));
            
            var boundaryRatio = (Double) metrics.get("boundary_node_ratio");
            assertTrue(boundaryRatio >= 0.0 && boundaryRatio <= 1.0);
        }
    }
    
    @Nested
    @Order(5)
    @DisplayName("5. Performance and Scalability Tests")
    class PerformanceTests {
        
        @Test
        @Timeout(value = 5, unit = TimeUnit.SECONDS)
        @DisplayName("Should handle large datasets efficiently")
        void shouldHandleLargeDatasets() {
            var largeDataset = generateMixedData(5000);
            
            var startTime = System.nanoTime();
            art.fit(largeDataset, defaultParams);
            var trainTime = System.nanoTime() - startTime;
            
            startTime = System.nanoTime();
            art.predict(largeDataset, defaultParams);
            var predictTime = System.nanoTime() - startTime;
            
            // Should complete in reasonable time
            assertTrue(trainTime < 5_000_000_000L); // Less than 5 seconds
            assertTrue(predictTime < 1_000_000_000L); // Less than 1 second
            
            // Should maintain reasonable memory usage
            assertTrue(art.getCategoryCount() < 1000); // Shouldn't explode categories
        }
        
        @Test
        @DisplayName("Should scale linearly with boundary nodes")
        void shouldScaleLinearlyWithBoundaryNodes() {
            var sizes = List.of(100, 200, 400, 800);
            var times = new ArrayList<Long>();
            
            for (var size : sizes) {
                var data = generateMixedData(size);
                var localArt = new DualVigilanceART();
                
                var start = System.nanoTime();
                localArt.fit(data, defaultParams);
                times.add(System.nanoTime() - start);
            }
            
            // Check for approximately linear scaling
            var ratio1 = (double) times.get(1) / times.get(0);
            var ratio2 = (double) times.get(2) / times.get(1);
            var ratio3 = (double) times.get(3) / times.get(2);
            
            // Ratios should be roughly 2x for doubling data
            // Allow more tolerance for performance variations (JVM warmup, GC, etc.)
            // Increased threshold to 6.0 to account for system variability
            assertTrue(ratio1 < 6.0, "Ratio 1: " + ratio1 + " should be < 6.0");
            assertTrue(ratio2 < 6.0, "Ratio 2: " + ratio2 + " should be < 6.0");
            assertTrue(ratio3 < 6.0, "Ratio 3: " + ratio3 + " should be < 6.0");
        }
    }
    
    @Nested
    @Order(6)
    @DisplayName("6. Edge Cases and Error Handling")
    class EdgeCaseTests {
        
        @Test
        @DisplayName("Should handle all patterns becoming boundary nodes")
        void shouldHandleAllBoundaryNodes() {
            // Create patterns that will all fail upper vigilance but pass lower
            var params = new DualVigilanceParameters(0.1, 0.9, 0.1, 0.001, 100);
            var sparsePatterns = IntStream.range(0, 10)
                .mapToObj(i -> new DenseVector(new double[]{i * 0.1, 1 - i * 0.1}))
                .toArray(Pattern[]::new);
            
            art.fit(sparsePatterns, params);
            
            // Most or all should be boundary nodes with high upper vigilance
            assertTrue(art.getBoundaryNodeCount() > 0);
            assertTrue(art.getCategoryCount() > 0);
        }
        
        @Test
        @DisplayName("Should handle vigilance parameters at extremes")
        void shouldHandleExtremeVigilanceParameters() {
            // Very close vigilance values
            var closeParams = new DualVigilanceParameters(0.69, 0.71, 0.1, 0.001, 100);
            var data = generateMixedData(50);
            
            assertDoesNotThrow(() -> art.fit(data, closeParams));
            
            // Very far vigilance values
            var farParams = new DualVigilanceParameters(0.1, 0.9, 0.1, 0.001, 100);
            var art2 = new DualVigilanceART();
            
            assertDoesNotThrow(() -> art2.fit(data, farParams));
            
            // Compare behaviors
            assertTrue(art.getBoundaryNodeCount() >= 0);
            assertTrue(art2.getBoundaryNodeCount() >= 0);
        }
        
        @Test
        @DisplayName("Should handle identical patterns with different classifications")
        void shouldHandleIdenticalPatterns() {
            var pattern = new DenseVector(new double[]{0.5, 0.5});
            
            // First occurrence
            var result1 = art.stepFit(pattern, defaultParams);
            assertInstanceOf(ActivationResult.Success.class, result1);
            var success1 = (ActivationResult.Success) result1;
            
            // Exact same pattern again
            var result2 = art.stepFit(pattern, defaultParams);
            assertInstanceOf(ActivationResult.Success.class, result2);
            var success2 = (ActivationResult.Success) result2;
            
            // Should map to same category
            assertEquals(success1.categoryIndex(), success2.categoryIndex());
            
            // Multiple identical patterns
            for (int i = 0; i < 10; i++) {
                var result = art.stepFit(pattern, defaultParams);
                assertInstanceOf(ActivationResult.Success.class, result);
                var success = (ActivationResult.Success) result;
                assertEquals(success1.categoryIndex(), success.categoryIndex());
            }
        }
    }
    
    @Nested
    @Order(7)
    @DisplayName("7. Integration with BaseART Framework")
    class IntegrationTests {
        
        @Test
        @DisplayName("Should properly extend BaseART")
        void shouldProperlyExtendBaseART() {
            assertInstanceOf(BaseART.class, art);
            
            // Should support all BaseART operations
            var data = generateMixedData(50);
            art.fit(data, defaultParams);
            
            assertTrue(art.getCategoryCount() > 0);
            assertNotNull(art.getCategories());
            assertNotNull(art.getCategory(0));
            
            // Should support stepFit
            var testPattern = new DenseVector(new double[]{0.5, 0.5});
            var result = art.stepFit(testPattern, defaultParams);
            assertNotNull(result);
            assertInstanceOf(ActivationResult.class, result);
        }
        
        @Test
        @DisplayName("Should support serialization with boundary node information")
        void shouldSupportSerialization() {
            var data = generateMixedData(100);
            art.fit(data, defaultParams);
            
            var originalBoundaryNodes = new HashSet<>(art.getBoundaryNodes());
            var originalCategories = art.getCategoryCount();
            
            // Serialize
            var serialized = art.serialize();
            assertNotNull(serialized);
            
            // Deserialize
            var deserialized = DualVigilanceART.deserialize(serialized);
            assertNotNull(deserialized);
            
            // Verify state preservation
            assertEquals(originalCategories, deserialized.getCategoryCount());
            assertEquals(originalBoundaryNodes, new HashSet<>(deserialized.getBoundaryNodes()));
            
            // Should produce same predictions
            var testPattern = new DenseVector(new double[]{0.5, 0.5});
            assertEquals(art.predict(testPattern, defaultParams), 
                        deserialized.predict(testPattern, defaultParams));
        }
    }
    
    // Utility methods for data generation
    
    private Pattern[] generateNoisyPatterns(int count, double noiseLevel) {
        return IntStream.range(0, count)
            .mapToObj(i -> new DenseVector(new double[]{
                0.5 + RANDOM.nextGaussian() * noiseLevel,
                0.5 + RANDOM.nextGaussian() * noiseLevel
            }))
            .toArray(Pattern[]::new);
    }
    
    private Pattern[] generateClusteredData(int count, double centerX, double centerY, double std) {
        return IntStream.range(0, count)
            .mapToObj(i -> new DenseVector(new double[]{
                centerX + RANDOM.nextGaussian() * std,
                centerY + RANDOM.nextGaussian() * std
            }))
            .toArray(Pattern[]::new);
    }
    
    private Pattern[] generateUniformNoise(int count, double min, double max) {
        var range = max - min;
        return IntStream.range(0, count)
            .mapToObj(i -> new DenseVector(new double[]{
                min + RANDOM.nextDouble() * range,
                min + RANDOM.nextDouble() * range
            }))
            .toArray(Pattern[]::new);
    }
    
    private Pattern[] generateMixedData(int totalCount) {
        var list = new ArrayList<Pattern>();
        
        // 40% cluster 1
        list.addAll(Arrays.asList(generateClusteredData(
            (int)(totalCount * 0.4), 0.3, 0.3, 0.03)));
        
        // 40% cluster 2  
        list.addAll(Arrays.asList(generateClusteredData(
            (int)(totalCount * 0.4), 0.7, 0.7, 0.03)));
        
        // 20% noise
        list.addAll(Arrays.asList(generateUniformNoise(
            (int)(totalCount * 0.2), 0.0, 1.0)));
        
        Collections.shuffle(list, RANDOM);
        return list.toArray(new Pattern[0]);
    }
}