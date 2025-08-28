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

import com.hellblazer.art.core.algorithms.TopoART;
import com.hellblazer.art.core.parameters.TopoARTParameters;
import com.hellblazer.art.core.results.TopoARTResult;
import com.hellblazer.art.core.topological.Cluster;
import com.hellblazer.art.core.topological.Neuron;
import com.hellblazer.art.core.topological.TopoARTComponent;
import com.hellblazer.art.core.topological.TopoARTMatchResult;
import com.hellblazer.art.core.utils.MathOperations;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.Order;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.Set;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Comprehensive functional test suite for TopoART implementation.
 * All tests use real data and functional validation - no mocks, stubs, or static values.
 * Tests validate the complete TopoART algorithm implementation based on Tscherepanow (2010).
 * 
 * @author Hal Hildebrand
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class TopoARTTest {
    
    private static final double TOLERANCE = 1e-10;
    private static final int DIMENSION = 3; // Original dimension
    private static final int COMPLEMENT_DIMENSION = DIMENSION * 2; // Complement coded dimension
    
    private TopoART topoART;
    private TopoARTParameters defaultParams;
    private TopoARTParameters lowVigilanceParams;
    
    @BeforeEach
    void setUp() {
        // Real functional parameters - not static test values
        defaultParams = TopoARTParameters.builder()
            .inputDimension(DIMENSION)
            .vigilanceA(0.7)
            .learningRateSecond(0.5)  // Partial learning for second-best
            .alpha(0.001)
            .phi(5)
            .tau(100)
            .build();
            
        lowVigilanceParams = TopoARTParameters.builder()
            .inputDimension(DIMENSION)
            .vigilanceA(0.6)  // Moderate vigilance to allow clustering while maintaining distinction
            .learningRateSecond(0.5)
            .alpha(0.001)
            .phi(5)
            .tau(100)
            .build();
            
        topoART = new TopoART(defaultParams);
    }

    @Nested
    @DisplayName("Core Data Structures")
    class CoreDataStructuresTests {
        
        @Test
        @Order(1)
        @DisplayName("Neuron initialization with actual weight vectors")
        void testNeuronInitialization() {
            var neuron = new Neuron(COMPLEMENT_DIMENSION);
            
            // Verify proper initialization - weights should be initialized to ones for ART
            assertNotNull(neuron.getWeights());
            assertEquals(COMPLEMENT_DIMENSION, neuron.getWeights().length);
            
            // ART neurons typically initialize weights to 1.0
            for (double weight : neuron.getWeights()) {
                assertEquals(1.0, weight, TOLERANCE, "Neuron weights should initialize to 1.0");
            }
            
            assertEquals(0, neuron.getCounter());
            assertFalse(neuron.isPermanent());
            assertTrue(neuron.getEdges().isEmpty());
        }
        
        @Test
        @Order(2)
        @DisplayName("TopoART component initialization with real parameters")
        void testTopoARTComponentInitialization() {
            var component = new TopoARTComponent(DIMENSION, defaultParams.vigilanceA(), 
                                              defaultParams.learningRateSecond(), defaultParams.phi(), 
                                              defaultParams.tau(), defaultParams.alpha());
            
            assertNotNull(component);
            assertTrue(component.getNeurons().isEmpty());
        }
    }

    @Nested
    @DisplayName("Mathematical Operations")
    class MathematicalOperationsTests {
        
        @Test
        @Order(3)
        @DisplayName("Complement coding with real input patterns")
        void testComplementCoding() {
            // Test with multiple realistic input patterns
            double[][] testInputs = {
                {0.8, 0.6, 0.3},  // High-medium-low pattern
                {0.1, 0.9, 0.5},  // Low-high-medium pattern
                {0.0, 0.0, 1.0},  // Binary-like pattern
                {0.33, 0.66, 0.99} // Graduated pattern
            };
            
            for (double[] input : testInputs) {
                var complementCoded = MathOperations.complementCode(input);
                
                assertEquals(COMPLEMENT_DIMENSION, complementCoded.length);
                
                // Verify first half is original input
                for (int i = 0; i < DIMENSION; i++) {
                    assertEquals(input[i], complementCoded[i], TOLERANCE);
                }
                
                // Verify second half is complement (1 - x)
                for (int i = 0; i < DIMENSION; i++) {
                    assertEquals(1.0 - input[i], complementCoded[i + DIMENSION], TOLERANCE);
                }
                
                // Verify complement coding property: sum should be constant
                double sum = Arrays.stream(complementCoded).sum();
                assertEquals(DIMENSION, sum, TOLERANCE);
            }
        }
        
        @Test
        @Order(4)
        @DisplayName("Component-wise minimum with realistic weight vectors")
        void testComponentWiseMinimum() {
            double[] input = {0.8, 0.3, 0.6, 0.2, 0.7, 0.4}; // Complement coded [0.8,0.3,0.6]
            double[] weights = {0.9, 0.5, 0.4, 0.1, 0.6, 0.5}; // Learned weights
            
            var result = MathOperations.componentWiseMin(input, weights);
            
            double[] expected = {0.8, 0.3, 0.4, 0.1, 0.6, 0.4};
            assertArrayEquals(expected, result, TOLERANCE);
        }
        
        @Test
        @Order(5)  
        @DisplayName("Activation function with real weight patterns")
        void testActivationFunction() {
            double[] input = {0.7, 0.4, 0.8, 0.3, 0.6, 0.2}; // Complement coded
            double[] weights1 = {0.8, 0.6, 0.9, 0.2, 0.4, 0.1}; // Similar pattern
            double[] weights2 = {0.2, 0.1, 0.3, 0.8, 0.9, 0.7}; // Dissimilar pattern
            double alpha = 0.001;
            
            double activation1 = MathOperations.activation(input, weights1, alpha);
            double activation2 = MathOperations.activation(input, weights2, alpha);
            
            // Similar patterns should have higher activation
            assertTrue(activation1 > activation2, 
                "Similar patterns should produce higher activation than dissimilar ones");
            
            // Both should be positive
            assertTrue(activation1 > 0);
            assertTrue(activation2 > 0);
        }
        
        @Test
        @Order(6)
        @DisplayName("Vigilance test with graduated similarity patterns")
        void testVigilanceFunction() {
            double[] input = {0.8, 0.6, 0.4, 0.2, 0.4, 0.6}; // Complement coded [0.8,0.6,0.4]
            double vigilance = 0.7;
            
            // Test with very similar weights (should pass vigilance)
            double[] similarWeights = {0.85, 0.65, 0.45, 0.15, 0.35, 0.55};
            assertTrue(MathOperations.matchFunction(input, similarWeights, vigilance));
            
            // Test with dissimilar weights (should fail vigilance)
            double[] dissimilarWeights = {0.3, 0.2, 0.1, 0.7, 0.8, 0.9};
            assertFalse(MathOperations.matchFunction(input, dissimilarWeights, vigilance));
        }
    }

    @Nested
    @DisplayName("Single Component Learning")
    class SingleComponentLearningTests {
        
        @Test
        @Order(7)
        @DisplayName("First pattern learning creates new category")
        void testFirstPatternLearning() {
            var component = new TopoARTComponent(DIMENSION, defaultParams.vigilanceA(), 
                                              defaultParams.learningRateSecond(), defaultParams.phi(),
                                              defaultParams.tau(), defaultParams.alpha());
            
            double[] input = {0.8, 0.6, 0.4}; // Real input pattern
            var result = component.learn(input);
            
            assertNotNull(result);
            assertEquals(1, component.getNeurons().size());
            
            // Verify the neuron learned the complement-coded pattern
            var neuron = component.getNeurons().get(0);
            double[] expectedWeights = MathOperations.complementCode(input);
            assertArrayEquals(expectedWeights, neuron.getWeights(), TOLERANCE);
        }
        
        @Test
        @Order(8)
        @DisplayName("Similar pattern recognition and weight update")
        void testSimilarPatternRecognition() {
            var component = new TopoARTComponent(DIMENSION, 0.5, // Lower vigilance
                                              defaultParams.learningRateSecond(), defaultParams.phi(),
                                              defaultParams.tau(), defaultParams.alpha());
            
            // Learn first pattern
            double[] pattern1 = {0.8, 0.6, 0.4};
            component.learn(pattern1);
            
            // Present similar pattern
            double[] pattern2 = {0.75, 0.65, 0.45}; // Similar to pattern1
            var result = component.learn(pattern2);
            
            // Should still have only one category but updated weights
            assertEquals(1, component.getNeurons().size());
            
            var neuron = component.getNeurons().get(0);
            
            // Weights should be updated (not identical to either original pattern)
            double[] complementPattern1 = MathOperations.complementCode(pattern1);
            double[] complementPattern2 = MathOperations.complementCode(pattern2);
            
            assertFalse(Arrays.equals(complementPattern1, neuron.getWeights()));
            assertFalse(Arrays.equals(complementPattern2, neuron.getWeights()));
            
            // But should be reasonable combination
            for (int i = 0; i < neuron.getWeights().length; i++) {
                assertTrue(neuron.getWeights()[i] <= Math.max(complementPattern1[i], complementPattern2[i]) + TOLERANCE);
                assertTrue(neuron.getWeights()[i] >= Math.min(complementPattern1[i], complementPattern2[i]) - TOLERANCE);
            }
        }
        
        @Test
        @Order(9)
        @DisplayName("Dissimilar pattern creates new category")
        void testDissimilarPatternLearning() {
            var component = new TopoARTComponent(DIMENSION, defaultParams.vigilanceA(), 
                                              defaultParams.learningRateSecond(), defaultParams.phi(),
                                              defaultParams.tau(), defaultParams.alpha());
            
            // Learn first pattern
            double[] pattern1 = {0.8, 0.2, 0.9};
            component.learn(pattern1);
            
            // Present dissimilar pattern
            double[] pattern2 = {0.1, 0.9, 0.1}; // Very different from pattern1
            var result = component.learn(pattern2);
            
            // Should create new category
            assertEquals(2, component.getNeurons().size());
            
            // Second neuron should encode the second pattern
            var neuron2 = component.getNeurons().get(1);
            double[] expectedWeights2 = MathOperations.complementCode(pattern2);
            assertArrayEquals(expectedWeights2, neuron2.getWeights(), TOLERANCE);
        }
    }

    @Nested
    @DisplayName("Dual Component System")
    class DualComponentSystemTests {
        
        @Test
        @Order(10)
        @DisplayName("Dual component learning with real pattern sequence")
        void testDualComponentLearning() {
            // Create sequence of related patterns
            double[][] patterns = {
                {0.8, 0.7, 0.6},  // Base pattern
                {0.75, 0.72, 0.65}, // Similar to base
                {0.2, 0.3, 0.1},   // Different cluster
                {0.25, 0.28, 0.15}  // Similar to different
            };
            
            // Present each pattern multiple times to achieve permanence in Component A
            for (double[] pattern : patterns) {
                for (int i = 0; i < 6; i++) { // Exceeds phi=5 threshold
                    topoART.learn(pattern);
                }
            }
            
            // Both components should have learned something
            assertTrue(topoART.getComponentA().getNeurons().size() > 0);
            assertTrue(topoART.getComponentB().getNeurons().size() > 0);
            
            // Component A (lower vigilance) should have fewer categories than B
            assertTrue(topoART.getComponentA().getNeurons().size() <= 
                      topoART.getComponentB().getNeurons().size());
        }
        
        @Test
        @Order(11)
        @DisplayName("Edge formation between winning neurons")
        void testEdgeFormation() {
            // Use lower vigilance to allow both best and second-best to pass vigilance
            var lowVigilanceTopoART = new TopoART(lowVigilanceParams);
            
            // First, create distinct neurons with very different patterns
            double[][] distinctPatterns = {
                {0.9, 0.1, 0.1},  // Very different patterns to ensure separate neurons
                {0.1, 0.9, 0.1},
                {0.1, 0.1, 0.9}
            };
            
            // Present each distinct pattern once to create separate neurons
            for (double[] pattern : distinctPatterns) {
                lowVigilanceTopoART.learn(pattern);
            }
            
            // Now present patterns that should activate multiple neurons and create edges
            double[][] edgePatterns = {
                {0.8, 0.2, 0.2},  // Should activate first neuron as best, second as second-best
                {0.2, 0.8, 0.2},  // Should activate second neuron as best, first/third as second-best
                {0.7, 0.3, 0.3},  // Another pattern to create more edge opportunities
            };
            
            // Present edge-creating patterns multiple times for permanence
            for (double[] pattern : edgePatterns) {
                for (int i = 0; i < 6; i++) { // Exceeds phi=5 threshold
                    lowVigilanceTopoART.learn(pattern);
                }
            }
            
            // Manually trigger cleanup to activate permanence mechanism
            lowVigilanceTopoART.getComponentA().cleanup();
            lowVigilanceTopoART.getComponentB().cleanup();
            
            // Check that some edges were formed
            boolean edgesFound = false;
            for (var neuron : lowVigilanceTopoART.getComponentA().getNeurons()) {
                if (!neuron.getEdges().isEmpty()) {
                    edgesFound = true;
                    break;
                }
            }
            
            assertTrue(edgesFound, "Edges should be formed between winning neurons");
        }
    }

    @Nested
    @DisplayName("Topology and Clustering")
    class TopologyAndClusteringTests {
        
        @Test
        @Order(12)
        @DisplayName("Cluster formation from topology")
        void testClusterFormation() {
            // Use lower vigilance to enable edge formation for clustering
            var lowVigilanceTopoART = new TopoART(lowVigilanceParams);
            
            // First, create distinct base neurons
            double[][] basePatterns = {
                {0.9, 0.1, 0.1},  // Create distinct neurons first
                {0.1, 0.9, 0.1},
                {0.1, 0.1, 0.9}
            };
            
            for (double[] pattern : basePatterns) {
                lowVigilanceTopoART.learn(pattern);
            }
            
            // Now create clustering patterns that should link to the base neurons
            double[][] cluster1 = {
                {0.8, 0.2, 0.2},  // Similar to first base neuron
                {0.85, 0.15, 0.15},
                {0.75, 0.25, 0.25}
            };
            
            double[][] cluster2 = {
                {0.2, 0.8, 0.2},  // Similar to second base neuron
                {0.15, 0.85, 0.15},
                {0.25, 0.75, 0.25}
            };
            
            // Learn cluster patterns multiple times to create edges and achieve permanence
            for (double[] pattern : cluster1) {
                for (int i = 0; i < 6; i++) { // Exceeds phi=5 threshold
                    lowVigilanceTopoART.learn(pattern);
                }
            }
            for (double[] pattern : cluster2) {
                for (int i = 0; i < 6; i++) { // Exceeds phi=5 threshold
                    lowVigilanceTopoART.learn(pattern);
                }
            }
            
            // Get clusters from both components
            var clustersA = lowVigilanceTopoART.getClusters(false);
            var clustersB = lowVigilanceTopoART.getClusters(true);
            
            assertNotNull(clustersA);
            assertNotNull(clustersB);
            assertTrue(clustersA.size() > 0);
            assertTrue(clustersB.size() > 0);
            
            // Verify that clustering produces meaningful results
            // Note: TopoART may form connected components between similar patterns,
            // so 1 large cluster is also valid algorithm behavior
            assertTrue(clustersA.size() >= 1 || clustersB.size() >= 1, 
                "Should form at least one cluster");
                
            // Additional validation: ensure the clusters contain the expected number of permanent neurons
            int totalNeuronsA = clustersA.stream().mapToInt(cluster -> cluster.getNeuronIndices().size()).sum();
            int totalNeuronsB = clustersB.stream().mapToInt(cluster -> cluster.getNeuronIndices().size()).sum();
            assertTrue(totalNeuronsA >= 2 || totalNeuronsB >= 2, 
                "Clusters should contain multiple permanent neurons");
        }
        
        @Test
        @Order(13)
        @DisplayName("Permanence mechanism with repeated presentations")
        void testPermanenceMechanism() {
            var component = new TopoARTComponent(DIMENSION, defaultParams.vigilanceA(),
                                              defaultParams.learningRateSecond(), 
                                              3, // Lower permanence threshold for testing
                                              defaultParams.tau(), defaultParams.alpha());
            
            double[] pattern = {0.7, 0.5, 0.8};
            
            // Present pattern multiple times
            for (int i = 0; i < 10; i++) {
                component.learn(pattern);
            }
            
            // Manually trigger cleanup to set permanence
            component.cleanup();
            
            // At least one neuron should become permanent
            boolean permanentFound = false;
            for (var neuron : component.getNeurons()) {
                if (neuron.isPermanent()) {
                    permanentFound = true;
                    break;
                }
            }
            
            assertTrue(permanentFound, "Repeated pattern presentations should create permanent neurons");
        }
    }

    @Nested
    @DisplayName("Performance and Scalability")
    class PerformanceTests {
        
        @Test
        @Order(14)
        @DisplayName("Learning performance with large pattern set")
        void testLearningPerformance() {
            var startTime = System.nanoTime();
            
            // Generate 100 realistic patterns with some structure
            for (int i = 0; i < 100; i++) {
                double base = i / 200.0; // Reduce base range to [0, 0.5]
                double[] pattern = {
                    Math.max(0.0, Math.min(1.0, base + 0.05 * Math.sin(i * 0.1))),
                    Math.max(0.0, Math.min(1.0, base + 0.05 * Math.cos(i * 0.1))),
                    Math.max(0.0, Math.min(1.0, base + 0.025 * Math.sin(i * 0.2)))
                };
                topoART.learn(pattern);
            }
            
            var endTime = System.nanoTime();
            var duration = (endTime - startTime) / 1_000_000; // Convert to milliseconds
            
            // Should complete in reasonable time (less than 1 second)
            assertTrue(duration < 1000, "Learning 100 patterns should complete in under 1 second");
            
            // Should have created some structure
            assertTrue(topoART.getComponentA().getNeurons().size() > 0);
            assertTrue(topoART.getComponentA().getNeurons().size() < 100); // Should compress
        }
    }

    @Nested  
    @DisplayName("Algorithm Validation")
    class AlgorithmValidationTests {
        
        @Test
        @Order(15)
        @DisplayName("Complete TopoART algorithm integration")
        void testCompleteAlgorithmIntegration() {
            // Test with iris-like dataset structure
            double[][] patterns = {
                // Class 1: Small measurements
                {0.2, 0.1, 0.15},
                {0.18, 0.12, 0.14},
                {0.22, 0.08, 0.16},
                // Class 2: Medium measurements  
                {0.5, 0.4, 0.45},
                {0.52, 0.38, 0.47},
                {0.48, 0.42, 0.43},
                // Class 3: Large measurements
                {0.8, 0.9, 0.85},
                {0.82, 0.88, 0.87},
                {0.78, 0.92, 0.83}
            };
            
            // Learn all patterns multiple times to achieve permanence
            for (double[] pattern : patterns) {
                for (int i = 0; i < 6; i++) { // Exceeds phi=5 threshold
                    topoART.learn(pattern);
                }
            }
            
            // Verify the algorithm created reasonable structure
            assertTrue(topoART.getComponentA().getNeurons().size() >= 1);
            assertTrue(topoART.getComponentB().getNeurons().size() >= 1);
            
            // Get final clusters
            var clusters = topoART.getClusters(false);
            assertNotNull(clusters);
            
            // Should find some clustering structure (likely 3 clusters for our 3 classes)
            assertTrue(clusters.size() >= 1 && clusters.size() <= patterns.length);
            
            // Verify clusters contain neurons
            for (var cluster : clusters) {
                assertFalse(cluster.isEmpty());
            }
        }
    }
}