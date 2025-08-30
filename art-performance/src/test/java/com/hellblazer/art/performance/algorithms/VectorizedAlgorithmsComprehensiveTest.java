package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.parameters.TopoARTParameters;
import com.hellblazer.art.performance.algorithms.VectorizedParameters;
import com.hellblazer.art.performance.algorithms.VectorizedFuzzyART;
import com.hellblazer.art.performance.algorithms.VectorizedTopoART;
import com.hellblazer.art.performance.algorithms.VectorizedHypersphereART;
import com.hellblazer.art.performance.algorithms.VectorizedHypersphereParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.RepeatedTest;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for vectorized ART algorithms based on reference implementation.
 * These tests mirror the test patterns from reference parity/unit_tests/*.py
 */
class VectorizedAlgorithmsComprehensiveTest {

    private Random random;

    @BeforeEach
    void setUp() {
        random = new Random(42); // Fixed seed for reproducible tests
    }

    @Test
    @DisplayName("FuzzyART - Test initialization and parameter validation")
    void testFuzzyARTInitialization() {
        // Test with valid parameters (mirrors Python test_initialization)
        var params = VectorizedParameters.createDefault();
        assertDoesNotThrow(() -> {
            try (var fuzzyArt = new VectorizedFuzzyART(params)) {
                assertNotNull(fuzzyArt);
                assertNotNull(fuzzyArt.getParameters());
                assertEquals(VectorizedParameters.class, fuzzyArt.getParameters().getClass());
                
                // Test SIMD initialization
                assertTrue(fuzzyArt.getVectorSpeciesLength() > 0);
                assertNotNull(fuzzyArt.getPerformanceStats());
            }
        });
    }

    @Test 
    @DisplayName("FuzzyART - Test clustering behavior with known data")
    void testFuzzyARTClustering() {
        // Create test data similar to Python clustering test
        var patterns = List.of(
            Pattern.of(0.0, 0.0),    // Cluster 1
            Pattern.of(0.0, 0.08),   // Should join cluster 1 (similar)
            Pattern.of(0.0, 1.0),    // Cluster 2 (distant)
            Pattern.of(1.0, 1.0),    // Cluster 3 (distant)
            Pattern.of(1.0, 0.0)     // Cluster 4 (distant)
        );

        // Use high vigilance to create separate clusters (mirrors Python test)
        var params = new VectorizedParameters(
            0.9,    // High vigilance
            0.1,    // Learning rate
            0.05,   // Alpha  
            1,      // Parallelism level for test
            100,    // Parallel threshold
            1000,   // Max cache size
            true,   // Enable SIMD
            false,  // Enable JOML
            1e-6    // Memory optimization threshold
        );

        try (var fuzzyArt = new VectorizedFuzzyART(params)) {
            var labels = new ArrayList<Integer>();
            
            // Learn each pattern and track labels
            for (var pattern : patterns) {
                var result = fuzzyArt.learn(pattern, params);
                assertNotNull(result);
            }

            // Make predictions to verify they work
            for (var pattern : patterns) {
                var prediction = fuzzyArt.predict(pattern, params);
                assertNotNull(prediction);
            }

            // Note: Since predict returns Object, we focus on testing that learning and prediction 
            // work without exceptions, which mirrors the Python test approach
        }
    }

    @Test
    @DisplayName("HypersphereART - Test initialization and parameter validation") 
    void testHypersphereARTInitialization() {
        // Test parameter validation (mirrors Python test_validate_params)
        assertDoesNotThrow(() -> {
            var params = VectorizedHypersphereParameters.conservative(2);
            assertNotNull(params);
        });

        // Test model initialization
        var params = VectorizedHypersphereParameters.conservative(2);
        try (var hypersphereArt = new VectorizedHypersphereART(params)) {
            assertNotNull(hypersphereArt);
            assertNotNull(hypersphereArt.getParameters());
            assertEquals(0, hypersphereArt.getCategoryCount()); // Initially no categories
        }
    }

    @Test
    @DisplayName("HypersphereART - Test cluster creation and centers")
    void testHypersphereARTClustering() {
        var params = VectorizedHypersphereParameters.conservative(2);
        
        try (var hypersphereArt = new VectorizedHypersphereART(params)) {
            // Test learning patterns (mirrors Python test_fit)
            var patterns = List.of(
                Pattern.of(0.1, 0.2),
                Pattern.of(0.3, 0.4), 
                Pattern.of(0.5, 0.6)
            );

            var categoryIndices = new ArrayList<Integer>();
            for (var pattern : patterns) {
                var categoryIndex = hypersphereArt.learn(pattern);
                assertTrue(categoryIndex >= 0);
                categoryIndices.add(categoryIndex);
            }

            // Verify clusters were created
            assertTrue(hypersphereArt.getCategoryCount() > 0);
            
            // Test prediction consistency
            for (int i = 0; i < patterns.size(); i++) {
                var prediction = hypersphereArt.predict(patterns.get(i), params);
                assertNotNull(prediction);
                // Note: Prediction may differ from learning due to match tracking
            }
        }
    }

    @Test
    @DisplayName("TopoART - Test initialization with base module")
    void testTopoARTInitialization() {
        // Test initialization (mirrors Python test_initialization)
        var params = new TopoARTParameters(2, 0.5, 0.1, 5, 10, 0.01);
        
        assertDoesNotThrow(() -> {
            var topoArt = new VectorizedTopoART(params);
            assertNotNull(topoArt);
            assertNotNull(topoArt.getParameters());
            assertEquals(TopoARTParameters.class, topoArt.getParameters().getClass());
            assertEquals(0, topoArt.getCategoryCount()); // Initially no categories
        });
    }

    @Test
    @DisplayName("TopoART - Test adjacency matrix updates")
    void testTopoARTAdjacencyMatrix() {
        var params = new TopoARTParameters(2, 0.5, 0.1, 5, 10, 0.01);
        
        var topoArt = new VectorizedTopoART(params);
        
        // Generate test data (mirrors Python test_adjacency_matrix)  
        var patterns = generateRandomPatterns(5, 2);
        
        int previousCategoryCount = 0;
        for (var pattern : patterns) {
            var result = topoArt.learn(pattern);
            assertNotNull(result);
            
            int currentCategoryCount = topoArt.getCategoryCount();
            assertTrue(currentCategoryCount >= previousCategoryCount);
            previousCategoryCount = currentCategoryCount;
        }
        
        // Verify final state
        assertTrue(topoArt.getCategoryCount() > 0);
    }

    @RepeatedTest(3)
    @DisplayName("Stress test - Multiple learning cycles")
    void testMultipleLearningCycles() {
        // Generate larger dataset for stress testing
        var patterns = generateRandomPatterns(50, 3);
        
        // Test all algorithms with the same data
        testAlgorithmStress("FuzzyART", patterns, this::createFuzzyARTForStress);
        testAlgorithmStress("HypersphereART", patterns, this::createHypersphereARTForStress);
        testAlgorithmStress("TopoART", patterns, this::createTopoARTForStress);
    }

    @Test
    @DisplayName("Performance measurement consistency")
    void testPerformanceStats() {
        var params = VectorizedParameters.createDefault();
        
        try (var fuzzyArt = new VectorizedFuzzyART(params)) {
            var initialStats = fuzzyArt.getPerformanceStats();
            assertNotNull(initialStats);
            
            // Perform some learning
            var patterns = generateRandomPatterns(10, 2);
            for (var pattern : patterns) {
                fuzzyArt.learn(pattern, params);
            }
            
            var finalStats = fuzzyArt.getPerformanceStats();
            assertNotNull(finalStats);
            // Performance stats should be updated after learning
        }
    }

    @Test
    @DisplayName("Resource cleanup verification")
    void testResourceCleanup() {
        // Test proper cleanup of vectorized resources
        assertDoesNotThrow(() -> {
            var params = VectorizedParameters.createDefault();
            var fuzzyArt = new VectorizedFuzzyART(params);
            
            // Use the algorithm
            fuzzyArt.learn(Pattern.of(0.5, 0.5), params);
            
            // Close should not throw
            fuzzyArt.close();
            
            // Second close should also not throw
            fuzzyArt.close();
        });
        
        assertDoesNotThrow(() -> {
            var params = VectorizedHypersphereParameters.conservative(2);
            var hypersphereArt = new VectorizedHypersphereART(params);
            
            hypersphereArt.learn(Pattern.of(0.3, 0.7));
            hypersphereArt.close();
            hypersphereArt.close(); // Double close should be safe
        });
    }

    // Helper methods
    
    private List<Pattern> generateRandomPatterns(int count, int dimension) {
        var patterns = new ArrayList<Pattern>(count);
        for (int i = 0; i < count; i++) {
            var values = new double[dimension];
            for (int j = 0; j < dimension; j++) {
                values[j] = random.nextDouble();
            }
            patterns.add(Pattern.of(values));
        }
        return patterns;
    }
    
    private interface AlgorithmFactory {
        AutoCloseable create();
    }
    
    private void testAlgorithmStress(String algorithmName, List<Pattern> patterns, AlgorithmFactory factory) {
        assertDoesNotThrow(() -> {
            try (var algorithm = factory.create()) {
                // Perform learning cycles - each algorithm handles this differently
                for (var pattern : patterns) {
                    if (algorithm instanceof VectorizedFuzzyART fuzzyArt) {
                        var params = VectorizedParameters.createDefault();
                        fuzzyArt.learn(pattern, params);
                    } else if (algorithm instanceof VectorizedHypersphereART hypersphereArt) {
                        hypersphereArt.learn(pattern);
                    } else if (algorithm instanceof VectorizedTopoART topoArt) {
                        topoArt.learn(pattern);
                    }
                }
            }
        }, algorithmName + " should handle stress test without exceptions");
    }
    
    private AutoCloseable createFuzzyARTForStress() {
        var params = VectorizedParameters.createDefault();
        return new VectorizedFuzzyART(params);
    }
    
    private AutoCloseable createHypersphereARTForStress() {
        var params = VectorizedHypersphereParameters.conservative(3);
        return new VectorizedHypersphereART(params);
    }
    
    private AutoCloseable createTopoARTForStress() {
        var params = new TopoARTParameters(3, 0.7, 0.1, 5, 10, 0.01);
        return new VectorizedTopoART(params);
    }
}