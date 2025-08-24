package com.hellblazer.art.algorithms;

import com.hellblazer.art.core.*;
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for VectorizedART implementation.
 * Tests vectorized operations, JOML optimizations, parallel processing, and performance characteristics.
 */
class VectorizedARTTest {
    
    private VectorizedART art;
    private VectorizedParameters params;
    
    @BeforeEach
    void setUp() {
        params = new VectorizedParameters(
            0.8,     // vigilanceThreshold
            0.1,     // learningRate
            0.01,    // alpha
            4,       // parallelismLevel
            100,     // parallelThreshold
            1000,    // maxCacheSize
            true,    // enableSIMD
            true,    // enableJOML
            0.8      // memoryOptimizationThreshold
        );
        art = new VectorizedART(params);
    }
    
    @AfterEach
    void tearDown() {
        if (art != null) {
            art.close();
        }
    }
    
    @Test
    @DisplayName("Constructor validation")
    void testConstructorValidation() {
        assertThrows(NullPointerException.class, () -> new VectorizedART(null));
        
        // Valid construction should succeed
        assertDoesNotThrow(() -> new VectorizedART(params));
    }
    
    @Test
    @DisplayName("Basic step fit functionality")
    void testBasicStepFit() {
        var input = Pattern.of(0.8, 0.6, 0.4);
        
        // First input should create new category
        var result = art.stepFit(input, params);
        assertNotNull(result);
        assertTrue(result instanceof ActivationResult.Success);
        
        var success = (ActivationResult.Success) result;
        assertEquals(0, success.categoryIndex());
        assertEquals(1, art.getCategoryCount());
        
        // Similar input should activate same category
        var similarInput = Pattern.of(0.75, 0.55, 0.35);
        var result2 = art.stepFit(similarInput, params);
        assertTrue(result2 instanceof ActivationResult.Success);
        
        var success2 = (ActivationResult.Success) result2;
        assertEquals(0, success2.categoryIndex()); // Should match same category
    }
    
    @Test
    @DisplayName("Enhanced step fit functionality")
    void testEnhancedStepFit() {
        // Test enhanced step fit with normal parameters
        var testInput = Pattern.of(0.5, 0.5, 0.5);
        var result = art.stepFitEnhanced(testInput, params);
        
        assertNotNull(result);
        assertTrue(result instanceof ActivationResult.Success);
        
        var success = (ActivationResult.Success) result;
        assertEquals(0, success.categoryIndex()); // First category
        assertEquals(1, art.getCategoryCount());
        
        // Test with different input - should create or match existing category
        var testInput2 = Pattern.of(0.1, 0.2, 0.3);
        var result2 = art.stepFitEnhanced(testInput2, params);
        
        assertNotNull(result2);
        assertTrue(result2 instanceof ActivationResult.Success);
        
        // Verify performance stats are being tracked
        var stats = art.getPerformanceStats();
        assertTrue(stats.totalVectorOperations() > 0);
        assertTrue(stats.avgComputeTimeMs() >= 0.0);
        assertTrue(stats.categoryCount() > 0);
        
        // Test with lower parallel threshold to trigger parallel processing path
        var lowThresholdParams = new VectorizedParameters(
            0.8,     // vigilanceThreshold
            0.1,     // learningRate
            0.01,    // alpha
            4,       // parallelismLevel
            1,       // parallelThreshold = 1 to trigger parallel path with few categories
            1000,    // maxCacheSize
            true,    // enableSIMD
            true,    // enableJOML
            0.8      // memoryOptimizationThreshold
        );
        
        // Create fresh ART instance with low threshold
        var testArt = new VectorizedART(lowThresholdParams);
        try {
            // Add some categories first
            testArt.stepFit(Pattern.of(0.1, 0.1, 0.1), lowThresholdParams);
            testArt.stepFit(Pattern.of(0.9, 0.9, 0.9), lowThresholdParams);
            
            // Now test enhanced step fit should trigger parallel path
            var parallelResult = testArt.stepFitEnhanced(Pattern.of(0.5, 0.5, 0.5), lowThresholdParams);
            assertNotNull(parallelResult);
            assertTrue(parallelResult instanceof ActivationResult.Success);
            
            // Verify parallel tasks were executed
            var parallelStats = testArt.getPerformanceStats();
            assertTrue(parallelStats.totalParallelTasks() > 0, "Should have executed parallel tasks");
        } finally {
            testArt.close();
        }
    }
    
    @Test
    @DisplayName("Vectorized activation calculation")
    void testVectorizedActivation() {
        var input = Pattern.of(0.8, 0.6, 0.4);
        var weight = VectorizedWeight.fromInput(Pattern.of(0.7, 0.5, 0.3), params);
        
        // Test activation calculation
        var activation = art.calculateActivation(input, weight, params);
        assertTrue(activation >= 0.0);
        assertTrue(activation <= 1.0);
        
        // Test with different dimensions
        var input4D = Pattern.of(0.8, 0.6, 0.4, 0.2);
        var weight4D = VectorizedWeight.fromInput(input4D, params);
        var activation4D = art.calculateActivation(input4D, weight4D, params);
        assertTrue(activation4D >= 0.0);
        assertTrue(activation4D <= 1.0);
    }
    
    @Test
    @DisplayName("Vigilance testing")
    void testVigilanceTesting() {
        var input = Pattern.of(0.8, 0.6, 0.4);
        var similarWeight = VectorizedWeight.fromInput(Pattern.of(0.85, 0.65, 0.45), params);
        var differentWeight = VectorizedWeight.fromInput(Pattern.of(0.1, 0.1, 0.1), params);
        
        // Similar input should pass vigilance
        var similarResult = art.checkVigilance(input, similarWeight, params);
        assertTrue(similarResult.isAccepted());
        
        // Different input should fail vigilance
        var differentResult = art.checkVigilance(input, differentWeight, params);
        assertTrue(differentResult.isRejected());
    }
    
    @Test
    @DisplayName("Weight updates")
    void testWeightUpdates() {
        var input = Pattern.of(0.8, 0.6, 0.4);
        var originalWeight = VectorizedWeight.fromInput(Pattern.of(0.7, 0.5, 0.3), params);
        
        var updatedWeight = art.updateWeights(input, originalWeight, params);
        assertNotNull(updatedWeight);
        assertNotSame(originalWeight, updatedWeight); // Should be immutable
        
        // Updated weights should be between original and input
        for (int i = 0; i < input.dimension(); i++) {
            double original = originalWeight.get(i);
            double inputVal = input.get(i);
            double updated = updatedWeight.get(i);
            
            if (original < inputVal) {
                assertTrue(updated >= original);
                assertTrue(updated <= inputVal);
            } else {
                assertTrue(updated <= original);
                assertTrue(updated >= inputVal);
            }
        }
    }
    
    @Test
    @DisplayName("Initial weight creation")
    void testInitialWeightCreation() {
        var input = Pattern.of(0.8, 0.6, 0.4);
        var weight = art.createInitialWeight(input, params);
        
        assertNotNull(weight);
        assertTrue(weight instanceof VectorizedWeight);
        assertEquals(input.dimension(), weight.dimension());
        
        // Initial weight should match input
        for (int i = 0; i < input.dimension(); i++) {
            assertEquals(input.get(i), weight.get(i), 1e-10);
        }
    }
    
    @ParameterizedTest
    @ValueSource(ints = {2, 3, 4, 5, 8, 16})
    @DisplayName("Multi-dimensional vector processing")
    void testMultiDimensionalVectors(int dimension) {
        // Create random input of specified dimension
        var values = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            values[i] = Math.random();
        }
        var input = Pattern.of(values);
        
        // Test step fit with various dimensions
        var result = art.stepFit(input, params);
        assertNotNull(result);
        assertTrue(result instanceof ActivationResult.Success);
        
        var success = (ActivationResult.Success) result;
        assertEquals(input.dimension(), success.updatedWeight().dimension());
    }
    
    @Test
    @DisplayName("JOML optimization for 3D vectors")
    void testJOMLOptimization3D() {
        var params3D = new VectorizedParameters(
            0.8, 0.1, 0.01, 4, 100, 
            1000,  // maxCacheSize
            false, // enableSIMD = false to test JOML specifically
            true,  // enableJOML = true
            0.8    // memoryOptimizationThreshold
        );
        var art3D = new VectorizedART(params3D);
        
        try {
            var input = Pattern.of(0.8, 0.6, 0.4);
            var result = art3D.stepFit(input, params3D);
            
            assertNotNull(result);
            assertTrue(result instanceof ActivationResult.Success);
        } finally {
            art3D.close();
        }
    }
    
    @Test
    @DisplayName("JOML optimization for 4D vectors")
    void testJOMLOptimization4D() {
        var params4D = new VectorizedParameters(
            0.8, 0.1, 0.01, 4, 100,
            1000,  // maxCacheSize
            false, // enableSIMD = false to test JOML specifically
            true,  // enableJOML = true
            0.8    // memoryOptimizationThreshold
        );
        var art4D = new VectorizedART(params4D);
        
        try {
            var input = Pattern.of(0.8, 0.6, 0.4, 0.2);
            var result = art4D.stepFit(input, params4D);
            
            assertNotNull(result);
            assertTrue(result instanceof ActivationResult.Success);
        } finally {
            art4D.close();
        }
    }
    
    @Test
    @DisplayName("SIMD optimization for larger vectors")
    void testSIMDOptimization() {
        var paramsSIMD = new VectorizedParameters(
            0.8, 0.1, 0.01, 4, 100,
            1000,  // maxCacheSize
            true,  // enableSIMD = true
            false, // enableJOML = false
            0.8    // memoryOptimizationThreshold
        );
        var artSIMD = new VectorizedART(paramsSIMD);
        
        try {
            // Create larger vector to trigger SIMD
            var values = new double[16];
            for (int i = 0; i < values.length; i++) {
                values[i] = Math.random();
            }
            var input = Pattern.of(values);
            
            var result = artSIMD.stepFit(input, paramsSIMD);
            assertNotNull(result);
            assertTrue(result instanceof ActivationResult.Success);
        } finally {
            artSIMD.close();
        }
    }
    
    @Test
    @DisplayName("Performance statistics tracking")
    void testPerformanceStats() {
        // Perform several operations
        for (int i = 0; i < 10; i++) {
            var input = Pattern.of(Math.random(), Math.random(), Math.random());
            art.stepFit(input, params);
        }
        
        var stats = art.getPerformanceStats();
        assertNotNull(stats);
        assertTrue(stats.totalVectorOperations() > 0);
        assertTrue(stats.avgComputeTimeMs() >= 0.0);
        assertTrue(stats.categoryCount() > 0);
        
        // Test reset
        art.resetPerformanceTracking();
        var resetStats = art.getPerformanceStats();
        assertEquals(0, resetStats.totalVectorOperations());
        assertEquals(0, resetStats.totalParallelTasks());
        assertEquals(0.0, resetStats.avgComputeTimeMs());
    }
    
    @Test
    @DisplayName("Memory optimization")
    void testMemoryOptimization() {
        // Fill cache beyond limit
        for (int i = 0; i < params.maxCacheSize() + 100; i++) {
            var input = Pattern.of(Math.random(), Math.random(), Math.random());
            art.stepFit(input, params);
        }
        
        var statsBefore = art.getPerformanceStats();
        art.optimizeMemory();
        var statsAfter = art.getPerformanceStats();
        
        // Cache should be cleared if it exceeded limit
        assertTrue(statsAfter.cacheSize() <= statsBefore.cacheSize());
    }
    
    @Test
    @DisplayName("Thread safety and concurrent access")
    void testThreadSafety() throws InterruptedException {
        var threads = new Thread[4];
        var results = new ActivationResult[threads.length];
        var exceptions = new Exception[threads.length];
        
        // Create concurrent step fit operations
        for (int i = 0; i < threads.length; i++) {
            final int threadIndex = i;
            threads[i] = new Thread(() -> {
                try {
                    var input = Pattern.of(
                        0.1 + threadIndex * 0.2,
                        0.2 + threadIndex * 0.2,
                        0.3 + threadIndex * 0.2
                    );
                    results[threadIndex] = art.stepFit(input, params);
                } catch (Exception e) {
                    exceptions[threadIndex] = e;
                    System.err.printf("Thread %d failed with exception: %s%n", threadIndex, e.getMessage());
                    e.printStackTrace();
                }
            });
        }
        
        // Start all threads
        for (var thread : threads) {
            thread.start();
        }
        
        // Wait for completion
        for (var thread : threads) {
            thread.join();
        }
        
        // Check for exceptions first
        for (int i = 0; i < exceptions.length; i++) {
            if (exceptions[i] != null) {
                fail("Thread " + i + " failed with exception: " + exceptions[i].getMessage(), exceptions[i]);
            }
        }
        
        // Verify all operations completed successfully
        for (int i = 0; i < results.length; i++) {
            assertNotNull(results[i], "Result from thread " + i + " is null");
            assertTrue(results[i] instanceof ActivationResult.Success, 
                      "Result from thread " + i + " is not Success: " + results[i]);
        }
    }
    
    @Test
    @DisplayName("Resource cleanup")
    void testResourceCleanup() {
        var testArt = new VectorizedART(params);
        
        // Use the ART instance
        var input = Pattern.of(0.5, 0.5, 0.5);
        testArt.stepFit(input, params);
        
        // Close should not throw exceptions
        assertDoesNotThrow(testArt::close);
        
        // Multiple closes should be safe
        assertDoesNotThrow(testArt::close);
    }
    
    @Test
    @DisplayName("Error handling and validation")
    void testErrorHandling() {
        var input = Pattern.of(0.5, 0.5, 0.5);
        
        // Null input should throw
        assertThrows(NullPointerException.class, () -> art.stepFit(null, params));
        assertThrows(NullPointerException.class, () -> art.stepFit(input, null));
        
        // Invalid parameters should throw
        assertThrows(IllegalArgumentException.class, () -> 
            art.stepFit(input, "invalid parameters"));
    }
    
    @Test
    @DisplayName("Conversion between weight types")
    void testWeightTypeConversion() {
        // Create a FuzzyWeight and ensure VectorizedART can handle it
        var originalInput = Pattern.of(0.7, 0.6, 0.5);
        var fuzzyWeight = FuzzyWeight.fromInput(originalInput);
        
        // Create complement-coded input to match fuzzy weight dimension
        var complementInput = Pattern.of(0.8, 0.6, 0.4, 0.2, 0.4, 0.6);
        
        // VectorizedART should be able to process any WeightVector type
        var activation = art.calculateActivation(complementInput, fuzzyWeight, params);
        assertTrue(activation >= 0.0);
        assertTrue(activation <= 1.0);
        
        var vigilanceResult = art.checkVigilance(complementInput, fuzzyWeight, params);
        assertNotNull(vigilanceResult);
        
        var updatedWeight = art.updateWeights(complementInput, fuzzyWeight, params);
        assertNotNull(updatedWeight);
        assertEquals(complementInput.dimension(), updatedWeight.dimension());
    }
}