package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.parameters.TopoARTParameters;
import com.hellblazer.art.performance.algorithms.VectorizedParameters;
import com.hellblazer.art.performance.algorithms.VectorizedFuzzyART;
import com.hellblazer.art.performance.algorithms.VectorizedTopoART;
import com.hellblazer.art.performance.algorithms.VectorizedHypersphereART;
import com.hellblazer.art.performance.algorithms.VectorizedHypersphereParameters;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Basic tests for vectorized algorithms to verify they can be instantiated and used.
 * These tests use only methods that actually exist in the implementations.
 */
class VectorizedAlgorithmsBasicTest {

    @Test
    @DisplayName("Test VectorizedFuzzyART basic functionality")
    void testVectorizedFuzzyARTBasic() {
        // Create parameters using actual factory method
        var params = VectorizedParameters.createDefault();
        
        assertDoesNotThrow(() -> {
            var fuzzyArt = new VectorizedFuzzyART(params);
            assertNotNull(fuzzyArt);
            
            // Test getParameters method that definitely exists
            var retrievedParams = fuzzyArt.getParameters();
            assertNotNull(retrievedParams);
            
            // Test getVectorSpeciesLength method that exists
            int vectorLength = fuzzyArt.getVectorSpeciesLength();
            assertTrue(vectorLength >= 0);
            
            // Test getPerformanceStats method that exists
            var stats = fuzzyArt.getPerformanceStats();
            assertNotNull(stats);
            
            // Test learn method with actual signature
            var pattern = Pattern.of(0.5, 0.5);
            var result = fuzzyArt.learn(pattern, params);
            assertNotNull(result);
            
            // Test predict method with actual signature  
            var prediction = fuzzyArt.predict(pattern, params);
            assertNotNull(prediction);
            
            fuzzyArt.close();
        });
    }

    @Test
    @DisplayName("Test VectorizedTopoART basic functionality")
    void testVectorizedTopoARTBasic() {
        // Create TopoART parameters
        var params = new TopoARTParameters(2, 0.5, 0.1, 5, 10, 0.01);
        
        assertDoesNotThrow(() -> {
            var topoArt = new VectorizedTopoART(params);
            assertNotNull(topoArt);
            
            // Test getParameters method
            var retrievedParams = topoArt.getParameters();
            assertNotNull(retrievedParams);
            assertEquals(TopoARTParameters.class, retrievedParams.getClass());
            
            // Test learn method with Pattern
            var pattern = Pattern.of(0.3, 0.7);
            var result = topoArt.learn(pattern);
            assertNotNull(result);
            
            // Test getCategoryCount method that exists
            int categoryCount = topoArt.getCategoryCount();
            assertTrue(categoryCount >= 0);
        });
    }

    @Test
    @DisplayName("Test VectorizedHypersphereART basic functionality")  
    void testVectorizedHypersphereARTBasic() {
        // Create parameters using factory method
        var params = VectorizedHypersphereParameters.conservative(2);
        
        assertDoesNotThrow(() -> {
            var hypersphereArt = new VectorizedHypersphereART(params);
            assertNotNull(hypersphereArt);
            
            // Test getParameters method
            var retrievedParams = hypersphereArt.getParameters();
            assertNotNull(retrievedParams);
            
            // Test learn method with Pattern
            var pattern = Pattern.of(0.4, 0.6);
            var result = hypersphereArt.learn(pattern);
            assertTrue(result >= 0); // Should return category index
            
            // Test getCategoryCount method
            int categoryCount = hypersphereArt.getCategoryCount();
            assertTrue(categoryCount > 0); // Should have at least one category after learning
            
            // Test predict method
            var prediction = hypersphereArt.predict(pattern, params);
            assertNotNull(prediction);
            
            hypersphereArt.close();
        });
    }

    @Test
    @DisplayName("Test parameter validation")
    void testParameterValidation() {
        // Test VectorizedParameters validation
        assertDoesNotThrow(() -> {
            var params = VectorizedParameters.createDefault();
            assertNotNull(params);
        });
        
        // Test VectorizedHypersphereParameters validation
        assertDoesNotThrow(() -> {
            var params = VectorizedHypersphereParameters.conservative(3);
            assertNotNull(params);
        });
        
        // Test TopoARTParameters validation  
        assertDoesNotThrow(() -> {
            var params = new TopoARTParameters(2, 0.7, 0.1, 5, 10, 0.01);
            assertNotNull(params);
        });
    }

    @Test
    @DisplayName("Test basic learning and prediction workflow")
    void testBasicLearningWorkflow() {
        // Test with simple 2D patterns
        var pattern1 = Pattern.of(0.1, 0.9);
        var pattern2 = Pattern.of(0.9, 0.1);
        var pattern3 = Pattern.of(0.5, 0.5);
        
        // Test FuzzyART workflow
        var fuzzyParams = VectorizedParameters.createDefault();
        var fuzzyArt = new VectorizedFuzzyART(fuzzyParams);
        
        assertDoesNotThrow(() -> {
            // Learn patterns
            fuzzyArt.learn(pattern1, fuzzyParams);
            fuzzyArt.learn(pattern2, fuzzyParams);
            fuzzyArt.learn(pattern3, fuzzyParams);
            
            // Make predictions
            fuzzyArt.predict(pattern1, fuzzyParams);
            fuzzyArt.predict(pattern2, fuzzyParams);
            fuzzyArt.predict(pattern3, fuzzyParams);
        });
        
        fuzzyArt.close();
        
        // Test HypersphereART workflow
        var hypersphereParams = VectorizedHypersphereParameters.conservative(2);
        var hypersphereArt = new VectorizedHypersphereART(hypersphereParams);
        
        assertDoesNotThrow(() -> {
            // Learn patterns
            hypersphereArt.learn(pattern1);
            hypersphereArt.learn(pattern2);
            hypersphereArt.learn(pattern3);
            
            // Verify categories were created
            assertTrue(hypersphereArt.getCategoryCount() > 0);
            
            // Make predictions
            hypersphereArt.predict(pattern1, hypersphereParams);
            hypersphereArt.predict(pattern2, hypersphereParams);
            hypersphereArt.predict(pattern3, hypersphereParams);
        });
        
        hypersphereArt.close();
    }
}