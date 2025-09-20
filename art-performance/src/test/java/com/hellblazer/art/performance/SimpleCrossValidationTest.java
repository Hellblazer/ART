package com.hellblazer.art.performance;

import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import com.hellblazer.art.core.*;
import com.hellblazer.art.core.algorithms.*;
import com.hellblazer.art.core.parameters.*;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.performance.algorithms.*;

/**
 * Simple cross-validation test to debug compilation issues.
 */
@DisplayName("Simple Cross-Validation Test")
class SimpleCrossValidationTest {

    @Test
    @DisplayName("Basic FuzzyART vs VectorizedFuzzyART Test")
    void testBasicFuzzyARTCrossValidation() throws Exception {
        // Create simple test patterns
        var pattern1 = Pattern.of(0.1, 0.2, 0.3, 0.4);
        var pattern2 = Pattern.of(0.9, 0.8, 0.7, 0.6);
        
        // Core FuzzyART with parameters
        var coreFuzzyART = new FuzzyART();
        var coreParams = new FuzzyParameters(0.7, 0.1, 0.9);
        
        // Vectorized FuzzyART
        var vectorizedParams = new VectorizedParameters(
            0.7,    // vigilanceThreshold
            0.9,    // learningRate  
            0.1,    // alpha
            1,      // parallelismLevel (single thread for fair comparison)
            1000,   // parallelThreshold
            1000,   // maxCacheSize
            true,   // enableSIMD
            true,   // enableJOML
            0.8     // memoryOptimizationThreshold
        );
        var vectorizedFuzzyART = new VectorizedFuzzyART(vectorizedParams);
        
        try {
            // Test basic functionality
            var coreResult1 = coreFuzzyART.stepFit(pattern1, coreParams);
            var vectorizedResult1 = vectorizedFuzzyART.learn(pattern1, vectorizedParams);
            
            var coreResult2 = coreFuzzyART.stepFit(pattern2, coreParams);
            var vectorizedResult2 = vectorizedFuzzyART.learn(pattern2, vectorizedParams);
            
            // Basic validation - just ensure they both work
            assertNotNull(coreResult1);
            assertNotNull(vectorizedResult1);
            assertTrue(vectorizedResult1 instanceof ActivationResult.Success);
            assertNotNull(coreResult2);
            assertNotNull(vectorizedResult2);
            assertTrue(vectorizedResult2 instanceof ActivationResult.Success);
            
            System.out.printf("Core categories: %d, Vectorized categories: %d%n", 
                             coreFuzzyART.getCategoryCount(), vectorizedFuzzyART.getCategoryCount());
            
        } finally {
            vectorizedFuzzyART.close();
        }
    }
}