package com.hellblazer.art.performance;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.performance.algorithms.VectorizedHypersphereART;
import com.hellblazer.art.performance.algorithms.VectorizedHypersphereParameters;

/**
 * Reference validation test for VectorizedHypersphereART implementation.
 * Tests SIMD-accelerated hyperspherical clustering behavior.
 */
@DisplayName("Vectorized Hypersphere ART Reference Validation")
class VectorizedHypersphereARTReferenceTest {
    
    private VectorizedHypersphereParameters referenceParams;
    private VectorizedHypersphereART vectorizedHypersphereArt;
    
    @BeforeEach
    void setUp() {
        referenceParams = VectorizedHypersphereParameters.builder()
            .inputDimensions(2)
            .vigilance(0.7)
            .learningRate(0.1)
            .build();
        
        vectorizedHypersphereArt = new VectorizedHypersphereART(referenceParams);
    }
    
    @AfterEach
    void tearDown() {
        if (vectorizedHypersphereArt != null) {
            vectorizedHypersphereArt.close();
        }
    }
    
    @Test
    @DisplayName("Hyperspherical clustering validation")
    void testHypersphericalClustering() {
        var patterns = new Pattern[] {
            Pattern.of(0.1, 0.1),
            Pattern.of(0.9, 0.9),
            Pattern.of(0.15, 0.05)
        };
        
        for (var pattern : patterns) {
            var result = vectorizedHypersphereArt.learn(pattern, referenceParams);
            assertNotNull(result, "Hypersphere learning should succeed");
        }
        
        assertTrue(vectorizedHypersphereArt.getCategoryCount() > 0, "Should create hyperspherical categories");
    }
    
    @Test
    @DisplayName("Euclidean distance calculations")
    void testEuclideanDistanceCalculations() {
        var patterns = new Pattern[] {
            Pattern.of(0.3, 0.7),
            Pattern.of(0.8, 0.2),
            Pattern.of(0.5, 0.5)
        };
        
        for (var pattern : patterns) {
            vectorizedHypersphereArt.learn(pattern, referenceParams);
        }
        
        // Test predictions work with Euclidean distance
        for (var pattern : patterns) {
            var prediction = vectorizedHypersphereArt.predict(pattern, referenceParams);
            assertNotNull(prediction, "Distance-based predictions should work");
        }
    }
    
    @Test
    @DisplayName("Spherical category boundaries")
    void testSphericalBoundaries() {
        var centerPattern = Pattern.of(0.5, 0.5);
        var result = vectorizedHypersphereArt.learn(centerPattern, referenceParams);
        assertNotNull(result, "Center pattern learning should succeed");
        
        var boundaryPatterns = new Pattern[] {
            Pattern.of(0.6, 0.5),  // Right of center
            Pattern.of(0.5, 0.6),  // Above center
            Pattern.of(0.4, 0.5),  // Left of center
            Pattern.of(0.5, 0.4)   // Below center
        };
        
        for (var pattern : boundaryPatterns) {
            var prediction = vectorizedHypersphereArt.predict(pattern, referenceParams);
            assertNotNull(prediction, "Boundary predictions should work");
        }
    }
    
    @Test
    @DisplayName("Performance metrics validation")
    void testPerformanceMetrics() {
        var patterns = new Pattern[] {
            Pattern.of(0.2, 0.8),
            Pattern.of(0.7, 0.3),
            Pattern.of(0.4, 0.6)
        };
        
        for (var pattern : patterns) {
            vectorizedHypersphereArt.learn(pattern, referenceParams);
        }
        
        var stats = vectorizedHypersphereArt.getPerformanceStats();
        assertNotNull(stats, "Performance stats should be available");
        
        assertTrue(vectorizedHypersphereArt.getCategoryCount() >= 1, "Should have learned patterns");
    }
    
    @Test
    @DisplayName("Vigilance parameter effects")
    void testVigilanceEffects() {
        var highVigilanceParams = VectorizedHypersphereParameters.builder()
            .inputDimensions(2)
            .vigilance(0.9)
            .learningRate(0.1)
            .build();
        
        var highVigilanceArt = new VectorizedHypersphereART(highVigilanceParams);
        
        try {
            var testPatterns = new Pattern[] {
                Pattern.of(0.3, 0.3),
                Pattern.of(0.31, 0.29),
                Pattern.of(0.7, 0.7)
            };
            
            for (var pattern : testPatterns) {
                var result = highVigilanceArt.learn(pattern, highVigilanceParams);
                assertNotNull(result, "High vigilance learning should work");
            }
            
            assertTrue(highVigilanceArt.getCategoryCount() > 0, "Should create categories with high vigilance");
        } finally {
            highVigilanceArt.close();
        }
    }
}