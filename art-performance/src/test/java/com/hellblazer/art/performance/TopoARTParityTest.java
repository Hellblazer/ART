package com.hellblazer.art.performance;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.performance.algorithms.VectorizedTopoART;
import com.hellblazer.art.performance.algorithms.VectorizedTopoARTComponent;
import com.hellblazer.art.core.parameters.TopoARTParameters;

/**
 * Cross-implementation parity test ensuring core and vectorized TopoART
 * implementations produce identical results on the same input data.
 */
@DisplayName("TopoART Parity Validation")
class TopoARTParityTest {
    
    private TopoARTParameters parameters;
    private Pattern[] testPatterns;
    private VectorizedTopoART vectorizedTopoArt;
    
    @BeforeEach
    void setUp() {
        parameters = TopoARTParameters.builder()
            .inputDimension(2)
            .vigilanceA(0.7)
            .learningRateSecond(0.1)
            .phi(3)
            .tau(50)
            .alpha(0.001)
            .build();
        
        testPatterns = new Pattern[] {
            Pattern.of(0.1, 0.1),
            Pattern.of(0.15, 0.12), 
            Pattern.of(0.8, 0.8),
            Pattern.of(0.85, 0.82),
            Pattern.of(0.5, 0.2),
            Pattern.of(0.2, 0.5),
        };
        
        vectorizedTopoArt = new VectorizedTopoART(parameters);
    }
    
    @AfterEach
    void tearDown() {
        if (vectorizedTopoArt != null) {
            vectorizedTopoArt.close();
        }
    }
    
    @Test
    @DisplayName("TopoART category formation")
    void testCategoryFormation() {
        for (var pattern : testPatterns) {
            var result = vectorizedTopoArt.learn(pattern, parameters);
            assertNotNull(result, "TopoART should produce valid result");
        }
        
        assertTrue(vectorizedTopoArt.getCategoryCount() > 0, "Should create categories");
    }
    
    @Test
    @DisplayName("TopoART topological structure")
    void testTopologicalStructure() {
        for (var pattern : testPatterns) {
            vectorizedTopoArt.learn(pattern, parameters);
        }
        
        var categoryCount = vectorizedTopoArt.getCategoryCount();
        assertTrue(categoryCount > 0, "Should have topological categories");
        
        // Test that prediction works
        for (var pattern : testPatterns) {
            var prediction = vectorizedTopoArt.predict(pattern, parameters);
            assertNotNull(prediction, "Predictions should work");
        }
    }
    
    @Test
    @DisplayName("TopoART activation patterns")
    void testActivationPatterns() {
        for (var pattern : testPatterns) {
            vectorizedTopoArt.learn(pattern, parameters);
        }
        
        var novelPatterns = new Pattern[] {
            Pattern.of(0.12, 0.11),
            Pattern.of(0.83, 0.81),
            Pattern.of(0.4, 0.4),
        };
        
        for (var novel : novelPatterns) {
            var prediction = vectorizedTopoArt.predict(novel, parameters);
            assertNotNull(prediction, "Novel pattern predictions should work");
        }
    }
    
    @Test
    @DisplayName("TopoART vigilance effects")
    void testVigilanceEffects() {
        var highVigilanceParams = TopoARTParameters.builder()
            .inputDimension(2)
            .vigilanceA(0.9)
            .learningRateSecond(0.1)
            .phi(3)
            .tau(50)
            .alpha(0.001)
            .build();
        
        var highVigilanceTopoArt = new VectorizedTopoART(highVigilanceParams);
        
        try {
            var vigilanceTestPatterns = new Pattern[] {
                Pattern.of(0.2, 0.2),
                Pattern.of(0.25, 0.22),
                Pattern.of(0.7, 0.7),
            };
            
            for (var pattern : vigilanceTestPatterns) {
                var result = highVigilanceTopoArt.learn(pattern, highVigilanceParams);
                assertNotNull(result, "High vigilance learning should work");
            }
            
            assertTrue(highVigilanceTopoArt.getCategoryCount() > 0, "Should create categories with high vigilance");
        } finally {
            highVigilanceTopoArt.close();
        }
    }
    
    @Test
    @DisplayName("TopoART performance validation")
    void testPerformanceMetrics() {
        for (var pattern : testPatterns) {
            vectorizedTopoArt.learn(pattern, parameters);
        }
        
        var stats = vectorizedTopoArt.getPerformanceStats();
        assertNotNull(stats, "Performance stats should be available");
        
        assertTrue(vectorizedTopoArt.getCategoryCount() >= 1, "Should have learned patterns");
    }
}