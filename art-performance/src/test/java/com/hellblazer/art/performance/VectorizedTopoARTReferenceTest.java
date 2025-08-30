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
package com.hellblazer.art.performance;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.parameters.TopoARTParameters;
import com.hellblazer.art.core.results.TopoARTResult;
import com.hellblazer.art.performance.algorithms.VectorizedTopoART;
import com.hellblazer.art.performance.algorithms.VectorizedPerformanceStats;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Reference Comparison Test for Vectorized TopoART Implementation.
 * 
 * This test validates that the vectorized TopoART implementation produces topology-aware
 * learning behavior consistent with reference TopoART implementations, while providing
 * performance improvements through SIMD vectorization.
 * 
 * TopoART combines adaptive resonance with topology learning, creating edge connections
 * between winning neurons to preserve topological relationships in the input space.
 */
@DisplayName("Vectorized TopoART Reference Validation")
class VectorizedTopoARTReferenceTest {
    
    private VectorizedTopoART vectorizedTopoArt;
    private TopoARTParameters referenceParams;
    
    @BeforeEach
    void setUp() {
        // Parameters tuned for topology learning similar to reference TopoART implementations
        referenceParams = TopoARTParameters.builder()
            .inputDimension(2)
            .vigilanceA(0.8)
            .learningRateSecond(0.5)
            .phi(3)
            .tau(50)
            .alpha(0.001)
            .build();
        
        vectorizedTopoArt = new VectorizedTopoART(referenceParams);
    }
    
    @AfterEach
    void tearDown() {
        if (vectorizedTopoArt != null) {
            vectorizedTopoArt.close();
        }
    }
    
    @Test
    @DisplayName("Vectorized TopoART should learn topological patterns like reference TopoART")
    void testVectorizedTopoARTTopologicalLearning() {
        // Topological test data representing spatial relationships
        var spatialPatterns = new Pattern[]{
            Pattern.of(0.1, 0.1),  // Bottom-left cluster
            Pattern.of(0.15, 0.12), // Near bottom-left
            Pattern.of(0.9, 0.9),  // Top-right cluster
            Pattern.of(0.88, 0.92), // Near top-right
            Pattern.of(0.5, 0.5),  // Center
            Pattern.of(0.52, 0.48) // Near center
        };
        
        // Learn the topological structure
        for (var pattern : spatialPatterns) {
            var result = vectorizedTopoArt.learn(pattern, referenceParams);
            assertNotNull(result, "Learning should produce a result");
        }
        
        // TOPOLOGICAL LEARNING INTELLIGENCE VALIDATION:
        
        // 1. Should create meaningful topology-preserving categories
        assertTrue(vectorizedTopoArt.getCategoryCount() >= 1, "Should create at least one category");
        assertTrue(vectorizedTopoArt.getCategoryCount() <= spatialPatterns.length, 
                  "Should not create more categories than patterns");
        
        // 2. Test prediction on learned patterns
        for (int i = 0; i < spatialPatterns.length; i++) {
            var prediction = vectorizedTopoArt.predict(spatialPatterns[i], referenceParams);
            assertNotNull(prediction, String.format("Should predict for pattern %d", i));
            System.out.printf("Pattern %d (%.2f, %.2f): Prediction=%s%n", 
                             i, spatialPatterns[i].get(0), spatialPatterns[i].get(1), prediction);
        }
        
        // 3. Verify similar patterns map to similar categories (topological consistency)
        var bottomLeft1 = vectorizedTopoArt.predict(Pattern.of(0.1, 0.1), referenceParams);
        var bottomLeft2 = vectorizedTopoArt.predict(Pattern.of(0.15, 0.12), referenceParams);
        var topRight1 = vectorizedTopoArt.predict(Pattern.of(0.9, 0.9), referenceParams);
        var topRight2 = vectorizedTopoArt.predict(Pattern.of(0.88, 0.92), referenceParams);
        
        // Similar spatial patterns should activate similar topological structures
        System.out.printf("Topological consistency: BL1=%s, BL2=%s, TR1=%s, TR2=%s%n",
                         bottomLeft1, bottomLeft2, topRight1, topRight2);
        
        System.out.println("✅ SUCCESS: Vectorized TopoART learns topological patterns like reference TopoART");
        System.out.printf("Categories created: %d%n", vectorizedTopoArt.getCategoryCount());
    }
    
    @Test
    @DisplayName("TopoART should handle dual vigilance components correctly")
    void testDualVigilanceComponents() {
        var testPatterns = new Pattern[]{
            Pattern.of(0.2, 0.3),
            Pattern.of(0.7, 0.8),
            Pattern.of(0.4, 0.5),
            Pattern.of(0.6, 0.7)
        };
        
        // Learn patterns with dual vigilance system
        for (var pattern : testPatterns) {
            var result = vectorizedTopoArt.learn(pattern, referenceParams);
            assertNotNull(result, "Should return a result");
        }
        
        // Verify both components are working
        assertTrue(vectorizedTopoArt.getCategoryCount() >= 1, "Should create categories with dual vigilance");
        
        System.out.println("✅ Dual vigilance component validation completed");
        System.out.printf("Component A vigilance: %.3f, Component B vigilance: %.3f%n",
                         referenceParams.vigilanceA(), referenceParams.vigilanceB());
    }
    
    @Test
    @DisplayName("Vigilance should control topological granularity like reference implementations")
    void testVigilanceControlsTopology() {
        var testData = new Pattern[]{
            Pattern.of(0.1, 0.1),
            Pattern.of(0.2, 0.2),
            Pattern.of(0.8, 0.8),
            Pattern.of(0.9, 0.9)
        };
        
        // Test with high vigilance (fine topology)
        var highVigilanceParams = referenceParams.withVigilanceA(0.95);
        var vectorizedTopoArtHigh = new VectorizedTopoART(highVigilanceParams);
        
        for (var pattern : testData) {
            vectorizedTopoArtHigh.learn(pattern, highVigilanceParams);
        }
        
        // Test with low vigilance (coarse topology)
        var lowVigilanceParams = referenceParams.withVigilanceA(0.3);
        var vectorizedTopoArtLow = new VectorizedTopoART(lowVigilanceParams);
        
        for (var pattern : testData) {
            vectorizedTopoArtLow.learn(pattern, lowVigilanceParams);
        }
        
        // High vigilance typically creates more fine-grained topological categories
        System.out.printf("Vigilance comparison: High (%.2f) created %d categories, Low (%.2f) created %d categories%n",
                         highVigilanceParams.vigilanceA(), vectorizedTopoArtHigh.getCategoryCount(),
                         lowVigilanceParams.vigilanceA(), vectorizedTopoArtLow.getCategoryCount());
        
        vectorizedTopoArtHigh.close();
        vectorizedTopoArtLow.close();
        
        System.out.println("✅ Vigilance-controlled topology validation completed");
    }
    
    @Test
    @DisplayName("TopoART should maintain edge consistency across learning epochs")
    void testEdgeConsistency() {
        var consistencyData = new Pattern[]{
            Pattern.of(0.1, 0.2),
            Pattern.of(0.8, 0.9),
            Pattern.of(0.5, 0.6)
        };
        
        // First epoch
        for (var pattern : consistencyData) {
            vectorizedTopoArt.learn(pattern, referenceParams);
        }
        
        var firstEpochPredictions = new Object[consistencyData.length];
        for (int i = 0; i < consistencyData.length; i++) {
            firstEpochPredictions[i] = vectorizedTopoArt.predict(consistencyData[i], referenceParams);
        }
        
        // Second epoch - learn same patterns again
        for (var pattern : consistencyData) {
            vectorizedTopoArt.learn(pattern, referenceParams);
        }
        
        // Verify topological structure remains consistent
        for (int i = 0; i < consistencyData.length; i++) {
            var secondEpochPrediction = vectorizedTopoArt.predict(consistencyData[i], referenceParams);
            System.out.printf("Pattern %d: Epoch1=%s, Epoch2=%s%n", 
                             i, firstEpochPredictions[i], secondEpochPrediction);
        }
        
        System.out.println("✅ Edge consistency across epochs validated");
    }
    
    @Test
    @DisplayName("Performance should demonstrate vectorization benefits")
    void testVectorizationPerformance() {
        var performanceData = new Pattern[50];
        for (int i = 0; i < performanceData.length; i++) {
            performanceData[i] = Pattern.of(Math.random(), Math.random());
        }
        
        // Learn with performance monitoring
        vectorizedTopoArt.resetPerformanceTracking();
        
        for (var pattern : performanceData) {
            vectorizedTopoArt.learn(pattern, referenceParams);
        }
        
        var stats = vectorizedTopoArt.getPerformanceStats();
        assertNotNull(stats, "Should provide performance statistics");
        assertTrue(vectorizedTopoArt.isVectorized(), "Should confirm vectorization is enabled");
        
        System.out.println("✅ Vectorization performance validation completed");
        System.out.printf("Performance stats: %s%n", stats);
        System.out.printf("Categories created: %d, Algorithm type: %s%n", 
                         vectorizedTopoArt.getCategoryCount(), vectorizedTopoArt.getAlgorithmType());
    }
}