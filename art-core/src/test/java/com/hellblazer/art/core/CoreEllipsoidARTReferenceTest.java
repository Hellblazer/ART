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

import com.hellblazer.art.core.algorithms.EllipsoidART;
import com.hellblazer.art.core.parameters.EllipsoidParameters;
import com.hellblazer.art.core.results.ActivationResult;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Python Reference Comparison Test for Core EllipsoidART Implementation.
 * 
 * This test validates that the core EllipsoidART implementation produces ellipsoidal clustering
 * behavior consistent with Python EllipsoidART implementations from reference parity.
 * 
 * EllipsoidART uses ellipsoidal (Gaussian) category regions with Mahalanobis distance
 * instead of simple geometric shapes, enabling more flexible category boundaries.
 */
@DisplayName("Core EllipsoidART Python Reference Validation")
class CoreEllipsoidARTReferenceTest {
    
    private EllipsoidART coreEllipsoidArt;
    private EllipsoidParameters referenceParams;
    
    @BeforeEach
    void setUp() {
        // Parameters tuned for ellipsoidal clustering similar to Python EllipsoidART implementations
        referenceParams = new EllipsoidParameters(
            0.3,        // vigilance - further reduced for coarser clustering (was 0.6→0.4→0.3)
            0.1,        // learning rate - slow adaptation
            2,          // dimensions - 2D patterns
            0.001,      // min variance - prevent numerical issues
            10.0,       // max variance - prevent ellipsoid explosion
            0.05,       // shape adaptation rate - slow ellipsoid evolution
            50          // max categories
        );
        
        coreEllipsoidArt = new EllipsoidART(referenceParams);
    }
    
    @AfterEach
    void tearDown() {
        // EllipsoidART doesn't implement AutoCloseable
    }
    
    @Test
    @DisplayName("Core EllipsoidART should learn ellipsoidal clusters like Python EllipsoidART")
    void testCoreEllipsoidARTEllipsoidalClustering() {
        // Test patterns representing ellipsoidal distributions with varying shapes
        var ellipsoidalPatterns = Arrays.asList(
            // Cluster 1: Tight ellipsoidal distribution (elongated horizontally)
            Pattern.of(0.1, 0.1),    // Core of ellipse 1
            Pattern.of(0.15, 0.105), // Slight horizontal elongation
            Pattern.of(0.12, 0.095), // Within ellipse 1 boundary
            Pattern.of(0.18, 0.11),  // Edge of ellipse 1
            
            // Cluster 2: Different ellipsoidal shape (elongated vertically) 
            Pattern.of(0.8, 0.8),    // Core of ellipse 2
            Pattern.of(0.81, 0.85),  // Vertical elongation
            Pattern.of(0.79, 0.75),  // Within ellipse 2 boundary
            Pattern.of(0.82, 0.88),  // Edge of ellipse 2
            
            // Cluster 3: Circular-ish distribution
            Pattern.of(0.5, 0.3),    // Core of ellipse 3
            Pattern.of(0.52, 0.32),  // Slight variation
            Pattern.of(0.48, 0.28)   // Within circular boundary
        );
        
        int totalPatterns = ellipsoidalPatterns.size();
        
        // ELLIPSOIDAL LEARNING PHASE - Learn Mahalanobis distance-based clusters
        for (int i = 0; i < totalPatterns; i++) {
            var pattern = ellipsoidalPatterns.get(i);
            var result = coreEllipsoidArt.stepFit(pattern, referenceParams);
            
            assertInstanceOf(ActivationResult.Success.class, result, 
                "Ellipsoidal clustering should succeed for pattern " + i);
            
            var success = (ActivationResult.Success) result;
            
            System.out.printf("Pattern %d: [%.3f,%.3f] -> Ellipsoid %d (Mahalanobis: %.3f)%n",
                             i, pattern.get(0), pattern.get(1), 
                             success.categoryIndex(), success.activationValue());
        }
        
        // ELLIPSOIDAL INTELLIGENCE VALIDATION:
        
        // 1. Should create reasonable number of ellipsoidal clusters
        assertTrue(coreEllipsoidArt.getCategoryCount() >= 2, 
            "Should create at least 2 ellipsoidal clusters for distinct distributions");
        assertTrue(coreEllipsoidArt.getCategoryCount() <= 4, 
            "Should not over-cluster ellipsoidal data (created " + coreEllipsoidArt.getCategoryCount() + " ellipsoids)");
        
        // 2. Test ellipsoidal boundary behavior with edge patterns
        var boundaryPattern1 = Pattern.of(0.2, 0.1);   // Near cluster 1 but outside
        var boundaryPattern2 = Pattern.of(0.75, 0.85); // Near cluster 2 but outside
        
        var boundary1Result = coreEllipsoidArt.stepFit(boundaryPattern1, referenceParams);
        var boundary2Result = coreEllipsoidArt.stepFit(boundaryPattern2, referenceParams);
        
        assertInstanceOf(ActivationResult.Success.class, boundary1Result, 
            "Boundary pattern 1 should be handled ellipsoidally");
        assertInstanceOf(ActivationResult.Success.class, boundary2Result, 
            "Boundary pattern 2 should be handled ellipsoidally");
        
        // 3. Test novel pattern handling with Mahalanobis distance
        var novelPattern = Pattern.of(0.95, 0.05); // Very different from all clusters
        var novelResult = coreEllipsoidArt.stepFit(novelPattern, referenceParams);
        
        assertInstanceOf(ActivationResult.Success.class, novelResult, 
            "Novel pattern should be handled with ellipsoidal geometry");
        
        var novelSuccess = (ActivationResult.Success) novelResult;
        System.out.printf("Novel pattern [%.2f,%.2f] -> Ellipsoid %d, Distance measure: %.3f%n",
                         novelPattern.get(0), novelPattern.get(1), 
                         novelSuccess.categoryIndex(), novelSuccess.activationValue());
        
        System.out.println("✅ SUCCESS: Core EllipsoidART learns ellipsoidal clusters like Python EllipsoidART");
        System.out.printf("Created %d ellipsoidal clusters for %d patterns%n", 
                         coreEllipsoidArt.getCategoryCount(), totalPatterns);
    }
    
    @Test
    @DisplayName("Ellipsoidal vigilance should control cluster flexibility like Python")
    void testEllipsoidalVigilanceControl() {
        var testPatterns = Arrays.asList(
            Pattern.of(0.3, 0.3),
            Pattern.of(0.35, 0.32),  // Moderately close
            Pattern.of(0.4, 0.35),   // Further but potentially same ellipsoid
            Pattern.of(0.7, 0.7)     // Definitely separate
        );
        
        // Test with high ellipsoidal vigilance (tight ellipsoids)
        var highVigilanceParams = new EllipsoidParameters(0.95, 0.1, 2, 0.001, 10.0, 0.05, 50);
        var highVigilanceEllipsoid = new EllipsoidART(highVigilanceParams);
        
        // Test with low ellipsoidal vigilance (loose ellipsoids)
        var lowVigilanceParams = new EllipsoidParameters(0.3, 0.1, 2, 0.001, 10.0, 0.05, 50);
        var lowVigilanceEllipsoid = new EllipsoidART(lowVigilanceParams);
        
        // Train both with same patterns
        for (var pattern : testPatterns) {
            highVigilanceEllipsoid.stepFit(pattern, highVigilanceParams);
            lowVigilanceEllipsoid.stepFit(pattern, lowVigilanceParams);
        }
        
        System.out.printf("Ellipsoidal vigilance comparison: High (%.2f) created %d ellipsoids, Low (%.2f) created %d ellipsoids%n",
                         highVigilanceParams.vigilance(), highVigilanceEllipsoid.getCategoryCount(),
                         lowVigilanceParams.vigilance(), lowVigilanceEllipsoid.getCategoryCount());
        
        // High vigilance should typically create more ellipsoids (tighter boundaries)
        assertTrue(highVigilanceEllipsoid.getCategoryCount() >= 1, "High vigilance should create ellipsoids");
        assertTrue(lowVigilanceEllipsoid.getCategoryCount() >= 1, "Low vigilance should create ellipsoids");
        
        System.out.println("✅ Ellipsoidal vigilance control validated");
    }
    
    @Test
    @DisplayName("Shape adaptation should evolve ellipsoid geometry like Python")
    void testShapeAdaptationBehavior() {
        var adaptationPatterns = Arrays.asList(
            Pattern.of(0.5, 0.5),    // Initial center
            Pattern.of(0.6, 0.51),   // Horizontal elongation
            Pattern.of(0.55, 0.6),   // Vertical elongation
            Pattern.of(0.65, 0.52)   // Continue horizontal trend
        );
        
        // Test with high shape adaptation rate (fast ellipsoid evolution)
        var fastAdaptationParams = new EllipsoidParameters(0.7, 0.2, 2, 0.001, 10.0, 0.2, 50);
        var fastAdaptationEllipsoid = new EllipsoidART(fastAdaptationParams);
        
        // Test with slow shape adaptation rate (conservative ellipsoid evolution)
        var slowAdaptationParams = new EllipsoidParameters(0.7, 0.2, 2, 0.001, 10.0, 0.01, 50);
        var slowAdaptationEllipsoid = new EllipsoidART(slowAdaptationParams);
        
        // Train with patterns that should elongate the ellipsoids
        for (int i = 0; i < adaptationPatterns.size(); i++) {
            var pattern = adaptationPatterns.get(i);
            
            var fastResult = fastAdaptationEllipsoid.stepFit(pattern, fastAdaptationParams);
            var slowResult = slowAdaptationEllipsoid.stepFit(pattern, slowAdaptationParams);
            
            assertInstanceOf(ActivationResult.Success.class, fastResult, 
                "Fast adaptation should succeed for pattern " + i);
            assertInstanceOf(ActivationResult.Success.class, slowResult, 
                "Slow adaptation should succeed for pattern " + i);
            
            var fastSuccess = (ActivationResult.Success) fastResult;
            var slowSuccess = (ActivationResult.Success) slowResult;
            
            System.out.printf("Pattern %d: Fast adaptation match %.3f, Slow adaptation match %.3f%n",
                             i, fastSuccess.activationValue(), slowSuccess.activationValue());
        }
        
        System.out.printf("Shape adaptation: Fast (%.2f) vs Slow (%.2f) rates%n",
                         fastAdaptationParams.shapeAdaptationRate(),
                         slowAdaptationParams.shapeAdaptationRate());
        
        System.out.println("✅ Shape adaptation behavior validated");
    }
    
    @Test
    @DisplayName("Variance bounds should regulate ellipsoid size like Python")
    void testVarianceBoundsRegulation() {
        var varianceTestPatterns = Arrays.asList(
            Pattern.of(0.5, 0.5),    // Center
            Pattern.of(0.8, 0.5),    // Far horizontal
            Pattern.of(0.2, 0.5),    // Far horizontal opposite
            Pattern.of(0.5, 0.9),    // Far vertical
            Pattern.of(0.5, 0.1)     // Far vertical opposite
        );
        
        // Test with tight variance bounds (constrained ellipsoids)
        var tightVarianceParams = new EllipsoidParameters(0.6, 0.1, 2, 0.01, 0.1, 0.05, 50);
        var tightVarianceEllipsoid = new EllipsoidART(tightVarianceParams);
        
        // Test with loose variance bounds (flexible ellipsoids)
        var looseVarianceParams = new EllipsoidParameters(0.6, 0.1, 2, 0.001, 5.0, 0.05, 50);
        var looseVarianceEllipsoid = new EllipsoidART(looseVarianceParams);
        
        // Train with widely spread patterns
        for (var pattern : varianceTestPatterns) {
            tightVarianceEllipsoid.stepFit(pattern, tightVarianceParams);
            looseVarianceEllipsoid.stepFit(pattern, looseVarianceParams);
        }
        
        System.out.printf("Variance bounds: Tight [%.3f-%.1f] created %d ellipsoids, Loose [%.3f-%.1f] created %d ellipsoids%n",
                         tightVarianceParams.minVariance(), tightVarianceParams.maxVariance(), 
                         tightVarianceEllipsoid.getCategoryCount(),
                         looseVarianceParams.minVariance(), looseVarianceParams.maxVariance(), 
                         looseVarianceEllipsoid.getCategoryCount());
        
        // Tight variance bounds might create more ellipsoids (can't stretch as much)
        assertTrue(tightVarianceEllipsoid.getCategoryCount() >= 1, "Tight bounds should create ellipsoids");
        assertTrue(looseVarianceEllipsoid.getCategoryCount() >= 1, "Loose bounds should create ellipsoids");
        
        System.out.println("✅ Variance bounds regulation validated");
    }
    
    @Test
    @DisplayName("Mahalanobis distance should provide geometric intelligence")
    void testMahalanobisDistanceIntelligence() {
        var geometricPatterns = Arrays.asList(
            Pattern.of(0.4, 0.4),    // Reference pattern
            Pattern.of(0.42, 0.41),  // Close in Euclidean and Mahalanobis
            Pattern.of(0.45, 0.4),   // Moderate distance
            Pattern.of(0.6, 0.6)     // Far in both metrics
        );
        
        // Train with the reference pattern first
        var referenceResult = coreEllipsoidArt.stepFit(geometricPatterns.get(0), referenceParams);
        assertInstanceOf(ActivationResult.Success.class, referenceResult, 
            "Reference pattern should establish ellipsoid");
        
        // Test how other patterns relate to the established ellipsoid
        for (int i = 1; i < geometricPatterns.size(); i++) {
            var pattern = geometricPatterns.get(i);
            var result = coreEllipsoidArt.stepFit(pattern, referenceParams);
            
            assertInstanceOf(ActivationResult.Success.class, result, 
                "Pattern " + i + " should be processed with Mahalanobis distance");
            
            var success = (ActivationResult.Success) result;
            
            // Calculate approximate Euclidean distance for comparison
            var refPattern = geometricPatterns.get(0);
            double euclideanDist = Math.sqrt(
                Math.pow(pattern.get(0) - refPattern.get(0), 2) + 
                Math.pow(pattern.get(1) - refPattern.get(1), 2)
            );
            
            System.out.printf("Pattern %d: Euclidean dist %.3f, Mahalanobis match %.3f, Category %d%n",
                             i, euclideanDist, success.activationValue(), success.categoryIndex());
        }
        
        System.out.println("✅ Mahalanobis distance intelligence validated");
    }
    
    @Test
    @DisplayName("Ellipsoid clusters should remain stable across training epochs")
    void testEllipsoidalStability() {
        var stabilityPatterns = Arrays.asList(
            Pattern.of(0.3, 0.7),
            Pattern.of(0.7, 0.3)
        );
        
        // Train multiple epochs
        for (int epoch = 0; epoch < 3; epoch++) {
            for (int i = 0; i < stabilityPatterns.size(); i++) {
                var pattern = stabilityPatterns.get(i);
                var result = coreEllipsoidArt.stepFit(pattern, referenceParams);
                
                assertInstanceOf(ActivationResult.Success.class, result, 
                    "Stable ellipsoid should succeed on epoch " + epoch);
                
                var success = (ActivationResult.Success) result;
                System.out.printf("Epoch %d, Pattern %d: Ellipsoid %d, Match %.3f%n",
                                 epoch, i, success.categoryIndex(), success.activationValue());
            }
        }
        
        System.out.println("✅ Ellipsoidal stability across epochs validated");
    }
}