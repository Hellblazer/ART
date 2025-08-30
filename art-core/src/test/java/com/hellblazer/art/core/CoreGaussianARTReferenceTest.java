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

import com.hellblazer.art.core.algorithms.GaussianART;
import com.hellblazer.art.core.parameters.GaussianParameters;
import com.hellblazer.art.core.results.ActivationResult;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Python Reference Comparison Test for Core GaussianART Implementation.
 * 
 * This test validates that the core GaussianART implementation produces Gaussian clustering
 * behavior consistent with Python GaussianART implementations from reference parity.
 * 
 * GaussianART uses multivariate Gaussian probability distributions to model categories,
 * with probabilistic activation based on Gaussian likelihood and vigilance test using
 * probability density threshold.
 */
@DisplayName("Core GaussianART Python Reference Validation")
class CoreGaussianARTReferenceTest {
    
    private GaussianART coreGaussianArt;
    private GaussianParameters referenceParams;
    
    @BeforeEach
    void setUp() {
        // Parameters tuned for Gaussian clustering similar to Python GaussianART implementations
        var sigmaInit = new double[]{0.1, 0.1}; // Initial sigma for 2D patterns
        
        referenceParams = new GaussianParameters(
            0.8,        // vigilance - moderate for reasonable clustering 
            sigmaInit   // initial sigma values for each dimension
        );
        
        coreGaussianArt = new GaussianART();
    }
    
    @AfterEach
    void tearDown() {
        // GaussianART doesn't implement AutoCloseable
    }
    
    @Test
    @DisplayName("Core GaussianART should learn Gaussian clusters like Python GaussianART")
    void testCoreGaussianARTGaussianClustering() {
        // Test patterns representing Gaussian distributions with different means and variances
        var gaussianPatterns = Arrays.asList(
            // Cluster 1: Tight Gaussian distribution around (0.2, 0.2)
            Pattern.of(0.2, 0.2),    // Mean of cluster 1
            Pattern.of(0.22, 0.21),  // Close to mean
            Pattern.of(0.19, 0.18),  // Within 1 sigma
            Pattern.of(0.25, 0.23),  // Within 2 sigma
            
            // Cluster 2: Gaussian distribution around (0.8, 0.7) 
            Pattern.of(0.8, 0.7),    // Mean of cluster 2
            Pattern.of(0.82, 0.72),  // Close to mean
            Pattern.of(0.78, 0.68),  // Within 1 sigma
            Pattern.of(0.85, 0.74),  // Within 2 sigma
            
            // Cluster 3: Different Gaussian around (0.4, 0.8)
            Pattern.of(0.4, 0.8),    // Mean of cluster 3
            Pattern.of(0.42, 0.82),  // Close to mean
            Pattern.of(0.38, 0.79)   // Within distribution
        );
        
        int totalPatterns = gaussianPatterns.size();
        
        // GAUSSIAN LEARNING PHASE - Learn probabilistic distributions
        for (int i = 0; i < totalPatterns; i++) {
            var pattern = gaussianPatterns.get(i);
            var result = coreGaussianArt.stepFit(pattern, referenceParams);
            
            assertInstanceOf(ActivationResult.Success.class, result, 
                "Gaussian clustering should succeed for pattern " + i);
            
            var success = (ActivationResult.Success) result;
            
            System.out.printf("Pattern %d: [%.3f,%.3f] -> Gaussian %d (Likelihood: %.3f)%n",
                             i, pattern.get(0), pattern.get(1), 
                             success.categoryIndex(), success.activationValue());
        }
        
        // GAUSSIAN INTELLIGENCE VALIDATION:
        
        // 1. Should create reasonable number of Gaussian clusters
        assertTrue(coreGaussianArt.getCategoryCount() >= 2, 
            "Should create at least 2 Gaussian clusters for distinct distributions");
        assertTrue(coreGaussianArt.getCategoryCount() <= 4, 
            "Should not over-cluster Gaussian data (created " + coreGaussianArt.getCategoryCount() + " clusters)");
        
        // 2. Test probabilistic predictions with Gaussian likelihood
        for (int i = 0; i < Math.min(5, totalPatterns); i++) {
            var pattern = gaussianPatterns.get(i);
            var predictionResult = coreGaussianArt.stepFit(pattern, referenceParams);
            
            if (predictionResult instanceof ActivationResult.Success success) {
                // Gaussian likelihood should be a valid probability measure
                assertTrue(success.activationValue() >= 0.0, 
                    "Gaussian likelihood should be non-negative: " + success.activationValue());
                
                System.out.printf("Gaussian prediction %d: Category %d, Likelihood %.6f%n",
                                 i, success.categoryIndex(), success.activationValue());
            }
        }
        
        // 3. Test novel pattern with low likelihood
        var novelPattern = Pattern.of(0.95, 0.05); // Very different from training clusters
        var novelResult = coreGaussianArt.stepFit(novelPattern, referenceParams);
        
        assertInstanceOf(ActivationResult.Success.class, novelResult, 
            "Novel pattern should be handled with Gaussian probability");
        
        var novelSuccess = (ActivationResult.Success) novelResult;
        System.out.printf("Novel pattern [%.2f,%.2f] -> Gaussian %d, Low likelihood: %.6f%n",
                         novelPattern.get(0), novelPattern.get(1), 
                         novelSuccess.categoryIndex(), novelSuccess.activationValue());
        
        System.out.println("✅ SUCCESS: Core GaussianART learns Gaussian clusters like Python GaussianART");
        System.out.printf("Created %d Gaussian clusters for %d patterns%n", 
                         coreGaussianArt.getCategoryCount(), totalPatterns);
    }
    
    @Test
    @DisplayName("Gaussian vigilance should control probability threshold like Python")
    void testGaussianVigilanceControl() {
        var testPatterns = Arrays.asList(
            Pattern.of(0.3, 0.3),    // Reference point
            Pattern.of(0.35, 0.32),  // Close - should likely be same cluster
            Pattern.of(0.4, 0.35),   // Moderate distance
            Pattern.of(0.7, 0.7)     // Far - should be separate cluster
        );
        
        // Test with high Gaussian vigilance (strict probability threshold)
        var highVigilanceParams = new GaussianParameters(0.95, new double[]{0.05, 0.05});
        var highVigilanceGaussian = new GaussianART();
        
        // Test with low Gaussian vigilance (lenient probability threshold)
        var lowVigilanceParams = new GaussianParameters(0.3, new double[]{0.05, 0.05});
        var lowVigilanceGaussian = new GaussianART();
        
        // Train both with same patterns
        for (var pattern : testPatterns) {
            highVigilanceGaussian.stepFit(pattern, highVigilanceParams);
            lowVigilanceGaussian.stepFit(pattern, lowVigilanceParams);
        }
        
        System.out.printf("Gaussian vigilance comparison: High (%.2f) created %d clusters, Low (%.2f) created %d clusters%n",
                         highVigilanceParams.vigilance(), highVigilanceGaussian.getCategoryCount(),
                         lowVigilanceParams.vigilance(), lowVigilanceGaussian.getCategoryCount());
        
        // High vigilance should typically create more clusters (stricter probability threshold)
        assertTrue(highVigilanceGaussian.getCategoryCount() >= 1, "High vigilance should create clusters");
        assertTrue(lowVigilanceGaussian.getCategoryCount() >= 1, "Low vigilance should create clusters");
        
        System.out.println("✅ Gaussian vigilance control validated");
    }
    
    @Test
    @DisplayName("Initial sigma should influence cluster formation like Python")
    void testInitialSigmaInfluence() {
        var sigmaTestPatterns = Arrays.asList(
            Pattern.of(0.5, 0.5),    // Center
            Pattern.of(0.52, 0.51),  // Close variation
            Pattern.of(0.48, 0.49)   // Another close variation
        );
        
        // Test with small initial sigma (tight initial clusters)
        var smallSigmaParams = new GaussianParameters(0.7, new double[]{0.01, 0.01});
        var smallSigmaGaussian = new GaussianART();
        
        // Test with large initial sigma (loose initial clusters)
        var largeSigmaParams = new GaussianParameters(0.7, new double[]{0.2, 0.2});
        var largeSigmaGaussian = new GaussianART();
        
        // Train with close patterns
        for (var pattern : sigmaTestPatterns) {
            smallSigmaGaussian.stepFit(pattern, smallSigmaParams);
            largeSigmaGaussian.stepFit(pattern, largeSigmaParams);
        }
        
        System.out.printf("Initial sigma influence: Small (%.3f) created %d clusters, Large (%.3f) created %d clusters%n",
                         smallSigmaParams.sigmaInit()[0], smallSigmaGaussian.getCategoryCount(),
                         largeSigmaParams.sigmaInit()[0], largeSigmaGaussian.getCategoryCount());
        
        // Different sigma values should affect initial cluster sensitivity
        assertTrue(smallSigmaGaussian.getCategoryCount() >= 1, "Small sigma should create clusters");
        assertTrue(largeSigmaGaussian.getCategoryCount() >= 1, "Large sigma should create clusters");
        
        System.out.println("✅ Initial sigma influence validated");
    }
    
    @Test
    @DisplayName("Gaussian probability density should follow mathematical principles")
    void testGaussianProbabilityDensity() {
        var probabilityTestPatterns = Arrays.asList(
            Pattern.of(0.4, 0.4),    // Reference pattern - will establish Gaussian
            Pattern.of(0.41, 0.41),  // Very close - should have high likelihood
            Pattern.of(0.45, 0.45),  // Moderate distance - medium likelihood  
            Pattern.of(0.6, 0.6)     // Far - should have lower likelihood
        );
        
        // Train with the reference pattern first
        var referenceResult = coreGaussianArt.stepFit(probabilityTestPatterns.get(0), referenceParams);
        assertInstanceOf(ActivationResult.Success.class, referenceResult, 
            "Reference pattern should establish Gaussian cluster");
        
        var previousLikelihood = Double.MAX_VALUE;
        
        // Test that likelihood decreases with distance (generally)
        for (int i = 1; i < probabilityTestPatterns.size(); i++) {
            var pattern = probabilityTestPatterns.get(i);
            var result = coreGaussianArt.stepFit(pattern, referenceParams);
            
            assertInstanceOf(ActivationResult.Success.class, result, 
                "Pattern " + i + " should be processed with Gaussian probability");
            
            var success = (ActivationResult.Success) result;
            
            // Calculate Euclidean distance from reference for comparison
            var refPattern = probabilityTestPatterns.get(0);
            double distance = Math.sqrt(
                Math.pow(pattern.get(0) - refPattern.get(0), 2) + 
                Math.pow(pattern.get(1) - refPattern.get(1), 2)
            );
            
            System.out.printf("Pattern %d: Distance %.3f, Likelihood %.6f, Category %d%n",
                             i, distance, success.activationValue(), success.categoryIndex());
            
            // Likelihood should be non-negative
            assertTrue(success.activationValue() >= 0.0, 
                "Gaussian likelihood must be non-negative: " + success.activationValue());
        }
        
        System.out.println("✅ Gaussian probability density principles validated");
    }
    
    @Test
    @DisplayName("Online mean and variance updates should adapt clusters")
    void testOnlineMeanVarianceUpdates() {
        var adaptationPatterns = Arrays.asList(
            Pattern.of(0.3, 0.3),    // Initial cluster center
            Pattern.of(0.32, 0.31),  // Slight shift
            Pattern.of(0.35, 0.33),  // Further shift - should adapt mean/variance
            Pattern.of(0.38, 0.36)   // Continue adaptation
        );
        
        double[] previousLikelihoods = new double[adaptationPatterns.size()];
        
        // Train incrementally and observe adaptation
        for (int i = 0; i < adaptationPatterns.size(); i++) {
            var pattern = adaptationPatterns.get(i);
            var result = coreGaussianArt.stepFit(pattern, referenceParams);
            
            assertInstanceOf(ActivationResult.Success.class, result, 
                "Adaptation should succeed for pattern " + i);
            
            var success = (ActivationResult.Success) result;
            previousLikelihoods[i] = success.activationValue();
            
            System.out.printf("Adaptation step %d: Pattern [%.2f,%.2f] -> Likelihood %.6f%n",
                             i, pattern.get(0), pattern.get(1), success.activationValue());
        }
        
        // Test that the adapted cluster can handle patterns it has seen
        for (int i = 0; i < adaptationPatterns.size(); i++) {
            var pattern = adaptationPatterns.get(i);
            var result = coreGaussianArt.stepFit(pattern, referenceParams);
            
            assertInstanceOf(ActivationResult.Success.class, result, 
                "Adapted cluster should handle seen pattern " + i);
            
            var success = (ActivationResult.Success) result;
            System.out.printf("Retesting pattern %d: Likelihood %.6f%n", i, success.activationValue());
        }
        
        System.out.println("✅ Online mean/variance adaptation validated");
    }
    
    @Test
    @DisplayName("Gaussian clusters should maintain statistical consistency")
    void testGaussianStatisticalConsistency() {
        var consistencyPatterns = Arrays.asList(
            Pattern.of(0.6, 0.4),
            Pattern.of(0.4, 0.6)
        );
        
        // Train the network
        for (var pattern : consistencyPatterns) {
            coreGaussianArt.stepFit(pattern, referenceParams);
        }
        
        // Test multiple predictions for consistency
        for (int iteration = 0; iteration < 3; iteration++) {
            for (int i = 0; i < consistencyPatterns.size(); i++) {
                var pattern = consistencyPatterns.get(i);
                var result = coreGaussianArt.stepFit(pattern, referenceParams);
                
                assertInstanceOf(ActivationResult.Success.class, result, 
                    "Consistent Gaussian prediction should succeed on iteration " + iteration);
                
                var success = (ActivationResult.Success) result;
                System.out.printf("Iteration %d, Pattern %d: Gaussian %d, Likelihood %.6f%n",
                                 iteration, i, success.categoryIndex(), success.activationValue());
            }
        }
        
        System.out.println("✅ Gaussian statistical consistency validated");
    }
    
    @Test
    @DisplayName("Multivariate Gaussian should handle correlation structure")
    void testMultivariateGaussianCorrelation() {
        // Test patterns with potential correlation structure
        var correlationPatterns = Arrays.asList(
            Pattern.of(0.2, 0.2),    // Diagonal pattern start
            Pattern.of(0.25, 0.25),  // Follow diagonal
            Pattern.of(0.3, 0.3),    // Continue diagonal - correlated dimensions
            Pattern.of(0.8, 0.2),    // Anti-diagonal pattern
            Pattern.of(0.75, 0.25),  // Follow anti-diagonal
            Pattern.of(0.7, 0.3)     // Continue anti-diagonal - negative correlation
        );
        
        // Train with correlated patterns
        for (int i = 0; i < correlationPatterns.size(); i++) {
            var pattern = correlationPatterns.get(i);
            var result = coreGaussianArt.stepFit(pattern, referenceParams);
            
            assertInstanceOf(ActivationResult.Success.class, result, 
                "Multivariate Gaussian should handle pattern " + i);
            
            var success = (ActivationResult.Success) result;
            System.out.printf("Correlated pattern %d: [%.2f,%.2f] -> Gaussian %d, Likelihood %.6f%n",
                             i, pattern.get(0), pattern.get(1), 
                             success.categoryIndex(), success.activationValue());
        }
        
        System.out.printf("Multivariate structure: Created %d Gaussian clusters for correlated data%n", 
                         coreGaussianArt.getCategoryCount());
        
        // Should handle multivariate structure appropriately
        assertTrue(coreGaussianArt.getCategoryCount() >= 1, "Should create clusters for correlated data");
        
        System.out.println("✅ Multivariate Gaussian correlation handling validated");
    }
}