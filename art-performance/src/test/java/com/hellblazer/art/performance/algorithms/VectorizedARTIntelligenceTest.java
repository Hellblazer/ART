package com.hellblazer.art.performance.algorithms;

import com.hellblazer.art.core.Pattern;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the actual INTELLIGENCE and LEARNING BEHAVIOR of ART algorithms.
 * These tests verify that the algorithms demonstrate adaptive learning,
 * pattern recognition, and the stability-plasticity dilemma resolution.
 */
class VectorizedARTIntelligenceTest {

    @Test
    @DisplayName("AI Test: Does FuzzyART learn to distinguish clusters vs noise?")
    void testClusterVsNoiseDistinction() {
        var params = new VectorizedParameters(
            0.8,    // High vigilance for good discrimination
            0.1,    // Slow learning for stability
            0.01,   // Small alpha
            1, 100, 1000, true, false, 1e-6
        );
        
        try (var fuzzyArt = new VectorizedFuzzyART(params)) {
            // Create two distinct clusters in 2D space
            var cluster1 = List.of(
                Pattern.of(0.1, 0.1), Pattern.of(0.15, 0.12), Pattern.of(0.08, 0.13),
                Pattern.of(0.12, 0.09), Pattern.of(0.11, 0.14)
            );
            var cluster2 = List.of(
                Pattern.of(0.8, 0.9), Pattern.of(0.82, 0.88), Pattern.of(0.79, 0.91),
                Pattern.of(0.83, 0.87), Pattern.of(0.81, 0.92)
            );
            
            // Add random noise patterns
            var random = new Random(42);
            var noise = new ArrayList<Pattern>();
            for (int i = 0; i < 5; i++) {
                noise.add(Pattern.of(random.nextDouble(), random.nextDouble()));
            }
            
            // Learn all patterns
            for (var pattern : cluster1) fuzzyArt.learn(pattern, params);
            for (var pattern : cluster2) fuzzyArt.learn(pattern, params);  
            for (var pattern : noise) fuzzyArt.learn(pattern, params);
            
            int totalCategories = fuzzyArt.getCategoryCount();
            
            // AI BEHAVIOR TEST: Should create ~2-7 categories
            // 2 for tight clusters + a few for noise, but NOT 15 (one per pattern)
            assertTrue(totalCategories >= 2, "Should recognize at least the 2 main clusters");
            assertTrue(totalCategories <= 10, "Should generalize, not memorize every pattern");
            
            // Test generalization: new patterns similar to cluster centers should be recognized
            var newCluster1Pattern = Pattern.of(0.105, 0.115); // Between cluster1 patterns
            var newCluster2Pattern = Pattern.of(0.805, 0.895); // Between cluster2 patterns
            
            var prediction1 = fuzzyArt.predict(newCluster1Pattern, params);
            var prediction2 = fuzzyArt.predict(newCluster2Pattern, params);
            
            assertNotNull(prediction1, "Should recognize pattern similar to learned cluster");
            assertNotNull(prediction2, "Should recognize pattern similar to learned cluster");
        }
    }

    @Test  
    @DisplayName("AI Test: Vigilance parameter controls learning granularity (ART's key insight)")
    void testVigilanceControlsLearning() {
        // LOW vigilance - should create FEW, BROAD categories  
        var lowVigilance = new VectorizedParameters(0.3, 0.1, 0.01, 1, 100, 1000, true, false, 1e-6);
        // HIGH vigilance - should create MANY, SPECIFIC categories
        var highVigilance = new VectorizedParameters(0.95, 0.1, 0.01, 1, 100, 1000, true, false, 1e-6);
        
        // Same learning data for both
        var patterns = List.of(
            Pattern.of(0.1, 0.1), Pattern.of(0.2, 0.2), Pattern.of(0.3, 0.3),
            Pattern.of(0.4, 0.4), Pattern.of(0.5, 0.5), Pattern.of(0.6, 0.6)
        );
        
        int lowVigilanceCategories, highVigilanceCategories;
        
        try (var lowVigArt = new VectorizedFuzzyART(lowVigilance)) {
            for (var pattern : patterns) lowVigArt.learn(pattern, lowVigilance);
            lowVigilanceCategories = lowVigArt.getCategoryCount();
        }
        
        try (var highVigArt = new VectorizedFuzzyART(highVigilance)) {
            for (var pattern : patterns) highVigArt.learn(pattern, highVigilance);  
            highVigilanceCategories = highVigArt.getCategoryCount();
        }
        
        // AI BEHAVIOR TEST: Vigilance should control granularity
        assertTrue(lowVigilanceCategories < highVigilanceCategories, 
            String.format("Low vigilance (%d categories) should create fewer categories than high vigilance (%d categories)", 
                lowVigilanceCategories, highVigilanceCategories));
        
        assertTrue(lowVigilanceCategories <= 3, "Low vigilance should create broad categories");
        assertTrue(highVigilanceCategories >= 4, "High vigilance should create specific categories");
    }

    @Test
    @DisplayName("AI Test: Does learning order affect final representation? (Stability-Plasticity)")
    void testStabilityPlasticityDilemma() {
        var params = new VectorizedParameters(0.7, 0.1, 0.01, 1, 100, 1000, true, false, 1e-6);
        
        // Same patterns in different orders
        var patterns = List.of(
            Pattern.of(0.1, 0.9), Pattern.of(0.9, 0.1), Pattern.of(0.5, 0.5),
            Pattern.of(0.2, 0.8), Pattern.of(0.8, 0.2)
        );
        
        int categories1, categories2;
        
        // Learn in original order
        try (var art1 = new VectorizedFuzzyART(params)) {
            for (var pattern : patterns) art1.learn(pattern, params);
            categories1 = art1.getCategoryCount();
        }
        
        // Learn in reverse order  
        try (var art2 = new VectorizedFuzzyART(params)) {
            for (int i = patterns.size() - 1; i >= 0; i--) {
                art2.learn(patterns.get(i), params);
            }
            categories2 = art2.getCategoryCount();
        }
        
        // AI BEHAVIOR TEST: Should be relatively stable to learning order
        // (Perfect stability impossible, but shouldn't be wildly different)
        int difference = Math.abs(categories1 - categories2);
        assertTrue(difference <= 2, 
            String.format("Learning order shouldn't drastically change representation (diff=%d)", difference));
    }

    @Test
    @DisplayName("AI Test: Can HypersphereART handle overlapping clusters intelligently?")
    void testOverlappingClustersIntelligence() {
        var params = VectorizedHypersphereParameters.conservative(2);
        
        try (var hypersphereArt = new VectorizedHypersphereART(params)) {
            // Create overlapping clusters - this is where intelligence matters
            var cluster1Center = List.of(
                Pattern.of(0.3, 0.3), Pattern.of(0.32, 0.31), Pattern.of(0.29, 0.33)
            );
            var cluster2Center = List.of(  
                Pattern.of(0.35, 0.35), Pattern.of(0.37, 0.36), Pattern.of(0.34, 0.38)
            );
            var farPattern = Pattern.of(0.8, 0.8); // Definitely separate
            
            // Learn patterns
            for (var pattern : cluster1Center) hypersphereArt.learn(pattern);
            for (var pattern : cluster2Center) hypersphereArt.learn(pattern);
            hypersphereArt.learn(farPattern);
            
            int totalCategories = hypersphereArt.getCategoryCount();
            
            // AI BEHAVIOR TEST: Should handle overlap intelligently
            // Could merge overlapping clusters OR keep separate - both are valid AI decisions
            assertTrue(totalCategories >= 2, "Should recognize at least separate + overlapping regions");
            assertTrue(totalCategories <= 5, "Shouldn't over-fragment overlapping regions");
            
            // Test boundary decision: pattern right between the clusters
            var boundaryPattern = Pattern.of(0.325, 0.325);
            var prediction = hypersphereArt.predict(boundaryPattern, params);
            assertNotNull(prediction, "Should make a decision about boundary patterns");
        }
    }

    @Test
    @DisplayName("AI Test: Progressive learning - does performance improve with more data?")
    void testProgressiveLearning() {
        var params = new VectorizedParameters(0.6, 0.1, 0.01, 1, 100, 1000, true, false, 1e-6);
        
        try (var fuzzyArt = new VectorizedFuzzyART(params)) {
            // Create a clear pattern: XOR-like problem
            var trainingData = List.of(
                Pattern.of(0.0, 0.0), Pattern.of(0.0, 0.1), Pattern.of(0.1, 0.0), // Group A
                Pattern.of(0.9, 0.9), Pattern.of(0.9, 1.0), Pattern.of(1.0, 0.9)  // Group B  
            );
            
            int categoriesAfter3 = 0, categoriesAfter6 = 0;
            
            // Learn first half
            for (int i = 0; i < 3; i++) {
                fuzzyArt.learn(trainingData.get(i), params);
            }
            categoriesAfter3 = fuzzyArt.getCategoryCount();
            
            // Learn second half
            for (int i = 3; i < 6; i++) {
                fuzzyArt.learn(trainingData.get(i), params);
            }
            categoriesAfter6 = fuzzyArt.getCategoryCount();
            
            // AI BEHAVIOR TEST: Should discover the second cluster
            assertTrue(categoriesAfter6 > categoriesAfter3, 
                "Should discover new patterns as more data becomes available");
            
            // Test if it learned the pattern structure
            var testGroupA = Pattern.of(0.05, 0.05);
            var testGroupB = Pattern.of(0.95, 0.95); 
            
            var predA = fuzzyArt.predict(testGroupA, params);
            var predB = fuzzyArt.predict(testGroupB, params);
            
            assertNotNull(predA, "Should recognize patterns similar to learned Group A");
            assertNotNull(predB, "Should recognize patterns similar to learned Group B");
        }
    }
}