package com.hellblazer.art.core;

import org.junit.jupiter.api.Test;

public class CVIARTDebugTest {
    
    @Test
    void debugSimilarityCalculation() {
        // Create CVIART instance
        var cviart = new CVIART();
        
        // Create test patterns - EXACTLY as in the failing test
        var pattern1 = new DenseVector(new double[]{0.1, 0.1, 0.1});
        var pattern2 = new DenseVector(new double[]{0.5, 0.5, 0.5});
        var pattern3 = new DenseVector(new double[]{0.9, 0.9, 0.9});
        
        // Create parameters with DEFAULT vigilance as in the failing test
        var params = new CVIART.CVIARTParameters();
        // Use default vigilance of 0.5
        
        // Learn first pattern - should create category 0
        System.out.println("\n=== Learning pattern 1: [0.1, 0.1, 0.1] ===");
        var result1 = cviart.learn(pattern1, params);
        System.out.println("Result: " + result1);
        System.out.println("Categories: " + cviart.getCategoryCount());
        System.out.println("Current vigilance: " + cviart.getCurrentVigilance());
        
        // Learn second pattern - should create category 1 (different from first)
        System.out.println("\n=== Learning pattern 2: [0.5, 0.5, 0.5] ===");
        var result2 = cviart.learn(pattern2, params);
        System.out.println("Result: " + result2);
        System.out.println("Categories: " + cviart.getCategoryCount());
        System.out.println("Current vigilance: " + cviart.getCurrentVigilance());
        
        // Learn third pattern - should create category 2 (different from both)
        System.out.println("\n=== Learning pattern 3: [0.9, 0.9, 0.9] ===");
        var result3 = cviart.learn(pattern3, params);
        System.out.println("Result: " + result3);
        System.out.println("Categories: " + cviart.getCategoryCount());
        System.out.println("Current vigilance: " + cviart.getCurrentVigilance());
        
        // Now let's manually test similarity
        System.out.println("\n=== Manual similarity test ===");
        
        // Test similarity between [0.1, 0.1, 0.1] and [0.5, 0.5, 0.5]
        double sumMin1 = Math.min(0.1, 0.5) + Math.min(0.1, 0.5) + Math.min(0.1, 0.5);
        double sumInput1 = 0.5 + 0.5 + 0.5;
        double similarity1 = sumMin1 / sumInput1;
        System.out.println("Similarity between [0.1,0.1,0.1] and [0.5,0.5,0.5]: " + similarity1);
        System.out.println("Meets vigilance 0.5? " + (similarity1 >= 0.5));
        
        // Test similarity between [0.5, 0.5, 0.5] and [0.9, 0.9, 0.9]
        // After pattern 2 learns, the weight gets UPDATED to be closer to pattern 2
        // Let's assume weight becomes [0.5, 0.5, 0.5] after learning
        double sumMin2 = Math.min(0.5, 0.9) + Math.min(0.5, 0.9) + Math.min(0.5, 0.9);
        double sumInput2 = 0.9 + 0.9 + 0.9;
        double similarity2 = sumMin2 / sumInput2;
        System.out.println("\nSimilarity between [0.5,0.5,0.5] and [0.9,0.9,0.9]: " + similarity2);
        System.out.println("Meets vigilance 0.5? " + (similarity2 >= 0.5));
        
        // sumMin = 0.5 + 0.5 + 0.5 = 1.5
        // sumInput = 0.9 + 0.9 + 0.9 = 2.7
        // similarity = 1.5 / 2.7 = 0.556
        // This is > 0.5, so pattern 3 WILL match category 1!
    }
}