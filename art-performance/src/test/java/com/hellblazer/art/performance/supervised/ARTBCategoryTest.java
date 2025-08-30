package com.hellblazer.art.performance.supervised;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.performance.algorithms.VectorizedART;
import com.hellblazer.art.performance.algorithms.VectorizedParameters;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class ARTBCategoryTest {
    
    @Test
    void testARTBCategoryCreation() {
        var params = VectorizedParameters.createDefault().withVigilance(0.92);
        var artB = new VectorizedART(params);
        
        // Test with one-hot encoded patterns for proper category separation
        var patterns = new Pattern[] {
            Pattern.of(1.0, 0.0, 0.0, 0.0, 0.0),  // Class 0
            Pattern.of(0.0, 1.0, 0.0, 0.0, 0.0),  // Class 1
            Pattern.of(0.0, 0.0, 1.0, 0.0, 0.0),  // Class 2
            Pattern.of(0.0, 0.0, 0.0, 1.0, 0.0),  // Class 3
            Pattern.of(0.0, 0.0, 0.0, 0.0, 1.0)   // Class 4
        };
        
        System.out.println("Testing ARTb category creation with one-hot encoded patterns:");
        for (var pattern : patterns) {
            var result = artB.stepFitEnhanced(pattern, params);
            System.out.printf("Pattern %s -> Category count: %d, Result: %s%n", 
                pattern, artB.getCategoryCount(), result);
        }
        
        // ARTb should create different categories for different one-hot patterns
        assertTrue(artB.getCategoryCount() > 1, 
            "ARTb should create multiple categories for different one-hot patterns, got: " + artB.getCategoryCount());
    }
    
    @Test
    void testComplementCodingEffect() {
        var params = VectorizedParameters.createDefault().withVigilance(0.92);
        var artB = new VectorizedART(params);
        
        // Test with one-hot encoded patterns to demonstrate proper category separation
        var value1 = Pattern.of(1.0, 0.0, 0.0);  // Class 0
        var value2 = Pattern.of(0.0, 1.0, 0.0);  // Class 1
        var value3 = Pattern.of(0.0, 0.0, 1.0);  // Class 2
        
        System.out.println("\\nComplement coding analysis with one-hot encoding:");
        System.out.printf("Class 0: raw=%s, dimension=%d%n", value1, value1.dimension());
        System.out.printf("Class 1: raw=%s, dimension=%d%n", value2, value2.dimension());
        System.out.printf("Class 2: raw=%s, dimension=%d%n", value3, value3.dimension());
        
        // Train on these values
        var result1 = artB.stepFitEnhanced(value1, params);
        var result2 = artB.stepFitEnhanced(value2, params);
        var result3 = artB.stepFitEnhanced(value3, params);
        
        System.out.printf("After training: Category count = %d%n", artB.getCategoryCount());
        System.out.printf("Result1: %s%n", result1);
        System.out.printf("Result2: %s%n", result2);
        System.out.printf("Result3: %s%n", result3);
        
        // Check if distinct categories were created
        assertTrue(artB.getCategoryCount() >= 3, 
            "Should create at least 3 categories for 3 distinct one-hot patterns, got: " + artB.getCategoryCount());
    }
}