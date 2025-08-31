package com.hellblazer.art.core.algorithms;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.parameters.ART1Parameters;
import com.hellblazer.art.core.results.ActivationResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Simple test for ART1 implementation using BaseART framework.
 */
class ART1SimpleTest {
    
    private ART1 art1;
    private ART1Parameters params;
    
    @BeforeEach
    void setUp() {
        art1 = new ART1();
        params = new ART1Parameters.Builder()
            .vigilance(0.5)
            .L(2.0)
            .build();
    }
    
    @Test
    void testEmptyNetwork() {
        assertEquals(0, art1.getCategoryCount());
        assertTrue(art1.getCategories().isEmpty());
    }
    
    @Test
    void testBinaryPatternValidation() {
        // Valid binary patterns
        var validPattern1 = Pattern.of(1.0, 0.0, 1.0);
        var validPattern2 = Pattern.of(0.0, 0.0, 0.0);
        var validPattern3 = Pattern.of(1.0, 1.0, 1.0);
        
        // Should not throw exceptions
        assertDoesNotThrow(() -> art1.stepFit(validPattern1, params));
        
        // Invalid binary patterns
        var invalidPattern1 = Pattern.of(0.5, 0.0, 1.0);
        var invalidPattern2 = Pattern.of(1.0, 2.0, 0.0);
        var invalidPattern3 = Pattern.of(-1.0, 1.0, 0.0);
        
        // Should throw exceptions for non-binary values
        assertThrows(IllegalArgumentException.class, () -> art1.stepFit(invalidPattern1, params));
        assertThrows(IllegalArgumentException.class, () -> art1.stepFit(invalidPattern2, params));
        assertThrows(IllegalArgumentException.class, () -> art1.stepFit(invalidPattern3, params));
    }
    
    @Test
    void testSinglePatternLearning() {
        var pattern = Pattern.of(1.0, 0.0, 1.0);
        
        // First presentation should create a new category
        var result = art1.stepFit(pattern, params);
        assertNotNull(result);
        assertEquals(1, art1.getCategoryCount());
    }
    
    @Test
    void testMultipleIdenticalPatterns() {
        var pattern = Pattern.of(1.0, 0.0, 1.0);
        
        // Present same pattern multiple times
        art1.stepFit(pattern, params);
        art1.stepFit(pattern, params);
        art1.stepFit(pattern, params);
        
        // Should still have only one category
        assertEquals(1, art1.getCategoryCount());
    }
    
    @Test
    void testDifferentPatterns() {
        var pattern1 = Pattern.of(1.0, 0.0, 1.0);
        var pattern2 = Pattern.of(0.0, 1.0, 0.0);
        
        // Present different patterns
        art1.stepFit(pattern1, params);
        art1.stepFit(pattern2, params);
        
        // Should create separate categories
        assertEquals(2, art1.getCategoryCount());
    }
    
    @Test
    void testVigilanceEffect() {
        // Low vigilance - should group patterns together
        var lowVigilanceParams = new ART1Parameters.Builder()
            .vigilance(0.1)
            .L(2.0)
            .build();
            
        // High vigilance - should keep patterns separate  
        var highVigilanceParams = new ART1Parameters.Builder()
            .vigilance(0.9)
            .L(2.0)
            .build();
        
        var pattern1 = Pattern.of(1.0, 1.0, 0.0);
        var pattern2 = Pattern.of(1.0, 0.0, 0.0);
        
        // Test with low vigilance
        var lowVigilanceART = new ART1();
        lowVigilanceART.stepFit(pattern1, lowVigilanceParams);
        lowVigilanceART.stepFit(pattern2, lowVigilanceParams);
        
        // Test with high vigilance
        var highVigilanceART = new ART1();
        highVigilanceART.stepFit(pattern1, highVigilanceParams);
        highVigilanceART.stepFit(pattern2, highVigilanceParams);
        
        // High vigilance should create more categories
        assertTrue(highVigilanceART.getCategoryCount() >= lowVigilanceART.getCategoryCount());
    }
    
    @Test
    void testParameterValidation() {
        var pattern = Pattern.of(1.0, 0.0, 1.0);
        
        // Null parameters should throw exception
        assertThrows(NullPointerException.class, () -> art1.stepFit(pattern, null));
        
        // Wrong parameter type should now throw IllegalArgumentException with prophylactic validation
        var wrongParams = "wrong type";
        var exception = assertThrows(IllegalArgumentException.class, 
            () -> art1.stepFit(pattern, wrongParams));
        
        // Verify the error message is helpful
        assertTrue(exception.getMessage().contains("ART1 requires ART1Parameters"));
        assertTrue(exception.getMessage().contains("java.lang.String"));
    }
    
    @Test
    void testToString() {
        var result = art1.toString();
        assertNotNull(result);
        assertTrue(result.contains("ART1"));
        assertTrue(result.contains("categories=0"));
        
        // After adding a category
        art1.stepFit(Pattern.of(1.0, 0.0), params);
        result = art1.toString();
        assertTrue(result.contains("categories=1"));
    }
}