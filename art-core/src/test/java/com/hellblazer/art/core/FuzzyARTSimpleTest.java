package com.hellblazer.art.core;

import com.hellblazer.art.core.algorithms.FuzzyART;
import com.hellblazer.art.core.parameters.FuzzyParameters;
import com.hellblazer.art.core.results.ActivationResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Simple test to verify FuzzyART compilation and basic functionality.
 */
class FuzzyARTSimpleTest {
    
    private FuzzyART fuzzyART;
    private FuzzyParameters defaultParams;
    
    @BeforeEach
    void setUp() {
        fuzzyART = new FuzzyART();
        defaultParams = FuzzyParameters.defaults();
    }
    
    @Test
    @DisplayName("FuzzyART constructor creates empty network")
    void testConstructor() {
        assertEquals(0, fuzzyART.getCategoryCount());
        assertTrue(fuzzyART.getCategories().isEmpty());
        assertEquals("FuzzyART{categories=0}", fuzzyART.toString());
    }
    
    @Test
    @DisplayName("First input creates initial category")
    void testFirstInputCreatesCategory() {
        var input = Pattern.of(0.3, 0.7);
        var result = fuzzyART.stepFit(input, defaultParams);
        
        // Should create new category
        assertTrue(result instanceof ActivationResult.Success);
        assertEquals(1, fuzzyART.getCategoryCount());
    }
}