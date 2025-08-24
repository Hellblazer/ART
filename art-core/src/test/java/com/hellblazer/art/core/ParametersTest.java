package com.hellblazer.art.core;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for ART algorithm parameters.
 * Tests parameter validation and immutability.
 */
class ParametersTest {
    
    @Test
    @DisplayName("FuzzyParameters creation and validation")
    void testFuzzyParameters() {
        var params = FuzzyParameters.of(0.7, 0.1, 0.9);
        
        assertEquals(0.7, params.vigilance(), 1e-10);
        assertEquals(0.1, params.alpha(), 1e-10);
        assertEquals(0.9, params.beta(), 1e-10);
    }
    
    @Test
    @DisplayName("FuzzyParameters default values")
    void testFuzzyParametersDefaults() {
        var params = FuzzyParameters.defaults();
        
        assertEquals(0.5, params.vigilance(), 1e-10);
        assertEquals(0.0, params.alpha(), 1e-10);
        assertEquals(1.0, params.beta(), 1e-10);
    }
    
    @ParameterizedTest
    @ValueSource(doubles = {-0.1, 1.1, -10.0, 2.0})
    @DisplayName("FuzzyParameters vigilance validation")
    void testFuzzyParametersVigilanceValidation(double invalidVigilance) {
        assertThrows(IllegalArgumentException.class, 
            () -> FuzzyParameters.of(invalidVigilance, 0.0, 1.0));
    }
    
    @ParameterizedTest
    @ValueSource(doubles = {-0.1, -10.0})
    @DisplayName("FuzzyParameters alpha validation")
    void testFuzzyParametersAlphaValidation(double invalidAlpha) {
        assertThrows(IllegalArgumentException.class, 
            () -> FuzzyParameters.of(0.5, invalidAlpha, 1.0));
    }
    
    @ParameterizedTest
    @ValueSource(doubles = {-0.1, 1.1, -10.0, 2.0})
    @DisplayName("FuzzyParameters beta validation")
    void testFuzzyParametersBetaValidation(double invalidBeta) {
        assertThrows(IllegalArgumentException.class, 
            () -> FuzzyParameters.of(0.5, 0.0, invalidBeta));
    }
    
    @Test
    @DisplayName("GaussianParameters creation and validation")
    void testGaussianParameters() {
        var sigmaInit = new double[]{1.0, 2.0, 0.5};
        var params = GaussianParameters.of(0.8, sigmaInit);
        
        assertEquals(0.8, params.vigilance(), 1e-10);
        assertArrayEquals(sigmaInit, params.sigmaInit(), 1e-10);
        
        // Test immutability - modifying original array shouldn't affect params
        sigmaInit[0] = 999.0;
        assertEquals(1.0, params.sigmaInit()[0], 1e-10);
    }
    
    @Test
    @DisplayName("GaussianParameters default sigma initialization")
    void testGaussianParametersDefaultSigma() {
        var params = GaussianParameters.withDimension(3);
        
        assertEquals(0.5, params.vigilance(), 1e-10);
        assertEquals(3, params.sigmaInit().length);
        
        for (double sigma : params.sigmaInit()) {
            assertEquals(1.0, sigma, 1e-10);
        }
    }
    
    @Test
    @DisplayName("GaussianParameters sigma validation")
    void testGaussianParametersSigmaValidation() {
        // Empty sigma array
        assertThrows(IllegalArgumentException.class, 
            () -> GaussianParameters.of(0.5, new double[0]));
        
        // Negative sigma values
        assertThrows(IllegalArgumentException.class, 
            () -> GaussianParameters.of(0.5, new double[]{1.0, -0.5, 2.0}));
        
        // Zero sigma values
        assertThrows(IllegalArgumentException.class, 
            () -> GaussianParameters.of(0.5, new double[]{1.0, 0.0, 2.0}));
        
        // Null sigma array
        assertThrows(NullPointerException.class, 
            () -> GaussianParameters.of(0.5, null));
    }
    
    @Test
    @DisplayName("HypersphereParameters creation and validation")
    void testHypersphereParameters() {
        var params = HypersphereParameters.of(0.6, 2.5, true);
        
        assertEquals(0.6, params.vigilance(), 1e-10);
        assertEquals(2.5, params.defaultRadius(), 1e-10);
        assertTrue(params.adaptiveRadius());
    }
    
    @Test
    @DisplayName("HypersphereParameters defaults")
    void testHypersphereParametersDefaults() {
        var params = HypersphereParameters.defaults();
        
        assertEquals(0.5, params.vigilance(), 1e-10);
        assertEquals(1.0, params.defaultRadius(), 1e-10);
        assertFalse(params.adaptiveRadius());
    }
    
    @Test
    @DisplayName("HypersphereParameters radius validation")
    void testHypersphereParametersRadiusValidation() {
        assertThrows(IllegalArgumentException.class, 
            () -> HypersphereParameters.of(0.5, -1.0, false));
        assertThrows(IllegalArgumentException.class, 
            () -> HypersphereParameters.of(0.5, 0.0, false));
    }
    
    @Test
    @DisplayName("Parameters equality and hashcode")
    void testParametersEquality() {
        var fuzzy1 = FuzzyParameters.of(0.7, 0.1, 0.9);
        var fuzzy2 = FuzzyParameters.of(0.7, 0.1, 0.9);
        var fuzzy3 = FuzzyParameters.of(0.8, 0.1, 0.9);
        
        assertEquals(fuzzy1, fuzzy2);
        assertNotEquals(fuzzy1, fuzzy3);
        assertEquals(fuzzy1.hashCode(), fuzzy2.hashCode());
        
        var gauss1 = GaussianParameters.of(0.5, new double[]{1.0, 2.0});
        var gauss2 = GaussianParameters.of(0.5, new double[]{1.0, 2.0});
        var gauss3 = GaussianParameters.of(0.5, new double[]{1.0, 3.0});
        
        assertEquals(gauss1, gauss2);
        assertNotEquals(gauss1, gauss3);
        assertEquals(gauss1.hashCode(), gauss2.hashCode());
    }
    
    @Test
    @DisplayName("Parameters builder pattern")
    void testParametersBuilder() {
        var fuzzyBuilder = FuzzyParameters.builder()
            .vigilance(0.8)
            .choiceParameter(0.2)
            .learningRate(0.7);
        var params = fuzzyBuilder.build();
        
        assertEquals(0.8, params.vigilance(), 1e-10);
        assertEquals(0.2, params.alpha(), 1e-10);
        assertEquals(0.7, params.beta(), 1e-10);
    }
    
    @Test
    @DisplayName("Parameters with method for immutable updates")
    void testParametersWithMethods() {
        var original = FuzzyParameters.of(0.5, 0.0, 1.0);
        var updated = original.withVigilance(0.8);
        
        assertEquals(0.5, original.vigilance(), 1e-10);
        assertEquals(0.8, updated.vigilance(), 1e-10);
        assertEquals(original.alpha(), updated.alpha(), 1e-10);
        assertEquals(original.beta(), updated.beta(), 1e-10);
    }
}