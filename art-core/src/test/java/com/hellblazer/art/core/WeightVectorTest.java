package com.hellblazer.art.core;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for WeightVector interface and implementations.
 * Tests all weight vector types used in ART algorithms.
 */
class WeightVectorTest {
    
    @Test
    @DisplayName("FuzzyWeight creation and operations")
    void testFuzzyWeight() {
        var data = new double[]{0.8, 0.3, 0.9, 0.2, 0.7, 0.1}; // complement-coded
        var weight = FuzzyWeight.of(data, 3); // original dimension is 3
        
        assertEquals(6, weight.dimension());
        assertEquals(3, weight.originalDimension());
        assertEquals(0.8, weight.get(0), 1e-10);
        assertEquals(0.1, weight.get(5), 1e-10);
        
        // Test L1 norm
        assertEquals(3.0, weight.l1Norm(), 1e-10); // 0.8+0.3+0.9+0.2+0.7+0.1
    }
    
    @Test
    @DisplayName("FuzzyWeight from input vector with complement coding")
    void testFuzzyWeightFromVector() {
        var input = Vector.of(0.2, 0.8, 0.5);
        var weight = FuzzyWeight.fromInput(input);
        
        assertEquals(6, weight.dimension());
        assertEquals(3, weight.originalDimension());
        
        // Check complement coding: [0.2, 0.8, 0.5, 0.8, 0.2, 0.5]
        assertEquals(0.2, weight.get(0), 1e-10);
        assertEquals(0.8, weight.get(1), 1e-10);  
        assertEquals(0.5, weight.get(2), 1e-10);
        assertEquals(0.8, weight.get(3), 1e-10); // complement of 0.2
        assertEquals(0.2, weight.get(4), 1e-10); // complement of 0.8
        assertEquals(0.5, weight.get(5), 1e-10); // complement of 0.5
    }
    
    @Test
    @DisplayName("FuzzyWeight update with learning rate")
    void testFuzzyWeightUpdate() {
        var input = Vector.of(0.6, 0.4, 0.8);
        var complementInput = Vector.of(0.6, 0.4, 0.8, 0.4, 0.6, 0.2); // hand-calculated
        var weight = FuzzyWeight.of(new double[]{1.0, 1.0, 1.0, 0.0, 0.0, 0.0}, 3);
        var params = FuzzyParameters.of(0.5, 0.0, 0.7); // beta = 0.7
        
        var updated = weight.update(complementInput, params);
        
        // Updated weight should be: β * min(input, weight) + (1-β) * weight
        // For first element: 0.7 * min(0.6, 1.0) + 0.3 * 1.0 = 0.7 * 0.6 + 0.3 = 0.72
        assertEquals(0.72, updated.get(0), 1e-10);
    }
    
    @Test
    @DisplayName("GaussianWeight creation and operations")
    void testGaussianWeight() {
        var mean = new double[]{0.0, 1.0, -0.5};
        var sigma = new double[]{1.0, 2.0, 0.5};
        var weight = GaussianWeight.of(mean, sigma, 1L);
        
        assertEquals(3, weight.dimension());
        assertArrayEquals(mean, weight.mean(), 1e-10);
        assertArrayEquals(sigma, weight.sigma(), 1e-10);
        assertEquals(1L, weight.sampleCount());
        
        // Check computed inverse sigma and determinant
        assertNotNull(weight.invSigma());
        assertTrue(weight.sqrtDetSigma() > 0);
    }
    
    @Test
    @DisplayName("GaussianWeight statistical update")
    void testGaussianWeightUpdate() {
        var initialMean = new double[]{0.0, 0.0};
        var initialSigma = new double[]{1.0, 1.0};
        var weight = GaussianWeight.of(initialMean, initialSigma, 1L);
        
        var newSample = Vector.of(1.0, 2.0);
        var updated = (GaussianWeight) weight.update(newSample, GaussianParameters.withDimension(2));
        
        assertEquals(2L, updated.sampleCount());
        
        // New mean should be (old_mean * old_count + new_sample) / new_count
        assertEquals(0.5, updated.mean()[0], 1e-10); // (0*1 + 1*1) / 2 = 0.5
        assertEquals(1.0, updated.mean()[1], 1e-10); // (0*1 + 2*1) / 2 = 1.0
    }
    
    @Test
    @DisplayName("HypersphereWeight creation and operations")
    void testHypersphereWeight() {
        var center = new double[]{1.0, 2.0, 3.0};
        var weight = HypersphereWeight.of(center, 2.5);
        
        assertEquals(3, weight.dimension());
        assertArrayEquals(center, weight.center(), 1e-10);
        assertEquals(2.5, weight.radius(), 1e-10);
    }
    
    @Test
    @DisplayName("HypersphereWeight radius expansion")
    void testHypersphereWeightExpansion() {
        var center = new double[]{0.0, 0.0};
        var weight = HypersphereWeight.of(center, 1.0);
        
        // Point outside current radius should expand it
        var outsidePoint = Vector.of(3.0, 4.0); // Distance = 5.0
        var expanded = weight.expandToInclude(outsidePoint);
        
        assertEquals(5.0, expanded.radius(), 1e-10);
        assertArrayEquals(center, expanded.center(), 1e-10); // Center unchanged
    }
    
    @Test
    @DisplayName("WeightVector dimension validation")
    void testWeightVectorDimensionValidation() {
        // FuzzyWeight validation
        assertThrows(IllegalArgumentException.class, 
            () -> FuzzyWeight.of(new double[]{1.0, 2.0, 3.0}, 2)); // Odd length for complement coding
        
        // GaussianWeight validation  
        assertThrows(IllegalArgumentException.class,
            () -> GaussianWeight.of(new double[]{1.0}, new double[]{1.0, 2.0}, 1L)); // Mismatched dimensions
            
        // Empty dimensions
        assertThrows(IllegalArgumentException.class,
            () -> GaussianWeight.of(new double[0], new double[0], 1L));
    }
    
    @Test
    @DisplayName("WeightVector null input handling")
    void testWeightVectorNullHandling() {
        assertThrows(NullPointerException.class, 
            () -> FuzzyWeight.of(null, 2));
        assertThrows(NullPointerException.class, 
            () -> GaussianWeight.of(null, new double[]{1.0}, 1L));
        assertThrows(NullPointerException.class, 
            () -> HypersphereWeight.of(null, 1.0));
    }
    
    @Test
    @DisplayName("WeightVector immutability")
    void testWeightVectorImmutability() {
        var originalData = new double[]{1.0, 2.0, 3.0, 4.0};
        var weight = FuzzyWeight.of(originalData, 2);
        
        // Modify original array
        originalData[0] = 999.0;
        
        // Weight should be unaffected
        assertEquals(1.0, weight.get(0), 1e-10);
        
        // Test that returned arrays are defensive copies
        var mean = new double[]{1.0, 2.0};
        var sigma = new double[]{1.0, 1.0};
        var gaussWeight = GaussianWeight.of(mean, sigma, 1L);
        
        mean[0] = 999.0;
        sigma[0] = 999.0;
        
        assertEquals(1.0, gaussWeight.mean()[0], 1e-10);
        assertEquals(1.0, gaussWeight.sigma()[0], 1e-10);
    }
    
    @Test
    @DisplayName("WeightVector equality and hashcode")
    void testWeightVectorEquality() {
        var data = new double[]{1.0, 2.0, 3.0, 4.0};
        var weight1 = FuzzyWeight.of(data, 2);
        var weight2 = FuzzyWeight.of(data, 2);
        var weight3 = FuzzyWeight.of(new double[]{1.0, 2.0, 3.0, 5.0}, 2);
        
        assertEquals(weight1, weight2);
        assertNotEquals(weight1, weight3);
        assertEquals(weight1.hashCode(), weight2.hashCode());
    }
    
    @ParameterizedTest
    @ValueSource(ints = {1, 2, 5, 10, 50})
    @DisplayName("WeightVector scaling with different dimensions")
    void testWeightVectorScaling(int originalDimension) {
        var data = new double[originalDimension * 2];
        for (int i = 0; i < data.length; i++) {
            data[i] = Math.random();
        }
        
        var weight = FuzzyWeight.of(data, originalDimension);
        assertEquals(originalDimension * 2, weight.dimension());
        assertEquals(originalDimension, weight.originalDimension());
    }
    
    @Test
    @DisplayName("WeightVector type safety with sealed interface")
    void testWeightVectorTypePattern() {
        WeightVector weight = FuzzyWeight.fromInput(Vector.of(0.5, 0.5));
        
        var result = switch (weight) {
            case FuzzyWeight fw -> "fuzzy:" + fw.originalDimension();
            case GaussianWeight gw -> "gaussian:" + gw.sampleCount();
            case HypersphereWeight hw -> "hypersphere:" + hw.radius();
            case ARTAWeight aw -> "arta:" + aw.dimension();
            case ARTSTARWeight asw -> "artstar:" + asw.dimension();
            case ARTEWeight aew -> "arte:" + aew.dimension();
            default -> "unknown:" + weight.getClass().getSimpleName();
        };
        
        assertEquals("fuzzy:2", result);
    }
}
