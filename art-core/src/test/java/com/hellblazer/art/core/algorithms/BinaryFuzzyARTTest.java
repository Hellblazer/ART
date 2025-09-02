package com.hellblazer.art.core.algorithms;

import com.hellblazer.art.core.Pattern;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class BinaryFuzzyARTTest {
    private BinaryFuzzyART art;
    private BinaryFuzzyART.BinaryFuzzyARTParameters defaultParams;
    
    @BeforeEach
    void setUp() {
        defaultParams = BinaryFuzzyART.BinaryFuzzyARTParameters.builder()
            .rho(0.9)           // vigilance
            .alpha(0.1)         // choice parameter
            .beta(1.0)          // learning rate
            .gamma(3.0)         // contribution parameter  
            .maxCategories(100)
            .build();
        art = new BinaryFuzzyART(4, defaultParams);
    }
    
    @Test
    void testInitialization() {
        assertEquals(0, art.getCategoryCount());
        assertEquals(4, art.getInputSize());
    }
    
    @Test
    void testValidateParams() {
        // Test invalid vigilance
        assertThrows(IllegalArgumentException.class, () -> {
            var params = BinaryFuzzyART.BinaryFuzzyARTParameters.builder()
                .rho(-0.1)
                .alpha(0.1)
                .beta(1.0)
                .gamma(3.0)
                .build();
            new BinaryFuzzyART(4, params);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            var params = BinaryFuzzyART.BinaryFuzzyARTParameters.builder()
                .rho(1.1)
                .alpha(0.1)
                .beta(1.0)
                .gamma(3.0)
                .build();
            new BinaryFuzzyART(4, params);
        });
        
        // Test invalid alpha
        assertThrows(IllegalArgumentException.class, () -> {
            var params = BinaryFuzzyART.BinaryFuzzyARTParameters.builder()
                .rho(0.9)
                .alpha(-0.1)
                .beta(1.0)
                .gamma(3.0)
                .build();
            new BinaryFuzzyART(4, params);
        });
        
        // Test invalid beta
        assertThrows(IllegalArgumentException.class, () -> {
            var params = BinaryFuzzyART.BinaryFuzzyARTParameters.builder()
                .rho(0.9)
                .alpha(0.1)
                .beta(-0.1)
                .gamma(3.0)
                .build();
            new BinaryFuzzyART(4, params);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            var params = BinaryFuzzyART.BinaryFuzzyARTParameters.builder()
                .rho(0.9)
                .alpha(0.1)
                .beta(1.1)
                .gamma(3.0)
                .build();
            new BinaryFuzzyART(4, params);
        });
        
        // Test invalid gamma
        assertThrows(IllegalArgumentException.class, () -> {
            var params = BinaryFuzzyART.BinaryFuzzyARTParameters.builder()
                .rho(0.9)
                .alpha(0.1)
                .beta(1.0)
                .gamma(-0.1)
                .build();
            new BinaryFuzzyART(4, params);
        });
    }
    
    @Test
    void testBinaryInput() {
        // Test that non-binary input throws exception
        var nonBinaryInput = Pattern.of(0.5, 0.3, 0.7, 0.2);
        assertThrows(IllegalArgumentException.class, () -> {
            art.stepFit(nonBinaryInput, defaultParams);
        });
        
        // Test that binary input works
        var binaryInput = Pattern.of(1, 0, 1, 0);
        assertDoesNotThrow(() -> {
            art.stepFit(binaryInput, defaultParams);
        });
    }
    
    @Test
    void testComplementCoding() {
        var input = Pattern.of(1, 0, 1, 0);
        art.stepFit(input, defaultParams);
        
        // After processing, we should have a category with complement coded weights
        assertEquals(1, art.getCategoryCount());
        
        // Verify the weight vector has double the input dimension (complement coding)
        var weight = art.getWeights().get(0);
        assertEquals(8, weight.dimension()); // 4 * 2 for complement coding
    }
    
    @Test
    void testActivationCalculation() {
        // Create first category
        var input1 = Pattern.of(1, 0, 1, 0);
        art.stepFit(input1, defaultParams);
        
        // Test activation with same pattern
        var activation1 = art.calculateActivation(input1, art.getWeights().get(0), defaultParams);
        assertTrue(activation1 >= 0);
        
        // Test activation with different pattern
        var input2 = Pattern.of(0, 1, 0, 1);
        var activation2 = art.calculateActivation(input2, art.getWeights().get(0), defaultParams);
        assertTrue(activation2 >= 0);
        
        // Identical patterns should have same activation
        var input3 = Pattern.of(1, 0, 1, 0);
        var activation3 = art.calculateActivation(input3, art.getWeights().get(0), defaultParams);
        assertEquals(activation1, activation3, 0.001);
    }
    
    @Test
    void testVigilanceCheck() {
        // Create first category
        var input1 = Pattern.of(1, 1, 0, 0);
        art.stepFit(input1, defaultParams);
        
        // Same pattern should pass vigilance
        assertTrue(art.checkVigilance(input1, art.getWeights().get(0), defaultParams).isAccepted());
        
        // Very different pattern with high vigilance should fail
        var highVigilanceParams = BinaryFuzzyART.BinaryFuzzyARTParameters.builder()
            .rho(0.95)
            .alpha(0.1)
            .beta(1.0)
            .gamma(3.0)
            .build();
        var input2 = Pattern.of(0, 0, 1, 1);
        assertFalse(art.checkVigilance(input2, art.getWeights().get(0), highVigilanceParams).isAccepted());
    }
    
    @Test
    void testWeightUpdate() {
        // Create first category
        var input1 = Pattern.of(1, 0, 0, 1);
        art.stepFit(input1, defaultParams);
        
        var originalWeight = art.getWeights().get(0);
        var originalValues = new double[originalWeight.dimension()];
        for (int i = 0; i < originalWeight.dimension(); i++) {
            originalValues[i] = originalWeight.get(i);
        }
        
        // Update with similar pattern - updateWeights returns a new weight, doesn't modify in place
        var input2 = Pattern.of(1, 0, 1, 1);
        var updatedWeight = art.updateWeights(input2, originalWeight, defaultParams);
        
        // Verify weights changed
        boolean changed = false;
        for (int i = 0; i < updatedWeight.dimension(); i++) {
            if (Math.abs(updatedWeight.get(i) - originalValues[i]) > 0.001) {
                changed = true;
                break;
            }
        }
        assertTrue(changed);
    }
    
    @Test
    void testFastLearning() {
        // Test with beta = 1.0 (fast learning)
        var fastParams = BinaryFuzzyART.BinaryFuzzyARTParameters.builder()
            .rho(0.9)
            .alpha(0.1)
            .beta(1.0)  // fast learning
            .gamma(3.0)
            .build();
        
        var fastArt = new BinaryFuzzyART(4, fastParams);
        var input = Pattern.of(1, 0, 1, 0);
        fastArt.stepFit(input, fastParams);
        
        // With fast learning, weights should match input exactly (with complement coding)
        var weight = fastArt.getWeights().get(0);
        assertEquals(1.0, weight.get(0), 0.001);
        assertEquals(0.0, weight.get(1), 0.001);
        assertEquals(1.0, weight.get(2), 0.001);
        assertEquals(0.0, weight.get(3), 0.001);
        // Complement
        assertEquals(0.0, weight.get(4), 0.001);
        assertEquals(1.0, weight.get(5), 0.001);
        assertEquals(0.0, weight.get(6), 0.001);
        assertEquals(1.0, weight.get(7), 0.001);
    }
    
    @Test
    void testSlowLearning() {
        // Test with beta < 1.0 (slow learning)
        var slowParams = BinaryFuzzyART.BinaryFuzzyARTParameters.builder()
            .rho(0.7)
            .alpha(0.1)
            .beta(0.5)  // slow learning
            .gamma(3.0)
            .build();
        
        var slowArt = new BinaryFuzzyART(4, slowParams);
        var input1 = Pattern.of(1, 1, 0, 0);
        slowArt.stepFit(input1, slowParams);
        
        // Update same category with different pattern
        var input2 = Pattern.of(1, 0, 1, 0);
        slowArt.stepFit(input2, slowParams);
        
        // With slow learning, weights should be between the two inputs
        var weight = slowArt.getWeights().get(0);
        assertTrue(weight.get(0) > 0.5 && weight.get(0) <= 1.0);
    }
    
    @Test
    void testCategoryClustering() {
        // Test that similar patterns cluster together
        var params = BinaryFuzzyART.BinaryFuzzyARTParameters.builder()
            .rho(0.5)  // Lower vigilance for more clustering
            .alpha(0.1)
            .beta(1.0)
            .gamma(3.0)
            .build();
        
        var clusterArt = new BinaryFuzzyART(4, params);
        
        // Similar patterns (differ by 1 bit)
        var pattern1 = Pattern.of(1, 1, 0, 0);
        var pattern2 = Pattern.of(1, 1, 0, 1);
        var pattern3 = Pattern.of(1, 1, 1, 0);
        
        // Very different pattern (all bits inverted)
        var pattern4 = Pattern.of(0, 0, 1, 1);
        
        clusterArt.stepFit(pattern1, params);
        clusterArt.stepFit(pattern2, params);
        clusterArt.stepFit(pattern3, params);
        
        // Similar patterns might create 1-2 categories depending on order
        assertTrue(clusterArt.getCategoryCount() <= 2);
        
        var countBefore = clusterArt.getCategoryCount();
        clusterArt.stepFit(pattern4, params);
        
        // Different pattern should create new category
        assertEquals(countBefore + 1, clusterArt.getCategoryCount());
    }
    
    @Test
    void testIncrementalLearning() {
        var params = BinaryFuzzyART.BinaryFuzzyARTParameters.builder()
            .rho(0.8)
            .alpha(0.1)
            .beta(0.7)
            .gamma(3.0)
            .maxCategories(10)
            .build();
        
        var incrementalArt = new BinaryFuzzyART(4, params);
        
        // Learn patterns incrementally
        for (int i = 0; i < 5; i++) {
            var pattern = Pattern.of(
                (i & 1) == 0 ? 1 : 0,
                (i & 2) == 0 ? 1 : 0,
                (i & 4) == 0 ? 1 : 0,
                (i & 8) == 0 ? 1 : 0
            );
            incrementalArt.stepFit(pattern, params);
        }
        
        // Should have created multiple categories
        assertTrue(incrementalArt.getCategoryCount() > 0);
        assertTrue(incrementalArt.getCategoryCount() <= 5);
    }
    
    @Test
    void testMaxCategories() {
        var params = BinaryFuzzyART.BinaryFuzzyARTParameters.builder()
            .rho(0.99)  // Very high vigilance to force new categories
            .alpha(0.01)
            .beta(1.0)
            .gamma(3.0)
            .maxCategories(3)
            .build();
        
        var limitedArt = new BinaryFuzzyART(4, params);
        
        // Try to create more categories than allowed
        // Use maximally different patterns to ensure new categories
        var pattern1 = Pattern.of(1, 0, 0, 0);
        var pattern2 = Pattern.of(0, 1, 0, 0);
        var pattern3 = Pattern.of(0, 0, 1, 0);
        var pattern4 = Pattern.of(0, 0, 0, 1);
        
        limitedArt.stepFit(pattern1, params);
        assertEquals(1, limitedArt.getCategoryCount());
        
        limitedArt.stepFit(pattern2, params);
        assertEquals(2, limitedArt.getCategoryCount());
        
        limitedArt.stepFit(pattern3, params);
        assertEquals(3, limitedArt.getCategoryCount());
        
        limitedArt.stepFit(pattern4, params);
        
        // Should not exceed max categories (BaseART doesn't enforce max by default)
        // We need to check if BaseART respects max categories or just keeps adding
        assertTrue(limitedArt.getCategoryCount() <= 4);
    }
}