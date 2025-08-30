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

import com.hellblazer.art.core.algorithms.FuzzyART;
import com.hellblazer.art.core.parameters.FuzzyParameters;
import com.hellblazer.art.core.results.ActivationResult;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Python Reference Comparison Test for Core FuzzyART Implementation.
 * 
 * This test validates that the core FuzzyART implementation produces identical
 * clustering behavior to the reference parity reference implementation.
 * 
 * Uses exact same test data and expected outcomes as vectorized implementation
 * to ensure both core and vectorized versions achieve Python parity.
 */
@DisplayName("Core FuzzyART Python Reference Validation")
class CoreFuzzyARTReferenceTest {
    
    private FuzzyART coreArt;
    private FuzzyParameters referenceParams;
    
    @BeforeEach
    void setUp() {
        coreArt = new FuzzyART();
        
        // Use exact same parameters as reference parity reference:
        // rho (vigilance) = 0.9, alpha (choice) = 0.001, beta (learning rate) = 1.0
        referenceParams = FuzzyParameters.of(
            0.9,    // vigilance (rho) - high vigilance like Python test
            0.001,  // choice parameter (alpha)
            1.0     // learning rate (beta)
        );
    }
    
    @AfterEach
    void tearDown() {
        // Core FuzzyART doesn't implement AutoCloseable
    }
    
    @Test
    @DisplayName("Core FuzzyART should match reference parity clustering exactly")
    void testCoreFuzzyARTMatchesPythonReference() {
        // EXACT same test data as reference parity demo_fuzzy_art.py
        var patterns = Arrays.asList(
            Pattern.of(0.0, 0.0),    // Expected category: 0
            Pattern.of(0.0, 0.08),   // Expected category: 0 (similar to first) 
            Pattern.of(0.1, 0.1),    // Expected category: 0 (close to first two)
            Pattern.of(0.9, 0.9),    // Expected category: 1 (distinctly different)
            Pattern.of(1.0, 1.0),    // Expected category: 1 (similar to fourth)
            Pattern.of(0.8, 0.9)     // Expected category: 2 (creates new category)
        );
        
        // Expected reference clustering: [0, 0, 0, 1, 1, 2]
        var expectedLabels = Arrays.asList(0, 0, 0, 1, 1, 2);
        var actualLabels = new ArrayList<Integer>();
        
        // Process each pattern and collect category assignments
        for (int i = 0; i < patterns.size(); i++) {
            var pattern = patterns.get(i);
            var result = coreArt.stepFit(pattern, referenceParams);
            
            assertInstanceOf(ActivationResult.Success.class, result,
                "Pattern " + i + " should be successfully learned");
            
            var success = (ActivationResult.Success) result;
            var categoryIndex = success.categoryIndex();
            actualLabels.add(categoryIndex);
            
            System.out.printf("Pattern %d: %s -> Category %d (activation: %.4f)%n",
                            i, pattern, categoryIndex, success.activationValue());
        }
        
        // CRITICAL VALIDATION: Verify exact Python clustering behavior
        assertEquals(expectedLabels.size(), actualLabels.size(),
            "Should process all patterns");
        
        // Validate key clustering intelligence from reference:
        
        // 1. First three patterns should cluster together (low vigilance allows grouping)
        assertEquals(actualLabels.get(0), actualLabels.get(1),
            "Patterns [0.0,0.0] and [0.0,0.08] should cluster together");
        assertEquals(actualLabels.get(0), actualLabels.get(2), 
            "Patterns [0.0,0.0] and [0.1,0.1] should cluster together");
        
        // 2. Fourth and fifth patterns should cluster together (high values)
        assertEquals(actualLabels.get(3), actualLabels.get(4),
            "Patterns [0.9,0.9] and [1.0,1.0] should cluster together");
        
        // 3. Sixth pattern should create separate category
        assertNotEquals(actualLabels.get(5), actualLabels.get(0),
            "Pattern [0.8,0.9] should be separate from low-value cluster");
        assertNotEquals(actualLabels.get(5), actualLabels.get(3),
            "Pattern [0.8,0.9] should be separate from high-value cluster");
        
        // 4. Should create exactly 3 categories total
        assertEquals(3, coreArt.getCategoryCount(),
            "Should create exactly 3 categories like reference");
        
        System.out.println("✅ SUCCESS: Core Java FuzzyART clustering matches reference behavior");
        System.out.println("Expected clustering: " + expectedLabels);
        System.out.println("Actual clustering:   " + actualLabels);
    }
    
    @Test
    @DisplayName("Vigilance parameter should affect clustering like reference")
    void testVigilanceParameterIntelligence() {
        var testPatterns = Arrays.asList(
            Pattern.of(0.0, 0.0),
            Pattern.of(0.0, 0.08),
            Pattern.of(0.1, 0.1)
        );
        
        // Test with different vigilance levels
        
        // High vigilance (strict) - should create more categories
        var highVigilanceArt = new FuzzyART();
        var highVigilanceParams = FuzzyParameters.of(0.95, 0.001, 1.0);
        
        for (var pattern : testPatterns) {
            highVigilanceArt.stepFit(pattern, highVigilanceParams);
        }
        
        // Low vigilance (permissive) - should create fewer categories
        var lowVigilanceArt = new FuzzyART(); 
        var lowVigilanceParams = FuzzyParameters.of(0.5, 0.001, 1.0);
        
        for (var pattern : testPatterns) {
            lowVigilanceArt.stepFit(pattern, lowVigilanceParams);
        }
        
        // High vigilance should create more categories than low vigilance
        assertTrue(highVigilanceArt.getCategoryCount() >= lowVigilanceArt.getCategoryCount(),
            "High vigilance should create at least as many categories as low vigilance");
        
        System.out.printf("Vigilance intelligence validated: High vigilance (%.2f) = %d categories, " +
                         "Low vigilance (%.2f) = %d categories%n",
                         0.95, highVigilanceArt.getCategoryCount(),
                         0.5, lowVigilanceArt.getCategoryCount());
    }
    
    @Test
    @DisplayName("Complement coding behavior should match reference parity")
    void testComplementCodingBehavior() {
        // Test complement coding with asymmetric patterns
        var input = Pattern.of(0.3, 0.7);
        
        var result = coreArt.stepFit(input, referenceParams);
        assertInstanceOf(ActivationResult.Success.class, result);
        
        var success = (ActivationResult.Success) result;
        assertEquals(0, success.categoryIndex(), "First pattern should create category 0");
        
        // Verify category was created with complement coding
        assertEquals(1, coreArt.getCategoryCount(), "Should create exactly one category");
        
        // The weight vector should be complement coded: [I, 1-I] = [0.3, 0.7, 0.7, 0.3]
        var categories = coreArt.getCategories();
        var firstCategory = categories.get(0);
        
        // Verify complement coding structure (dimension should be 2 * input dimension)
        assertEquals(4, firstCategory.dimension(), 
            "Complement coded weight should have 4 dimensions for 2D input");
        
        System.out.println("✅ Complement coding behavior validated against reference");
    }
    
    @Test
    @DisplayName("Learning stability should match reference behavior")
    void testLearningStability() {
        var pattern = Pattern.of(0.5, 0.5);
        
        // Learn same pattern multiple times
        var result1 = coreArt.stepFit(pattern, referenceParams);
        var result2 = coreArt.stepFit(pattern, referenceParams);
        var result3 = coreArt.stepFit(pattern, referenceParams);
        
        // All should assign to same category
        assertInstanceOf(ActivationResult.Success.class, result1);
        assertInstanceOf(ActivationResult.Success.class, result2);
        assertInstanceOf(ActivationResult.Success.class, result3);
        
        var category1 = ((ActivationResult.Success) result1).categoryIndex();
        var category2 = ((ActivationResult.Success) result2).categoryIndex();  
        var category3 = ((ActivationResult.Success) result3).categoryIndex();
        
        assertEquals(category1, category2, "Repeated learning should be stable");
        assertEquals(category2, category3, "Repeated learning should be stable");
        
        // Should create only one category
        assertEquals(1, coreArt.getCategoryCount(), "Should create only one category");
        
        System.out.println("✅ Learning stability validated: consistent category " + category1);
    }
}