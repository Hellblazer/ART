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
package com.hellblazer.art.core.visualization;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Simple test for visualization framework components.
 * 
 * @author Hal Hildebrand
 */
public class SimpleVisualizationTest {
    
    /**
     * Test that VAT and iVAT algorithms can be instantiated and run.
     */
    @Test
    void testVATandIVAT() {
        // Simple test data - 4 points that should form 2 clusters
        double[][] data = {
            {0.0, 0.0},  // Cluster 1
            {0.1, 0.1},  // Cluster 1  
            {5.0, 5.0},  // Cluster 2
            {5.1, 5.1}   // Cluster 2
        };
        
        // Test VAT
        VATResult vatResult = VAT.compute(data);
        assertNotNull(vatResult);
        assertEquals(4, vatResult.getSize());
        assertTrue(vatResult.getClusterClarity() >= 0.0);
        assertTrue(vatResult.estimateClusterCount() >= 1);
        
        // Test iVAT
        VATResult ivatResult = iVAT.compute(data);
        assertNotNull(ivatResult);
        assertEquals(4, ivatResult.getSize());
        assertTrue(ivatResult.getClusterClarity() >= 0.0);
    }
    
    /**
     * Test Visualizable interface default implementations.
     */
    @Test
    void testVisualizableDefaults() {
        TestVisualizable visualizable = new TestVisualizable();
        
        assertFalse(visualizable.isVisualizationEnabled());
        assertTrue(visualizable.getVisualizationData().isEmpty());
        assertEquals("No visualization available", visualizable.getVisualizationDescription());
        
        // Test no-op methods don't throw exceptions
        assertDoesNotThrow(() -> visualizable.setVisualizationEnabled(true));
        assertDoesNotThrow(() -> visualizable.updateVisualizationData());
        assertDoesNotThrow(() -> visualizable.clearVisualizationData());
    }
    
    /**
     * Test VisualizationData creation.
     */
    @Test
    void testVisualizationDataCreation() {
        var vizData = VisualizationData.builder()
            .algorithmType("Test")
            .build();
        
        assertNotNull(vizData);
        assertEquals("Test", vizData.getAlgorithmType());
        assertFalse(vizData.hasClusters());
        assertFalse(vizData.hasDataPoints());
        assertTrue(vizData.getTimestamp() > 0);
    }
    
    private static class TestVisualizable implements Visualizable {
        // Uses all default implementations
    }
}