package com.hellblazer.art.core;

import com.hellblazer.art.core.algorithms.TopoART;
import com.hellblazer.art.core.parameters.TopoARTParameters;
import com.hellblazer.art.core.utils.MathOperations;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Simple test to verify TopoART compilation and basic functionality.
 */
class SimpleTopoARTTest {
    
    @Test
    void testBasicTopoARTCreation() {
        var params = TopoARTParameters.builder()
            .inputDimension(3)
            .vigilanceA(0.7)
            .learningRateSecond(0.5)
            .alpha(0.001)
            .phi(5)
            .tau(100)
            .build();
            
        var topoART = new TopoART(params);
        assertNotNull(topoART);
    }
    
    @Test
    void testBasicLearning() {
        var params = TopoARTParameters.builder()
            .inputDimension(3)
            .vigilanceA(0.7)
            .learningRateSecond(0.5)
            .alpha(0.001)
            .phi(3)  // Lower threshold for testing
            .tau(100)
            .build();
            
        var topoART = new TopoART(params);
        
        // Present pattern multiple times to achieve permanence
        double[] pattern = {0.5, 0.5, 0.5};
        for (int i = 0; i < 5; i++) {
            topoART.learn(pattern);
        }
        
        // Component A should always have learned
        assertTrue(topoART.getComponentA().getNeurons().size() > 0);
        
        // Component B should now have learned (after permanence achieved)
        assertTrue(topoART.getComponentB().getNeurons().size() > 0);
    }
}