package com.hellblazer.art.core;

import com.hellblazer.art.core.algorithms.TopoART;
import com.hellblazer.art.core.parameters.TopoARTParameters;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test that TopoART maintains BaseART compatibility after refactoring.
 */
public class TopoARTBaseARTCompatibilityTest {
    
    @Test
    @DisplayName("TopoART extends BaseART")
    void testTopoARTExtendsBaseART() {
        var params = TopoARTParameters.builder()
            .inputDimension(2)
            .vigilanceA(0.7)
            .learningRateSecond(0.5)
            .alpha(0.001)
            .phi(3)
            .tau(10)
            .build();
            
        var topoART = new TopoART(params);
        
        // Verify inheritance
        assertInstanceOf(BaseART.class, topoART, "TopoART should extend BaseART");
        
        // Verify TopoART functionality still works
        assertNotNull(topoART.getComponentA(), "Component A should be accessible");
        assertNotNull(topoART.getComponentB(), "Component B should be accessible");
        
        // Test basic learning still works
        double[] pattern = {0.5, 0.8};
        assertDoesNotThrow(() -> topoART.learn(pattern), "Learning should work");
        
        assertEquals(1, topoART.getLearningCycle(), "Learning cycle should increment");
    }
    
    @Test
    @DisplayName("TopoART preserves dual-component behavior")
    void testDualComponentBehavior() {
        var params = TopoARTParameters.builder()
            .inputDimension(2)
            .vigilanceA(0.6)
            .learningRateSecond(0.1)
            .alpha(0.001)
            .phi(2)  // Low threshold for quick permanence
            .tau(100)
            .build();
            
        var topoART = new TopoART(params);
        
        // Learn pattern multiple times to achieve permanence
        double[] pattern = {0.7, 0.3};
        for (int i = 0; i < 5; i++) {
            topoART.learn(pattern);
        }
        
        // Component A should have learned
        assertTrue(topoART.getComponentA().getNeuronCount() > 0, 
            "Component A should have neurons");
            
        // Component B should also have learned after permanence
        assertTrue(topoART.getComponentB().getNeuronCount() > 0,
            "Component B should have neurons after permanence achieved");
    }
}