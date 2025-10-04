package com.hellblazer.art.laminar;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.laminar.builders.LaminarCircuitBuilder;
import com.hellblazer.art.laminar.core.LaminarCircuit;
import com.hellblazer.art.laminar.impl.DefaultLaminarParameters;
import com.hellblazer.art.laminar.impl.DefaultLearningParameters;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Basic compilation and functionality test for art-laminar module.
 */
class BasicCompilationTest {

    @Test
    void testModuleCompiles() {
        var parameters = DefaultLaminarParameters.builder()
            .withLearningParameters(new DefaultLearningParameters(0.5, 0.0, false, 0.0))
            .withVigilance(0.8)
            .build();

        LaminarCircuit<DefaultLaminarParameters> circuit =
            new LaminarCircuitBuilder<DefaultLaminarParameters>()
                .withParameters(parameters)
                .withInputLayer(4, false)
                .withFeatureLayer(4)
                .withCategoryLayer(10)
                .withStandardConnections()
                .withVigilance(0.8)
                .build();

        assertNotNull(circuit);
        assertEquals(3, circuit.getLayers().size());
    }

    @Test
    void testBasicLearning() {
        var parameters = DefaultLaminarParameters.builder()
            .withLearningParameters(new DefaultLearningParameters(0.5, 0.0, false, 0.0))
            .build();

        var circuit = new LaminarCircuitBuilder<DefaultLaminarParameters>()
            .withParameters(parameters)
            .withInputLayer(4, false)
            .withFeatureLayer(4)
            .withCategoryLayer(10)
            .withStandardConnections()
            .build();

        var pattern = new DenseVector(new double[]{0.5, 0.5, 0.5, 0.5});
        var result = circuit.learn(pattern, parameters);

        assertNotNull(result);
        assertEquals(1, circuit.getCategoryCount());
    }

    @Test
    void testProcessCycle() {
        var parameters = DefaultLaminarParameters.builder().build();

        var circuit = new LaminarCircuitBuilder<DefaultLaminarParameters>()
            .withParameters(parameters)
            .withInputLayer(4, false)
            .withFeatureLayer(4)
            .withCategoryLayer(10)
            .withStandardConnections()
            .build();

        var pattern = new DenseVector(new double[]{0.7, 0.3, 0.2, 0.1});
        var result = circuit.processCycle(pattern, parameters);

        assertNotNull(result);
        assertTrue(result.getProcessingCycles() > 0);
    }
}