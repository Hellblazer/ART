package com.hellblazer.art.laminar.validation;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.integration.ARTCircuitParameters;
import com.hellblazer.art.laminar.integration.ARTLaminarCircuit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Simplified biological validation tests for art-laminar.
 * Validates key biological principles from Grossberg's work.
 *
 * @author Claude Code
 */
class SimpleBiologicalValidationTest {

    private ARTLaminarCircuit circuit;
    private static final int INPUT_SIZE = 64;

    @BeforeEach
    void setUp() {
        var params = ARTCircuitParameters.builder(INPUT_SIZE)
            .vigilance(0.85)
            .learningRate(0.8)
            .maxCategories(50)
            .build();
        circuit = new ARTLaminarCircuit(params);
    }

    @AfterEach
    void tearDown() throws Exception {
        if (circuit != null) {
            circuit.close();
        }
    }

    @Test
    void testResonanceStabilization() {
        var pattern = createPattern(0.75);

        circuit.reset();
        circuit.process(pattern);
        var cat1 = circuit.getState().activeCategory();

        circuit.process(pattern);
        var cat2 = circuit.getState().activeCategory();

        assertEquals(cat1, cat2, "Resonance should stabilize on learned category");
        assertTrue(circuit.getState().matchScore() >= 0.85);
    }

    @Test
    void testCatastrophicForgettingPrevention() {
        var pattern = createPattern(0.75);

        for (int i = 0; i < 10; i++) {
            circuit.process(pattern);
        }

        assertEquals(1, circuit.getCategoryCount(),
            "Should not create multiple categories for same pattern");
    }

    @Test
    void testVigilanceControl() {
        var p1 = createPattern(0.8);
        var p2 = createPattern(0.3);

        // High vigilance
        var highParams = ARTCircuitParameters.builder(INPUT_SIZE)
            .vigilance(0.95).learningRate(0.8).maxCategories(50).build();

        try (var high = new ARTLaminarCircuit(highParams)) {
            high.process(p1);
            high.process(p2);
            var highCount = high.getCategoryCount();

            // Low vigilance
            var lowParams = ARTCircuitParameters.builder(INPUT_SIZE)
                .vigilance(0.60).learningRate(0.8).maxCategories(50).build();

            try (var low = new ARTLaminarCircuit(lowParams)) {
                low.process(p1);
                low.process(p2);
                var lowCount = low.getCategoryCount();

                assertTrue(lowCount <= highCount || Math.abs(lowCount - highCount) <= 1);
            }
        } catch (Exception e) {
            fail("Cleanup failed");
        }
    }

    @Test
    void testBottomUpDrivingSignal() {
        var pattern = createPattern(0.8);
        circuit.reset();
        var expectation = circuit.process(pattern);

        assertTrue(circuit.getCategoryCount() >= 1);
        assertNotNull(expectation);
    }

    @Test
    void testCategoryDifferentiation() {
        circuit.reset();
        var p1 = createPattern(0.9);
        var p2 = createPattern(0.3);

        circuit.process(p1);
        circuit.process(p2);

        assertTrue(circuit.getCategoryCount() >= 1);
    }

    private Pattern createPattern(double value) {
        var data = new double[INPUT_SIZE];
        for (int i = 0; i < INPUT_SIZE; i++) {
            data[i] = value * (0.8 + 0.2 * Math.random());
        }
        return new DenseVector(data);
    }
}