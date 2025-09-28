package com.hellblazer.art.temporal.integration;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for TemporalARTParameters.
 */
public class TemporalARTParametersTest {

    @Test
    public void testDefaultParameters() {
        var params = TemporalARTParameters.defaults();

        assertNotNull(params);
        assertEquals(0.75, params.getVigilance(), 0.001);
        assertEquals(0.1, params.getLearningRate(), 0.001);
        assertEquals(0.01, params.getTimeStep(), 0.001);
        assertEquals(100, params.getMaxCategories());
        assertNotNull(params.getWorkingMemoryParameters());
        assertNotNull(params.getMaskingFieldParameters());
    }

    @Test
    public void testSpeechDefaults() {
        var params = TemporalARTParameters.speechDefaults();

        assertEquals(0.7, params.getVigilance(), 0.001);
        assertEquals(0.5, params.getLearningRate(), 0.001);
        assertEquals(0.01, params.getTimeStep(), 0.001);
        assertEquals(200, params.getMaxCategories());
    }

    @Test
    public void testListLearningDefaults() {
        var params = TemporalARTParameters.listLearningDefaults();

        assertEquals(0.8, params.getVigilance(), 0.001);
        assertEquals(0.3, params.getLearningRate(), 0.001);
        assertEquals(0.05, params.getTimeStep(), 0.001);
        assertEquals(100, params.getMaxCategories());
    }

    @Test
    public void testParameterValidation() {
        // Valid parameters
        assertDoesNotThrow(() -> {
            TemporalARTParameters.builder()
                .vigilance(0.9)
                .learningRate(0.5)
                .timeStep(0.02)
                .maxCategories(50)
                .build();
        });

        // Invalid vigilance
        assertThrows(IllegalArgumentException.class, () -> {
            TemporalARTParameters.builder()
                .vigilance(1.5)
                .build();
        });

        // Invalid learning rate
        assertThrows(IllegalArgumentException.class, () -> {
            TemporalARTParameters.builder()
                .learningRate(-0.1)
                .build();
        });

        // Invalid time step
        assertThrows(IllegalArgumentException.class, () -> {
            TemporalARTParameters.builder()
                .timeStep(0.0)
                .build();
        });

        // Invalid max categories
        assertThrows(IllegalArgumentException.class, () -> {
            TemporalARTParameters.builder()
                .maxCategories(0)
                .build();
        });
    }

    @Test
    public void testGetAllParameters() {
        var params = TemporalARTParameters.builder()
            .vigilance(0.85)
            .learningRate(0.25)
            .build();

        var allParams = params.getAllParameters();

        assertNotNull(allParams);
        assertTrue(allParams.containsKey("vigilance"));
        assertTrue(allParams.containsKey("learningRate"));
        assertTrue(allParams.containsKey("timeStep"));

        // Check nested parameters
        assertTrue(allParams.containsKey("wm.capacity"));
        assertTrue(allParams.containsKey("mf.maxItemNodes"));

        assertEquals(0.85, allParams.get("vigilance"), 0.001);
        assertEquals(0.25, allParams.get("learningRate"), 0.001);
    }

    @Test
    public void testGetParameter() {
        var params = TemporalARTParameters.defaults();

        var vigilance = params.getParameter("vigilance");
        assertTrue(vigilance.isPresent());
        assertEquals(0.75, vigilance.get(), 0.001);

        // Test nested parameter
        var wmCapacity = params.getParameter("wm.capacity");
        assertTrue(wmCapacity.isPresent());

        var nonExistent = params.getParameter("nonExistent");
        assertFalse(nonExistent.isPresent());
    }

    @Test
    public void testWithParameter() {
        var params = TemporalARTParameters.defaults();

        // Modify direct parameter
        var modified = (TemporalARTParameters) params.withParameter("vigilance", 0.9);
        assertEquals(0.9, modified.getVigilance(), 0.001);

        // Original unchanged
        assertEquals(0.75, params.getVigilance(), 0.001);

        // Modify nested parameter
        var modifiedNested = (TemporalARTParameters) params.withParameter("wm.capacity", 10.0);
        assertEquals(10.0, modifiedNested.getParameter("wm.capacity").orElse(0.0), 0.001);

        // Test boolean parameter
        var modifiedBool = (TemporalARTParameters) params.withParameter("fastLearning", 1.0);
        assertTrue(modifiedBool.isFastLearning());

        // Test invalid parameter
        assertThrows(IllegalArgumentException.class, () -> {
            params.withParameter("invalidParam", 1.0);
        });
    }

    @Test
    public void testBuilderChaining() {
        var params = TemporalARTParameters.builder()
            .vigilance(0.8)
            .learningRate(0.2)
            .timeStep(0.05)
            .maxCategories(150)
            .matchThreshold(0.6)
            .fastLearning(true)
            .build();

        assertEquals(0.8, params.getVigilance(), 0.001);
        assertEquals(0.2, params.getLearningRate(), 0.001);
        assertEquals(0.05, params.getTimeStep(), 0.001);
        assertEquals(150, params.getMaxCategories());
        assertEquals(0.6, params.getMatchThreshold(), 0.001);
        assertTrue(params.isFastLearning());
    }
}