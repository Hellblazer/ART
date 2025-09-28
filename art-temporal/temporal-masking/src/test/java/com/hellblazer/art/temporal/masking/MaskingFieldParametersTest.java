package com.hellblazer.art.temporal.masking;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for MaskingFieldParameters validation and configurations.
 */
public class MaskingFieldParametersTest {

    @Test
    public void testDefaultParameters() {
        var params = MaskingFieldParameters.builder().build();

        assertEquals(50, params.getMaxItemNodes());
        assertEquals(10, params.getMaxChunks());
        assertEquals(3, params.getMinChunkSize());
        assertEquals(7, params.getMaxChunkSize());
        assertEquals(0.05, params.getIntegrationTimeStep(), 0.001);
        assertEquals(0.2, params.getMinChunkInterval(), 0.001);
        assertEquals(3.0, params.getMaxTemporalGap(), 0.001);
        assertTrue(params.isNormalizationEnabled());
        assertTrue(params.isResetAfterChunk());
    }

    @Test
    public void testPhoneNumberDefaults() {
        var params = MaskingFieldParameters.phoneNumberDefaults();

        assertEquals(50, params.getMaxItemNodes());
        assertEquals(3, params.getMinChunkSize());
        assertEquals(4, params.getMaxChunkSize());
        assertEquals(0.05, params.getIntegrationTimeStep(), 0.001);
        assertEquals(0.3, params.getMinChunkInterval(), 0.001);
        assertEquals(2.0, params.getSpatialScale(), 0.001);
        assertEquals(0.6, params.getCompetitionStrength(), 0.001);
        assertEquals(0.35, params.getWinnerThreshold(), 0.001);
        assertTrue(params.isResetAfterChunk());
    }

    @Test
    public void testListLearningDefaults() {
        var params = MaskingFieldParameters.listLearningDefaults();

        assertEquals(50, params.getMaxItemNodes());
        assertEquals(2, params.getMinChunkSize());
        assertEquals(7, params.getMaxChunkSize());
        assertEquals(0.05, params.getIntegrationTimeStep(), 0.001);
        assertEquals(0.2, params.getMinChunkInterval(), 0.001);
        assertEquals(5.0, params.getMaxTemporalGap(), 0.001);
        assertEquals(0.5, params.getCompetitionStrength(), 0.001);
        assertFalse(params.isResetAfterChunk());
    }

    @Test
    public void testParameterValidation() {
        // Valid parameters
        assertDoesNotThrow(() -> {
            MaskingFieldParameters.builder()
                .maxItemNodes(50)
                .minChunkSize(2)
                .maxChunkSize(5)
                .integrationTimeStep(0.05)
                .winnerThreshold(0.5)
                .learningRate(0.1)
                .matchingThreshold(0.8)
                .build();
        });

        // Invalid max item nodes (too small)
        assertThrows(IllegalArgumentException.class, () -> {
            MaskingFieldParameters.builder()
                .maxItemNodes(5)
                .build();
        });

        // Invalid max item nodes (too large)
        assertThrows(IllegalArgumentException.class, () -> {
            MaskingFieldParameters.builder()
                .maxItemNodes(200)
                .build();
        });

        // Invalid chunk size (min > max)
        assertThrows(IllegalArgumentException.class, () -> {
            MaskingFieldParameters.builder()
                .minChunkSize(5)
                .maxChunkSize(3)
                .build();
        });

        // Invalid integration time step
        assertThrows(IllegalArgumentException.class, () -> {
            MaskingFieldParameters.builder()
                .integrationTimeStep(0.2)
                .build();
        });

        // Invalid winner threshold
        assertThrows(IllegalArgumentException.class, () -> {
            MaskingFieldParameters.builder()
                .winnerThreshold(1.5)
                .build();
        });

        // Invalid learning rate
        assertThrows(IllegalArgumentException.class, () -> {
            MaskingFieldParameters.builder()
                .learningRate(1.5)
                .build();
        });
    }

    @Test
    public void testParameterGetters() {
        var params = MaskingFieldParameters.builder()
            .maxItemNodes(40)
            .maxChunks(8)
            .minChunkSize(2)
            .maxChunkSize(6)
            .integrationTimeStep(0.04)
            .minChunkInterval(0.15)
            .maxTemporalGap(4.0)
            .spatialScale(1.5)
            .excitationRange(0.8)
            .inhibitionRange(2.5)
            .competitionStrength(0.4)
            .winnerThreshold(0.25)
            .itemDecayRate(0.04)
            .chunkDecayRate(0.015)
            .maxActivation(0.9)
            .initialActivation(0.4)
            .activationBoost(0.15)
            .activeChunkBoost(0.25)
            .learningRate(0.08)
            .matchingThreshold(0.75)
            .selfExcitation(0.08)
            .normalizationEnabled(false)
            .resetAfterChunk(false)
            .resetDecayFactor(0.25)
            .build();

        assertEquals(40, params.getMaxItemNodes());
        assertEquals(8, params.getMaxChunks());
        assertEquals(2, params.getMinChunkSize());
        assertEquals(6, params.getMaxChunkSize());
        assertEquals(0.04, params.getIntegrationTimeStep(), 0.001);
        assertEquals(0.15, params.getMinChunkInterval(), 0.001);
        assertEquals(4.0, params.getMaxTemporalGap(), 0.001);
        assertEquals(1.5, params.getSpatialScale(), 0.001);
        assertEquals(0.8, params.getExcitationRange(), 0.001);
        assertEquals(2.5, params.getInhibitionRange(), 0.001);
        assertEquals(0.4, params.getCompetitionStrength(), 0.001);
        assertEquals(0.25, params.getWinnerThreshold(), 0.001);
        assertEquals(0.04, params.getItemDecayRate(), 0.001);
        assertEquals(0.015, params.getChunkDecayRate(), 0.001);
        assertEquals(0.9, params.getMaxActivation(), 0.001);
        assertEquals(0.4, params.getInitialActivation(), 0.001);
        assertEquals(0.15, params.getActivationBoost(), 0.001);
        assertEquals(0.25, params.getActiveChunkBoost(), 0.001);
        assertEquals(0.08, params.getLearningRate(), 0.001);
        assertEquals(0.75, params.getMatchingThreshold(), 0.001);
        assertEquals(0.08, params.getSelfExcitation(), 0.001);
        assertFalse(params.isNormalizationEnabled());
        assertFalse(params.isResetAfterChunk());
        assertEquals(0.25, params.getResetDecayFactor(), 0.001);
    }

    @Test
    public void testGetAllParameters() {
        var params = MaskingFieldParameters.builder().build();
        var allParams = params.getAllParameters();

        assertNotNull(allParams);
        assertTrue(allParams.containsKey("maxItemNodes"));
        assertTrue(allParams.containsKey("integrationTimeStep"));
        assertTrue(allParams.containsKey("competitionStrength"));
        assertTrue(allParams.containsKey("learningRate"));

        assertEquals(50.0, allParams.get("maxItemNodes"));
        assertEquals(0.05, allParams.get("integrationTimeStep"));
    }

    @Test
    public void testGetParameter() {
        var params = MaskingFieldParameters.builder()
            .learningRate(0.15)
            .build();

        var learningRate = params.getParameter("learningRate");
        assertTrue(learningRate.isPresent());
        assertEquals(0.15, learningRate.get(), 0.001);

        var nonExistent = params.getParameter("nonExistentParam");
        assertFalse(nonExistent.isPresent());
    }

    @Test
    public void testWithParameter() {
        var params = MaskingFieldParameters.builder().build();

        // Modify single parameter
        var modified = (MaskingFieldParameters) params.withParameter("learningRate", 0.25);
        assertEquals(0.25, modified.getLearningRate(), 0.001);

        // Original should be unchanged
        assertEquals(0.1, params.getLearningRate(), 0.001);

        // Modify boolean parameter
        var modifiedBool = (MaskingFieldParameters) params.withParameter("normalizationEnabled", 0.0);
        assertFalse(modifiedBool.isNormalizationEnabled());

        // Test invalid parameter name
        assertThrows(IllegalArgumentException.class, () -> {
            params.withParameter("invalidParam", 1.0);
        });
    }

    @Test
    public void testTimeScaleRange() {
        var params = MaskingFieldParameters.builder().build();

        // Integration time step should be in masking field range (50-500ms)
        double timeStepMs = params.getIntegrationTimeStep() * 1000;
        assertTrue(timeStepMs >= 10 && timeStepMs <= 100,
                  "Time step should be in masking field range");

        // Chunk interval should be appropriate for time scale
        double intervalMs = params.getMinChunkInterval() * 1000;
        assertTrue(intervalMs >= 50 && intervalMs <= 1000,
                  "Chunk interval should be appropriate for masking field");
    }

    @Test
    public void testCompetitionParameterConsistency() {
        var params = MaskingFieldParameters.builder().build();

        // Excitation range should be smaller than inhibition range
        assertTrue(params.getExcitationRange() < params.getInhibitionRange(),
                  "Excitation should have shorter range than inhibition for Mexican hat");

        // Competition strength should be moderate
        assertTrue(params.getCompetitionStrength() > 0 && params.getCompetitionStrength() < 1,
                  "Competition strength should be in (0, 1)");
    }
}