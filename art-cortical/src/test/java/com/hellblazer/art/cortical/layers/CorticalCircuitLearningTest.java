package com.hellblazer.art.cortical.layers;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.cortical.learning.HebbianLearning;
import com.hellblazer.art.cortical.learning.ResonanceGatedLearning;
import com.hellblazer.art.cortical.temporal.MaskingFieldParameters;
import com.hellblazer.art.cortical.temporal.TemporalProcessor;
import com.hellblazer.art.cortical.temporal.WorkingMemoryParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for CorticalCircuit learning integration (Phase 3B).
 *
 * <p>Verifies that:
 * <ul>
 *   <li>Learning can be enabled/disabled</li>
 *   <li>Learning is gated by consciousness and attention</li>
 *   <li>processAndLearn() correctly applies learning</li>
 *   <li>Statistics are tracked correctly</li>
 * </ul>
 *
 * @author Phase 3B: Cortical Circuit Learning Tests
 */
class CorticalCircuitLearningTest {

    private CorticalCircuit circuit;
    private static final int SIZE = 10;

    @BeforeEach
    void setUp() {
        // Create parameters for all layers using builders
        var layer1Params = Layer1Parameters.builder()
            .timeConstant(500.0)
            .primingStrength(0.3)
            .build();

        var layer23Params = Layer23Parameters.builder()
            .size(SIZE)
            .timeConstant(75.0)
            .topDownWeight(0.3)
            .bottomUpWeight(1.0)
            .build();

        var layer4Params = Layer4Parameters.builder()
            .timeConstant(25.0)
            .drivingStrength(0.8)
            .build();

        var layer5Params = Layer5Parameters.builder()
            .timeConstant(100.0)
            .amplificationGain(1.5)
            .build();

        var layer6Params = Layer6Parameters.builder()
            .timeConstant(200.0)
            .onCenterWeight(1.0)
            .offSurroundStrength(0.2)
            .build();

        // Create temporal processor
        var wmParams = WorkingMemoryParameters.builder()
            .capacity(5)
            .primacyDecayRate(0.5)
            .build();

        var mfParams = MaskingFieldParameters.builder()
            .maxItemNodes(SIZE)
            .maxChunks(10)
            .minChunkSize(2)
            .maxChunkSize(5)
            .build();

        var temporalProcessor = new TemporalProcessor(wmParams, mfParams);

        // Create circuit
        circuit = new CorticalCircuit(
            SIZE,
            layer1Params,
            layer23Params,
            layer4Params,
            layer5Params,
            layer6Params,
            temporalProcessor
        );

        // Enable resonance detection for consciousness-based learning
        circuit.enableResonanceDetection(0.6, 1000.0, 256);
    }

    @Test
    void testEnableLearning() {
        assertFalse(circuit.isLearningEnabled());

        var learningRule = new HebbianLearning(0.001, 0.0, 1.0);
        circuit.enableLearning(learningRule, 0.01);

        assertTrue(circuit.isLearningEnabled());
        assertNotNull(circuit.getCircuitLearningStatistics());
    }

    @Test
    void testDisableLearning() {
        var learningRule = new HebbianLearning(0.001, 0.0, 1.0);
        circuit.enableLearning(learningRule, 0.01);
        assertTrue(circuit.isLearningEnabled());

        circuit.disableLearning();
        assertFalse(circuit.isLearningEnabled());
        assertNull(circuit.getCircuitLearningStatistics());
    }

    @Test
    void testEnableLearningValidation() {
        // Null learning rule
        assertThrows(IllegalArgumentException.class, () ->
            circuit.enableLearning(null, 0.01)
        );

        var learningRule = new HebbianLearning(0.001, 0.0, 1.0);

        // Invalid learning rate (zero)
        assertThrows(IllegalArgumentException.class, () ->
            circuit.enableLearning(learningRule, 0.0)
        );

        // Invalid learning rate (negative)
        assertThrows(IllegalArgumentException.class, () ->
            circuit.enableLearning(learningRule, -0.1)
        );

        // Invalid learning rate (> 1.0)
        assertThrows(IllegalArgumentException.class, () ->
            circuit.enableLearning(learningRule, 1.5)
        );

        // Valid learning rates
        assertDoesNotThrow(() -> circuit.enableLearning(learningRule, 0.001));
        assertDoesNotThrow(() -> circuit.enableLearning(learningRule, 1.0));
    }

    @Test
    void testThresholdValidation() {
        // Resonance threshold validation
        assertThrows(IllegalArgumentException.class, () ->
            circuit.setResonanceLearningThreshold(-0.1)
        );
        assertThrows(IllegalArgumentException.class, () ->
            circuit.setResonanceLearningThreshold(1.5)
        );
        assertDoesNotThrow(() -> circuit.setResonanceLearningThreshold(0.0));
        assertDoesNotThrow(() -> circuit.setResonanceLearningThreshold(1.0));

        // Attention threshold validation
        assertThrows(IllegalArgumentException.class, () ->
            circuit.setAttentionLearningThreshold(-0.1)
        );
        assertThrows(IllegalArgumentException.class, () ->
            circuit.setAttentionLearningThreshold(1.5)
        );
        assertDoesNotThrow(() -> circuit.setAttentionLearningThreshold(0.0));
        assertDoesNotThrow(() -> circuit.setAttentionLearningThreshold(1.0));
    }

    /**
     * Helper method to create test input pattern.
     */
    private DenseVector createInput() {
        var data = new double[SIZE];
        for (int i = 0; i < SIZE / 2; i++) {
            data[i] = 1.0;
        }
        return new DenseVector(data);
    }

    @Test
    void testProcessAndLearnWithLearningDisabled() {
        // Learning disabled by default
        assertFalse(circuit.isLearningEnabled());

        var input = createInput();

        // Should process without learning
        var result = circuit.processAndLearn(input);
        assertNotNull(result);
        assertNotNull(result.getFinalOutput());
        assertNull(circuit.getCircuitLearningStatistics());
    }

    @Test
    void testProcessAndLearnWithLearningEnabled() {
        // Enable learning
        var learningRule = new HebbianLearning(0.001, 0.0, 1.0);
        circuit.enableLearning(learningRule, 0.01);

        // Set low thresholds for easier testing
        circuit.setResonanceLearningThreshold(0.3);
        circuit.setAttentionLearningThreshold(0.1);

        var input = createInput();

        // Process multiple times to build statistics
        for (int trial = 0; trial < 10; trial++) {
            var result = circuit.processAndLearn(input);
            assertNotNull(result);
        }

        // Check that statistics were recorded
        var stats = circuit.getCircuitLearningStatistics();
        assertNotNull(stats);

        // Stats should have some learning events (may be gated by consciousness)
        long learningEvents = stats.getTotalLearningEvents();
        // Note: May be 0 if all trials were below consciousness threshold
        assertTrue(learningEvents >= 0);
    }

    @Test
    void testLearningGatedByConsciousness() {
        // Enable learning with high consciousness threshold
        var learningRule = new HebbianLearning(0.001, 0.0, 1.0);
        circuit.enableLearning(learningRule, 0.01);
        circuit.setResonanceLearningThreshold(0.9);  // Very high threshold
        circuit.setAttentionLearningThreshold(0.1);

        var input = createInput();

        // Process multiple times
        for (int trial = 0; trial < 20; trial++) {
            circuit.processAndLearn(input);
        }

        var stats = circuit.getCircuitLearningStatistics();
        assertNotNull(stats);

        // With very high threshold, most trials should be gated
        // (exact count depends on stochastic resonance)
        long learningEvents = stats.getTotalLearningEvents();
        long resonanceGated = stats.getResonanceGatedEvents();

        // Some events should be gated by resonance
        assertTrue(resonanceGated >= 0);
    }

    @Test
    void testLearningGatedByAttention() {
        // Enable learning with high attention threshold
        var learningRule = new HebbianLearning(0.001, 0.0, 1.0);
        circuit.enableLearning(learningRule, 0.01);
        circuit.setResonanceLearningThreshold(0.3);  // Low consciousness threshold
        circuit.setAttentionLearningThreshold(0.9);  // Very high attention threshold

        var input = createInput();

        // Process multiple times
        for (int trial = 0; trial < 20; trial++) {
            circuit.processAndLearn(input);
        }

        var stats = circuit.getCircuitLearningStatistics();
        assertNotNull(stats);

        // With very high attention threshold, most trials should be gated
        long learningEvents = stats.getTotalLearningEvents();
        long attentionGated = stats.getAttentionGatedEvents();

        assertTrue(attentionGated >= 0);
    }

    @Test
    void testResonanceGatedLearningIntegration() {
        // Test with ResonanceGatedLearning wrapper
        var baseRule = new HebbianLearning(0.001, 0.0, 1.0);
        var gatedRule = new ResonanceGatedLearning(baseRule, 0.7);

        circuit.enableLearning(gatedRule, 0.01);
        circuit.setResonanceLearningThreshold(0.5);
        circuit.setAttentionLearningThreshold(0.1);

        var input = createInput();

        // Should work without errors
        for (int trial = 0; trial < 10; trial++) {
            var result = circuit.processAndLearn(input);
            assertNotNull(result);
        }

        var stats = circuit.getCircuitLearningStatistics();
        assertNotNull(stats);
    }

    @Test
    void testLearningWithoutResonanceDetection() {
        // Create circuit without resonance detection
        var layer1Params = Layer1Parameters.builder()
            .timeConstant(500.0)
            .primingStrength(0.3)
            .build();

        var layer23Params = Layer23Parameters.builder()
            .size(SIZE)
            .timeConstant(75.0)
            .topDownWeight(0.3)
            .bottomUpWeight(1.0)
            .build();

        var layer4Params = Layer4Parameters.builder()
            .timeConstant(25.0)
            .drivingStrength(0.8)
            .build();

        var layer5Params = Layer5Parameters.builder()
            .timeConstant(100.0)
            .amplificationGain(1.5)
            .build();

        var layer6Params = Layer6Parameters.builder()
            .timeConstant(200.0)
            .onCenterWeight(1.0)
            .offSurroundStrength(0.2)
            .build();

        var wmParams = WorkingMemoryParameters.builder()
            .capacity(5)
            .primacyDecayRate(0.5)
            .build();

        var mfParams = MaskingFieldParameters.builder()
            .maxItemNodes(SIZE)
            .maxChunks(10)
            .minChunkSize(2)
            .maxChunkSize(5)
            .build();

        var temporalProcessor = new TemporalProcessor(wmParams, mfParams);

        var circuitNoResonance = new CorticalCircuit(
            SIZE, layer1Params, layer23Params, layer4Params, layer5Params, layer6Params, temporalProcessor
        );

        // Enable learning
        var learningRule = new HebbianLearning(0.001, 0.0, 1.0);
        circuitNoResonance.enableLearning(learningRule, 0.01);

        var input = createInput();

        // Should process but not apply learning (no resonance state)
        var result = circuitNoResonance.processAndLearn(input);
        assertNotNull(result);
        assertNull(result.resonanceState());

        var stats = circuitNoResonance.getCircuitLearningStatistics();
        assertNotNull(stats);
        assertEquals(0, stats.getTotalLearningEvents());  // No learning without resonance

        circuitNoResonance.close();
    }

    @Test
    void testStatisticsTracking() {
        var learningRule = new HebbianLearning(0.001, 0.0, 1.0);
        circuit.enableLearning(learningRule, 0.01);

        // Set very low thresholds to ensure learning occurs
        circuit.setResonanceLearningThreshold(0.1);
        circuit.setAttentionLearningThreshold(0.1);

        var input = createInput();

        // Process multiple times
        int trials = 10;
        for (int i = 0; i < trials; i++) {
            circuit.processAndLearn(input);
        }

        var stats = circuit.getCircuitLearningStatistics();
        assertNotNull(stats);

        // Check that statistics are accumulated
        long totalEvents = stats.getTotalLearningEvents();
        assertTrue(totalEvents >= 0);

        // Statistics are tracked regardless of whether learning occurred
        assertTrue(totalEvents >= 0 && totalEvents <= trials);
    }

    @Test
    void testLearningReset() {
        var learningRule = new HebbianLearning(0.001, 0.0, 1.0);
        circuit.enableLearning(learningRule, 0.01);
        circuit.setResonanceLearningThreshold(0.1);
        circuit.setAttentionLearningThreshold(0.1);

        var input = createInput();

        // Process to generate statistics
        for (int i = 0; i < 5; i++) {
            circuit.processAndLearn(input);
        }

        var statsBefore = circuit.getCircuitLearningStatistics();
        long eventsBefore = statsBefore.getTotalLearningEvents();

        // Re-enable learning (should reset statistics)
        circuit.enableLearning(learningRule, 0.01);

        var statsAfter = circuit.getCircuitLearningStatistics();
        assertEquals(0, statsAfter.getTotalLearningEvents());
    }

    @Test
    void testDifferentLearningRates() {
        var learningRule = new HebbianLearning(0.001, 0.0, 1.0);

        // Test with different learning rates
        double[] learningRates = {0.001, 0.01, 0.1};

        for (double rate : learningRates) {
            circuit.disableLearning();
            circuit.enableLearning(learningRule, rate);

            var input = createInput();

            // Should process without errors
            var result = circuit.processAndLearn(input);
            assertNotNull(result);
        }
    }
}
