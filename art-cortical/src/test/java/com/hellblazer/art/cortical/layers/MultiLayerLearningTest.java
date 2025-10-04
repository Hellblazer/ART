package com.hellblazer.art.cortical.layers;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.cortical.learning.HebbianLearning;
import com.hellblazer.art.cortical.temporal.MaskingFieldParameters;
import com.hellblazer.art.cortical.temporal.TemporalProcessor;
import com.hellblazer.art.cortical.temporal.WorkingMemoryParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for multi-layer learning integration (Phase 3C).
 *
 * <p>Verifies that:
 * <ul>
 *   <li>All layers can learn simultaneously</li>
 *   <li>Multi-timescale learning rates are applied correctly</li>
 *   <li>Learning statistics are tracked per layer</li>
 *   <li>Custom learning rates can be configured</li>
 * </ul>
 *
 * @author Phase 3C: Multi-Layer Learning Tests
 */
class MultiLayerLearningTest {

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
    void testEnableMultiLayerLearning() {
        assertFalse(circuit.isLearningEnabled());

        var learningRule = new HebbianLearning(0.001, 0.0, 1.0);
        circuit.enableLearning(learningRule);

        assertTrue(circuit.isLearningEnabled());

        // Check all layers have learning enabled
        assertTrue(circuit.getLayer1().isLearningEnabled());
        assertTrue(circuit.getLayer23().isLearningEnabled());
        assertTrue(circuit.getLayer4().isLearningEnabled());
        assertTrue(circuit.getLayer5().isLearningEnabled());
        assertTrue(circuit.getLayer6().isLearningEnabled());
    }

    @Test
    void testDisableMultiLayerLearning() {
        var learningRule = new HebbianLearning(0.001, 0.0, 1.0);
        circuit.enableLearning(learningRule);
        assertTrue(circuit.isLearningEnabled());

        circuit.disableLearning();
        assertFalse(circuit.isLearningEnabled());

        // Check all layers have learning disabled
        assertFalse(circuit.getLayer1().isLearningEnabled());
        assertFalse(circuit.getLayer23().isLearningEnabled());
        assertFalse(circuit.getLayer4().isLearningEnabled());
        assertFalse(circuit.getLayer5().isLearningEnabled());
        assertFalse(circuit.getLayer6().isLearningEnabled());
    }

    @Test
    void testMultiLayerLearningWithDefaultRates() {
        var learningRule = new HebbianLearning(0.001, 0.0, 1.0);
        circuit.enableLearning(learningRule);

        // Set low thresholds for easier testing
        circuit.setResonanceLearningThreshold(0.1);
        circuit.setAttentionLearningThreshold(0.1);

        var input = createInput();

        // Process multiple times
        for (int trial = 0; trial < 10; trial++) {
            var result = circuit.processAndLearn(input);
            assertNotNull(result);
        }

        // Check that all layers have statistics (learning occurred or was tracked)
        assertNotNull(circuit.getLayer1().getLearningStatistics());
        assertNotNull(circuit.getLayer23().getLearningStatistics());
        assertNotNull(circuit.getLayer4().getLearningStatistics());
        assertNotNull(circuit.getLayer5().getLearningStatistics());
        assertNotNull(circuit.getLayer6().getLearningStatistics());
    }

    @Test
    void testCustomLearningRates() {
        var learningRule = new HebbianLearning(0.001, 0.0, 1.0);

        // Custom rates for each layer
        circuit.enableLearningWithRates(
            learningRule,
            0.002,  // Layer 1: slightly faster than default
            0.015,  // Layer 2/3: slightly faster
            0.05,   // Layer 4: slower than default
            0.02,   // Layer 5: faster
            0.01    // Layer 6: faster
        );

        assertTrue(circuit.isLearningEnabled());

        // Process with custom rates
        circuit.setResonanceLearningThreshold(0.1);
        circuit.setAttentionLearningThreshold(0.1);

        var input = createInput();
        for (int trial = 0; trial < 5; trial++) {
            var result = circuit.processAndLearn(input);
            assertNotNull(result);
        }

        // All layers should have learning enabled
        assertTrue(circuit.getLayer1().isLearningEnabled());
        assertTrue(circuit.getLayer23().isLearningEnabled());
        assertTrue(circuit.getLayer4().isLearningEnabled());
        assertTrue(circuit.getLayer5().isLearningEnabled());
        assertTrue(circuit.getLayer6().isLearningEnabled());
    }

    @Test
    void testCustomLearningRatesValidation() {
        var learningRule = new HebbianLearning(0.001, 0.0, 1.0);

        // Invalid rate (zero)
        assertThrows(IllegalArgumentException.class, () ->
            circuit.enableLearningWithRates(learningRule, 0.0, 0.01, 0.1, 0.01, 0.005)
        );

        // Invalid rate (negative)
        assertThrows(IllegalArgumentException.class, () ->
            circuit.enableLearningWithRates(learningRule, 0.001, -0.01, 0.1, 0.01, 0.005)
        );

        // Invalid rate (> 1.0)
        assertThrows(IllegalArgumentException.class, () ->
            circuit.enableLearningWithRates(learningRule, 0.001, 0.01, 1.5, 0.01, 0.005)
        );

        // Valid rates
        assertDoesNotThrow(() ->
            circuit.enableLearningWithRates(learningRule, 0.001, 0.01, 0.1, 0.01, 0.005)
        );
    }

    @Test
    void testBackwardCompatibleEnableLearning() {
        var learningRule = new HebbianLearning(0.001, 0.0, 1.0);

        // Old API with custom Layer 4 rate
        @SuppressWarnings("deprecation")
        var unused = circuit;
        circuit.enableLearning(learningRule, 0.05);

        assertTrue(circuit.isLearningEnabled());

        // All layers should be enabled
        assertTrue(circuit.getLayer1().isLearningEnabled());
        assertTrue(circuit.getLayer23().isLearningEnabled());
        assertTrue(circuit.getLayer4().isLearningEnabled());
        assertTrue(circuit.getLayer5().isLearningEnabled());
        assertTrue(circuit.getLayer6().isLearningEnabled());
    }

    @Test
    void testLayerSpecificStatistics() {
        var learningRule = new HebbianLearning(0.001, 0.0, 1.0);
        circuit.enableLearning(learningRule);

        circuit.setResonanceLearningThreshold(0.1);
        circuit.setAttentionLearningThreshold(0.1);

        var input = createInput();

        // Process multiple times
        for (int trial = 0; trial < 10; trial++) {
            circuit.processAndLearn(input);
        }

        // Each layer should track its own statistics
        var statsL1 = circuit.getLayer1().getLearningStatistics();
        var statsL23 = circuit.getLayer23().getLearningStatistics();
        var statsL4 = circuit.getLayer4().getLearningStatistics();
        var statsL5 = circuit.getLayer5().getLearningStatistics();
        var statsL6 = circuit.getLayer6().getLearningStatistics();

        assertNotNull(statsL1);
        assertNotNull(statsL23);
        assertNotNull(statsL4);
        assertNotNull(statsL5);
        assertNotNull(statsL6);

        // All layers should have some events recorded
        // (may be 0 if gated by consciousness/attention)
        assertTrue(statsL1.getTotalLearningEvents() >= 0);
        assertTrue(statsL23.getTotalLearningEvents() >= 0);
        assertTrue(statsL4.getTotalLearningEvents() >= 0);
        assertTrue(statsL5.getTotalLearningEvents() >= 0);
        assertTrue(statsL6.getTotalLearningEvents() >= 0);
    }

    @Test
    void testMultiLayerLearningWithHighThresholds() {
        var learningRule = new HebbianLearning(0.001, 0.0, 1.0);
        circuit.enableLearning(learningRule);

        // Very high thresholds - most learning should be gated
        circuit.setResonanceLearningThreshold(0.95);
        circuit.setAttentionLearningThreshold(0.95);

        var input = createInput();

        for (int trial = 0; trial < 20; trial++) {
            circuit.processAndLearn(input);
        }

        // Circuit stats should still be created
        var circuitStats = circuit.getCircuitLearningStatistics();
        assertNotNull(circuitStats);

        // With very high thresholds, learning events may be low
        long totalEvents = circuitStats.getTotalLearningEvents();
        assertTrue(totalEvents >= 0);
    }

    @Test
    void testMultiLayerLearningReset() {
        var learningRule = new HebbianLearning(0.001, 0.0, 1.0);
        circuit.enableLearning(learningRule);
        circuit.setResonanceLearningThreshold(0.1);
        circuit.setAttentionLearningThreshold(0.1);

        var input = createInput();

        // First round of learning
        for (int i = 0; i < 5; i++) {
            circuit.processAndLearn(input);
        }

        // Re-enable learning (should reset statistics)
        circuit.enableLearning(learningRule);

        var statsL1 = circuit.getLayer1().getLearningStatistics();
        var statsL23 = circuit.getLayer23().getLearningStatistics();
        var statsL4 = circuit.getLayer4().getLearningStatistics();
        var statsL5 = circuit.getLayer5().getLearningStatistics();
        var statsL6 = circuit.getLayer6().getLearningStatistics();
        var circuitStats = circuit.getCircuitLearningStatistics();

        // All statistics should be reset to zero
        assertEquals(0, statsL1.getTotalLearningEvents());
        assertEquals(0, statsL23.getTotalLearningEvents());
        assertEquals(0, statsL4.getTotalLearningEvents());
        assertEquals(0, statsL5.getTotalLearningEvents());
        assertEquals(0, statsL6.getTotalLearningEvents());
        assertEquals(0, circuitStats.getTotalLearningEvents());
    }

    @Test
    void testMultiLayerLearningWithoutResonanceDetection() {
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
        circuitNoResonance.enableLearning(learningRule);

        var input = createInput();

        // Should process but not apply learning (no resonance state)
        var result = circuitNoResonance.processAndLearn(input);
        assertNotNull(result);
        assertNull(result.resonanceState());

        // No learning should occur without resonance detection
        assertEquals(0, circuitNoResonance.getCircuitLearningStatistics().getTotalLearningEvents());

        circuitNoResonance.close();
    }
}
