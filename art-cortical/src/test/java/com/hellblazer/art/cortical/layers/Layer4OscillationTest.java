package com.hellblazer.art.cortical.layers;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.cortical.analysis.OscillationMetrics;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for Layer 4 oscillation analysis integration.
 *
 * <p>Phase 2C: Tests oscillation tracking in Layer 4 during pattern processing.
 *
 * @author Phase 2: Oscillatory Dynamics Integration
 */
public class Layer4OscillationTest {

    private static final int LAYER_SIZE = 64;
    private static final double SAMPLING_RATE = 1000.0;  // 1ms timesteps
    private static final int HISTORY_SIZE = 256;

    /**
     * Test Layer 4 with oscillation tracking enabled.
     */
    @Test
    public void testLayer4WithOscillationTracking() {
        var layer = new Layer4("L4", LAYER_SIZE);
        layer.enableOscillationTracking(SAMPLING_RATE, HISTORY_SIZE);

        var params = Layer4Parameters.builder().build();

        // Process oscillatory input (40 Hz gamma)
        var frequency = 40.0;
        for (int t = 0; t < HISTORY_SIZE; t++) {
            double time = t / SAMPLING_RATE;
            var input = generateOscillatoryInput(frequency, time, LAYER_SIZE);

            var result = layer.processBottomUp(input, params);

            // Check activation is computed
            assertNotNull(result, "Layer should produce activation");
            assertEquals(LAYER_SIZE, result.dimension(), "Activation dimension should match layer size");
        }

        // After filling history buffer, oscillation metrics should be available
        var metrics = layer.getOscillationMetrics();
        assertNotNull(metrics, "Oscillation metrics should be available after history filled");

        // Should detect gamma oscillation
        assertTrue(metrics.isGammaOscillation(),
            "Should detect gamma oscillation in 40 Hz input");
        assertEquals(40.0, metrics.dominantFrequency(), 5.0,
            "Dominant frequency should be near 40 Hz");
    }

    /**
     * Test Layer 4 without oscillation tracking (default).
     */
    @Test
    public void testLayer4WithoutOscillationTracking() {
        var layer = new Layer4("L4", LAYER_SIZE);

        var params = Layer4Parameters.builder().build();
        var input = new DenseVector(new double[LAYER_SIZE]);

        var result = layer.processBottomUp(input, params);

        assertNotNull(result, "Layer should produce activation");

        // No oscillation metrics when tracking disabled
        var metrics = layer.getOscillationMetrics();
        assertNull(metrics, "Oscillation metrics should be null when tracking disabled");
    }

    /**
     * Test oscillation tracking can be disabled.
     */
    @Test
    public void testDisableOscillationTracking() {
        var layer = new Layer4("L4", LAYER_SIZE);
        layer.enableOscillationTracking(SAMPLING_RATE, HISTORY_SIZE);

        // Process some patterns
        var params = Layer4Parameters.builder().build();
        for (int i = 0; i < 10; i++) {
            layer.processBottomUp(new DenseVector(new double[LAYER_SIZE]), params);
        }

        // Disable tracking
        layer.disableOscillationTracking();

        // No metrics after disabling
        var metrics = layer.getOscillationMetrics();
        assertNull(metrics, "Oscillation metrics should be null after disabling");
    }

    /**
     * Test oscillation metrics update on each timestep.
     */
    @Test
    public void testOscillationMetricsUpdate() {
        var layer = new Layer4("L4", LAYER_SIZE);
        layer.enableOscillationTracking(SAMPLING_RATE, HISTORY_SIZE);

        var params = Layer4Parameters.builder().build();
        var frequency = 40.0;

        // Fill history buffer
        for (int t = 0; t < HISTORY_SIZE; t++) {
            double time = t / SAMPLING_RATE;
            var input = generateOscillatoryInput(frequency, time, LAYER_SIZE);
            layer.processBottomUp(input, params);
        }

        var metrics1 = layer.getOscillationMetrics();
        assertNotNull(metrics1, "First metrics should be available");

        // Process one more timestep
        var time2 = HISTORY_SIZE / SAMPLING_RATE;
        var input2 = generateOscillatoryInput(frequency, time2, LAYER_SIZE);
        layer.processBottomUp(input2, params);

        var metrics2 = layer.getOscillationMetrics();
        assertNotNull(metrics2, "Second metrics should be available");

        // Timestamps should be different
        assertTrue(metrics2.timestamp() > metrics1.timestamp(),
            "Timestamp should advance");
    }

    /**
     * Test oscillation tracking with reset.
     */
    @Test
    public void testOscillationTrackingReset() {
        var layer = new Layer4("L4", LAYER_SIZE);
        layer.enableOscillationTracking(SAMPLING_RATE, HISTORY_SIZE);

        var params = Layer4Parameters.builder().build();

        // Process some patterns
        for (int i = 0; i < HISTORY_SIZE; i++) {
            layer.processBottomUp(new DenseVector(new double[LAYER_SIZE]), params);
        }

        assertNotNull(layer.getOscillationMetrics(), "Metrics should exist before reset");

        // Reset layer
        layer.reset();

        // Metrics should be cleared
        var metricsAfterReset = layer.getOscillationMetrics();
        assertNull(metricsAfterReset, "Metrics should be null after reset");
    }

    /**
     * Test phase tracking over time.
     */
    @Test
    public void testPhaseTracking() {
        var layer = new Layer4("L4", LAYER_SIZE);
        layer.enableOscillationTracking(SAMPLING_RATE, HISTORY_SIZE);

        var params = Layer4Parameters.builder().build();
        var frequency = 40.0;

        // Fill history with oscillatory input
        for (int t = 0; t < HISTORY_SIZE + 10; t++) {
            double time = t / SAMPLING_RATE;
            var input = generateOscillatoryInput(frequency, time, LAYER_SIZE);
            layer.processBottomUp(input, params);
        }

        var metrics = layer.getOscillationMetrics();
        assertNotNull(metrics, "Metrics should be available");

        // Phase should be in valid range
        assertTrue(metrics.phase() >= -Math.PI && metrics.phase() <= Math.PI,
            "Phase should be in [-π, π]");
    }

    /**
     * Test constant input produces low frequency.
     */
    @Test
    public void testConstantInput() {
        var layer = new Layer4("L4", LAYER_SIZE);
        layer.enableOscillationTracking(SAMPLING_RATE, HISTORY_SIZE);

        var params = Layer4Parameters.builder().build();

        // Create constant input
        var constantData = new double[LAYER_SIZE];
        for (int i = 0; i < LAYER_SIZE; i++) {
            constantData[i] = 0.5;  // Constant 0.5
        }
        var constantInput = new DenseVector(constantData);

        for (int t = 0; t < HISTORY_SIZE; t++) {
            layer.processBottomUp(constantInput, params);
        }

        var metrics = layer.getOscillationMetrics();
        assertNotNull(metrics, "Metrics should be available");

        // Constant input should have very low dominant frequency
        assertTrue(metrics.dominantFrequency() < 5.0,
            "Constant input should have low dominant frequency, got: " + metrics.dominantFrequency());
    }

    // ============== Helper Methods ==============

    /**
     * Generate oscillatory input pattern.
     */
    private Pattern generateOscillatoryInput(double frequency, double time, int size) {
        var data = new double[size];
        for (int i = 0; i < size; i++) {
            // Pure sine wave (no DC offset)
            data[i] = Math.sin(2 * Math.PI * frequency * time);
        }
        return new DenseVector(data);
    }
}
