package com.hellblazer.art.laminar.attention;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.AttentionParameters;
import com.hellblazer.art.laminar.parameters.GainModulationParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for GainModulator - attention-based multiplicative gain modulation.
 *
 * Tests verify biological accuracy of gain modulation mechanisms:
 * - Multiplicative gain (not additive)
 * - Layer-specific modulation weights
 * - Proper bounds enforcement
 *
 * @author Hal Hildebrand
 */
public class GainModulatorTest {

    private GainModulator modulator;
    private GainModulationParameters defaultParams;

    @BeforeEach
    public void setUp() {
        defaultParams = new GainModulationParameters();
        modulator = new GainModulator();
    }

    /**
     * Test 1: Multiplicative Gain Application
     *
     * Verifies that gain modulation is MULTIPLICATIVE (not additive), which is
     * the biologically correct mechanism for attention.
     *
     * Biological basis: Attention operates through multiplicative gain changes
     * in neural responses (Reynolds & Heeger, 2009; Treue & Martinez-Trujillo, 1999).
     *
     * Formula: modulatedSignal(i) = originalSignal(i) * (1.0 + attentionGain(i) * modulationStrength)
     */
    @Test
    public void testMultiplicativeGainApplication() {
        // Create a signal with varying amplitudes
        var signal = new DenseVector(new double[]{
            0.0, 0.2, 0.5, 0.8, 1.0
        });

        // Create a gain field with varying attention
        double[] gainField = new double[]{
            0.0,   // No attention
            0.5,   // Moderate attention
            1.0,   // Strong attention
            1.5,   // Very strong attention
            2.0    // Maximum attention
        };

        // Apply modulation
        var modulated = modulator.modulateSignal(signal, gainField, defaultParams);
        var modulatedData = ((DenseVector) modulated).data();

        // Verify multiplicative property
        // For signal = 0, modulation should keep it at 0 (multiplicative)
        assertEquals(0.0, modulatedData[0], 0.001,
                     "Zero signal should remain zero (multiplicative, not additive)");

        // For non-zero signals, check multiplicative formula
        var originalData = signal.data();
        for (int i = 1; i < originalData.length; i++) {
            double expected = originalData[i] * (1.0 + gainField[i] * defaultParams.modulationStrength());
            // Clamp to max gain
            expected = Math.min(expected, originalData[i] * defaultParams.maxGain());
            assertEquals(expected, modulatedData[i], 0.001,
                         String.format("Signal[%d] should follow multiplicative formula", i));

            // Verify enhancement - modulated should be >= original
            assertTrue(modulatedData[i] >= originalData[i],
                       String.format("Modulated signal[%d] should be >= original", i));
        }

        // Verify stronger attention produces stronger modulation (for same signal)
        // signal[1] = 0.2 with gain 0.5
        // signal[2] = 0.5 with gain 1.0
        // Normalized by signal amplitude, signal[2] should show more enhancement
        double enhancement1 = modulatedData[1] / originalData[1];
        double enhancement2 = modulatedData[2] / originalData[2];
        assertTrue(enhancement2 >= enhancement1,
                   "Higher attention gain should produce equal or greater enhancement ratio");

        // Verify bounds enforcement - check maximum gain
        assertTrue(modulatedData[4] <= originalData[4] * defaultParams.maxGain(),
                   "Modulated signal should not exceed maxGain");
    }

    /**
     * Test 2: Layer-Specific Modulation
     *
     * Verifies that different cortical layers can have different gain modulation
     * strengths, reflecting their different functional roles.
     *
     * Biological basis: Attention has layer-specific effects in cortex, with
     * stronger modulation in superficial layers (Buffalo et al., 2010).
     *
     * Formula: layerGain = baseGain * attentionLevel * layerWeight
     */
    @Test
    public void testLayerSpecificModulation() {
        // Create parameters with layer-specific weights
        var layerWeights = Map.of(
            "layer1", 1.5,   // Strong modulation in Layer 1 (top-down)
            "layer23", 1.2,  // Moderate-strong in Layer 2/3 (grouping)
            "layer4", 0.8,   // Weaker in Layer 4 (input)
            "layer5", 1.0,   // Baseline in Layer 5
            "layer6", 0.9    // Slightly reduced in Layer 6
        );
        var params = new GainModulationParameters(
            0.5,  // modulationStrength
            0.5,  // minGain
            3.0,  // maxGain
            layerWeights
        );

        // Test gain computation for different layers with same attention level
        double attentionLevel = 1.0;

        double layer1Gain = modulator.computeLayerSpecificGain("layer1", attentionLevel, params);
        double layer23Gain = modulator.computeLayerSpecificGain("layer23", attentionLevel, params);
        double layer4Gain = modulator.computeLayerSpecificGain("layer4", attentionLevel, params);
        double layer5Gain = modulator.computeLayerSpecificGain("layer5", attentionLevel, params);
        double layer6Gain = modulator.computeLayerSpecificGain("layer6", attentionLevel, params);

        // Verify layer hierarchy
        assertTrue(layer1Gain > layer23Gain, "Layer 1 should have highest gain");
        assertTrue(layer23Gain > layer5Gain, "Layer 2/3 should have more gain than Layer 5");
        assertTrue(layer5Gain > layer6Gain, "Layer 5 should have more gain than Layer 6");
        assertTrue(layer6Gain > layer4Gain, "Layer 6 should have more gain than Layer 4");

        // Verify unknown layers get default weight (1.0)
        double unknownGain = modulator.computeLayerSpecificGain("unknown", attentionLevel, params);
        double expectedDefaultGain = attentionLevel * 1.0;  // Default weight is 1.0
        assertEquals(expectedDefaultGain, unknownGain, 0.001,
                     "Unknown layers should use default weight of 1.0");

        // Verify attention level scaling
        double highAttention = 2.0;
        double highLayer1Gain = modulator.computeLayerSpecificGain("layer1", highAttention, params);
        assertTrue(highLayer1Gain > layer1Gain * 1.5,
                   "Higher attention should proportionally increase gain");

        // Test with spatial attention modulation
        var signal = new DenseVector(new double[]{0.5, 0.5, 0.5, 0.5, 0.5});
        var attentionParams = new AttentionParameters();

        // Apply layer-specific spatial modulation
        var modulated = modulator.modulateWithSpatialAttention(
            signal,
            2, 2,  // Center attention
            attentionParams,
            params
        );

        // Verify signal was modulated
        assertNotNull(modulated, "Modulated signal should not be null");
        var modulatedData = ((DenseVector) modulated).data();
        assertEquals(signal.data().length, modulatedData.length,
                     "Modulated signal should have same dimensions");

        // At least some elements should show enhancement
        boolean hasEnhancement = false;
        for (int i = 0; i < modulatedData.length; i++) {
            if (modulatedData[i] > signal.data()[i]) {
                hasEnhancement = true;
                break;
            }
        }
        assertTrue(hasEnhancement, "Spatial attention should enhance at least some signal elements");
    }
}