package com.hellblazer.art.laminar.attention;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.AttentionParameters;
import com.hellblazer.art.laminar.parameters.GainModulationParameters;

/**
 * GainModulator - Applies attention-based multiplicative gain modulation to pathway signals.
 *
 * Implements the biologically correct mechanism for attention: multiplicative gain
 * modulation (Reynolds & Heeger, 2009; Treue & Martinez-Trujillo, 1999).
 *
 * Key properties:
 * - Gain is MULTIPLICATIVE, not additive
 * - Zero signals remain zero (contrast with additive modulation)
 * - Layer-specific modulation weights
 * - Bounded gain values (minGain to maxGain)
 *
 * Gain Formula:
 * modulatedSignal(i) = originalSignal(i) * (1.0 + attentionGain(i) * modulationStrength)
 *
 * Layer-Specific Gain:
 * layerGain = attentionLevel * layerWeight
 *
 * where layerWeight reflects the layer's sensitivity to attention
 * (e.g., Layer 1 top-down = 1.5, Layer 4 input = 0.8).
 *
 * @author Hal Hildebrand
 */
public class GainModulator {

    /**
     * Create a gain modulator.
     */
    public GainModulator() {
        // Stateless - parameters passed to each method
    }

    /**
     * Apply multiplicative gain modulation to a signal.
     *
     * Uses the formula:
     * modulated[i] = signal[i] * (1.0 + gainField[i] * modulationStrength)
     *
     * Key property: Zero signals remain zero (multiplicative, not additive).
     *
     * @param signal Original signal pattern
     * @param gainField Attention gain for each element (typically 0.0 to 2.0)
     * @param params Gain modulation parameters
     * @return Modulated signal with attention-based enhancement
     */
    public Pattern modulateSignal(Pattern signal, double[] gainField, GainModulationParameters params) {
        var signalData = ((DenseVector) signal).data();

        if (signalData.length != gainField.length) {
            throw new IllegalArgumentException(
                "Signal and gain field must have same dimensions: " +
                signalData.length + " vs " + gainField.length
            );
        }

        var modulated = new double[signalData.length];

        for (int i = 0; i < signalData.length; i++) {
            // Apply multiplicative gain formula
            double gain = 1.0 + gainField[i] * params.modulationStrength();

            // Ensure gain is within bounds
            gain = Math.max(params.minGain(), Math.min(params.maxGain(), gain));

            // Apply gain (multiplicative)
            modulated[i] = signalData[i] * gain;
        }

        return new DenseVector(modulated);
    }

    /**
     * Apply gain modulation with spatial attention.
     *
     * Convenience method that computes spatial gain field from attention controller
     * and applies it to the signal.
     *
     * This is useful for spatially-organized layers where attention is focused
     * on specific locations.
     *
     * @param signal Original signal pattern (must match spatial dimensions)
     * @param centerX Center X of spatial attention
     * @param centerY Center Y of spatial attention
     * @param attentionParams Attention parameters for computing spatial gain
     * @param modulationParams Gain modulation parameters
     * @return Modulated signal with spatial attention enhancement
     */
    public Pattern modulateWithSpatialAttention(Pattern signal,
                                                 int centerX, int centerY,
                                                 AttentionParameters attentionParams,
                                                 GainModulationParameters modulationParams) {
        var signalData = ((DenseVector) signal).data();

        // For 1D signal, map to 2D spatial grid (simplified)
        // In full implementation, would need explicit width/height
        int width = (int) Math.sqrt(signalData.length);
        int height = signalData.length / width;

        // Create attention controller for this spatial layout
        var controller = new AttentionController(width, height, attentionParams);
        controller.setAttentionLocation(centerX, centerY);

        // Compute gain field
        var gainField = new double[signalData.length];
        for (int i = 0; i < signalData.length; i++) {
            int x = i % width;
            int y = i / width;
            gainField[i] = controller.computeSpatialGain(x, y);
        }

        // Apply modulation
        return modulateSignal(signal, gainField, modulationParams);
    }

    /**
     * Compute layer-specific gain for attention modulation.
     *
     * Different cortical layers have different sensitivity to attention.
     * This method computes the effective gain for a specific layer.
     *
     * Formula: layerGain = attentionLevel * layerWeight
     *
     * Example layer weights:
     * - Layer 1 (top-down): 1.5 (strong modulation)
     * - Layer 2/3 (grouping): 1.2 (moderate-strong)
     * - Layer 4 (input): 0.8 (weaker modulation)
     * - Layer 5 (output): 1.0 (baseline)
     * - Layer 6 (feedback): 0.9 (slightly reduced)
     *
     * @param layerId Layer identifier (e.g., "layer1", "layer23")
     * @param attentionLevel Current attention level (typically 0.0 to 2.0)
     * @param params Gain modulation parameters with layer weights
     * @return Effective gain for this layer
     */
    public double computeLayerSpecificGain(String layerId, double attentionLevel,
                                           GainModulationParameters params) {
        double layerWeight = params.getLayerWeight(layerId);
        return attentionLevel * layerWeight;
    }

    /**
     * Reset modulator state (currently stateless, but included for API consistency).
     */
    public void reset() {
        // Stateless implementation - nothing to reset
    }

    /**
     * Apply layer-specific modulation to a pathway signal.
     *
     * This is a convenience method that combines layer-specific gain computation
     * with signal modulation.
     *
     * @param signal Original signal
     * @param layerId Layer identifier
     * @param attentionLevel Current attention level
     * @param params Gain modulation parameters
     * @return Modulated signal with layer-specific attention
     */
    public Pattern modulateWithLayerGain(Pattern signal, String layerId,
                                         double attentionLevel,
                                         GainModulationParameters params) {
        var signalData = ((DenseVector) signal).data();

        // Compute uniform gain for this layer
        double layerGain = computeLayerSpecificGain(layerId, attentionLevel, params);

        // Create uniform gain field
        var gainField = new double[signalData.length];
        for (int i = 0; i < gainField.length; i++) {
            gainField[i] = layerGain;
        }

        return modulateSignal(signal, gainField, params);
    }

    /**
     * Apply combined spatial and layer-specific modulation.
     *
     * This integrates both spatial attention (location-specific) and
     * layer-specific modulation weights.
     *
     * @param signal Original signal
     * @param layerId Layer identifier
     * @param centerX Spatial attention center X
     * @param centerY Spatial attention center Y
     * @param attentionParams Attention parameters
     * @param modulationParams Modulation parameters
     * @return Modulated signal with combined attention
     */
    public Pattern modulateWithCombinedAttention(Pattern signal, String layerId,
                                                  int centerX, int centerY,
                                                  AttentionParameters attentionParams,
                                                  GainModulationParameters modulationParams) {
        var signalData = ((DenseVector) signal).data();

        // Compute spatial dimensions
        int width = (int) Math.sqrt(signalData.length);
        int height = signalData.length / width;

        // Create attention controller
        var controller = new AttentionController(width, height, attentionParams);
        controller.setAttentionLocation(centerX, centerY);

        // Get layer-specific weight
        double layerWeight = modulationParams.getLayerWeight(layerId);

        // Compute combined gain field
        var gainField = new double[signalData.length];
        for (int i = 0; i < signalData.length; i++) {
            int x = i % width;
            int y = i / width;
            double spatialGain = controller.computeSpatialGain(x, y);
            // Combine spatial and layer-specific gains
            gainField[i] = spatialGain * layerWeight;
        }

        return modulateSignal(signal, gainField, modulationParams);
    }
}