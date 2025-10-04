package com.hellblazer.art.laminar.parameters;

import java.util.Map;

/**
 * Parameters for attention-based gain modulation in cortical pathways.
 *
 * Implements multiplicative gain modulation, the biologically correct mechanism
 * for attention (Reynolds & Heeger, 2009; Treue & Martinez-Trujillo, 1999).
 *
 * Layer-specific modulation reflects the finding that attention has different
 * effects in different cortical layers (Buffalo et al., 2010).
 *
 * @author Hal Hildebrand
 *
 * @param modulationStrength Overall strength of gain modulation (default: 0.5).
 *                          Controls how much attention affects neural responses.
 *                          Range: 0.0-1.0.
 *
 * @param minGain Minimum multiplicative gain (default: 0.5).
 *                Prevents complete suppression of unattended signals.
 *
 * @param maxGain Maximum multiplicative gain (default: 3.0).
 *                Caps enhancement to biologically realistic levels.
 *                Neurophysiology shows 2-4x enhancement is typical.
 *
 * @param layerWeights Layer-specific modulation weights (default: empty map = uniform 1.0).
 *                    Maps layer IDs to relative modulation strengths.
 *                    Example: Layer 1 (top-down) might have weight 1.5,
 *                            Layer 4 (input) might have weight 0.8.
 */
public record GainModulationParameters(
    double modulationStrength,
    double minGain,
    double maxGain,
    Map<String, Double> layerWeights
) {
    /**
     * Default constructor with standard parameters.
     */
    public GainModulationParameters() {
        this(0.5, 0.5, 3.0, Map.of());
    }

    /**
     * Validate parameters are in acceptable ranges.
     */
    public GainModulationParameters {
        if (modulationStrength < 0 || modulationStrength > 1.0) {
            throw new IllegalArgumentException("modulationStrength must be in [0, 1]");
        }
        if (minGain < 0 || minGain > 1.0) {
            throw new IllegalArgumentException("minGain must be in [0, 1]");
        }
        if (maxGain < 1.0) {
            throw new IllegalArgumentException("maxGain must be >= 1.0");
        }
        if (maxGain < minGain) {
            throw new IllegalArgumentException("maxGain must be >= minGain");
        }
        // Validate layer weights are positive
        if (layerWeights != null) {
            for (var entry : layerWeights.entrySet()) {
                if (entry.getValue() < 0) {
                    throw new IllegalArgumentException(
                        "Layer weight for " + entry.getKey() + " must be non-negative"
                    );
                }
            }
        }
    }

    /**
     * Get the modulation weight for a specific layer.
     * Returns 1.0 for unknown layers (default weight).
     */
    public double getLayerWeight(String layerId) {
        if (layerWeights == null) {
            return 1.0;
        }
        return layerWeights.getOrDefault(layerId, 1.0);
    }
}