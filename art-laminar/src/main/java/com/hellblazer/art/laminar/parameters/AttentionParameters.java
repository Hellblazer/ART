package com.hellblazer.art.laminar.parameters;

/**
 * Parameters for attention control mechanisms in the canonical cortical circuit.
 *
 * Implements spatial, feature-based, and object-based attention following
 * Grossberg's LAMINART model and modern attention theories.
 *
 * Biological basis:
 * - Spatial attention: Gaussian gain fields (Reynolds & Heeger, 2009)
 * - Feature attention: Similarity-based enhancement (Maunsell & Treue, 2006)
 * - Object attention: Template matching (Scholl, 2001)
 *
 * @author Hal Hildebrand
 *
 * @param spatialSigma Gaussian spread for spatial attention (default: 5.0).
 *                     Controls how quickly attention falls off with distance.
 *                     Larger values = broader attention window.
 *
 * @param maxSpatialGain Maximum multiplicative gain for spatial attention (default: 2.0).
 *                       Biological range: 1.5-3.0 based on neurophysiology.
 *
 * @param featureAlpha Feature similarity enhancement factor (default: 1.0).
 *                     Controls strength of feature-based attention.
 *
 * @param maxFeatureGain Maximum multiplicative gain for feature attention (default: 1.5).
 *                       Typically weaker than spatial attention.
 *
 * @param objectBeta Object template matching weight (default: 1.5).
 *                   Controls strength of object-based attention.
 *
 * @param maxObjectGain Maximum multiplicative gain for object attention (default: 2.0).
 *                      Similar to spatial attention strength.
 *
 * @param attentionDecayRate Decay rate when location/feature is not attended (default: 0.1).
 *                           Controls how quickly attention fades.
 *
 * @param shiftSpeed Speed of attention shifting between locations (default: 0.3).
 *                   Higher values = faster shifts. Range: 0.1-0.5.
 */
public record AttentionParameters(
    double spatialSigma,
    double maxSpatialGain,
    double featureAlpha,
    double maxFeatureGain,
    double objectBeta,
    double maxObjectGain,
    double attentionDecayRate,
    double shiftSpeed
) {
    /**
     * Default constructor with biologically plausible parameters.
     */
    public AttentionParameters() {
        this(5.0, 2.0, 1.0, 1.5, 1.5, 2.0, 0.1, 0.3);
    }

    /**
     * Validate parameters are in acceptable ranges.
     */
    public AttentionParameters {
        if (spatialSigma <= 0) {
            throw new IllegalArgumentException("spatialSigma must be positive");
        }
        if (maxSpatialGain < 1.0) {
            throw new IllegalArgumentException("maxSpatialGain must be >= 1.0");
        }
        if (featureAlpha < 0) {
            throw new IllegalArgumentException("featureAlpha must be non-negative");
        }
        if (maxFeatureGain < 1.0) {
            throw new IllegalArgumentException("maxFeatureGain must be >= 1.0");
        }
        if (objectBeta < 0) {
            throw new IllegalArgumentException("objectBeta must be non-negative");
        }
        if (maxObjectGain < 1.0) {
            throw new IllegalArgumentException("maxObjectGain must be >= 1.0");
        }
        if (attentionDecayRate < 0 || attentionDecayRate > 1.0) {
            throw new IllegalArgumentException("attentionDecayRate must be in [0, 1]");
        }
        if (shiftSpeed <= 0 || shiftSpeed > 1.0) {
            throw new IllegalArgumentException("shiftSpeed must be in (0, 1]");
        }
    }
}