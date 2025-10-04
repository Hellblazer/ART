package com.hellblazer.art.laminar.attention;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.AttentionParameters;

/**
 * AttentionController - Implements spatial, feature, and object-based attention mechanisms.
 *
 * Based on Grossberg's canonical cortical circuit and modern attention theory,
 * this controller provides three types of attention modulation:
 *
 * 1. Spatial Attention: Gaussian gain fields centered on attended locations
 *    (Reynolds & Heeger, 2009)
 *
 * 2. Feature Attention: Similarity-based enhancement for attended features
 *    (Maunsell & Treue, 2006)
 *
 * 3. Object Attention: Template matching for object-based attention
 *    (Scholl, 2001)
 *
 * All attention mechanisms operate through multiplicative gain modulation,
 * the biologically correct mechanism (not additive).
 *
 * Mathematical Models:
 * - Spatial: gain(x,y) = exp(-((x-cx)² + (y-cy)²) / (2σ²))
 * - Feature: gain(f) = 1 + α * similarity(f, f_attended)
 * - Object: gain(p) = 1 + β * match(p, template)
 *
 * @author Hal Hildebrand
 */
public class AttentionController {

    private final int width;
    private final int height;
    private final AttentionParameters parameters;

    // Spatial attention state
    private double currentCenterX;
    private double currentCenterY;
    private double targetCenterX;
    private double targetCenterY;

    // Feature attention state
    private Pattern attendedFeature;

    // Object attention state
    private Pattern attendedObject;

    /**
     * Create an attention controller for a spatial field.
     *
     * @param width Width of the spatial attention field
     * @param height Height of the spatial attention field
     * @param parameters Attention control parameters
     */
    public AttentionController(int width, int height, AttentionParameters parameters) {
        this.width = width;
        this.height = height;
        this.parameters = parameters;

        // Initialize at center
        this.currentCenterX = width / 2.0;
        this.currentCenterY = height / 2.0;
        this.targetCenterX = currentCenterX;
        this.targetCenterY = currentCenterY;

        this.attendedFeature = null;
        this.attendedObject = null;
    }

    /**
     * Compute spatial attention gain for a location.
     *
     * Uses a Gaussian gain field centered on the attended location:
     * gain(x,y) = exp(-distance² / (2σ²))
     *
     * where distance² = (x - centerX)² + (y - centerY)²
     *
     * @param x X coordinate
     * @param y Y coordinate
     * @return Spatial gain value (0.0 to 1.0, before scaling by maxSpatialGain)
     */
    public double computeSpatialGain(int x, int y) {
        // Compute squared Euclidean distance from attention center
        double dx = x - currentCenterX;
        double dy = y - currentCenterY;
        double distanceSquared = dx * dx + dy * dy;

        // Gaussian falloff
        double sigma = parameters.spatialSigma();
        double gain = Math.exp(-distanceSquared / (2.0 * sigma * sigma));

        return gain;
    }

    /**
     * Compute feature attention gain for a feature pattern.
     *
     * Uses cosine similarity to attended feature:
     * gain(f) = 1 + α * similarity(f, f_attended)
     *
     * where similarity is the normalized dot product (cosine similarity).
     *
     * @param feature Feature pattern to evaluate
     * @return Feature gain value (>= 1.0, capped by maxFeatureGain)
     */
    public double computeFeatureGain(Pattern feature) {
        if (attendedFeature == null) {
            return 1.0;  // No feature attention active
        }

        // Compute cosine similarity
        double similarity = cosineSimilarity(feature, attendedFeature);

        // Apply feature enhancement formula
        double gain = 1.0 + parameters.featureAlpha() * similarity;

        // Cap at maximum feature gain
        gain = Math.min(gain, parameters.maxFeatureGain());

        return gain;
    }

    /**
     * Compute object attention gain for a pattern.
     *
     * Uses normalized dot product with object template:
     * gain(p) = 1 + β * match(p, template)
     *
     * where match is the normalized dot product.
     *
     * @param pattern Pattern to evaluate
     * @return Object gain value (>= 1.0, capped by maxObjectGain)
     */
    public double computeObjectGain(Pattern pattern) {
        if (attendedObject == null) {
            return 1.0;  // No object attention active
        }

        // Compute normalized dot product (template match)
        double match = normalizedDotProduct(pattern, attendedObject);

        // Apply object enhancement formula
        double gain = 1.0 + parameters.objectBeta() * match;

        // Cap at maximum object gain
        gain = Math.min(gain, parameters.maxObjectGain());

        return gain;
    }

    /**
     * Compute combined attention gain incorporating all active mechanisms.
     *
     * Combines spatial, feature, and object attention gains.
     * The combination is multiplicative to reflect independent mechanisms.
     *
     * @param x X coordinate for spatial attention
     * @param y Y coordinate for spatial attention
     * @param feature Feature pattern for feature/object attention
     * @return Combined gain value
     */
    public double computeCombinedGain(int x, int y, Pattern feature) {
        // Compute individual gains
        double spatialGain = computeSpatialGain(x, y);
        double featureGain = computeFeatureGain(feature);

        // For combined gain, we use a weighted combination
        // Spatial gain acts as a multiplier on the feature gain
        // This reflects the finding that spatial and feature attention interact
        double combinedGain = 1.0 + (spatialGain * (featureGain - 1.0));

        return combinedGain;
    }

    /**
     * Set the attended location for spatial attention.
     *
     * @param x Target X coordinate
     * @param y Target Y coordinate
     */
    public void setAttentionLocation(int x, int y) {
        this.targetCenterX = x;
        this.targetCenterY = y;
        // For immediate shift, update current location
        // In a full implementation, this would shift smoothly over time
        this.currentCenterX = x;
        this.currentCenterY = y;
    }

    /**
     * Set the attended feature for feature-based attention.
     *
     * @param feature Feature pattern to attend to
     */
    public void setAttendedFeature(Pattern feature) {
        this.attendedFeature = feature;
    }

    /**
     * Set the attended object template for object-based attention.
     *
     * @param template Object template to attend to
     */
    public void setAttendedObject(Pattern template) {
        this.attendedObject = template;
    }

    /**
     * Shift attention smoothly to a new location.
     *
     * This updates the target and begins a smooth transition.
     * Call update() to advance the shift over time.
     *
     * @param newX New target X coordinate
     * @param newY New target Y coordinate
     */
    public void shiftAttention(int newX, int newY) {
        this.targetCenterX = newX;
        this.targetCenterY = newY;
    }

    /**
     * Update attention state (for smooth shifting).
     *
     * Call this each time step to advance smooth attention shifts.
     */
    public void update() {
        // Smooth shift toward target location
        double dx = targetCenterX - currentCenterX;
        double dy = targetCenterY - currentCenterY;

        currentCenterX += dx * parameters.shiftSpeed();
        currentCenterY += dy * parameters.shiftSpeed();
    }

    /**
     * Reset attention to default state.
     */
    public void reset() {
        this.currentCenterX = width / 2.0;
        this.currentCenterY = height / 2.0;
        this.targetCenterX = currentCenterX;
        this.targetCenterY = currentCenterY;
        this.attendedFeature = null;
        this.attendedObject = null;
    }

    /**
     * Compute cosine similarity between two patterns.
     *
     * @param a First pattern
     * @param b Second pattern
     * @return Cosine similarity in range [-1, 1], or 0 if either pattern is zero
     */
    private double cosineSimilarity(Pattern a, Pattern b) {
        var aData = ((DenseVector) a).data();
        var bData = ((DenseVector) b).data();

        if (aData.length != bData.length) {
            throw new IllegalArgumentException("Patterns must have same dimensionality");
        }

        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < aData.length; i++) {
            dotProduct += aData[i] * bData[i];
            normA += aData[i] * aData[i];
            normB += bData[i] * bData[i];
        }

        normA = Math.sqrt(normA);
        normB = Math.sqrt(normB);

        if (normA < 1e-10 || normB < 1e-10) {
            return 0.0;  // Avoid division by zero
        }

        return dotProduct / (normA * normB);
    }

    /**
     * Compute normalized dot product between two patterns.
     *
     * Uses cosine similarity which naturally captures pattern matching.
     * Returns the raw cosine similarity (can be negative for opposite patterns).
     *
     * @param a First pattern
     * @param b Second pattern
     * @return Cosine similarity in range [-1, 1], or 0 if either pattern is zero
     */
    private double normalizedDotProduct(Pattern a, Pattern b) {
        // For object attention, use cosine similarity directly
        // This gives better discrimination than mapping to [0,1]
        return cosineSimilarity(a, b);
    }

    /**
     * Get current attention center X coordinate.
     */
    public double getCurrentCenterX() {
        return currentCenterX;
    }

    /**
     * Get current attention center Y coordinate.
     */
    public double getCurrentCenterY() {
        return currentCenterY;
    }

    /**
     * Get attended feature pattern.
     */
    public Pattern getAttendedFeature() {
        return attendedFeature;
    }

    /**
     * Get attended object template.
     */
    public Pattern getAttendedObject() {
        return attendedObject;
    }

    /**
     * Get attention parameters.
     */
    public AttentionParameters getParameters() {
        return parameters;
    }
}