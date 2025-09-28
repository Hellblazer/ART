package com.hellblazer.art.laminar.parameters;

import java.io.Serializable;

/**
 * Base interface for laminar circuit parameters.
 *
 * @author Hal Hildebrand
 */
public interface ILaminarParameters extends Serializable {

    /**
     * Get parameters for specific layer.
     *
     * @param layerId Layer identifier
     * @return Layer parameters
     */
    ILayerParameters getLayerParameters(String layerId);

    /**
     * Get parameters for specific pathway.
     *
     * @param pathwayId Pathway identifier
     * @return Pathway parameters
     */
    IPathwayParameters getPathwayParameters(String pathwayId);

    /**
     * Get resonance control parameters.
     *
     * @return Resonance parameters
     */
    IResonanceParameters getResonanceParameters();

    /**
     * Get shunting equation parameters.
     *
     * @return Shunting parameters
     */
    IShuntingParameters getShuntingParameters();

    /**
     * Get learning parameters.
     *
     * @return Learning parameters
     */
    ILearningParameters getLearningParameters();

    /**
     * Validate parameter consistency.
     *
     * @return true if parameters are valid
     */
    boolean validate();

    /**
     * Create a deep copy of parameters.
     *
     * @return Parameter copy
     */
    ILaminarParameters copy();
}