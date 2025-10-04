package com.hellblazer.art.laminar.parameters;

/**
 * Parameters for laminar ART circuits.
 *
 * @author Hal Hildebrand
 */
public interface LaminarParameters {

    LearningParameters getLearningParameters();
    LaminarShuntingParameters getShuntingParameters();
    ResonanceParameters getResonanceParameters();

    LayerParameters getLayerParameters(String layerId);
    PathwayParameters getPathwayParameters(String pathwayId);

    double getVigilance();
    boolean isComplementCoding();
}