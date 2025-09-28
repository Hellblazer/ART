package com.hellblazer.art.laminar.parameters;

/**
 * Parameters for resonance control.
 *
 * @author Hal Hildebrand
 */
public interface IResonanceParameters {
    double getVigilance();
    double getMatchThreshold();
    int getMaxSearchCycles();
    boolean useFastLearning();
}