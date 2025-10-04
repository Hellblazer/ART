package com.hellblazer.art.laminar.parameters;

/**
 * Parameters for resonance control.
 *
 * @author Hal Hildebrand
 */
public interface ResonanceParameters {
    double getInitialVigilance();
    double getVigilanceIncrement();
    double getMaxVigilance();
    boolean isAdaptiveVigilance();
}