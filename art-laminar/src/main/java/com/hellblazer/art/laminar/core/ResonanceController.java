package com.hellblazer.art.laminar.core;

import com.hellblazer.art.core.Pattern;

/**
 * Interface for controlling resonance in a laminar circuit.
 *
 * @author Hal Hildebrand
 */
public interface ResonanceController {

    // Match calculation
    double calculateMatch(Pattern bottomUp, Pattern topDown);

    // Reset control
    boolean shouldReset(double matchScore);

    // Vigilance management
    double getVigilance();
    void setVigilance(double vigilance);

    // Adaptive vigilance
    boolean isAdaptiveVigilance();
    void adjustVigilance(double matchScore);

    // State
    void reset();
}