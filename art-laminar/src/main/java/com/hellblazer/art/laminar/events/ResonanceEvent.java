package com.hellblazer.art.laminar.events;

/**
 * Event fired when resonance is achieved.
 */
public record ResonanceEvent(int getCategoryIndex, double getMatchScore) {}