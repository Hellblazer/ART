package com.hellblazer.art.laminar.core;

/**
 * Types of pathways in a laminar circuit.
 *
 * @author Hal Hildebrand
 */
public enum PathwayType {
    BOTTOM_UP("Bottom-up (feedforward) pathway"),
    TOP_DOWN("Top-down (feedback) pathway"),
    LATERAL("Lateral (horizontal) pathway"),
    MODULATORY("Modulatory pathway"),
    CUSTOM("Custom pathway type");

    private final String description;

    PathwayType(String description) {
        this.description = description;
    }

    public String getDescription() {
        return description;
    }
}