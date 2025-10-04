package com.hellblazer.art.laminar.core;

/**
 * Types of layers in a laminar cortical circuit.
 *
 * @author Hal Hildebrand
 */
public enum LayerType {
    INPUT("F0 - Input Layer"),
    FEATURE("F1 - Feature Processing Layer"),
    CATEGORY("F2 - Category Representation Layer"),
    ATTENTION("Attention Control Layer"),
    EXPECTATION("Expectation Layer"),
    CUSTOM("Custom Layer Type");

    private final String description;

    LayerType(String description) {
        this.description = description;
    }

    public String getDescription() {
        return description;
    }
}