package com.hellblazer.art.laminar.core;

/**
 * Layer types for the laminar circuit.
 *
 * @author Hal Hildebrand
 */
public enum LayerType {
    INPUT,          // F0: Input preprocessing layer
    FEATURE,        // F1: Feature representation layer
    CATEGORY,       // F2: Category representation layer
    ATTENTION,      // Attentional gain control layer
    EXPECTATION,    // Top-down expectation layer
    BOUNDARY,       // Boundary completion layer
    SURFACE,        // Surface filling-in layer
    CUSTOM          // User-defined layer type
}