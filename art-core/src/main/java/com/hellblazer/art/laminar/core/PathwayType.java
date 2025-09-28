package com.hellblazer.art.laminar.core;

/**
 * Types of pathways in laminar circuits.
 *
 * @author Hal Hildebrand
 */
public enum PathwayType {
    BOTTOM_UP,      // Feedforward connections
    TOP_DOWN,       // Feedback connections
    HORIZONTAL,     // Lateral connections within layer
    DIAGONAL,       // Skip connections across layers
    MODULATORY      // Gain control connections
}