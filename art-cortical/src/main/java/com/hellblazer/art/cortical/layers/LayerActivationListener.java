package com.hellblazer.art.cortical.layers;

import com.hellblazer.art.core.Pattern;

/**
 * Listener interface for layer activation changes.
 * Enables monitoring and logging of cortical layer dynamics during processing.
 *
 * <p>Use cases:
 * <ul>
 *   <li>Debugging cortical circuit dynamics</li>
 *   <li>Visualization of layer activations over time</li>
 *   <li>Performance monitoring and profiling</li>
 *   <li>Event-driven circuit orchestration</li>
 * </ul>
 *
 * <p>Thread safety: Implementations must be thread-safe if layers
 * process patterns concurrently.
 *
 * @author Migrated from art-laminar to art-cortical (Phase 3, Milestone 3)
 */
@FunctionalInterface
public interface LayerActivationListener {

    /**
     * Called when a layer's activation changes.
     *
     * @param layerId unique identifier for the layer (e.g., "L4", "L2/3")
     * @param oldActivation previous activation pattern (may be null)
     * @param newActivation new activation pattern (never null)
     */
    void onActivationChanged(String layerId, Pattern oldActivation, Pattern newActivation);
}
