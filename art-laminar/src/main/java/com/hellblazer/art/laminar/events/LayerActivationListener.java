package com.hellblazer.art.laminar.events;

import com.hellblazer.art.core.Pattern;

/**
 * Listener for layer activation changes.
 */
public interface LayerActivationListener {
    void onActivationChanged(String layerId, Pattern oldActivation, Pattern newActivation);
}