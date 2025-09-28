package com.hellblazer.art.laminar.events;

import com.hellblazer.art.core.Pattern;

/**
 * Listener for layer activation events.
 *
 * @author Hal Hildebrand
 */
public interface ILayerActivationListener {

    void onActivationChange(String layerId, Pattern oldActivation,
                           Pattern newActivation, long timestamp);
    void onThresholdReached(String layerId, Pattern activation);
    void onSaturation(String layerId, double saturationLevel);
}