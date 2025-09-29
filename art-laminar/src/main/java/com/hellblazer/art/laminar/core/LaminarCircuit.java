package com.hellblazer.art.laminar.core;

import com.hellblazer.art.core.ARTAlgorithm;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.results.LaminarActivationResult;
import com.hellblazer.art.laminar.events.CircuitEventListener;
import com.hellblazer.art.laminar.parameters.LaminarParameters;

import java.util.List;
import java.util.Map;

/**
 * Core interface for laminar ART circuits implementing Grossberg's cortical dynamics.
 *
 * @author Hal Hildebrand
 */
public interface LaminarCircuit<P extends LaminarParameters> extends ARTAlgorithm<P> {

    // Layer management
    LaminarCircuit<P> addLayer(Layer layer, int depth);
    Layer getLayer(int depth);
    Map<Integer, Layer> getLayers();

    // Pathway management
    LaminarCircuit<P> connectLayers(Pathway pathway);
    List<Pathway> getPathways();
    List<Pathway> getPathwaysForLayer(String layerId);

    // Processing
    LaminarActivationResult processCycle(Pattern input, P parameters);

    // Resonance control
    LaminarCircuit<P> setResonanceController(ResonanceController controller);
    ResonanceController getResonanceController();
    boolean isResonant();
    double getResonanceScore();

    // Circuit control
    void resetActivations();
    CircuitState getState();

    // Event handling
    void addListener(CircuitEventListener listener);
    void removeListener(CircuitEventListener listener);
}