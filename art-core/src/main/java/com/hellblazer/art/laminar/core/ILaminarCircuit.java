package com.hellblazer.art.laminar.core;

import com.hellblazer.art.core.ARTAlgorithm;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.ILaminarParameters;
import com.hellblazer.art.laminar.events.ICircuitEventListener;
import java.util.List;
import java.util.Map;

/**
 * Core interface for laminar ART circuit implementations.
 * Extends ARTAlgorithm to maintain compatibility with existing infrastructure
 * while adding laminar-specific functionality.
 *
 * @param <P> Parameter type extending ILaminarParameters
 * @author Hal Hildebrand
 */
public interface ILaminarCircuit<P extends ILaminarParameters>
    extends ARTAlgorithm<P> {

    // === Layer Management ===

    /**
     * Add a layer to the circuit at specified depth.
     *
     * @param layer The layer to add
     * @param depth Laminar depth (0 = input layer, higher = deeper processing)
     * @return This circuit for fluent configuration
     */
    ILaminarCircuit<P> addLayer(ILayer layer, int depth);

    /**
     * Get layer at specified depth.
     *
     * @param depth The laminar depth
     * @return The layer at that depth, or null if not present
     */
    ILayer getLayer(int depth);

    /**
     * Get all layers organized by depth.
     *
     * @return Map of depth to layer
     */
    Map<Integer, ILayer> getLayers();

    // === Pathway Management ===

    /**
     * Connect layers with a pathway.
     *
     * @param pathway The pathway to add
     * @return This circuit for fluent configuration
     */
    ILaminarCircuit<P> connectLayers(IPathway pathway);

    /**
     * Get all pathways in the circuit.
     *
     * @return List of all pathways
     */
    List<IPathway> getPathways();

    /**
     * Get pathways connected to a specific layer.
     *
     * @param layerId The layer identifier
     * @return List of connected pathways
     */
    List<IPathway> getPathwaysForLayer(String layerId);

    // === Resonance Control ===

    /**
     * Set the resonance controller for the circuit.
     *
     * @param controller The resonance controller
     * @return This circuit for fluent configuration
     */
    ILaminarCircuit<P> setResonanceController(IResonanceController controller);

    /**
     * Get the current resonance controller.
     *
     * @return The resonance controller
     */
    IResonanceController getResonanceController();

    // === Circuit Dynamics ===

    /**
     * Perform one processing cycle through the circuit.
     *
     * @param input Input pattern
     * @param parameters Circuit parameters
     * @return Processing result with layer activations
     */
    com.hellblazer.art.core.results.LaminarActivationResult processCycle(Pattern input, P parameters);

    /**
     * Check if the circuit has reached resonance.
     *
     * @return true if in resonant state
     */
    boolean isResonant();

    /**
     * Get the current resonance score (0.0 to 1.0).
     *
     * @return Current resonance level
     */
    double getResonanceScore();

    /**
     * Reset circuit to initial state without clearing learned weights.
     */
    void resetActivations();

    // === Monitoring and Events ===

    /**
     * Register a listener for circuit events.
     *
     * @param listener The event listener
     */
    void addListener(ICircuitEventListener listener);

    /**
     * Remove a registered listener.
     *
     * @param listener The listener to remove
     */
    void removeListener(ICircuitEventListener listener);

    /**
     * Get circuit state snapshot for visualization/debugging.
     *
     * @return Current circuit state
     */
    CircuitState getState();
}