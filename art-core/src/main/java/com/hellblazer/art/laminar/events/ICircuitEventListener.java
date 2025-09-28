package com.hellblazer.art.laminar.events;

/**
 * Listener for circuit-level events.
 *
 * @author Hal Hildebrand
 */
public interface ICircuitEventListener {

    void onResonance(ResonanceEvent event);
    void onReset(ResetEvent event);
    void onCategoryCreated(CategoryEvent event);
    void onAttentionShift(AttentionEvent event);
    void onCycleComplete(CycleEvent event);
}