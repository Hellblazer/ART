package com.hellblazer.art.laminar.events;

/**
 * Listener interface for circuit events.
 *
 * @author Hal Hildebrand
 */
public interface CircuitEventListener {
    void onResonance(ResonanceEvent event);
    void onReset(ResetEvent event);
    void onCycleComplete(CycleEvent event);
    void onCategoryCreated(CategoryEvent event);
    void onAttentionShift(AttentionEvent event);
}