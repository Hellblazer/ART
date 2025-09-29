package com.hellblazer.art.laminar.events;

/**
 * Event fired when a processing cycle completes.
 */
public record CycleEvent(int getCycleNumber, double resonanceScore) {}