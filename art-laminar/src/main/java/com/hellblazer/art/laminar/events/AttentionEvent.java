package com.hellblazer.art.laminar.events;

/**
 * Event fired when attention shifts.
 */
public record AttentionEvent(int fromCategory, int toCategory) {}