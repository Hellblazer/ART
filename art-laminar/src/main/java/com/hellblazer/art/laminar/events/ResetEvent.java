package com.hellblazer.art.laminar.events;

/**
 * Event fired when a reset occurs.
 */
public record ResetEvent(int getCategoryIndex, double matchScore, String getReason) {}