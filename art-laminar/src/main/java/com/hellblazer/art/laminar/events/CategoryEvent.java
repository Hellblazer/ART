package com.hellblazer.art.laminar.events;

/**
 * Event fired when a new category is created.
 */
public record CategoryEvent(int getCategoryIndex, double activationLevel) {}