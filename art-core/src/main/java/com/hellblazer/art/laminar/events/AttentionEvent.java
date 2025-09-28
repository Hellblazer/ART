package com.hellblazer.art.laminar.events;

import java.io.Serializable;

public class AttentionEvent implements Serializable {
    private static final long serialVersionUID = 1L;
    private final int[] focusedFeatures;
    private final double attentionStrength;
    private final long timestamp = System.currentTimeMillis();

    public AttentionEvent(int[] focusedFeatures, double attentionStrength) {
        this.focusedFeatures = focusedFeatures;
        this.attentionStrength = attentionStrength;
    }

    public int[] getFocusedFeatures() { return focusedFeatures; }
    public double getAttentionStrength() { return attentionStrength; }
    public long getTimestamp() { return timestamp; }
}