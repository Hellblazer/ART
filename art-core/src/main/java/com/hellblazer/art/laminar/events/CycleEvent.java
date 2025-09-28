package com.hellblazer.art.laminar.events;

import java.io.Serializable;

public class CycleEvent implements Serializable {
    private static final long serialVersionUID = 1L;
    private final int cycleNumber;
    private final double resonanceScore;
    private final long timestamp = System.currentTimeMillis();

    public CycleEvent(int cycleNumber, double resonanceScore) {
        this.cycleNumber = cycleNumber;
        this.resonanceScore = resonanceScore;
    }

    public int getCycleNumber() { return cycleNumber; }
    public double getResonanceScore() { return resonanceScore; }
    public long getTimestamp() { return timestamp; }
}