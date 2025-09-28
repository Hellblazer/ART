package com.hellblazer.art.laminar.events;

import java.io.Serializable;

public class ResonanceEvent implements Serializable {
    private static final long serialVersionUID = 1L;
    private final int categoryIndex;
    private final double matchScore;
    private final long timestamp = System.currentTimeMillis();

    public ResonanceEvent(int categoryIndex, double matchScore) {
        this.categoryIndex = categoryIndex;
        this.matchScore = matchScore;
    }

    public int getCategoryIndex() { return categoryIndex; }
    public double getMatchScore() { return matchScore; }
    public long getTimestamp() { return timestamp; }
}