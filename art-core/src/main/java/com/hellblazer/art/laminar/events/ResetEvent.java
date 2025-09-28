package com.hellblazer.art.laminar.events;

import java.io.Serializable;

/**
 * Event fired when a reset occurs in the circuit.
 *
 * @author Hal Hildebrand
 */
public class ResetEvent implements Serializable {
    private static final long serialVersionUID = 1L;

    private final int categoryIndex;
    private final String reason;
    private final double matchScore;
    private final long timestamp;

    public ResetEvent(int categoryIndex, String reason, double matchScore) {
        this.categoryIndex = categoryIndex;
        this.reason = reason;
        this.matchScore = matchScore;
        this.timestamp = System.currentTimeMillis();
    }

    public int getCategoryIndex() {
        return categoryIndex;
    }

    public String getReason() {
        return reason;
    }

    public double getMatchScore() {
        return matchScore;
    }

    public long getTimestamp() {
        return timestamp;
    }
}