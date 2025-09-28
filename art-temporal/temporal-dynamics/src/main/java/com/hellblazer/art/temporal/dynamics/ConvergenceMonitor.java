package com.hellblazer.art.temporal.dynamics;

import com.hellblazer.art.temporal.core.State;
import java.util.ArrayList;
import java.util.List;

/**
 * Monitors convergence of dynamical systems to steady state.
 * Tracks state history and detects when changes become negligible.
 *
 * @param <S> State type
 */
public class ConvergenceMonitor<S extends State> {
    private static final int DEFAULT_HISTORY_SIZE = 10;
    private static final double DEFAULT_TOLERANCE = 1e-6;

    private final List<S> stateHistory;
    private final List<Double> changeHistory;
    private final int maxHistorySize;
    private S previousState;
    private int stepsSinceLastSignificantChange;
    private double averageChange;

    public ConvergenceMonitor() {
        this(DEFAULT_HISTORY_SIZE);
    }

    public ConvergenceMonitor(int maxHistorySize) {
        this.maxHistorySize = maxHistorySize;
        this.stateHistory = new ArrayList<>(maxHistorySize);
        this.changeHistory = new ArrayList<>(maxHistorySize);
        this.stepsSinceLastSignificantChange = 0;
        this.averageChange = Double.MAX_VALUE;
    }

    /**
     * Check if the system has converged to steady state.
     *
     * @param currentState Current state
     * @param tolerance Convergence tolerance
     * @return true if converged
     */
    @SuppressWarnings("unchecked")
    public boolean hasConverged(S currentState, double tolerance) {
        if (previousState == null) {
            previousState = (S) currentState.copy();
            stateHistory.add((S) currentState.copy());
            return false;
        }

        // Calculate change from previous state
        var change = currentState.distance(previousState);
        changeHistory.add(change);

        // Update history
        stateHistory.add((S) currentState.copy());
        if (stateHistory.size() > maxHistorySize) {
            stateHistory.remove(0);
            changeHistory.remove(0);
        }

        // Check for significant change
        if (change < tolerance) {
            stepsSinceLastSignificantChange++;
        } else {
            stepsSinceLastSignificantChange = 0;
        }

        // Update average change
        if (changeHistory.size() >= 3) {
            averageChange = changeHistory.stream()
                .skip(Math.max(0, changeHistory.size() - 3))
                .mapToDouble(Double::doubleValue)
                .average()
                .orElse(Double.MAX_VALUE);
        }

        previousState = (S) currentState.copy();

        // Convergence criteria:
        // 1. Recent changes are all below tolerance
        // 2. Average change is decreasing
        // 3. No significant changes for multiple steps
        return stepsSinceLastSignificantChange >= 3 && averageChange < tolerance;
    }

    /**
     * Check for oscillatory behavior.
     *
     * @return true if oscillations detected
     */
    public boolean isOscillating() {
        if (changeHistory.size() < 4) {
            return false;
        }

        // Look for alternating increase/decrease pattern
        int signChanges = 0;
        for (int i = 1; i < changeHistory.size(); i++) {
            var prev = changeHistory.get(i - 1);
            var curr = changeHistory.get(i);
            if ((prev < curr && i > 1 && changeHistory.get(i - 2) > prev) ||
                (prev > curr && i > 1 && changeHistory.get(i - 2) < prev)) {
                signChanges++;
            }
        }

        return signChanges >= changeHistory.size() / 2;
    }

    /**
     * Reset the convergence monitor.
     */
    public void reset() {
        stateHistory.clear();
        changeHistory.clear();
        previousState = null;
        stepsSinceLastSignificantChange = 0;
        averageChange = Double.MAX_VALUE;
    }

    /**
     * Get the rate of convergence (change per step).
     */
    public double getConvergenceRate() {
        if (changeHistory.size() < 2) {
            return Double.MAX_VALUE;
        }

        // Linear regression on log(change) vs step
        var n = changeHistory.size();
        var sumX = 0.0;
        var sumY = 0.0;
        var sumXY = 0.0;
        var sumX2 = 0.0;

        for (int i = 0; i < n; i++) {
            var x = (double) i;
            var y = Math.log(changeHistory.get(i) + 1e-10);
            sumX += x;
            sumY += y;
            sumXY += x * y;
            sumX2 += x * x;
        }

        var slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        return Math.exp(slope); // Convert back from log scale
    }

    public double getAverageChange() {
        return averageChange;
    }

    public int getStepsSinceLastSignificantChange() {
        return stepsSinceLastSignificantChange;
    }

    public List<Double> getChangeHistory() {
        return new ArrayList<>(changeHistory);
    }
}