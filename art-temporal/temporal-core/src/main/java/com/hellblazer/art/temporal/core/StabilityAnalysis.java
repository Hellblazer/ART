package com.hellblazer.art.temporal.core;

import java.util.HashMap;
import java.util.Map;

/**
 * Stability analysis results for dynamical systems.
 */
public class StabilityAnalysis {
    private final double[] eigenvalues;
    private final double spectralRadius;
    private final boolean isStable;
    private final boolean isAsymptoticallyStable;
    private final Map<String, Boolean> additionalConditions;

    private StabilityAnalysis(Matrix jacobian) {
        this.eigenvalues = jacobian.eigenvalues();
        this.spectralRadius = jacobian.spectralRadius();
        this.isStable = checkStability();
        this.isAsymptoticallyStable = checkAsymptoticStability();
        this.additionalConditions = new HashMap<>();
    }

    public static StabilityAnalysis fromJacobian(Matrix jacobian) {
        return new StabilityAnalysis(jacobian);
    }

    private boolean checkStability() {
        // System is stable if all eigenvalues have non-positive real parts
        for (double ev : eigenvalues) {
            if (ev > 1e-10) { // Small tolerance for numerical errors
                return false;
            }
        }
        return true;
    }

    private boolean checkAsymptoticStability() {
        // System is asymptotically stable if all eigenvalues have negative real parts
        for (double ev : eigenvalues) {
            if (ev >= -1e-10) { // Small tolerance for numerical errors
                return false;
            }
        }
        return true;
    }

    public void addCondition(String name, boolean satisfied) {
        additionalConditions.put(name, satisfied);
    }

    public boolean isStable() {
        return isStable;
    }

    public boolean isAsymptoticallyStable() {
        return isAsymptoticallyStable;
    }

    public double getSpectralRadius() {
        return spectralRadius;
    }

    public double[] getEigenvalues() {
        return eigenvalues.clone();
    }

    public double getLargestEigenvalue() {
        double max = Double.NEGATIVE_INFINITY;
        for (double ev : eigenvalues) {
            max = Math.max(max, ev);
        }
        return max;
    }

    public Map<String, Boolean> getAdditionalConditions() {
        return new HashMap<>(additionalConditions);
    }

    public boolean allConditionsSatisfied() {
        if (!isStable) return false;
        return additionalConditions.values().stream().allMatch(b -> b);
    }

    @Override
    public String toString() {
        var sb = new StringBuilder();
        sb.append("StabilityAnalysis {\n");
        sb.append("  Stable: ").append(isStable).append("\n");
        sb.append("  Asymptotically Stable: ").append(isAsymptoticallyStable).append("\n");
        sb.append("  Spectral Radius: ").append(spectralRadius).append("\n");
        sb.append("  Largest Eigenvalue: ").append(getLargestEigenvalue()).append("\n");

        if (!additionalConditions.isEmpty()) {
            sb.append("  Additional Conditions:\n");
            additionalConditions.forEach((name, satisfied) ->
                sb.append("    ").append(name).append(": ").append(satisfied).append("\n"));
        }

        sb.append("}");
        return sb.toString();
    }
}