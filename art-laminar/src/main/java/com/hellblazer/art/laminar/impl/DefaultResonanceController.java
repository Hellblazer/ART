package com.hellblazer.art.laminar.impl;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.core.ResonanceController;

/**
 * Default implementation of resonance controller.
 *
 * @author Hal Hildebrand
 */
public class DefaultResonanceController implements ResonanceController {
    private double vigilance = 0.9;
    private boolean adaptiveVigilance = false;
    private double vigilanceIncrement = 0.01;
    private double maxVigilance = 0.99;

    @Override
    public double calculateMatch(Pattern bottomUp, Pattern topDown) {
        if (bottomUp == null || topDown == null) {
            return 0.0;
        }

        // ART match function: |I ∧ F| / |I|
        var dimension = Math.min(bottomUp.dimension(), topDown.dimension());
        var intersection = 0.0;
        var bottomNorm = 0.0;

        for (int i = 0; i < dimension; i++) {
            var bottom = bottomUp.get(i);
            var top = topDown.get(i);
            intersection += Math.min(bottom, top); // Fuzzy AND
            bottomNorm += bottom;
        }

        return bottomNorm > 0 ? intersection / bottomNorm : 0.0;
    }

    @Override
    public boolean shouldReset(double matchScore) {
        return matchScore < vigilance;
    }

    @Override
    public double getVigilance() {
        return vigilance;
    }

    @Override
    public void setVigilance(double vigilance) {
        this.vigilance = Math.max(0.0, Math.min(1.0, vigilance));
    }

    @Override
    public boolean isAdaptiveVigilance() {
        return adaptiveVigilance;
    }

    @Override
    public void adjustVigilance(double matchScore) {
        if (adaptiveVigilance && matchScore < vigilance) {
            vigilance = Math.min(vigilance + vigilanceIncrement, maxVigilance);
        }
    }

    @Override
    public void reset() {
        // Reset to initial vigilance if adaptive
        if (adaptiveVigilance) {
            vigilance = 0.9;
        }
    }
}