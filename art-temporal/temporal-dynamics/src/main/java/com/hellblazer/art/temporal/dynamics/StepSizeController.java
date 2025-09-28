package com.hellblazer.art.temporal.dynamics;

import com.hellblazer.art.temporal.core.DynamicalSystem;

/**
 * Adaptive step size controller for numerical integration.
 * Adjusts time steps based on error estimates and system time scales.
 */
public class StepSizeController {
    private static final double SAFETY_FACTOR = 0.9;
    private static final double MAX_INCREASE_FACTOR = 2.0;
    private static final double MAX_DECREASE_FACTOR = 0.1;
    private static final double ERROR_EXPONENT = 0.2; // For 4th order methods

    private final double minStepSize;
    private final double maxStepSize;
    private final double initialStepSize;
    private double previousError;
    private int consecutiveGoodSteps;

    public StepSizeController(DynamicalSystem.TimeScale timeScale) {
        // Set step sizes based on characteristic time scale
        var typicalTime = timeScale.getTypicalMillis() / 1000.0; // Convert to seconds
        this.minStepSize = typicalTime * 0.001;  // 0.1% of typical time
        this.maxStepSize = typicalTime * 0.1;    // 10% of typical time
        this.initialStepSize = typicalTime * 0.01; // 1% of typical time
        this.previousError = 1.0;
        this.consecutiveGoodSteps = 0;
    }

    public StepSizeController(double minStep, double maxStep, double initialStep) {
        this.minStepSize = minStep;
        this.maxStepSize = maxStep;
        this.initialStepSize = initialStep;
        this.previousError = 1.0;
        this.consecutiveGoodSteps = 0;
    }

    /**
     * Get the initial step size for integration.
     */
    public double getInitialStepSize() {
        return initialStepSize;
    }

    /**
     * Calculate new step size based on error estimate.
     *
     * @param currentStepSize Current step size
     * @param errorRatio Ratio of actual error to desired tolerance
     * @return Adjusted step size
     */
    public double adjustStepSize(double currentStepSize, double errorRatio) {
        double factor;

        if (errorRatio < 1.0) {
            // Error is acceptable, can increase step size
            consecutiveGoodSteps++;

            // Be more aggressive with increases after consistent success
            if (consecutiveGoodSteps > 5) {
                factor = Math.min(MAX_INCREASE_FACTOR, Math.pow(errorRatio, -ERROR_EXPONENT));
            } else {
                factor = Math.min(1.5, Math.pow(errorRatio, -ERROR_EXPONENT * 0.5));
            }
        } else {
            // Error too large, must decrease step size
            consecutiveGoodSteps = 0;
            factor = Math.max(MAX_DECREASE_FACTOR, SAFETY_FACTOR * Math.pow(errorRatio, -ERROR_EXPONENT));
        }

        // Apply PI controller for smoother adaptation
        if (previousError > 0) {
            var piCorrection = Math.sqrt(previousError / errorRatio);
            factor *= piCorrection;
        }

        previousError = errorRatio;

        // Apply bounds
        var newStepSize = currentStepSize * factor;
        return Math.min(maxStepSize, Math.max(minStepSize, newStepSize));
    }

    /**
     * Increase step size when error is very small.
     */
    public double increaseStepSize(double currentStepSize) {
        consecutiveGoodSteps++;
        var factor = consecutiveGoodSteps > 3 ? 1.5 : 1.2;
        return Math.min(maxStepSize, currentStepSize * factor);
    }

    /**
     * Decrease step size when error is too large.
     */
    public double decreaseStepSize(double currentStepSize, double errorRatio) {
        consecutiveGoodSteps = 0;
        var factor = SAFETY_FACTOR / Math.pow(errorRatio, ERROR_EXPONENT);
        return Math.max(minStepSize, currentStepSize * factor);
    }

    /**
     * Get minimum allowed step size.
     */
    public double getMinStepSize() {
        return minStepSize;
    }

    /**
     * Get maximum allowed step size.
     */
    public double getMaxStepSize() {
        return maxStepSize;
    }

    /**
     * Reset the controller state.
     */
    public void reset() {
        previousError = 1.0;
        consecutiveGoodSteps = 0;
    }

    /**
     * Estimate optimal initial step size using derivative information.
     */
    public static double estimateInitialStepSize(double stateNorm, double derivativeNorm,
                                                 double tolerance) {
        if (derivativeNorm < 1e-10) {
            return 1.0; // System at equilibrium
        }

        // Estimate based on local linearity assumption
        var h0 = tolerance / derivativeNorm;

        // Apply safety factor
        return SAFETY_FACTOR * h0;
    }
}