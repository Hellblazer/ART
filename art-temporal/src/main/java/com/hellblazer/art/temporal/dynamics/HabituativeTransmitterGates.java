package com.hellblazer.art.temporal.dynamics;

import java.util.Arrays;

/**
 * Habituative Transmitter Gates implementation
 * Activity-dependent gates that prevent perseveration
 * Based on Equation 7 from Kazerounian & Grossberg (2014)
 */
public class HabituativeTransmitterGates {

    private final float recoveryRate;     // ε parameter - transmitter recovery
    private final float depletionRate;    // λ parameter - linear depletion
    private final float quadraticRate;    // μ parameter - quadratic depletion
    private final float deltaTime;

    // Gate states
    private float[] transmitterLevels;
    private float[] previousActivations;
    private int numGates;

    // Thresholds
    private final float minTransmitter = 0.0f;
    private final float maxTransmitter = 1.0f;
    private final float depletionThreshold = 0.1f;

    public HabituativeTransmitterGates(float recoveryRate, float depletionRate,
                                      float quadraticRate, int numGates, float deltaTime) {
        this.recoveryRate = recoveryRate;
        this.depletionRate = depletionRate;
        this.quadraticRate = quadraticRate;
        this.numGates = numGates;
        this.deltaTime = deltaTime;

        initializeGates();
    }

    /**
     * Update transmitter gates based on neural activations
     * dZi/dt = ε(1-Zi) - Zi(λxi + μxi²)
     *
     * @param activations Current neural activations
     */
    public void updateGates(float[] activations) {
        for (int i = 0; i < Math.min(numGates, activations.length); i++) {
            updateSingleGate(i, activations[i]);
        }

        // Store activations for next update
        System.arraycopy(activations, 0, previousActivations, 0,
                        Math.min(activations.length, numGates));
    }

    /**
     * Update a single transmitter gate
     */
    private void updateSingleGate(int gateIndex, float activation) {
        var currentLevel = transmitterLevels[gateIndex];

        // Recovery term: ε(1-Z)
        var recovery = recoveryRate * (maxTransmitter - currentLevel);

        // Depletion term: Z(λx + μx²)
        var linearDepletion = depletionRate * activation;
        var quadraticDepletion = quadraticRate * activation * activation;
        var totalDepletion = currentLevel * (linearDepletion + quadraticDepletion);

        // Update equation
        var derivative = recovery - totalDepletion;
        var newLevel = currentLevel + derivative * deltaTime;

        // Bound transmitter level
        transmitterLevels[gateIndex] = Math.max(minTransmitter, Math.min(maxTransmitter, newLevel));
    }

    /**
     * Apply gates to input signals
     *
     * @param inputs Input signals to be gated
     * @return Gated output signals
     */
    public float[] applyGates(float[] inputs) {
        var gatedOutputs = new float[inputs.length];

        for (int i = 0; i < inputs.length; i++) {
            if (i < numGates) {
                // Apply transmitter gate
                gatedOutputs[i] = inputs[i] * transmitterLevels[i];
            } else {
                // Pass through if no gate
                gatedOutputs[i] = inputs[i];
            }
        }

        return gatedOutputs;
    }

    /**
     * Check if a gate is depleted (below threshold)
     */
    public boolean isGateDepleted(int gateIndex) {
        return gateIndex < numGates && transmitterLevels[gateIndex] < depletionThreshold;
    }

    /**
     * Check if any gates are depleted
     */
    public boolean hasDepletedGates() {
        for (int i = 0; i < numGates; i++) {
            if (transmitterLevels[i] < depletionThreshold) {
                return true;
            }
        }
        return false;
    }

    /**
     * Reset specific gate to full transmitter level
     */
    public void resetGate(int gateIndex) {
        if (gateIndex < numGates) {
            transmitterLevels[gateIndex] = maxTransmitter;
            previousActivations[gateIndex] = 0;
        }
    }

    /**
     * Reset all gates to full transmitter levels
     */
    public void resetAllGates() {
        Arrays.fill(transmitterLevels, maxTransmitter);
        Arrays.fill(previousActivations, 0);
    }

    /**
     * Partial reset for gates below threshold
     */
    public void resetDepletedGates() {
        for (int i = 0; i < numGates; i++) {
            if (transmitterLevels[i] < depletionThreshold) {
                transmitterLevels[i] = maxTransmitter;
                previousActivations[i] = 0;
            }
        }
    }

    /**
     * Get current transmitter levels
     */
    public float[] getTransmitterLevels() {
        return Arrays.copyOf(transmitterLevels, numGates);
    }

    /**
     * Get transmitter level for specific gate
     */
    public float getTransmitterLevel(int gateIndex) {
        return gateIndex < numGates ? transmitterLevels[gateIndex] : maxTransmitter;
    }

    /**
     * Set transmitter level for specific gate (for testing or initialization)
     */
    public void setTransmitterLevel(int gateIndex, float level) {
        if (gateIndex < numGates) {
            transmitterLevels[gateIndex] = Math.max(minTransmitter, Math.min(maxTransmitter, level));
        }
    }

    /**
     * Compute equilibrium transmitter level for constant activation
     */
    public float computeEquilibrium(float constantActivation) {
        // At equilibrium: dZ/dt = 0
        // ε(1-Z) - Z(λx + μx²) = 0
        // Z = ε / (ε + λx + μx²)

        var denominator = recoveryRate + depletionRate * constantActivation +
                         quadraticRate * constantActivation * constantActivation;

        if (denominator > 0) {
            return recoveryRate / denominator;
        }
        return maxTransmitter;
    }

    /**
     * Estimate time to recovery from depletion
     */
    public float estimateRecoveryTime(int gateIndex) {
        if (gateIndex >= numGates) {
            return 0;
        }

        var currentLevel = transmitterLevels[gateIndex];
        if (currentLevel >= maxTransmitter * 0.95f) {
            return 0;  // Already recovered
        }

        // Simplified estimate assuming no activation
        // dZ/dt ≈ ε(1-Z)
        var timeConstant = 1.0f / recoveryRate;
        var recoveryFraction = (maxTransmitter - currentLevel) / maxTransmitter;

        return (float)(timeConstant * (-Math.log(1 - recoveryFraction * 0.95)));
    }

    // Helper methods

    private void initializeGates() {
        transmitterLevels = new float[numGates];
        previousActivations = new float[numGates];

        // Initialize all gates to full transmitter
        Arrays.fill(transmitterLevels, maxTransmitter);
    }

    // Getters for monitoring

    public int getNumGates() {
        return numGates;
    }

    public float getRecoveryRate() {
        return recoveryRate;
    }

    public float getDepletionRate() {
        return depletionRate;
    }

    public float getQuadraticRate() {
        return quadraticRate;
    }

    public float getAverageTransmitterLevel() {
        var sum = 0.0f;
        for (var level : transmitterLevels) {
            sum += level;
        }
        return sum / numGates;
    }
}