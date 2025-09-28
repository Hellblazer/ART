package com.hellblazer.art.temporal.dynamics;

/**
 * Shunting dynamics calculator for on-center off-surround networks
 * Implements the core differential equations from Kazerounian & Grossberg (2014)
 */
public class ShuntingDynamics {

    private final float decayRate;      // A parameter
    private final float upperBound;     // B parameter
    private final float lowerBound;     // C parameter
    private final float deltaTime;

    public ShuntingDynamics(float decayRate, float upperBound, float lowerBound, float deltaTime) {
        this.decayRate = decayRate;
        this.upperBound = upperBound;
        this.lowerBound = lowerBound;
        this.deltaTime = deltaTime;
    }

    /**
     * Update activation using shunting equation
     * dx/dt = -Ax + (B-x)E - (x+C)I
     *
     * @param currentActivation Current activation value
     * @param excitation Excitatory input
     * @param inhibition Inhibitory input
     * @return Updated activation
     */
    public float updateActivation(float currentActivation, float excitation, float inhibition) {
        var decay = -decayRate * currentActivation;
        var excitatory = (upperBound - currentActivation) * excitation;
        var inhibitory = (currentActivation + lowerBound) * inhibition;

        var derivative = decay + excitatory - inhibitory;
        var newActivation = currentActivation + derivative * deltaTime;

        // Bound the activation
        return Math.max(0, Math.min(upperBound, newActivation));
    }

    /**
     * Update a vector of activations
     */
    public void updateActivations(float[] activations, float[] excitations, float[] inhibitions) {
        for (int i = 0; i < activations.length; i++) {
            var excitation = i < excitations.length ? excitations[i] : 0;
            var inhibition = i < inhibitions.length ? inhibitions[i] : 0;
            activations[i] = updateActivation(activations[i], excitation, inhibition);
        }
    }

    /**
     * Update with on-center off-surround architecture
     */
    public float updateWithSurround(float currentActivation, float selfExcitation,
                                   float input, float surroundInhibition) {
        var totalExcitation = selfExcitation + input;
        return updateActivation(currentActivation, totalExcitation, surroundInhibition);
    }

    /**
     * Compute equilibrium activation given constant inputs
     */
    public float computeEquilibrium(float excitation, float inhibition) {
        // At equilibrium: dx/dt = 0
        // Solving: -Ax + (B-x)E - (x+C)I = 0
        // x(A + E + I) = BE - CI
        // x = (BE - CI) / (A + E + I)

        var numerator = upperBound * excitation - lowerBound * inhibition;
        var denominator = decayRate + excitation + inhibition;

        if (denominator > 0) {
            var equilibrium = numerator / denominator;
            return Math.max(0, Math.min(upperBound, equilibrium));
        }
        return 0;
    }

    /**
     * Check if activation has reached steady state
     */
    public boolean isAtSteadyState(float activation, float excitation, float inhibition, float tolerance) {
        var equilibrium = computeEquilibrium(excitation, inhibition);
        return Math.abs(activation - equilibrium) < tolerance;
    }

    // Getters for parameters
    public float getDecayRate() { return decayRate; }
    public float getUpperBound() { return upperBound; }
    public float getLowerBound() { return lowerBound; }
    public float getDeltaTime() { return deltaTime; }
}