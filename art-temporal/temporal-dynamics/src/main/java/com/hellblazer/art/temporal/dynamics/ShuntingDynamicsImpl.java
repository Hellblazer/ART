package com.hellblazer.art.temporal.dynamics;

import com.hellblazer.art.temporal.core.ActivationState;

/**
 * Shunting dynamics implementation for competitive neural networks.
 * Implements the equation: dx_i/dt = -A_i * x_i + (B - x_i) * S_i^+ - x_i * S_i^-
 * Based on Grossberg (1973) and used in Kazerounian & Grossberg (2014).
 *
 * This is the baseline scalar implementation.
 * For SIMD-optimized version, see VectorizedShuntingDynamics in temporal-performance module.
 *
 * @author Hal Hildebrand
 */
public class ShuntingDynamicsImpl {

    private final ShuntingParameters parameters;
    private double[] activations;
    private double[] excitatory;
    private double[] inhibitory;
    private int dimension;

    public ShuntingDynamicsImpl(ShuntingParameters parameters, int dimension) {
        this.parameters = parameters;
        this.dimension = dimension;
        this.activations = new double[dimension];
        this.excitatory = new double[dimension];
        this.inhibitory = new double[dimension];
    }

    public ActivationState evolve(ActivationState currentState, double deltaT) {
        var current = currentState.getActivations();
        var result = new double[dimension];

        // Compute shunting dynamics for each unit
        for (int i = 0; i < dimension; i++) {
            // dx_i/dt = -A_i * x_i + (B - x_i) * S_i^+ - x_i * S_i^-
            double decay = parameters.getDecayRate(i);
            double ceiling = parameters.getCeiling();
            double floor = parameters.getFloor();

            double excitation = computeExcitation(i, current);
            double inhibition = computeInhibition(i, current);

            // Shunting equation
            double derivative = -decay * current[i] +
                               (ceiling - current[i]) * excitation -
                               (current[i] - floor) * inhibition;

            // Euler integration
            result[i] = current[i] + deltaT * derivative;

            // Ensure bounds
            result[i] = Math.max(floor, Math.min(ceiling, result[i]));
        }

        return new ActivationState(result);
    }

    public ActivationState getState() {
        return new ActivationState(activations.clone());
    }

    public void setState(ActivationState state) {
        var acts = state.getActivations();
        System.arraycopy(acts, 0, activations, 0, Math.min(acts.length, dimension));
    }

    /**
     * Compute excitatory input for unit i.
     */
    private double computeExcitation(int i, double[] current) {
        double total = 0.0;

        // Self-excitation
        total += parameters.getSelfExcitation() * current[i];

        // Lateral excitation from nearby units
        for (int j = 0; j < dimension; j++) {
            if (i != j) {
                double weight = computeExcitatoryWeight(i, j);
                total += weight * current[j];
            }
        }

        // External input if any
        if (excitatory[i] > 0) {
            total += excitatory[i];
        }

        return Math.max(0, total);  // Rectify
    }

    /**
     * Compute inhibitory input for unit i.
     */
    private double computeInhibition(int i, double[] current) {
        double total = 0.0;

        // Lateral inhibition from all units
        for (int j = 0; j < dimension; j++) {
            if (i != j) {
                double weight = computeInhibitoryWeight(i, j);
                total += weight * current[j];
            }
        }

        // External inhibition if any
        if (inhibitory[i] > 0) {
            total += inhibitory[i];
        }

        return Math.max(0, total);  // Rectify
    }

    /**
     * Compute excitatory weight between units i and j.
     * Uses a Gaussian kernel for local excitation.
     */
    private double computeExcitatoryWeight(int i, int j) {
        double distance = Math.abs(i - j);
        double sigma = parameters.getExcitatoryRange();
        return parameters.getExcitatoryStrength() *
               Math.exp(-distance * distance / (2.0 * sigma * sigma));
    }

    /**
     * Compute inhibitory weight between units i and j.
     * Uses broader inhibition for competitive dynamics.
     */
    private double computeInhibitoryWeight(int i, int j) {
        double distance = Math.abs(i - j);
        double sigma = parameters.getInhibitoryRange();
        return parameters.getInhibitoryStrength() *
               Math.exp(-distance * distance / (2.0 * sigma * sigma));
    }

    /**
     * Set external excitatory input.
     */
    public void setExcitatoryInput(double[] input) {
        System.arraycopy(input, 0, excitatory, 0, Math.min(input.length, dimension));
    }

    /**
     * Set external inhibitory input.
     */
    public void setInhibitoryInput(double[] input) {
        System.arraycopy(input, 0, inhibitory, 0, Math.min(input.length, dimension));
    }

    /**
     * Clear all inputs.
     */
    public void clearInputs() {
        for (int i = 0; i < dimension; i++) {
            excitatory[i] = 0.0;
            inhibitory[i] = 0.0;
        }
    }

    /**
     * Reset all activations to zero.
     */
    public void reset() {
        for (int i = 0; i < dimension; i++) {
            activations[i] = 0.0;
            excitatory[i] = 0.0;
            inhibitory[i] = 0.0;
        }
    }

    /**
     * Get parameter reference.
     */
    public ShuntingParameters getParameters() {
        return parameters;
    }

    /**
     * Get dimension.
     */
    public int getDimension() {
        return dimension;
    }

    /**
     * Get current activations.
     */
    public double[] getActivations() {
        return activations.clone();
    }

    /**
     * Compute total network energy (Lyapunov function).
     */
    public double computeEnergy() {
        double energy = 0.0;

        for (int i = 0; i < dimension; i++) {
            // Decay term
            energy += 0.5 * parameters.getDecayRate(i) * activations[i] * activations[i];

            // Interaction terms
            for (int j = i + 1; j < dimension; j++) {
                double excWeight = computeExcitatoryWeight(i, j);
                double inhWeight = computeInhibitoryWeight(i, j);
                energy -= excWeight * activations[i] * activations[j];
                energy += inhWeight * activations[i] * activations[j];
            }
        }

        return energy;
    }

    /**
     * Check if dynamics have converged.
     */
    public boolean hasConverged(double tolerance) {
        var state = new ActivationState(activations);
        var evolved = evolve(state, parameters.getTimeStep());

        double maxChange = 0.0;
        var newActs = evolved.getActivations();
        for (int i = 0; i < dimension; i++) {
            double change = Math.abs(newActs[i] - activations[i]);
            maxChange = Math.max(maxChange, change);
        }

        return maxChange < tolerance;
    }
}