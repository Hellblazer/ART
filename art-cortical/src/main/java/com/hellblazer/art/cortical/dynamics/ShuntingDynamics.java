package com.hellblazer.art.cortical.dynamics;

/**
 * Shunting dynamics implementation for competitive neural networks.
 * Implements the on-center off-surround equation from Grossberg (1973):
 *
 * <pre>
 * dx_i/dt = -A_i * x_i + (B - x_i) * S_i^+ - (x_i - C) * S_i^-
 * </pre>
 *
 * Where:
 * <ul>
 *   <li>x_i: activation of unit i</li>
 *   <li>A_i: passive decay rate</li>
 *   <li>B: activation ceiling (upper saturation)</li>
 *   <li>C: activation floor (lower saturation)</li>
 *   <li>S_i^+: total excitatory input (on-center)</li>
 *   <li>S_i^-: total inhibitory input (off-surround)</li>
 * </ul>
 *
 * <p>This implements:
 * <ul>
 *   <li>On-center excitation (narrow Gaussian kernel)</li>
 *   <li>Off-surround inhibition (broad Gaussian kernel)</li>
 *   <li>Mexican hat spatial interaction profile</li>
 *   <li>Winner-take-all competitive dynamics</li>
 * </ul>
 *
 * <p>Based on:
 * <ul>
 *   <li>Grossberg, S. (1973). Contour enhancement, short term memory, and
 *       constancies in reverberating neural networks. Studies in Applied Mathematics.</li>
 *   <li>Kazerounian, S., & Grossberg, S. (2014). Real-time learning of predictive
 *       recognition categories that chunk sequences of items stored in working memory.
 *       Frontiers in Psychology.</li>
 * </ul>
 *
 * @author Migrated from art-temporal/temporal-dynamics to art-cortical (Phase 1)
 */
public class ShuntingDynamics implements NeuralDynamics {

    private final ShuntingParameters parameters;
    private final int dimension;
    private final double[] activations;
    private final double[] excitatoryInput;
    private final double[] inhibitoryInput;

    /**
     * Create shunting dynamics with given parameters.
     *
     * @param parameters shunting dynamics parameters
     * @throws IllegalArgumentException if parameters are invalid
     */
    public ShuntingDynamics(ShuntingParameters parameters) {
        this.parameters = parameters;
        this.dimension = parameters.getDimension();
        this.activations = new double[dimension];
        this.excitatoryInput = new double[dimension];
        this.inhibitoryInput = new double[dimension];

        // Initialize activations
        for (var i = 0; i < dimension; i++) {
            activations[i] = parameters.initialActivation();
        }
    }

    @Override
    public double[] update(double timeStep) {
        if (timeStep <= 0) {
            throw new IllegalArgumentException("Time step must be positive: " + timeStep);
        }

        var result = new double[dimension];

        // Compute shunting dynamics for each unit
        for (var i = 0; i < dimension; i++) {
            var decay = parameters.getDecayRate(i);
            var ceiling = parameters.ceiling();
            var floor = parameters.floor();

            var excitation = computeExcitation(i);
            var inhibition = computeInhibition(i);

            // Shunting equation (Grossberg 1973, Equation 1)
            var derivative = -decay * activations[i] +
                            (ceiling - activations[i]) * excitation -
                            (activations[i] - floor) * inhibition;

            // Euler integration
            result[i] = activations[i] + timeStep * derivative;

            // Enforce bounds (saturation at ceiling and floor)
            result[i] = Math.max(floor, Math.min(ceiling, result[i]));
        }

        // Update internal state
        System.arraycopy(result, 0, activations, 0, dimension);

        return result.clone();
    }

    @Override
    public void reset() {
        for (var i = 0; i < dimension; i++) {
            activations[i] = parameters.initialActivation();
            excitatoryInput[i] = 0.0;
            inhibitoryInput[i] = 0.0;
        }
    }

    @Override
    public double[] getActivation() {
        return activations.clone();
    }

    @Override
    public int size() {
        return dimension;
    }

    @Override
    public boolean hasConverged() {
        // Check if activation changes are below tolerance
        var tolerance = 1e-6;
        var tempActivations = activations.clone();

        update(parameters.timeStep());

        var maxChange = 0.0;
        for (var i = 0; i < dimension; i++) {
            var change = Math.abs(activations[i] - tempActivations[i]);
            maxChange = Math.max(maxChange, change);
        }

        // Restore previous state
        System.arraycopy(tempActivations, 0, activations, 0, dimension);

        return maxChange < tolerance;
    }

    /**
     * Set external excitatory input for all units.
     */
    public void setExcitatoryInput(double[] input) {
        System.arraycopy(input, 0, excitatoryInput, 0, Math.min(input.length, dimension));
    }

    /**
     * Set external inhibitory input for all units.
     */
    public void setInhibitoryInput(double[] input) {
        System.arraycopy(input, 0, inhibitoryInput, 0, Math.min(input.length, dimension));
    }

    /**
     * Clear all external inputs.
     */
    public void clearInputs() {
        for (var i = 0; i < dimension; i++) {
            excitatoryInput[i] = 0.0;
            inhibitoryInput[i] = 0.0;
        }
    }

    /**
     * Compute total excitatory input for unit i.
     * Includes self-excitation, lateral excitation, and external input.
     */
    private double computeExcitation(int i) {
        var total = 0.0;

        // Self-excitation (recurrent)
        total += parameters.selfExcitation() * activations[i];

        // Lateral excitation from nearby units (on-center, narrow Gaussian)
        for (var j = 0; j < dimension; j++) {
            if (i != j) {
                var weight = computeExcitatoryWeight(i, j);
                total += weight * activations[j];
            }
        }

        // External excitatory input
        total += excitatoryInput[i];

        return Math.max(0, total);  // Rectify (non-negative)
    }

    /**
     * Compute total inhibitory input for unit i.
     * Includes lateral inhibition and external input.
     */
    private double computeInhibition(int i) {
        var total = 0.0;

        // Lateral inhibition from all units (off-surround, broad Gaussian)
        for (var j = 0; j < dimension; j++) {
            if (i != j) {
                var weight = computeInhibitoryWeight(i, j);
                total += weight * activations[j];
            }
        }

        // External inhibitory input
        total += inhibitoryInput[i];

        return Math.max(0, total);  // Rectify (non-negative)
    }

    /**
     * Compute excitatory weight between units i and j.
     * Uses a Gaussian kernel for local on-center excitation.
     */
    private double computeExcitatoryWeight(int i, int j) {
        var distance = Math.abs(i - j);
        var sigma = parameters.excitatoryRange();
        return parameters.excitatoryStrength() *
               Math.exp(-distance * distance / (2.0 * sigma * sigma));
    }

    /**
     * Compute inhibitory weight between units i and j.
     * Uses a broader Gaussian kernel for off-surround inhibition.
     */
    private double computeInhibitoryWeight(int i, int j) {
        var distance = Math.abs(i - j);
        var sigma = parameters.inhibitoryRange();
        return parameters.inhibitoryStrength() *
               Math.exp(-distance * distance / (2.0 * sigma * sigma));
    }

    /**
     * Compute total network energy (Lyapunov function).
     * Decreases over time, proving convergence.
     */
    public double computeEnergy() {
        var energy = 0.0;

        for (var i = 0; i < dimension; i++) {
            // Decay term
            energy += 0.5 * parameters.getDecayRate(i) * activations[i] * activations[i];

            // Interaction terms
            for (var j = i + 1; j < dimension; j++) {
                var excWeight = computeExcitatoryWeight(i, j);
                var inhWeight = computeInhibitoryWeight(i, j);
                energy -= excWeight * activations[i] * activations[j];
                energy += inhWeight * activations[i] * activations[j];
            }
        }

        return energy;
    }

    /**
     * Get parameters (immutable).
     */
    public ShuntingParameters getParameters() {
        return parameters;
    }
}
