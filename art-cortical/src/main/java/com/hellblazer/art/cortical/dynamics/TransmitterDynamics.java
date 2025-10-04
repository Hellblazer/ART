package com.hellblazer.art.cortical.dynamics;

/**
 * Transmitter habituation dynamics for temporal sequence learning.
 * Implements Equation 2 from Kazerounian & Grossberg (2014):
 *
 * <pre>
 * dZ_i/dt = ε(1 - Z_i) - Z_i(λ * S_i + μ * S_i²)
 * </pre>
 *
 * <p>Functionality:
 * <ul>
 *   <li>Transmitter gates signal transmission (multiplicative gating)</li>
 *   <li>Depletes with usage (habituates to repeated stimulation)</li>
 *   <li>Recovers slowly toward baseline</li>
 *   <li>Creates primacy gradient: earlier items in sequence get stronger weight</li>
 * </ul>
 *
 * <p>Biological Motivation:
 * <ul>
 *   <li>Models synaptic depression from vesicle depletion</li>
 *   <li>Provides temporal context through differential depletion</li>
 *   <li>Enables working memory to distinguish item position in sequence</li>
 * </ul>
 *
 * @author Migrated from art-temporal/temporal-dynamics to art-cortical (Phase 1)
 */
public class TransmitterDynamics implements NeuralDynamics {

    private final TransmitterParameters parameters;
    private final int dimension;
    private final double[] transmitters;
    private final double[] signals;

    /**
     * Create transmitter dynamics with given parameters.
     *
     * @param parameters transmitter habituation parameters
     * @param dimension number of transmitter gates
     */
    public TransmitterDynamics(TransmitterParameters parameters, int dimension) {
        this.parameters = parameters;
        this.dimension = dimension;
        this.transmitters = new double[dimension];
        this.signals = new double[dimension];

        // Initialize to baseline (full strength)
        for (var i = 0; i < dimension; i++) {
            transmitters[i] = parameters.baselineLevel();
        }
    }

    @Override
    public double[] update(double timeStep) {
        if (timeStep <= 0) {
            throw new IllegalArgumentException("Time step must be positive: " + timeStep);
        }

        var result = new double[dimension];

        for (var i = 0; i < dimension; i++) {
            // Equation 2: dZ_i/dt = ε(1 - Z_i) - Z_i(λ * S_i + μ * S_i²)
            var recovery = parameters.recoveryRate();
            var linearDepletion = parameters.linearDepletionRate();
            var quadraticDepletion = parameters.quadraticDepletionRate();
            var signal = signals[i];

            // Habituation dynamics
            var derivative = recovery * (1.0 - transmitters[i]) -
                            transmitters[i] * (linearDepletion * signal +
                                             quadraticDepletion * signal * signal);

            // Euler integration
            result[i] = transmitters[i] + timeStep * derivative;

            // Enforce bounds [0, 1]
            result[i] = Math.max(0.0, Math.min(1.0, result[i]));
        }

        // Update internal state
        System.arraycopy(result, 0, transmitters, 0, dimension);

        return result.clone();
    }

    @Override
    public void reset() {
        for (var i = 0; i < dimension; i++) {
            transmitters[i] = parameters.baselineLevel();
            signals[i] = 0.0;
        }
    }

    @Override
    public double[] getActivation() {
        return transmitters.clone();
    }

    @Override
    public int size() {
        return dimension;
    }

    @Override
    public boolean hasConverged() {
        // Check if transmitters have stabilized
        var tolerance = 1e-6;
        var tempTransmitters = transmitters.clone();

        update(parameters.getEffectiveTimeStep());

        var maxChange = 0.0;
        for (var i = 0; i < dimension; i++) {
            var change = Math.abs(transmitters[i] - tempTransmitters[i]);
            maxChange = Math.max(maxChange, change);
        }

        // Restore previous state
        System.arraycopy(tempTransmitters, 0, transmitters, 0, dimension);

        return maxChange < tolerance;
    }

    /**
     * Set signal inputs that drive transmitter depletion.
     */
    public void setSignals(double[] inputSignals) {
        System.arraycopy(inputSignals, 0, signals, 0, Math.min(inputSignals.length, dimension));
    }

    /**
     * Apply habituation based on activation pattern.
     * Activations become signals that deplete transmitters.
     */
    public void habituate(double[] activations, double timeStep) {
        // Convert activations to signals (rectified)
        for (var i = 0; i < Math.min(activations.length, dimension); i++) {
            signals[i] = Math.max(0, activations[i]);
        }

        // Update transmitter levels
        update(timeStep);
    }

    /**
     * Compute gated output: activation * transmitter.
     * This implements multiplicative gating where transmitter
     * modulates signal strength.
     *
     * @param activations input activation pattern
     * @return gated output (activation × transmitter)
     */
    public double[] computeGatedOutput(double[] activations) {
        var output = new double[dimension];
        for (var i = 0; i < dimension; i++) {
            output[i] = activations[i] * transmitters[i];
        }
        return output;
    }

    /**
     * Partial reset with decay factor (for sequence boundaries).
     * Allows transmitters to partially recover between items.
     *
     * @param factor reset strength (0.0 = no reset, 1.0 = full reset)
     */
    public void partialReset(double factor) {
        var baseline = parameters.baselineLevel();
        for (var i = 0; i < dimension; i++) {
            transmitters[i] = factor * baseline + (1.0 - factor) * transmitters[i];
            signals[i] *= (1.0 - factor);
        }
    }

    /**
     * Get current transmitter levels.
     */
    public double[] getTransmitterLevels() {
        return transmitters.clone();
    }

    /**
     * Get average transmitter level across all gates.
     */
    public double getAverageLevel() {
        var sum = 0.0;
        for (var level : transmitters) {
            sum += level;
        }
        return sum / dimension;
    }

    /**
     * Get depletion amount for each transmitter gate.
     * Depletion = baseline - current_level
     */
    public double[] getDepletions() {
        var depletions = new double[dimension];
        var baseline = parameters.baselineLevel();
        for (var i = 0; i < dimension; i++) {
            depletions[i] = baseline - transmitters[i];
        }
        return depletions;
    }

    /**
     * Check if transmitters have recovered above threshold.
     */
    public boolean hasRecovered(double threshold) {
        for (var level : transmitters) {
            if (level < threshold) {
                return false;
            }
        }
        return true;
    }

    /**
     * Compute recovery time constant at current state.
     * Indicates how long it takes for transmitters to recover.
     */
    public double computeRecoveryTimeConstant() {
        var avgSignal = 0.0;
        for (var s : signals) {
            avgSignal += s;
        }
        avgSignal /= dimension;

        var recovery = parameters.recoveryRate();
        var depletion = parameters.linearDepletionRate() * avgSignal +
                       parameters.quadraticDepletionRate() * avgSignal * avgSignal;

        if (recovery + depletion > 0) {
            return 1.0 / (recovery + depletion);
        }
        return Double.POSITIVE_INFINITY;
    }

    /**
     * Get parameters (immutable).
     */
    public TransmitterParameters getParameters() {
        return parameters;
    }
}
