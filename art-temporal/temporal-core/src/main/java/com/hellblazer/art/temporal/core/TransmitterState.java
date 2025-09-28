package com.hellblazer.art.temporal.core;

/**
 * State representation for habituative transmitter gates.
 * Based on Equation 7 from Kazerounian & Grossberg (2014):
 * dZ_i/dt = ε(1 - Z_i) - Z_i(λ * S_i + μ * S_i²)
 */
public class TransmitterState extends State {
    private final double[] transmitterLevels;  // Z_i values (0 to 1)
    private final double[] presynapticSignals; // S_i values
    private final double[] depletionHistory;   // Track cumulative depletion for reset decisions

    public TransmitterState(double[] transmitterLevels, double[] presynapticSignals, double[] depletionHistory) {
        validateTransmitterLevels(transmitterLevels);
        this.transmitterLevels = transmitterLevels.clone();
        this.presynapticSignals = presynapticSignals.clone();
        this.depletionHistory = depletionHistory != null ? depletionHistory.clone() : new double[transmitterLevels.length];
    }

    public TransmitterState(int dimension) {
        this.transmitterLevels = new double[dimension];
        this.presynapticSignals = new double[dimension];
        this.depletionHistory = new double[dimension];
        // Initialize transmitters to full capacity
        for (int i = 0; i < dimension; i++) {
            transmitterLevels[i] = 1.0;
        }
    }

    private void validateTransmitterLevels(double[] levels) {
        for (double level : levels) {
            if (level < 0.0 || level > 1.0) {
                throw new IllegalArgumentException("Transmitter levels must be in range [0, 1]");
            }
        }
    }

    public double[] getTransmitterLevels() {
        return transmitterLevels.clone();
    }

    public double[] getLevels() {
        return transmitterLevels.clone();
    }

    public double[] getPresynapticSignals() {
        return presynapticSignals.clone();
    }

    public double[] getDepletionHistory() {
        return depletionHistory.clone();
    }

    public double getTransmitterLevel(int index) {
        return transmitterLevels[index];
    }

    public void setPresynapticSignal(int index, double signal) {
        presynapticSignals[index] = signal;
    }

    /**
     * Check if any transmitter is critically depleted (below threshold).
     */
    public boolean hasDepletedTransmitters(double threshold) {
        for (double level : transmitterLevels) {
            if (level < threshold) {
                return true;
            }
        }
        return false;
    }

    /**
     * Get the average transmitter level across all gates.
     */
    public double getAverageTransmitterLevel() {
        double sum = 0.0;
        for (double level : transmitterLevels) {
            sum += level;
        }
        return sum / transmitterLevels.length;
    }

    /**
     * Apply gating to input pattern.
     */
    public double[] applyGating(double[] input) {
        if (input.length != transmitterLevels.length) {
            throw new IllegalArgumentException("Input dimension must match transmitter dimension");
        }

        var gated = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            gated[i] = input[i] * transmitterLevels[i];
        }
        return gated;
    }

    @Override
    public State add(State other) {
        if (!(other instanceof TransmitterState t)) {
            throw new IllegalArgumentException("Can only add TransmitterState to TransmitterState");
        }

        var result = vectorizedOperation(transmitterLevels, t.transmitterLevels, (a, b) -> a.add(b));

        // Clamp result to [0, 1] range for transmitter levels
        for (int i = 0; i < result.length; i++) {
            result[i] = Math.max(0.0, Math.min(1.0, result[i]));
        }

        var signals = vectorizedOperation(presynapticSignals, t.presynapticSignals, (a, b) -> a.add(b));

        return new TransmitterState(result, signals, depletionHistory);
    }

    @Override
    public State scale(double scalar) {
        var result = new double[transmitterLevels.length];
        for (int i = 0; i < transmitterLevels.length; i++) {
            result[i] = Math.min(1.0, Math.max(0.0, transmitterLevels[i] * scalar));
        }
        return new TransmitterState(result, presynapticSignals, depletionHistory);
    }

    @Override
    public double distance(State other) {
        if (!(other instanceof TransmitterState t)) {
            throw new IllegalArgumentException("Can only compute distance to TransmitterState");
        }

        double sum = 0.0;
        for (int i = 0; i < transmitterLevels.length; i++) {
            var diff = transmitterLevels[i] - t.transmitterLevels[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    @Override
    public int dimension() {
        return transmitterLevels.length;
    }

    @Override
    public State copy() {
        return new TransmitterState(transmitterLevels, presynapticSignals, depletionHistory);
    }

    @Override
    public double[] toArray() {
        return transmitterLevels.clone();
    }

    @Override
    public State fromArray(double[] values) {
        validateTransmitterLevels(values);
        return new TransmitterState(values, presynapticSignals, depletionHistory);
    }

    /**
     * Compute equilibrium transmitter level for given signal.
     * Z_eq = ε / (ε + λS + μS²)
     */
    public static double computeEquilibrium(double signal, double epsilon, double lambda, double mu) {
        return epsilon / (epsilon + lambda * signal + mu * signal * signal);
    }
}