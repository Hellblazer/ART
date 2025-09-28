package com.hellblazer.art.temporal.core;

/**
 * State representation for shunting dynamics.
 */
public class ShuntingState extends State {
    private final double[] activations;
    private final double[] excitatoryInputs;

    public ShuntingState(double[] activations, double[] excitatoryInputs) {
        this.activations = activations.clone();
        this.excitatoryInputs = excitatoryInputs.clone();
    }

    public double[] getActivations() {
        return activations.clone();
    }

    public double[] getExcitatoryInputs() {
        return excitatoryInputs.clone();
    }

    public double getExcitatoryInput(int index) {
        return excitatoryInputs[index];
    }

    @Override
    public State add(State other) {
        if (!(other instanceof ShuntingState s)) {
            throw new IllegalArgumentException("Can only add ShuntingState to ShuntingState");
        }

        var result = vectorizedOperation(activations, s.activations,
            (a, b) -> a.add(b));

        return new ShuntingState(result, excitatoryInputs);
    }

    @Override
    public State scale(double scalar) {
        var result = new double[activations.length];
        for (int i = 0; i < activations.length; i++) {
            result[i] = activations[i] * scalar;
        }
        return new ShuntingState(result, excitatoryInputs);
    }

    @Override
    public double distance(State other) {
        if (!(other instanceof ShuntingState s)) {
            throw new IllegalArgumentException("Can only compute distance to ShuntingState");
        }

        double sum = 0.0;
        for (int i = 0; i < activations.length; i++) {
            var diff = activations[i] - s.activations[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    @Override
    public int dimension() {
        return activations.length;
    }

    @Override
    public State copy() {
        return new ShuntingState(activations, excitatoryInputs);
    }

    @Override
    public double[] toArray() {
        return activations.clone();
    }

    @Override
    public State fromArray(double[] values) {
        return new ShuntingState(values, excitatoryInputs);
    }
}