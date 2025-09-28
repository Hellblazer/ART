package com.hellblazer.art.temporal.core;

/**
 * Activation state for neural field dynamics.
 * Represents the current activation levels of neural units.
 */
public class ActivationState extends State {

    private final double[] activations;

    public ActivationState(double[] activations) {
        this.activations = activations.clone();
    }

    public ActivationState(int dimension) {
        this.activations = new double[dimension];
    }

    @Override
    public int getDimension() {
        return activations.length;
    }

    @Override
    public double[] toArray() {
        return activations.clone();
    }

    @Override
    public State add(State other) {
        if (!(other instanceof ActivationState)) {
            throw new IllegalArgumentException("Can only add ActivationState to ActivationState");
        }
        var otherActivation = (ActivationState) other;
        if (getDimension() != otherActivation.getDimension()) {
            throw new IllegalArgumentException("Dimension mismatch");
        }

        var result = new double[getDimension()];
        for (int i = 0; i < getDimension(); i++) {
            result[i] = activations[i] + otherActivation.activations[i];
        }
        return new ActivationState(result);
    }

    @Override
    public State scale(double factor) {
        var result = new double[getDimension()];
        for (int i = 0; i < getDimension(); i++) {
            result[i] = activations[i] * factor;
        }
        return new ActivationState(result);
    }

    @Override
    public double norm() {
        double sum = 0;
        for (double a : activations) {
            sum += a * a;
        }
        return Math.sqrt(sum);
    }

    @Override
    public boolean isValid() {
        for (double a : activations) {
            if (Double.isNaN(a) || Double.isInfinite(a)) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void clamp(double min, double max) {
        for (int i = 0; i < activations.length; i++) {
            activations[i] = Math.max(min, Math.min(max, activations[i]));
        }
    }

    public double[] getActivations() {
        return activations.clone();
    }

    public double getActivation(int index) {
        return activations[index];
    }

    public void setActivation(int index, double value) {
        activations[index] = value;
    }

    public double getMaxActivation() {
        double max = activations[0];
        for (int i = 1; i < activations.length; i++) {
            max = Math.max(max, activations[i]);
        }
        return max;
    }

    public double getMinActivation() {
        double min = activations[0];
        for (int i = 1; i < activations.length; i++) {
            min = Math.min(min, activations[i]);
        }
        return min;
    }

    public double getTotalActivation() {
        double sum = 0;
        for (double a : activations) {
            sum += a;
        }
        return sum;
    }

    @Override
    public int dimension() {
        return activations.length;
    }

    @Override
    public State copy() {
        return new ActivationState(activations);
    }

    @Override
    public State fromArray(double[] values) {
        return new ActivationState(values);
    }

    @Override
    public double distance(State other) {
        if (!(other instanceof ActivationState)) {
            throw new IllegalArgumentException("Can only compute distance to ActivationState");
        }
        var otherActivation = (ActivationState) other;
        double sum = 0;
        for (int i = 0; i < getDimension(); i++) {
            double diff = activations[i] - otherActivation.activations[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
}