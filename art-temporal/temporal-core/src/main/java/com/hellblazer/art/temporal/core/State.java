package com.hellblazer.art.temporal.core;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Abstract state representation for dynamical systems.
 * Supports both standard and vectorized operations.
 */
public abstract class State {
    protected static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    /**
     * Add another state to this state (vector addition).
     */
    public abstract State add(State other);

    /**
     * Scale this state by a scalar (scalar multiplication).
     */
    public abstract State scale(double scalar);

    /**
     * Compute distance to another state (typically L2 norm).
     */
    public abstract double distance(State other);

    /**
     * Get the dimension of this state vector.
     */
    public abstract int dimension();

    /**
     * Get the dimension of this state (alias for dimension()).
     */
    public int getDimension() {
        return dimension();
    }

    /**
     * Clone this state for immutable operations.
     */
    public abstract State copy();

    /**
     * Convert to array representation.
     */
    public abstract double[] toArray();

    /**
     * Create from array representation.
     */
    public abstract State fromArray(double[] values);

    /**
     * Compute norm of this state.
     */
    public double norm() {
        var array = toArray();
        double sum = 0.0;
        for (double v : array) {
            sum += v * v;
        }
        return Math.sqrt(sum);
    }

    /**
     * Check if state is valid (no NaN or infinite values).
     */
    public boolean isValid() {
        var array = toArray();
        for (double v : array) {
            if (Double.isNaN(v) || Double.isInfinite(v)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Clamp state values to range [min, max].
     */
    public void clamp(double min, double max) {
        var array = toArray();
        for (int i = 0; i < array.length; i++) {
            array[i] = Math.max(min, Math.min(max, array[i]));
        }
        fromArray(array);
    }

    /**
     * Apply element-wise operation using Vector API.
     */
    protected double[] vectorizedOperation(double[] a, double[] b, VectorOperation op) {
        var result = new double[a.length];
        var length = a.length;
        var i = 0;

        // Vectorized loop
        for (; i < SPECIES.loopBound(length); i += SPECIES.length()) {
            var va = DoubleVector.fromArray(SPECIES, a, i);
            var vb = DoubleVector.fromArray(SPECIES, b, i);
            var vr = op.apply(va, vb);
            vr.intoArray(result, i);
        }

        // Scalar tail
        for (; i < length; i++) {
            result[i] = op.applyScalar(a[i], b[i]);
        }

        return result;
    }

    @FunctionalInterface
    protected interface VectorOperation {
        DoubleVector apply(DoubleVector a, DoubleVector b);

        default double applyScalar(double a, double b) {
            return apply(DoubleVector.broadcast(SPECIES, a),
                        DoubleVector.broadcast(SPECIES, b))
                        .lane(0);
        }
    }
}