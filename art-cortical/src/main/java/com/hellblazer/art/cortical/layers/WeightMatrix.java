package com.hellblazer.art.cortical.layers;

/**
 * Weight matrix for synaptic connections between layers.
 * Represents a 2D array of connection weights for Hebbian learning
 * and other plasticity mechanisms.
 *
 * <p>Design: Simple, immutable-structure mutable-weights pattern.
 * - Matrix dimensions fixed at creation
 * - Individual weights mutable for learning
 * - Thread-safe for read operations
 *
 * @author Migrated from art-laminar to art-cortical (Phase 3, Milestone 3)
 */
public final class WeightMatrix {

    private final double[][] weights;
    private final int rows;
    private final int cols;

    /**
     * Create a weight matrix with given dimensions.
     * All weights initialized to zero.
     *
     * @param rows number of rows (post-synaptic units)
     * @param cols number of columns (pre-synaptic units)
     * @throws IllegalArgumentException if rows or cols <= 0
     */
    public WeightMatrix(int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException(
                "Dimensions must be positive: rows=" + rows + ", cols=" + cols);
        }
        this.rows = rows;
        this.cols = cols;
        this.weights = new double[rows][cols];
    }

    /**
     * Get weight at position (i, j).
     *
     * @param i row index (post-synaptic unit)
     * @param j column index (pre-synaptic unit)
     * @return weight value
     * @throws IndexOutOfBoundsException if indices out of range
     */
    public double get(int i, int j) {
        return weights[i][j];
    }

    /**
     * Set weight at position (i, j).
     *
     * @param i row index (post-synaptic unit)
     * @param j column index (pre-synaptic unit)
     * @param value new weight value
     * @throws IndexOutOfBoundsException if indices out of range
     */
    public void set(int i, int j, double value) {
        weights[i][j] = value;
    }

    /**
     * Get number of rows (post-synaptic units).
     *
     * @return row count
     */
    public int getRows() {
        return rows;
    }

    /**
     * Get number of columns (pre-synaptic units).
     *
     * @return column count
     */
    public int getCols() {
        return cols;
    }

    /**
     * Get direct reference to underlying weight array.
     * WARNING: Modifying this array directly bypasses encapsulation.
     * Use for performance-critical operations only.
     *
     * @return weight array (not a copy)
     */
    public double[][] getData() {
        return weights;
    }

    /**
     * Reset all weights to zero.
     */
    public void reset() {
        for (var i = 0; i < rows; i++) {
            for (var j = 0; j < cols; j++) {
                weights[i][j] = 0.0;
            }
        }
    }

    /**
     * Initialize weights randomly in range [-range, range].
     *
     * @param range maximum absolute weight value
     */
    public void randomize(double range) {
        for (var i = 0; i < rows; i++) {
            for (var j = 0; j < cols; j++) {
                weights[i][j] = (Math.random() * 2.0 - 1.0) * range;
            }
        }
    }
}
