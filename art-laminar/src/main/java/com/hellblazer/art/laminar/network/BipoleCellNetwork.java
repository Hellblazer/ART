package com.hellblazer.art.laminar.network;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.parameters.BipoleCellParameters;

import java.util.ArrayList;
import java.util.List;

/**
 * Network of bipole cells implementing horizontal connections for boundary completion
 * and perceptual grouping in Layer 2/3.
 *
 * Key features:
 * - Long-range horizontal connections (5-20 units)
 * - Three-way firing logic for each cell
 * - Collinear facilitation
 * - Boundary completion across gaps
 * - Illusory contour formation
 *
 * @author Hal Hildebrand
 */
public class BipoleCellNetwork {

    private final BipoleCellParameters parameters;
    private final List<BipoleCell> cells;
    private final int networkSize;
    private final double[][] connectionWeights;  // Precomputed connection weights
    private boolean propagationEnabled = false;
    private static final double TIME_STEP = 0.01;  // 10ms integration step for faster convergence

    public BipoleCellNetwork(BipoleCellParameters parameters) {
        this.parameters = parameters;
        this.networkSize = parameters.networkSize();
        this.cells = new ArrayList<>(networkSize);

        // Initialize bipole cells
        for (int i = 0; i < networkSize; i++) {
            cells.add(new BipoleCell(i, parameters));
        }

        // Precompute connection weights for efficiency
        this.connectionWeights = computeConnectionWeights();
    }

    /**
     * Process input through the bipole network.
     *
     * @param input Input pattern
     * @return Output activation pattern after horizontal processing
     */
    public DenseVector process(Pattern input) {
        // Pattern is either DenseVector or similar, need to extract data
        var inputData = ((DenseVector) input).data();
        var outputData = new double[networkSize];

        // Initialize cells with input values only once at the beginning
        for (int i = 0; i < networkSize; i++) {
            cells.get(i).setActivation(inputData[i]);
        }

        // Multiple iterations to allow horizontal propagation and convergence
        // More iterations needed for proper temporal dynamics convergence
        int iterations = propagationEnabled ? 15 : 10;

        for (int iter = 0; iter < iterations; iter++) {
            // First pass: Compute all horizontal inputs from CURRENT state
            double[] leftInputs = new double[networkSize];
            double[] rightInputs = new double[networkSize];

            for (int i = 0; i < networkSize; i++) {
                // Compute left horizontal input
                double leftInput = 0.0;
                for (int j = Math.max(0, i - parameters.maxHorizontalRange()); j < i; j++) {
                    double weight = connectionWeights[i][j];
                    if (weight > 0) {
                        leftInput += weight * cells.get(j).getActivation();
                    }
                }
                leftInputs[i] = leftInput;

                // Compute right horizontal input
                double rightInput = 0.0;
                for (int j = i + 1; j < Math.min(networkSize, i + parameters.maxHorizontalRange() + 1); j++) {
                    double weight = connectionWeights[i][j];
                    if (weight > 0) {
                        rightInput += weight * cells.get(j).getActivation();
                    }
                }
                rightInputs[i] = rightInput;
            }

            // Second pass: Update all cells using precomputed horizontal inputs
            for (int i = 0; i < networkSize; i++) {
                var cell = cells.get(i);

                // Store current activation before update
                double currentActivation = cell.getActivation();

                // Update cell activation using three-way logic
                // Use full direct input throughout, as the cell dynamics handle temporal integration
                double directInput = inputData[i];

                // Temporarily set activation back to ensure computeActivation uses correct base
                cell.setActivation(currentActivation);
                double activation = cell.computeActivation(directInput, leftInputs[i], rightInputs[i], TIME_STEP);
                outputData[i] = activation;
            }

            // Third pass: Synchronously update all cell activations for next iteration
            for (int i = 0; i < networkSize; i++) {
                cells.get(i).setActivation(outputData[i]);
            }
        }

        return new DenseVector(outputData);
    }

    /**
     * Precompute connection weights between all cells.
     */
    private double[][] computeConnectionWeights() {
        var weights = new double[networkSize][networkSize];

        for (int i = 0; i < networkSize; i++) {
            for (int j = 0; j < networkSize; j++) {
                if (i != j) {
                    int distance = Math.abs(i - j);
                    if (distance > 0 && distance <= parameters.maxHorizontalRange()) {
                        // Exponential decay with distance
                        double distanceWeight = parameters.maxWeight() *
                            Math.exp(-distance / parameters.distanceSigma());

                        // For now, ignore orientation (will be set per test)
                        weights[i][j] = distanceWeight;
                    }
                }
            }
        }

        return weights;
    }

    /**
     * Get connection weights (for testing).
     *
     * @return Connection weight matrix
     */
    public double[][] getConnectionWeights() {
        return connectionWeights;
    }

    /**
     * Set orientation for a specific cell.
     *
     * @param position Cell position
     * @param orientation Orientation in radians
     */
    public void setOrientation(int position, double orientation) {
        if (position >= 0 && position < networkSize) {
            cells.get(position).setOrientation(orientation);
            // Recompute connection weights for this cell
            updateConnectionWeights(position);
        }
    }

    /**
     * Update connection weights for a specific cell after orientation change.
     */
    private void updateConnectionWeights(int position) {
        var cell = cells.get(position);

        // Update outgoing connections
        for (int j = 0; j < networkSize; j++) {
            if (position != j) {
                connectionWeights[position][j] = cell.computeConnectionWeight(
                    cells.get(j).getPosition(),
                    cells.get(j).getOrientation()
                );
            }
        }

        // Update incoming connections
        for (int i = 0; i < networkSize; i++) {
            if (i != position) {
                connectionWeights[i][position] = cells.get(i).computeConnectionWeight(
                    position,
                    cell.getOrientation()
                );
            }
        }
    }

    /**
     * Enable or disable multi-iteration propagation.
     *
     * @param enabled Whether to enable propagation
     */
    public void enablePropagation(boolean enabled) {
        this.propagationEnabled = enabled;
        // Also enable propagation mode in cells for wave-like spreading
        for (var cell : cells) {
            cell.setPropagationMode(enabled);
        }
    }

    /**
     * Reset all cells in the network.
     */
    public void reset() {
        for (var cell : cells) {
            cell.reset();
        }
    }

    /**
     * Get cell at specific position.
     *
     * @param position Cell position
     * @return Bipole cell at position
     */
    public BipoleCell getCell(int position) {
        if (position >= 0 && position < networkSize) {
            return cells.get(position);
        }
        return null;
    }

    /**
     * Get network size.
     *
     * @return Number of cells in network
     */
    public int getSize() {
        return networkSize;
    }

    /**
     * Get parameters.
     *
     * @return Network parameters
     */
    public BipoleCellParameters getParameters() {
        return parameters;
    }
}