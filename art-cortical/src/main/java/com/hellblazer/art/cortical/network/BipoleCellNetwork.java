package com.hellblazer.art.cortical.network;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;

import java.util.ArrayList;
import java.util.List;

/**
 * Network of bipole cells implementing horizontal connections for boundary completion
 * and perceptual grouping in Layer 1.
 *
 * <p>Key features from Grossberg's BCS (Boundary Contour System):
 * <ul>
 *   <li>Long-range horizontal connections (5-20 cortical columns)</li>
 *   <li>Three-way firing logic for each cell</li>
 *   <li>Collinear facilitation for contour integration</li>
 *   <li>Boundary completion across gaps (illusory contours)</li>
 *   <li>Orientation-selective grouping</li>
 * </ul>
 *
 * <p>Processing algorithm:
 * <ol>
 *   <li>Initialize cells with bottom-up input</li>
 *   <li>Compute horizontal inputs (left/right) from current activations</li>
 *   <li>Update each cell via three-way firing logic</li>
 *   <li>Iterate for temporal convergence (10-15 steps)</li>
 * </ol>
 *
 * <p>References:
 * <ul>
 *   <li>Grossberg, S. (2013). Adaptive Resonance Theory. Scholarpedia 8(1): 1569</li>
 *   <li>Grossberg, S., & Mingolla, E. (1985). Neural dynamics of form perception. Psych Review 92(2): 173</li>
 * </ul>
 *
 * @author Migrated from art-laminar to art-cortical (Phase 3, Milestone 2)
 */
public class BipoleCellNetwork {

    private final BipoleCellParameters parameters;
    private final List<BipoleCell> cells;
    private final int networkSize;
    private final double[][] connectionWeights;  // Precomputed connection weights [i][j]
    private boolean propagationEnabled = false;

    private static final double TIME_STEP = 0.01;  // 10ms integration step

    /**
     * Create a new bipole cell network.
     *
     * @param parameters Network parameters
     */
    public BipoleCellNetwork(BipoleCellParameters parameters) {
        this.parameters = parameters;
        this.networkSize = parameters.networkSize();
        this.cells = new ArrayList<>(networkSize);

        // Initialize bipole cells
        for (var i = 0; i < networkSize; i++) {
            cells.add(new BipoleCell(i, parameters));
        }

        // Precompute connection weights for efficiency
        this.connectionWeights = computeConnectionWeights();
    }

    /**
     * Process input through the bipole network.
     *
     * <p>Implements iterative horizontal processing:
     * <ol>
     *   <li>Initialize cells with input</li>
     *   <li>For each iteration:
     *     <ul>
     *       <li>Compute horizontal inputs from current state</li>
     *       <li>Update all cells with three-way firing logic</li>
     *       <li>Synchronously update activations</li>
     *     </ul>
     *   </li>
     *   <li>Return converged activation pattern</li>
     * </ol>
     *
     * @param input Input pattern (bottom-up from Layer 4)
     * @return Output activation pattern after horizontal processing
     */
    public DenseVector process(Pattern input) {
        // Extract input data
        var inputData = ((DenseVector) input).data();
        var outputData = new double[networkSize];

        // Initialize cells with input values
        for (var i = 0; i < networkSize; i++) {
            cells.get(i).setActivation(inputData[i]);
        }

        // Multiple iterations for horizontal propagation and convergence
        // More iterations needed with propagation enabled for wave-like spreading
        var iterations = propagationEnabled ? 15 : 10;

        for (var iter = 0; iter < iterations; iter++) {
            // First pass: Compute all horizontal inputs from CURRENT state
            var leftInputs = new double[networkSize];
            var rightInputs = new double[networkSize];

            for (var i = 0; i < networkSize; i++) {
                // Compute left horizontal input (from cells to the left)
                var leftInput = 0.0;
                var leftStart = Math.max(0, i - parameters.maxHorizontalRange());
                for (var j = leftStart; j < i; j++) {
                    var weight = connectionWeights[i][j];
                    if (weight > 0) {
                        leftInput += weight * cells.get(j).getActivation();
                    }
                }
                leftInputs[i] = leftInput;

                // Compute right horizontal input (from cells to the right)
                var rightInput = 0.0;
                var rightEnd = Math.min(networkSize, i + parameters.maxHorizontalRange() + 1);
                for (var j = i + 1; j < rightEnd; j++) {
                    var weight = connectionWeights[i][j];
                    if (weight > 0) {
                        rightInput += weight * cells.get(j).getActivation();
                    }
                }
                rightInputs[i] = rightInput;
            }

            // Second pass: Update all cells using precomputed horizontal inputs
            for (var i = 0; i < networkSize; i++) {
                var cell = cells.get(i);

                // Store current activation before update
                var currentActivation = cell.getActivation();

                // Update cell activation using three-way logic
                // Direct input remains constant (bottom-up drive)
                var directInput = inputData[i];

                // Compute new activation with temporal dynamics
                cell.setActivation(currentActivation);
                var activation = cell.computeActivation(
                    directInput,
                    leftInputs[i],
                    rightInputs[i],
                    TIME_STEP
                );
                outputData[i] = activation;
            }

            // Third pass: Synchronously update all cell activations for next iteration
            for (var i = 0; i < networkSize; i++) {
                cells.get(i).setActivation(outputData[i]);
            }
        }

        return new DenseVector(outputData);
    }

    /**
     * Precompute connection weights between all cells.
     *
     * <p>Connection weight formula:
     * <pre>
     * w[i][j] = w_max * exp(-|i-j|/σ_d) * g(θ_i, θ_j)
     * </pre>
     *
     * <p>where g(θ_i, θ_j) is the orientation selectivity factor (if enabled).
     *
     * @return Connection weight matrix [i][j]
     */
    private double[][] computeConnectionWeights() {
        var weights = new double[networkSize][networkSize];

        for (var i = 0; i < networkSize; i++) {
            for (var j = 0; j < networkSize; j++) {
                if (i != j) {
                    var distance = Math.abs(i - j);
                    if (distance > 0 && distance <= parameters.maxHorizontalRange()) {
                        // Exponential decay with distance
                        // w(d) = w_max * exp(-d/σ_d)
                        var distanceWeight = parameters.maxWeight() *
                            Math.exp(-distance / parameters.distanceSigma());

                        // Initially set distance-based weight
                        // Orientation will be updated per-cell if enabled
                        weights[i][j] = distanceWeight;
                    }
                }
            }
        }

        return weights;
    }

    /**
     * Set orientation for a specific cell.
     * Updates connection weights accordingly if orientation selectivity is enabled.
     *
     * @param position Cell position (0-based index)
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
     * Recalculates both incoming and outgoing connection weights.
     *
     * @param position Cell position
     */
    private void updateConnectionWeights(int position) {
        var cell = cells.get(position);

        // Update outgoing connections (from this cell to others)
        for (var j = 0; j < networkSize; j++) {
            if (position != j) {
                connectionWeights[position][j] = cell.computeConnectionWeight(
                    cells.get(j).getPosition(),
                    cells.get(j).getOrientation()
                );
            }
        }

        // Update incoming connections (from others to this cell)
        for (var i = 0; i < networkSize; i++) {
            if (i != position) {
                connectionWeights[i][position] = cells.get(i).computeConnectionWeight(
                    position,
                    cell.getOrientation()
                );
            }
        }
    }

    /**
     * Enable or disable multi-iteration propagation mode.
     *
     * <p>When enabled:
     * <ul>
     *   <li>More iterations for convergence (15 vs 10)</li>
     *   <li>Cells use unilateral propagation (Condition 4)</li>
     *   <li>Enables wave-like contour spreading</li>
 * </ul>
     *
     * @param enabled Whether to enable propagation
     */
    public void enablePropagation(boolean enabled) {
        this.propagationEnabled = enabled;
        // Enable propagation mode in all cells for wave-like spreading
        for (var cell : cells) {
            cell.setPropagationMode(enabled);
        }
    }

    /**
     * Reset all cells in the network to initial state.
     * Clears activations but preserves connection weights and orientations.
     */
    public void reset() {
        for (var cell : cells) {
            cell.reset();
        }
    }

    // ==================== Accessors ====================

    /**
     * Get cell at specific position.
     *
     * @param position Cell position (0-based index)
     * @return Bipole cell at position, or null if position invalid
     */
    public BipoleCell getCell(int position) {
        if (position >= 0 && position < networkSize) {
            return cells.get(position);
        }
        return null;
    }

    /**
     * Get network size (number of cells).
     *
     * @return Number of cells in network
     */
    public int getSize() {
        return networkSize;
    }

    /**
     * Get network parameters.
     *
     * @return Network parameters (immutable)
     */
    public BipoleCellParameters getParameters() {
        return parameters;
    }

    /**
     * Get connection weight matrix (for testing/analysis).
     *
     * @return Connection weight matrix [i][j]
     */
    public double[][] getConnectionWeights() {
        return connectionWeights;
    }
}
