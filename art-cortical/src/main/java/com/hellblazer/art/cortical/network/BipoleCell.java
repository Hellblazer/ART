package com.hellblazer.art.cortical.network;

/**
 * Individual Bipole Cell implementing three-way firing logic for boundary completion.
 *
 * <p>A bipole cell fires under three conditions:
 * <ol>
 *   <li>Strong direct bottom-up activation alone</li>
 *   <li>Simultaneous horizontal inputs from BOTH sides (collinear)</li>
 *   <li>Both bottom-up AND horizontal inputs present</li>
 * </ol>
 *
 * <p>This implements the canonical cortical circuit's horizontal grouping mechanism
 * for Layer 1 boundary completion and surface contour processing.
 *
 * <p>Equations (from Grossberg, 2013):
 * <pre>
 * Activation dynamics:
 *   dx/dt = -x/τ + (1-x)[D + H_L + H_R] - x[inhibition]
 *
 * Three-way OR logic:
 *   Fire if: (D > θ_strong) OR
 *            (H_L > θ_h AND H_R > θ_h) OR
 *            (D > θ_weak AND (H_L > θ_h OR H_R > θ_h))
 *
 * Horizontal weight: w(d) = w_max * exp(-d/σ_d) * g(Δθ)
 *   where g(Δθ) = exp(-Δθ²/(2σ_θ²)) for orientation selectivity
 * </pre>
 *
 * @author Migrated from art-laminar to art-cortical (Phase 3, Milestone 2)
 */
public class BipoleCell {

    private final int position;
    private final BipoleCellParameters parameters;

    private double activation;
    private double directInput;
    private double leftHorizontalInput;
    private double rightHorizontalInput;
    private double orientation;  // Cell's preferred orientation in radians
    private double previousActivation;
    private boolean propagationMode = false;  // Allow unilateral propagation when enabled

    /**
     * Create a new bipole cell at the specified position.
     *
     * @param position Position in the network (0-based index)
     * @param parameters Cell parameters
     */
    public BipoleCell(int position, BipoleCellParameters parameters) {
        this.position = position;
        this.parameters = parameters;
        this.activation = 0.0;
        this.directInput = 0.0;
        this.leftHorizontalInput = 0.0;
        this.rightHorizontalInput = 0.0;
        this.orientation = 0.0;  // Default horizontal orientation
        this.previousActivation = 0.0;
    }

    /**
     * Compute cell activation using three-way firing logic.
     *
     * <p>Implements the three firing conditions with temporal dynamics:
     * <ol>
     *   <li>Condition 1: Strong direct input (D > θ_strong) fires cell alone</li>
     *   <li>Condition 2: Bilateral horizontal (H_L > θ AND H_R > θ) enables gap-filling</li>
     *   <li>Condition 3: Weak direct + unilateral horizontal combines bottom-up and grouping</li>
     * </ol>
     *
     * @param direct Direct bottom-up input
     * @param leftInput Summed input from left horizontal connections
     * @param rightInput Summed input from right horizontal connections
     * @param timeStep Integration time step (seconds)
     * @return Updated activation level [0,1]
     */
    public double computeActivation(double direct, double leftInput, double rightInput, double timeStep) {
        this.directInput = direct;
        this.leftHorizontalInput = leftInput;
        this.rightHorizontalInput = rightInput;

        // Store previous activation for dynamics
        previousActivation = activation;

        // Three-way OR logic for bipole cell firing
        var shouldFire = false;
        var targetActivation = 0.0;

        // Condition 1: Strong direct input alone
        // This allows bottom-up signals to drive the cell independently
        if (directInput > parameters.strongDirectThreshold()) {
            shouldFire = true;
            targetActivation = Math.max(targetActivation, directInput);
        }

        // Condition 2: Both horizontal sides active (boundary completion)
        // Use a lower threshold when both sides are active for gap-filling
        // This is the key bipole mechanism for illusory contour completion
        var bilateralThreshold = 0.1; // Lower threshold for bilateral activation
        if (leftHorizontalInput > bilateralThreshold &&
            rightHorizontalInput > bilateralThreshold) {
            shouldFire = true;
            // Strong bilateral activation for gap filling
            var bilateralActivation = (leftHorizontalInput + rightHorizontalInput) * 0.8;
            targetActivation = Math.max(targetActivation, Math.min(1.0, bilateralActivation));
        }

        // Condition 3: Weak direct + at least one horizontal side
        // Combines bottom-up with lateral grouping
        if (directInput > parameters.weakDirectThreshold()) {
            if (leftHorizontalInput > parameters.horizontalThreshold() ||
                rightHorizontalInput > parameters.horizontalThreshold()) {
                shouldFire = true;
                var horizontalSupport = Math.max(leftHorizontalInput, rightHorizontalInput);
                var combinedActivation = (directInput + horizontalSupport) / 2.0;
                targetActivation = Math.max(targetActivation, combinedActivation);
            }
        }

        // Condition 4: In propagation mode, allow unilateral horizontal propagation
        // This enables contour propagation along edges
        if (propagationMode && !shouldFire) {
            var totalHorizontal = leftHorizontalInput + rightHorizontalInput;
            if (totalHorizontal > parameters.horizontalThreshold() * 0.5) {
                shouldFire = true;
                // Propagate with decay to prevent unbounded spreading
                targetActivation = Math.max(targetActivation, totalHorizontal * 0.6);
            }
        }

        // Update activation with temporal dynamics
        // Exponential approach to target (shunting equation)
        if (shouldFire) {
            // Approach target with time constant τ
            var tau = parameters.timeConstant();
            var alpha = timeStep / tau;
            activation = activation + alpha * (targetActivation - activation);
        } else {
            // Decay to zero when not firing (slower decay for hysteresis)
            var tau = parameters.timeConstant() * 2.0;  // Slower decay preserves boundaries
            var alpha = timeStep / tau;
            activation = activation * (1.0 - alpha);
        }

        // Clamp to [0, 1]
        activation = Math.max(0.0, Math.min(1.0, activation));

        return activation;
    }

    /**
     * Compute connection weight to another cell based on distance and orientation.
     *
     * <p>Implements spatial connection profile:
     * <pre>
     * w(d, Δθ) = w_max * exp(-d/σ_d) * exp(-Δθ²/(2σ_θ²))
     * </pre>
     *
     * <p>Distance decay prevents long-range spurious connections.
     * Orientation selectivity ensures collinear grouping.
     *
     * @param otherPosition Position of the other cell
     * @param otherOrientation Orientation of the other cell (radians)
     * @return Connection weight [0, w_max]
     */
    public double computeConnectionWeight(int otherPosition, double otherOrientation) {
        var distance = Math.abs(position - otherPosition);

        // Check if within range
        if (distance == 0 || distance > parameters.maxHorizontalRange()) {
            return 0.0;
        }

        // Exponential decay with distance
        // w_d(d) = w_max * exp(-d/σ_d)
        var distanceWeight = parameters.maxWeight() *
            Math.exp(-distance / parameters.distanceSigma());

        // Orientation selectivity if enabled
        if (parameters.orientationSelectivity()) {
            var orientationDiff = Math.abs(orientation - otherOrientation);
            // Wrap around at π (orientations are mod π)
            if (orientationDiff > Math.PI) {
                orientationDiff = 2 * Math.PI - orientationDiff;
            }

            // Gaussian tuning for orientation
            // g(Δθ) = exp(-Δθ²/(2σ_θ²))
            var orientationWeight = Math.exp(-orientationDiff * orientationDiff /
                (2 * parameters.orientationSigma() * parameters.orientationSigma()));

            return distanceWeight * orientationWeight;
        }

        return distanceWeight;
    }

    /**
     * Reset cell state to initial conditions.
     * Clears activation and inputs but preserves orientation preference.
     */
    public void reset() {
        activation = 0.0;
        directInput = 0.0;
        leftHorizontalInput = 0.0;
        rightHorizontalInput = 0.0;
        previousActivation = 0.0;
    }

    // ==================== Accessors ====================

    public double getActivation() {
        return activation;
    }

    public void setActivation(double activation) {
        this.activation = Math.max(0.0, Math.min(1.0, activation));
    }

    public int getPosition() {
        return position;
    }

    public double getOrientation() {
        return orientation;
    }

    public void setOrientation(double orientation) {
        this.orientation = orientation;
    }

    public double getDirectInput() {
        return directInput;
    }

    public double getLeftHorizontalInput() {
        return leftHorizontalInput;
    }

    public double getRightHorizontalInput() {
        return rightHorizontalInput;
    }

    public double getPreviousActivation() {
        return previousActivation;
    }

    public void setPropagationMode(boolean propagationMode) {
        this.propagationMode = propagationMode;
    }

    public boolean isPropagationMode() {
        return propagationMode;
    }
}
