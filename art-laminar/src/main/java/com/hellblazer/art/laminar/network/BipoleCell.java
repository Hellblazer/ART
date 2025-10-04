package com.hellblazer.art.laminar.network;

import com.hellblazer.art.laminar.parameters.BipoleCellParameters;

/**
 * Individual Bipole Cell implementing three-way firing logic for boundary completion.
 *
 * A bipole cell fires under three conditions:
 * 1. Strong direct bottom-up activation alone
 * 2. Simultaneous horizontal inputs from BOTH sides (collinear)
 * 3. Both bottom-up AND horizontal inputs present
 *
 * This implements the canonical cortical circuit's horizontal grouping mechanism.
 *
 * @author Hal Hildebrand
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
     * @param direct Direct bottom-up input
     * @param leftInput Summed input from left horizontal connections
     * @param rightInput Summed input from right horizontal connections
     * @param timeStep Integration time step
     * @return Updated activation
     */
    public double computeActivation(double direct, double leftInput, double rightInput, double timeStep) {
        this.directInput = direct;
        this.leftHorizontalInput = leftInput;
        this.rightHorizontalInput = rightInput;

        // Store previous activation for dynamics
        previousActivation = activation;

        // Three-way OR logic for bipole cell firing
        boolean shouldFire = false;
        double targetActivation = 0.0;

        // Condition 1: Strong direct input alone
        if (directInput > parameters.strongDirectThreshold()) {
            shouldFire = true;
            targetActivation = Math.max(targetActivation, directInput);
        }

        // Condition 2: Both horizontal sides active (boundary completion)
        // Use a lower threshold when both sides are active for gap-filling
        double bilateralThreshold = 0.1; // Lower threshold for bilateral activation
        if (leftHorizontalInput > bilateralThreshold &&
            rightHorizontalInput > bilateralThreshold) {
            shouldFire = true;
            // Strong bilateral activation for gap filling
            double bilateralActivation = (leftHorizontalInput + rightHorizontalInput) * 0.8;
            targetActivation = Math.max(targetActivation, Math.min(1.0, bilateralActivation));
        }

        // Condition 3: Weak direct + at least one horizontal side
        if (directInput > parameters.weakDirectThreshold()) {
            if (leftHorizontalInput > parameters.horizontalThreshold() ||
                rightHorizontalInput > parameters.horizontalThreshold()) {
                shouldFire = true;
                double horizontalSupport = Math.max(leftHorizontalInput, rightHorizontalInput);
                double combinedActivation = (directInput + horizontalSupport) / 2.0;
                targetActivation = Math.max(targetActivation, combinedActivation);
            }
        }

        // Condition 4: In propagation mode, allow unilateral horizontal propagation
        if (propagationMode && !shouldFire) {
            double totalHorizontal = leftHorizontalInput + rightHorizontalInput;
            if (totalHorizontal > parameters.horizontalThreshold() * 0.5) {
                shouldFire = true;
                // Propagate with decay
                targetActivation = Math.max(targetActivation, totalHorizontal * 0.6);
            }
        }

        // Update activation with temporal dynamics
        if (shouldFire) {
            // Exponential approach to target with time constant
            double tau = parameters.timeConstant();
            double alpha = timeStep / tau;
            activation = activation + alpha * (targetActivation - activation);
        } else {
            // Decay to zero when not firing
            double tau = parameters.timeConstant() * 2.0;  // Slower decay
            double alpha = timeStep / tau;
            activation = activation * (1.0 - alpha);
        }

        // Clamp to [0, 1]
        activation = Math.max(0.0, Math.min(1.0, activation));

        return activation;
    }

    /**
     * Compute connection weight to another cell based on distance and orientation.
     *
     * @param otherPosition Position of the other cell
     * @param otherOrientation Orientation of the other cell
     * @return Connection weight
     */
    public double computeConnectionWeight(int otherPosition, double otherOrientation) {
        int distance = Math.abs(position - otherPosition);

        // Check if within range
        if (distance == 0 || distance > parameters.maxHorizontalRange()) {
            return 0.0;
        }

        // Exponential decay with distance
        double distanceWeight = parameters.maxWeight() *
            Math.exp(-distance / parameters.distanceSigma());

        // Orientation selectivity if enabled
        if (parameters.orientationSelectivity()) {
            double orientationDiff = Math.abs(orientation - otherOrientation);
            // Wrap around at π
            if (orientationDiff > Math.PI) {
                orientationDiff = 2 * Math.PI - orientationDiff;
            }

            // Gaussian tuning for orientation
            double orientationWeight = Math.exp(-orientationDiff * orientationDiff /
                (2 * parameters.orientationSigma() * parameters.orientationSigma()));

            return distanceWeight * orientationWeight;
        }

        return distanceWeight;
    }

    /**
     * Reset cell state.
     */
    public void reset() {
        activation = 0.0;
        directInput = 0.0;
        leftHorizontalInput = 0.0;
        rightHorizontalInput = 0.0;
        previousActivation = 0.0;
    }

    // Getters and setters

    public double getActivation() {
        return activation;
    }

    public void setActivation(double activation) {
        this.activation = activation;
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

    public void setPropagationMode(boolean propagationMode) {
        this.propagationMode = propagationMode;
    }
}