package com.hellblazer.art.temporal.memory;

import com.hellblazer.art.temporal.core.*;

/**
 * Controls and maintains the primacy gradient in working memory.
 * Implements the key insight from Kazerounian & Grossberg (2014) that
 * primacy gradients emerge from transmitter depletion dynamics.
 */
public class PrimacyGradientController {

    private final ShuntingDynamics shuntingDynamics;
    private final TransmitterDynamics transmitterDynamics;
    private final WorkingMemoryParameters parameters;

    // Tracking primacy gradient evolution
    private double[] positionActivations;
    private double[] transmitterLevels;
    private double[] cumulativeInputs;
    private int sequenceLength;

    public PrimacyGradientController(WorkingMemoryParameters parameters) {
        this.parameters = parameters;
        this.shuntingDynamics = new ShuntingDynamics();
        this.transmitterDynamics = new TransmitterDynamics();

        int capacity = parameters.getCapacity();
        this.positionActivations = new double[capacity];
        this.transmitterLevels = new double[capacity];
        this.cumulativeInputs = new double[capacity];
        this.sequenceLength = 0;

        // Initialize transmitter levels to full
        for (int i = 0; i < capacity; i++) {
            transmitterLevels[i] = 1.0;
        }
    }

    /**
     * Initialize primacy gradient for a new sequence.
     */
    public void initializeForSequence(int expectedLength) {
        if (expectedLength > parameters.getCapacity()) {
            expectedLength = parameters.getCapacity();
        }

        this.sequenceLength = expectedLength;

        // Set position-dependent initial activations
        for (int i = 0; i < expectedLength; i++) {
            positionActivations[i] = computeInitialActivation(i, expectedLength);
            transmitterLevels[i] = 1.0; // Start with full transmitter
        }

        // Clear rest
        for (int i = expectedLength; i < parameters.getCapacity(); i++) {
            positionActivations[i] = 0.0;
            transmitterLevels[i] = 1.0;
        }
    }

    /**
     * Compute initial activation with primacy gradient.
     * Earlier positions get higher initial activation.
     */
    private double computeInitialActivation(int position, int totalLength) {
        // Exponential primacy gradient
        double decayFactor = parameters.getPrimacyDecayRate();
        double baseActivation = parameters.getMaxActivation();

        // Additional boost for first item (strong primacy)
        double primacyBoost = (position == 0) ? 1.2 : 1.0;

        // Exponential decay with position
        double activation = baseActivation * primacyBoost * Math.exp(-decayFactor * position);

        // Normalize by sequence length
        double lengthNormalization = 1.0 + Math.log(totalLength) / 10.0;
        activation /= lengthNormalization;

        return Math.min(parameters.getMaxActivation(), activation);
    }

    /**
     * Update gradient when new item is stored at position.
     */
    public void updateForNewItem(int position, double inputStrength, double duration) {
        if (position >= parameters.getCapacity()) return;

        // Record cumulative input
        cumulativeInputs[position] += inputStrength * duration;

        // Create states for dynamics computation
        var shuntingState = new ShuntingState(positionActivations, cumulativeInputs);
        var transmitterState = new TransmitterState(transmitterLevels, cumulativeInputs, null);

        // Create parameter objects
        var shuntingParams = ShuntingParameters.builder()
            .decayRate(computePositionDependentDecay(position))
            .upperBound(parameters.getMaxActivation())
            .lowerBound(0.0)
            .selfExcitation(parameters.getSelfExcitation())
            .lateralInhibition(parameters.getLateralInhibition())
            .build();

        var transmitterParams = TransmitterParameters.builder()
            .epsilon(parameters.getTransmitterRecoveryRate())
            .lambda(parameters.getTransmitterDepletionLinear())
            .mu(parameters.getTransmitterDepletionQuadratic())
            .depletionThreshold(0.2)
            .build();

        // Evolve dynamics
        double dt = parameters.getTimeStep();
        int steps = (int)(duration / dt);

        for (int step = 0; step < steps; step++) {
            // Update shunting dynamics
            shuntingState = shuntingDynamics.step(shuntingState, shuntingParams, step * dt, dt);

            // Update transmitter dynamics
            transmitterState = transmitterDynamics.step(transmitterState, transmitterParams, step * dt, dt);
        }

        // Extract updated values
        positionActivations = shuntingState.getActivations();
        transmitterLevels = transmitterState.getTransmitterLevels();

        // Apply lateral inhibition from new item to previous items
        applyRetrospectiveInhibition(position);
    }

    /**
     * Compute position-dependent decay rate.
     * Later positions decay faster, enhancing primacy gradient.
     */
    private double computePositionDependentDecay(int position) {
        double baseDecay = parameters.getDecayRate();
        double positionFactor = 1.0 + 0.05 * position; // 5% increase per position
        return baseDecay * positionFactor;
    }

    /**
     * Apply retrospective inhibition from new item to earlier items.
     * This implements the competitive dynamics that limit capacity.
     */
    private void applyRetrospectiveInhibition(int newPosition) {
        if (newPosition == 0) return;

        double inhibitionStrength = parameters.getLateralInhibition();
        double newActivation = positionActivations[newPosition];

        for (int i = 0; i < newPosition; i++) {
            // Distance-based inhibition
            int distance = newPosition - i;
            double distanceFactor = Math.exp(-distance / 3.0);

            // Apply inhibition proportional to new item's strength
            double inhibition = inhibitionStrength * newActivation * distanceFactor;
            positionActivations[i] *= (1.0 - inhibition);

            // Ensure non-negative
            if (positionActivations[i] < 0) {
                positionActivations[i] = 0;
            }
        }
    }

    /**
     * Get current gradient strength (positive = primacy, negative = recency).
     */
    public double getGradientStrength() {
        if (sequenceLength < 2) return 0.0;

        double firstHalfSum = 0.0;
        double secondHalfSum = 0.0;
        int midpoint = sequenceLength / 2;

        for (int i = 0; i < midpoint; i++) {
            firstHalfSum += positionActivations[i] * transmitterLevels[i];
        }

        for (int i = midpoint; i < sequenceLength; i++) {
            secondHalfSum += positionActivations[i] * transmitterLevels[i];
        }

        double firstHalfAvg = firstHalfSum / midpoint;
        double secondHalfAvg = secondHalfSum / (sequenceLength - midpoint);

        return (firstHalfAvg - secondHalfAvg) / (firstHalfAvg + secondHalfAvg + 1e-10);
    }

    /**
     * Get gradient profile showing activation by position.
     */
    public GradientProfile getGradientProfile() {
        double[] effectiveActivations = new double[sequenceLength];
        double[] rawActivations = new double[sequenceLength];
        double[] transmitters = new double[sequenceLength];

        for (int i = 0; i < sequenceLength; i++) {
            rawActivations[i] = positionActivations[i];
            transmitters[i] = transmitterLevels[i];
            effectiveActivations[i] = positionActivations[i] * transmitterLevels[i];
        }

        return new GradientProfile(effectiveActivations, rawActivations, transmitters);
    }

    /**
     * Check if gradient has degraded (too flat or reversed).
     */
    public boolean hasGradientDegraded() {
        double strength = getGradientStrength();
        // Gradient is degraded if it's too weak or has reversed (recency)
        return strength < 0.1 || strength < 0;
    }

    /**
     * Apply recovery to partially restore gradient.
     */
    public void applyRecovery(double recoveryDuration) {
        // Allow transmitters to recover
        double recoveryRate = parameters.getTransmitterRecoveryRate();
        double recovery = recoveryRate * recoveryDuration;

        for (int i = 0; i < sequenceLength; i++) {
            transmitterLevels[i] += recovery * (1.0 - transmitterLevels[i]);
            transmitterLevels[i] = Math.min(1.0, transmitterLevels[i]);
        }
    }

    /**
     * Reset to initial state.
     */
    public void reset() {
        for (int i = 0; i < parameters.getCapacity(); i++) {
            positionActivations[i] = 0.0;
            transmitterLevels[i] = 1.0;
            cumulativeInputs[i] = 0.0;
        }
        sequenceLength = 0;
    }

    // Inner classes

    /**
     * Profile of the primacy gradient.
     */
    public record GradientProfile(
        double[] effectiveActivations,
        double[] rawActivations,
        double[] transmitterLevels
    ) {
        /**
         * Get the position with maximum effective activation.
         */
        public int getPeakPosition() {
            int peak = 0;
            double maxActivation = effectiveActivations[0];

            for (int i = 1; i < effectiveActivations.length; i++) {
                if (effectiveActivations[i] > maxActivation) {
                    maxActivation = effectiveActivations[i];
                    peak = i;
                }
            }

            return peak;
        }

        /**
         * Compute gradient slope (activation difference per position).
         */
        public double getSlope() {
            if (effectiveActivations.length < 2) return 0.0;

            double firstActivation = effectiveActivations[0];
            double lastActivation = effectiveActivations[effectiveActivations.length - 1];
            return (firstActivation - lastActivation) / (effectiveActivations.length - 1);
        }
    }
}