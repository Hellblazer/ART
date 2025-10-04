package com.hellblazer.art.cortical.temporal;

import com.hellblazer.art.cortical.dynamics.ShuntingDynamics;
import com.hellblazer.art.cortical.dynamics.ShuntingParameters;
import com.hellblazer.art.cortical.dynamics.TransmitterDynamics;
import com.hellblazer.art.cortical.dynamics.TransmitterParameters;

/**
 * Controls and maintains the primacy gradient in working memory.
 *
 * <p>Implements the key insight from Kazerounian & Grossberg (2014) that
 * primacy gradients emerge from the interaction of:
 * <ul>
 *   <li>Position-dependent initial activations</li>
 *   <li>Transmitter depletion dynamics during encoding</li>
 *   <li>Competitive lateral inhibition</li>
 * </ul>
 *
 * <p>The primacy gradient ensures earlier items in a sequence are encoded
 * more strongly than later items, consistent with serial position effects
 * in working memory (Murdock, 1962).
 *
 * @author Migrated from art-temporal/temporal-memory to art-cortical (Phase 2)
 */
public class PrimacyGradientController {

    private final WorkingMemoryParameters parameters;
    private final ShuntingDynamics shuntingDynamics;
    private final TransmitterDynamics transmitterDynamics;

    // State tracking
    private double[] positionActivations;
    private double[] cumulativeInputs;
    private int sequenceLength;

    /**
     * Create primacy gradient controller with given parameters.
     */
    public PrimacyGradientController(WorkingMemoryParameters parameters) {
        this.parameters = parameters;

        int capacity = parameters.capacity();

        // Create uniform decay rates for now (could be position-dependent)
        var decayRates = new double[capacity];
        for (int i = 0; i < capacity; i++) {
            decayRates[i] = parameters.decayRate();
        }

        // Create shunting dynamics for position-specific activations
        var shuntingParams = new ShuntingParameters(
            decayRates,                         // per-unit decay rates
            parameters.ceiling(),               // ceiling
            0.0,                                // floor
            parameters.selfExcitation(),        // self-excitation
            0.3,                                // excitatory strength
            parameters.lateralInhibition(),     // inhibitory strength
            2.0,                                // excitatory range
            5.0,                                // inhibitory range
            0.0,                                // initial activation
            parameters.timeStep()               // time step
        );
        this.shuntingDynamics = new ShuntingDynamics(shuntingParams);

        // Create transmitter dynamics for habituation
        var transmitterParams = new TransmitterParameters(
            parameters.transmitterRecoveryRate(),
            parameters.transmitterDepletionLinear(),
            parameters.transmitterDepletionQuadratic(),
            1.0,                                // baseline level
            0.01                                // effective time step
        );
        this.transmitterDynamics = new TransmitterDynamics(transmitterParams, capacity);

        // Initialize tracking arrays
        this.positionActivations = new double[capacity];
        this.cumulativeInputs = new double[capacity];
        this.sequenceLength = 0;

        // Initialize transmitter levels to full
        transmitterDynamics.reset();
    }

    /**
     * Initialize primacy gradient for a new sequence.
     */
    public void initializeForSequence(int expectedLength) {
        if (expectedLength > parameters.capacity()) {
            expectedLength = parameters.capacity();
        }

        this.sequenceLength = expectedLength;

        // Set position-dependent initial activations
        for (int i = 0; i < expectedLength; i++) {
            positionActivations[i] = computeInitialActivation(i, expectedLength);
        }

        // Clear rest
        for (int i = expectedLength; i < parameters.capacity(); i++) {
            positionActivations[i] = 0.0;
        }

        // Clear cumulative inputs
        for (int i = 0; i < parameters.capacity(); i++) {
            cumulativeInputs[i] = 0.0;
        }

        // Reset transmitters
        transmitterDynamics.reset();
    }

    /**
     * Compute initial activation with primacy gradient.
     * Earlier positions get higher initial activation.
     */
    private double computeInitialActivation(int position, int totalLength) {
        var decayFactor = parameters.primacyDecayRate();
        var baseActivation = parameters.maxActivation();

        // Additional boost for first item (strong primacy)
        double primacyBoost = (position == 0) ? 1.2 : 1.0;

        // Exponential decay with position
        double activation = baseActivation * primacyBoost * Math.exp(-decayFactor * position);

        // Normalize by sequence length
        double lengthNormalization = 1.0 + Math.log(totalLength) / 10.0;
        activation /= lengthNormalization;

        return Math.min(parameters.maxActivation(), activation);
    }

    /**
     * Update gradient when new item is stored at position.
     */
    public void updateForNewItem(int position, double inputStrength, double duration) {
        if (position >= parameters.capacity()) {
            return;
        }

        // Record cumulative input
        cumulativeInputs[position] += inputStrength * duration;

        // Set input for shunting dynamics
        var excitatoryInput = new double[parameters.capacity()];
        excitatoryInput[position] = inputStrength;
        shuntingDynamics.setExcitatoryInput(excitatoryInput);

        // Set signal for transmitter depletion
        var signals = new double[parameters.capacity()];
        signals[position] = inputStrength;
        transmitterDynamics.setSignals(signals);

        // Evolve dynamics for duration
        double dt = parameters.timeStep();
        int steps = (int)(duration / dt);

        for (int step = 0; step < steps; step++) {
            // Update shunting dynamics (fast time scale: ~10ms)
            positionActivations = shuntingDynamics.update(dt);

            // Update transmitter dynamics periodically (slow time scale: ~500ms)
            if (step % 10 == 0) {
                transmitterDynamics.update(dt * 10);
            }
        }

        // Apply lateral inhibition from new item to previous items
        applyRetrospectiveInhibition(position);
    }

    /**
     * Compute position-dependent decay rate.
     * Later positions decay faster, enhancing primacy gradient.
     */
    @SuppressWarnings("unused")
    private double computePositionDependentDecay(int position) {
        double baseDecay = parameters.decayRate();
        double positionFactor = 1.0 + 0.05 * position; // 5% increase per position
        return baseDecay * positionFactor;
    }

    /**
     * Apply retrospective inhibition from new item to earlier items.
     * This implements the competitive dynamics that limit capacity.
     */
    private void applyRetrospectiveInhibition(int newPosition) {
        if (newPosition == 0) {
            return;
        }

        double inhibitionStrength = parameters.lateralInhibition();
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
        if (sequenceLength < 2) {
            return 0.0;
        }

        var transmitterLevels = transmitterDynamics.getTransmitterLevels();

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
        var transmitters = transmitterDynamics.getTransmitterLevels();
        var effectiveActivations = new double[sequenceLength];
        var rawActivations = new double[sequenceLength];
        var transmitterLevels = new double[sequenceLength];

        for (int i = 0; i < sequenceLength; i++) {
            rawActivations[i] = positionActivations[i];
            transmitterLevels[i] = transmitters[i];
            effectiveActivations[i] = positionActivations[i] * transmitters[i];
        }

        return new GradientProfile(effectiveActivations, rawActivations, transmitterLevels);
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
        // Let transmitter dynamics handle recovery naturally
        transmitterDynamics.setSignals(new double[parameters.capacity()]); // No signal
        transmitterDynamics.update(recoveryDuration);
    }

    /**
     * Reset to initial state.
     */
    public void reset() {
        for (int i = 0; i < parameters.capacity(); i++) {
            positionActivations[i] = 0.0;
            cumulativeInputs[i] = 0.0;
        }
        sequenceLength = 0;
        shuntingDynamics.reset();
        transmitterDynamics.reset();
    }

    /**
     * Get current position activations.
     */
    public double[] getPositionActivations() {
        return positionActivations.clone();
    }

    /**
     * Get current transmitter levels.
     */
    public double[] getTransmitterLevels() {
        return transmitterDynamics.getTransmitterLevels();
    }

    // Inner classes

    /**
     * Profile of the primacy gradient showing activation distribution.
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
            if (effectiveActivations.length < 2) {
                return 0.0;
            }

            double firstActivation = effectiveActivations[0];
            double lastActivation = effectiveActivations[effectiveActivations.length - 1];
            return (firstActivation - lastActivation) / (effectiveActivations.length - 1);
        }
    }
}
