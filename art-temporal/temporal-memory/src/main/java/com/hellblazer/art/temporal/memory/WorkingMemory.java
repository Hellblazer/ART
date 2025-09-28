package com.hellblazer.art.temporal.memory;

import com.hellblazer.art.temporal.core.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Working Memory implementation based on STORE 2 model from Kazerounian & Grossberg (2014).
 *
 * Key features:
 * - Primacy gradient through position-dependent decay rates
 * - Shunting dynamics for item storage
 * - Transmitter gates for habituation
 * - Competitive dynamics for capacity limits
 *
 * This creates the primacy gradient effect where early items in a sequence
 * are encoded more strongly than later items.
 */
public class WorkingMemory {

    private final WorkingMemoryParameters parameters;
    private final ShuntingDynamics shuntingDynamics;
    private final TransmitterDynamics transmitterDynamics;

    private ShuntingState shuntingState;
    private TransmitterState transmitterState;
    private List<MemoryItem> storedSequence;

    private int currentPosition;
    private double currentTime;

    public WorkingMemory(WorkingMemoryParameters parameters) {
        this.parameters = parameters;
        this.shuntingDynamics = new ShuntingDynamics();
        this.transmitterDynamics = new TransmitterDynamics();

        // Initialize states
        this.shuntingState = new ShuntingState(
            new double[parameters.getCapacity()],
            new double[parameters.getCapacity()]
        );
        this.transmitterState = new TransmitterState(parameters.getCapacity());
        this.storedSequence = new ArrayList<>();

        this.currentPosition = 0;
        this.currentTime = 0.0;
    }

    /**
     * Store a new item in working memory with primacy gradient.
     */
    public void storeItem(double[] pattern, double duration) {
        if (currentPosition >= parameters.getCapacity()) {
            // Memory full - need to handle overflow
            if (parameters.isOverflowResetEnabled()) {
                reset();
            } else {
                return; // Ignore new items
            }
        }

        // Create memory item with position-dependent activation
        var primacyActivation = computePrimacyActivation(currentPosition);
        var memoryItem = new MemoryItem(pattern, currentPosition, primacyActivation, currentTime);
        storedSequence.add(memoryItem);

        // Update shunting state with new item
        var excitatory = shuntingState.getExcitatoryInputs();
        double inputStrength = computeInputStrength(pattern);
        excitatory[currentPosition] = primacyActivation * inputStrength;
        shuntingState = new ShuntingState(shuntingState.getActivations(), excitatory);

        // Update transmitter state with strong signal for visible depletion
        transmitterState.setPresynapticSignal(currentPosition, inputStrength * 5.0);

        // Evolve dynamics for duration
        evolveDynamics(duration);

        currentPosition++;
        currentTime += duration;
    }

    /**
     * Store a complete sequence with automatic primacy gradient.
     */
    public void storeSequence(List<double[]> patterns, double itemDuration) {
        reset();
        for (var pattern : patterns) {
            storeItem(pattern, itemDuration);
        }
    }

    /**
     * Compute primacy-gradient activation for position.
     * Early positions get higher initial activation.
     */
    private double computePrimacyActivation(int position) {
        // Exponential decay with position as in STORE 2 model
        // Increase decay factor based on current sequence length for proper gradient
        double adjustedDecayFactor = parameters.getPrimacyDecayRate() * (1.0 + currentPosition * 0.1);
        // Ensure minimum activation stays above retrieval threshold
        double minActivation = parameters.getRetrievalThreshold() * 2.0;
        double baseActivation = parameters.getMaxActivation() * Math.exp(-adjustedDecayFactor * position);
        return Math.max(baseActivation, minActivation);
    }

    /**
     * Compute input strength from pattern.
     */
    private double computeInputStrength(double[] pattern) {
        double sum = 0.0;
        for (double value : pattern) {
            sum += value * value;
        }
        return Math.sqrt(sum) / Math.sqrt(pattern.length);
    }

    /**
     * Evolve the dynamics for given duration.
     */
    private void evolveDynamics(double duration) {
        double dt = parameters.getTimeStep();
        int steps = Math.max(1, (int)(duration / dt));  // Ensure at least one step

        var shuntingParams = ShuntingParameters.builder()
            .decayRate(parameters.getDecayRate())
            .upperBound(parameters.getMaxActivation())
            .lowerBound(0.0)
            .selfExcitation(parameters.getSelfExcitation())
            .lateralInhibition(parameters.getLateralInhibition())
            .enableNormalization(true)
            .build();

        var transmitterParams = TransmitterParameters.builder()
            .epsilon(parameters.getTransmitterRecoveryRate())
            .lambda(parameters.getTransmitterDepletionLinear() * 10.0)  // Much stronger depletion
            .mu(parameters.getTransmitterDepletionQuadratic() * 10.0)
            .depletionThreshold(0.2)
            .initialLevel(1.0)
            .enableQuadratic(true)
            .build();

        for (int i = 0; i < steps; i++) {
            // Update shunting dynamics
            shuntingState = shuntingDynamics.step(shuntingState, shuntingParams, currentTime, dt);

            // Update transmitter dynamics with active signals for depletion
            transmitterState = transmitterDynamics.step(transmitterState, transmitterParams, currentTime, dt);

            // Apply transmitter gating to shunting activations
            applyTransmitterGating();

            currentTime += dt;
        }
    }

    /**
     * Apply transmitter gating to modulate activations.
     */
    private void applyTransmitterGating() {
        var activations = shuntingState.getActivations();
        var transmitters = transmitterState.getTransmitterLevels();
        var gatedActivations = new double[activations.length];

        for (int i = 0; i < activations.length; i++) {
            gatedActivations[i] = activations[i] * transmitters[i];
        }

        shuntingState = new ShuntingState(gatedActivations, shuntingState.getExcitatoryInputs());
    }

    /**
     * Retrieve the current working memory state as a temporal pattern.
     */
    public TemporalPattern getTemporalPattern() {
        var activations = shuntingState.getActivations();
        var transmitters = transmitterState.getTransmitterLevels();

        // Extract patterns weighted by activation and transmitter levels
        List<double[]> weightedPatterns = new ArrayList<>();
        List<Double> weights = new ArrayList<>();

        for (int i = 0; i < storedSequence.size(); i++) {
            var item = storedSequence.get(i);
            double weight = activations[i] * transmitters[i];

            // Include all stored items, even with low weights
            // The weight itself indicates retrievability
            weightedPatterns.add(item.pattern());
            weights.add(weight);
        }

        return new TemporalPattern(weightedPatterns, weights, computePrimacyGradientStrength());
    }

    /**
     * Compute the strength of the primacy gradient.
     */
    public double computePrimacyGradientStrength() {
        var activations = shuntingState.getActivations();
        if (currentPosition < 2) return 0.0;

        double earlySum = 0.0;
        double lateSum = 0.0;
        int midpoint = currentPosition / 2;

        for (int i = 0; i < midpoint; i++) {
            earlySum += activations[i];
        }
        for (int i = midpoint; i < currentPosition; i++) {
            lateSum += activations[i];
        }

        double earlyAvg = earlySum / midpoint;
        double lateAvg = lateSum / (currentPosition - midpoint);

        return (earlyAvg - lateAvg) / (earlyAvg + lateAvg + 1e-10);
    }

    /**
     * Check if memory should trigger reset due to transmitter depletion.
     */
    public boolean shouldReset() {
        return transmitterDynamics.shouldReset(
            transmitterState,
            TransmitterParameters.paperDefaults()
        );
    }

    /**
     * Reset working memory to initial state.
     */
    public void reset() {
        shuntingState = new ShuntingState(
            new double[parameters.getCapacity()],
            new double[parameters.getCapacity()]
        );
        transmitterState = new TransmitterState(parameters.getCapacity());
        storedSequence.clear();
        currentPosition = 0;
        currentTime = 0.0;
    }

    /**
     * Get current memory utilization (0 to 1).
     */
    public double getUtilization() {
        return (double) currentPosition / parameters.getCapacity();
    }

    /**
     * Get detailed memory state for analysis.
     */
    public MemoryState getDetailedState() {
        return new MemoryState(
            shuntingState,
            transmitterState,
            new ArrayList<>(storedSequence),
            currentPosition,
            currentTime,
            computePrimacyGradientStrength()
        );
    }

    // Inner classes

    /**
     * Represents a single item stored in working memory.
     */
    public record MemoryItem(
        double[] pattern,
        int position,
        double initialActivation,
        double storageTime
    ) {}

    /**
     * Represents a temporal pattern extracted from working memory.
     */
    public record TemporalPattern(
        List<double[]> patterns,
        List<Double> weights,
        double primacyGradient
    ) {
        public boolean isValid() {
            return !patterns.isEmpty() && patterns.size() == weights.size();
        }

        public int sequenceLength() {
            return patterns.size();
        }

        /**
         * Get combined pattern weighted by activations.
         */
        public double[] getCombinedPattern() {
            if (patterns.isEmpty()) return new double[0];

            int dim = patterns.get(0).length;
            double[] combined = new double[dim];
            double totalWeight = 0.0;

            for (int i = 0; i < patterns.size(); i++) {
                var pattern = patterns.get(i);
                var weight = weights.get(i);
                totalWeight += weight;

                for (int j = 0; j < dim; j++) {
                    combined[j] += pattern[j] * weight;
                }
            }

            // Normalize
            if (totalWeight > 0) {
                for (int j = 0; j < dim; j++) {
                    combined[j] /= totalWeight;
                }
            }

            return combined;
        }
    }

    /**
     * Get current state of working memory.
     */
    public WorkingMemoryState getState() {
        // Convert internal data to state format
        var items = new double[parameters.getCapacity()][parameters.getItemDimension()];
        for (int i = 0; i < Math.min(storedSequence.size(), parameters.getCapacity()); i++) {
            var item = storedSequence.get(i);
            System.arraycopy(item.pattern(), 0, items[i], 0, Math.min(item.pattern().length, parameters.getItemDimension()));
        }

        var primacyWeights = computePrimacyWeights();
        var recencyWeights = computeRecencyWeights();

        return new WorkingMemoryState(
            items,
            primacyWeights,
            recencyWeights,
            currentPosition,
            storedSequence.size()
        );
    }

    private double[] computePrimacyWeights() {
        var weights = new double[parameters.getCapacity()];
        double gradient = parameters.getPrimacyGradient();
        for (int i = 0; i < weights.length; i++) {
            weights[i] = Math.exp(-gradient * i);
        }
        return weights;
    }

    private double[] computeRecencyWeights() {
        var weights = new double[parameters.getCapacity()];
        double gradient = parameters.getRecencyGradient();
        for (int i = 0; i < weights.length; i++) {
            weights[i] = Math.exp(-gradient * (weights.length - 1 - i));
        }
        return weights;
    }


    /**
     * Complete memory state for analysis.
     */
    public record MemoryState(
        ShuntingState shuntingState,
        TransmitterState transmitterState,
        List<MemoryItem> sequence,
        int position,
        double time,
        double primacyGradient
    ) {}
}