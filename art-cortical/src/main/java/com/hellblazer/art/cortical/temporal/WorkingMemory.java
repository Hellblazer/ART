package com.hellblazer.art.cortical.temporal;

import com.hellblazer.art.cortical.dynamics.ShuntingDynamics;
import com.hellblazer.art.cortical.dynamics.ShuntingParameters;
import com.hellblazer.art.cortical.dynamics.TransmitterDynamics;
import com.hellblazer.art.cortical.dynamics.TransmitterParameters;

import java.util.ArrayList;
import java.util.List;

/**
 * Working Memory implementation based on STORE 2 model from Kazerounian & Grossberg (2014).
 *
 * <p>Key features:
 * <ul>
 *   <li>Primacy gradient through position-dependent decay rates</li>
 *   <li>Shunting dynamics for item storage</li>
 *   <li>Transmitter gates for habituation</li>
 *   <li>Competitive dynamics for capacity limits (Miller's 7±2)</li>
 * </ul>
 *
 * <p>This creates the primacy gradient effect where early items in a sequence
 * are encoded more strongly than later items, consistent with serial position
 * effects observed in human working memory.
 *
 * <p>The primacy gradient emerges from:
 * <ol>
 *   <li>Position-dependent initial activations (exponential decay)</li>
 *   <li>Transmitter depletion during encoding</li>
 *   <li>Competitive lateral inhibition</li>
 * </ol>
 *
 * @author Migrated from art-temporal/temporal-memory to art-cortical (Phase 2)
 */
public class WorkingMemory {

    private final WorkingMemoryParameters parameters;
    private final ShuntingDynamics shuntingDynamics;
    private final TransmitterDynamics transmitterDynamics;

    private List<MemoryItem> storedSequence;
    private int currentPosition;
    private double currentTime;

    /**
     * Create working memory with given parameters.
     */
    public WorkingMemory(WorkingMemoryParameters parameters) {
        this.parameters = parameters;

        // Create shunting dynamics for activation
        var shuntingParams = createShuntingParameters();
        this.shuntingDynamics = new ShuntingDynamics(shuntingParams);

        // Create transmitter dynamics for habituation
        var transmitterParams = createTransmitterParameters();
        this.transmitterDynamics = new TransmitterDynamics(transmitterParams, parameters.capacity());

        // Initialize state
        this.storedSequence = new ArrayList<>();
        this.currentPosition = 0;
        this.currentTime = 0.0;
    }

    /**
     * Create shunting dynamics parameters from working memory parameters.
     */
    private ShuntingParameters createShuntingParameters() {
        var decayRates = new double[parameters.capacity()];
        for (int i = 0; i < parameters.capacity(); i++) {
            decayRates[i] = parameters.decayRate();
        }

        return new ShuntingParameters(
            decayRates,
            parameters.maxActivation(),
            0.0,  // floor
            parameters.selfExcitation(),
            0.3,  // excitatory strength
            parameters.lateralInhibition(),
            2.0,  // excitatory range
            5.0,  // inhibitory range
            0.0,  // initial activation
            parameters.timeStep()
        );
    }

    /**
     * Create transmitter dynamics parameters from working memory parameters.
     */
    private TransmitterParameters createTransmitterParameters() {
        return new TransmitterParameters(
            parameters.transmitterRecoveryRate(),
            parameters.transmitterDepletionLinear(),
            parameters.transmitterDepletionQuadratic(),
            1.0,  // baseline level
            parameters.timeStep()
        );
    }

    /**
     * Store a new item in working memory with primacy gradient.
     */
    public void storeItem(double[] pattern, double duration) {
        if (currentPosition >= parameters.capacity()) {
            // Memory full - need to handle overflow
            if (parameters.overflowResetEnabled()) {
                reset();
            } else {
                return; // Ignore new items
            }
        }

        // Create memory item with position-dependent activation
        var primacyActivation = computePrimacyActivation(currentPosition);
        var memoryItem = new MemoryItem(pattern, currentPosition, primacyActivation, currentTime);
        storedSequence.add(memoryItem);

        // Set excitatory input for this position
        var excitatoryInput = new double[parameters.capacity()];
        double inputStrength = computeInputStrength(pattern);
        excitatoryInput[currentPosition] = primacyActivation * inputStrength;
        shuntingDynamics.setExcitatoryInput(excitatoryInput);

        // Set transmitter signal for THIS position only
        var signals = new double[parameters.capacity()];
        signals[currentPosition] = inputStrength;
        transmitterDynamics.setSignals(signals);

        // Evolve dynamics for duration
        evolveDynamics(duration);

        // After evolution, clear signals for this position
        signals[currentPosition] = 0.0;
        transmitterDynamics.setSignals(signals);

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
        double adjustedDecayFactor = parameters.primacyDecayRate() * (1.0 + currentPosition * 0.1);
        double minActivation = parameters.retrievalThreshold() * 2.0;
        double baseActivation = parameters.maxActivation() * Math.exp(-adjustedDecayFactor * position);
        return Math.max(baseActivation, minActivation);
    }

    /**
     * Compute input strength from pattern (L2 norm).
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
     * Multi-scale integration: fast shunting, slow transmitters.
     */
    private void evolveDynamics(double duration) {
        double dt = parameters.timeStep();
        int steps = Math.max(1, (int)(duration / dt));

        // Multi-scale integration: fast shunting, slow transmitters
        int transmitterUpdateInterval = 10;

        for (int i = 0; i < steps; i++) {
            // Update shunting dynamics (fast time scale: ~10ms)
            shuntingDynamics.update(dt);

            // Update transmitter dynamics periodically (slow time scale: ~500ms)
            if (i % transmitterUpdateInterval == 0) {
                transmitterDynamics.update(dt * transmitterUpdateInterval);
            }
        }

        // Apply transmitter gating ONCE at the end after activations stabilize
        applyTransmitterGating();
    }

    /**
     * Apply transmitter gating to modulate activations.
     * Implements multiplicative gating: activation × transmitter.
     */
    private void applyTransmitterGating() {
        var activations = shuntingDynamics.getActivation();
        var gatedActivations = transmitterDynamics.computeGatedOutput(activations);

        // Update shunting dynamics with gated values
        shuntingDynamics.setExcitatoryInput(gatedActivations);
        shuntingDynamics.update(parameters.timeStep());
    }

    /**
     * Retrieve the current working memory state as a temporal pattern.
     */
    public TemporalPattern getTemporalPattern() {
        var activations = shuntingDynamics.getActivation();
        var transmitters = transmitterDynamics.getTransmitterLevels();

        // Extract patterns weighted by activation and transmitter levels
        List<double[]> weightedPatterns = new ArrayList<>();
        List<Double> weights = new ArrayList<>();

        for (int i = 0; i < storedSequence.size(); i++) {
            var item = storedSequence.get(i);
            double weight = activations[i] * transmitters[i];

            // Include all stored items
            weightedPatterns.add(item.pattern());
            weights.add(weight);
        }

        return new TemporalPattern(weightedPatterns, weights, computePrimacyGradientStrength());
    }

    /**
     * Compute the strength of the primacy gradient.
     * Positive = primacy effect, Negative = recency effect.
     */
    public double computePrimacyGradientStrength() {
        var activations = shuntingDynamics.getActivation();
        if (currentPosition < 2) {
            return 0.0;
        }

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
        var avgLevel = transmitterDynamics.getAverageLevel();
        return avgLevel < 0.3; // Reset if average depletion exceeds 70%
    }

    /**
     * Reset working memory to initial state.
     */
    public void reset() {
        shuntingDynamics.reset();
        transmitterDynamics.reset();
        storedSequence.clear();
        currentPosition = 0;
        currentTime = 0.0;
    }

    /**
     * Get current memory utilization (0 to 1).
     */
    public double getUtilization() {
        return (double) currentPosition / parameters.capacity();
    }

    /**
     * Get detailed memory state for analysis.
     */
    public MemoryState getDetailedState() {
        return new MemoryState(
            shuntingDynamics.getActivation(),
            transmitterDynamics.getTransmitterLevels(),
            new ArrayList<>(storedSequence),
            currentPosition,
            currentTime,
            computePrimacyGradientStrength()
        );
    }

    /**
     * Get current state of working memory.
     */
    public WorkingMemoryState getState() {
        // Convert internal data to state format
        var items = new double[parameters.capacity()][parameters.itemDimension()];
        for (int i = 0; i < Math.min(storedSequence.size(), parameters.capacity()); i++) {
            var item = storedSequence.get(i);
            System.arraycopy(item.pattern(), 0, items[i], 0,
                           Math.min(item.pattern().length, parameters.itemDimension()));
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
        var weights = new double[parameters.capacity()];
        double gradient = parameters.getPrimacyGradient();
        for (int i = 0; i < weights.length; i++) {
            weights[i] = Math.exp(-gradient * i);
        }
        return weights;
    }

    private double[] computeRecencyWeights() {
        var weights = new double[parameters.capacity()];
        double gradient = parameters.getRecencyGradient();
        for (int i = 0; i < weights.length; i++) {
            weights[i] = Math.exp(-gradient * (weights.length - 1 - i));
        }
        return weights;
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
     * Complete memory state for analysis.
     */
    public record MemoryState(
        double[] activations,
        double[] transmitterLevels,
        List<MemoryItem> sequence,
        int position,
        double time,
        double primacyGradient
    ) {
        /**
         * Get shunting state wrapper for backward compatibility.
         */
        public ShuntingStateWrapper shuntingState() {
            return new ShuntingStateWrapper(activations);
        }

        /**
         * Get transmitter state wrapper for backward compatibility.
         */
        public TransmitterStateWrapper transmitterState() {
            return new TransmitterStateWrapper(transmitterLevels);
        }
    }

    /**
     * Wrapper for shunting dynamics state (backward compatibility).
     */
    public record ShuntingStateWrapper(double[] activations) {
        public double[] getActivations() {
            return activations.clone();
        }
    }

    /**
     * Wrapper for transmitter dynamics state (backward compatibility).
     */
    public record TransmitterStateWrapper(double[] transmitterLevels) {
        public double[] getTransmitterLevels() {
            return transmitterLevels.clone();
        }
    }
}
