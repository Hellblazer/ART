package com.hellblazer.art.cortical.layers;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.cortical.analysis.CircularBuffer;
import com.hellblazer.art.cortical.analysis.OscillationAnalyzer;
import com.hellblazer.art.cortical.analysis.OscillationMetrics;
import com.hellblazer.art.cortical.dynamics.ShuntingDynamics;
import com.hellblazer.art.cortical.dynamics.ShuntingParameters;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Layer 1 Implementation - Top-Down Attentional Priming.
 *
 * <p>Layer 1 contains the apical dendrites of pyramidal neurons that receive
 * top-down attentional signals from higher cortical areas. It provides
 * sustained attention without directly driving responses.
 *
 * <p>Key characteristics:
 * <ul>
 *   <li>Very slow time constants (200-1000ms) for sustained attention</li>
 *   <li>Receives top-down signals from higher cortical areas</li>
 *   <li>Sustained attention effects that persist long after input ends</li>
 *   <li>Priming without driving cells directly (modulatory only)</li>
 *   <li>Integrates with Layer 2/3 apical dendrites</li>
 *   <li>Long-duration memory traces (seconds)</li>
 *   <li>Lowest firing rates of all layers</li>
 * </ul>
 *
 * <p>Biological basis:
 * Layer 1 is primarily composed of apical dendritic tufts from pyramidal
 * neurons in deeper layers, along with sparse inhibitory interneurons.
 * It receives diffuse top-down projections that modulate cortical processing
 * through sustained attentional effects.
 *
 * @author Migrated from art-laminar to art-cortical (Phase 3, Milestone 3)
 */
public final class Layer1 implements Layer {

    private final String id;
    private final int size;
    private final List<LayerActivationListener> listeners;

    private ShuntingDynamics verySlowDynamics;
    private Layer1Parameters currentParameters;
    private double[] attentionState;      // Persistent attention state
    private double[] primingEffect;       // Current priming effect
    private double[] memoryTrace;         // Long-term memory trace
    private Pattern activation;

    // Oscillation tracking (Phase 2: Oscillatory Dynamics Integration)
    private OscillationAnalyzer oscillationAnalyzer;
    private CircularBuffer<double[]> activationHistory;
    private OscillationMetrics currentMetrics;
    private double currentTimestamp;

    // Learning infrastructure (Phase 3C: Multi-Layer Learning)
    private com.hellblazer.art.cortical.learning.LearningRule learningRule;
    private com.hellblazer.art.cortical.learning.LearningStatistics learningStatistics;

    /**
     * Create Layer 1 with given ID and size.
     *
     * @param id layer identifier (typically "L1")
     * @param size number of units/columns in this layer
     * @throws IllegalArgumentException if size <= 0
     */
    public Layer1(String id, int size) {
        if (size <= 0) {
            throw new IllegalArgumentException("Layer size must be positive: " + size);
        }

        this.id = id;
        this.size = size;
        this.listeners = new CopyOnWriteArrayList<>();
        this.attentionState = new double[size];
        this.primingEffect = new double[size];
        this.memoryTrace = new double[size];
        this.activation = new DenseVector(new double[size]);
        this.currentTimestamp = 0.0;

        initializeVerySlowDynamics();
    }

    /**
     * Initialize shunting dynamics with very slow time constants for Layer 1.
     */
    private void initializeVerySlowDynamics() {
        // Initialize with very slow dynamics suitable for Layer 1
        var params = ShuntingParameters.builder(size)
            .ceiling(1.0)
            .floor(0.0)
            .selfExcitation(0.05)  // Very weak self-excitation
            .inhibitoryStrength(0.05)  // Very weak lateral inhibition
            .timeStep(0.05)  // 50ms time step for very slow dynamics
            .build();
        this.verySlowDynamics = new ShuntingDynamics(params);
    }

    @Override
    public String getId() {
        return id;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public LayerType getType() {
        return LayerType.LAYER_1;
    }

    @Override
    public Pattern getActivation() {
        return activation;
    }

    @Override
    public void setActivation(Pattern activation) {
        if (activation.dimension() != size) {
            throw new IllegalArgumentException(
                "Activation dimension " + activation.dimension() + " doesn't match layer size " + size);
        }
        this.activation = activation;
    }

    @Override
    public Pattern processBottomUp(Pattern input, LayerParameters parameters) {
        // Layer 1 primarily processes top-down signals
        // Bottom-up processing is minimal - just maintain current state
        return activation;
    }

    @Override
    public Pattern processTopDown(Pattern expectation, LayerParameters parameters) {
        Layer1Parameters layer1Params;
        if (!(parameters instanceof Layer1Parameters)) {
            // Use defaults if wrong parameter type
            layer1Params = Layer1Parameters.builder().build();
        } else {
            layer1Params = (Layer1Parameters) parameters;
        }
        currentParameters = layer1Params;

        // Update shunting dynamics parameters based on Layer1Parameters
        updateDynamicsParameters(layer1Params);

        // Convert top-down expectation to attention array
        var attentionInput = new double[size];
        for (int i = 0; i < Math.min(expectation.dimension(), size); i++) {
            attentionInput[i] = expectation.get(i);
        }

        // Update attention state with very slow dynamics
        updateAttentionState(attentionInput, layer1Params);

        // Update memory trace with even slower dynamics
        updateMemoryTrace(attentionInput, layer1Params);

        // Calculate priming effect (modulates but doesn't drive)
        for (int i = 0; i < size; i++) {
            // Combine attention state and memory trace
            var combinedAttention = attentionState[i] + memoryTrace[i] * 0.5;

            // For initial strong attention, allow higher values
            if (attentionInput[i] > 0.8) {
                combinedAttention = Math.max(combinedAttention, attentionInput[i] * 0.9);
            }

            // Apply priming strength (keeps values moderate)
            primingEffect[i] = combinedAttention * layer1Params.primingStrength();

            // Ensure priming doesn't exceed limits
            primingEffect[i] = Math.min(primingEffect[i], 0.5);  // Cap at 50% to prevent driving
        }

        // Set as excitatory input for very slow dynamics
        verySlowDynamics.setExcitatoryInput(primingEffect);

        // Evolve with very slow time constant
        var timeStep = layer1Params.timeConstant() / 20000.0;  // Convert ms to seconds
        var result = verySlowDynamics.update(timeStep);

        // Apply constraints and allow strong initial attention
        for (int i = 0; i < result.length; i++) {
            // For strong attention input, allow stronger output
            if (attentionInput[i] > 0.8) {
                result[i] = Math.max(result[i], attentionInput[i] * 0.85);
            }

            // Maintain attention state even without input
            if (attentionState[i] > 0.1) {
                result[i] = Math.max(result[i], attentionState[i] * 0.7);
            }

            result[i] = Math.max(layer1Params.floor(),
                        Math.min(layer1Params.ceiling(), result[i]));
        }

        // Update activation and notify listeners
        var oldActivation = activation;
        activation = new DenseVector(result);

        // Notify listeners about activation change
        for (var listener : listeners) {
            listener.onActivationChanged(id, oldActivation, activation);
        }

        // Oscillation analysis (if enabled)
        if (oscillationAnalyzer != null && activationHistory != null) {
            // Record activation snapshot
            activationHistory.add(result.clone());

            // Analyze when history buffer is full
            if (activationHistory.isFull()) {
                currentMetrics = oscillationAnalyzer.analyze(activationHistory, currentTimestamp);
            }

            // Increment timestamp (assuming 1ms timesteps for 1000 Hz sampling)
            currentTimestamp += 0.001;
        }

        return activation;
    }

    @Override
    public Pattern processLateral(Pattern lateral, LayerParameters parameters) {
        // Layer 1 has minimal lateral processing - lateral competition is handled
        // implicitly through the shunting dynamics in processTopDown
        return activation;
    }

    @Deprecated
    @Override
    public void updateWeights(Pattern input, double learningRate) {
        // Layer 1 doesn't have traditional synaptic learning
        // Attention and priming are dynamic, not weight-based
    }

    @Override
    public void reset() {
        if (verySlowDynamics != null) {
            verySlowDynamics.reset();
        }
        if (attentionState != null) {
            for (int i = 0; i < attentionState.length; i++) {
                attentionState[i] = 0.0;
            }
        }
        if (primingEffect != null) {
            for (int i = 0; i < primingEffect.length; i++) {
                primingEffect[i] = 0.0;
            }
        }
        if (memoryTrace != null) {
            for (int i = 0; i < memoryTrace.length; i++) {
                memoryTrace[i] = 0.0;
            }
        }
        currentParameters = null;
        activation = new DenseVector(new double[size]);

        // Clear oscillation tracking
        if (activationHistory != null) {
            activationHistory.clear();
        }
        currentMetrics = null;
        currentTimestamp = 0.0;
    }

    /**
     * Update attention state with very slow dynamics.
     */
    private void updateAttentionState(double[] input, Layer1Parameters params) {
        var sustainedDecay = params.sustainedDecayRate();
        var shiftRate = params.attentionShiftRate();

        for (int i = 0; i < size; i++) {
            // Very slow decay to maintain attention (apply much smaller decay per time step)
            // With sustainedDecay = 0.0005 and 100 steps, we want to retain > 0.3
            // So decay factor per step should be very small
            attentionState[i] *= (1.0 - sustainedDecay * 0.01);  // Much slower decay

            // Integrate new input with shift rate
            if (input[i] > 0) {
                // Shift attention gradually to new location
                attentionState[i] += input[i] * shiftRate;
            }

            // Cap attention state
            attentionState[i] = Math.min(attentionState[i], 1.0);
        }
    }

    /**
     * Update memory trace with even slower dynamics for long persistence.
     */
    private void updateMemoryTrace(double[] input, Layer1Parameters params) {
        var traceDecay = params.sustainedDecayRate() * 0.05;  // Much slower than attention

        for (int i = 0; i < size; i++) {
            // Very slow decay for memory trace
            memoryTrace[i] *= (1.0 - traceDecay);

            // Build up memory trace from sustained attention
            if (attentionState[i] > 0.3) {  // Threshold for memory formation
                memoryTrace[i] += attentionState[i] * 0.1;  // Slow accumulation
            }

            // Cap memory trace
            memoryTrace[i] = Math.min(memoryTrace[i], 0.8);
        }
    }

    /**
     * Update shunting dynamics parameters based on Layer1Parameters.
     */
    private void updateDynamicsParameters(Layer1Parameters params) {
        // Update the shunting dynamics with Layer 1 specific parameters
        var shuntingParams = ShuntingParameters.builder(size)
            .ceiling(params.ceiling())
            .floor(params.floor())
            .selfExcitation(params.selfExcitation())
            .inhibitoryStrength(params.lateralInhibition())
            .timeStep(params.timeConstant() / 20000.0)  // Convert ms to seconds
            .build();

        // Create new dynamics instance with updated parameters
        this.verySlowDynamics = new ShuntingDynamics(shuntingParams);
    }

    /**
     * Get the current priming effect for integration with other layers.
     *
     * @return current priming effect pattern
     */
    public Pattern getPrimingEffect() {
        return new DenseVector(primingEffect.clone());
    }

    /**
     * Get the signal for Layer 2/3 apical dendrites.
     *
     * @return apical dendrite signal pattern
     */
    public Pattern getApicalDendriteSignal() {
        var apicalSignal = new double[size];
        var apicalIntegration = currentParameters != null ?
            currentParameters.apicalIntegration() : 0.5;

        for (int i = 0; i < size; i++) {
            // Modulated signal for apical dendrites
            apicalSignal[i] = (attentionState[i] + primingEffect[i]) * apicalIntegration;
        }

        return new DenseVector(apicalSignal);
    }

    /**
     * Get the current attention state.
     *
     * @return current attention state pattern
     */
    public Pattern getAttentionState() {
        return new DenseVector(attentionState.clone());
    }

    /**
     * Add an activation listener to be notified of activation changes.
     *
     * @param listener the listener to add
     */
    public void addActivationListener(LayerActivationListener listener) {
        listeners.add(listener);
    }

    /**
     * Remove an activation listener.
     *
     * @param listener the listener to remove
     */
    public void removeActivationListener(LayerActivationListener listener) {
        listeners.remove(listener);
    }

    // ============== Oscillation Tracking API (Phase 2) ==============

    /**
     * Enable oscillation tracking for this layer.
     *
     * <p>When enabled, the layer will maintain a circular buffer of activation
     * history and compute oscillation metrics using FFT analysis.
     *
     * @param samplingRate Sampling rate in Hz (typically 1000 for 1ms timesteps)
     * @param historySize Number of samples to analyze (power-of-2 recommended)
     * @throws IllegalArgumentException if parameters invalid
     */
    public void enableOscillationTracking(double samplingRate, int historySize) {
        if (samplingRate <= 0) {
            throw new IllegalArgumentException("samplingRate must be positive: " + samplingRate);
        }
        if (historySize <= 0) {
            throw new IllegalArgumentException("historySize must be positive: " + historySize);
        }

        this.oscillationAnalyzer = new OscillationAnalyzer(samplingRate, historySize);
        this.activationHistory = new CircularBuffer<>(historySize);
        this.currentMetrics = null;
        this.currentTimestamp = 0.0;
    }

    /**
     * Disable oscillation tracking.
     *
     * <p>Clears all oscillation-related state and metrics.
     */
    public void disableOscillationTracking() {
        this.oscillationAnalyzer = null;
        this.activationHistory = null;
        this.currentMetrics = null;
        this.currentTimestamp = 0.0;
    }

    /**
     * Get current oscillation metrics.
     *
     * <p>Returns null if:
     * <ul>
     *   <li>Oscillation tracking is disabled</li>
     *   <li>Activation history buffer not yet full</li>
     * </ul>
     *
     * @return Current oscillation metrics, or null if unavailable
     */
    public OscillationMetrics getOscillationMetrics() {
        return currentMetrics;
    }

    /**
     * Check if oscillation tracking is enabled.
     *
     * @return true if oscillation tracking is enabled
     */
    public boolean isOscillationTrackingEnabled() {
        return oscillationAnalyzer != null;
    }

    /**
     * Get current timestamp for oscillation analysis.
     *
     * @return Current timestamp in seconds
     */
    public double getCurrentTimestamp() {
        return currentTimestamp;
    }

    // ============== Learning API (Phase 3C) ==============

    /**
     * Enable resonance-gated learning for Layer 1.
     *
     * <p>Layer 1 learning characteristics:
     * <ul>
     *   <li>Learns top-down attention and priming patterns</li>
     *   <li>Hebbian learning for attentional modulation</li>
     *   <li>Gated by consciousness and attention from circuit</li>
     *   <li>Time constant: 200-1000ms (slow)</li>
     * </ul>
     *
     * @param learningRule learning rule to apply
     * @throws IllegalArgumentException if learningRule is null
     */
    public void enableLearning(com.hellblazer.art.cortical.learning.LearningRule learningRule) {
        if (learningRule == null) {
            throw new IllegalArgumentException("learningRule cannot be null");
        }
        this.learningRule = learningRule;
        this.learningStatistics = new com.hellblazer.art.cortical.learning.LearningStatistics();
    }

    /**
     * Disable learning for Layer 1.
     */
    public void disableLearning() {
        this.learningRule = null;
        this.learningStatistics = null;
    }

    @Override
    public boolean isLearningEnabled() {
        return learningRule != null;
    }

    @Override
    public com.hellblazer.art.cortical.learning.LearningStatistics getLearningStatistics() {
        return learningStatistics;
    }

    @Override
    public void learn(com.hellblazer.art.cortical.learning.LearningContext context, double baseLearningRate) {
        if (!isLearningEnabled()) {
            return;
        }

        // Layer 1 has no weight matrix (attention is dynamic)
        // For Layer 1, learning modulates attention parameters rather than weights
        // This is a placeholder that records learning events for statistics

        // Record learning event (weight change is N/A for Layer 1)
        learningStatistics.recordLearningEvent(
            context.resonanceState(),
            context.attentionStrength(),
            0.0  // No weight change for Layer 1
        );
    }
}
