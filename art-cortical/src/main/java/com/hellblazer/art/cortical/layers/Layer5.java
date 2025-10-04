package com.hellblazer.art.cortical.layers;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.cortical.analysis.CircularBuffer;
import com.hellblazer.art.cortical.analysis.OscillationAnalyzer;
import com.hellblazer.art.cortical.analysis.OscillationMetrics;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Layer 5 Implementation - Motor Output & Action Selection.
 *
 * <p>Layer 5 projects processed signals from Layer 2/3 to higher cortical areas
 * and subcortical structures. Implements decision formation and action selection.
 *
 * <p>Key characteristics:
 * <ul>
 *   <li>Medium time constants (50-200ms)</li>
 *   <li>Receives input from Layer 2/3 pyramidal cells</li>
 *   <li>Amplification/gating for salient features</li>
 *   <li>Output normalization for stable signaling</li>
 *   <li>Category signal generation</li>
 *   <li>Burst firing capability for important signals</li>
 * </ul>
 *
 * @author Migrated from art-laminar to art-cortical (Phase 3)
 */
public final class Layer5 implements Layer {

    private final String id;
    private final int size;
    private final WeightMatrix weights;
    private final List<LayerActivationListener> listeners;

    private Pattern activation;
    private double[] previousActivation;
    private Layer5Parameters currentParameters;

    // Oscillation tracking
    private OscillationAnalyzer oscillationAnalyzer;
    private CircularBuffer<double[]> activationHistory;
    private OscillationMetrics currentMetrics;
    private double currentTimestamp;

    // Learning infrastructure (Phase 3C: Multi-Layer Learning)
    private com.hellblazer.art.cortical.learning.LearningRule learningRule;
    private com.hellblazer.art.cortical.learning.LearningStatistics learningStatistics;

    public Layer5(String id, int size) {
        if (size <= 0) {
            throw new IllegalArgumentException("Layer size must be positive: " + size);
        }

        this.id = id;
        this.size = size;
        this.weights = new WeightMatrix(size, size);
        this.listeners = new CopyOnWriteArrayList<>();
        this.activation = new DenseVector(new double[size]);
        this.previousActivation = new double[size];
        this.currentTimestamp = 0.0;
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
        return LayerType.LAYER_5;
    }

    @Override
    public Pattern getActivation() {
        return activation;
    }

    @Override
    public void setActivation(Pattern activation) {
        if (activation.dimension() != size) {
            throw new IllegalArgumentException(
                "Activation dimension " + activation.dimension() +
                " does not match layer size " + size);
        }

        var oldActivation = this.activation;
        this.activation = activation;

        // Notify listeners
        for (var listener : listeners) {
            listener.onActivationChanged(id, oldActivation, activation);
        }
    }

    @Override
    public Pattern processBottomUp(Pattern input, LayerParameters parameters) {
        if (!(parameters instanceof Layer5Parameters)) {
            parameters = Layer5Parameters.builder().build();
        }
        currentParameters = (Layer5Parameters) parameters;

        // Convert input to array
        var inputArray = new double[size];
        for (var i = 0; i < Math.min(input.dimension(), size); i++) {
            inputArray[i] = input.get(i);
        }

        // Apply amplification gain
        var amplificationGain = currentParameters.amplificationGain();
        for (var i = 0; i < size; i++) {
            inputArray[i] *= amplificationGain;
        }

        // Check for burst firing conditions
        var burstThreshold = currentParameters.burstThreshold();
        var burstAmplification = currentParameters.burstAmplification();

        var shouldBurst = false;
        for (var i = 0; i < size; i++) {
            if (inputArray[i] > burstThreshold) {
                shouldBurst = true;
                break;
            }
        }

        // Apply burst amplification if needed
        if (shouldBurst) {
            for (var i = 0; i < size; i++) {
                if (inputArray[i] > burstThreshold) {
                    inputArray[i] *= burstAmplification;
                }
            }
        }

        // Blend with previous activation for state persistence
        var persistence = 1.0 - currentParameters.decayRate() * 0.01;
        var result = new double[size];
        for (var i = 0; i < size; i++) {
            result[i] = inputArray[i] + previousActivation[i] * persistence;
        }

        // Apply output gain
        var outputGain = currentParameters.outputGain();
        for (var i = 0; i < size; i++) {
            result[i] *= outputGain;
        }

        // Apply normalization
        var sum = 0.0;
        for (var i = 0; i < size; i++) {
            sum += result[i];
        }

        var normalization = currentParameters.outputNormalization();
        if (sum > 0.01 && normalization > 0) {
            var normalizer = 1.0 / (1.0 + normalization * sum);
            for (var i = 0; i < size; i++) {
                result[i] *= normalizer;
            }
        }

        // Apply ceiling and floor
        for (var i = 0; i < size; i++) {
            result[i] = Math.max(currentParameters.floor(),
                                Math.min(currentParameters.ceiling(), result[i]));
        }

        // Store for state persistence
        System.arraycopy(result, 0, previousActivation, 0, size);

        // Update activation and notify
        var oldActivation = this.activation;
        activation = new DenseVector(result);

        for (var listener : listeners) {
            listener.onActivationChanged(id, oldActivation, activation);
        }

        // Oscillation analysis (if enabled)
        if (oscillationAnalyzer != null && activationHistory != null) {
            activationHistory.add(result.clone());
            if (activationHistory.isFull()) {
                currentMetrics = oscillationAnalyzer.analyze(activationHistory, currentTimestamp);
            }
            currentTimestamp += 0.001;
        }

        return activation;
    }

    @Override
    public Pattern processTopDown(Pattern expectation, LayerParameters parameters) {
        // Layer 5 receives minimal top-down (it's an output layer)
        return activation;
    }

    @Override
    public Pattern processLateral(Pattern lateral, LayerParameters parameters) {
        // Layer 5 has weak lateral processing
        return activation;
    }

    @Deprecated
    @Override
    public void updateWeights(Pattern input, double learningRate) {
        if (learningRate < 0.0 || learningRate > 1.0) {
            throw new IllegalArgumentException(
                "Learning rate must be in [0, 1]: " + learningRate);
        }

        // Hebbian learning
        for (var i = 0; i < weights.getRows(); i++) {
            for (var j = 0; j < weights.getCols(); j++) {
                if (j < input.dimension()) {
                    var currentWeight = weights.get(i, j);
                    var inputValue = input.get(j);
                    var activationValue = activation.get(i);
                    var deltaW = learningRate * inputValue * activationValue;
                    weights.set(i, j, currentWeight + deltaW);
                }
            }
        }
    }

    @Override
    public void reset() {
        activation = new DenseVector(new double[size]);
        previousActivation = new double[size];
        currentParameters = null;

        // Clear oscillation tracking
        if (activationHistory != null) {
            activationHistory.clear();
        }
        currentMetrics = null;
        currentTimestamp = 0.0;
    }

    @Override
    public void close() {
        reset();
    }

    /**
     * Detect category formation based on threshold.
     *
     * @return true if any activation exceeds category threshold
     */
    public boolean isCategoryFormed() {
        if (currentParameters == null) {
            return false;
        }

        var threshold = currentParameters.categoryThreshold();
        for (var i = 0; i < size; i++) {
            if (activation.get(i) > threshold) {
                return true;
            }
        }
        return false;
    }

    /**
     * Get weight matrix for this layer.
     *
     * @return weight matrix
     */
    public WeightMatrix getWeights() {
        return weights;
    }

    /**
     * Add activation listener for monitoring layer dynamics.
     *
     * @param listener activation change listener
     */
    public void addActivationListener(LayerActivationListener listener) {
        if (listener != null) {
            listeners.add(listener);
        }
    }

    /**
     * Remove activation listener.
     *
     * @param listener listener to remove
     */
    public void removeActivationListener(LayerActivationListener listener) {
        listeners.remove(listener);
    }

    // ============== Oscillation Tracking API ==============

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
     * Enable resonance-gated learning for Layer 5.
     *
     * <p>Layer 5 learning characteristics:
     * <ul>
     *   <li>Learns motor output and action selection</li>
     *   <li>Hebbian learning for output mappings</li>
     *   <li>Gated by consciousness and attention from circuit</li>
     *   <li>Time constant: 50-200ms (medium)</li>
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
     * Disable learning for Layer 5.
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

        // Apply learning rule with modulated learning rate
        var newWeights = learningRule.update(
            context.preActivation(),
            context.postActivation(),
            weights,
            baseLearningRate * context.getLearningRateModulation()
        );

        // Compute weight change magnitude for statistics
        double weightChange = computeWeightChangeMagnitude(weights, newWeights);

        // Copy new weights to current weights (in-place update)
        copyWeights(newWeights, weights);

        // Update statistics
        learningStatistics.recordLearningEvent(
            context.resonanceState(),
            context.attentionStrength(),
            weightChange
        );
    }

    /**
     * Compute magnitude of weight change between two weight matrices.
     */
    private double computeWeightChangeMagnitude(WeightMatrix oldWeights, WeightMatrix newWeights) {
        double sumSquares = 0.0;
        for (int i = 0; i < oldWeights.getRows(); i++) {
            for (int j = 0; j < oldWeights.getCols(); j++) {
                double diff = newWeights.get(i, j) - oldWeights.get(i, j);
                sumSquares += diff * diff;
            }
        }
        return Math.sqrt(sumSquares);
    }

    /**
     * Copy weights from source to destination (in-place).
     */
    private void copyWeights(WeightMatrix source, WeightMatrix destination) {
        for (int i = 0; i < source.getRows(); i++) {
            for (int j = 0; j < source.getCols(); j++) {
                destination.set(i, j, source.get(i, j));
            }
        }
    }
}
