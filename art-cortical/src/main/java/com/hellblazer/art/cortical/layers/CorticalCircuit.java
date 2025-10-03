package com.hellblazer.art.cortical.layers;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.cortical.learning.LearningContext;
import com.hellblazer.art.cortical.learning.LearningRule;
import com.hellblazer.art.cortical.learning.LearningStatistics;
import com.hellblazer.art.cortical.resonance.EnhancedResonanceDetector;
import com.hellblazer.art.cortical.resonance.ResonanceState;
import com.hellblazer.art.cortical.temporal.TemporalProcessor;

/**
 * Complete 6-layer cortical circuit with temporal processing integration.
 *
 * <p>Integrates the full cortical architecture:
 * <ul>
 *   <li><b>Temporal Processing</b>: Working memory → masking field chunking (Phase 2)</li>
 *   <li><b>Layer 4</b>: Thalamic driving input (fast, 10-50ms)</li>
 *   <li><b>Layer 2/3</b>: Inter-areal communication, prediction (medium, 30-150ms)</li>
 *   <li><b>Layer 1</b>: Apical dendrites, top-down attention (slow, 200-1000ms)</li>
 *   <li><b>Layer 6</b>: Corticothalamic feedback, gain modulation (slow, 100-500ms)</li>
 *   <li><b>Layer 5</b>: Motor output, action selection (medium, 50-200ms)</li>
 * </ul>
 *
 * <p><b>Processing Pathways</b>:
 * <ol>
 *   <li><b>Temporal Chunking</b>: Input → TemporalProcessor → chunked patterns</li>
 *   <li><b>Bottom-Up</b>: L4 → L2/3 → L1 (feedforward sensory processing)</li>
 *   <li><b>Top-Down</b>: L6 → L2/3 → L4 (feedback modulation/prediction)</li>
 *   <li><b>Output</b>: L2/3 → L5 (motor/action signals)</li>
 * </ol>
 *
 * <p><b>Biological References</b>:
 * <ul>
 *   <li>Douglas & Martin (2004). Neuronal circuits of the neocortex. Ann Rev Neurosci.</li>
 *   <li>Grossberg (2013). Adaptive Resonance Theory. Neural Networks, 37, 1-47.</li>
 *   <li>Kazerounian & Grossberg (2014). LIST PARSE model. Psych Review, 121(4), 621-671.</li>
 *   <li>Larkum et al. (2009). Synaptic integration in tuft dendrites. Science, 325, 756-760.</li>
 * </ul>
 *
 * @author Created for art-cortical Phase 3, Milestone 5 (Final Integration)
 */
public final class CorticalCircuit implements AutoCloseable {

    // 6-layer cortical circuit
    private final Layer1 layer1;
    private final Layer23 layer23;
    private final Layer4 layer4;
    private final Layer5 layer5;
    private final Layer6 layer6;

    // Temporal processing (Phase 2 integration)
    private final TemporalProcessor temporalProcessor;

    // Enhanced resonance detection (Phase 2D)
    private EnhancedResonanceDetector resonanceDetector;
    private double currentTimestamp;

    // Learning infrastructure (Phase 3B/3C: Multi-Layer Learning)
    private boolean learningEnabled;
    private double resonanceLearningThreshold;
    private double attentionLearningThreshold;

    // Layer-specific learning rates (multi-timescale learning)
    private double baseLearningRateL1;   // Slow: 0.001 (attention/priming)
    private double baseLearningRateL23;  // Medium: 0.01 (inter-areal grouping)
    private double baseLearningRateL4;   // Fast: 0.1 (bottom-up features)
    private double baseLearningRateL5;   // Medium: 0.01 (motor output)
    private double baseLearningRateL6;   // Medium-slow: 0.005 (corticothalamic)

    private LearningStatistics circuitLearningStats;

    // Circuit parameters
    private final Layer1Parameters layer1Params;
    private final Layer23Parameters layer23Params;
    private final Layer4Parameters layer4Params;
    private final Layer5Parameters layer5Params;
    private final Layer6Parameters layer6Params;

    /**
     * Create cortical circuit with all layers and temporal processing.
     *
     * @param size number of units per layer (uniform sizing)
     * @param layer1Params Layer 1 parameters
     * @param layer23Params Layer 2/3 parameters
     * @param layer4Params Layer 4 parameters
     * @param layer5Params Layer 5 parameters
     * @param layer6Params Layer 6 parameters
     * @param temporalProcessor temporal processing pipeline
     */
    public CorticalCircuit(int size,
                          Layer1Parameters layer1Params,
                          Layer23Parameters layer23Params,
                          Layer4Parameters layer4Params,
                          Layer5Parameters layer5Params,
                          Layer6Parameters layer6Params,
                          TemporalProcessor temporalProcessor) {
        // Initialize all 6 layers
        this.layer1 = new Layer1("L1", size);
        this.layer23 = new Layer23("L2/3", size);
        this.layer4 = new Layer4("L4", size);
        this.layer5 = new Layer5("L5", size);
        this.layer6 = new Layer6("L6", size);

        // Store parameters
        this.layer1Params = layer1Params;
        this.layer23Params = layer23Params;
        this.layer4Params = layer4Params;
        this.layer5Params = layer5Params;
        this.layer6Params = layer6Params;

        // Temporal processing
        this.temporalProcessor = temporalProcessor;

        // Enhanced resonance detection (disabled by default)
        this.resonanceDetector = null;
        this.currentTimestamp = 0.0;

        // Learning (disabled by default, Phase 3C: Multi-timescale learning)
        this.learningEnabled = false;
        this.resonanceLearningThreshold = 0.7;
        this.attentionLearningThreshold = 0.3;

        // Multi-timescale learning rates (biologically motivated)
        this.baseLearningRateL1 = 0.001;   // Slow: attention/priming
        this.baseLearningRateL23 = 0.01;   // Medium: inter-areal grouping
        this.baseLearningRateL4 = 0.1;     // Fast: bottom-up features
        this.baseLearningRateL5 = 0.01;    // Medium: motor output
        this.baseLearningRateL6 = 0.005;   // Medium-slow: corticothalamic

        this.circuitLearningStats = new LearningStatistics();
    }

    /**
     * Process input through complete cortical circuit.
     *
     * <p>Processing sequence:
     * <ol>
     *   <li>Temporal chunking via working memory</li>
     *   <li>Bottom-up: L4 → L2/3 → L1</li>
     *   <li>Top-down: L6 → L2/3 → L4</li>
     *   <li>Output: L2/3 → L5</li>
     * </ol>
     *
     * @param input input pattern (sensory/thalamic)
     * @return processed output from Layer 5
     */
    public Pattern process(Pattern input) {
        // Step 1: Temporal chunking (Phase 2)
        var inputArray = ((DenseVector) input).data();
        var temporalResult = temporalProcessor.processItem(inputArray);

        // Extract chunked pattern for spatial processing
        var wmState = temporalResult.workingMemoryState();
        var chunkedPattern = new DenseVector(wmState.getCombinedPattern());

        // Step 2: Bottom-up pathway (L4 → L2/3 → L1)
        var l4Output = layer4.processBottomUp(chunkedPattern, layer4Params);
        var l23Output = layer23.processBottomUp(l4Output, layer23Params);
        var l1Output = layer1.processBottomUp(l23Output, layer1Params);

        // Step 3: Top-down pathway (L6 → L2/3 → L4)
        // L6 generates expectations based on L2/3 categories
        var l6Output = layer6.processBottomUp(l23Output, layer6Params);

        // Top-down modulation back to L2/3 and L4
        var l23Modulated = layer23.processTopDown(l6Output, layer23Params);
        var l4Modulated = layer4.processTopDown(l23Modulated, layer4Params);

        // Step 4: Layer 1 top-down priming to L2/3
        var l23WithL1 = layer23.processTopDown(l1Output, layer23Params);

        // Step 5: Output pathway (L2/3 → L5)
        var l5Output = layer5.processBottomUp(l23WithL1, layer5Params);

        return l5Output;
    }

    /**
     * Process input with explicit bottom-up and top-down separation.
     * Useful for testing and analyzing pathway contributions.
     *
     * @param input input pattern
     * @return circuit processing result with pathway activations
     */
    public CircuitResult processDetailed(Pattern input) {
        // Temporal processing
        var inputArray = ((DenseVector) input).data();
        var temporalResult = temporalProcessor.processItem(inputArray);
        var wmState = temporalResult.workingMemoryState();
        var chunkedPattern = new DenseVector(wmState.getCombinedPattern());

        // Bottom-up pathway
        var l4BottomUp = layer4.processBottomUp(chunkedPattern, layer4Params);
        var l23BottomUp = layer23.processBottomUp(l4BottomUp, layer23Params);
        var l1BottomUp = layer1.processBottomUp(l23BottomUp, layer1Params);

        // Top-down pathway
        var l6TopDown = layer6.processBottomUp(l23BottomUp, layer6Params);
        var l23TopDown = layer23.processTopDown(l6TopDown, layer23Params);
        var l4TopDown = layer4.processTopDown(l23TopDown, layer4Params);

        // Layer 1 priming
        var l23WithL1 = layer23.processTopDown(l1BottomUp, layer23Params);

        // Output
        var l5Output = layer5.processBottomUp(l23WithL1, layer5Params);

        // Enhanced resonance detection (Phase 2D)
        ResonanceState resonanceState = null;
        if (resonanceDetector != null) {
            // Record activation history for oscillation analysis
            var l4Data = ((DenseVector) l4BottomUp).data();
            var l1Data = ((DenseVector) l1BottomUp).data();
            resonanceDetector.recordBottomUp(l4Data);  // Bottom-up from Layer 4
            resonanceDetector.recordTopDown(l1Data);   // Top-down from Layer 1

            // Detect resonance between bottom-up features and top-down expectations
            var l23Data = ((DenseVector) l23BottomUp).data();
            var l6Data = ((DenseVector) l6TopDown).data();
            resonanceState = resonanceDetector.detectResonance(l23Data, l6Data, currentTimestamp);

            currentTimestamp += 0.001;  // 1ms timesteps
        }

        return new CircuitResult(
            chunkedPattern,
            l4BottomUp,
            l23BottomUp,
            l1BottomUp,
            l6TopDown,
            l23TopDown,
            l4TopDown,
            l23WithL1,
            l5Output,
            temporalResult,
            resonanceState
        );
    }

    /**
     * Get Layer 1 (apical dendrites, attention).
     */
    public Layer1 getLayer1() {
        return layer1;
    }

    /**
     * Get Layer 2/3 (inter-areal, prediction).
     */
    public Layer23 getLayer23() {
        return layer23;
    }

    /**
     * Get Layer 4 (thalamic input).
     */
    public Layer4 getLayer4() {
        return layer4;
    }

    /**
     * Get Layer 5 (motor output).
     */
    public Layer5 getLayer5() {
        return layer5;
    }

    /**
     * Get Layer 6 (corticothalamic feedback).
     */
    public Layer6 getLayer6() {
        return layer6;
    }

    /**
     * Get temporal processor.
     */
    public TemporalProcessor getTemporalProcessor() {
        return temporalProcessor;
    }

    /**
     * Enable enhanced resonance detection with consciousness metrics.
     *
     * <p>This enables tracking of:
     * <ul>
     *   <li>Bottom-up/top-down phase synchronization</li>
     *   <li>Gamma oscillations (30-50 Hz)</li>
     *   <li>Consciousness likelihood estimation</li>
     * </ul>
     *
     * @param vigilanceThreshold ART vigilance threshold [0, 1]
     * @param samplingRate Sampling rate in Hz (typically 1000 for 1ms timesteps)
     * @param historySize Number of samples for oscillation analysis (power-of-2 recommended)
     */
    public void enableResonanceDetection(double vigilanceThreshold, double samplingRate, int historySize) {
        this.resonanceDetector = new EnhancedResonanceDetector(vigilanceThreshold, samplingRate, historySize);
        this.currentTimestamp = 0.0;
    }

    /**
     * Disable enhanced resonance detection.
     */
    public void disableResonanceDetection() {
        this.resonanceDetector = null;
        this.currentTimestamp = 0.0;
    }

    /**
     * Check if resonance detection is enabled.
     *
     * @return true if resonance detection is enabled
     */
    public boolean isResonanceDetectionEnabled() {
        return resonanceDetector != null;
    }

    /**
     * Get the current resonance detector (if enabled).
     *
     * @return resonance detector or null if disabled
     */
    public EnhancedResonanceDetector getResonanceDetector() {
        return resonanceDetector;
    }

    /**
     * Reset all layers and temporal processing.
     */
    public void reset() {
        layer1.reset();
        layer23.reset();
        layer4.reset();
        layer5.reset();
        layer6.reset();
        temporalProcessor.reset();
        if (resonanceDetector != null) {
            resonanceDetector.reset();
        }
        currentTimestamp = 0.0;
    }

    /**
     * Update weights across all layers based on learning.
     *
     * @param input input pattern
     * @param learningRate global learning rate
     * @deprecated Use {@link #processAndLearn(Pattern)} with {@link #enableLearning(LearningRule, double)} instead
     */
    @Deprecated
    public void learn(Pattern input, double learningRate) {
        // Process through circuit
        var result = processDetailed(input);

        // Update weights in each layer (Hebbian-style)
        layer4.updateWeights(result.layer4Output(), learningRate);
        layer23.updateWeights(result.layer23Output(), learningRate);
        layer1.updateWeights(result.layer1Output(), learningRate);
        layer6.updateWeights(result.layer6Output(), learningRate);
        layer5.updateWeights(result.layer5Output(), learningRate);
    }

    /**
     * Enable resonance-gated learning for the cortical circuit (all layers).
     *
     * <p>Phase 3C: Multi-timescale learning across all cortical layers:
     * <ul>
     *   <li>Layer 1: Slow learning (0.001) - top-down attention/priming</li>
     *   <li>Layer 2/3: Medium learning (0.01) - inter-areal grouping</li>
     *   <li>Layer 4: Fast learning (0.1) - bottom-up features</li>
     *   <li>Layer 5: Medium learning (0.01) - motor output</li>
     *   <li>Layer 6: Medium-slow learning (0.005) - corticothalamic feedback</li>
     * </ul>
     *
     * <p>Learning is gated by:
     * <ul>
     *   <li>Resonance threshold (consciousness likelihood, default 0.7)</li>
     *   <li>Attention threshold (Layer 1 strength, default 0.3)</li>
     * </ul>
     *
     * <p>Uses default biologically-motivated learning rates for each layer.
     *
     * @param learningRule Learning rule to apply (e.g., HebbianLearning, ResonanceGatedLearning)
     * @throws IllegalArgumentException if learningRule is null
     */
    public void enableLearning(LearningRule learningRule) {
        if (learningRule == null) {
            throw new IllegalArgumentException("learningRule cannot be null");
        }

        this.learningEnabled = true;

        // Enable learning in all layers with appropriate learning rules
        layer1.enableLearning(learningRule);
        layer23.enableLearning(learningRule);
        layer4.enableLearning(learningRule);
        layer5.enableLearning(learningRule);
        layer6.enableLearning(learningRule);

        // Reset statistics
        this.circuitLearningStats = new LearningStatistics();
    }

    /**
     * Enable resonance-gated learning with custom learning rate for Layer 4.
     *
     * <p>Backward compatible method - uses custom rate for Layer 4, defaults for others.
     *
     * @param learningRule Learning rule to apply
     * @param baseLearningRateL4 Base learning rate for Layer 4 (typically 0.01-0.1)
     * @throws IllegalArgumentException if learningRule is null or rates invalid
     * @deprecated Use {@link #enableLearning(LearningRule)} or {@link #enableLearningWithRates}
     */
    @Deprecated
    public void enableLearning(LearningRule learningRule, double baseLearningRateL4) {
        if (baseLearningRateL4 <= 0.0 || baseLearningRateL4 > 1.0) {
            throw new IllegalArgumentException("baseLearningRateL4 must be in (0, 1]: " + baseLearningRateL4);
        }
        this.baseLearningRateL4 = baseLearningRateL4;
        enableLearning(learningRule);
    }

    /**
     * Enable learning with custom rates for all layers.
     *
     * <p>Allows fine-grained control over learning rates for each layer.
     *
     * @param learningRule Learning rule to apply
     * @param rateL1 Layer 1 learning rate (slow, typically 0.001)
     * @param rateL23 Layer 2/3 learning rate (medium, typically 0.01)
     * @param rateL4 Layer 4 learning rate (fast, typically 0.1)
     * @param rateL5 Layer 5 learning rate (medium, typically 0.01)
     * @param rateL6 Layer 6 learning rate (medium-slow, typically 0.005)
     * @throws IllegalArgumentException if any rate is invalid
     */
    public void enableLearningWithRates(
            LearningRule learningRule,
            double rateL1,
            double rateL23,
            double rateL4,
            double rateL5,
            double rateL6) {

        if (learningRule == null) {
            throw new IllegalArgumentException("learningRule cannot be null");
        }

        validateLearningRate(rateL1, "rateL1");
        validateLearningRate(rateL23, "rateL23");
        validateLearningRate(rateL4, "rateL4");
        validateLearningRate(rateL5, "rateL5");
        validateLearningRate(rateL6, "rateL6");

        this.baseLearningRateL1 = rateL1;
        this.baseLearningRateL23 = rateL23;
        this.baseLearningRateL4 = rateL4;
        this.baseLearningRateL5 = rateL5;
        this.baseLearningRateL6 = rateL6;

        enableLearning(learningRule);
    }

    /**
     * Validate learning rate is in valid range.
     */
    private void validateLearningRate(double rate, String name) {
        if (rate <= 0.0 || rate > 1.0) {
            throw new IllegalArgumentException(name + " must be in (0, 1]: " + rate);
        }
    }

    /**
     * Disable learning for the cortical circuit (all layers).
     */
    public void disableLearning() {
        this.learningEnabled = false;
        layer1.disableLearning();
        layer23.disableLearning();
        layer4.disableLearning();
        layer5.disableLearning();
        layer6.disableLearning();
    }

    /**
     * Check if learning is enabled.
     *
     * @return true if learning is enabled
     */
    public boolean isLearningEnabled() {
        return learningEnabled;
    }

    /**
     * Set resonance threshold for learning.
     *
     * @param threshold consciousness likelihood threshold [0, 1] (typically 0.7)
     * @throws IllegalArgumentException if threshold out of range
     */
    public void setResonanceLearningThreshold(double threshold) {
        if (threshold < 0.0 || threshold > 1.0) {
            throw new IllegalArgumentException("threshold must be in [0, 1]: " + threshold);
        }
        this.resonanceLearningThreshold = threshold;
    }

    /**
     * Set attention threshold for learning.
     *
     * @param threshold attention strength threshold [0, 1] (typically 0.3)
     * @throws IllegalArgumentException if threshold out of range
     */
    public void setAttentionLearningThreshold(double threshold) {
        if (threshold < 0.0 || threshold > 1.0) {
            throw new IllegalArgumentException("threshold must be in [0, 1]: " + threshold);
        }
        this.attentionLearningThreshold = threshold;
    }

    /**
     * Process input and apply learning if enabled.
     *
     * <p>This is the primary method for training the circuit:
     * <ol>
     *   <li>Process input through all layers (processDetailed)</li>
     *   <li>Check if learning should occur (resonance + attention)</li>
     *   <li>Apply learning to layers with gated learning rate</li>
     *   <li>Update circuit-level statistics</li>
     * </ol>
     *
     * @param input input pattern
     * @return circuit processing result (same as processDetailed)
     */
    public CircuitResult processAndLearn(Pattern input) {
        // Process through circuit
        var result = processDetailed(input);

        // Apply learning if enabled
        if (learningEnabled && shouldLearn(result)) {
            applyLearning(result);
        }

        return result;
    }

    /**
     * Get circuit-level learning statistics.
     *
     * @return learning statistics or null if learning disabled
     */
    public LearningStatistics getCircuitLearningStatistics() {
        return learningEnabled ? circuitLearningStats : null;
    }

    /**
     * Check if learning should occur for this processing result.
     *
     * @param result circuit processing result
     * @return true if learning should occur
     */
    private boolean shouldLearn(CircuitResult result) {
        // Require resonance detection to be enabled
        if (result.resonanceState() == null) {
            return false;
        }

        // Check consciousness threshold
        double consciousness = result.resonanceState().consciousnessLikelihood();
        if (consciousness < resonanceLearningThreshold) {
            return false;
        }

        // Check attention threshold (from Layer 1 activation)
        double attention = computeAttentionStrength(result.layer1Output());
        if (attention < attentionLearningThreshold) {
            return false;
        }

        return true;
    }

    /**
     * Compute attention strength from Layer 1 activation.
     *
     * @param layer1Activation Layer 1 activation pattern
     * @return attention strength [0, 1]
     */
    private double computeAttentionStrength(Pattern layer1Activation) {
        // Use L2 norm of Layer 1 activation as attention strength
        var data = ((DenseVector) layer1Activation).data();
        double sumSquares = 0.0;
        for (double v : data) {
            sumSquares += v * v;
        }
        return Math.sqrt(sumSquares / data.length);
    }

    /**
     * Apply learning to all circuit layers (Phase 3C: Multi-Layer Learning).
     *
     * <p>Multi-timescale learning across cortical hierarchy:
     * <ol>
     *   <li>Layer 4: Fast learning (bottom-up feature extraction)</li>
     *   <li>Layer 2/3: Medium learning (inter-areal grouping)</li>
     *   <li>Layer 6: Medium-slow learning (corticothalamic modulation)</li>
     *   <li>Layer 5: Medium learning (motor output selection)</li>
     *   <li>Layer 1: Slow learning (top-down attention/priming)</li>
     * </ol>
     *
     * @param result circuit processing result with activations
     */
    private void applyLearning(CircuitResult result) {
        // Compute attention strength for learning modulation
        double attentionStrength = computeAttentionStrength(result.layer1Output());

        // Layer 4: Fast learning (bottom-up features)
        // Pre: temporally chunked input, Post: Layer 4 activation
        var layer4Context = new LearningContext(
            result.temporalPattern(),
            result.layer4Output(),
            result.resonanceState(),
            attentionStrength,
            currentTimestamp
        );
        layer4.learn(layer4Context, baseLearningRateL4);

        // Layer 2/3: Medium learning (inter-areal integration)
        // Pre: Layer 4 output, Post: Layer 2/3 activation
        var layer23Context = new LearningContext(
            result.layer4Output(),
            result.layer23Output(),
            result.resonanceState(),
            attentionStrength,
            currentTimestamp
        );
        layer23.learn(layer23Context, baseLearningRateL23);

        // Layer 6: Medium-slow learning (corticothalamic feedback)
        // Pre: Layer 2/3 output, Post: Layer 6 activation
        var layer6Context = new LearningContext(
            result.layer23Output(),
            result.layer6Output(),
            result.resonanceState(),
            attentionStrength,
            currentTimestamp
        );
        layer6.learn(layer6Context, baseLearningRateL6);

        // Layer 5: Medium learning (motor output)
        // Pre: Layer 2/3 with L1 priming, Post: Layer 5 activation
        var layer5Context = new LearningContext(
            result.layer23WithL1(),
            result.layer5Output(),
            result.resonanceState(),
            attentionStrength,
            currentTimestamp
        );
        layer5.learn(layer5Context, baseLearningRateL5);

        // Layer 1: Slow learning (top-down attention)
        // Pre: Layer 2/3 output, Post: Layer 1 activation
        var layer1Context = new LearningContext(
            result.layer23Output(),
            result.layer1Output(),
            result.resonanceState(),
            attentionStrength,
            currentTimestamp
        );
        layer1.learn(layer1Context, baseLearningRateL1);

        // Update circuit-level statistics
        circuitLearningStats.recordLearningEvent(
            result.resonanceState(),
            attentionStrength,
            0.0  // Weight change computed internally by layers
        );
    }

    @Override
    public void close() {
        layer1.close();
        layer23.close();
        layer4.close();
        layer5.close();
        layer6.close();
    }

    /**
     * Detailed circuit processing result showing all pathway activations.
     *
     * @param temporalPattern Temporally chunked input pattern
     * @param layer4Output Layer 4 bottom-up output
     * @param layer23Output Layer 2/3 bottom-up output
     * @param layer1Output Layer 1 output (top-down priming)
     * @param layer6Output Layer 6 output (expectation/modulation)
     * @param layer23TopDown Layer 2/3 after top-down from L6
     * @param layer4TopDown Layer 4 after top-down modulation
     * @param layer23WithL1 Layer 2/3 with L1 priming
     * @param layer5Output Final output from Layer 5
     * @param temporalResult Temporal processing result
     * @param resonanceState Enhanced resonance state (null if detection disabled)
     */
    public record CircuitResult(
        Pattern temporalPattern,
        Pattern layer4Output,
        Pattern layer23Output,
        Pattern layer1Output,
        Pattern layer6Output,
        Pattern layer23TopDown,
        Pattern layer4TopDown,
        Pattern layer23WithL1,
        Pattern layer5Output,
        TemporalProcessor.TemporalResult temporalResult,
        ResonanceState resonanceState
    ) {
        /**
         * Get final circuit output (from Layer 5).
         */
        public Pattern getFinalOutput() {
            return layer5Output;
        }

        /**
         * Check if temporal chunking occurred.
         */
        public boolean hasTemporalChunks() {
            return temporalResult.hasChunks();
        }

        /**
         * Get number of temporal chunks formed.
         */
        public int getChunkCount() {
            return temporalResult.chunkCount();
        }

        /**
         * Check if resonance was detected (requires resonance detection enabled).
         */
        public boolean hasResonance() {
            return resonanceState != null && resonanceState.artResonance();
        }

        /**
         * Check if likely conscious perception (requires resonance detection enabled).
         *
         * @param threshold consciousness likelihood threshold (typically 0.7)
         * @return true if likely conscious perception
         */
        public boolean isLikelyConscious(double threshold) {
            return resonanceState != null && resonanceState.isLikelyConscious(threshold);
        }

        /**
         * Get consciousness likelihood [0, 1] (requires resonance detection enabled).
         *
         * @return consciousness likelihood or 0.0 if detection disabled
         */
        public double getConsciousnessLikelihood() {
            return resonanceState != null ? resonanceState.consciousnessLikelihood() : 0.0;
        }
    }
}
