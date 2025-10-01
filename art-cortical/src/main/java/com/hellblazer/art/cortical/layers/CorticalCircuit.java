package com.hellblazer.art.cortical.layers;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
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
            temporalResult
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
     * Reset all layers and temporal processing.
     */
    public void reset() {
        layer1.reset();
        layer23.reset();
        layer4.reset();
        layer5.reset();
        layer6.reset();
        temporalProcessor.reset();
    }

    /**
     * Update weights across all layers based on learning.
     *
     * @param input input pattern
     * @param learningRate global learning rate
     */
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
        TemporalProcessor.TemporalResult temporalResult
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
    }
}
