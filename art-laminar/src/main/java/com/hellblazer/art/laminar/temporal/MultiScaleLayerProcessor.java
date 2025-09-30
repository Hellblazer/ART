package com.hellblazer.art.laminar.temporal;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.canonical.TemporallyIntegratedPathway;
import com.hellblazer.art.laminar.parameters.LayerParameters;

/**
 * Processes layer activations with coordinated multi-scale dynamics.
 *
 * Coordinates three temporal scales:
 * 1. Fast shunting dynamics (pathway level)
 * 2. Medium chunking dynamics (layer level)
 * 3. Slow transmitter dynamics (pathway level)
 *
 * @author Hal Hildebrand
 */
public class MultiScaleLayerProcessor {

    private final TemporalChunkingLayer layer;
    private final TemporallyIntegratedPathway bottomUpPathway;
    private final MultiScaleCoordinator coordinator;

    private int fastStepCount;
    private int chunkingUpdateCount;
    private int slowUpdateCount;

    public MultiScaleLayerProcessor(TemporalChunkingLayer layer,
                                   TemporallyIntegratedPathway bottomUpPathway,
                                   MultiScaleCoordinator coordinator) {
        this.layer = layer;
        this.bottomUpPathway = bottomUpPathway;
        this.coordinator = coordinator;

        this.fastStepCount = 0;
        this.chunkingUpdateCount = 0;
        this.slowUpdateCount = 0;
    }

    /**
     * Create processor with standard multi-scale coordination.
     */
    public static MultiScaleLayerProcessor standard(TemporalChunkingLayer layer,
                                                    TemporallyIntegratedPathway pathway) {
        return new MultiScaleLayerProcessor(layer, pathway, MultiScaleCoordinator.standard());
    }

    /**
     * Create processor for real-time processing.
     */
    public static MultiScaleLayerProcessor realTime(TemporalChunkingLayer layer,
                                                   TemporallyIntegratedPathway pathway) {
        return new MultiScaleLayerProcessor(layer, pathway, MultiScaleCoordinator.realTime());
    }

    /**
     * Process a single input pattern through all temporal scales.
     *
     * @param input Input pattern
     * @param parameters Layer parameters
     * @return Processed pattern with multi-scale dynamics
     */
    public Pattern process(Pattern input, LayerParameters parameters) {
        // 1. Fast time scale: Process through pathway (includes shunting dynamics)
        double fastTimeStep = coordinator.advanceFastTimeStep();
        var processed = layer.processBottomUp(input, parameters);
        fastStepCount++;

        // 2. Medium time scale: Update temporal chunking if interval reached
        if (coordinator.shouldUpdateChunking()) {
            double chunkingTimeStep = coordinator.getChunkingTimeStep();
            layer.processWithChunking(processed, chunkingTimeStep);
            chunkingUpdateCount++;
        }

        // 3. Slow time scale: Update transmitter dynamics if interval reached
        if (coordinator.shouldUpdateSlowDynamics()) {
            double slowTimeStep = coordinator.getSlowTimeStep();
            if (bottomUpPathway.isTemporalDynamicsEnabled()) {
                bottomUpPathway.updateDynamics(slowTimeStep);
            }
            slowUpdateCount++;
        }

        return processed;
    }

    /**
     * Process a sequence of patterns with multi-scale coordination.
     *
     * @param inputs Sequence of input patterns
     * @param parameters Layer parameters
     * @return Final layer state after processing sequence
     */
    public LayerState processSequence(Iterable<Pattern> inputs, LayerParameters parameters) {
        Pattern lastOutput = null;

        for (var input : inputs) {
            lastOutput = process(input, parameters);
        }

        return layer.getLayerState();
    }

    /**
     * Get the current layer state with temporal context.
     */
    public LayerState getLayerState() {
        return layer.getLayerState();
    }

    /**
     * Reset all temporal state.
     */
    public void reset() {
        layer.reset();
        coordinator.reset();
        fastStepCount = 0;
        chunkingUpdateCount = 0;
        slowUpdateCount = 0;
    }

    // Statistics

    public record ProcessingStatistics(
        int totalFastSteps,
        int totalChunkingUpdates,
        int totalSlowUpdates,
        double actualChunkingToFastRatio,
        double actualSlowToFastRatio,
        MultiScaleCoordinator.TimingStatistics timing,
        ChunkingStatistics chunking
    ) {
        public static ProcessingStatistics from(MultiScaleLayerProcessor processor) {
            return new ProcessingStatistics(
                processor.fastStepCount,
                processor.chunkingUpdateCount,
                processor.slowUpdateCount,
                processor.fastStepCount > 0 ?
                    (double) processor.fastStepCount / processor.chunkingUpdateCount : 0.0,
                processor.fastStepCount > 0 ?
                    (double) processor.fastStepCount / processor.slowUpdateCount : 0.0,
                processor.coordinator.getStatistics(),
                processor.layer.getChunkingStatistics()
            );
        }
    }

    public ProcessingStatistics getStatistics() {
        return ProcessingStatistics.from(this);
    }

    // Getters

    public TemporalChunkingLayer getLayer() {
        return layer;
    }

    public TemporallyIntegratedPathway getBottomUpPathway() {
        return bottomUpPathway;
    }

    public MultiScaleCoordinator getCoordinator() {
        return coordinator;
    }

    public int getFastStepCount() {
        return fastStepCount;
    }

    public int getChunkingUpdateCount() {
        return chunkingUpdateCount;
    }

    public int getSlowUpdateCount() {
        return slowUpdateCount;
    }
}