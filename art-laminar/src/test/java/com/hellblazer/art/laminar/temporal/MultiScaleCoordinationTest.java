package com.hellblazer.art.laminar.temporal;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.canonical.*;
import com.hellblazer.art.laminar.core.Layer;
import com.hellblazer.art.laminar.core.LayerType;
import com.hellblazer.art.laminar.events.LayerActivationListener;
import com.hellblazer.art.laminar.parameters.LayerParameters;
import com.hellblazer.art.temporal.core.ShuntingParameters;
import com.hellblazer.art.temporal.core.TransmitterParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for multi-scale temporal coordination.
 *
 * @author Hal Hildebrand
 */
@DisplayName("Multi-Scale Coordination Tests")
class MultiScaleCoordinationTest extends CanonicalCircuitTestBase {

    private MultiScaleCoordinator coordinator;

    @BeforeEach
    void setUp() {
        coordinator = MultiScaleCoordinator.standard();
    }

    @Test
    @DisplayName("Coordinator initializes with correct time scales")
    void testInitialization() {
        assertEquals(TimeScale.FAST, coordinator.getFastTimeScale());
        assertEquals(TimeScale.MEDIUM, coordinator.getMediumTimeScale());
        assertEquals(TimeScale.SLOW, coordinator.getSlowTimeScale());

        assertEquals(0.0, coordinator.getCurrentTime(), 1e-10);
    }

    @Test
    @DisplayName("Fast time step advances correctly")
    void testFastTimeStep() {
        double dt1 = coordinator.advanceFastTimeStep();
        assertEquals(TimeScale.FAST.getTypicalTimeStep(), dt1, 1e-10);

        double time1 = coordinator.getCurrentTime();
        assertTrue(time1 > 0, "Time should advance");

        double dt2 = coordinator.advanceFastTimeStep();
        assertEquals(dt1, dt2, 1e-10, "Time steps should be consistent");

        double time2 = coordinator.getCurrentTime();
        assertEquals(time1 + dt2, time2, 1e-10);
    }

    @Test
    @DisplayName("Chunking updates at correct intervals")
    void testChunkingUpdateInterval() {
        int fastSteps = 0;
        int chunkingUpdates = 0;

        // Run for sufficient time to trigger chunking updates
        for (int i = 0; i < 100; i++) {
            coordinator.advanceFastTimeStep();
            fastSteps++;

            if (coordinator.shouldUpdateChunking()) {
                coordinator.getChunkingTimeStep();
                chunkingUpdates++;
            }
        }

        assertTrue(chunkingUpdates > 0, "Should have some chunking updates");
        double ratio = (double) fastSteps / chunkingUpdates;

        // Should be close to expected ratio
        double expectedRatio = coordinator.getChunkingToFastRatio();
        assertTrue(Math.abs(ratio - expectedRatio) < 2.0,
                  String.format("Ratio %.2f should be close to expected %.2f",
                               ratio, expectedRatio));
    }

    @Test
    @DisplayName("Slow updates at correct intervals")
    void testSlowUpdateInterval() {
        int fastSteps = 0;
        int slowUpdates = 0;

        // Run for sufficient time
        for (int i = 0; i < 1000; i++) {
            coordinator.advanceFastTimeStep();
            fastSteps++;

            if (coordinator.shouldUpdateSlowDynamics()) {
                coordinator.getSlowTimeStep();
                slowUpdates++;
            }
        }

        assertTrue(slowUpdates > 0, "Should have some slow updates");
        double ratio = (double) fastSteps / slowUpdates;

        double expectedRatio = coordinator.getSlowToFastRatio();
        assertTrue(Math.abs(ratio - expectedRatio) < 10.0,
                  String.format("Ratio %.2f should be close to expected %.2f",
                               ratio, expectedRatio));
    }

    @Test
    @DisplayName("Time scale separation ratios are correct")
    void testTimeScaleSeparation() {
        double chunkingToFast = coordinator.getChunkingToFastRatio();
        double slowToFast = coordinator.getSlowToFastRatio();
        double slowToChunking = coordinator.getSlowToChunkingRatio();

        assertTrue(chunkingToFast > 1.0,
                  "Medium should be slower than fast");
        assertTrue(slowToFast > chunkingToFast,
                  "Slow should be slower than medium");
        assertTrue(slowToChunking > 1.0,
                  "Slow should be slower than chunking");

        // Verify multiplication property
        assertEquals(slowToFast, chunkingToFast * slowToChunking, 1e-6,
                    "Ratios should multiply correctly");
    }

    @Test
    @DisplayName("Reset clears all timing state")
    void testReset() {
        // Advance time
        for (int i = 0; i < 10; i++) {
            coordinator.advanceFastTimeStep();
        }

        assertTrue(coordinator.getCurrentTime() > 0, "Time should have advanced");

        // Reset
        coordinator.reset();

        assertEquals(0.0, coordinator.getCurrentTime(), 1e-10,
                    "Time should be reset");
        assertFalse(coordinator.shouldUpdateChunking(),
                   "Should not trigger chunking update immediately after reset");
        assertFalse(coordinator.shouldUpdateSlowDynamics(),
                   "Should not trigger slow update immediately after reset");
    }

    @Test
    @DisplayName("Statistics track timing correctly")
    void testStatistics() {
        // Advance through several cycles
        for (int i = 0; i < 50; i++) {
            coordinator.advanceFastTimeStep();
            if (coordinator.shouldUpdateChunking()) {
                coordinator.getChunkingTimeStep();
            }
            if (coordinator.shouldUpdateSlowDynamics()) {
                coordinator.getSlowTimeStep();
            }
        }

        var stats = coordinator.getStatistics();

        assertTrue(stats.currentTime() > 0, "Should have advanced time");
        assertTrue(stats.chunkingToFastRatio() > 1.0, "Medium slower than fast");
        assertTrue(stats.slowToFastRatio() > stats.chunkingToFastRatio(),
                  "Slow slower than medium");
    }

    @Test
    @DisplayName("Real-time coordinator has faster updates")
    void testRealTimeCoordinator() {
        var realTime = MultiScaleCoordinator.realTime();

        double standardRatio = coordinator.getChunkingToFastRatio();
        double realTimeRatio = realTime.getChunkingToFastRatio();

        assertTrue(realTimeRatio <= standardRatio,
                  "Real-time should have faster chunking updates");
    }

    @Test
    @DisplayName("Multi-scale processor coordinates all dynamics")
    void testMultiScaleProcessor() {
        // Create test components
        var baseLayer = new MockLayer(10);
        var chunkingLayer = new TemporalChunkingLayerDecorator(
            baseLayer,
            ChunkingParameters.fastChunking()
        );

        var shuntingParams = createStandardShuntingParameters();
        var transmitterParams = createStandardTransmitterParameters();

        var pathway = new MockPathway(10, shuntingParams, transmitterParams);

        var processor = MultiScaleLayerProcessor.standard(chunkingLayer, pathway);

        // Process multiple patterns
        LayerParameters params = null;  // Use null for test
        for (int i = 0; i < 50; i++) {
            var input = createTestPattern(0.7, 10);
            processor.process(input, params);
        }

        var stats = processor.getStatistics();

        assertTrue(stats.totalFastSteps() > 0, "Should have fast steps");
        assertTrue(stats.totalChunkingUpdates() > 0, "Should have chunking updates");

        // Verify ratios
        assertTrue(stats.actualChunkingToFastRatio() > 1.0,
                  "Should have more fast steps than chunking updates");
    }

    @Test
    @DisplayName("Processor maintains correct update counts")
    void testProcessorUpdateCounts() {
        var baseLayer = new MockLayer(10);
        var chunkingLayer = new TemporalChunkingLayerDecorator(
            baseLayer,
            ChunkingParameters.fastChunking()
        );

        var pathway = new MockPathway(10,
                                     createStandardShuntingParameters(),
                                     createStandardTransmitterParameters());

        var processor = MultiScaleLayerProcessor.standard(chunkingLayer, pathway);

        // Process exactly 100 patterns
        LayerParameters params = null;  // Use null for test
        for (int i = 0; i < 100; i++) {
            var input = createTestPattern(0.7, 10);
            processor.process(input, params);
        }

        assertEquals(100, processor.getFastStepCount(),
                    "Should have 100 fast steps");
        assertTrue(processor.getChunkingUpdateCount() < 100,
                  "Should have fewer chunking updates than fast steps");
        assertTrue(processor.getSlowUpdateCount() < processor.getChunkingUpdateCount(),
                  "Should have fewer slow updates than chunking updates");
    }

    @Test
    @DisplayName("Processor handles sequence processing")
    void testSequenceProcessing() {
        var baseLayer = new MockLayer(10);
        var chunkingLayer = new TemporalChunkingLayerDecorator(
            baseLayer,
            ChunkingParameters.paperDefaults()
        );

        var pathway = new MockPathway(10,
                                     createStandardShuntingParameters(),
                                     createStandardTransmitterParameters());

        var processor = MultiScaleLayerProcessor.standard(chunkingLayer, pathway);

        // Create sequence
        List<Pattern> sequence = new ArrayList<>();
        for (int i = 0; i < 20; i++) {
            sequence.add(createTestPattern(0.8, 10));
        }

        LayerParameters params = null;  // Use null for test
        var finalState = processor.processSequence(sequence, params);

        assertNotNull(finalState, "Should return final state");
        assertEquals(20, processor.getFastStepCount(), "Should process all patterns");
    }

    @Test
    @DisplayName("Processor reset clears all state")
    void testProcessorReset() {
        var baseLayer = new MockLayer(10);
        var chunkingLayer = new TemporalChunkingLayerDecorator(
            baseLayer,
            ChunkingParameters.fastChunking()
        );

        var pathway = new MockPathway(10,
                                     createStandardShuntingParameters(),
                                     createStandardTransmitterParameters());

        var processor = MultiScaleLayerProcessor.standard(chunkingLayer, pathway);

        // Process some patterns
        LayerParameters params = null;  // Use null for test
        for (int i = 0; i < 20; i++) {
            processor.process(createTestPattern(0.7, 10), params);
        }

        assertTrue(processor.getFastStepCount() > 0, "Should have counts");

        // Reset
        processor.reset();

        assertEquals(0, processor.getFastStepCount(), "Counts should be reset");
        assertEquals(0, processor.getChunkingUpdateCount(), "Counts should be reset");
        assertEquals(0, processor.getSlowUpdateCount(), "Counts should be reset");
    }

    // Helper methods

    private Pattern createTestPattern(double value, int size) {
        double[] values = new double[size];
        for (int i = 0; i < size; i++) {
            values[i] = value;
        }
        return new DenseVector(values);
    }

    // Mock implementations

    private static class MockLayer implements Layer {
        private final int size;
        private Pattern activation;

        MockLayer(int size) {
            this.size = size;
            this.activation = new DenseVector(new double[size]);
        }

        @Override public String getId() { return "mock"; }
        @Override public int size() { return size; }
        @Override public LayerType getType() { return LayerType.INPUT; }
        @Override public Pattern getActivation() { return activation; }
        @Override public void setActivation(Pattern activation) { this.activation = activation; }

        @Override
        public Pattern processBottomUp(Pattern input, LayerParameters parameters) {
            setActivation(input);
            return input;
        }

        @Override
        public Pattern processTopDown(Pattern expectation, LayerParameters parameters) {
            return expectation;
        }

        @Override
        public Pattern processLateral(Pattern lateral, LayerParameters parameters) {
            return lateral;
        }

        @Override public com.hellblazer.art.laminar.core.WeightMatrix getWeights() { return null; }
        @Override public void setWeights(com.hellblazer.art.laminar.core.WeightMatrix weights) {}
        @Override public void updateWeights(Pattern input, double learningRate) {}
        @Override public void reset() { activation = new DenseVector(new double[size]); }
        @Override public void addActivationListener(LayerActivationListener listener) {}
    }

    private static class MockPathway implements TemporallyIntegratedPathway {
        private boolean temporalEnabled = true;

        MockPathway(int size, ShuntingParameters shuntingParams, TransmitterParameters transmitterParams) {
        }

        @Override public com.hellblazer.art.temporal.core.ShuntingState getShuntingState() { return null; }
        @Override public com.hellblazer.art.temporal.core.TransmitterState getTransmitterState() { return null; }
        @Override public void updateDynamics(double timeStep) {}
        @Override public boolean hasReachedEquilibrium(double threshold) { return false; }
        @Override public void resetDynamics() {}
        @Override public TimeScale getTimeScale() { return TimeScale.FAST; }
        @Override public void setTemporalDynamicsEnabled(boolean enabled) { this.temporalEnabled = enabled; }
        @Override public boolean isTemporalDynamicsEnabled() { return temporalEnabled; }

        @Override public String getId() { return "mock-pathway"; }
        @Override public String getSourceLayerId() { return "source"; }
        @Override public String getTargetLayerId() { return "target"; }
        @Override public com.hellblazer.art.laminar.core.PathwayType getType() {
            return com.hellblazer.art.laminar.core.PathwayType.BOTTOM_UP;
        }
        @Override public Pattern propagate(Pattern signal, com.hellblazer.art.laminar.parameters.PathwayParameters parameters) {
            return signal;
        }
        @Override public double getGain() { return 1.0; }
        @Override public void setGain(double gain) {}
        @Override public boolean isAdaptive() { return false; }
        @Override public void updateWeights(Pattern input, Pattern output, double learningRate) {}
        @Override public void reset() {}
    }
}