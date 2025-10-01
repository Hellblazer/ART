package com.hellblazer.art.laminar.examples;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.integration.ARTCircuitParameters;
import com.hellblazer.art.laminar.integration.ARTLaminarCircuit;
import com.hellblazer.art.laminar.temporal.*;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Demonstration: Temporal Chunking with Current Implementation
 *
 * Shows how the ART laminar circuit chunks SIMILAR temporal patterns.
 * Current implementation uses cosine similarity - chunks form when consecutive
 * patterns are similar (high coherence).
 *
 * Real-world examples:
 * - Repeating sensor readings (stable signal periods)
 * - Similar activity patterns (walking, running bursts)
 * - Clustered events (multiple requests from same source)
 * - Pattern repetition (rhythm, periodic signals)
 *
 * @author Hal Hildebrand
 */
public class SequenceLearningDemo {

    /**
     * Demo 1: Basic Temporal Chunking with Repeated Patterns
     *
     * Shows how similar consecutive patterns form chunks.
     */
    @Test
    void demo1_RepeatedPatternChunking() {
        System.out.println("\n=== DEMO 1: Repeated Pattern Chunking ===\n");

        var params = ARTCircuitParameters.builder(10)
            .vigilance(0.85)
            .learningRate(0.7)
            .maxCategories(50)
            .build();

        try (var circuit = new ARTLaminarCircuit(params)) {
            // Use low coherence threshold for easier chunk formation
            var chunkingParams = ChunkingParameters.builder()
                .maxHistorySize(10)
                .chunkFormationThreshold(0.3)  // Low threshold - easy to activate
                .chunkCoherenceThreshold(0.3)  // Low threshold - similar patterns chunk
                .chunkDecayRate(0.05)
                .minChunkSize(2)
                .maxChunkSize(5)
                .build();

            var chunkingLayer = new TemporalChunkingLayerDecorator(
                circuit.getLayer23(), chunkingParams);

            // Sequence: AAA BBB CCC (repeated patterns)
            System.out.println("Processing sequence: A A A  B B B  C C C");
            System.out.println("(Repeated patterns should form chunks)\n");

            var patternA = createPattern(new double[]{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
            var patternB = createPattern(new double[]{0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
            var patternC = createPattern(new double[]{0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

            // Process AAA
            for (int i = 0; i < 3; i++) {
                chunkingLayer.processWithChunking(patternA, 0.1);
                circuit.process(patternA);
            }

            // Process BBB
            for (int i = 0; i < 3; i++) {
                chunkingLayer.processWithChunking(patternB, 0.1);
                circuit.process(patternB);
            }

            // Process CCC
            for (int i = 0; i < 3; i++) {
                chunkingLayer.processWithChunking(patternC, 0.1);
                circuit.process(patternC);
            }

            var statistics = chunkingLayer.getChunkingStatistics();
            System.out.println("Chunk Formation Statistics:");
            System.out.printf("  Total chunks formed: %d%n", statistics.totalChunks());
            System.out.printf("  Average chunk size: %.1f items%n", statistics.averageChunkSize());
            System.out.printf("  Average coherence: %.3f%n", statistics.averageCoherence());

            System.out.println("\nExpected Behavior:");
            System.out.println("- AAA → 1 chunk (3 identical patterns, coherence = 1.0)");
            System.out.println("- BBB → 1 chunk (3 identical patterns, coherence = 1.0)");
            System.out.println("- CCC → 1 chunk (3 identical patterns, coherence = 1.0)");
            System.out.println("- Total: 3 chunks with high coherence");

            assertTrue(statistics.totalChunks() >= 1,
                "Should form at least 1 chunk from repeated patterns");
        } catch (Exception e) {
            fail("Demo failed: " + e.getMessage());
        }
    }

    /**
     * Demo 2: Sensor Reading Chunking (Stable Periods)
     *
     * Real-world: Temperature sensor with stable periods and changes.
     */
    @Test
    void demo2_SensorReadingChunking() {
        System.out.println("\n=== DEMO 2: Sensor Reading Chunking ===\n");

        var params = ARTCircuitParameters.builder(5)
            .vigilance(0.85)
            .learningRate(0.7)
            .maxCategories(50)
            .build();

        try (var circuit = new ARTLaminarCircuit(params)) {
            var chunkingParams = ChunkingParameters.builder()
                .maxHistorySize(15)
                .chunkFormationThreshold(0.3)
                .chunkCoherenceThreshold(0.4)  // Allow some variation
                .chunkDecayRate(0.05)
                .minChunkSize(3)
                .maxChunkSize(7)
                .build();

            var chunkingLayer = new TemporalChunkingLayerDecorator(
                circuit.getLayer23(), chunkingParams);

            System.out.println("Simulating temperature sensor readings:");
            System.out.println("Period 1: Stable at 20°C (5 readings)");
            System.out.println("Period 2: Stable at 25°C (5 readings)");
            System.out.println("Period 3: Stable at 20°C (5 readings)\n");

            var random = new Random(42);

            // Stable period 1: ~20°C with minor noise
            for (int i = 0; i < 5; i++) {
                var reading = createSensorReading(20.0 + random.nextGaussian() * 0.5);
                chunkingLayer.processWithChunking(reading, 0.1);
                circuit.process(reading);
            }

            // Stable period 2: ~25°C with minor noise
            for (int i = 0; i < 5; i++) {
                var reading = createSensorReading(25.0 + random.nextGaussian() * 0.5);
                chunkingLayer.processWithChunking(reading, 0.1);
                circuit.process(reading);
            }

            // Stable period 3: back to ~20°C
            for (int i = 0; i < 5; i++) {
                var reading = createSensorReading(20.0 + random.nextGaussian() * 0.5);
                chunkingLayer.processWithChunking(reading, 0.1);
                circuit.process(reading);
            }

            var stats = chunkingLayer.getChunkingStatistics();
            System.out.println("Chunking Results:");
            System.out.printf("  Total chunks: %d%n", stats.totalChunks());
            System.out.printf("  Average chunk size: %.1f readings%n", stats.averageChunkSize());
            System.out.printf("  Average coherence: %.3f%n", stats.averageCoherence());

            System.out.println("\nExpected: 2-3 chunks for stable temperature periods");
            System.out.println("High coherence within each stable period (similar readings)");

            assertTrue(stats.totalChunks() >= 1,
                "Should form chunks during stable periods");
        } catch (Exception e) {
            fail("Demo failed: " + e.getMessage());
        }
    }

    /**
     * Demo 3: Activity Burst Detection
     *
     * Real-world: Network traffic with burst patterns.
     */
    @Test
    void demo3_ActivityBurstDetection() {
        System.out.println("\n=== DEMO 3: Activity Burst Detection ===\n");

        var params = ARTCircuitParameters.builder(8)
            .vigilance(0.85)
            .learningRate(0.7)
            .maxCategories(50)
            .build();

        try (var circuit = new ARTLaminarCircuit(params)) {
            var chunkingParams = ChunkingParameters.fastChunking();  // Pre-configured

            var chunkingLayer = new TemporalChunkingLayerDecorator(
                circuit.getLayer23(), chunkingParams);

            System.out.println("Simulating network traffic:");
            System.out.println("Burst 1: High activity (4 similar high-load patterns)");
            System.out.println("Quiet period: Low activity (3 low-load patterns)");
            System.out.println("Burst 2: High activity (4 similar high-load patterns)\n");

            // Burst 1: High activity patterns
            for (int i = 0; i < 4; i++) {
                var highLoad = createTrafficPattern(0.8, 0.7, 0.9);  // Similar high values
                chunkingLayer.processWithChunking(highLoad, 0.1);
                circuit.process(highLoad);
            }

            // Quiet period: Low activity
            for (int i = 0; i < 3; i++) {
                var lowLoad = createTrafficPattern(0.2, 0.1, 0.3);  // Similar low values
                chunkingLayer.processWithChunking(lowLoad, 0.1);
                circuit.process(lowLoad);
            }

            // Burst 2: High activity again
            for (int i = 0; i < 4; i++) {
                var highLoad = createTrafficPattern(0.85, 0.75, 0.95);
                chunkingLayer.processWithChunking(highLoad, 0.1);
                circuit.process(highLoad);
            }

            var stats = chunkingLayer.getChunkingStatistics();
            System.out.println("Burst Detection Results:");
            System.out.printf("  Total chunks: %d%n", stats.totalChunks());
            System.out.printf("  Average chunk size: %.1f events%n", stats.averageChunkSize());
            System.out.printf("  Average coherence: %.3f%n", stats.averageCoherence());

            System.out.println("\nExpected: 2-3 chunks");
            System.out.println("- Chunk for high-activity burst");
            System.out.println("- Chunk for quiet period");
            System.out.println("- Possibly another chunk for second burst");

            assertTrue(stats.totalChunks() >= 1,
                "Should detect activity bursts as chunks");
        } catch (Exception e) {
            fail("Demo failed: " + e.getMessage());
        }
    }

    /**
     * Demo 4: Pattern Repetition and Rhythms
     *
     * Shows chunking of rhythmic/periodic patterns.
     */
    @Test
    void demo4_RhythmicPatternChunking() {
        System.out.println("\n=== DEMO 4: Rhythmic Pattern Chunking ===\n");

        var params = ARTCircuitParameters.builder(6)
            .vigilance(0.85)
            .learningRate(0.7)
            .maxCategories(50)
            .build();

        try (var circuit = new ARTLaminarCircuit(params)) {
            var chunkingParams = ChunkingParameters.builder()
                .maxHistorySize(12)
                .chunkFormationThreshold(0.3)
                .chunkCoherenceThreshold(0.5)  // Medium coherence
                .chunkDecayRate(0.05)
                .minChunkSize(2)
                .maxChunkSize(6)
                .build();

            var chunkingLayer = new TemporalChunkingLayerDecorator(
                circuit.getLayer23(), chunkingParams);

            System.out.println("Processing rhythmic pattern: X X X Y Y Y X X X");
            System.out.println("(Repeating groups should form chunks)\n");

            var patternX = createPattern(new double[]{1.0, 0.0, 0.0, 0.0, 0.0, 0.0});
            var patternY = createPattern(new double[]{0.0, 0.0, 0.0, 1.0, 0.0, 0.0});

            // Group 1: XXX
            for (int i = 0; i < 3; i++) {
                chunkingLayer.processWithChunking(patternX, 0.1);
                circuit.process(patternX);
            }

            // Group 2: YYY
            for (int i = 0; i < 3; i++) {
                chunkingLayer.processWithChunking(patternY, 0.1);
                circuit.process(patternY);
            }

            // Group 3: XXX (repeat)
            for (int i = 0; i < 3; i++) {
                chunkingLayer.processWithChunking(patternX, 0.1);
                circuit.process(patternX);
            }

            var stats = chunkingLayer.getChunkingStatistics();
            var chunks = chunkingLayer.getTemporalChunks();

            System.out.println("Rhythm Chunking Results:");
            System.out.printf("  Total chunks: %d%n", stats.totalChunks());
            System.out.printf("  Average chunk size: %.1f patterns%n", stats.averageChunkSize());
            System.out.printf("  Average coherence: %.3f%n", stats.averageCoherence());

            System.out.println("\nActive Chunks:");
            for (int i = 0; i < chunks.size(); i++) {
                var chunk = chunks.get(i);
                System.out.printf("  Chunk %d: size=%d, coherence=%.3f, strength=%.3f%n",
                    i, chunk.size(), chunk.getCoherence(), chunk.getStrength());
            }

            System.out.println("\nExpected: Chunks for XXX and YYY groups");
            System.out.println("Perfect coherence (1.0) within each group");

            assertTrue(stats.totalChunks() >= 1,
                "Should form chunks from rhythmic patterns");
        } catch (Exception e) {
            fail("Demo failed: " + e.getMessage());
        }
    }

    /**
     * Demo 5: Working Memory Capacity
     *
     * Shows how chunk history is maintained and decays.
     */
    @Test
    void demo5_WorkingMemoryCapacity() {
        System.out.println("\n=== DEMO 5: Working Memory Capacity ===\n");

        var params = ARTCircuitParameters.builder(8)
            .vigilance(0.85)
            .learningRate(0.7)
            .maxCategories(50)
            .build();

        try (var circuit = new ARTLaminarCircuit(params)) {
            var chunkingParams = ChunkingParameters.builder()
                .maxHistorySize(7)  // Miller's 7±2
                .chunkFormationThreshold(0.3)
                .chunkCoherenceThreshold(0.4)
                .chunkDecayRate(0.1)  // Faster decay
                .minChunkSize(2)
                .maxChunkSize(5)
                .build();

            var chunkingLayer = new TemporalChunkingLayerDecorator(
                circuit.getLayer23(), chunkingParams);

            System.out.println("Testing working memory capacity:");
            System.out.println("Creating 3 chunks of similar patterns");
            System.out.println("Observing chunk decay over time\n");

            var random = new Random(42);

            // Create 3 distinct groups of similar patterns
            for (int group = 0; group < 3; group++) {
                System.out.printf("Group %d: ", group + 1);
                double baseValue = 0.3 * (group + 1);  // 0.3, 0.6, 0.9

                for (int i = 0; i < 3; i++) {
                    var pattern = createClusteredPattern(baseValue, 0.1, random);
                    chunkingLayer.processWithChunking(pattern, 0.1);
                    circuit.process(pattern);
                    System.out.print(".");
                }
                System.out.println();
            }

            System.out.println("\nInitial State:");
            var initialStats = chunkingLayer.getChunkingStatistics();
            System.out.printf("  Chunks formed: %d%n", initialStats.totalChunks());
            System.out.printf("  Average coherence: %.3f%n", initialStats.averageCoherence());

            // Check active chunks
            var activeChunks = chunkingLayer.getTemporalChunks();
            System.out.printf("  Active chunks in memory: %d%n", activeChunks.size());

            System.out.println("\nWorking Memory Properties:");
            System.out.println("- Capacity: 7±2 items (Miller's law)");
            System.out.println("- Chunks decay with decay rate = 0.1");
            System.out.println("- Active chunks represent recent temporal context");

            assertTrue(initialStats.totalChunks() >= 1,
                "Should form chunks within working memory capacity");
        } catch (Exception e) {
            fail("Demo failed: " + e.getMessage());
        }
    }

    // ========== Helper Methods ==========

    private Pattern createPattern(double[] values) {
        return new DenseVector(values);
    }

    private Pattern createSensorReading(double temperature) {
        // Normalize temperature to [0,1] range (assume 0-50°C)
        double normalized = Math.max(0.0, Math.min(1.0, temperature / 50.0));
        var data = new double[5];
        data[0] = normalized;
        data[1] = normalized * normalized;  // Non-linear component
        data[2] = Math.sin(normalized * Math.PI);  // Periodic component
        data[3] = normalized;
        data[4] = 1.0 - normalized;  // Complement
        return new DenseVector(data);
    }

    private Pattern createTrafficPattern(double load1, double load2, double load3) {
        var data = new double[8];
        data[0] = load1;
        data[1] = load2;
        data[2] = load3;
        data[3] = (load1 + load2 + load3) / 3.0;  // Average
        data[4] = Math.max(load1, Math.max(load2, load3));  // Peak
        data[5] = Math.min(load1, Math.min(load2, load3));  // Min
        data[6] = load1 * load2;  // Correlation
        data[7] = 1.0 - data[3];  // Complement
        return new DenseVector(data);
    }

    private Pattern createClusteredPattern(double center, double spread, Random random) {
        var data = new double[8];
        for (int i = 0; i < 8; i++) {
            data[i] = Math.max(0.0, Math.min(1.0,
                center + random.nextGaussian() * spread));
        }
        return new DenseVector(data);
    }
}
