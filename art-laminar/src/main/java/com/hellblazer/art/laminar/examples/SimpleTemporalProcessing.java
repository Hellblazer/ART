/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 *
 * This file is part of Java ART Neural Networks.
 *
 * Java ART Neural Networks is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Java ART Neural Networks is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with Java ART Neural Networks. If not, see <https://www.gnu.org/licenses/>.
 */
package com.hellblazer.art.laminar.examples;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.core.results.LaminarActivationResult;
import com.hellblazer.art.laminar.builders.LaminarCircuitBuilder;
import com.hellblazer.art.laminar.impl.DefaultLaminarParameters;
import com.hellblazer.art.laminar.impl.DefaultLearningParameters;
import com.hellblazer.art.laminar.performance.VectorizedLaminarCircuit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Example demonstrating temporal sequence processing with laminar circuits.
 *
 * Shows how laminar circuits with shunting dynamics naturally handle:
 * - Temporal sequences with memory effects
 * - Pattern transitions and sequence learning
 * - Temporal context integration
 * - Rhythm and timing patterns
 *
 * The shunting dynamics provide natural temporal integration through
 * the differential equation: dx/dt = -Ax + (B-x)E - (x+C)I
 */
public class SimpleTemporalProcessing {

    private static final Logger log = LoggerFactory.getLogger(SimpleTemporalProcessing.class);

    private final VectorizedLaminarCircuit<DefaultLaminarParameters> circuit;
    private final DefaultLaminarParameters parameters;
    private final int inputDimension;

    // Track sequence patterns
    private final Map<String, Pattern> namedPatterns = new HashMap<>();
    private final List<SequenceStep> sequenceHistory = new ArrayList<>();

    public SimpleTemporalProcessing(int inputDim) {
        this.inputDimension = inputDim;

        // Parameters with temporal characteristics
        this.parameters = DefaultLaminarParameters.builder()
            .withLearningParameters(new DefaultLearningParameters(
                0.4,    // Moderate learning rate for stability
                0.0,    // No baseline activation
                false,  // No fast commitment
                0.0     // No choice parameter
            ))
            .withVigilance(0.75)  // Medium vigilance for sequence generalization
            .build();

        // Create vectorized circuit
        this.circuit = new VectorizedLaminarCircuit<>(parameters);

        // Configure layers for temporal processing
        new LaminarCircuitBuilder<DefaultLaminarParameters>()
            .withParameters(parameters)
            .withInputLayer(inputDim, true)      // Complement coding for contrast
            .withFeatureLayer(inputDim * 2)      // Larger for temporal features
            .withCategoryLayer(50)                // Enough for sequence variations
            .withStandardConnections()
            .withVigilance(0.75)
            .build();

        log.info("Temporal processing circuit initialized with {} dimensions", inputDim);
    }

    /**
     * Main demonstration of temporal processing capabilities.
     */
    public void demonstrateTemporalProcessing() {
        log.info("=== Temporal Sequence Processing Demonstration ===\n");

        // 1. Simple repeating sequence
        demonstrateRepeatingSequence();

        // 2. Rhythm patterns
        demonstrateRhythmPatterns();

        // 3. Sequence prediction
        demonstrateSequenceLearning();

        // 4. Temporal context effects
        demonstrateTemporalContext();

        // Display statistics
        displayStatistics();
    }

    /**
     * Demonstrate learning of repeating sequences.
     */
    private void demonstrateRepeatingSequence() {
        log.info("--- 1. Repeating Sequence (A-B-C-A-B-C) ---");

        // Create distinct patterns
        var patternA = createNamedPattern("A");
        var patternB = createNamedPattern("B");
        var patternC = createNamedPattern("C");

        // Train on repeating sequence
        log.info("Training on sequence...");
        for (int cycle = 0; cycle < 3; cycle++) {
            processPattern(patternA, "A", cycle * 3);
            processPattern(patternB, "B", cycle * 3 + 1);
            processPattern(patternC, "C", cycle * 3 + 2);
        }

        // Test recognition
        log.info("Testing sequence recognition...");
        testPattern(patternA, "A");
        testPattern(patternB, "B");
        testPattern(patternC, "C");

        log.info("Categories learned: {}\n", circuit.getCategoryCount());
    }

    /**
     * Demonstrate rhythm pattern processing.
     */
    private void demonstrateRhythmPatterns() {
        log.info("--- 2. Rhythm Patterns ---");

        // Create rhythm: Short-Short-Long pattern
        var shortBeat = createRhythmPattern(0.3, "short");
        var longBeat = createRhythmPattern(0.8, "long");

        log.info("Training rhythm: short-short-long...");
        for (int measure = 0; measure < 4; measure++) {
            processPattern(shortBeat, "short", measure * 3);
            processPattern(shortBeat, "short", measure * 3 + 1);
            processPattern(longBeat, "long", measure * 3 + 2);
        }

        // Test rhythm recognition
        log.info("Testing rhythm recognition...");
        testPattern(shortBeat, "short beat");
        testPattern(longBeat, "long beat");

        // Test intermediate pattern
        var mediumBeat = createRhythmPattern(0.5, "medium");
        testPattern(mediumBeat, "medium beat (novel)");

        log.info("");
    }

    /**
     * Demonstrate sequence learning and completion.
     */
    private void demonstrateSequenceLearning() {
        log.info("--- 3. Sequence Learning ---");

        // Create a melody sequence
        var notes = new String[]{"Do", "Re", "Mi", "Fa", "Sol"};
        var melody = new ArrayList<Pattern>();

        log.info("Training melody sequence: Do-Re-Mi-Fa-Sol");
        for (int i = 0; i < notes.length; i++) {
            var pattern = createMelodyPattern(i, notes[i]);
            melody.add(pattern);
            processPattern(pattern, notes[i], i);
        }

        // Train multiple times for stronger learning
        log.info("Reinforcing sequence...");
        for (int trial = 1; trial < 3; trial++) {
            for (int i = 0; i < melody.size(); i++) {
                processPattern(melody.get(i), notes[i], trial * 5 + i);
            }
        }

        // Test partial sequence recognition
        log.info("Testing partial sequences...");
        testPattern(melody.get(0), "Do (start)");
        testPattern(melody.get(2), "Mi (middle)");
        testPattern(melody.get(4), "Sol (end)");

        log.info("");
    }

    /**
     * Demonstrate temporal context effects.
     */
    private void demonstrateTemporalContext() {
        log.info("--- 4. Temporal Context Effects ---");

        // Same pattern in different contexts
        var ambiguousPattern = createNamedPattern("X");

        // Context 1: After pattern A
        log.info("Context 1: A → X");
        processPattern(createNamedPattern("A"), "A", 100);
        processPattern(ambiguousPattern, "X after A", 101);

        // Context 2: After pattern B
        log.info("Context 2: B → X");
        processPattern(createNamedPattern("B"), "B", 102);
        processPattern(ambiguousPattern, "X after B", 103);

        // The circuit should learn different representations based on context
        log.info("Categories after contextual learning: {}\n", circuit.getCategoryCount());
    }

    /**
     * Process a pattern through the circuit.
     */
    private void processPattern(Pattern pattern, String label, int timeStep) {
        // Learn the pattern
        var result = circuit.learn(pattern, parameters);

        // Process for temporal dynamics
        var laminarResult = circuit.processCycle(pattern, parameters);

        // Record in history
        var category = extractCategory(result);
        sequenceHistory.add(new SequenceStep(timeStep, label, category, laminarResult.isResonant()));

        log.debug("t={}: {} → category {} (resonant: {})",
            timeStep, label, category, laminarResult.isResonant());
    }

    /**
     * Test pattern recognition without learning.
     */
    private void testPattern(Pattern pattern, String label) {
        var result = circuit.predict(pattern, parameters);
        var category = extractCategory(result);

        var laminarResult = circuit.processCycle(pattern, parameters);

        log.info("  Test '{}': category {} (resonant: {})",
            label, category, laminarResult.isResonant());
    }

    /**
     * Create a named pattern with consistent characteristics.
     */
    private Pattern createNamedPattern(String name) {
        if (namedPatterns.containsKey(name)) {
            return namedPatterns.get(name);
        }

        var random = new Random(name.hashCode());
        var values = new double[inputDimension];

        // Create distinctive pattern
        for (int i = 0; i < inputDimension; i++) {
            if (random.nextDouble() < 0.3) {  // Sparse pattern
                values[i] = 0.5 + random.nextGaussian() * 0.2;
                values[i] = Math.max(0, Math.min(1, values[i]));
            }
        }

        var pattern = new DenseVector(values);
        namedPatterns.put(name, pattern);
        return pattern;
    }

    /**
     * Create a rhythm pattern with specific intensity.
     */
    private Pattern createRhythmPattern(double intensity, String label) {
        var key = "rhythm_" + label;
        if (namedPatterns.containsKey(key)) {
            return namedPatterns.get(key);
        }

        var values = new double[inputDimension];

        // Create rhythm pattern with harmonics
        for (int i = 0; i < inputDimension; i++) {
            double freq = (i + 1) * 0.1;
            values[i] = intensity * Math.sin(freq * Math.PI);
            values[i] = Math.max(0, Math.abs(values[i]));
        }

        var pattern = new DenseVector(values);
        namedPatterns.put(key, pattern);
        return pattern;
    }

    /**
     * Create a melody pattern for a specific note.
     */
    private Pattern createMelodyPattern(int noteIndex, String noteName) {
        var key = "melody_" + noteName;
        if (namedPatterns.containsKey(key)) {
            return namedPatterns.get(key);
        }

        var values = new double[inputDimension];

        // Create frequency pattern for the note
        double baseFreq = 1.0 + noteIndex * 0.2;  // Increasing frequency
        for (int i = 0; i < inputDimension; i++) {
            // Primary frequency + harmonics
            values[i] = 0.5 * Math.sin(baseFreq * i * 0.1) +
                       0.3 * Math.sin(2 * baseFreq * i * 0.1) +
                       0.2 * Math.sin(3 * baseFreq * i * 0.1);
            values[i] = (values[i] + 1.0) / 2.0;  // Normalize to [0,1]
        }

        var pattern = new DenseVector(values);
        namedPatterns.put(key, pattern);
        return pattern;
    }

    /**
     * Extract category index from activation result.
     */
    private int extractCategory(ActivationResult result) {
        if (result instanceof ActivationResult.Success success) {
            return success.categoryIndex();
        } else if (result instanceof LaminarActivationResult laminar) {
            return laminar.getCategoryIndex();
        }
        return -1;
    }

    /**
     * Display processing statistics.
     */
    private void displayStatistics() {
        log.info("=== Temporal Processing Statistics ===");
        log.info("Total categories learned: {}", circuit.getCategoryCount());
        log.info("Unique patterns created: {}", namedPatterns.size());
        log.info("Sequence steps processed: {}", sequenceHistory.size());

        // Analyze resonance patterns
        long resonantCount = sequenceHistory.stream()
            .filter(step -> step.resonant)
            .count();
        double resonanceRate = (100.0 * resonantCount) / sequenceHistory.size();
        log.info("Resonance rate: {:.1f}%", resonanceRate);

        // Performance metrics
        var perfStats = circuit.getPerformanceStats();
        log.info("\nPerformance Metrics:");
        log.info("  Vector operations: {}", perfStats.totalVectorOperations());
        log.info("  Activation calls: {}", perfStats.activationCalls());
        log.info("  Learning calls: {}", perfStats.learningCalls());
        log.info("  Estimated speedup: {:.1f}x", perfStats.getEstimatedSpeedup());
        log.info("  Vector lane width: {} ({})",
            perfStats.vectorLaneWidth(),
            perfStats.vectorLaneWidth() == 4 ? "128-bit SIMD" : "Other");
    }

    /**
     * Record of a sequence processing step.
     */
    private record SequenceStep(
        int timeStep,
        String label,
        int category,
        boolean resonant
    ) {}

    /**
     * Main method to run the demonstration.
     */
    public static void main(String[] args) {
        // Create processor with 50-dimensional patterns
        var processor = new SimpleTemporalProcessing(50);

        // Run demonstration
        processor.demonstrateTemporalProcessing();

        // Cleanup
        processor.circuit.close();

        log.info("\nTemporal processing demonstration complete.");
    }
}