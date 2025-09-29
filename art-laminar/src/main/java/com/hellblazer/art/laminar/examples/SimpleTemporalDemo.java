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
import com.hellblazer.art.laminar.builders.LaminarCircuitBuilder;
import com.hellblazer.art.laminar.impl.DefaultLaminarParameters;
import com.hellblazer.art.laminar.impl.DefaultLearningParameters;
import com.hellblazer.art.laminar.impl.LaminarCircuitImpl;

import java.util.*;

/**
 * Simple demonstration of temporal sequence processing with laminar circuits.
 *
 * This example shows how the shunting dynamics in laminar circuits
 * naturally provide temporal integration and sequence learning capabilities.
 */
public class SimpleTemporalDemo {

    private final LaminarCircuitImpl<DefaultLaminarParameters> circuit;
    private final DefaultLaminarParameters parameters;
    private final Map<String, Pattern> patterns = new HashMap<>();
    private final List<String> sequenceHistory = new ArrayList<>();

    public SimpleTemporalDemo() {
        // Create parameters for temporal processing
        this.parameters = DefaultLaminarParameters.builder()
            .withLearningParameters(new DefaultLearningParameters(0.5, 0.0, false, 0.0))
            .withVigilance(0.7)
            .build();

        // Create standard laminar circuit
        this.circuit = (LaminarCircuitImpl<DefaultLaminarParameters>)
            new LaminarCircuitBuilder<DefaultLaminarParameters>()
                .withParameters(parameters)
                .withInputLayer(50, true)
                .withFeatureLayer(50)
                .withCategoryLayer(20)
                .withStandardConnections()
                .build();
    }

    /**
     * Run the temporal processing demonstration.
     */
    public void runDemo() {
        System.out.println("=== Temporal Sequence Processing Demo ===\n");

        // Demo 1: Learn a simple repeating sequence
        learnRepeatingSequence();

        // Demo 2: Learn and recognize rhythm patterns
        learnRhythmPatterns();

        // Demo 3: Demonstrate temporal context effects
        demonstrateContextEffects();

        // Display results
        displayResults();
    }

    /**
     * Learn a repeating sequence A-B-C.
     */
    private void learnRepeatingSequence() {
        System.out.println("1. Learning Repeating Sequence (A-B-C)");
        System.out.println("--------------------------------------");

        var patternA = createPattern("A", 0.1);
        var patternB = createPattern("B", 0.5);
        var patternC = createPattern("C", 0.9);

        // Train the sequence multiple times
        for (int cycle = 0; cycle < 3; cycle++) {
            System.out.printf("Cycle %d: ", cycle + 1);

            processAndRecord(patternA, "A");
            System.out.print("A");

            processAndRecord(patternB, "B");
            System.out.print("-B");

            processAndRecord(patternC, "C");
            System.out.println("-C");
        }

        // Test recognition
        System.out.print("\nTesting recognition: ");
        testRecognition(patternA, "A");
        testRecognition(patternB, "B");
        testRecognition(patternC, "C");
        System.out.println("\n");
    }

    /**
     * Learn rhythm patterns (short-short-long).
     */
    private void learnRhythmPatterns() {
        System.out.println("2. Learning Rhythm Patterns");
        System.out.println("----------------------------");

        var shortBeat = createPattern("short", 0.3);
        var longBeat = createPattern("long", 0.8);

        // Train rhythm pattern: short-short-long
        for (int measure = 0; measure < 4; measure++) {
            System.out.printf("Measure %d: ", measure + 1);

            processAndRecord(shortBeat, "S");
            System.out.print("S");

            processAndRecord(shortBeat, "S");
            System.out.print("-S");

            processAndRecord(longBeat, "L");
            System.out.println("-L");
        }

        System.out.println();
    }

    /**
     * Demonstrate how context affects pattern recognition.
     */
    private void demonstrateContextEffects() {
        System.out.println("3. Temporal Context Effects");
        System.out.println("---------------------------");

        var contextA = createPattern("CtxA", 0.2);
        var contextB = createPattern("CtxB", 0.7);
        var targetPattern = createPattern("Target", 0.5);

        // Same pattern after different contexts
        System.out.println("Context A → Target:");
        processAndRecord(contextA, "ContextA");
        var resultA = circuit.learn(targetPattern, parameters);
        var categoryA = getCategoryIndex(resultA);
        System.out.println("  Target assigned to category: " + categoryA);

        System.out.println("\nContext B → Target:");
        processAndRecord(contextB, "ContextB");
        var resultB = circuit.learn(targetPattern, parameters);
        var categoryB = getCategoryIndex(resultB);
        System.out.println("  Target assigned to category: " + categoryB);

        // Due to temporal integration in shunting dynamics,
        // the same pattern may be assigned to different categories
        // based on temporal context
        if (!categoryA.equals(categoryB)) {
            System.out.println("\n  ✓ Context affected categorization!");
        } else {
            System.out.println("\n  × Same category (context effect not observed)");
        }
        System.out.println();
    }

    /**
     * Create a distinctive pattern.
     */
    private Pattern createPattern(String name, double centerValue) {
        if (patterns.containsKey(name)) {
            return patterns.get(name);
        }

        var random = new Random(name.hashCode());
        var values = new double[50];

        // Create pattern centered around centerValue
        for (int i = 0; i < values.length; i++) {
            if (random.nextDouble() < 0.3) {
                values[i] = centerValue + random.nextGaussian() * 0.1;
                values[i] = Math.max(0, Math.min(1, values[i]));
            }
        }

        var pattern = new DenseVector(values);
        patterns.put(name, pattern);
        return pattern;
    }

    /**
     * Process pattern and record in history.
     */
    private void processAndRecord(Pattern pattern, String label) {
        circuit.learn(pattern, parameters);
        sequenceHistory.add(label);
    }

    /**
     * Test pattern recognition.
     */
    private void testRecognition(Pattern pattern, String label) {
        var result = circuit.predict(pattern, parameters);
        var category = getCategoryIndex(result);
        System.out.printf("%s→%s ", label, category);
    }

    /**
     * Extract category index from result.
     */
    private String getCategoryIndex(ActivationResult result) {
        if (result instanceof ActivationResult.Success success) {
            return String.valueOf(success.categoryIndex());
        }
        return "?";
    }

    /**
     * Display final results.
     */
    private void displayResults() {
        System.out.println("=== Results Summary ===");
        System.out.println("Total categories learned: " + circuit.getCategoryCount());
        System.out.println("Sequence length processed: " + sequenceHistory.size());
        System.out.println("\nSequence history: " + String.join("-", sequenceHistory));

        // Calculate pattern statistics
        var patternCounts = new HashMap<String, Integer>();
        for (var label : sequenceHistory) {
            patternCounts.merge(label, 1, Integer::sum);
        }

        System.out.println("\nPattern frequencies:");
        patternCounts.forEach((label, count) ->
            System.out.printf("  %s: %d times\n", label, count));
    }

    /**
     * Main method to run the demonstration.
     */
    public static void main(String[] args) {
        var demo = new SimpleTemporalDemo();
        demo.runDemo();

        System.out.println("\n✓ Temporal processing demonstration complete!");
    }
}