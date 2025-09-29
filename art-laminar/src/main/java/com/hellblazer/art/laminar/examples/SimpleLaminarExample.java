package com.hellblazer.art.laminar.examples;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.laminar.builders.LaminarCircuitBuilder;
import com.hellblazer.art.laminar.core.LaminarCircuit;
import com.hellblazer.art.laminar.impl.DefaultLaminarParameters;
import com.hellblazer.art.laminar.impl.DefaultLearningParameters;
import com.hellblazer.art.laminar.events.*;
import com.hellblazer.art.core.results.LaminarActivationResult;

import java.util.Random;

/**
 * Simple example demonstrating basic laminar ART circuit usage.
 * Shows pattern learning, prediction, and resonance dynamics.
 *
 * @author Hal Hildebrand
 */
public class SimpleLaminarExample {

    public static void main(String[] args) {
        System.out.println("=== Laminar ART Circuit Example ===\n");

        // Create a simple circuit for 4-dimensional patterns
        var inputDim = 4;
        System.out.println("Creating laminar circuit for " + inputDim + "-dimensional patterns");

        // Build circuit with basic configuration
        var circuit = createBasicCircuit(inputDim);

        // Create some example patterns
        System.out.println("\nCreating training patterns...");
        var patterns = createExamplePatterns();

        // Train the circuit
        System.out.println("\nTraining circuit with patterns:");
        trainCircuit(circuit, patterns);

        // Test prediction
        System.out.println("\nTesting prediction with known patterns:");
        testPrediction(circuit, patterns);

        // Test with novel pattern
        System.out.println("\nTesting with novel pattern:");
        testNovelPattern(circuit);

        System.out.println("\n=== Example Complete ===");
    }

    /**
     * Create a basic laminar circuit.
     */
    private static LaminarCircuit<DefaultLaminarParameters> createBasicCircuit(int inputDim) {
        var builder = new LaminarCircuitBuilder<DefaultLaminarParameters>();

        // Configure layers
        builder.withInputLayer(inputDim, false)  // No complement coding for simplicity
               .withFeatureLayer(inputDim)
               .withCategoryLayer(10);            // Max 10 categories

        // Set up connections
        builder.withStandardConnections();

        // Configure resonance
        builder.withVigilance(0.8);

        // Set learning parameters
        var learningParams = new DefaultLearningParameters(
            0.5,    // learning rate
            0.8,    // momentum
            false,  // not fast learning
            0.01    // weight decay
        );

        builder.withParameters(DefaultLaminarParameters.builder()
                .withLearningParameters(learningParams)
                .build());

        // Add event listener for monitoring
        builder.withListener(new CircuitMonitor());

        return builder.build();
    }

    /**
     * Create example patterns for training.
     */
    private static Pattern[] createExamplePatterns() {
        return new Pattern[] {
            // Pattern 1: High first component
            new DenseVector(new double[]{0.9, 0.1, 0.1, 0.2}),

            // Pattern 2: High second component
            new DenseVector(new double[]{0.1, 0.9, 0.2, 0.1}),

            // Pattern 3: High third component
            new DenseVector(new double[]{0.2, 0.1, 0.9, 0.1}),

            // Pattern 4: Mixed pattern
            new DenseVector(new double[]{0.5, 0.5, 0.3, 0.4})
        };
    }

    /**
     * Train the circuit with patterns.
     */
    private static void trainCircuit(LaminarCircuit<DefaultLaminarParameters> circuit,
                                    Pattern[] patterns) {
        var params = DefaultLaminarParameters.builder()
                .withLearningParameters(new DefaultLearningParameters(0.5, 0.8, false, 0.01))
                .build();

        for (int i = 0; i < patterns.length; i++) {
            System.out.println("\n  Training pattern " + (i + 1) + ": " + patternToString(patterns[i]));

            var result = circuit.learn(patterns[i], params);

            if (result instanceof LaminarActivationResult laminar) {
                if (laminar.isResonant()) {
                    System.out.println("    → Learned as category " + laminar.getCategoryIndex() +
                                     " (resonance: " + String.format("%.3f", laminar.getResonanceScore()) + ")");
                } else {
                    System.out.println("    → Failed to achieve resonance");
                }
            }
        }

        System.out.println("\n  Total categories created: " + circuit.getCategoryCount());
    }

    /**
     * Test prediction with known patterns.
     */
    private static void testPrediction(LaminarCircuit<DefaultLaminarParameters> circuit,
                                      Pattern[] patterns) {
        var params = DefaultLaminarParameters.builder().build();

        // Test with first pattern (should recognize)
        var testPattern = patterns[0];
        System.out.println("\n  Testing with known pattern: " + patternToString(testPattern));

        var result = circuit.predict(testPattern, params);

        if (result instanceof LaminarActivationResult laminar && laminar.isResonant()) {
            System.out.println("    → Recognized as category " + laminar.getCategoryIndex() +
                             " (resonance: " + String.format("%.3f", laminar.getResonanceScore()) + ")");
        } else {
            System.out.println("    → Pattern not recognized");
        }
    }

    /**
     * Test with a novel pattern.
     */
    private static void testNovelPattern(LaminarCircuit<DefaultLaminarParameters> circuit) {
        var params = DefaultLaminarParameters.builder().build();

        // Create a novel pattern
        var novelPattern = new DenseVector(new double[]{0.3, 0.7, 0.5, 0.8});
        System.out.println("\n  Testing novel pattern: " + patternToString(novelPattern));

        var result = circuit.predict(novelPattern, params);

        if (result instanceof LaminarActivationResult laminar) {
            if (laminar.isResonant()) {
                System.out.println("    → Matched to category " + laminar.getCategoryIndex() +
                                 " (resonance: " + String.format("%.3f", laminar.getResonanceScore()) + ")");
            } else {
                System.out.println("    → No match found (would create new category if learning)");
            }
        }
    }

    /**
     * Convert pattern to string representation.
     */
    private static String patternToString(Pattern pattern) {
        var sb = new StringBuilder("[");
        for (int i = 0; i < pattern.dimension(); i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.2f", pattern.get(i)));
        }
        sb.append("]");
        return sb.toString();
    }

    /**
     * Event listener for monitoring circuit behavior.
     */
    static class CircuitMonitor implements CircuitEventListener {
        @Override
        public void onResonance(ResonanceEvent event) {
            System.out.println("      [Event] Resonance achieved - Category: " + event.getCategoryIndex() +
                             ", Score: " + String.format("%.3f", event.getMatchScore()));
        }

        @Override
        public void onReset(ResetEvent event) {
            System.out.println("      [Event] Reset - Category: " + event.getCategoryIndex() +
                             ", Reason: " + event.getReason());
        }

        @Override
        public void onCycleComplete(CycleEvent event) {
            // Silent for cleaner output
        }

        @Override
        public void onCategoryCreated(CategoryEvent event) {
            System.out.println("      [Event] New category created - Index: " + event.getCategoryIndex());
        }

        @Override
        public void onAttentionShift(AttentionEvent event) {
            // Silent for cleaner output
        }
    }
}