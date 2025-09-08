package com.art.textgen;

import com.art.textgen.core.WorkingMemory;
import com.art.textgen.dynamics.*;
import com.art.textgen.memory.*;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;
import java.util.stream.Collectors;

/**
 * JUnit test class for the ART-based text generation system
 */
public class TestTextGeneration {
    
    @Test
    @DisplayName("ART Text Generation System Integration Test")
    public void testARTTextGenerationSystem() {
        System.out.println("=== ART-Based Text Generation System Test ===\n");
        
        // Test individual components
        testWorkingMemory();
        testShuntingDynamics();
        testResonanceDetection();
        testAttentionalDynamics();
        testPredictiveOscillator();
        testIntegratedDynamics();
        
        // Test full text generation
        testFullGeneration();
    }
    
    private static void testWorkingMemory() {
        System.out.println("Testing Working Memory...");
        
        WorkingMemory<String> memory = new WorkingMemory<>(7, 1.0);
        
        // Add items
        String[] words = {"The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"};
        for (String word : words) {
            memory.addItem(word, 1.0);
            System.out.println("Added: " + word);
            System.out.println("Recent items: " + memory.getRecentItems(3));
            System.out.println("Activation gradient: " + 
                Arrays.toString(memory.getActivationGradient()));
        }
        
        System.out.println("\nFinal state:");
        System.out.println("Items in memory: " + memory.getRecentItems(10));
        System.out.println("Capacity used: " + !memory.hasCapacity());
        System.out.println();
    }
    
    private static void testShuntingDynamics() {
        System.out.println("Testing Shunting Dynamics...");
        
        ShuntingEquations shunting = new ShuntingEquations();
        ShuntingEquations.NeuralField field = new ShuntingEquations.NeuralField(10);
        
        // Set random input
        Random rand = new Random();
        double[] input = new double[10];
        for (int i = 0; i < input.length; i++) {
            input[i] = rand.nextDouble();
        }
        
        field.setExcitatory(input);
        System.out.println("Initial input: " + Arrays.toString(input));
        
        // Run dynamics
        for (int iter = 0; iter < 10; iter++) {
            shunting.updateField(field);
            shunting.applyLateralInhibition(field, 2.0);
        }
        
        System.out.println("After dynamics: " + Arrays.toString(field.getActivations()));
        System.out.println("Winner neuron: " + field.getWinner());
        System.out.println("Total activation: " + field.getTotalActivation());
        System.out.println();
    }
    
    private static void testResonanceDetection() {
        System.out.println("Testing Resonance Detection...");
        
        ResonanceDetector resonance = new ResonanceDetector(0.7);
        
        // Create test patterns
        double[][] patterns = {
            {1.0, 0.0, 0.0, 0.5, 0.0},
            {0.0, 1.0, 0.0, 0.5, 0.0},
            {1.0, 0.0, 0.0, 0.5, 0.0}, // Similar to first
            {0.0, 0.0, 1.0, 0.0, 0.5}
        };
        
        for (int i = 0; i < patterns.length; i++) {
            ResonanceDetector.ResonanceState state = resonance.searchResonance(patterns[i]);
            
            System.out.println("Pattern " + i + ": " + Arrays.toString(patterns[i]));
            System.out.println("  Resonant: " + state.isResonant);
            System.out.println("  Strength: " + state.resonanceStrength);
            System.out.println("  Category: " + 
                (state.resonantCategory != null ? state.resonantCategory.id : "new"));
            System.out.println("  Iterations: " + state.iterations);
        }
        
        System.out.println("\nTotal categories created: " + resonance.getCategories().size());
        System.out.println();
    }
    
    private static void testAttentionalDynamics() {
        System.out.println("Testing Attentional Dynamics...");
        
        AttentionalDynamics attention = new AttentionalDynamics();
        
        // Process sequence with shifting attention
        double[][] sequence = {
            {0.1, 0.1, 0.9, 0.1, 0.1},
            {0.1, 0.9, 0.1, 0.1, 0.1},
            {0.9, 0.1, 0.1, 0.1, 0.1},
            {0.1, 0.1, 0.1, 0.9, 0.1}
        };
        
        for (int t = 0; t < sequence.length; t++) {
            double[] output = attention.processWithAttention(sequence[t]);
            AttentionalDynamics.AttentionalState state = attention.getState();
            
            System.out.println("Time " + t + ":");
            System.out.println("  Input: " + Arrays.toString(sequence[t]));
            System.out.println("  Output: " + Arrays.toString(output));
            System.out.println("  Attention load: " + state.attentionalLoad);
            System.out.println("  Needs shift: " + state.needsShift);
        }
        
        System.out.println();
    }
    
    private static void testPredictiveOscillator() {
        System.out.println("Testing Predictive Oscillator...");
        
        PredictiveOscillator oscillator = new PredictiveOscillator(4);
        
        // Create rhythmic input
        double[] rhythmicInput = new double[50];
        for (int i = 0; i < rhythmicInput.length; i++) {
            rhythmicInput[i] = Math.sin(2 * Math.PI * i / 10.0); // Period of 10
        }
        
        // Entrain to rhythm
        oscillator.entrain(rhythmicInput, 0.2);
        
        // Generate prediction
        PredictiveOscillator.TemporalPrediction prediction = 
            oscillator.predict(rhythmicInput, 20);
        
        System.out.println("Input rhythm (first 10): " + 
            Arrays.toString(Arrays.copyOf(rhythmicInput, 10)));
        System.out.println("Predictions (next 10): " + 
            Arrays.toString(Arrays.copyOf(prediction.predictions, 10)));
        System.out.println("Confidence: " + prediction.confidence);
        System.out.println("Best oscillator: " + prediction.bestOscillator);
        
        System.out.println();
    }
    
    private static void testIntegratedDynamics() {
        System.out.println("Testing Integrated Dynamics...");
        
        IntegratedDynamics dynamics = new IntegratedDynamics();
        
        // Process sequence
        double[] input = {0.5, 0.8, 0.3, 0.9, 0.2, 0.7, 0.4, 0.6};
        
        IntegratedDynamics.DynamicsState state = dynamics.process(input);
        
        System.out.println("Input: " + Arrays.toString(input));
        System.out.println("Neural activations: " + 
            Arrays.toString(state.neuralActivations));
        System.out.println("Resonance achieved: " + state.resonanceState.isResonant);
        System.out.println("Attention load: " + state.attentionalState.attentionalLoad);
        System.out.println("Temporal confidence: " + state.temporalPrediction.confidence);
        System.out.println("Overall coherence: " + state.coherence);
        
        // Generate prediction
        double[] prediction = dynamics.predictNext(state, 5);
        System.out.println("Next prediction: " + Arrays.toString(prediction));
        
        // Get diagnostics
        Map<String, Object> diagnostics = dynamics.getDiagnostics();
        System.out.println("System diagnostics: " + diagnostics);
        
        dynamics.shutdown();
        System.out.println();
    }
    
    private static void testFullGeneration() {
        System.out.println("Testing Full Text Generation...");
        System.out.println("=" + "=".repeat(50));
        
        GrossbergTextGenerator generator = new GrossbergTextGenerator();
        
        // Test prompts
        String[] prompts = {
            "The future of artificial intelligence",
            "Once upon a time in a",
            "The key to understanding consciousness"
        };
        
        for (String prompt : prompts) {
            System.out.println("\nPrompt: \"" + prompt + "\"");
            System.out.println("Generated continuation:");
            
            // Generate text
            List<String> generated = generator.generate(prompt, 20)
                .collect(Collectors.toList());
            
            // Print generated text
            System.out.print("  " + prompt + " ");
            for (String token : generated) {
                System.out.print(token + " ");
            }
            System.out.println();
        }
        
        generator.shutdown();
        System.out.println("\n=== Test Complete ===");
    }
    
    /**
     * Helper method to demonstrate memory hierarchy
     */
    private static void demonstrateMemoryHierarchy() {
        System.out.println("\n=== Memory Hierarchy Demonstration ===");
        
        RecursiveHierarchicalMemory hierarchy = new RecursiveHierarchicalMemory(3);
        
        // Add many tokens to trigger compression
        String text = "The quick brown fox jumps over the lazy dog. " +
                     "This is a test of the hierarchical memory system. " +
                     "It should compress and maintain context over long sequences.";
        
        String[] tokens = text.split("\\s+");
        
        for (String token : tokens) {
            hierarchy.addToken(token);
        }
        
        // Get active context
        List<Object> context = hierarchy.getActiveContext(50);
        System.out.println("Active context (50 items max):");
        for (Object item : context) {
            System.out.println("  " + item);
        }
        
        System.out.println("Effective capacity: " + hierarchy.getEffectiveCapacity());
    }
    
    /**
     * Helper method to demonstrate multi-timescale processing
     */
    private static void demonstrateMultiTimescale() {
        System.out.println("\n=== Multi-Timescale Processing ===");
        
        MultiTimescaleMemoryBank bank = new MultiTimescaleMemoryBank();
        
        // Process tokens at different rates
        String[] tokens = "The quick brown fox jumps".split("\\s+");
        
        for (String token : tokens) {
            bank.update(token);
            
            Map<MultiTimescaleMemoryBank.TimeScale, 
                MultiTimescaleMemoryBank.Prediction> predictions = 
                bank.generatePredictions();
            
            System.out.println("After token: " + token);
            for (Map.Entry<MultiTimescaleMemoryBank.TimeScale, 
                            MultiTimescaleMemoryBank.Prediction> entry : 
                 predictions.entrySet()) {
                System.out.println("  " + entry.getKey() + ": " + 
                    (entry.getValue().content != null ? entry.getValue().content : "none") +
                    " (weight: " + entry.getValue().weight + ")");
            }
        }
    }
}
