package com.art.textgen.dynamics;

import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Integrated Neural Dynamics System
 * Combines all Grossberg dynamics for coherent text generation
 */
public class IntegratedDynamics {
    
    private final ShuntingEquations shunting;
    private final ResonanceDetector resonance;
    private final AttentionalDynamics attention;
    private final PredictiveOscillator oscillator;
    private final ExecutorService executor;
    
    public static class DynamicsState {
        public final double[] neuralActivations;
        public final ResonanceDetector.ResonanceState resonanceState;
        public final AttentionalDynamics.AttentionalState attentionalState;
        public final PredictiveOscillator.TemporalPrediction temporalPrediction;
        public final double coherence;
        public final long timestamp;
        
        public DynamicsState(double[] activations, 
                            ResonanceDetector.ResonanceState resonance,
                            AttentionalDynamics.AttentionalState attention,
                            PredictiveOscillator.TemporalPrediction temporal,
                            double coherence) {
            this.neuralActivations = activations.clone();
            this.resonanceState = resonance;
            this.attentionalState = attention;
            this.temporalPrediction = temporal;
            this.coherence = coherence;
            this.timestamp = System.currentTimeMillis();
        }
    }
    
    public IntegratedDynamics() {
        this.shunting = new ShuntingEquations();
        this.resonance = new ResonanceDetector();
        this.attention = new AttentionalDynamics();
        this.oscillator = new PredictiveOscillator();
        this.executor = Executors.newFixedThreadPool(4);
    }
    
    /**
     * Process input through all dynamics systems
     */
    public DynamicsState process(double[] input) {
        // Process in parallel where possible
        CompletableFuture<double[]> attentionFuture = CompletableFuture.supplyAsync(
            () -> attention.processWithAttention(input), executor);
        
        CompletableFuture<ResonanceDetector.ResonanceState> resonanceFuture = 
            CompletableFuture.supplyAsync(
                () -> resonance.searchResonance(input), executor);
        
        CompletableFuture<PredictiveOscillator.TemporalPrediction> temporalFuture = 
            CompletableFuture.supplyAsync(
                () -> oscillator.predict(input, 50), executor);
        
        // Wait for all to complete
        CompletableFuture.allOf(attentionFuture, resonanceFuture, temporalFuture).join();
        
        try {
            // Get results
            double[] attendedInput = attentionFuture.get();
            ResonanceDetector.ResonanceState resonanceState = resonanceFuture.get();
            PredictiveOscillator.TemporalPrediction temporalPred = temporalFuture.get();
            
            // Process through shunting dynamics with resonance modulation
            double[] activations = processWithResonance(attendedInput, resonanceState);
            
            // Apply temporal modulation
            activations = applyTemporalModulation(activations, temporalPred);
            
            // Update oscillator based on current state
            oscillator.update(0.01);
            
            // Compute overall coherence
            double coherence = computeCoherence(resonanceState, 
                                               attention.getState(), 
                                               temporalPred);
            
            return new DynamicsState(
                activations,
                resonanceState,
                attention.getState(),
                temporalPred,
                coherence
            );
            
        } catch (Exception e) {
            throw new RuntimeException("Failed to process dynamics", e);
        }
    }
    
    /**
     * Process with resonance modulation
     */
    private double[] processWithResonance(double[] input, 
                                         ResonanceDetector.ResonanceState resonanceState) {
        ShuntingEquations.NeuralField field = 
            new ShuntingEquations.NeuralField(input.length);
        
        // Set excitatory input
        field.setExcitatory(input);
        
        // If resonant, enhance with top-down expectations
        if (resonanceState.isResonant && resonanceState.resonantCategory != null) {
            double[] topDown = resonanceState.resonantCategory.topDownWeights;
            double[] modulated = new double[input.length];
            
            for (int i = 0; i < input.length; i++) {
                modulated[i] = input[i] * (1.0 + topDown[i] * resonanceState.resonanceStrength);
            }
            
            field.setExcitatory(modulated);
        }
        
        // Update dynamics with damping for stability
        for (int iter = 0; iter < 20; iter++) {
            shunting.updateField(field, 0.95); // Slightly higher damping for integrated dynamics
            
            // Apply lateral inhibition with reduced strength
            shunting.applyLateralInhibition(field, 1.5);
            
            // Check for winner-take-all if highly resonant
            if (resonanceState.resonanceStrength > 0.9) {
                shunting.winnerTakeAll(field, 0.5);
            }
        }
        
        return field.getActivations();
    }
    
    /**
     * Apply temporal modulation from oscillators
     */
    private double[] applyTemporalModulation(double[] activations, 
                                            PredictiveOscillator.TemporalPrediction temporal) {
        double[] modulated = new double[activations.length];
        
        // Get current oscillator pattern
        double[] rhythm = oscillator.getCurrentPattern(activations.length);
        
        for (int i = 0; i < activations.length; i++) {
            // Combine activation with rhythmic modulation
            double rhythmicGain = 1.0 + 0.3 * rhythm[i] * temporal.confidence;
            modulated[i] = activations[i] * rhythmicGain;
            
            // Add predictive component
            if (i < temporal.predictions.length) {
                modulated[i] += temporal.predictions[i] * 0.2 * temporal.confidence;
            }
            
            // Bound values
            modulated[i] = Math.max(0, Math.min(1, modulated[i]));
        }
        
        return modulated;
    }
    
    /**
     * Compute overall system coherence
     */
    private double computeCoherence(ResonanceDetector.ResonanceState resonance,
                                   AttentionalDynamics.AttentionalState attention,
                                   PredictiveOscillator.TemporalPrediction temporal) {
        double resonanceCoherence = resonance.isResonant ? resonance.resonanceStrength : 0.5;
        double attentionalCoherence = 1.0 - attention.attentionalLoad / 10.0; // Normalize
        double temporalCoherence = temporal.confidence;
        
        // Weighted average
        return 0.4 * resonanceCoherence + 
               0.3 * attentionalCoherence + 
               0.3 * temporalCoherence;
    }
    
    /**
     * Adapt system based on feedback
     */
    public void adaptToFeedback(double[] target, double[] actual) {
        // Compute error
        double[] error = new double[target.length];
        double totalError = 0.0;
        
        for (int i = 0; i < target.length; i++) {
            error[i] = target[i] - actual[i];
            totalError += error[i] * error[i];
        }
        
        totalError = Math.sqrt(totalError);
        
        // Adjust vigilance based on error
        if (totalError > 0.5) {
            // High error - increase vigilance
            resonance.setVigilance(Math.min(1.0, 
                resonance.getCategories().size() > 0 ? 0.8 : 0.7));
        } else {
            // Low error - decrease vigilance for broader matching
            resonance.setVigilance(Math.max(0.5, 0.6));
        }
        
        // Adjust attention based on error distribution
        int maxErrorIndex = 0;
        double maxError = Math.abs(error[0]);
        
        for (int i = 1; i < error.length; i++) {
            if (Math.abs(error[i]) > maxError) {
                maxError = Math.abs(error[i]);
                maxErrorIndex = i;
            }
        }
        
        // Create attentional bias towards high error region
        attention.createBias("error_region_" + maxErrorIndex, maxError);
        
        // Entrain oscillators if there's rhythmic structure
        oscillator.entrain(target, 0.1);
    }
    
    /**
     * Generate prediction for next token
     */
    public double[] predictNext(DynamicsState currentState, int steps) {
        double[] prediction = new double[currentState.neuralActivations.length];
        
        // Start from current activations
        System.arraycopy(currentState.neuralActivations, 0, prediction, 0, prediction.length);
        
        // Evolve forward
        for (int step = 0; step < steps; step++) {
            // Apply dynamics without input (autonomous evolution)
            ShuntingEquations.NeuralField field = 
                new ShuntingEquations.NeuralField(prediction.length);
            field.setExcitatory(prediction);
            
            // Decay and update
            for (int i = 0; i < prediction.length; i++) {
                prediction[i] *= 0.95; // Decay
            }
            
            shunting.updateField(field);
            prediction = field.getActivations();
            
            // Apply temporal prediction
            if (step < currentState.temporalPrediction.predictions.length) {
                for (int i = 0; i < prediction.length; i++) {
                    prediction[i] = 0.7 * prediction[i] + 
                                   0.3 * currentState.temporalPrediction.predictions[step];
                }
            }
        }
        
        return prediction;
    }
    
    /**
     * Reset dynamics to initial state
     */
    public void reset() {
        resonance.reset();
        attention.reset();
        oscillator.resetPhases();
    }
    
    /**
     * Shutdown executor
     */
    public void shutdown() {
        executor.shutdown();
    }
    
    /**
     * Set system parameters
     */
    public void setParameters(double vigilance, double gain, double coupling) {
        resonance.setVigilance(vigilance);
        attention.setGlobalGain(gain);
        // Oscillator coupling would need additional method
    }
    
    /**
     * Get system diagnostics
     */
    public Map<String, Object> getDiagnostics() {
        Map<String, Object> diagnostics = new HashMap<>();
        
        diagnostics.put("num_categories", resonance.getCategories().size());
        diagnostics.put("attention_load", attention.getState().attentionalLoad);
        diagnostics.put("oscillator_count", oscillator.getOscillators().size());
        
        // Add category statistics
        if (!resonance.getCategories().isEmpty()) {
            double avgCommitment = resonance.getCategories().stream()
                .mapToDouble(cat -> cat.commitment)
                .average()
                .orElse(0.0);
            diagnostics.put("avg_category_commitment", avgCommitment);
        }
        
        return diagnostics;
    }
}
