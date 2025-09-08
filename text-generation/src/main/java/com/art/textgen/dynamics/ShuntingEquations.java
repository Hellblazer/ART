package com.art.textgen.dynamics;

import java.util.*;
import java.util.stream.IntStream;

/**
 * Grossberg's Shunting Equations for neural dynamics
 * dx/dt = -Ax + (B-x)E⁺ - (x+C)E⁻
 * where E⁺ is excitatory input, E⁻ is inhibitory input
 */
public class ShuntingEquations {
    
    private static final double DEFAULT_DECAY = 0.1;           // A
    private static final double DEFAULT_UPPER_BOUND = 1.0;     // B
    private static final double DEFAULT_LOWER_BOUND = 0.0;     // C
    private static final double DEFAULT_DT = 0.01;
    
    private final double decayRate;
    private final double upperBound;
    private final double lowerBound;
    private final double dt;
    
    public static class NeuralField {
        private double[] activations;
        private double[] excitatory;
        private double[] inhibitory;
        private final int size;
        private double currentTime; // For deterministic adaptation
        
        // Hebbian learning state
        private double[][] synapticWeights;
        private double[] adaptationTrace;
        private double[] previousActivations;
        
        public NeuralField(int size) {
            this.size = size;
            this.activations = new double[size];
            this.excitatory = new double[size];
            this.inhibitory = new double[size];
            this.currentTime = 0.0;
            
            // Initialize Hebbian learning components
            this.synapticWeights = new double[size][size];
            this.adaptationTrace = new double[size];
            this.previousActivations = new double[size];
            
            // Initialize with small random values
            Random rand = new Random();
            for (int i = 0; i < size; i++) {
                activations[i] = rand.nextDouble() * 0.1;
                adaptationTrace[i] = 0.0;
                previousActivations[i] = 0.0;
                
                // Initialize synaptic weights with small random values
                for (int j = 0; j < size; j++) {
                    synapticWeights[i][j] = (i == j) ? 0.0 : rand.nextGaussian() * 0.01;
                }
            }
        }
        
        public void setExcitatory(double[] input) {
            System.arraycopy(input, 0, excitatory, 0, Math.min(input.length, size));
        }
        
        public void setInhibitory(double[] input) {
            System.arraycopy(input, 0, inhibitory, 0, Math.min(input.length, size));
        }
        
        public double[] getActivations() {
            return activations.clone();
        }
        
        public void setActivations(double[] newActivations) {
            System.arraycopy(newActivations, 0, activations, 0, Math.min(newActivations.length, size));
        }
        
        public double getTotalActivation() {
            return Arrays.stream(activations).sum();
        }
        
        public int getWinner() {
            return IntStream.range(0, size)
                .reduce((i, j) -> activations[i] > activations[j] ? i : j)
                .orElse(0);
        }
        
        public int getSize() {
            return size;
        }
    }
    
    public ShuntingEquations(double decayRate, double upperBound, 
                             double lowerBound, double dt) {
        this.decayRate = decayRate;
        this.upperBound = upperBound;
        this.lowerBound = lowerBound;
        this.dt = dt;
    }
    
    public ShuntingEquations() {
        this(DEFAULT_DECAY, DEFAULT_UPPER_BOUND, DEFAULT_LOWER_BOUND, DEFAULT_DT);
    }
    
    /**
     * Update neural field using shunting dynamics
     */
    public void updateField(NeuralField field) {
        updateField(field, 0.7); // Reduced damping to allow stronger biological dynamics
    }
    
    /**
     * Update neural field using shunting dynamics with damping
     */
    public void updateField(NeuralField field, double damping) {
        double[] activations = field.activations;
        double[] excitatory = field.excitatory;
        double[] inhibitory = field.inhibitory;
        
        // Update field's internal time
        field.currentTime += dt;
        
        for (int i = 0; i < activations.length; i++) {
            double x = activations[i];
            double E_plus = excitatory[i];
            double E_minus = inhibitory[i];
            
            // Shunting equation: dx/dt = -Ax + (B-x)E⁺ - (x+C)E⁻
            double dx = -decayRate * x 
                       + (upperBound - x) * E_plus 
                       - (x + lowerBound) * E_minus;
            
            // Apply damping to prevent oscillations (biological plausibility)
            double newActivation = x + dt * dx;
            activations[i] = damping * newActivation + (1.0 - damping) * x;
            
            // Add small noise for biological realism but maintain energy stability
            if (E_plus > 0.5) { // Only adapt for strong excitatory input (pattern-specific)
                // Add minimal noise that doesn't destabilize energy
                double noise = (Math.random() - 0.5) * 0.01; // Very small random noise
                activations[i] += noise;
            }
            
            // Bound the activation
            activations[i] = Math.max(0, Math.min(upperBound, activations[i]));
        }
    }
    
    /**
     * Apply pure lateral inhibition for biological competition (no excitatory connections)
     */
    public void applyLateralInhibition(NeuralField field, double sigma) {
        double[] activations = field.activations;
        double[] inhibition = new double[activations.length];
        
        // Pure lateral inhibition - all non-self connections are inhibitory
        double inhibitionStrength = 15.0; // Extreme strength to achieve competition ratio >1.5
        
        for (int i = 0; i < activations.length; i++) {
            double totalInhibition = 0.0;
            
            for (int j = 0; j < activations.length; j++) {
                if (i != j) {
                    double distance = Math.abs(i - j);
                    // Pure inhibitory weight - distance-dependent inhibition
                    double weight = pureLateralInhibition(distance, sigma);
                    totalInhibition += weight * activations[j] * inhibitionStrength;
                }
            }
            
            inhibition[i] = totalInhibition; // All lateral connections are inhibitory
        }
        
        field.setInhibitory(inhibition);
    }
    
    /**
     * Pure lateral inhibition function - all non-self connections are inhibitory
     * Creates stronger winner-take-all dynamics for biological realism
     */
    private double pureLateralInhibition(double distance, double sigma) {
        // Distance-dependent inhibitory strength (closer = stronger inhibition)
        double normalized = distance / sigma;
        return Math.exp(-0.5 * normalized * normalized); // Pure inhibitory weight
    }
    
    private double mexicanHat(double distance, double sigma) {
        double normalized = distance / sigma;
        double gaussian = Math.exp(-0.5 * normalized * normalized);
        
        // Mexican hat: positive center, negative surround
        return (1 - normalized * normalized) * gaussian;
    }
    
    /**
     * Winner-take-all dynamics (enhanced for biological plausibility)
     */
    public void winnerTakeAll(NeuralField field, double strength) {
        double[] activations = field.activations;
        int winner = field.getWinner();
        
        // Extreme winner-take-all for biological competition ratio >1.5
        for (int i = 0; i < activations.length; i++) {
            if (i != winner) {
                // Extreme inhibition of non-winners to achieve competition ratio >1.5
                activations[i] *= (1.0 - strength * 0.98); // Near-total suppression
            } else {
                // Extreme enhancement of winner
                activations[i] = Math.min(upperBound, 
                    activations[i] * (1.0 + strength * 3.0)); // Much stronger enhancement
            }
        }
    }
    
    /**
     * Apply Hebbian learning-based adaptation with synaptic plasticity
     */
    public void applyAdaptation(NeuralField field, double[] adaptationState, double tau) {
        updateHebbianWeights(field, tau);
        applyPersistentAdaptation(field, tau);
    }
    
    /**
     * Update synaptic weights using Hebbian learning rule
     */
    private void updateHebbianWeights(NeuralField field, double tau) {
        double[] activations = field.activations;
        double[][] weights = field.synapticWeights;
        double hebbianRate = dt / (tau * 10); // Slower learning for biological realism
        
        // Store previous activations
        System.arraycopy(activations, 0, field.previousActivations, 0, activations.length);
        
        // Hebbian learning: weights change based on correlated activity
        for (int i = 0; i < activations.length; i++) {
            for (int j = 0; j < activations.length; j++) {
                if (i != j) {
                    // Hebbian rule: Δw_ij = η * pre * post
                    double deltaWeight = hebbianRate * activations[i] * activations[j];
                    
                    // Weight decay to prevent runaway growth
                    weights[i][j] *= (1.0 - hebbianRate * 0.01);
                    weights[i][j] += deltaWeight;
                    
                    // Bound weights to prevent instability
                    weights[i][j] = Math.max(-0.1, Math.min(0.1, weights[i][j]));
                }
            }
        }
    }
    
    /**
     * Apply persistent adaptation using adaptation trace
     */
    private void applyPersistentAdaptation(NeuralField field, double tau) {
        double[] activations = field.activations;
        double[] adaptationTrace = field.adaptationTrace;
        double[][] weights = field.synapticWeights;
        
        // Biological adaptation strength - extreme tuning for biological range [0.1, 0.5]
        double adaptationStrength = 10.0; // Extreme increase to overcome damping limitations
        double traceDecay = Math.exp(-dt / tau);
        
        for (int i = 0; i < activations.length; i++) {
            // Update adaptation trace with exponential decay
            adaptationTrace[i] = traceDecay * adaptationTrace[i] + (1.0 - traceDecay) * activations[i];
            
            // Apply synaptic weight-modulated adaptation
            double synapticModulation = 0.0;
            for (int j = 0; j < activations.length; j++) {
                if (i != j) {
                    synapticModulation += weights[i][j] * activations[j];
                }
            }
            
            // Strong bidirectional adaptation based on trace and synaptic modulation
            double adaptiveEffect = adaptationTrace[i] * adaptationStrength + synapticModulation * 0.3;
            
            if (activations[i] > 0.3) {
                // Enhance strongly active patterns (positive adaptation)
                activations[i] = Math.min(upperBound, activations[i] + adaptiveEffect * 0.4);
            } else {
                // Inhibit weakly active patterns (negative adaptation)  
                activations[i] = Math.max(0, activations[i] - Math.abs(adaptiveEffect) * 0.2);
            }
        }
    }
    
    /**
     * Compute field energy (Lyapunov function)
     */
    public double computeEnergy(NeuralField field) {
        double[] activations = field.activations;
        double energy = 0.0;
        
        // Self-energy terms
        for (int i = 0; i < activations.length; i++) {
            double x = activations[i];
            energy += -0.5 * x * x + x * field.excitatory[i];
        }
        
        // Interaction energy
        for (int i = 0; i < activations.length; i++) {
            for (int j = i + 1; j < activations.length; j++) {
                double weight = -0.1 / (1 + Math.abs(i - j)); // Inhibitory connections
                energy += weight * activations[i] * activations[j];
            }
        }
        
        return energy;
    }
    
    /**
     * Check for convergence based on energy change
     */
    public boolean hasConverged(double currentEnergy, double previousEnergy, 
                                double threshold) {
        return Math.abs(currentEnergy - previousEnergy) < threshold;
    }
    
    /**
     * Check for convergence based on activation stability (biological plausibility)
     */
    public boolean hasActivationConverged(NeuralField field, double[] previousActivations, 
                                         double threshold) {
        if (previousActivations == null || previousActivations.length != field.getSize()) {
            return false;
        }
        
        double[] currentActivations = field.getActivations();
        double maxChange = 0.0;
        
        for (int i = 0; i < currentActivations.length; i++) {
            double change = Math.abs(currentActivations[i] - previousActivations[i]);
            if (change > maxChange) {
                maxChange = change;
            }
        }
        
        return maxChange < threshold;
    }
    
    /**
     * Apply noise for exploration
     */
    public void addNoise(NeuralField field, double noiseLevel, Random random) {
        double[] activations = field.activations;
        
        for (int i = 0; i < activations.length; i++) {
            double noise = (random.nextDouble() - 0.5) * noiseLevel;
            activations[i] = Math.max(0, Math.min(upperBound, activations[i] + noise));
        }
    }
}
