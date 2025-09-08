package com.art.textgen.dynamics;

import java.util.*;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Attentional Dynamics based on Grossberg's ART attention subsystem
 * Manages focus, gain control, and attentional shifts
 */
public class AttentionalDynamics {
    
    private static final double DEFAULT_GAIN = 2.0;
    private static final double ATTENTION_DECAY = 0.95;
    private static final double SHIFT_THRESHOLD = 0.3;
    
    private final ShuntingEquations shuntingDynamics;
    private final Map<String, AttentionFocus> foci;
    private AttentionFocus currentFocus;
    private double globalGain;
    
    public static class AttentionFocus {
        public final String id;
        public final double[] spatialCenter;
        public double spatialWidth;
        public double temporalPersistence;
        public double strength;
        public final Map<String, Double> features;
        public long creationTime;
        public int accessCount;
        
        public AttentionFocus(String id, double[] center, double width) {
            this.id = id;
            this.spatialCenter = center.clone();
            this.spatialWidth = width;
            this.temporalPersistence = 1.0;
            this.strength = 1.0;
            this.features = new HashMap<>();
            this.creationTime = System.currentTimeMillis();
            this.accessCount = 0;
        }
        
        public double computeActivation(double[] input) {
            // Gaussian activation around spatial center
            double distance = 0.0;
            for (int i = 0; i < Math.min(input.length, spatialCenter.length); i++) {
                distance += Math.pow(input[i] - spatialCenter[i], 2);
            }
            distance = Math.sqrt(distance);
            
            return strength * Math.exp(-distance * distance / (2 * spatialWidth * spatialWidth));
        }
        
        public void decay() {
            strength *= ATTENTION_DECAY;
            temporalPersistence *= ATTENTION_DECAY;
        }
        
        public boolean isActive() {
            return strength > 0.1;
        }
    }
    
    public static class AttentionalState {
        public final AttentionFocus primaryFocus;
        public final List<AttentionFocus> peripheralFoci;
        public final double attentionalLoad;
        public final boolean needsShift;
        
        public AttentionalState(AttentionFocus primary, List<AttentionFocus> peripheral,
                               double load, boolean shift) {
            this.primaryFocus = primary;
            this.peripheralFoci = new ArrayList<>(peripheral);
            this.attentionalLoad = load;
            this.needsShift = shift;
        }
    }
    
    public AttentionalDynamics() {
        this.shuntingDynamics = new ShuntingEquations();
        this.foci = new HashMap<>();
        this.globalGain = DEFAULT_GAIN;
        this.currentFocus = null;
    }
    
    /**
     * Process input through attentional system
     */
    public double[] processWithAttention(double[] input) {
        // Apply current attentional focus
        double[] modulated = modulateByAttention(input);
        
        // Apply gain control
        modulated = applyGainControl(modulated);
        
        // Process through shunting dynamics
        ShuntingEquations.NeuralField field = 
            new ShuntingEquations.NeuralField(modulated.length);
        field.setExcitatory(modulated);
        
        // Update field dynamics
        for (int i = 0; i < 10; i++) {
            shuntingDynamics.updateField(field);
            shuntingDynamics.applyLateralInhibition(field, 3.0);
        }
        
        // Check for attentional shift
        if (shouldShiftAttention(field)) {
            shiftAttention(input);
        }
        
        return field.getActivations();
    }
    
    /**
     * Modulate input by current attentional focus
     */
    private double[] modulateByAttention(double[] input) {
        if (currentFocus == null) {
            return input.clone();
        }
        
        double[] modulated = new double[input.length];
        double focusActivation = currentFocus.computeActivation(input);
        
        for (int i = 0; i < input.length; i++) {
            // Enhance attended features
            double enhancement = 1.0 + focusActivation * globalGain;
            modulated[i] = input[i] * enhancement;
            
            // Suppress unattended features (surround inhibition)
            for (AttentionFocus peripheral : getPeripheralFoci()) {
                double suppression = peripheral.computeActivation(input) * 0.3;
                modulated[i] *= (1.0 - suppression);
            }
        }
        
        return modulated;
    }
    
    /**
     * Apply gain control based on arousal/vigilance
     */
    private double[] applyGainControl(double[] input) {
        double[] controlled = new double[input.length];
        
        // Compute input statistics
        double mean = Arrays.stream(input).average().orElse(0.0);
        double variance = Arrays.stream(input)
            .map(x -> Math.pow(x - mean, 2))
            .average().orElse(0.0);
        
        // Adaptive gain based on input variance
        double adaptiveGain = globalGain / (1.0 + Math.sqrt(variance));
        
        // Apply sigmoidal gain control
        for (int i = 0; i < input.length; i++) {
            controlled[i] = sigmoid(input[i] * adaptiveGain);
        }
        
        return controlled;
    }
    
    /**
     * Determine if attention should shift
     */
    private boolean shouldShiftAttention(ShuntingEquations.NeuralField field) {
        if (currentFocus == null) return true;
        
        // Check if current focus is still strong
        if (currentFocus.strength < SHIFT_THRESHOLD) {
            return true;
        }
        
        // Check for competing activation peaks
        double[] activations = field.getActivations();
        int peakIndex = field.getWinner();
        double peakValue = activations[peakIndex];
        
        // Find second peak
        double secondPeak = 0.0;
        for (int i = 0; i < activations.length; i++) {
            if (i != peakIndex && activations[i] > secondPeak) {
                secondPeak = activations[i];
            }
        }
        
        // Shift if competition is strong
        return secondPeak / peakValue > 0.7;
    }
    
    /**
     * Shift attention to new focus
     */
    private void shiftAttention(double[] input) {
        // Decay current focus
        if (currentFocus != null) {
            currentFocus.decay();
        }
        
        // Find most salient region
        int peakIndex = findSalientRegion(input);
        
        // Create new focus
        double[] center = new double[input.length];
        center[peakIndex] = 1.0;
        
        // Smooth around peak
        for (int i = Math.max(0, peakIndex - 2); 
             i < Math.min(input.length, peakIndex + 3); i++) {
            if (i != peakIndex) {
                center[i] = 0.5 * Math.exp(-Math.abs(i - peakIndex) / 2.0);
            }
        }
        
        String focusId = "focus_" + System.currentTimeMillis();
        AttentionFocus newFocus = new AttentionFocus(focusId, center, 3.0);
        
        foci.put(focusId, newFocus);
        currentFocus = newFocus;
    }
    
    /**
     * Find most salient region in input
     */
    private int findSalientRegion(double[] input) {
        // Compute saliency map
        double[] saliency = new double[input.length];
        
        for (int i = 0; i < input.length; i++) {
            // Local contrast
            double localMean = 0.0;
            int count = 0;
            
            for (int j = Math.max(0, i - 3); 
                 j < Math.min(input.length, i + 4); j++) {
                localMean += input[j];
                count++;
            }
            
            if (count > 0) {
                localMean /= count;
                saliency[i] = Math.abs(input[i] - localMean);
            }
            
            // Temporal novelty (if we have history)
            if (currentFocus != null) {
                double expectedValue = currentFocus.spatialCenter[i];
                saliency[i] += Math.abs(input[i] - expectedValue) * 0.5;
            }
        }
        
        // Find peak saliency
        int peakIndex = 0;
        double peakSaliency = saliency[0];
        
        for (int i = 1; i < saliency.length; i++) {
            if (saliency[i] > peakSaliency) {
                peakSaliency = saliency[i];
                peakIndex = i;
            }
        }
        
        return peakIndex;
    }
    
    /**
     * Get peripheral (non-primary) foci
     */
    private List<AttentionFocus> getPeripheralFoci() {
        List<AttentionFocus> peripheral = new ArrayList<>();
        
        for (AttentionFocus focus : foci.values()) {
            if (focus != currentFocus && focus.isActive()) {
                peripheral.add(focus);
            }
        }
        
        // Sort by strength
        peripheral.sort((a, b) -> Double.compare(b.strength, a.strength));
        
        // Return top 3 peripheral foci
        return peripheral.subList(0, Math.min(3, peripheral.size()));
    }
    
    /**
     * Create attentional bias towards specific features
     */
    public void createBias(String featureName, double strength) {
        if (currentFocus != null) {
            currentFocus.features.put(featureName, strength);
        }
    }
    
    /**
     * Get current attentional state
     */
    public AttentionalState getState() {
        double load = computeAttentionalLoad();
        boolean needsShift = currentFocus == null || 
                            currentFocus.strength < SHIFT_THRESHOLD;
        
        return new AttentionalState(currentFocus, getPeripheralFoci(), 
                                   load, needsShift);
    }
    
    /**
     * Compute total attentional load
     */
    private double computeAttentionalLoad() {
        return foci.values().stream()
            .filter(AttentionFocus::isActive)
            .mapToDouble(f -> f.strength)
            .sum();
    }
    
    /**
     * Set global gain parameter
     */
    public void setGlobalGain(double gain) {
        this.globalGain = Math.max(0.1, Math.min(10.0, gain));
    }
    
    /**
     * Reset attentional system
     */
    public void reset() {
        foci.clear();
        currentFocus = null;
        globalGain = DEFAULT_GAIN;
    }
    
    /**
     * Prune old inactive foci
     */
    public void pruneFoci() {
        long currentTime = System.currentTimeMillis();
        
        foci.entrySet().removeIf(entry -> {
            AttentionFocus focus = entry.getValue();
            long age = currentTime - focus.creationTime;
            return !focus.isActive() && age > 60000; // Remove after 1 minute inactive
        });
    }
    
    /**
     * Sigmoid activation function
     */
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    /**
     * Apply vigilance-based modulation
     */
    public double[] applyVigilance(double[] input, double vigilance) {
        double[] modulated = new double[input.length];
        
        // High vigilance = narrow focus, low vigilance = broad focus
        double focusWidth = 10.0 * (1.0 - vigilance) + 1.0;
        
        if (currentFocus != null) {
            // Temporarily adjust focus width
            double originalWidth = currentFocus.spatialWidth;
            currentFocus.spatialWidth = focusWidth;
            
            modulated = modulateByAttention(input);
            
            // Restore original width
            currentFocus.spatialWidth = originalWidth;
        } else {
            modulated = input.clone();
        }
        
        return modulated;
    }
}
