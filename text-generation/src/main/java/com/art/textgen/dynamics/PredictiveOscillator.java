package com.art.textgen.dynamics;

import java.util.*;
import java.util.stream.IntStream;

/**
 * Predictive Oscillator for temporal pattern generation
 * Based on Grossberg's neural oscillator models for timing and rhythm
 */
public class PredictiveOscillator {
    
    private static final int DEFAULT_OSCILLATORS = 8;
    private static final double DEFAULT_COUPLING = 0.1;
    
    private final List<Oscillator> oscillators;
    private final double[][] couplingMatrix;
    private double globalPhase;
    private final AdaptiveRhythm adaptiveRhythm;
    
    public static class Oscillator {
        public final int id;
        public double frequency;
        public double phase;
        public double amplitude;
        public double adaptiveFreq;
        public double phaseError;
        public List<Double> phaseHistory;
        
        public Oscillator(int id, double frequency, double amplitude) {
            this.id = id;
            this.frequency = frequency;
            this.phase = Math.random() * 2 * Math.PI;
            this.amplitude = amplitude;
            this.adaptiveFreq = frequency;
            this.phaseError = 0.0;
            this.phaseHistory = new ArrayList<>();
        }
        
        public double getActivation(double time) {
            return amplitude * Math.sin(frequency * time + phase);
        }
        
        public void updatePhase(double dt) {
            phase += adaptiveFreq * dt;
            
            // Keep phase in [0, 2π]
            while (phase > 2 * Math.PI) {
                phase -= 2 * Math.PI;
            }
            while (phase < 0) {
                phase += 2 * Math.PI;
            }
            
            phaseHistory.add(phase);
            if (phaseHistory.size() > 100) {
                phaseHistory.remove(0);
            }
        }
    }
    
    public static class TemporalPrediction {
        public final double[] predictions;
        public final double confidence;
        public final int bestOscillator;
        public final double predictedPhase;
        public final Map<Integer, Double> oscillatorContributions;
        
        public TemporalPrediction(double[] predictions, double confidence, 
                                 int bestOscillator, double phase) {
            this.predictions = predictions.clone();
            this.confidence = confidence;
            this.bestOscillator = bestOscillator;
            this.predictedPhase = phase;
            this.oscillatorContributions = new HashMap<>();
        }
    }
    
    public static class AdaptiveRhythm {
        private final List<Double> beatTimes;
        private double estimatedPeriod;
        private double periodVariance;
        
        public AdaptiveRhythm() {
            this.beatTimes = new ArrayList<>();
            this.estimatedPeriod = 1.0;
            this.periodVariance = 0.1;
        }
        
        public void addBeat(double time) {
            beatTimes.add(time);
            
            if (beatTimes.size() > 1) {
                updatePeriodEstimate();
            }
            
            // Keep history bounded
            if (beatTimes.size() > 50) {
                beatTimes.remove(0);
            }
        }
        
        private void updatePeriodEstimate() {
            if (beatTimes.size() < 2) return;
            
            List<Double> intervals = new ArrayList<>();
            for (int i = 1; i < beatTimes.size(); i++) {
                intervals.add(beatTimes.get(i) - beatTimes.get(i - 1));
            }
            
            // Compute mean and variance of intervals
            double mean = intervals.stream()
                .mapToDouble(Double::doubleValue)
                .average()
                .orElse(1.0);
            
            double variance = intervals.stream()
                .mapToDouble(interval -> Math.pow(interval - mean, 2))
                .average()
                .orElse(0.1);
            
            // Smooth update
            estimatedPeriod = 0.7 * estimatedPeriod + 0.3 * mean;
            periodVariance = 0.7 * periodVariance + 0.3 * variance;
        }
        
        public double getPeriod() {
            return estimatedPeriod;
        }
        
        public double getConfidence() {
            // Lower variance = higher confidence
            return Math.exp(-periodVariance);
        }
    }
    
    public PredictiveOscillator(int numOscillators) {
        this.oscillators = new ArrayList<>();
        this.couplingMatrix = new double[numOscillators][numOscillators];
        this.globalPhase = 0.0;
        this.adaptiveRhythm = new AdaptiveRhythm();
        
        // Initialize oscillators with different frequencies
        for (int i = 0; i < numOscillators; i++) {
            // Frequencies from 0.5 Hz to 8 Hz (covers speech rhythms)
            double frequency = 0.5 * Math.pow(2, i / 2.0);
            double amplitude = 1.0 / (1.0 + i * 0.2); // Decay amplitude
            
            oscillators.add(new Oscillator(i, frequency, amplitude));
        }
        
        // Initialize coupling matrix (weak coupling)
        initializeCoupling();
    }
    
    public PredictiveOscillator() {
        this(DEFAULT_OSCILLATORS);
    }
    
    /**
     * Initialize coupling between oscillators
     */
    private void initializeCoupling() {
        Random rand = new Random();
        
        for (int i = 0; i < oscillators.size(); i++) {
            for (int j = 0; j < oscillators.size(); j++) {
                if (i != j) {
                    // Harmonic relationships have stronger coupling
                    double freqRatio = oscillators.get(i).frequency / 
                                      oscillators.get(j).frequency;
                    
                    if (isHarmonic(freqRatio)) {
                        couplingMatrix[i][j] = DEFAULT_COUPLING * 2;
                    } else {
                        couplingMatrix[i][j] = DEFAULT_COUPLING * rand.nextDouble();
                    }
                }
            }
        }
    }
    
    /**
     * Check if frequency ratio is harmonic
     */
    private boolean isHarmonic(double ratio) {
        double[] harmonics = {0.5, 1.0, 2.0, 3.0, 4.0};
        
        for (double harmonic : harmonics) {
            if (Math.abs(ratio - harmonic) < 0.1) {
                return true;
            }
        }
        
        return false;
    }
    
    /**
     * Update oscillator network
     */
    public void update(double dt) {
        // Store current phases
        double[] currentPhases = oscillators.stream()
            .mapToDouble(o -> o.phase)
            .toArray();
        
        // Update each oscillator with coupling
        for (int i = 0; i < oscillators.size(); i++) {
            Oscillator osc = oscillators.get(i);
            
            // Compute coupling influence
            double couplingForce = 0.0;
            
            for (int j = 0; j < oscillators.size(); j++) {
                if (i != j) {
                    double phaseDiff = currentPhases[j] - currentPhases[i];
                    couplingForce += couplingMatrix[i][j] * Math.sin(phaseDiff);
                }
            }
            
            // Adaptive frequency adjustment
            osc.adaptiveFreq = osc.frequency + couplingForce;
            
            // Update phase
            osc.updatePhase(dt);
        }
        
        // Update global phase
        globalPhase += dt;
    }
    
    /**
     * Generate temporal prediction
     */
    public TemporalPrediction predict(double[] input, int horizon) {
        double[] predictions = new double[horizon];
        
        // Find best matching oscillator based on input rhythm
        int bestOsc = findBestOscillator(input);
        Oscillator best = oscillators.get(bestOsc);
        
        // Generate predictions using oscillator ensemble
        for (int t = 0; t < horizon; t++) {
            double futureTime = globalPhase + t * 0.01; // 10ms steps
            
            // Weighted combination of oscillators
            double prediction = 0.0;
            double totalWeight = 0.0;
            
            for (Oscillator osc : oscillators) {
                double weight = computeOscillatorWeight(osc, best);
                prediction += weight * osc.getActivation(futureTime);
                totalWeight += weight;
            }
            
            if (totalWeight > 0) {
                predictions[t] = prediction / totalWeight;
            }
        }
        
        // Compute prediction confidence
        double confidence = computeConfidence(input, best);
        
        TemporalPrediction result = new TemporalPrediction(
            predictions, confidence, bestOsc, best.phase
        );
        
        // Add oscillator contributions
        for (Oscillator osc : oscillators) {
            result.oscillatorContributions.put(osc.id, 
                computeOscillatorWeight(osc, best));
        }
        
        return result;
    }
    
    /**
     * Find best matching oscillator for input rhythm
     */
    private int findBestOscillator(double[] input) {
        double[] matchScores = new double[oscillators.size()];
        
        for (int i = 0; i < oscillators.size(); i++) {
            Oscillator osc = oscillators.get(i);
            
            // Compute phase alignment with input
            double alignment = 0.0;
            
            for (int t = 0; t < input.length; t++) {
                double oscValue = osc.getActivation(t * 0.01);
                alignment += input[t] * oscValue;
            }
            
            // Normalize by input energy
            double inputEnergy = Arrays.stream(input)
                .map(x -> x * x)
                .sum();
            
            if (inputEnergy > 0) {
                matchScores[i] = alignment / Math.sqrt(inputEnergy);
            }
        }
        
        // Find maximum score
        return IntStream.range(0, matchScores.length)
            .reduce((i, j) -> matchScores[i] > matchScores[j] ? i : j)
            .orElse(0);
    }
    
    /**
     * Compute weight for oscillator contribution
     */
    private double computeOscillatorWeight(Oscillator osc, Oscillator reference) {
        // Weight based on frequency similarity and phase coherence
        double freqDiff = Math.abs(osc.frequency - reference.frequency);
        double freqWeight = Math.exp(-freqDiff / reference.frequency);
        
        double phaseDiff = Math.abs(osc.phase - reference.phase);
        double phaseWeight = Math.cos(phaseDiff);
        
        return freqWeight * Math.max(0, phaseWeight);
    }
    
    /**
     * Compute prediction confidence
     */
    private double computeConfidence(double[] input, Oscillator best) {
        // Confidence based on phase stability and rhythm regularity
        if (best.phaseHistory.size() < 10) {
            return 0.5; // Low confidence with little history
        }
        
        // Compute phase stability
        double phaseVariance = 0.0;
        double meanPhaseChange = 0.0;
        
        for (int i = 1; i < best.phaseHistory.size(); i++) {
            double change = best.phaseHistory.get(i) - best.phaseHistory.get(i - 1);
            
            // Unwrap phase
            if (change > Math.PI) change -= 2 * Math.PI;
            if (change < -Math.PI) change += 2 * Math.PI;
            
            meanPhaseChange += change;
        }
        
        meanPhaseChange /= (best.phaseHistory.size() - 1);
        
        for (int i = 1; i < best.phaseHistory.size(); i++) {
            double change = best.phaseHistory.get(i) - best.phaseHistory.get(i - 1);
            
            // Unwrap phase
            if (change > Math.PI) change -= 2 * Math.PI;
            if (change < -Math.PI) change += 2 * Math.PI;
            
            phaseVariance += Math.pow(change - meanPhaseChange, 2);
        }
        
        phaseVariance /= (best.phaseHistory.size() - 1);
        
        // Convert variance to confidence
        double stability = Math.exp(-phaseVariance);
        
        // Combine with adaptive rhythm confidence
        double rhythmConfidence = adaptiveRhythm.getConfidence();
        
        return stability * rhythmConfidence;
    }
    
    /**
     * Entrain oscillators to external rhythm
     */
    public void entrain(double[] rhythmSignal, double strength) {
        // Detect beats in rhythm signal
        List<Integer> beatIndices = detectBeats(rhythmSignal);
        
        // Update adaptive rhythm
        for (int beatIndex : beatIndices) {
            adaptiveRhythm.addBeat(beatIndex * 0.01); // Convert to time
        }
        
        // Adjust oscillator frequencies
        double targetPeriod = adaptiveRhythm.getPeriod();
        double targetFreq = 1.0 / targetPeriod;
        
        for (Oscillator osc : oscillators) {
            // Find closest harmonic
            double harmonic = Math.round(osc.frequency / targetFreq);
            if (harmonic < 1) harmonic = 1;
            
            double targetOscFreq = harmonic * targetFreq;
            
            // Gradual adjustment
            osc.frequency += strength * (targetOscFreq - osc.frequency);
        }
    }
    
    /**
     * Detect beats in rhythm signal
     */
    private List<Integer> detectBeats(double[] signal) {
        List<Integer> beats = new ArrayList<>();
        
        // Simple peak detection
        for (int i = 1; i < signal.length - 1; i++) {
            if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1]) {
                // Local maximum
                if (signal[i] > 0.5) { // Threshold
                    beats.add(i);
                }
            }
        }
        
        return beats;
    }
    
    /**
     * Reset oscillator phases
     */
    public void resetPhases() {
        for (Oscillator osc : oscillators) {
            osc.phase = 0.0;
            osc.phaseHistory.clear();
        }
        globalPhase = 0.0;
    }
    
    /**
     * Get current oscillator states
     */
    public List<Oscillator> getOscillators() {
        return new ArrayList<>(oscillators);
    }
    
    /**
     * Apply phase reset at specific oscillator
     */
    public void phaseReset(int oscillatorId, double newPhase) {
        if (oscillatorId >= 0 && oscillatorId < oscillators.size()) {
            oscillators.get(oscillatorId).phase = newPhase;
        }
    }
    
    /**
     * Get rhythmic pattern at current time
     */
    public double[] getCurrentPattern(int length) {
        double[] pattern = new double[length];
        
        for (int i = 0; i < length; i++) {
            double time = globalPhase + i * 0.01;
            
            // Sum all oscillator contributions
            for (Oscillator osc : oscillators) {
                pattern[i] += osc.getActivation(time);
            }
            
            // Normalize
            pattern[i] /= oscillators.size();
        }
        
        return pattern;
    }
}
