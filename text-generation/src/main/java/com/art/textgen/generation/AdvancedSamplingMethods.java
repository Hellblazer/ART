package com.art.textgen.generation;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Advanced sampling methods for text generation
 * Implements Top-k, Top-p (Nucleus), and adaptive temperature sampling
 * Based on Phase 4.2 of EXECUTION_PLAN.md
 */
public class AdvancedSamplingMethods {
    
    private double temperature = 1.0;
    private int topK = 40;
    private double topP = 0.9;
    private boolean adaptiveTemperature = true;
    
    // Temperature range for adaptive scaling
    private final double MIN_TEMPERATURE = 0.5;
    private final double MAX_TEMPERATURE = 1.5;
    
    public AdvancedSamplingMethods() {
        // Default constructor with standard parameters
    }
    
    public AdvancedSamplingMethods(double temperature, int topK, double topP) {
        this.temperature = temperature;
        this.topK = topK;
        this.topP = topP;
    }
    
    /**
     * Sample from probability distribution using selected method
     */
    public int sample(double[] logits, SamplingMethod method, double contextUncertainty) {
        // Apply adaptive temperature if enabled
        if (adaptiveTemperature) {
            temperature = calculateAdaptiveTemperature(contextUncertainty);
        }
        
        // Convert logits to probabilities with temperature
        double[] probabilities = softmaxWithTemperature(logits, temperature);
        
        // Apply sampling method
        switch (method) {
            case TOP_K:
                return topKSampling(probabilities, topK);
            case TOP_P:
                return topPSampling(probabilities, topP);
            case TOP_K_P:
                return topKPCombinedSampling(probabilities, topK, topP);
            case GREEDY:
                return greedySampling(probabilities);
            case TEMPERATURE:
                return temperatureSampling(probabilities);
            default:
                return temperatureSampling(probabilities);
        }
    }
    
    /**
     * Top-k sampling: keep only top k most likely tokens
     */
    public int topKSampling(double[] probabilities, int k) {
        if (k <= 0 || k > probabilities.length) {
            k = probabilities.length;
        }
        
        // Create indices array
        Integer[] indices = new Integer[probabilities.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        
        // Sort indices by probability (descending)
        Arrays.sort(indices, (i, j) -> Double.compare(probabilities[j], probabilities[i]));
        
        // Keep only top k
        double[] topKProbs = new double[k];
        int[] topKIndices = new int[k];
        double sum = 0;
        
        for (int i = 0; i < k; i++) {
            topKProbs[i] = probabilities[indices[i]];
            topKIndices[i] = indices[i];
            sum += topKProbs[i];
        }
        
        // Renormalize
        for (int i = 0; i < k; i++) {
            topKProbs[i] /= sum;
        }
        
        // Sample from top-k distribution
        int sampledIdx = sampleFromDistribution(topKProbs);
        return topKIndices[sampledIdx];
    }
    
    /**
     * Top-p (Nucleus) sampling: keep tokens until cumulative probability > p
     */
    public int topPSampling(double[] probabilities, double p) {
        if (p <= 0 || p > 1) {
            p = 1.0;
        }
        
        // Create indices array
        Integer[] indices = new Integer[probabilities.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        
        // Sort indices by probability (descending)
        Arrays.sort(indices, (i, j) -> Double.compare(probabilities[j], probabilities[i]));
        
        // Find nucleus
        double cumSum = 0;
        int nucleusSize = 0;
        
        for (int i = 0; i < indices.length; i++) {
            cumSum += probabilities[indices[i]];
            nucleusSize++;
            if (cumSum >= p) {
                break;
            }
        }
        
        // Keep at least one token
        nucleusSize = Math.max(1, nucleusSize);
        
        // Create nucleus distribution
        double[] nucleusProbs = new double[nucleusSize];
        int[] nucleusIndices = new int[nucleusSize];
        double sum = 0;
        
        for (int i = 0; i < nucleusSize; i++) {
            nucleusProbs[i] = probabilities[indices[i]];
            nucleusIndices[i] = indices[i];
            sum += nucleusProbs[i];
        }
        
        // Renormalize
        for (int i = 0; i < nucleusSize; i++) {
            nucleusProbs[i] /= sum;
        }
        
        // Sample from nucleus
        int sampledIdx = sampleFromDistribution(nucleusProbs);
        return nucleusIndices[sampledIdx];
    }
    
    /**
     * Combined Top-k and Top-p sampling
     * First apply top-k, then top-p on the result
     */
    public int topKPCombinedSampling(double[] probabilities, int k, double p) {
        // First apply top-k filtering
        if (k > 0 && k < probabilities.length) {
            // Create indices array
            Integer[] indices = new Integer[probabilities.length];
            for (int i = 0; i < indices.length; i++) {
                indices[i] = i;
            }
            
            // Sort indices by probability (descending)
            final double[] probs = probabilities;
            Arrays.sort(indices, (i, j) -> Double.compare(probs[j], probs[i]));
            
            // Create filtered distribution with top-k
            double[] filteredProbs = new double[probabilities.length];
            for (int i = 0; i < Math.min(k, indices.length); i++) {
                filteredProbs[indices[i]] = probabilities[indices[i]];
            }
            
            // Renormalize
            double sum = Arrays.stream(filteredProbs).sum();
            if (sum > 0) {
                for (int i = 0; i < filteredProbs.length; i++) {
                    filteredProbs[i] /= sum;
                }
            }
            
            probabilities = filteredProbs;
        }
        
        // Then apply top-p sampling
        return topPSampling(probabilities, p);
    }
    
    /**
     * Greedy sampling: always pick the most likely token
     */
    public int greedySampling(double[] probabilities) {
        int maxIdx = 0;
        double maxProb = probabilities[0];
        
        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                maxIdx = i;
            }
        }
        
        return maxIdx;
    }
    
    /**
     * Temperature sampling: sample from the full distribution
     */
    public int temperatureSampling(double[] probabilities) {
        return sampleFromDistribution(probabilities);
    }
    
    /**
     * Calculate adaptive temperature based on context uncertainty
     * Higher uncertainty -> higher temperature (more randomness)
     * Lower uncertainty -> lower temperature (more focused)
     */
    private double calculateAdaptiveTemperature(double contextUncertainty) {
        // Map uncertainty [0, 1] to temperature range [MIN, MAX]
        double adaptedTemp = MIN_TEMPERATURE + 
            (MAX_TEMPERATURE - MIN_TEMPERATURE) * contextUncertainty;
        
        return Math.max(MIN_TEMPERATURE, Math.min(MAX_TEMPERATURE, adaptedTemp));
    }
    
    /**
     * Apply softmax with temperature to logits
     */
    private double[] softmaxWithTemperature(double[] logits, double temp) {
        if (temp <= 0) {
            temp = 1e-10; // Avoid division by zero
        }
        
        double[] scaled = new double[logits.length];
        double maxLogit = Arrays.stream(logits).max().orElse(0);
        
        // Scale by temperature and subtract max for numerical stability
        for (int i = 0; i < logits.length; i++) {
            scaled[i] = (logits[i] - maxLogit) / temp;
        }
        
        // Compute exp
        double[] exp = new double[scaled.length];
        double sum = 0;
        for (int i = 0; i < scaled.length; i++) {
            exp[i] = Math.exp(scaled[i]);
            sum += exp[i];
        }
        
        // Normalize
        for (int i = 0; i < exp.length; i++) {
            exp[i] /= sum;
        }
        
        return exp;
    }
    
    /**
     * Sample an index from a probability distribution
     */
    private int sampleFromDistribution(double[] probabilities) {
        double random = Math.random();
        double cumSum = 0;
        
        for (int i = 0; i < probabilities.length; i++) {
            cumSum += probabilities[i];
            if (random < cumSum) {
                return i;
            }
        }
        
        // Fallback to last index (shouldn't happen with proper normalization)
        return probabilities.length - 1;
    }
    
    /**
     * Calculate entropy of a probability distribution
     * Used to measure uncertainty
     */
    public double calculateEntropy(double[] probabilities) {
        double entropy = 0;
        for (double p : probabilities) {
            if (p > 0) {
                entropy -= p * Math.log(p) / Math.log(2);
            }
        }
        return entropy;
    }
    
    /**
     * Calculate perplexity from log probabilities
     */
    public double calculatePerplexity(double[] logProbs) {
        double avgLogProb = Arrays.stream(logProbs).average().orElse(0);
        return Math.exp(-avgLogProb);
    }
    
    /**
     * Beam search for generating multiple hypotheses
     */
    public List<BeamHypothesis> beamSearch(TokenProbabilityProvider provider,
                                          String prompt,
                                          int beamWidth,
                                          int maxLength) {
        List<BeamHypothesis> beams = new ArrayList<>();
        beams.add(new BeamHypothesis(prompt, 0.0));
        
        for (int step = 0; step < maxLength; step++) {
            List<BeamHypothesis> candidates = new ArrayList<>();
            
            for (BeamHypothesis beam : beams) {
                if (beam.isComplete()) {
                    candidates.add(beam);
                    continue;
                }
                
                double[] probs = provider.getTokenProbabilities(beam.text);
                
                // Get top-k candidates for this beam
                Integer[] indices = new Integer[probs.length];
                for (int i = 0; i < indices.length; i++) {
                    indices[i] = i;
                }
                Arrays.sort(indices, (i, j) -> Double.compare(probs[j], probs[i]));
                
                for (int i = 0; i < Math.min(beamWidth, indices.length); i++) {
                    int tokenIdx = indices[i];
                    String token = provider.indexToToken(tokenIdx);
                    double logProb = Math.log(probs[tokenIdx]);
                    
                    BeamHypothesis newBeam = new BeamHypothesis(
                        beam.text + " " + token,
                        beam.score + logProb
                    );
                    
                    if (token.equals(".") || token.equals("!") || token.equals("?")) {
                        newBeam.setComplete(true);
                    }
                    
                    candidates.add(newBeam);
                }
            }
            
            // Sort candidates by score and keep top beam_width
            candidates.sort((a, b) -> Double.compare(b.score, a.score));
            beams = candidates.subList(0, Math.min(beamWidth, candidates.size()));
            
            // Early stopping if all beams are complete
            if (beams.stream().allMatch(BeamHypothesis::isComplete)) {
                break;
            }
        }
        
        return beams;
    }
    
    // Getters and setters
    
    public double getTemperature() { return temperature; }
    public void setTemperature(double temperature) { this.temperature = temperature; }
    
    public int getTopK() { return topK; }
    public void setTopK(int topK) { this.topK = topK; }
    
    public double getTopP() { return topP; }
    public void setTopP(double topP) { this.topP = topP; }
    
    public boolean isAdaptiveTemperature() { return adaptiveTemperature; }
    public void setAdaptiveTemperature(boolean adaptive) { this.adaptiveTemperature = adaptive; }
    
    /**
     * Sampling method enum
     */
    public enum SamplingMethod {
        GREEDY,
        TEMPERATURE,
        TOP_K,
        TOP_P,
        TOP_K_P  // Combined top-k and top-p
    }
    
    /**
     * Beam search hypothesis
     */
    public static class BeamHypothesis {
        public final String text;
        public final double score;
        private boolean complete;
        
        public BeamHypothesis(String text, double score) {
            this.text = text;
            this.score = score;
            this.complete = false;
        }
        
        public boolean isComplete() { return complete; }
        public void setComplete(boolean complete) { this.complete = complete; }
    }
    
    /**
     * Interface for providing token probabilities
     */
    public interface TokenProbabilityProvider {
        double[] getTokenProbabilities(String context);
        String indexToToken(int index);
    }
}
