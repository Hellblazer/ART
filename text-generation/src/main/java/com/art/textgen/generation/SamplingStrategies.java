package com.art.textgen.generation;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Advanced sampling strategies for text generation
 * Implements top-k, top-p (nucleus), and temperature-based sampling
 * Based on Phase 4.2 of EXECUTION_PLAN.md
 */
public class SamplingStrategies {
    
    /**
     * Token with its probability
     */
    public static class TokenProbability implements Comparable<TokenProbability> {
        public final String token;
        public final double probability;
        public final double logProb;
        
        public TokenProbability(String token, double probability) {
            this.token = token;
            this.probability = probability;
            this.logProb = Math.log(probability);
        }
        
        @Override
        public int compareTo(TokenProbability other) {
            return Double.compare(other.probability, this.probability); // Descending order
        }
    }
    
    /**
     * Configuration for sampling
     */
    public static class SamplingConfig {
        public double temperature = 1.0;       // Temperature for probability scaling
        public int topK = 40;                  // Top-k filtering
        public double topP = 0.9;              // Top-p (nucleus) filtering
        public double repetitionPenalty = 1.0; // Penalty for repeated tokens
        public boolean adaptiveTemp = false;   // Enable adaptive temperature
        public double minTemp = 0.5;          // Minimum temperature for adaptive
        public double maxTemp = 1.5;          // Maximum temperature for adaptive
        
        public static SamplingConfig defaultConfig() {
            return new SamplingConfig();
        }
        
        public static SamplingConfig conservative() {
            SamplingConfig config = new SamplingConfig();
            config.temperature = 0.7;
            config.topK = 20;
            config.topP = 0.8;
            return config;
        }
        
        public static SamplingConfig creative() {
            SamplingConfig config = new SamplingConfig();
            config.temperature = 1.2;
            config.topK = 50;
            config.topP = 0.95;
            return config;
        }
    }
    
    private final Random random;
    private final Map<String, Integer> recentTokens;
    private final int recentTokensWindow = 50;
    
    public SamplingStrategies() {
        this(new Random());
    }
    
    public SamplingStrategies(long seed) {
        this(new Random(seed));
    }
    
    public SamplingStrategies(Random random) {
        this.random = random;
        this.recentTokens = new LinkedHashMap<String, Integer>(recentTokensWindow + 1, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<String, Integer> eldest) {
                return size() > recentTokensWindow;
            }
        };
    }
    
    /**
     * Sample a token using the configured strategy
     */
    public String sample(List<TokenProbability> candidates, SamplingConfig config) {
        if (candidates.isEmpty()) {
            return null;
        }
        
        // Apply repetition penalty
        candidates = applyRepetitionPenalty(candidates, config.repetitionPenalty);
        
        // Apply temperature (with optional adaptive scaling)
        double temperature = config.adaptiveTemp ? 
            calculateAdaptiveTemperature(candidates, config) : config.temperature;
        candidates = applyTemperature(candidates, temperature);
        
        // Apply top-k filtering
        candidates = applyTopK(candidates, config.topK);
        
        // Apply top-p (nucleus) filtering
        candidates = applyTopP(candidates, config.topP);
        
        // Renormalize probabilities
        candidates = renormalize(candidates);
        
        // Sample from the filtered distribution
        String selected = sampleFromDistribution(candidates);
        
        // Update recent tokens for repetition penalty
        updateRecentTokens(selected);
        
        return selected;
    }
    
    /**
     * Apply temperature scaling to probabilities
     */
    private List<TokenProbability> applyTemperature(List<TokenProbability> candidates, double temperature) {
        if (temperature == 1.0) {
            return candidates;
        }
        
        return candidates.stream()
            .map(tp -> {
                double scaledLogProb = tp.logProb / temperature;
                double scaledProb = Math.exp(scaledLogProb);
                return new TokenProbability(tp.token, scaledProb);
            })
            .collect(Collectors.toList());
    }
    
    /**
     * Calculate adaptive temperature based on entropy
     */
    private double calculateAdaptiveTemperature(List<TokenProbability> candidates, SamplingConfig config) {
        // Calculate entropy of the distribution
        double entropy = 0;
        for (TokenProbability tp : candidates) {
            if (tp.probability > 0) {
                entropy -= tp.probability * Math.log(tp.probability);
            }
        }
        
        // Normalize entropy (0 to 1)
        double maxEntropy = Math.log(candidates.size());
        double normalizedEntropy = maxEntropy > 0 ? entropy / maxEntropy : 0;
        
        // Map entropy to temperature
        // Low entropy (confident) -> lower temperature
        // High entropy (uncertain) -> higher temperature
        double temperature = config.minTemp + (config.maxTemp - config.minTemp) * normalizedEntropy;
        
        return temperature;
    }
    
    /**
     * Apply repetition penalty to recently used tokens
     */
    private List<TokenProbability> applyRepetitionPenalty(List<TokenProbability> candidates, double penalty) {
        if (penalty == 1.0 || recentTokens.isEmpty()) {
            return candidates;
        }
        
        return candidates.stream()
            .map(tp -> {
                if (recentTokens.containsKey(tp.token)) {
                    int distance = recentTokensWindow - recentTokens.get(tp.token);
                    double penaltyFactor = Math.pow(penalty, (double) distance / recentTokensWindow);
                    double penalizedProb = tp.probability / penaltyFactor;
                    return new TokenProbability(tp.token, penalizedProb);
                }
                return tp;
            })
            .collect(Collectors.toList());
    }
    
    /**
     * Apply top-k filtering
     */
    private List<TokenProbability> applyTopK(List<TokenProbability> candidates, int k) {
        if (k <= 0 || k >= candidates.size()) {
            return candidates;
        }
        
        // Sort by probability (descending) and take top k
        return candidates.stream()
            .sorted()
            .limit(k)
            .collect(Collectors.toList());
    }
    
    /**
     * Apply top-p (nucleus) filtering
     */
    private List<TokenProbability> applyTopP(List<TokenProbability> candidates, double p) {
        if (p <= 0 || p >= 1.0) {
            return candidates;
        }
        
        // Sort by probability (descending)
        List<TokenProbability> sorted = candidates.stream()
            .sorted()
            .collect(Collectors.toList());
        
        // Keep tokens until cumulative probability exceeds p
        List<TokenProbability> nucleus = new ArrayList<>();
        double cumulative = 0;
        
        for (TokenProbability tp : sorted) {
            nucleus.add(tp);
            cumulative += tp.probability;
            if (cumulative >= p) {
                break;
            }
        }
        
        // Always keep at least one token
        if (nucleus.isEmpty() && !sorted.isEmpty()) {
            nucleus.add(sorted.get(0));
        }
        
        return nucleus;
    }
    
    /**
     * Renormalize probabilities to sum to 1
     */
    private List<TokenProbability> renormalize(List<TokenProbability> candidates) {
        double sum = candidates.stream()
            .mapToDouble(tp -> tp.probability)
            .sum();
        
        if (sum == 0) {
            // Uniform distribution if all probabilities are 0
            double uniformProb = 1.0 / candidates.size();
            return candidates.stream()
                .map(tp -> new TokenProbability(tp.token, uniformProb))
                .collect(Collectors.toList());
        }
        
        return candidates.stream()
            .map(tp -> new TokenProbability(tp.token, tp.probability / sum))
            .collect(Collectors.toList());
    }
    
    /**
     * Sample from a probability distribution
     */
    private String sampleFromDistribution(List<TokenProbability> candidates) {
        if (candidates.isEmpty()) {
            return null;
        }
        
        if (candidates.size() == 1) {
            return candidates.get(0).token;
        }
        
        // Create cumulative distribution
        double randomValue = random.nextDouble();
        double cumulative = 0;
        
        for (TokenProbability tp : candidates) {
            cumulative += tp.probability;
            if (randomValue <= cumulative) {
                return tp.token;
            }
        }
        
        // Fallback to last token (shouldn't happen with proper normalization)
        return candidates.get(candidates.size() - 1).token;
    }
    
    /**
     * Update recent tokens for repetition penalty
     */
    private void updateRecentTokens(String token) {
        recentTokens.put(token, recentTokensWindow);
        
        // Decrement positions of existing tokens
        for (Map.Entry<String, Integer> entry : recentTokens.entrySet()) {
            if (!entry.getKey().equals(token)) {
                entry.setValue(entry.getValue() - 1);
            }
        }
    }
    
    /**
     * Clear recent tokens history
     */
    public void clearHistory() {
        recentTokens.clear();
    }
    
    /**
     * Create a simple greedy sampler (always picks highest probability)
     */
    public static String greedySample(List<TokenProbability> candidates) {
        if (candidates.isEmpty()) {
            return null;
        }
        
        return candidates.stream()
            .max(Comparator.comparingDouble(tp -> tp.probability))
            .map(tp -> tp.token)
            .orElse(null);
    }
    
    /**
     * Create a beam search sampler
     */
    public static class BeamSearchSampler {
        private final int beamWidth;
        
        public BeamSearchSampler(int beamWidth) {
            this.beamWidth = beamWidth;
        }
        
        /**
         * Beam search for sequence generation
         */
        public List<Beam> search(List<TokenProbability> candidates, int maxLength) {
            List<Beam> beams = new ArrayList<>();
            
            // Initialize beams with top-k candidates
            candidates.stream()
                .sorted()
                .limit(beamWidth)
                .forEach(tp -> beams.add(new Beam(tp.token, tp.logProb)));
            
            return beams;
        }
        
        public static class Beam {
            public final List<String> tokens;
            public final double score;
            
            public Beam(String token, double score) {
                this.tokens = new ArrayList<>();
                this.tokens.add(token);
                this.score = score;
            }
            
            public Beam(List<String> tokens, double score) {
                this.tokens = new ArrayList<>(tokens);
                this.score = score;
            }
            
            public Beam extend(String token, double logProb) {
                List<String> newTokens = new ArrayList<>(tokens);
                newTokens.add(token);
                return new Beam(newTokens, score + logProb);
            }
        }
    }
}
