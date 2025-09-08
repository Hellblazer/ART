package com.art.textgen.generation;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Advanced sampling methods for text generation
 * Implements top-k, top-p (nucleus), and combined sampling strategies
 */
public class AdvancedSampler {
    
    private final Random random;
    private int topK = 40;
    private double topP = 0.9;
    private double temperature = 0.8;
    private boolean useTopK = true;
    private boolean useTopP = true;
    
    // Sampling statistics
    private int totalSamples = 0;
    private double averageCandidates = 0;
    
    public AdvancedSampler() {
        this.random = new Random();
    }
    
    public AdvancedSampler(long seed) {
        this.random = new Random(seed);
    }
    
    /**
     * Sample token using advanced sampling methods
     */
    public String sample(Map<String, Double> probabilities) {
        if (probabilities.isEmpty()) {
            return null;
        }
        
        // Apply temperature
        Map<String, Double> tempered = applyTemperature(probabilities);
        
        // Apply top-k filtering
        if (useTopK) {
            tempered = applyTopK(tempered);
        }
        
        // Apply top-p (nucleus) filtering
        if (useTopP) {
            tempered = applyTopP(tempered);
        }
        
        // Sample from final distribution
        String selected = sampleFromDistribution(tempered);
        
        // Update statistics
        updateStatistics(tempered.size());
        
        return selected;
    }
    
    /**
     * Apply temperature scaling to probabilities
     */
    private Map<String, Double> applyTemperature(Map<String, Double> probs) {
        if (temperature == 1.0) {
            return probs;
        }
        
        Map<String, Double> scaled = new HashMap<>();
        
        // Convert to log space for numerical stability
        double maxLogProb = Double.NEGATIVE_INFINITY;
        Map<String, Double> logProbs = new HashMap<>();
        
        for (Map.Entry<String, Double> entry : probs.entrySet()) {
            double logProb = Math.log(Math.max(1e-10, entry.getValue()));
            logProbs.put(entry.getKey(), logProb);
            maxLogProb = Math.max(maxLogProb, logProb);
        }
        
        // Apply temperature and convert back
        double sum = 0;
        for (Map.Entry<String, Double> entry : logProbs.entrySet()) {
            double scaledLogProb = (entry.getValue() - maxLogProb) / temperature;
            double prob = Math.exp(scaledLogProb);
            scaled.put(entry.getKey(), prob);
            sum += prob;
        }
        
        // Normalize
        return normalize(scaled, sum);
    }
    
    /**
     * Apply top-k filtering
     */
    private Map<String, Double> applyTopK(Map<String, Double> probs) {
        if (probs.size() <= topK) {
            return probs;
        }
        
        // Sort by probability and keep top-k
        List<Map.Entry<String, Double>> sorted = probs.entrySet().stream()
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .limit(topK)
            .collect(Collectors.toList());
        
        // Renormalize
        Map<String, Double> filtered = new HashMap<>();
        double sum = 0;
        
        for (Map.Entry<String, Double> entry : sorted) {
            filtered.put(entry.getKey(), entry.getValue());
            sum += entry.getValue();
        }
        
        return normalize(filtered, sum);
    }
    
    /**
     * Apply top-p (nucleus) filtering
     */
    private Map<String, Double> applyTopP(Map<String, Double> probs) {
        // Sort by probability
        List<Map.Entry<String, Double>> sorted = probs.entrySet().stream()
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .collect(Collectors.toList());
        
        // Find minimum set with cumulative probability > p
        Map<String, Double> nucleus = new HashMap<>();
        double cumulative = 0;
        double sum = 0;
        
        for (Map.Entry<String, Double> entry : sorted) {
            nucleus.put(entry.getKey(), entry.getValue());
            cumulative += entry.getValue();
            sum += entry.getValue();
            
            if (cumulative >= topP) {
                break;
            }
        }
        
        // Ensure at least one token
        if (nucleus.isEmpty() && !sorted.isEmpty()) {
            Map.Entry<String, Double> best = sorted.get(0);
            nucleus.put(best.getKey(), best.getValue());
            sum = best.getValue();
        }
        
        return normalize(nucleus, sum);
    }
    
    /**
     * Sample from probability distribution
     */
    private String sampleFromDistribution(Map<String, Double> probs) {
        if (probs.size() == 1) {
            return probs.keySet().iterator().next();
        }
        
        double rand = random.nextDouble();
        double cumulative = 0;
        
        for (Map.Entry<String, Double> entry : probs.entrySet()) {
            cumulative += entry.getValue();
            if (rand <= cumulative) {
                return entry.getKey();
            }
        }
        
        // Fallback to highest probability
        return probs.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse(null);
    }
    
    /**
     * Beam search for sequence generation
     */
    public List<Beam> beamSearch(TokenScorer scorer, String start, 
                                 int beamWidth, int maxLength) {
        List<Beam> beams = new ArrayList<>();
        beams.add(new Beam(Arrays.asList(start), 0.0));
        
        for (int step = 0; step < maxLength; step++) {
            List<Beam> candidates = new ArrayList<>();
            
            for (Beam beam : beams) {
                if (beam.isComplete()) {
                    candidates.add(beam);
                    continue;
                }
                
                // Get next token probabilities
                Map<String, Double> probs = scorer.scoreTokens(beam.tokens);
                
                // Apply sampling methods
                Map<String, Double> filtered = applyTopK(probs);
                filtered = applyTopP(filtered);
                
                // Generate candidates
                for (Map.Entry<String, Double> entry : filtered.entrySet()) {
                    List<String> newTokens = new ArrayList<>(beam.tokens);
                    newTokens.add(entry.getKey());
                    
                    double newScore = beam.score + Math.log(entry.getValue());
                    Beam newBeam = new Beam(newTokens, newScore);
                    
                    // Check if complete
                    if (entry.getKey().equals("<END>") || 
                        entry.getKey().equals(".") && step > 5) {
                        newBeam.setComplete(true);
                    }
                    
                    candidates.add(newBeam);
                }
            }
            
            // Select top beams
            beams = candidates.stream()
                .sorted((a, b) -> Double.compare(b.score, a.score))
                .limit(beamWidth)
                .collect(Collectors.toList());
            
            // Early stopping if all beams complete
            if (beams.stream().allMatch(Beam::isComplete)) {
                break;
            }
        }
        
        return beams;
    }
    
    /**
     * Normalize probability distribution
     */
    private Map<String, Double> normalize(Map<String, Double> probs, double sum) {
        if (sum == 0) return probs;
        
        Map<String, Double> normalized = new HashMap<>();
        for (Map.Entry<String, Double> entry : probs.entrySet()) {
            normalized.put(entry.getKey(), entry.getValue() / sum);
        }
        return normalized;
    }
    
    /**
     * Update sampling statistics
     */
    private void updateStatistics(int candidates) {
        totalSamples++;
        averageCandidates = (averageCandidates * (totalSamples - 1) + candidates) / totalSamples;
    }
    
    // Configuration methods
    
    public void setTopK(int k) {
        this.topK = Math.max(1, k);
    }
    
    public void setTopP(double p) {
        this.topP = Math.max(0.1, Math.min(1.0, p));
    }
    
    public void setTemperature(double temp) {
        this.temperature = Math.max(0.1, Math.min(2.0, temp));
    }
    
    public void setUseTopK(boolean use) {
        this.useTopK = use;
    }
    
    public void setUseTopP(boolean use) {
        this.useTopP = use;
    }
    
    /**
     * Get sampling statistics
     */
    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new HashMap<>();
        
        stats.put("total_samples", totalSamples);
        stats.put("average_candidates", averageCandidates);
        stats.put("top_k", topK);
        stats.put("top_p", topP);
        stats.put("temperature", temperature);
        stats.put("use_top_k", useTopK);
        stats.put("use_top_p", useTopP);
        
        return stats;
    }
    
    /**
     * Token scorer interface for beam search
     */
    public interface TokenScorer {
        Map<String, Double> scoreTokens(List<String> context);
    }
    
    /**
     * Beam for beam search
     */
    public static class Beam {
        public final List<String> tokens;
        public final double score;
        private boolean complete;
        
        public Beam(List<String> tokens, double score) {
            this.tokens = new ArrayList<>(tokens);
            this.score = score;
            this.complete = false;
        }
        
        public boolean isComplete() {
            return complete;
        }
        
        public void setComplete(boolean complete) {
            this.complete = complete;
        }
        
        public String toString() {
            return String.join(" ", tokens) + " [" + score + "]";
        }
    }
}
