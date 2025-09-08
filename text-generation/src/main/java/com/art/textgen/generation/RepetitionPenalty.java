package com.art.textgen.generation;

import java.util.*;

/**
 * Repetition penalty system to prevent redundant text generation
 * Tracks recent tokens and n-grams to penalize repetition
 */
public class RepetitionPenalty {
    
    // Token-level tracking
    private final LinkedList<String> recentTokens;
    private final Map<String, Integer> tokenDistances;
    private final int tokenWindowSize;
    
    // N-gram tracking
    private final Map<String, Integer> bigramCounts;
    private final Map<String, Integer> trigramCounts;
    private final Map<String, Double> penaltyCache;
    
    // Penalty parameters
    private double tokenPenaltyBase = 0.95;
    private double tokenPenaltyDecay = 0.05;
    private double ngramPenaltyFactor = 0.8;
    private double rarityBonus = 1.1;
    
    public RepetitionPenalty(int windowSize) {
        this.tokenWindowSize = windowSize;
        this.recentTokens = new LinkedList<>();
        this.tokenDistances = new HashMap<>();
        this.bigramCounts = new HashMap<>();
        this.trigramCounts = new HashMap<>();
        this.penaltyCache = new HashMap<>();
    }
    
    /**
     * Apply penalty to token probabilities based on repetition
     */
    public Map<String, Double> applyPenalty(Map<String, Double> tokenProbabilities, 
                                           List<String> context) {
        Map<String, Double> adjusted = new HashMap<>();
        
        for (Map.Entry<String, Double> entry : tokenProbabilities.entrySet()) {
            String token = entry.getKey();
            double prob = entry.getValue();
            
            // Calculate combined penalty
            double penalty = calculatePenalty(token, context);
            
            // Apply penalty
            adjusted.put(token, prob * penalty);
        }
        
        // Renormalize probabilities
        return normalize(adjusted);
    }
    
    /**
     * Calculate penalty for a specific token
     */
    private double calculatePenalty(String token, List<String> context) {
        double penalty = 1.0;
        
        // 1. Token-level repetition penalty
        if (tokenDistances.containsKey(token)) {
            int distance = tokenDistances.get(token);
            // Stronger penalty for more recent repetitions
            double tokenPenalty = tokenPenaltyBase - (tokenPenaltyDecay * distance);
            penalty *= Math.max(0.3, tokenPenalty);
        }
        
        // 2. Bigram repetition penalty
        if (context.size() >= 1) {
            String bigram = context.get(context.size() - 1) + " " + token;
            int bigramCount = bigramCounts.getOrDefault(bigram, 0);
            if (bigramCount > 0) {
                penalty *= Math.pow(ngramPenaltyFactor, bigramCount);
            }
        }
        
        // 3. Trigram repetition penalty (stronger)
        if (context.size() >= 2) {
            String trigram = context.get(context.size() - 2) + " " + 
                           context.get(context.size() - 1) + " " + token;
            int trigramCount = trigramCounts.getOrDefault(trigram, 0);
            if (trigramCount > 0) {
                penalty *= Math.pow(ngramPenaltyFactor * 0.7, trigramCount);
            }
        }
        
        // 4. Rarity bonus (encourage less common tokens)
        if (!tokenDistances.containsKey(token) && !isCommonWord(token)) {
            penalty *= rarityBonus;
        }
        
        return penalty;
    }
    
    /**
     * Update tracking after token generation
     */
    public void updateHistory(String token, List<String> context) {
        // Update recent tokens
        recentTokens.addLast(token);
        if (recentTokens.size() > tokenWindowSize) {
            String removed = recentTokens.removeFirst();
            tokenDistances.remove(removed);
        }
        
        // Update token distances
        updateTokenDistances();
        
        // Update n-gram counts
        if (context.size() >= 1) {
            String bigram = context.get(context.size() - 1) + " " + token;
            bigramCounts.merge(bigram, 1, Integer::sum);
        }
        
        if (context.size() >= 2) {
            String trigram = context.get(context.size() - 2) + " " + 
                           context.get(context.size() - 1) + " " + token;
            trigramCounts.merge(trigram, 1, Integer::sum);
        }
        
        // Clear cache periodically
        if (penaltyCache.size() > 1000) {
            penaltyCache.clear();
        }
    }
    
    /**
     * Update distance map for all recent tokens
     */
    private void updateTokenDistances() {
        tokenDistances.clear();
        int distance = 0;
        
        // Iterate from most recent to oldest
        Iterator<String> iter = recentTokens.descendingIterator();
        while (iter.hasNext()) {
            String token = iter.next();
            if (!tokenDistances.containsKey(token)) {
                tokenDistances.put(token, distance);
            }
            distance++;
        }
    }
    
    /**
     * Check if token is a common word (less penalty for function words)
     */
    private boolean isCommonWord(String token) {
        Set<String> common = Set.of(
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "as", "is", "was", "are",
            "were", "be", "been", "being", "have", "has", "had", "do",
            "does", "did", "will", "would", "could", "should", "may",
            "might", "must", "can", "shall", ".", ",", "!", "?", ";", ":"
        );
        return common.contains(token.toLowerCase());
    }
    
    /**
     * Normalize probability distribution
     */
    private Map<String, Double> normalize(Map<String, Double> probs) {
        double sum = probs.values().stream().mapToDouble(Double::doubleValue).sum();
        
        if (sum == 0) return probs;
        
        Map<String, Double> normalized = new HashMap<>();
        for (Map.Entry<String, Double> entry : probs.entrySet()) {
            normalized.put(entry.getKey(), entry.getValue() / sum);
        }
        
        return normalized;
    }
    
    /**
     * Reset history (for new generation sessions)
     */
    public void reset() {
        recentTokens.clear();
        tokenDistances.clear();
        bigramCounts.clear();
        trigramCounts.clear();
        penaltyCache.clear();
    }
    
    /**
     * Configure penalty parameters
     */
    public void setParameters(double tokenPenalty, double ngramPenalty, double rarity) {
        this.tokenPenaltyBase = Math.max(0.1, Math.min(1.0, tokenPenalty));
        this.ngramPenaltyFactor = Math.max(0.1, Math.min(1.0, ngramPenalty));
        this.rarityBonus = Math.max(1.0, Math.min(2.0, rarity));
    }
    
    /**
     * Get statistics about repetition
     */
    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new HashMap<>();
        
        stats.put("recent_tokens", recentTokens.size());
        stats.put("unique_recent", new HashSet<>(recentTokens).size());
        stats.put("bigram_patterns", bigramCounts.size());
        stats.put("trigram_patterns", trigramCounts.size());
        
        // Calculate repetition rate
        double repetitionRate = 1.0 - (new HashSet<>(recentTokens).size() / 
                                      (double) Math.max(1, recentTokens.size()));
        stats.put("repetition_rate", repetitionRate);
        
        // Most repeated patterns
        stats.put("top_repeated_bigrams", getTopPatterns(bigramCounts, 5));
        stats.put("top_repeated_trigrams", getTopPatterns(trigramCounts, 5));
        
        return stats;
    }
    
    /**
     * Get top repeated patterns
     */
    private List<String> getTopPatterns(Map<String, Integer> counts, int n) {
        return counts.entrySet().stream()
            .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
            .limit(n)
            .map(e -> e.getKey() + " (" + e.getValue() + "x)")
            .toList();
    }
}
