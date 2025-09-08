package com.art.textgen.generation;

import com.art.textgen.core.Vocabulary;
import com.art.textgen.dynamics.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Pattern-based text generator using ART resonance
 * Generates text by finding resonant patterns and continuing them
 */
public class PatternGenerator {
    
    protected final Vocabulary vocabulary;  // Changed from private to protected
    private final ResonanceDetector resonance;
    private final Map<String, List<PatternSequence>> patternBank;
    private final Random random;
    private double temperature;
    
    public static class PatternSequence {
        public final List<String> tokens;
        public final double frequency;
        public final Map<String, Double> continuations;
        public double resonanceStrength;
        
        public PatternSequence(List<String> tokens) {
            this.tokens = new ArrayList<>(tokens);
            this.frequency = 1.0;
            this.continuations = new HashMap<>();
            this.resonanceStrength = 0.0;
        }
        
        public void addContinuation(String next, double weight) {
            continuations.merge(next, weight, Double::sum);
        }
        
        public String selectContinuation(Random random, double temperature) {
            if (continuations.isEmpty()) return null;
            
            // Apply temperature to probabilities
            Map<String, Double> adjusted = new HashMap<>();
            double sum = 0.0;
            
            for (Map.Entry<String, Double> entry : continuations.entrySet()) {
                double score = Math.pow(entry.getValue(), 1.0 / temperature);
                adjusted.put(entry.getKey(), score);
                sum += score;
            }
            
            // Normalize and sample
            double rand = random.nextDouble() * sum;
            double cumulative = 0.0;
            
            for (Map.Entry<String, Double> entry : adjusted.entrySet()) {
                cumulative += entry.getValue();
                if (rand <= cumulative) {
                    return entry.getKey();
                }
            }
            
            // Fallback to most likely
            return continuations.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(null);
        }
    }
    
    public PatternGenerator(Vocabulary vocabulary, double temperature) {
        this.vocabulary = vocabulary;
        this.resonance = new ResonanceDetector(0.6); // Lower vigilance for more matches
        this.patternBank = new HashMap<>();
        this.random = new Random();
        this.temperature = temperature;
        
        // Initialize with seed patterns
        initializeSeedPatterns();
    }
    
    /**
     * Initialize seed patterns for generation
     */
    private void initializeSeedPatterns() {
        // Common linguistic patterns
        addPattern("the", "future", "of");
        addPattern("artificial", "intelligence", "is");
        addPattern("understanding", "consciousness", "requires");
        addPattern("once", "upon", "a", "time");
        addPattern("in", "the", "beginning");
        addPattern("the", "key", "to");
        
        // Grammatical patterns
        addPattern("is", "a");
        addPattern("can", "be");
        addPattern("will", "be");
        addPattern("has", "been");
        addPattern("are", "the");
        
        // Conceptual patterns
        addPattern("machine", "learning", "algorithms");
        addPattern("neural", "networks", "can");
        addPattern("cognitive", "processes", "involve");
        addPattern("human", "intelligence", "differs");
        addPattern("computational", "models", "suggest");
        
        // Story patterns
        addPattern("there", "was", "a");
        addPattern("long", "ago", "in");
        addPattern("the", "story", "begins");
        addPattern("it", "was", "a");
        
        // Build continuation probabilities
        buildContinuationProbabilities();
    }
    
    /**
     * Add pattern to bank
     */
    private void addPattern(String... tokens) {
        String key = String.join(" ", tokens);
        PatternSequence pattern = new PatternSequence(Arrays.asList(tokens));
        patternBank.computeIfAbsent(key, k -> new ArrayList<>()).add(pattern);
        
        // Add to vocabulary
        for (String token : tokens) {
            vocabulary.addToken(token);
        }
    }
    
    /**
     * Build continuation probabilities from patterns
     */
    private void buildContinuationProbabilities() {
        // Create bigram/trigram continuations
        Map<String, Map<String, Integer>> bigramCounts = new HashMap<>();
        Map<String, Map<String, Integer>> trigramCounts = new HashMap<>();
        
        // Collect from all patterns
        for (List<PatternSequence> patterns : patternBank.values()) {
            for (PatternSequence pattern : patterns) {
                List<String> tokens = pattern.tokens;
                
                // Build bigrams
                for (int i = 0; i < tokens.size() - 1; i++) {
                    String current = tokens.get(i);
                    String next = tokens.get(i + 1);
                    
                    bigramCounts.computeIfAbsent(current, k -> new HashMap<>())
                        .merge(next, 1, Integer::sum);
                }
                
                // Build trigrams
                for (int i = 0; i < tokens.size() - 2; i++) {
                    String bigram = tokens.get(i) + " " + tokens.get(i + 1);
                    String next = tokens.get(i + 2);
                    
                    trigramCounts.computeIfAbsent(bigram, k -> new HashMap<>())
                        .merge(next, 1, Integer::sum);
                }
            }
        }
        
        // Add common continuations
        addContinuations(bigramCounts);
        addContinuations(trigramCounts);
    }
    
    /**
     * Add continuation probabilities to patterns
     */
    private void addContinuations(Map<String, Map<String, Integer>> counts) {
        for (Map.Entry<String, Map<String, Integer>> entry : counts.entrySet()) {
            String context = entry.getKey();
            Map<String, Integer> continuations = entry.getValue();
            
            // Find matching patterns
            for (List<PatternSequence> patterns : patternBank.values()) {
                for (PatternSequence pattern : patterns) {
                    String patternStr = String.join(" ", pattern.tokens);
                    
                    if (patternStr.endsWith(context)) {
                        // Add continuations
                        int total = continuations.values().stream().mapToInt(Integer::intValue).sum();
                        
                        for (Map.Entry<String, Integer> cont : continuations.entrySet()) {
                            double prob = cont.getValue() / (double) total;
                            pattern.addContinuation(cont.getKey(), prob);
                        }
                    }
                }
            }
        }
        
        // Add default continuations
        addDefaultContinuations();
    }
    
    /**
     * Add default continuations for common patterns
     */
    private void addDefaultContinuations() {
        // Articles
        addDefaultContinuation("the", "future", "past", "present", "world", "system", "mind");
        addDefaultContinuation("a", "new", "simple", "complex", "different", "unique", "special");
        
        // Verbs
        addDefaultContinuation("is", "not", "always", "often", "sometimes", "never", "truly");
        addDefaultContinuation("can", "be", "help", "create", "understand", "learn", "process");
        
        // Prepositions
        addDefaultContinuation("of", "the", "human", "artificial", "natural", "complex", "simple");
        addDefaultContinuation("in", "the", "a", "this", "our", "their", "its");
        
        // Conceptual
        addDefaultContinuation("intelligence", "is", "can", "will", "should", "might", "could");
        addDefaultContinuation("consciousness", "is", "emerges", "arises", "exists", "develops");
        addDefaultContinuation("understanding", "requires", "involves", "needs", "demands", "means");
    }
    
    /**
     * Add default continuations for a token
     */
    private void addDefaultContinuation(String token, String... continuations) {
        for (List<PatternSequence> patterns : patternBank.values()) {
            for (PatternSequence pattern : patterns) {
                if (pattern.tokens.get(pattern.tokens.size() - 1).equals(token)) {
                    double weight = 1.0 / continuations.length;
                    for (String cont : continuations) {
                        pattern.addContinuation(cont, weight);
                        vocabulary.addToken(cont);
                    }
                }
            }
        }
    }
    
    /**
     * Generate next token based on context
     */
    public String generateNext(List<String> context) {
        if (context.isEmpty()) {
            return vocabulary.START_TOKEN;
        }
        
        // Find resonant patterns
        List<PatternSequence> resonantPatterns = findResonantPatterns(context);
        
        if (resonantPatterns.isEmpty()) {
            // No resonant patterns - use semantic associations
            return generateSemanticNext(context);
        }
        
        // Combine predictions from resonant patterns
        Map<String, Double> combinedPredictions = new HashMap<>();
        double totalResonance = 0.0;
        
        for (PatternSequence pattern : resonantPatterns) {
            double resonance = pattern.resonanceStrength;
            totalResonance += resonance;
            
            // Get continuation from this pattern
            String continuation = pattern.selectContinuation(random, temperature);
            
            if (continuation != null) {
                combinedPredictions.merge(continuation, resonance, Double::sum);
            }
        }
        
        // Normalize and select
        if (!combinedPredictions.isEmpty()) {
            return selectFromDistribution(combinedPredictions, totalResonance);
        }
        
        // Fallback to semantic generation
        return generateSemanticNext(context);
    }
    
    /**
     * Find patterns that resonate with context
     */
    private List<PatternSequence> findResonantPatterns(List<String> context) {
        List<PatternSequence> resonantPatterns = new ArrayList<>();
        
        // Convert context to embedding
        double[] contextEmbedding = computeContextEmbedding(context);
        
        // Test each pattern for resonance
        for (List<PatternSequence> patterns : patternBank.values()) {
            for (PatternSequence pattern : patterns) {
                double[] patternEmbedding = computePatternEmbedding(pattern);
                
                // Check resonance
                ResonanceDetector.ResonanceState state = 
                    resonance.searchResonance(combineEmbeddings(contextEmbedding, patternEmbedding));
                
                if (state.isResonant) {
                    pattern.resonanceStrength = state.resonanceStrength;
                    resonantPatterns.add(pattern);
                }
            }
        }
        
        // Sort by resonance strength
        resonantPatterns.sort((a, b) -> 
            Double.compare(b.resonanceStrength, a.resonanceStrength));
        
        // Return top patterns
        return resonantPatterns.stream()
            .limit(5)
            .collect(Collectors.toList());
    }
    
    /**
     * Compute context embedding
     */
    private double[] computeContextEmbedding(List<String> context) {
        int dim = vocabulary.getEmbedding(vocabulary.UNK_TOKEN).length;
        double[] embedding = new double[dim];
        
        // Weighted average with recency bias
        for (int i = 0; i < context.size(); i++) {
            String token = context.get(i);
            double[] tokenEmb = vocabulary.getEmbedding(token);
            double weight = Math.exp(-0.1 * (context.size() - i - 1)); // Recency weight
            
            for (int j = 0; j < dim; j++) {
                embedding[j] += weight * tokenEmb[j];
            }
        }
        
        return normalize(embedding);
    }
    
    /**
     * Compute pattern embedding
     */
    private double[] computePatternEmbedding(PatternSequence pattern) {
        int dim = vocabulary.getEmbedding(vocabulary.UNK_TOKEN).length;
        double[] embedding = new double[dim];
        
        // Average embeddings of pattern tokens
        for (String token : pattern.tokens) {
            double[] tokenEmb = vocabulary.getEmbedding(token);
            for (int j = 0; j < dim; j++) {
                embedding[j] += tokenEmb[j];
            }
        }
        
        // Normalize by pattern length
        for (int j = 0; j < dim; j++) {
            embedding[j] /= pattern.tokens.size();
        }
        
        return normalize(embedding);
    }
    
    /**
     * Combine embeddings for resonance check
     */
    private double[] combineEmbeddings(double[] a, double[] b) {
        double[] combined = new double[Math.max(a.length, b.length)];
        
        for (int i = 0; i < combined.length; i++) {
            if (i < a.length && i < b.length) {
                combined[i] = (a[i] + b[i]) / 2.0;
            } else if (i < a.length) {
                combined[i] = a[i];
            } else {
                combined[i] = b[i];
            }
        }
        
        return combined;
    }
    
    /**
     * Generate semantically coherent next token
     */
    private String generateSemanticNext(List<String> context) {
        if (context.isEmpty()) {
            return vocabulary.START_TOKEN;
        }
        
        // Get last few tokens
        List<String> recentContext = context.subList(
            Math.max(0, context.size() - 3), context.size());
        
        // Find semantic neighbors
        Map<String, Double> candidates = new HashMap<>();
        
        for (String contextToken : recentContext) {
            List<String> neighbors = vocabulary.getSemanticNeighbors(contextToken, 5);
            
            for (int i = 0; i < neighbors.size(); i++) {
                String neighbor = neighbors.get(i);
                double weight = 1.0 / (i + 1); // Distance-based weight
                candidates.merge(neighbor, weight, Double::sum);
            }
        }
        
        // Add some randomness
        if (!candidates.isEmpty()) {
            return selectFromDistribution(candidates, 
                candidates.values().stream().mapToDouble(Double::doubleValue).sum());
        }
        
        // Absolute fallback - random token
        Set<String> allTokens = vocabulary.getAllTokens();
        List<String> tokenList = new ArrayList<>(allTokens);
        return tokenList.get(random.nextInt(tokenList.size()));
    }
    
    /**
     * Select token from probability distribution
     */
    private String selectFromDistribution(Map<String, Double> distribution, double sum) {
        if (distribution.isEmpty()) return null;
        
        double rand = random.nextDouble() * sum;
        double cumulative = 0.0;
        
        for (Map.Entry<String, Double> entry : distribution.entrySet()) {
            cumulative += entry.getValue();
            if (rand <= cumulative) {
                return entry.getKey();
            }
        }
        
        // Fallback to highest probability
        return distribution.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse(null);
    }
    
    /**
     * Normalize vector
     */
    private double[] normalize(double[] vector) {
        double norm = 0.0;
        for (double v : vector) {
            norm += v * v;
        }
        
        if (norm == 0) return vector;
        
        norm = Math.sqrt(norm);
        double[] normalized = new double[vector.length];
        for (int i = 0; i < vector.length; i++) {
            normalized[i] = vector[i] / norm;
        }
        
        return normalized;
    }
    
    /**
     * Learn from generated sequence
     */
    public void learnPattern(List<String> sequence) {
        // Extract n-grams and add to pattern bank
        for (int n = 2; n <= 4; n++) {
            for (int i = 0; i <= sequence.size() - n; i++) {
                List<String> ngram = sequence.subList(i, i + n);
                String key = String.join(" ", ngram);
                
                PatternSequence pattern = new PatternSequence(ngram);
                patternBank.computeIfAbsent(key, k -> new ArrayList<>()).add(pattern);
                
                // Add continuation if available
                if (i + n < sequence.size()) {
                    pattern.addContinuation(sequence.get(i + n), 1.0);
                }
            }
        }
    }
    
    /**
     * Get current temperature setting
     */
    public double getTemperature() {
        return temperature;
    }
    
    /**
     * Set temperature for generation
     */
    public void setTemperature(double temperature) {
        if (temperature < 0.1) temperature = 0.1;
        if (temperature > 2.0) temperature = 2.0;
        this.temperature = temperature;
    }
    
    /**
     * Get statistics about the pattern generator
     */
    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new HashMap<>();
        
        int totalPatterns = 0;
        int maxLength = 0;
        Set<String> uniqueTokens = new HashSet<>();
        
        for (List<PatternSequence> patterns : patternBank.values()) {
            totalPatterns += patterns.size();
            for (PatternSequence pattern : patterns) {
                maxLength = Math.max(maxLength, pattern.tokens.size());
                uniqueTokens.addAll(pattern.tokens);
            }
        }
        
        stats.put("total_patterns", totalPatterns);
        stats.put("max_length", maxLength);
        stats.put("unique_tokens", uniqueTokens.size());
        stats.put("temperature", temperature);
        
        return stats;
    }
}
