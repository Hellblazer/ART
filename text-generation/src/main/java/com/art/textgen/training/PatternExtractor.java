package com.art.textgen.training;

import com.art.textgen.core.Vocabulary;
import com.art.textgen.generation.PatternGenerator;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Extract patterns from corpus for training
 * Includes n-grams, syntactic patterns, and semantic relationships
 */
public class PatternExtractor {
    
    private final Vocabulary vocabulary;
    private final Map<String, PatternInfo> ngramPatterns;
    private final Map<String, List<String>> syntacticPatterns;
    private final Map<String, Set<String>> semanticClusters;
    private final Map<String, Double> patternScores;
    
    public static class PatternInfo {
        public final String pattern;
        public final int frequency;
        public final Map<String, Integer> continuations;
        public final Set<String> contexts;
        public double importance;
        
        public PatternInfo(String pattern) {
            this.pattern = pattern;
            this.frequency = 1;
            this.continuations = new HashMap<>();
            this.contexts = new HashSet<>();
            this.importance = 0.0;
        }
        
        public void addOccurrence(String context, String continuation) {
            contexts.add(context);
            if (continuation != null) {
                continuations.merge(continuation, 1, Integer::sum);
            }
        }
        
        public void updateImportance(int totalPatterns) {
            // TF-IDF like scoring
            double tf = Math.log(1 + frequency);
            double idf = Math.log(totalPatterns / (1.0 + contexts.size()));
            importance = tf * idf;
        }
    }
    
    public PatternExtractor(Vocabulary vocabulary) {
        this.vocabulary = vocabulary;
        this.ngramPatterns = new HashMap<>();
        this.syntacticPatterns = new HashMap<>();
        this.semanticClusters = new HashMap<>();
        this.patternScores = new HashMap<>();
    }
    
    /**
     * Extract all patterns from sentences
     */
    public void extractPatterns(List<List<String>> sentences) {
        System.out.println("Extracting patterns from " + sentences.size() + " sentences");
        
        // Extract n-grams
        for (int n = 2; n <= 5; n++) {
            extractNGrams(sentences, n);
        }
        
        // Extract syntactic patterns
        extractSyntacticPatterns(sentences);
        
        // Extract semantic clusters
        extractSemanticClusters(sentences);
        
        // Calculate pattern importance
        calculatePatternImportance();
        
        System.out.println("Extracted " + ngramPatterns.size() + " n-gram patterns");
        System.out.println("Found " + syntacticPatterns.size() + " syntactic patterns");
        System.out.println("Created " + semanticClusters.size() + " semantic clusters");
    }
    
    /**
     * Extract n-grams from sentences
     */
    private void extractNGrams(List<List<String>> sentences, int n) {
        for (List<String> sentence : sentences) {
            if (sentence.size() < n) continue;
            
            for (int i = 0; i <= sentence.size() - n; i++) {
                List<String> ngram = sentence.subList(i, i + n);
                String pattern = String.join(" ", ngram);
                
                // Get continuation if available
                String continuation = null;
                if (i + n < sentence.size()) {
                    continuation = sentence.get(i + n);
                }
                
                // Get context (previous 2 tokens)
                String context = "";
                if (i >= 2) {
                    context = sentence.get(i - 2) + " " + sentence.get(i - 1);
                } else if (i >= 1) {
                    context = sentence.get(i - 1);
                }
                
                // Add to patterns
                PatternInfo info = ngramPatterns.computeIfAbsent(pattern, PatternInfo::new);
                info.addOccurrence(context, continuation);
            }
        }
    }
    
    /**
     * Extract syntactic patterns
     */
    private void extractSyntacticPatterns(List<List<String>> sentences) {
        for (List<String> sentence : sentences) {
            // Simple POS-like patterns
            List<String> pattern = simplifySentencePattern(sentence);
            String patternKey = String.join(" ", pattern);
            
            syntacticPatterns.computeIfAbsent(patternKey, k -> new ArrayList<>())
                .add(String.join(" ", sentence));
            
            // Extract phrase patterns
            extractPhrasePatterns(sentence);
        }
    }
    
    /**
     * Simplify sentence to pattern
     */
    private List<String> simplifySentencePattern(List<String> sentence) {
        List<String> pattern = new ArrayList<>();
        
        for (String token : sentence) {
            if (isArticle(token)) {
                pattern.add("DET");
            } else if (isPreposition(token)) {
                pattern.add("PREP");
            } else if (isConjunction(token)) {
                pattern.add("CONJ");
            } else if (isPronoun(token)) {
                pattern.add("PRON");
            } else if (isAuxiliary(token)) {
                pattern.add("AUX");
            } else if (token.endsWith("ing")) {
                pattern.add("VERB-ING");
            } else if (token.endsWith("ed")) {
                pattern.add("VERB-ED");
            } else if (token.endsWith("s") && !token.endsWith("ss")) {
                pattern.add("NOUN-PL");
            } else if (token.endsWith("ly")) {
                pattern.add("ADV");
            } else if (token.matches("[.!?]")) {
                pattern.add("PUNCT");
            } else if (Character.isUpperCase(token.charAt(0))) {
                pattern.add("PROPER");
            } else {
                pattern.add("WORD");
            }
        }
        
        return pattern;
    }
    
    /**
     * Extract phrase patterns
     */
    private void extractPhrasePatterns(List<String> sentence) {
        // Noun phrases (simple detection)
        for (int i = 0; i < sentence.size() - 1; i++) {
            String current = sentence.get(i);
            String next = sentence.get(i + 1);
            
            // Article + Noun
            if (isArticle(current) && !isVerb(next)) {
                String phrase = current + " " + next;
                int j = i + 2;
                
                // Extend phrase
                while (j < sentence.size() && !isVerb(sentence.get(j)) && 
                       !isPunctuation(sentence.get(j))) {
                    phrase += " " + sentence.get(j);
                    j++;
                }
                
                syntacticPatterns.computeIfAbsent("NP", k -> new ArrayList<>()).add(phrase);
            }
            
            // Verb phrases
            if (isVerb(current) || isAuxiliary(current)) {
                String phrase = current;
                int j = i + 1;
                
                while (j < sentence.size() && (isAdverb(sentence.get(j)) || 
                       isPreposition(sentence.get(j)))) {
                    phrase += " " + sentence.get(j);
                    j++;
                }
                
                syntacticPatterns.computeIfAbsent("VP", k -> new ArrayList<>()).add(phrase);
            }
        }
    }
    
    /**
     * Extract semantic clusters
     */
    private void extractSemanticClusters(List<List<String>> sentences) {
        // Co-occurrence based clustering
        Map<String, Map<String, Integer>> cooccurrence = new HashMap<>();
        
        for (List<String> sentence : sentences) {
            // Window-based co-occurrence
            for (int i = 0; i < sentence.size(); i++) {
                String word = sentence.get(i);
                if (isContentWord(word)) {
                    Map<String, Integer> neighbors = cooccurrence.computeIfAbsent(word, k -> new HashMap<>());
                    
                    // Look at window of ±3 words
                    for (int j = Math.max(0, i - 3); j < Math.min(sentence.size(), i + 4); j++) {
                        if (i != j && isContentWord(sentence.get(j))) {
                            neighbors.merge(sentence.get(j), 1, Integer::sum);
                        }
                    }
                }
            }
        }
        
        // Create clusters from strong co-occurrences
        for (Map.Entry<String, Map<String, Integer>> entry : cooccurrence.entrySet()) {
            String word = entry.getKey();
            Map<String, Integer> neighbors = entry.getValue();
            
            // Get top co-occurring words
            List<String> topNeighbors = neighbors.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .limit(10)
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
            
            if (!topNeighbors.isEmpty()) {
                Set<String> cluster = semanticClusters.computeIfAbsent(word, k -> new HashSet<>());
                cluster.addAll(topNeighbors);
            }
        }
    }
    
    /**
     * Calculate pattern importance scores
     */
    private void calculatePatternImportance() {
        int totalPatterns = ngramPatterns.size();
        
        for (PatternInfo info : ngramPatterns.values()) {
            info.updateImportance(totalPatterns);
            patternScores.put(info.pattern, info.importance);
        }
    }
    
    /**
     * Get top patterns by importance
     */
    public List<PatternInfo> getTopPatterns(int n) {
        return ngramPatterns.values().stream()
            .sorted((a, b) -> Double.compare(b.importance, a.importance))
            .limit(n)
            .collect(Collectors.toList());
    }
    
    /**
     * Get patterns for training
     */
    public Map<String, PatternInfo> getPatternsForTraining() {
        // Filter patterns with minimum frequency
        return ngramPatterns.entrySet().stream()
            .filter(e -> e.getValue().frequency >= 2)
            .collect(Collectors.toMap(
                Map.Entry::getKey,
                Map.Entry::getValue
            ));
    }
    
    /**
     * Get semantic associations for a word
     */
    public Set<String> getSemanticAssociations(String word) {
        return semanticClusters.getOrDefault(word, new HashSet<>());
    }
    
    /**
     * Export patterns for PatternGenerator
     */
    public void exportToPatternGenerator(PatternGenerator generator) {
        // Add n-gram patterns
        for (PatternInfo info : ngramPatterns.values()) {
            if (info.frequency >= 2) {
                // Learn the pattern
                List<String> tokens = Arrays.asList(info.pattern.split(" "));
                generator.learnPattern(tokens);
                
                // Add continuations
                for (Map.Entry<String, Integer> cont : info.continuations.entrySet()) {
                    List<String> extended = new ArrayList<>(tokens);
                    extended.add(cont.getKey());
                    generator.learnPattern(extended);
                }
            }
        }
    }
    
    // Helper methods for token classification
    private boolean isArticle(String token) {
        return Arrays.asList("a", "an", "the").contains(token.toLowerCase());
    }
    
    private boolean isPreposition(String token) {
        return Arrays.asList("in", "on", "at", "to", "for", "with", "by", "from", 
                           "of", "about", "through", "between", "under", "over")
            .contains(token.toLowerCase());
    }
    
    private boolean isConjunction(String token) {
        return Arrays.asList("and", "or", "but", "yet", "so", "nor", "for")
            .contains(token.toLowerCase());
    }
    
    private boolean isPronoun(String token) {
        return Arrays.asList("i", "you", "he", "she", "it", "we", "they", 
                           "me", "him", "her", "us", "them", "my", "your", 
                           "his", "her", "its", "our", "their")
            .contains(token.toLowerCase());
    }
    
    private boolean isAuxiliary(String token) {
        return Arrays.asList("is", "are", "was", "were", "be", "been", "being",
                           "have", "has", "had", "do", "does", "did", 
                           "will", "would", "could", "should", "may", "might", "must", "can")
            .contains(token.toLowerCase());
    }
    
    private boolean isVerb(String token) {
        return token.endsWith("ing") || token.endsWith("ed") || 
               isAuxiliary(token) || token.endsWith("s");
    }
    
    private boolean isAdverb(String token) {
        return token.endsWith("ly");
    }
    
    private boolean isPunctuation(String token) {
        return token.matches("[.!?,;:]");
    }
    
    private boolean isContentWord(String word) {
        return word.length() > 2 && 
               !isArticle(word) && 
               !isPreposition(word) && 
               !isConjunction(word) && 
               !isPronoun(word) && 
               !isAuxiliary(word) && 
               !isPunctuation(word);
    }
    
    /**
     * Get statistics
     */
    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new HashMap<>();
        
        stats.put("total_ngram_patterns", ngramPatterns.size());
        stats.put("unique_syntactic_patterns", syntacticPatterns.size());
        stats.put("semantic_clusters", semanticClusters.size());
        
        // Pattern frequency distribution
        Map<Integer, Long> freqDist = ngramPatterns.values().stream()
            .collect(Collectors.groupingBy(
                p -> p.frequency,
                Collectors.counting()
            ));
        stats.put("frequency_distribution", freqDist);
        
        // Top patterns
        stats.put("top_10_patterns", getTopPatterns(10).stream()
            .map(p -> p.pattern)
            .collect(Collectors.toList()));
        
        return stats;
    }
}
