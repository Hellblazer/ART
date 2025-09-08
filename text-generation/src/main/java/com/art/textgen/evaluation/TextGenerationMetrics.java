package com.art.textgen.evaluation;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Comprehensive metrics for evaluating text generation quality
 * Implements perplexity, diversity, coherence, and fluency metrics
 */
public class TextGenerationMetrics {
    
    private final Map<String, Double> metricsCache = new HashMap<>();
    
    /**
     * Calculate perplexity of generated text against test set
     * Lower is better (typical range: 20-100)
     */
    public double calculatePerplexity(List<String> testSet, TextProbabilityModel model) {
        double totalLogProb = 0;
        int totalTokens = 0;
        
        for (String text : testSet) {
            List<String> tokens = tokenize(text);
            for (int i = 1; i < tokens.size(); i++) {
                List<String> context = tokens.subList(Math.max(0, i - 10), i);
                double prob = model.getTokenProbability(context, tokens.get(i));
                
                // Avoid log(0) by using small epsilon
                prob = Math.max(prob, 1e-10);
                totalLogProb += Math.log(prob);
                totalTokens++;
            }
        }
        
        double perplexity = Math.exp(-totalLogProb / totalTokens);
        metricsCache.put("perplexity", perplexity);
        return perplexity;
    }
    
    /**
     * Calculate BLEU score for n-gram overlap
     * Range: 0-1, higher is better
     */
    public double calculateBLEU(String generated, String reference, int maxN) {
        double bleuScore = 0.0;
        double[] weights = new double[maxN];
        Arrays.fill(weights, 1.0 / maxN);
        
        for (int n = 1; n <= maxN; n++) {
            double precision = calculateNGramPrecision(generated, reference, n);
            bleuScore += weights[n-1] * Math.log(Math.max(precision, 1e-10));
        }
        
        bleuScore = Math.exp(bleuScore);
        
        // Apply brevity penalty
        int genLength = tokenize(generated).size();
        int refLength = tokenize(reference).size();
        if (genLength < refLength) {
            double brevityPenalty = Math.exp(1 - (double) refLength / genLength);
            bleuScore *= brevityPenalty;
        }
        
        metricsCache.put("bleu", bleuScore);
        return bleuScore;
    }
    
    /**
     * Calculate diversity score based on unique n-grams
     * Range: 0-1, higher is more diverse
     */
    public double calculateDiversity(List<String> generatedTexts, int n) {
        Set<String> uniqueNGrams = new HashSet<>();
        int totalNGrams = 0;
        
        for (String text : generatedTexts) {
            List<String> nGrams = extractNGrams(text, n);
            uniqueNGrams.addAll(nGrams);
            totalNGrams += nGrams.size();
        }
        
        double diversity = totalNGrams > 0 ? 
            (double) uniqueNGrams.size() / totalNGrams : 0.0;
        
        metricsCache.put("diversity_" + n, diversity);
        return diversity;
    }
    
    /**
     * Calculate coherence score using topic consistency
     * Range: 0-1, higher is more coherent
     */
    public double calculateCoherence(String text, int windowSize) {
        List<String> sentences = splitIntoSentences(text);
        if (sentences.size() < 2) return 1.0;
        
        double totalSimilarity = 0;
        int comparisons = 0;
        
        for (int i = 0; i < sentences.size() - 1; i++) {
            for (int j = i + 1; j <= Math.min(i + windowSize, sentences.size() - 1); j++) {
                double similarity = calculateSemanticSimilarity(sentences.get(i), sentences.get(j));
                totalSimilarity += similarity;
                comparisons++;
            }
        }
        
        double coherence = comparisons > 0 ? totalSimilarity / comparisons : 0.0;
        metricsCache.put("coherence", coherence);
        return coherence;
    }
    
    /**
     * Calculate fluency using simple grammar heuristics
     * Range: 0-1, higher is more fluent
     */
    public double calculateFluency(String text) {
        double score = 1.0;
        
        // Check for basic grammar patterns
        String[] sentences = text.split("[.!?]+");
        for (String sentence : sentences) {
            sentence = sentence.trim();
            if (sentence.isEmpty()) continue;
            
            // Penalize sentences without verbs
            if (!containsVerb(sentence)) score -= 0.1;
            
            // Penalize very short or very long sentences
            int wordCount = sentence.split("\\s+").length;
            if (wordCount < 3 || wordCount > 50) score -= 0.05;
            
            // Check for repeated words
            if (hasExcessiveRepetition(sentence)) score -= 0.1;
            
            // Check capitalization
            if (!Character.isUpperCase(sentence.charAt(0))) score -= 0.05;
        }
        
        score = Math.max(0, Math.min(1, score));
        metricsCache.put("fluency", score);
        return score;
    }
    
    /**
     * Calculate readability using Flesch Reading Ease
     * Range: 0-100, higher is easier to read
     */
    public double calculateReadability(String text) {
        int totalWords = 0;
        int totalSentences = 0;
        int totalSyllables = 0;
        
        String[] sentences = text.split("[.!?]+");
        totalSentences = sentences.length;
        
        for (String sentence : sentences) {
            String[] words = sentence.trim().split("\\s+");
            totalWords += words.length;
            for (String word : words) {
                totalSyllables += countSyllables(word);
            }
        }
        
        if (totalWords == 0 || totalSentences == 0) return 0;
        
        double avgWordsPerSentence = (double) totalWords / totalSentences;
        double avgSyllablesPerWord = (double) totalSyllables / totalWords;
        
        // Flesch Reading Ease formula
        double readability = 206.835 - 1.015 * avgWordsPerSentence - 84.6 * avgSyllablesPerWord;
        readability = Math.max(0, Math.min(100, readability));
        
        metricsCache.put("readability", readability);
        return readability;
    }
    
    /**
     * Get composite quality score combining all metrics
     * Range: 0-1, higher is better
     */
    public double getCompositeScore() {
        double score = 0;
        int count = 0;
        
        // Normalize and weight different metrics
        if (metricsCache.containsKey("perplexity")) {
            // Normalize perplexity (inverse, lower is better)
            double normPerplexity = 1.0 / (1.0 + metricsCache.get("perplexity") / 50.0);
            score += normPerplexity * 0.2;
            count++;
        }
        
        if (metricsCache.containsKey("bleu")) {
            score += metricsCache.get("bleu") * 0.2;
            count++;
        }
        
        if (metricsCache.containsKey("diversity_2")) {
            score += metricsCache.get("diversity_2") * 0.2;
            count++;
        }
        
        if (metricsCache.containsKey("coherence")) {
            score += metricsCache.get("coherence") * 0.2;
            count++;
        }
        
        if (metricsCache.containsKey("fluency")) {
            score += metricsCache.get("fluency") * 0.2;
            count++;
        }
        
        return count > 0 ? score : 0;
    }
    
    /**
     * Generate detailed metrics report
     */
    public String generateReport() {
        StringBuilder report = new StringBuilder();
        report.append("\n=== Text Generation Quality Metrics ===\n");
        report.append("=".repeat(40) + "\n\n");
        
        if (metricsCache.containsKey("perplexity")) {
            report.append(String.format("Perplexity:        %.2f (lower is better)\n", 
                metricsCache.get("perplexity")));
        }
        
        if (metricsCache.containsKey("bleu")) {
            report.append(String.format("BLEU Score:        %.3f (0-1, higher is better)\n", 
                metricsCache.get("bleu")));
        }
        
        if (metricsCache.containsKey("diversity_2")) {
            report.append(String.format("Diversity (2-gram): %.3f (0-1, higher is better)\n", 
                metricsCache.get("diversity_2")));
        }
        
        if (metricsCache.containsKey("coherence")) {
            report.append(String.format("Coherence:         %.3f (0-1, higher is better)\n", 
                metricsCache.get("coherence")));
        }
        
        if (metricsCache.containsKey("fluency")) {
            report.append(String.format("Fluency:           %.3f (0-1, higher is better)\n", 
                metricsCache.get("fluency")));
        }
        
        if (metricsCache.containsKey("readability")) {
            report.append(String.format("Readability:       %.1f (0-100, higher is easier)\n", 
                metricsCache.get("readability")));
        }
        
        report.append("\n");
        report.append(String.format("Composite Score:   %.3f (0-1, higher is better)\n", 
            getCompositeScore()));
        
        return report.toString();
    }
    
    // Helper methods
    
    private List<String> tokenize(String text) {
        return Arrays.asList(text.toLowerCase().split("\\s+"));
    }
    
    private List<String> extractNGrams(String text, int n) {
        List<String> tokens = tokenize(text);
        List<String> nGrams = new ArrayList<>();
        
        for (int i = 0; i <= tokens.size() - n; i++) {
            String nGram = String.join(" ", tokens.subList(i, i + n));
            nGrams.add(nGram);
        }
        
        return nGrams;
    }
    
    private double calculateNGramPrecision(String generated, String reference, int n) {
        List<String> genNGrams = extractNGrams(generated, n);
        List<String> refNGrams = extractNGrams(reference, n);
        
        if (genNGrams.isEmpty()) return 0;
        
        int matches = 0;
        for (String nGram : genNGrams) {
            if (refNGrams.contains(nGram)) {
                matches++;
                refNGrams.remove(nGram); // Avoid double counting
            }
        }
        
        return (double) matches / genNGrams.size();
    }
    
    private List<String> splitIntoSentences(String text) {
        return Arrays.asList(text.split("[.!?]+"))
            .stream()
            .map(String::trim)
            .filter(s -> !s.isEmpty())
            .collect(Collectors.toList());
    }
    
    private double calculateSemanticSimilarity(String s1, String s2) {
        Set<String> words1 = new HashSet<>(tokenize(s1));
        Set<String> words2 = new HashSet<>(tokenize(s2));
        
        Set<String> intersection = new HashSet<>(words1);
        intersection.retainAll(words2);
        
        Set<String> union = new HashSet<>(words1);
        union.addAll(words2);
        
        return union.isEmpty() ? 0 : (double) intersection.size() / union.size();
    }
    
    private boolean containsVerb(String sentence) {
        // Simple heuristic: check for common verb patterns
        String[] commonVerbs = {"is", "are", "was", "were", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might"};
        
        String lower = sentence.toLowerCase();
        for (String verb : commonVerbs) {
            if (lower.contains(" " + verb + " ") || lower.startsWith(verb + " ")) {
                return true;
            }
        }
        
        // Check for -ing, -ed endings
        return lower.matches(".*\\b\\w+(ing|ed)\\b.*");
    }
    
    private boolean hasExcessiveRepetition(String sentence) {
        List<String> words = tokenize(sentence);
        Set<String> uniqueWords = new HashSet<>(words);
        
        // If unique words are less than 60% of total, consider it repetitive
        return words.size() > 5 && uniqueWords.size() < words.size() * 0.6;
    }
    
    private int countSyllables(String word) {
        word = word.toLowerCase().replaceAll("[^a-z]", "");
        if (word.isEmpty()) return 0;
        
        int count = 0;
        boolean previousWasVowel = false;
        
        for (char c : word.toCharArray()) {
            boolean isVowel = "aeiou".indexOf(c) >= 0;
            if (isVowel && !previousWasVowel) {
                count++;
            }
            previousWasVowel = isVowel;
        }
        
        // Adjust for silent e
        if (word.endsWith("e") && count > 1) {
            count--;
        }
        
        return Math.max(1, count);
    }
    
    /**
     * Interface for probability models used in perplexity calculation
     */
    public interface TextProbabilityModel {
        double getTokenProbability(List<String> context, String token);
    }
}
