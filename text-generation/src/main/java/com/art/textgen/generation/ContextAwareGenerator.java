package com.art.textgen.generation;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Context-aware text generation system
 * Tracks topic keywords, maintains discourse coherence, and preserves stylistic consistency
 * Based on Phase 4.3 of EXECUTION_PLAN.md
 */
public class ContextAwareGenerator {
    
    // Topic tracking
    private final Map<String, Double> topicKeywords;
    private final Queue<String> recentKeywords;
    private final int keywordWindowSize = 50;
    
    // Discourse coherence
    private final List<String> discourseMarkers;
    private String currentDiscourseState;
    private final Map<String, List<String>> discourseTransitions;
    
    // Style consistency
    private StyleProfile currentStyle;
    private final Map<String, StyleProfile> styleProfiles;
    
    // Context memory
    private final List<String> generatedSentences;
    private final Map<String, Integer> entityMentions;
    private final Set<String> introducedConcepts;
    
    // Coherence parameters
    private double topicDriftPenalty = 0.3;
    private double styleConsistencyWeight = 0.5;
    private double discourseCoherenceWeight = 0.4;
    
    public ContextAwareGenerator() {
        this.topicKeywords = new HashMap<>();
        this.recentKeywords = new LinkedList<>();
        this.discourseMarkers = initializeDiscourseMarkers();
        this.discourseTransitions = initializeDiscourseTransitions();
        this.currentDiscourseState = "introduction";
        
        this.styleProfiles = initializeStyleProfiles();
        this.currentStyle = styleProfiles.get("neutral");
        
        this.generatedSentences = new ArrayList<>();
        this.entityMentions = new HashMap<>();
        this.introducedConcepts = new HashSet<>();
    }
    
    /**
     * Generate text with context awareness
     */
    public String generateWithContext(String prompt, int maxLength, 
                                     TokenGenerator baseGenerator) {
        // Extract initial context from prompt
        initializeContext(prompt);
        
        StringBuilder generated = new StringBuilder();
        String currentToken = "";
        int tokenCount = 0;
        
        while (tokenCount < maxLength && !isEndToken(currentToken)) {
            // Get base probabilities from underlying generator
            double[] baseProbabilities = baseGenerator.getTokenProbabilities(
                prompt + " " + generated.toString()
            );
            
            // Apply context-aware adjustments
            double[] adjustedProbabilities = applyContextAdjustments(
                baseProbabilities,
                generated.toString()
            );
            
            // Sample next token
            int tokenIdx = sampleToken(adjustedProbabilities);
            currentToken = baseGenerator.indexToToken(tokenIdx);
            
            // Update context
            updateContext(currentToken);
            
            generated.append(currentToken).append(" ");
            tokenCount++;
            
            // Check for sentence completion
            if (isSentenceEnd(currentToken)) {
                generatedSentences.add(getCurrentSentence(generated.toString()));
                updateDiscourseState();
            }
        }
        
        return generated.toString().trim();
    }
    
    /**
     * Apply context-aware adjustments to token probabilities
     */
    private double[] applyContextAdjustments(double[] baseProbabilities, 
                                            String generatedSoFar) {
        double[] adjusted = baseProbabilities.clone();
        
        // Apply topic coherence adjustment
        adjusted = applyTopicCoherence(adjusted, generatedSoFar);
        
        // Apply discourse coherence adjustment
        adjusted = applyDiscourseCoherence(adjusted, generatedSoFar);
        
        // Apply style consistency adjustment
        adjusted = applyStyleConsistency(adjusted, generatedSoFar);
        
        // Apply entity tracking adjustment
        adjusted = applyEntityTracking(adjusted, generatedSoFar);
        
        // Renormalize probabilities
        return normalize(adjusted);
    }
    
    /**
     * Apply topic coherence to maintain focus on current topics
     */
    private double[] applyTopicCoherence(double[] probabilities, String context) {
        double[] adjusted = probabilities.clone();
        
        // Boost probabilities for tokens related to current topics
        for (String keyword : topicKeywords.keySet()) {
            double relevance = topicKeywords.get(keyword);
            
            // Find tokens related to this keyword
            List<Integer> relatedIndices = findRelatedTokenIndices(keyword);
            
            for (int idx : relatedIndices) {
                // Boost probability based on topic relevance
                adjusted[idx] *= (1 + relevance * (1 - topicDriftPenalty));
            }
        }
        
        return adjusted;
    }
    
    /**
     * Apply discourse coherence for logical flow
     */
    private double[] applyDiscourseCoherence(double[] probabilities, String context) {
        double[] adjusted = probabilities.clone();
        
        // Get appropriate discourse transitions for current state
        List<String> validTransitions = discourseTransitions.get(currentDiscourseState);
        
        if (validTransitions != null) {
            for (String transition : validTransitions) {
                int tokenIdx = getTokenIndex(transition);
                if (tokenIdx >= 0 && tokenIdx < adjusted.length) {
                    // Boost probability for coherent discourse markers
                    adjusted[tokenIdx] *= (1 + discourseCoherenceWeight);
                }
            }
        }
        
        // Penalize inappropriate discourse markers
        for (String marker : discourseMarkers) {
            if (!validTransitions.contains(marker)) {
                int tokenIdx = getTokenIndex(marker);
                if (tokenIdx >= 0 && tokenIdx < adjusted.length) {
                    adjusted[tokenIdx] *= (1 - discourseCoherenceWeight * 0.5);
                }
            }
        }
        
        return adjusted;
    }
    
    /**
     * Apply style consistency to maintain writing style
     */
    private double[] applyStyleConsistency(double[] probabilities, String context) {
        double[] adjusted = probabilities.clone();
        
        // Adjust based on current style profile
        if (currentStyle != null) {
            // Boost formal/informal words based on style
            for (int i = 0; i < adjusted.length; i++) {
                String token = indexToToken(i);
                
                if (currentStyle.formalWords.contains(token)) {
                    adjusted[i] *= (1 + currentStyle.formalityLevel * styleConsistencyWeight);
                } else if (currentStyle.informalWords.contains(token)) {
                    adjusted[i] *= (1 + (1 - currentStyle.formalityLevel) * styleConsistencyWeight);
                }
                
                // Adjust for sentence length preference
                if (isSentenceEnd(token)) {
                    int currentLength = context.split("\\s+").length;
                    double lengthDiff = Math.abs(currentLength - currentStyle.avgSentenceLength);
                    adjusted[i] *= Math.exp(-lengthDiff / 10);
                }
            }
        }
        
        return adjusted;
    }
    
    /**
     * Apply entity tracking to maintain consistency
     */
    private double[] applyEntityTracking(double[] probabilities, String context) {
        double[] adjusted = probabilities.clone();
        
        // Track entity mentions and boost pronouns appropriately
        for (Map.Entry<String, Integer> entity : entityMentions.entrySet()) {
            if (entity.getValue() > 0 && shouldUsePronoun(entity.getKey(), context)) {
                // Boost pronoun probabilities
                List<String> pronouns = getPronouns(entity.getKey());
                for (String pronoun : pronouns) {
                    int idx = getTokenIndex(pronoun);
                    if (idx >= 0 && idx < adjusted.length) {
                        adjusted[idx] *= 1.5;
                    }
                }
                
                // Slightly penalize repeating the entity name
                int entityIdx = getTokenIndex(entity.getKey());
                if (entityIdx >= 0 && entityIdx < adjusted.length) {
                    adjusted[entityIdx] *= 0.7;
                }
            }
        }
        
        return adjusted;
    }
    
    /**
     * Initialize context from prompt
     */
    private void initializeContext(String prompt) {
        // Extract keywords from prompt
        String[] words = prompt.toLowerCase().split("\\s+");
        for (String word : words) {
            if (isContentWord(word)) {
                topicKeywords.put(word, 1.0);
                recentKeywords.offer(word);
            }
        }
        
        // Detect initial style
        detectStyle(prompt);
        
        // Set initial discourse state
        currentDiscourseState = "introduction";
        
        // Extract entities
        extractEntities(prompt);
    }
    
    /**
     * Update context after generating a token
     */
    private void updateContext(String token) {
        // Update keyword tracking
        if (isContentWord(token)) {
            recentKeywords.offer(token);
            if (recentKeywords.size() > keywordWindowSize) {
                String old = recentKeywords.poll();
                // Decay old keyword relevance
                if (topicKeywords.containsKey(old)) {
                    topicKeywords.put(old, topicKeywords.get(old) * 0.9);
                }
            }
            
            // Update or add keyword
            topicKeywords.merge(token.toLowerCase(), 0.5, (old, new_) -> old + new_);
        }
        
        // Update entity tracking
        if (isEntity(token)) {
            entityMentions.merge(token, 1, Integer::sum);
        }
        
        // Track introduced concepts
        if (isNewConcept(token)) {
            introducedConcepts.add(token);
        }
    }
    
    /**
     * Update discourse state after completing a sentence
     */
    private void updateDiscourseState() {
        // Simple state machine for discourse progression
        switch (currentDiscourseState) {
            case "introduction":
                if (generatedSentences.size() >= 2) {
                    currentDiscourseState = "development";
                }
                break;
            case "development":
                if (generatedSentences.size() >= 5) {
                    currentDiscourseState = "elaboration";
                }
                break;
            case "elaboration":
                if (generatedSentences.size() >= 8) {
                    currentDiscourseState = "conclusion";
                }
                break;
            case "conclusion":
                // Stay in conclusion state
                break;
        }
    }
    
    /**
     * Detect writing style from text
     */
    private void detectStyle(String text) {
        // Simple style detection based on vocabulary and structure
        int formalCount = 0;
        int informalCount = 0;
        
        String[] words = text.toLowerCase().split("\\s+");
        for (String word : words) {
            if (isFormalWord(word)) formalCount++;
            if (isInformalWord(word)) informalCount++;
        }
        
        if (formalCount > informalCount * 2) {
            currentStyle = styleProfiles.get("formal");
        } else if (informalCount > formalCount * 2) {
            currentStyle = styleProfiles.get("informal");
        } else {
            currentStyle = styleProfiles.get("neutral");
        }
    }
    
    /**
     * Extract entities from text
     */
    private void extractEntities(String text) {
        // Simple named entity detection (would use NER in production)
        String[] words = text.split("\\s+");
        for (String word : words) {
            if (Character.isUpperCase(word.charAt(0)) && word.length() > 1) {
                entityMentions.put(word, 0);
            }
        }
    }
    
    /**
     * Initialize discourse markers
     */
    private List<String> initializeDiscourseMarkers() {
        return Arrays.asList(
            "however", "therefore", "moreover", "furthermore",
            "nevertheless", "consequently", "additionally", "specifically",
            "for example", "in contrast", "similarly", "finally",
            "firstly", "secondly", "in conclusion", "to summarize"
        );
    }
    
    /**
     * Initialize discourse transitions
     */
    private Map<String, List<String>> initializeDiscourseTransitions() {
        Map<String, List<String>> transitions = new HashMap<>();
        
        transitions.put("introduction", Arrays.asList(
            "firstly", "to begin", "initially", "importantly"
        ));
        
        transitions.put("development", Arrays.asList(
            "furthermore", "moreover", "additionally", "for example",
            "specifically", "in particular"
        ));
        
        transitions.put("elaboration", Arrays.asList(
            "however", "nevertheless", "in contrast", "similarly",
            "consequently", "therefore"
        ));
        
        transitions.put("conclusion", Arrays.asList(
            "finally", "in conclusion", "to summarize", "ultimately",
            "in summary", "thus"
        ));
        
        return transitions;
    }
    
    /**
     * Initialize style profiles
     */
    private Map<String, StyleProfile> initializeStyleProfiles() {
        Map<String, StyleProfile> profiles = new HashMap<>();
        
        // Formal style
        StyleProfile formal = new StyleProfile();
        formal.formalityLevel = 0.8;
        formal.avgSentenceLength = 20;
        formal.formalWords = new HashSet<>(Arrays.asList(
            "therefore", "consequently", "furthermore", "moreover",
            "nonetheless", "regarding", "concerning", "establish"
        ));
        formal.informalWords = new HashSet<>(Arrays.asList(
            "gonna", "wanna", "yeah", "cool", "awesome", "stuff"
        ));
        profiles.put("formal", formal);
        
        // Informal style
        StyleProfile informal = new StyleProfile();
        informal.formalityLevel = 0.2;
        informal.avgSentenceLength = 12;
        informal.formalWords = formal.formalWords;
        informal.informalWords = formal.informalWords;
        profiles.put("informal", informal);
        
        // Neutral style
        StyleProfile neutral = new StyleProfile();
        neutral.formalityLevel = 0.5;
        neutral.avgSentenceLength = 15;
        neutral.formalWords = formal.formalWords;
        neutral.informalWords = formal.informalWords;
        profiles.put("neutral", neutral);
        
        return profiles;
    }
    
    // Helper methods
    
    private boolean isContentWord(String word) {
        // Check if word is a content word (noun, verb, adjective, adverb)
        return word.length() > 3 && !isStopWord(word);
    }
    
    private boolean isStopWord(String word) {
        Set<String> stopWords = new HashSet<>(Arrays.asList(
            "the", "a", "an", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "as", "is", "was",
            "are", "were", "been", "be", "have", "has", "had", "do", "does"
        ));
        return stopWords.contains(word.toLowerCase());
    }
    
    private boolean isEntity(String token) {
        return Character.isUpperCase(token.charAt(0)) && 
               token.length() > 1 && 
               !isSentenceStart(token);
    }
    
    private boolean isNewConcept(String token) {
        return isContentWord(token) && !introducedConcepts.contains(token);
    }
    
    private boolean isSentenceStart(String token) {
        // Check if token appears after sentence-ending punctuation
        return generatedSentences.isEmpty() || 
               generatedSentences.get(generatedSentences.size() - 1)
                   .matches(".*[.!?]\\s*$");
    }
    
    private boolean isSentenceEnd(String token) {
        return token.equals(".") || token.equals("!") || token.equals("?");
    }
    
    private boolean isEndToken(String token) {
        return token.equals("<END>") || token.equals("<EOS>");
    }
    
    private String getCurrentSentence(String text) {
        String[] sentences = text.split("[.!?]+");
        return sentences.length > 0 ? sentences[sentences.length - 1].trim() : "";
    }
    
    private boolean shouldUsePronoun(String entity, String context) {
        // Use pronoun if entity was recently mentioned
        Integer mentions = entityMentions.get(entity);
        return mentions != null && mentions > 0 && mentions % 3 == 0;
    }
    
    private List<String> getPronouns(String entity) {
        // Simple pronoun mapping (would use coreference resolution in production)
        if (isPersonName(entity)) {
            return Arrays.asList("he", "she", "they", "him", "her", "them");
        } else {
            return Arrays.asList("it", "its");
        }
    }
    
    private boolean isPersonName(String entity) {
        // Simple heuristic for person names
        Set<String> commonNames = new HashSet<>(Arrays.asList(
            "John", "Jane", "Bob", "Alice", "Mary", "James", "Sarah"
        ));
        return commonNames.contains(entity);
    }
    
    private boolean isFormalWord(String word) {
        return currentStyle != null && currentStyle.formalWords.contains(word);
    }
    
    private boolean isInformalWord(String word) {
        return currentStyle != null && currentStyle.informalWords.contains(word);
    }
    
    private List<Integer> findRelatedTokenIndices(String keyword) {
        // Find tokens semantically related to keyword
        // Simplified - would use word embeddings in production
        List<Integer> indices = new ArrayList<>();
        // Placeholder implementation
        return indices;
    }
    
    private int getTokenIndex(String token) {
        // Convert token to index (placeholder)
        return -1;
    }
    
    private String indexToToken(int index) {
        // Convert index to token (placeholder)
        return "";
    }
    
    private int sampleToken(double[] probabilities) {
        // Sample from probability distribution
        double random = Math.random();
        double cumSum = 0;
        
        for (int i = 0; i < probabilities.length; i++) {
            cumSum += probabilities[i];
            if (random < cumSum) {
                return i;
            }
        }
        
        return probabilities.length - 1;
    }
    
    private double[] normalize(double[] values) {
        double sum = Arrays.stream(values).sum();
        if (sum == 0) return values;
        
        double[] normalized = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            normalized[i] = values[i] / sum;
        }
        return normalized;
    }
    
    /**
     * Style profile for maintaining consistency
     */
    private static class StyleProfile {
        double formalityLevel;  // 0 = informal, 1 = formal
        int avgSentenceLength;
        Set<String> formalWords;
        Set<String> informalWords;
    }
    
    /**
     * Interface for base token generator
     */
    public interface TokenGenerator {
        double[] getTokenProbabilities(String context);
        String indexToToken(int index);
    }
    
    /**
     * Get current context state for debugging/monitoring
     */
    public Map<String, Object> getContextState() {
        Map<String, Object> state = new HashMap<>();
        state.put("discourse_state", currentDiscourseState);
        state.put("topic_keywords", topicKeywords);
        state.put("entity_mentions", entityMentions);
        state.put("style_formality", currentStyle != null ? currentStyle.formalityLevel : 0.5);
        state.put("sentences_generated", generatedSentences.size());
        state.put("introduced_concepts", introducedConcepts.size());
        return state;
    }
}
