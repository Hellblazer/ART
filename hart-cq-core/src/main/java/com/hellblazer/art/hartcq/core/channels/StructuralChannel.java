package com.hellblazer.art.hartcq.core.channels;

import com.hellblazer.art.hartcq.ProcessingWindow;
import com.hellblazer.art.hartcq.Token;
import com.hellblazer.art.hartcq.core.SlidingWindow;

import java.util.Set;

/**
 * Structural analysis channel that analyzes sentence structure and grammatical patterns.
 * Tracks syntactic relationships, grammatical constructions, and structural features.
 */
public class StructuralChannel implements Channel {
    
    private static final int STRUCTURAL_DIM = 56;
    
    // Grammatical function words
    private static final Set<String> DETERMINERS = Set.of(
        "the", "a", "an", "this", "that", "these", "those", "my", "your", "his", 
        "her", "its", "our", "their", "some", "any", "all", "each", "every"
    );
    
    private static final Set<String> PREPOSITIONS = Set.of(
        "in", "on", "at", "by", "for", "with", "without", "to", "from", "of",
        "about", "under", "over", "through", "between", "among", "during", "before", "after"
    );
    
    private static final Set<String> CONJUNCTIONS = Set.of(
        "and", "or", "but", "so", "yet", "for", "nor", "because", "since", "although",
        "though", "while", "if", "unless", "when", "where", "whereas", "however"
    );
    
    private static final Set<String> AUXILIARY_VERBS = Set.of(
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "shall", "should", "can", "could", "may", "might", "must"
    );
    
    private static final Set<String> PRONOUNS = Set.of(
        "I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "myself", "yourself", "himself", "herself", "itself", "ourselves", "themselves"
    );
    
    @Override
    public ChannelOutput process(ProcessingWindow window) {
        var startTime = System.nanoTime();
        var tokens = window.getTokens();
        var tokenArray = tokens.toArray(new Token[0]);
        var output = new float[STRUCTURAL_DIM];
        
        // Analyze grammatical structure
        analyzeGrammaticalCategories(tokenArray, output);
        
        // Analyze sentence structure patterns
        analyzeSentencePatterns(tokenArray, output);
        
        // Analyze dependency-like relationships
        analyzeDependencyPatterns(tokenArray, output);
        
        // Analyze punctuation structure
        analyzePunctuationStructure(tokenArray, output);
        
        // Normalize the output
        normalize(output);
        
        var processingTime = System.nanoTime() - startTime;
        return ChannelOutput.valid(output, getChannelType(), processingTime);
    }

    @Override
    public float[] processWindow(Token[] tokens) {
        var output = new float[STRUCTURAL_DIM];
        
        // Analyze grammatical structure
        analyzeGrammaticalCategories(tokens, output);
        
        // Analyze sentence structure patterns
        analyzeSentencePatterns(tokens, output);
        
        // Analyze dependency-like relationships
        analyzeDependencyPatterns(tokens, output);
        
        // Analyze punctuation structure
        analyzePunctuationStructure(tokens, output);
        
        // Normalize the output
        normalize(output);
        
        return output;
    }
    
    private void analyzeGrammaticalCategories(Token[] tokens, float[] output) {
        int detCount = 0, prepCount = 0, conjCount = 0, auxCount = 0, pronCount = 0;
        int totalWords = 0;
        
        for (int i = 0; i < tokens.length && i < SlidingWindow.WINDOW_SIZE; i++) {
            if (tokens[i] == null || tokens[i].getType() != Token.TokenType.WORD) {
                continue;
            }
            
            var text = tokens[i].getText();
            if (text == null || text.isEmpty()) {
                continue;
            }
            var word = text.toLowerCase();
            totalWords++;
            
            if (DETERMINERS.contains(word)) detCount++;
            if (PREPOSITIONS.contains(word)) prepCount++;
            if (CONJUNCTIONS.contains(word)) conjCount++;
            if (AUXILIARY_VERBS.contains(word)) auxCount++;
            var originalText = tokens[i].getText();
            if (PRONOUNS.contains(word.toLowerCase()) || (originalText != null && PRONOUNS.contains(originalText))) pronCount++;
        }
        
        // Store grammatical category ratios
        if (totalWords > 0) {
            output[0] = (float) detCount / totalWords;     // Determiner density
            output[1] = (float) prepCount / totalWords;    // Preposition density
            output[2] = (float) conjCount / totalWords;    // Conjunction density
            output[3] = (float) auxCount / totalWords;     // Auxiliary verb density
            output[4] = (float) pronCount / totalWords;    // Pronoun density
        }
        
        // Absolute counts (normalized by window size)
        output[5] = (float) detCount / SlidingWindow.WINDOW_SIZE;
        output[6] = (float) prepCount / SlidingWindow.WINDOW_SIZE;
        output[7] = (float) conjCount / SlidingWindow.WINDOW_SIZE;
    }
    
    private void analyzeSentencePatterns(Token[] tokens, float[] output) {
        int offset = 8;
        
        // Look for common structural patterns
        for (int i = 0; i < tokens.length - 2 && i < SlidingWindow.WINDOW_SIZE - 2; i++) {
            if (tokens[i] == null || tokens[i+1] == null || tokens[i+2] == null) continue;
            
            var word1 = getWordText(tokens[i]);
            var word2 = getWordText(tokens[i+1]);
            var word3 = getWordText(tokens[i+2]);
            
            if (word1 == null || word2 == null || word3 == null) continue;
            
            // Determiner + Adjective + Noun pattern (e.g., "the big house")
            if (DETERMINERS.contains(word1) && tokens[i+2].getType() == Token.TokenType.WORD) {
                output[offset] += 1.0f;
            }
            
            // Preposition + Determiner + Noun pattern (e.g., "in the house")
            if (PREPOSITIONS.contains(word1) && DETERMINERS.contains(word2)) {
                output[offset + 1] += 1.0f;
            }
            
            // Auxiliary + Verb pattern (e.g., "is running", "have seen")
            if (AUXILIARY_VERBS.contains(word1) && tokens[i+1].getType() == Token.TokenType.WORD) {
                output[offset + 2] += 1.0f;
            }
            
            // Pronoun + Auxiliary + Verb pattern (e.g., "I am going")
            if (PRONOUNS.contains(word1) && AUXILIARY_VERBS.contains(word2)) {
                output[offset + 3] += 1.0f;
            }
        }
    }
    
    private void analyzeDependencyPatterns(Token[] tokens, float[] output) {
        int offset = 12;
        
        // Simple heuristic dependency analysis
        for (int i = 0; i < tokens.length - 1 && i < SlidingWindow.WINDOW_SIZE - 1; i++) {
            if (tokens[i] == null || tokens[i+1] == null) continue;
            
            var word1 = getWordText(tokens[i]);
            var word2 = getWordText(tokens[i+1]);
            
            if (word1 == null || word2 == null) continue;
            
            // Subject-verb patterns
            if (PRONOUNS.contains(word1) && AUXILIARY_VERBS.contains(word2)) {
                output[offset] += 1.0f; // Subject-auxiliary dependency
            }
            
            // Determiner-noun dependencies
            if (DETERMINERS.contains(word1) && tokens[i+1].getType() == Token.TokenType.WORD) {
                output[offset + 1] += 1.0f; // Determiner-noun dependency
            }
            
            // Preposition-object dependencies
            if (PREPOSITIONS.contains(word1)) {
                output[offset + 2] += 1.0f; // Prepositional dependencies
            }
            
            // Conjunction coordination
            if (CONJUNCTIONS.contains(word1)) {
                output[offset + 3] += 1.0f; // Coordination patterns
            }
        }
        
        // Analyze word distance patterns (approximation of dependency distance)
        analyzeWordDistances(tokens, output, offset + 4);
    }
    
    private void analyzeWordDistances(Token[] tokens, float[] output, int offset) {
        // Calculate average distances between related word types
        int detNounDistance = 0, detNounCount = 0;
        int prepObjDistance = 0, prepObjCount = 0;
        
        for (int i = 0; i < tokens.length - 1 && i < SlidingWindow.WINDOW_SIZE - 1; i++) {
            if (tokens[i] == null) continue;
            var word1 = getWordText(tokens[i]);
            if (word1 == null) continue;
            
            // Look for related words within a small window
            for (int j = i + 1; j < Math.min(i + 5, tokens.length) && j < SlidingWindow.WINDOW_SIZE; j++) {
                if (tokens[j] == null) continue;
                var word2 = getWordText(tokens[j]);
                if (word2 == null) continue;
                
                // Determiner to next noun
                if (DETERMINERS.contains(word1) && tokens[j].getType() == Token.TokenType.WORD && 
                    !DETERMINERS.contains(word2) && !PREPOSITIONS.contains(word2)) {
                    detNounDistance += (j - i);
                    detNounCount++;
                    break;
                }
                
                // Preposition to object
                if (PREPOSITIONS.contains(word1) && tokens[j].getType() == Token.TokenType.WORD &&
                    !PREPOSITIONS.contains(word2) && !CONJUNCTIONS.contains(word2)) {
                    prepObjDistance += (j - i);
                    prepObjCount++;
                    break;
                }
            }
        }
        
        // Store average distances (normalized)
        if (detNounCount > 0) {
            output[offset] = (float) detNounDistance / (detNounCount * 5.0f); // Normalized by max distance
        }
        if (prepObjCount > 0) {
            output[offset + 1] = (float) prepObjDistance / (prepObjCount * 5.0f);
        }
    }
    
    private void analyzePunctuationStructure(Token[] tokens, float[] output) {
        int offset = 18;
        
        int commaCount = 0, periodCount = 0, questionCount = 0, exclamationCount = 0;
        int openParenCount = 0, closeParenCount = 0;
        
        // Analyze punctuation patterns
        for (int i = 0; i < tokens.length && i < SlidingWindow.WINDOW_SIZE; i++) {
            if (tokens[i] == null || tokens[i].getType() != Token.TokenType.PUNCTUATION) {
                continue;
            }
            
            var punct = tokens[i].getText();
            if (punct == null || punct.isEmpty()) {
                continue;
            }
            switch (punct) {
                case "," -> commaCount++;
                case "." -> periodCount++;
                case "?" -> questionCount++;
                case "!" -> exclamationCount++;
                case "(" -> openParenCount++;
                case ")" -> closeParenCount++;
            }
        }
        
        // Store punctuation features
        output[offset] = (float) commaCount / SlidingWindow.WINDOW_SIZE;       // Comma density
        output[offset + 1] = (float) periodCount / SlidingWindow.WINDOW_SIZE; // Period density
        output[offset + 2] = (float) questionCount / SlidingWindow.WINDOW_SIZE; // Question density
        output[offset + 3] = (float) exclamationCount / SlidingWindow.WINDOW_SIZE; // Exclamation density
        
        // Parentheses balance
        output[offset + 4] = Math.abs(openParenCount - closeParenCount) / (float) SlidingWindow.WINDOW_SIZE;
        
        // Sentence boundary indicators
        output[offset + 5] = (periodCount + questionCount + exclamationCount) > 0 ? 1.0f : 0.0f;
    }
    
    private String getWordText(Token token) {
        if (token == null || token.getType() != Token.TokenType.WORD) {
            return null;
        }
        var text = token.getText();
        if (text == null || text.isEmpty()) {
            return null;
        }
        return text.toLowerCase();
    }
    
    private void normalize(float[] vector) {
        float sum = 0;
        for (float v : vector) {
            sum += v * v;
        }
        
        if (sum > 0) {
            float norm = (float) Math.sqrt(sum);
            for (int i = 0; i < vector.length; i++) {
                vector[i] /= norm;
            }
        }
    }
    
    @Override
    public int getOutputDimension() {
        return STRUCTURAL_DIM;
    }
    
    @Override
    public String getName() {
        return "StructuralChannel";
    }
    
    @Override
    public ChannelType getChannelType() {
        return ChannelType.STRUCTURAL;
    }
    
    @Override
    public void reset() {
        // No state to reset - structural analysis is stateless
    }
    
    @Override
    public boolean isDeterministic() {
        return true; // Same input produces same structural analysis
    }
}