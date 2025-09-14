package com.hellblazer.art.hartcq.core.channels;

import com.hellblazer.art.hartcq.Token;
import com.hellblazer.art.hartcq.ProcessingWindow;
import com.hellblazer.art.hartcq.core.SlidingWindow;

import java.util.HashSet;
import java.util.Set;

/**
 * Semantic relationship channel that captures meaning relationships.
 */
public class SemanticChannel implements Channel {
    
    private static final int SEMANTIC_DIM = 48;
    
    // Semantic categories
    private static final Set<String> TEMPORAL_WORDS = Set.of(
        "today", "tomorrow", "yesterday", "now", "then", "when", "before", "after",
        "during", "while", "year", "month", "day", "hour", "minute", "second"
    );
    
    private static final Set<String> SPATIAL_WORDS = Set.of(
        "here", "there", "where", "above", "below", "beside", "between", "near",
        "far", "left", "right", "up", "down", "in", "out", "inside", "outside"
    );
    
    private static final Set<String> CAUSAL_WORDS = Set.of(
        "because", "therefore", "thus", "hence", "so", "since", "as", "for",
        "consequently", "accordingly", "due", "owing", "result", "cause", "effect"
    );
    
    private static final Set<String> MODAL_WORDS = Set.of(
        "can", "could", "may", "might", "must", "shall", "should", "will",
        "would", "ought", "need", "dare", "able", "possible", "necessary"
    );
    
    @Override
    public float[] processWindow(Token[] tokens) {
        var output = new float[SEMANTIC_DIM];
        
        // Count semantic categories
        int temporalCount = 0;
        int spatialCount = 0;
        int causalCount = 0;
        int modalCount = 0;
        
        // Analyze tokens for semantic patterns
        for (int i = 0; i < tokens.length && i < SlidingWindow.WINDOW_SIZE; i++) {
            if (tokens[i] == null || tokens[i].getType() != Token.TokenType.WORD) {
                continue;
            }
            
            var text = tokens[i].getText();
            if (text == null || text.isEmpty()) {
                continue;
            }
            var word = text.toLowerCase();
            
            if (TEMPORAL_WORDS.contains(word)) {
                temporalCount++;
                output[0] += 1.0f;
            }
            if (SPATIAL_WORDS.contains(word)) {
                spatialCount++;
                output[1] += 1.0f;
            }
            if (CAUSAL_WORDS.contains(word)) {
                causalCount++;
                output[2] += 1.0f;
            }
            if (MODAL_WORDS.contains(word)) {
                modalCount++;
                output[3] += 1.0f;
            }
        }
        
        // Add semantic density features
        output[4] = (float) temporalCount / SlidingWindow.WINDOW_SIZE;
        output[5] = (float) spatialCount / SlidingWindow.WINDOW_SIZE;
        output[6] = (float) causalCount / SlidingWindow.WINDOW_SIZE;
        output[7] = (float) modalCount / SlidingWindow.WINDOW_SIZE;
        
        // Analyze semantic transitions
        analyzeSemanticTransitions(tokens, output);
        
        // Add topic coherence features
        analyzeTopicCoherence(tokens, output);
        
        // Normalize
        normalize(output);
        
        return output;
    }
    
    private void analyzeSemanticTransitions(Token[] tokens, float[] output) {
        int offset = 8;
        
        for (int i = 0; i < tokens.length - 1 && i < SlidingWindow.WINDOW_SIZE - 1; i++) {
            if (tokens[i] == null || tokens[i+1] == null) continue;
            if (tokens[i].getType() != Token.TokenType.WORD || 
                tokens[i+1].getType() != Token.TokenType.WORD) continue;
            
            var text1 = tokens[i].getText();
            var text2 = tokens[i+1].getText();
            if (text1 == null || text1.isEmpty() || text2 == null || text2.isEmpty()) {
                continue;
            }
            var word1 = text1.toLowerCase();
            var word2 = text2.toLowerCase();
            
            // Temporal to spatial transition
            if (TEMPORAL_WORDS.contains(word1) && SPATIAL_WORDS.contains(word2)) {
                output[offset] += 1.0f;
            }
            // Causal relationships
            if (CAUSAL_WORDS.contains(word1) || CAUSAL_WORDS.contains(word2)) {
                output[offset + 1] += 1.0f;
            }
            // Modal transitions
            if (MODAL_WORDS.contains(word1) && !MODAL_WORDS.contains(word2)) {
                output[offset + 2] += 1.0f;
            }
        }
    }
    
    private void analyzeTopicCoherence(Token[] tokens, float[] output) {
        int offset = 11;
        
        // Simple topic coherence: count unique words vs total words
        Set<String> uniqueWords = new HashSet<>();
        int totalWords = 0;
        
        for (int i = 0; i < tokens.length && i < SlidingWindow.WINDOW_SIZE; i++) {
            if (tokens[i] != null && tokens[i].getType() == Token.TokenType.WORD) {
                var text = tokens[i].getText();
                if (text != null && !text.isEmpty()) {
                    uniqueWords.add(text.toLowerCase());
                }
                totalWords++;
            }
        }
        
        if (totalWords > 0) {
            // Lexical diversity
            output[offset] = (float) uniqueWords.size() / totalWords;
            
            // Repetition indicator
            output[offset + 1] = 1.0f - output[offset];
        }
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
        return SEMANTIC_DIM;
    }
    
    @Override
    public String getName() {
        return "SemanticChannel";
    }
    
    @Override
    public void reset() {
        // No state to reset
    }
    
    @Override
    public boolean isDeterministic() {
        return true;
    }

    @Override
    public ChannelType getChannelType() {
        return ChannelType.SEMANTIC;
    }

    @Override
    public ChannelOutput process(ProcessingWindow window) {
        var startTime = System.nanoTime();

        // Convert ProcessingWindow to Token array for legacy compatibility
        var tokens = window.getTokens().toArray(new Token[0]);
        var features = processWindow(tokens);

        var processingTime = System.nanoTime() - startTime;
        return ChannelOutput.valid(features, getChannelType(), processingTime);
    }
}