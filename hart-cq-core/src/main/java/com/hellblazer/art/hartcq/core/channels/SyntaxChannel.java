package com.hellblazer.art.hartcq.core.channels;

import com.hellblazer.art.hartcq.Token;
import com.hellblazer.art.hartcq.ProcessingWindow;
import com.hellblazer.art.hartcq.core.SlidingWindow;

/**
 * Syntax pattern channel that captures grammatical structures.
 */
public class SyntaxChannel implements Channel {
    
    private static final int SYNTAX_DIM = 32;
    
    // Common syntactic patterns
    private enum SyntaxPattern {
        SENTENCE_START,    // Capital letter at start
        SENTENCE_END,      // Period, question mark, exclamation
        COMMA_PAUSE,       // Comma-separated clauses
        QUOTED_TEXT,       // Text in quotes
        PARENTHETICAL,     // Text in parentheses
        NUMBER_SEQUENCE,   // Sequences of numbers
        CAPITALIZED_WORD,  // Proper nouns
        ALL_CAPS,          // Emphasis or acronyms
        PUNCTUATION_CLUSTER // Multiple punctuation marks
    }
    
    @Override
    public float[] processWindow(Token[] tokens) {
        var output = new float[SYNTAX_DIM];
        
        // Analyze syntactic patterns in the window
        for (int i = 0; i < tokens.length && i < SlidingWindow.WINDOW_SIZE; i++) {
            if (tokens[i] == null) continue;
            
            var token = tokens[i];
            var text = token.getText();
            
            // Check for sentence boundaries
            if (i == 0 && isCapitalized(text)) {
                output[SyntaxPattern.SENTENCE_START.ordinal()] += 1.0f;
            }
            
            if (isSentenceEnd(text)) {
                output[SyntaxPattern.SENTENCE_END.ordinal()] += 1.0f;
            }
            
            // Check for punctuation patterns
            if (token.getType() == Token.TokenType.PUNCTUATION) {
                if (text.equals(",")) {
                    output[SyntaxPattern.COMMA_PAUSE.ordinal()] += 1.0f;
                }
                if (text.equals("\"") || text.equals("'")) {
                    output[SyntaxPattern.QUOTED_TEXT.ordinal()] += 0.5f;
                }
                if (text.equals("(") || text.equals(")")) {
                    output[SyntaxPattern.PARENTHETICAL.ordinal()] += 0.5f;
                }
            }
            
            // Check for word patterns
            if (token.getType() == Token.TokenType.WORD) {
                if (isCapitalized(text) && i > 0) {
                    output[SyntaxPattern.CAPITALIZED_WORD.ordinal()] += 1.0f;
                }
                if (isAllCaps(text) && text.length() > 1) {
                    output[SyntaxPattern.ALL_CAPS.ordinal()] += 1.0f;
                }
            }
            
            // Check for number sequences
            if (token.getType() == Token.TokenType.NUMBER) {
                output[SyntaxPattern.NUMBER_SEQUENCE.ordinal()] += 1.0f;
            }
            
            // Look for punctuation clusters
            if (i > 0 && tokens[i-1] != null) {
                if (token.getType() == Token.TokenType.PUNCTUATION && 
                    tokens[i-1].getType() == Token.TokenType.PUNCTUATION) {
                    output[SyntaxPattern.PUNCTUATION_CLUSTER.ordinal()] += 1.0f;
                }
            }
        }
        
        // Add n-gram patterns (bigrams and trigrams)
        analyzeBigrams(tokens, output);
        
        // Normalize
        normalize(output);
        
        return output;
    }
    
    private void analyzeBigrams(Token[] tokens, float[] output) {
        int offset = SyntaxPattern.values().length;
        
        for (int i = 0; i < tokens.length - 1 && i < SlidingWindow.WINDOW_SIZE - 1; i++) {
            if (tokens[i] == null || tokens[i+1] == null) continue;
            
            var t1 = tokens[i].getType();
            var t2 = tokens[i+1].getType();
            
            // Common bigram patterns
            if (t1 == Token.TokenType.WORD && t2 == Token.TokenType.WORD) {
                output[offset] += 1.0f; // Word-Word
            } else if (t1 == Token.TokenType.WORD && t2 == Token.TokenType.PUNCTUATION) {
                output[offset + 1] += 1.0f; // Word-Punct
            } else if (t1 == Token.TokenType.NUMBER && t2 == Token.TokenType.WORD) {
                output[offset + 2] += 1.0f; // Number-Word
            }
        }
    }
    
    private boolean isCapitalized(String text) {
        return text != null && text.length() > 0 && Character.isUpperCase(text.charAt(0));
    }
    
    private boolean isAllCaps(String text) {
        return text != null && text.equals(text.toUpperCase());
    }
    
    private boolean isSentenceEnd(String text) {
        return text != null && (text.equals(".") || text.equals("!") || text.equals("?"));
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
        return SYNTAX_DIM;
    }
    
    @Override
    public String getName() {
        return "SyntaxChannel";
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
        return ChannelType.STRUCTURAL;
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