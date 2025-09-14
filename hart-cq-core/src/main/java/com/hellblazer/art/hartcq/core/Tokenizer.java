package com.hellblazer.art.hartcq.core;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Tokenizer for breaking text into tokens for HART-CQ processing.
 */
public class Tokenizer {
    
    private static final Pattern WORD_PATTERN = Pattern.compile("\\w+");
    private static final Pattern NUMBER_PATTERN = Pattern.compile("\\d+(\\.\\d+)?");
    private static final Pattern PUNCTUATION_PATTERN = Pattern.compile("[.,;:!?()\\[\\]{}'\"\\-]");
    private static final Pattern SPECIAL_PATTERN = Pattern.compile("[@#$%\\^&*+<>=~`]");
    private static final Pattern WHITESPACE_PATTERN = Pattern.compile("\\s+");
    
    /**
     * Tokenizes input text into a list of tokens.
     * @param text The text to tokenize
     * @return List of tokens
     */
    public List<Token> tokenize(String text) {
        if (text == null || text.isEmpty()) {
            return new ArrayList<>();
        }
        
        var tokens = new ArrayList<Token>();
        var chars = text.toCharArray();
        var buffer = new StringBuilder();
        int position = 0;
        
        for (int i = 0; i < chars.length; i++) {
            char c = chars[i];
            
            if (Character.isWhitespace(c)) {
                if (buffer.length() > 0) {
                    tokens.add(createToken(buffer.toString(), position++));
                    buffer.setLength(0);
                }
                // Collect consecutive whitespace
                var wsBuffer = new StringBuilder();
                while (i < chars.length && Character.isWhitespace(chars[i])) {
                    wsBuffer.append(chars[i]);
                    i++;
                }
                i--; // Back up one since the loop will increment
                // Don't add whitespace tokens to maintain compatibility with tests
            } else if (isSpecial(c)) {
                if (buffer.length() > 0) {
                    tokens.add(createToken(buffer.toString(), position++));
                    buffer.setLength(0);
                }
                tokens.add(new Token(String.valueOf(c), position++, Token.TokenType.SPECIAL));
            } else if (isPunctuation(c)) {
                if (buffer.length() > 0) {
                    tokens.add(createToken(buffer.toString(), position++));
                    buffer.setLength(0);
                }
                tokens.add(new Token(String.valueOf(c), position++, Token.TokenType.PUNCTUATION));
            } else {
                buffer.append(c);
            }
        }
        
        // Don't forget the last token
        if (buffer.length() > 0) {
            tokens.add(createToken(buffer.toString(), position));
        }
        
        return tokens;
    }
    
    /**
     * Creates a token with appropriate type detection.
     */
    private Token createToken(String text, int position) {
        if (NUMBER_PATTERN.matcher(text).matches()) {
            return new Token(text, position, Token.TokenType.NUMBER);
        } else if (WORD_PATTERN.matcher(text).matches()) {
            return new Token(text, position, Token.TokenType.WORD);
        } else {
            return new Token(text, position, Token.TokenType.UNKNOWN);
        }
    }
    
    /**
     * Checks if a character is punctuation.
     */
    private boolean isPunctuation(char c) {
        return PUNCTUATION_PATTERN.matcher(String.valueOf(c)).matches();
    }

    /**
     * Checks if a character is a special character.
     */
    private boolean isSpecial(char c) {
        return SPECIAL_PATTERN.matcher(String.valueOf(c)).matches();
    }
    
    /**
     * Tokenizes text and returns only word tokens (filtering out punctuation, numbers, etc).
     * @param text The text to tokenize
     * @return List of word tokens only
     */
    public List<Token> tokenizeWords(String text) {
        var allTokens = tokenize(text);
        var wordTokens = new ArrayList<Token>();
        
        for (Token token : allTokens) {
            if (token.getType() == Token.TokenType.WORD) {
                wordTokens.add(token);
            }
        }
        
        return wordTokens;
    }
    
    /**
     * Tokenizes text into fixed-size chunks for batch processing.
     * @param text The text to tokenize
     * @param chunkSize The size of each chunk
     * @return List of token chunks
     */
    public List<List<Token>> tokenizeInChunks(String text, int chunkSize) {
        var allTokens = tokenize(text);
        var chunks = new ArrayList<List<Token>>();
        
        for (int i = 0; i < allTokens.size(); i += chunkSize) {
            int end = Math.min(i + chunkSize, allTokens.size());
            chunks.add(new ArrayList<>(allTokens.subList(i, end)));
        }
        
        return chunks;
    }
}