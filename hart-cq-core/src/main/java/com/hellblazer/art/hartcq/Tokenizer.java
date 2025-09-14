package com.hellblazer.art.hartcq;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Tokenizer for converting text to tokens in the HART-CQ system.
 * This is a simplified interface to the core tokenizer functionality,
 * focused on the essential operations needed by the HART-CQ algorithm.
 */
public class Tokenizer {
    
    private final com.hellblazer.art.hartcq.core.Tokenizer coreTokenizer;
    
    // Commonly used patterns for token classification
    private static final Pattern WORD_PATTERN = Pattern.compile("\\w+");
    private static final Pattern NUMBER_PATTERN = Pattern.compile("\\d+(\\.\\d+)?");
    private static final Pattern PUNCTUATION_PATTERN = Pattern.compile("[.,;:!?()\\[\\]{}'\"\\-]");
    private static final Pattern SPECIAL_PATTERN = Pattern.compile("[#@$%^&*+=<>~`]");
    
    /**
     * Constructs a new Tokenizer using the core tokenization engine.
     */
    public Tokenizer() {
        this.coreTokenizer = new com.hellblazer.art.hartcq.core.Tokenizer();
    }
    
    /**
     * Tokenizes input text into a list of tokens.
     * This is the primary method for converting text to tokens.
     *
     * @param text the text to tokenize
     * @return list of tokens extracted from the text
     */
    public List<Token> tokenize(String text) {
        if (text == null || text.trim().isEmpty()) {
            return new ArrayList<>();
        }

        var coreTokens = coreTokenizer.tokenize(text);
        var tokens = new ArrayList<Token>();

        for (var coreToken : coreTokens) {
            tokens.add(convertToken(coreToken));
        }

        return tokens;
    }

    /**
     * Converts a core token to a root package token.
     */
    private Token convertToken(com.hellblazer.art.hartcq.core.Token coreToken) {
        var tokenType = convertTokenType(coreToken.getType());
        return new Token(coreToken.getText(), coreToken.getPosition(), tokenType);
    }

    /**
     * Converts core token type to root token type.
     */
    private Token.TokenType convertTokenType(com.hellblazer.art.hartcq.core.Token.TokenType coreType) {
        return switch (coreType) {
            case WORD -> Token.TokenType.WORD;
            case PUNCTUATION -> Token.TokenType.PUNCTUATION;
            case NUMBER -> Token.TokenType.NUMBER;
            case WHITESPACE -> Token.TokenType.WHITESPACE;
            case SPECIAL -> Token.TokenType.SPECIAL;
            case UNKNOWN -> Token.TokenType.UNKNOWN;
        };
    }
    
    /**
     * Tokenizes text and filters to return only word tokens.
     * Useful for linguistic analysis that focuses on content words.
     * 
     * @param text the text to tokenize
     * @return list containing only word tokens
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
     * Tokenizes text and returns tokens categorized by type.
     * Provides detailed breakdown of different token types found.
     * 
     * @param text the text to tokenize
     * @return TokenizedResult containing categorized tokens
     */
    public TokenizedResult tokenizeWithCategories(String text) {
        var allTokens = tokenize(text);
        var result = new TokenizedResult();
        
        for (Token token : allTokens) {
            result.addToken(token);
        }
        
        return result;
    }
    
    /**
     * Tokenizes text for sliding window processing.
     * Optimized for the 20-token sliding window used in HART-CQ.
     * 
     * @param text the text to tokenize
     * @param windowSize the size of each window (typically 20)
     * @return list of token windows
     */
    public List<List<Token>> tokenizeForWindows(String text, int windowSize) {
        var allTokens = tokenize(text);
        var windows = new ArrayList<List<Token>>();
        
        if (allTokens.size() <= windowSize) {
            windows.add(new ArrayList<>(allTokens));
            return windows;
        }
        
        // Create sliding windows
        for (int i = 0; i <= allTokens.size() - windowSize; i++) {
            var window = new ArrayList<Token>();
            for (int j = i; j < i + windowSize; j++) {
                window.add(allTokens.get(j));
            }
            windows.add(window);
        }
        
        return windows;
    }
    
    /**
     * Fast tokenization for performance-critical paths.
     * Uses simplified tokenization rules for maximum speed.
     * 
     * @param text the text to tokenize
     * @return list of tokens using fast tokenization
     */
    public List<Token> fastTokenize(String text) {
        if (text == null || text.trim().isEmpty()) {
            return new ArrayList<>();
        }
        
        var tokens = new ArrayList<Token>();
        var words = text.split("\\s+");
        var position = 0;
        
        for (String word : words) {
            if (word.trim().isEmpty()) continue;
            
            Token.TokenType type = determineTokenType(word);
            tokens.add(new Token(word.trim(), position++, type));
        }
        
        return tokens;
    }
    
    /**
     * Determines the type of a token based on its content.
     * 
     * @param text the token text to analyze
     * @return the appropriate TokenType
     */
    private Token.TokenType determineTokenType(String text) {
        if (NUMBER_PATTERN.matcher(text).matches()) {
            return Token.TokenType.NUMBER;
        } else if (WORD_PATTERN.matcher(text).matches()) {
            return Token.TokenType.WORD;
        } else if (PUNCTUATION_PATTERN.matcher(text).matches()) {
            return Token.TokenType.PUNCTUATION;
        } else if (SPECIAL_PATTERN.matcher(text).matches()) {
            return Token.TokenType.SPECIAL;
        } else {
            return Token.TokenType.UNKNOWN;
        }
    }
    
    /**
     * Counts tokens by type for statistical analysis.
     * 
     * @param tokens list of tokens to analyze
     * @return TokenStats containing count information
     */
    public TokenStats getTokenStats(List<Token> tokens) {
        var stats = new TokenStats();
        
        for (Token token : tokens) {
            stats.incrementType(token.getType());
        }
        
        return stats;
    }
    
    /**
     * Result of categorized tokenization.
     */
    public static class TokenizedResult {
        private final List<Token> words = new ArrayList<>();
        private final List<Token> numbers = new ArrayList<>();
        private final List<Token> punctuation = new ArrayList<>();
        private final List<Token> special = new ArrayList<>();
        private final List<Token> whitespace = new ArrayList<>();
        private final List<Token> unknown = new ArrayList<>();
        private final List<Token> allTokens = new ArrayList<>();
        
        public void addToken(Token token) {
            allTokens.add(token);
            
            switch (token.getType()) {
                case WORD -> words.add(token);
                case NUMBER -> numbers.add(token);
                case PUNCTUATION -> punctuation.add(token);
                case SPECIAL -> special.add(token);
                case WHITESPACE -> whitespace.add(token);
                case UNKNOWN -> unknown.add(token);
            }
        }
        
        public List<Token> getWords() { return new ArrayList<>(words); }
        public List<Token> getNumbers() { return new ArrayList<>(numbers); }
        public List<Token> getPunctuation() { return new ArrayList<>(punctuation); }
        public List<Token> getSpecial() { return new ArrayList<>(special); }
        public List<Token> getWhitespace() { return new ArrayList<>(whitespace); }
        public List<Token> getUnknown() { return new ArrayList<>(unknown); }
        public List<Token> getAllTokens() { return new ArrayList<>(allTokens); }
        
        public int getTotalTokens() { return allTokens.size(); }
        public int getWordCount() { return words.size(); }
        public int getNumberCount() { return numbers.size(); }
        public int getPunctuationCount() { return punctuation.size(); }
    }
    
    /**
     * Statistical information about tokenized text.
     */
    public static class TokenStats {
        private int wordCount = 0;
        private int numberCount = 0;
        private int punctuationCount = 0;
        private int specialCount = 0;
        private int whitespaceCount = 0;
        private int unknownCount = 0;
        
        public void incrementType(Token.TokenType type) {
            switch (type) {
                case WORD -> wordCount++;
                case NUMBER -> numberCount++;
                case PUNCTUATION -> punctuationCount++;
                case SPECIAL -> specialCount++;
                case WHITESPACE -> whitespaceCount++;
                case UNKNOWN -> unknownCount++;
            }
        }
        
        public int getWordCount() { return wordCount; }
        public int getNumberCount() { return numberCount; }
        public int getPunctuationCount() { return punctuationCount; }
        public int getSpecialCount() { return specialCount; }
        public int getWhitespaceCount() { return whitespaceCount; }
        public int getUnknownCount() { return unknownCount; }
        public int getTotalCount() { 
            return wordCount + numberCount + punctuationCount + specialCount + whitespaceCount + unknownCount; 
        }
        
        @Override
        public String toString() {
            return String.format("TokenStats[total=%d, words=%d, numbers=%d, punct=%d, special=%d, whitespace=%d, unknown=%d]",
                               getTotalCount(), wordCount, numberCount, punctuationCount, specialCount, whitespaceCount, unknownCount);
        }
    }
}