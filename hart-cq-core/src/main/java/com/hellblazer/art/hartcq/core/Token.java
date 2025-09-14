package com.hellblazer.art.hartcq.core;

import java.util.Objects;

/**
 * Represents a token in the HART-CQ system.
 * Tokens are the basic units of text processing.
 */
public class Token {
    private final String text;
    private final int position;
    private final TokenType type;
    
    public enum TokenType {
        WORD,
        PUNCTUATION,
        NUMBER,
        WHITESPACE,
        SPECIAL,
        UNKNOWN
    }
    
    public Token(String text, int position, TokenType type) {
        this.text = Objects.requireNonNull(text, "Token text cannot be null");
        this.position = position;
        this.type = Objects.requireNonNull(type, "Token type cannot be null");
    }
    
    public String getText() {
        return text;
    }
    
    public int getPosition() {
        return position;
    }
    
    public TokenType getType() {
        return type;
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        var token = (Token) o;
        return position == token.position && 
               Objects.equals(text, token.text) && 
               type == token.type;
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(text, position, type);
    }
    
    @Override
    public String toString() {
        return String.format("Token[text='%s', pos=%d, type=%s]", text, position, type);
    }
}