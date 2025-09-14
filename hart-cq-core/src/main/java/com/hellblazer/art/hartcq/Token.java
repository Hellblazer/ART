package com.hellblazer.art.hartcq;

import java.util.Objects;

/**
 * Represents a single token in the processing stream.
 * Immutable design for thread-safe processing.
 */
public class Token {
    private final String text;
    private final int position;
    private final TokenType type;
    private final long timestamp;

    public Token(String text, int position, TokenType type) {
        this.text = text != null ? text : "";  // Handle null gracefully by converting to empty string
        this.position = position;
        this.type = type != null ? type : TokenType.UNKNOWN;  // Handle null type gracefully
        this.timestamp = System.nanoTime();
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

    public long getTimestamp() {
        return timestamp;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Token token)) return false;
        return position == token.position &&
               text.equals(token.text) &&
               type == token.type;
    }

    @Override
    public int hashCode() {
        return Objects.hash(text, position, type);
    }

    @Override
    public String toString() {
        return String.format("Token[%s,%d,%s]", text, position, type);
    }

    /**
     * Token types for classification.
     */
    public enum TokenType {
        WORD,
        PUNCTUATION,
        NUMBER,
        SYMBOL,
        WHITESPACE,
        SPECIAL,
        UNKNOWN
    }
}