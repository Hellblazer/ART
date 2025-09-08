package com.hellblazer.art.nlp.core;

import java.util.Objects;

/**
 * Represents a named entity extracted from text.
 * Thread-safe immutable class.
 */
public final class Entity {
    private final String text;
    private final String type;
    private final int startToken;
    private final int endToken;
    private final double confidence;

    public Entity(String text, String type, int startToken) {
        this(text, type, startToken, startToken, 1.0);
    }

    public Entity(String text, String type, int startToken, int endToken) {
        this(text, type, startToken, endToken, 1.0);
    }

    public Entity(String text, String type, int startToken, int endToken, double confidence) {
        this.text = Objects.requireNonNull(text, "text cannot be null");
        this.type = Objects.requireNonNull(type, "type cannot be null");
        this.startToken = startToken;
        this.endToken = endToken;
        this.confidence = confidence;
        
        if (startToken < 0) {
            throw new IllegalArgumentException("startToken must be non-negative");
        }
        if (endToken < startToken) {
            throw new IllegalArgumentException("endToken must be >= startToken");
        }
        if (confidence < 0.0 || confidence > 1.0) {
            throw new IllegalArgumentException("confidence must be in [0.0, 1.0]");
        }
    }

    public String getText() {
        return text;
    }

    public String getType() {
        return type;
    }

    public int getStartToken() {
        return startToken;
    }

    public int getEndToken() {
        return endToken;
    }

    public double getConfidence() {
        return confidence;
    }

    public int getLength() {
        return endToken - startToken + 1;
    }

    /**
     * Creates a new entity with extended token range.
     * Used for BIO tagging when continuing an entity.
     */
    public Entity addToken(String additionalText) {
        var newText = this.text + " " + additionalText;
        return new Entity(newText, type, startToken, endToken + 1, confidence);
    }

    /**
     * Creates a new entity with updated confidence score.
     */
    public Entity withConfidence(double newConfidence) {
        return new Entity(text, type, startToken, endToken, newConfidence);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Entity entity = (Entity) obj;
        return startToken == entity.startToken &&
               endToken == entity.endToken &&
               Double.compare(entity.confidence, confidence) == 0 &&
               Objects.equals(text, entity.text) &&
               Objects.equals(type, entity.type);
    }

    @Override
    public int hashCode() {
        return Objects.hash(text, type, startToken, endToken, confidence);
    }

    @Override
    public String toString() {
        return String.format("Entity{text='%s', type='%s', tokens=[%d-%d], confidence=%.3f}",
                           text, type, startToken, endToken, confidence);
    }
}