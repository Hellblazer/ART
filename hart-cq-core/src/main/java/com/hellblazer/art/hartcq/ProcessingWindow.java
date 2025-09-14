package com.hellblazer.art.hartcq;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Represents a 20-token processing window.
 * Thread-safe container for tokens being processed together.
 */
public class ProcessingWindow {
    private final List<Token> tokens;
    private final long windowId;
    private final long creationTime;
    private final AtomicReference<WindowResult> processingResult;

    public ProcessingWindow(List<Token> tokens, long windowId) {
        this.tokens = Collections.unmodifiableList(new ArrayList<>(tokens));
        this.windowId = windowId;
        this.creationTime = System.nanoTime();
        this.processingResult = new AtomicReference<>();
    }

    public List<Token> getTokens() {
        return tokens;
    }

    public long getWindowId() {
        return windowId;
    }

    public long getCreationTime() {
        return creationTime;
    }

    public void setProcessingResult(WindowResult result) {
        processingResult.set(result);
    }

    public WindowResult getProcessingResult() {
        return processingResult.get();
    }

    public boolean isProcessed() {
        return processingResult.get() != null;
    }

    /**
     * Get text representation of the window.
     */
    public String getText() {
        var sb = new StringBuilder();
        for (var token : tokens) {
            if (token.getType() != Token.TokenType.WHITESPACE) {
                if (sb.length() > 0) sb.append(" ");
                sb.append(token.getText());
            }
        }
        return sb.toString();
    }

    /**
     * Get token at specific position.
     */
    public Token getTokenAt(int index) {
        if (index < 0 || index >= tokens.size()) {
            return null;
        }
        return tokens.get(index);
    }

    /**
     * Get window size.
     */
    public int size() {
        return tokens.size();
    }

    @Override
    public String toString() {
        return String.format("Window[id=%d, size=%d, processed=%b]",
            windowId, tokens.size(), isProcessed());
    }
}