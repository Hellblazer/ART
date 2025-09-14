package com.hellblazer.art.hartcq.core;

import java.util.*;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Implements a thread-safe sliding window for token processing.
 * The window maintains exactly 20 tokens for processing.
 */
public class SlidingWindow {
    public static final int WINDOW_SIZE = 20;
    
    private final LinkedList<Token> window;
    private final ReadWriteLock lock;
    private int totalTokensProcessed;
    
    public SlidingWindow() {
        this.window = new LinkedList<>();
        this.lock = new ReentrantReadWriteLock();
        this.totalTokensProcessed = 0;
    }
    
    /**
     * Adds a token to the window. If window is full, removes the oldest token.
     * @param token The token to add
     * @return The removed token if window was full, null otherwise
     */
    public Token addToken(Token token) {
        lock.writeLock().lock();
        try {
            Token removed = null;
            if (window.size() >= WINDOW_SIZE) {
                removed = window.removeFirst();
            }
            window.addLast(token);
            totalTokensProcessed++;
            return removed;
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Gets a snapshot of the current window contents.
     * @return An unmodifiable list of tokens in the window
     */
    public List<Token> getWindowSnapshot() {
        lock.readLock().lock();
        try {
            return Collections.unmodifiableList(new ArrayList<>(window));
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Gets the current window as an array for vectorization.
     * @return Array of tokens, padded with nulls if less than WINDOW_SIZE
     */
    public Token[] getWindowArray() {
        lock.readLock().lock();
        try {
            var array = new Token[WINDOW_SIZE];
            int i = 0;
            for (Token token : window) {
                array[i++] = token;
            }
            return array;
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Checks if the window is full (contains WINDOW_SIZE tokens).
     * @return true if window is full
     */
    public boolean isFull() {
        lock.readLock().lock();
        try {
            return window.size() == WINDOW_SIZE;
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Gets the current size of the window.
     * @return Number of tokens currently in the window
     */
    public int size() {
        lock.readLock().lock();
        try {
            return window.size();
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Clears the window.
     */
    public void clear() {
        lock.writeLock().lock();
        try {
            window.clear();
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Gets the total number of tokens processed.
     * @return Total tokens processed since creation
     */
    public int getTotalTokensProcessed() {
        lock.readLock().lock();
        try {
            return totalTokensProcessed;
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Gets the token at a specific position in the window.
     * @param index The index (0 = oldest, size-1 = newest)
     * @return The token at the index, or null if index is out of bounds
     */
    public Token getTokenAt(int index) {
        lock.readLock().lock();
        try {
            if (index < 0 || index >= window.size()) {
                return null;
            }
            return window.get(index);
        } finally {
            lock.readLock().unlock();
        }
    }
}