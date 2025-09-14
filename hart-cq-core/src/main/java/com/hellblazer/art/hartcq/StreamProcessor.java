package com.hellblazer.art.hartcq;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Core stream processor implementing 20-token sliding window mechanism.
 * This is the foundation of HART-CQ's deterministic text processing.
 */
public class StreamProcessor {
    private static final Logger log = LoggerFactory.getLogger(StreamProcessor.class);

    public static final int WINDOW_SIZE = 20;
    public static final int SLIDE_SIZE = 5; // Slide by 5 tokens for overlapping context

    private final Queue<Token> tokenBuffer;
    private final List<ProcessingWindow> activeWindows;
    private final ReentrantReadWriteLock windowLock;
    private final AtomicInteger processedTokenCount;
    private final WindowProcessor windowProcessor;

    // Performance metrics
    private long totalProcessingTime;
    private long windowCount;

    public StreamProcessor() {
        this.tokenBuffer = new ConcurrentLinkedQueue<>();
        this.activeWindows = new ArrayList<>();
        this.windowLock = new ReentrantReadWriteLock();
        this.processedTokenCount = new AtomicInteger(0);
        this.windowProcessor = new WindowProcessor();
        this.totalProcessingTime = 0;
        this.windowCount = 0;
    }

    /**
     * Add tokens to the stream for processing.
     */
    public void addTokens(List<Token> tokens) {
        tokenBuffer.addAll(tokens);
        processBufferedTokens();
    }

    /**
     * Process buffered tokens through sliding windows.
     */
    private void processBufferedTokens() {
        windowLock.writeLock().lock();
        try {
            while (tokenBuffer.size() >= WINDOW_SIZE) {
                var window = createWindow();
                if (window != null) {
                    processWindow(window);
                }
            }
        } finally {
            windowLock.writeLock().unlock();
        }
    }

    /**
     * Create a processing window from buffered tokens.
     */
    private ProcessingWindow createWindow() {
        if (tokenBuffer.size() < WINDOW_SIZE) {
            return null;
        }

        var windowTokens = new ArrayList<Token>(WINDOW_SIZE);
        var tempQueue = new LinkedList<>(tokenBuffer);

        // Extract window tokens
        for (int i = 0; i < WINDOW_SIZE && !tempQueue.isEmpty(); i++) {
            windowTokens.add(tempQueue.poll());
        }

        // Slide the buffer
        for (int i = 0; i < SLIDE_SIZE && !tokenBuffer.isEmpty(); i++) {
            tokenBuffer.poll();
            processedTokenCount.incrementAndGet();
        }

        return new ProcessingWindow(windowTokens, windowCount++);
    }

    /**
     * Process a single window through the pipeline.
     */
    private void processWindow(ProcessingWindow window) {
        long startTime = System.nanoTime();

        try {
            // Process through window processor
            var result = windowProcessor.process(window);

            // Store active window for hierarchical processing
            activeWindows.add(window);

            // Clean up old windows (keep last 10 for context)
            if (activeWindows.size() > 10) {
                activeWindows.remove(0);
            }

            window.setProcessingResult(result);

        } catch (Exception e) {
            log.error("Error processing window {}: {}", window.getWindowId(), e.getMessage());
        } finally {
            long processingTime = System.nanoTime() - startTime;
            totalProcessingTime += processingTime;
        }
    }

    /**
     * Get current processing statistics.
     */
    public ProcessingStats getStats() {
        windowLock.readLock().lock();
        try {
            return new ProcessingStats(
                processedTokenCount.get(),
                windowCount,
                totalProcessingTime,
                tokenBuffer.size(),
                activeWindows.size()
            );
        } finally {
            windowLock.readLock().unlock();
        }
    }

    /**
     * Clear all buffers and reset processor.
     */
    public void reset() {
        windowLock.writeLock().lock();
        try {
            tokenBuffer.clear();
            activeWindows.clear();
            processedTokenCount.set(0);
            windowCount = 0;
            totalProcessingTime = 0;
        } finally {
            windowLock.writeLock().unlock();
        }
    }

    /**
     * Get active windows for inspection.
     */
    public List<ProcessingWindow> getActiveWindows() {
        windowLock.readLock().lock();
        try {
            return new ArrayList<>(activeWindows);
        } finally {
            windowLock.readLock().unlock();
        }
    }

    /**
     * Processing statistics.
     */
    public record ProcessingStats(
        int processedTokens,
        long windowCount,
        long totalProcessingTimeNanos,
        int pendingTokens,
        int activeWindows
    ) {
        public double getAverageWindowProcessingTimeMs() {
            if (windowCount == 0) return 0;
            return (totalProcessingTimeNanos / 1_000_000.0) / windowCount;
        }

        public double getThroughputTokensPerSecond() {
            if (totalProcessingTimeNanos == 0) return 0;
            double seconds = totalProcessingTimeNanos / 1_000_000_000.0;
            return processedTokens / seconds;
        }
    }
}