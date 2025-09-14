package com.hellblazer.art.hartcq.core;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * Core stream processor for HART-CQ system.
 * Manages the 20-token sliding window and coordinates processing.
 */
public class StreamProcessor implements AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(StreamProcessor.class);
    
    private final SlidingWindow window;
    private final Tokenizer tokenizer;
    private final ExecutorService executor;
    private final List<Consumer<Token[]>> windowProcessors;
    private final List<Token[]> processedWindows;
    private final java.util.concurrent.atomic.AtomicInteger windowCount;
    private volatile boolean isProcessing;
    
    public StreamProcessor() {
        this(Runtime.getRuntime().availableProcessors());
    }
    
    public StreamProcessor(int threadPoolSize) {
        this.window = new SlidingWindow();
        this.tokenizer = new Tokenizer();
        this.executor = Executors.newFixedThreadPool(threadPoolSize);
        this.windowProcessors = new ArrayList<>();
        this.processedWindows = new ArrayList<>();
        this.windowCount = new java.util.concurrent.atomic.AtomicInteger(0);
        this.isProcessing = false;
    }
    
    /**
     * Processes a stream of text through the sliding window.
     * @param text The text to process
     * @return CompletableFuture that completes when processing is done
     */
    public CompletableFuture<ProcessingResult> processStream(String text) {
        return CompletableFuture.supplyAsync(() -> {
            isProcessing = true;
            var result = new ProcessingResult();
            
            try {
                var tokens = tokenizer.tokenize(text);
                result.setTotalTokens(tokens.size());
                
                for (Token token : tokens) {
                    if (!isProcessing) {
                        break;
                    }
                    
                    var removed = window.addToken(token);
                    
                    // Process window when it's full
                    if (window.isFull()) {
                        processWindow();
                        result.incrementWindowsProcessed();
                    }
                    
                    if (removed != null) {
                        result.addEvictedToken(removed);
                    }
                }
                
                // Process final partial window if needed
                if (window.size() > 0 && window.size() < SlidingWindow.WINDOW_SIZE) {
                    processWindow();
                    result.incrementWindowsProcessed();
                }
                
                result.setSuccessful(true);
            } catch (Exception e) {
                logger.error("Error processing stream", e);
                result.setSuccessful(false);
                result.setErrorMessage(e.getMessage());
            } finally {
                isProcessing = false;
            }
            
            return result;
        }, executor);
    }
    
    /**
     * Processes text in batches for improved performance.
     * @param text The text to process
     * @param batchSize Number of tokens to process in each batch
     * @return List of processing results for each batch
     */
    public List<ProcessingResult> processBatches(String text, int batchSize) {
        var results = new ArrayList<ProcessingResult>();
        var chunks = tokenizer.tokenizeInChunks(text, batchSize);
        
        for (var chunk : chunks) {
            var batchResult = new ProcessingResult();
            batchResult.setTotalTokens(chunk.size());
            
            for (Token token : chunk) {
                window.addToken(token);
                
                if (window.isFull()) {
                    processWindow();
                    batchResult.incrementWindowsProcessed();
                }
            }
            
            batchResult.setSuccessful(true);
            results.add(batchResult);
        }
        
        return results;
    }
    
    /**
     * Processes the current window state.
     */
    private void processWindow() {
        var windowArray = window.getWindowArray();
        
        // Store processed window
        processedWindows.add(windowArray);
        windowCount.incrementAndGet();
        
        // Notify all registered processors
        for (var processor : windowProcessors) {
            try {
                processor.accept(windowArray);
            } catch (Exception e) {
                logger.error("Error in window processor", e);
            }
        }
    }
    
    /**
     * Process a list of tokens through the sliding window.
     * @param tokens The tokens to process
     * @return CompletableFuture that completes when processing is done
     */
    public CompletableFuture<ProcessingResult> processTokens(List<Token> tokens) {
        return CompletableFuture.supplyAsync(() -> {
            isProcessing = true;
            var result = new ProcessingResult();
            
            try {
                result.setTotalTokens(tokens.size());
                
                for (Token token : tokens) {
                    if (!isProcessing) {
                        break;
                    }
                    
                    var removed = window.addToken(token);
                    
                    // Process window when it's full
                    if (window.isFull()) {
                        processWindow();
                        result.incrementWindowsProcessed();
                    }
                    
                    if (removed != null) {
                        result.addEvictedToken(removed);
                    }
                }
                
                // Process final partial window if needed
                if (window.size() > 0 && window.size() < SlidingWindow.WINDOW_SIZE) {
                    processWindow();
                    result.incrementWindowsProcessed();
                }
                
                result.setSuccessful(true);
            } catch (Exception e) {
                logger.error("Error processing tokens", e);
                result.setSuccessful(false);
                result.setErrorMessage(e.getMessage());
            } finally {
                isProcessing = false;
            }
            
            return result;
        }, executor);
    }
    
    /**
     * Registers a window processor that will be called for each full window.
     * @param processor The processor to register
     */
    public void registerWindowProcessor(Consumer<Token[]> processor) {
        windowProcessors.add(processor);
    }
    
    /**
     * Stops processing and clears the window.
     */
    public void stop() {
        isProcessing = false;
        window.clear();
    }
    
    /**
     * Gets the current window snapshot.
     * @return Current tokens in the window
     */
    public List<Token> getCurrentWindow() {
        return window.getWindowSnapshot();
    }
    
    /**
     * Gets the current window size.
     * @return Number of tokens in the window
     */
    public int getWindowSize() {
        return window.size();
    }
    
    /**
     * Gets total tokens processed.
     * @return Total number of tokens processed
     */
    public int getTotalTokensProcessed() {
        return window.getTotalTokensProcessed();
    }
    
    @Override
    public void close() {
        stop();
        executor.shutdown();
        try {
            if (!executor.awaitTermination(10, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
    
    /**
     * Result of stream processing operation.
     */
    public static class ProcessingResult {
        private int totalTokens;
        private int windowsProcessed;
        private List<Token> evictedTokens;
        private boolean successful;
        private String errorMessage;
        
        public ProcessingResult() {
            this.evictedTokens = new ArrayList<>();
            this.successful = false;
        }
        
        public int getTotalTokens() {
            return totalTokens;
        }
        
        public void setTotalTokens(int totalTokens) {
            this.totalTokens = totalTokens;
        }
        
        public int getWindowsProcessed() {
            return windowsProcessed;
        }
        
        public void incrementWindowsProcessed() {
            this.windowsProcessed++;
        }
        
        public List<Token> getEvictedTokens() {
            return evictedTokens;
        }
        
        public void addEvictedToken(Token token) {
            evictedTokens.add(token);
        }
        
        public boolean isSuccessful() {
            return successful;
        }
        
        public void setSuccessful(boolean successful) {
            this.successful = successful;
        }
        
        public String getErrorMessage() {
            return errorMessage;
        }
        
        public void setErrorMessage(String errorMessage) {
            this.errorMessage = errorMessage;
        }
    }
    
    /**
     * Reset the stream processor to initial state.
     */
    public void reset() {
        window.clear();
        processedWindows.clear();
        windowCount.set(0);
        logger.info("StreamProcessor reset");
    }
    
    /**
     * Get all processed windows.
     * @return List of processed token windows
     */
    public List<Token[]> getProcessedWindows() {
        return new ArrayList<>(processedWindows);
    }
    
    /**
     * Shutdown the stream processor.
     */
    public void shutdown() {
        executor.shutdown();
        processedWindows.clear();
        logger.info("StreamProcessor shutdown");
    }
}