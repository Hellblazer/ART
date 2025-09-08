package com.hellblazer.art.nlp.config;

/**
 * Configuration for performance and memory management settings.
 */
public class PerformanceConfig {
    private int threadPoolSize = 8;
    private int maxCategoriesPerChannel = 10000;
    private int pruningInterval = 1000;  // patterns processed before pruning
    private double pruningThreshold = 0.01;  // minimum activation frequency
    private int embeddingCacheSize = 10000;
    private boolean enableParallelProcessing = true;
    private long processingTimeoutMs = 5000;
    
    // Getters and setters
    public int getThreadPoolSize() { 
        return threadPoolSize; 
    }
    
    public void setThreadPoolSize(int threadPoolSize) {
        if (threadPoolSize < 1) {
            throw new IllegalArgumentException("Thread pool size must be positive: " + threadPoolSize);
        }
        this.threadPoolSize = threadPoolSize; 
    }
    
    public int getMaxCategoriesPerChannel() { 
        return maxCategoriesPerChannel; 
    }
    
    public void setMaxCategoriesPerChannel(int maxCategoriesPerChannel) {
        if (maxCategoriesPerChannel < 1) {
            throw new IllegalArgumentException("Max categories per channel must be positive: " + maxCategoriesPerChannel);
        }
        this.maxCategoriesPerChannel = maxCategoriesPerChannel; 
    }
    
    public int getPruningInterval() { 
        return pruningInterval; 
    }
    
    public void setPruningInterval(int pruningInterval) {
        if (pruningInterval < 1) {
            throw new IllegalArgumentException("Pruning interval must be positive: " + pruningInterval);
        }
        this.pruningInterval = pruningInterval; 
    }
    
    public double getPruningThreshold() { 
        return pruningThreshold; 
    }
    
    public void setPruningThreshold(double pruningThreshold) {
        if (pruningThreshold < 0.0 || pruningThreshold > 1.0) {
            throw new IllegalArgumentException("Pruning threshold must be in [0.0, 1.0]: " + pruningThreshold);
        }
        this.pruningThreshold = pruningThreshold; 
    }
    
    public int getEmbeddingCacheSize() { 
        return embeddingCacheSize; 
    }
    
    public void setEmbeddingCacheSize(int embeddingCacheSize) {
        if (embeddingCacheSize < 0) {
            throw new IllegalArgumentException("Embedding cache size must be non-negative: " + embeddingCacheSize);
        }
        this.embeddingCacheSize = embeddingCacheSize; 
    }
    
    public boolean isEnableParallelProcessing() { 
        return enableParallelProcessing; 
    }
    
    public void setEnableParallelProcessing(boolean enableParallelProcessing) { 
        this.enableParallelProcessing = enableParallelProcessing; 
    }
    
    public long getProcessingTimeoutMs() { 
        return processingTimeoutMs; 
    }
    
    public void setProcessingTimeoutMs(long processingTimeoutMs) {
        if (processingTimeoutMs < 0) {
            throw new IllegalArgumentException("Processing timeout must be non-negative: " + processingTimeoutMs);
        }
        this.processingTimeoutMs = processingTimeoutMs; 
    }
    
    /**
     * Create default performance configuration.
     * 
     * @return default PerformanceConfig
     */
    public static PerformanceConfig defaults() {
        return new PerformanceConfig();
    }
    
    @Override
    public String toString() {
        return String.format("PerformanceConfig{threadPoolSize=%d, maxCategories=%d, pruningInterval=%d, pruningThreshold=%.3f, cacheSize=%d, parallelProcessing=%s, timeout=%dms}",
                           threadPoolSize, maxCategoriesPerChannel, pruningInterval, pruningThreshold, 
                           embeddingCacheSize, enableParallelProcessing, processingTimeoutMs);
    }
}