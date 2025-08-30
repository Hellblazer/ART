package com.hellblazer.art.core;

import java.util.Optional;

/**
 * Cache for storing intermediate computation results during ART processing.
 * 
 * This mirrors the cache system used in reference parity for 
 * optimization of repeated computations within the same step_fit call.
 */
public final class ActivationCache {
    
    private final Object data;
    private final String algorithmType;
    
    private ActivationCache(Object data, String algorithmType) {
        this.data = data;
        this.algorithmType = algorithmType;
    }
    
    /**
     * Create a new cache with the specified data.
     */
    public static ActivationCache of(Object data, String algorithmType) {
        return new ActivationCache(data, algorithmType);
    }
    
    /**
     * Create an empty cache.
     */
    public static ActivationCache empty(String algorithmType) {
        return new ActivationCache(null, algorithmType);
    }
    
    /**
     * Get the cached data if present.
     */
    public Optional<Object> getData() {
        return Optional.ofNullable(data);
    }
    
    /**
     * Get the cached data cast to a specific type.
     */
    @SuppressWarnings("unchecked")
    public <T> Optional<T> getData(Class<T> type) {
        if (data != null && type.isInstance(data)) {
            return Optional.of((T) data);
        }
        return Optional.empty();
    }
    
    /**
     * Get the algorithm type this cache is for.
     */
    public String getAlgorithmType() {
        return algorithmType;
    }
    
    /**
     * Check if this cache has data.
     */
    public boolean hasData() {
        return data != null;
    }
    
    /**
     * Create a new cache with different data but same algorithm type.
     */
    public ActivationCache withData(Object newData) {
        return new ActivationCache(newData, algorithmType);
    }
    
    @Override
    public String toString() {
        return String.format("ActivationCache{algorithm=%s, hasData=%s}", 
                           algorithmType, hasData());
    }
}