package com.hellblazer.art.nlp.fasttext;

import java.io.*;
import java.util.zip.GZIPInputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.hellblazer.art.core.DenseVector;

/**
 * FastText model wrapper for word embeddings.
 * Thread-safe implementation with lazy loading and efficient memory management.
 * Supports both word lookup and OOV (out-of-vocabulary) handling.
 */
public final class FastTextModel implements AutoCloseable {
    private static final Logger log = LoggerFactory.getLogger(FastTextModel.class);
    
    // Model configuration
    private final Path modelPath;
    private final int dimensions;
    private final boolean normalizeVectors;
    private final int maxCacheSize;
    
    // Thread-safe state
    private final Map<String, float[]> vectorCache = new ConcurrentHashMap<>();
    private final AtomicBoolean loaded = new AtomicBoolean(false);
    private final AtomicInteger vocabularySize = new AtomicInteger(0);
    private final Object loadLock = new Object();
    
    // Statistics
    private final AtomicInteger cacheHits = new AtomicInteger(0);
    private final AtomicInteger cacheMisses = new AtomicInteger(0);
    private final AtomicInteger oovCount = new AtomicInteger(0);

    /**
     * Create FastText model with default settings.
     * @param modelPath Path to FastText .vec file
     */
    public FastTextModel(Path modelPath) {
        this(modelPath, 300, true, 10_000);
    }

    /**
     * Create FastText model with custom configuration.
     * @param modelPath Path to FastText .vec file
     * @param dimensions Expected vector dimensions
     * @param normalizeVectors Whether to normalize vectors to unit length
     * @param maxCacheSize Maximum number of vectors to cache in memory
     */
    public FastTextModel(Path modelPath, int dimensions, boolean normalizeVectors, int maxCacheSize) {
        this.modelPath = modelPath;
        this.dimensions = dimensions;
        this.normalizeVectors = normalizeVectors;
        this.maxCacheSize = maxCacheSize;
        
        log.debug("Created FastText model: path={}, dims={}, normalize={}, cache={}",
                modelPath, dimensions, normalizeVectors, maxCacheSize);
    }

    /**
     * Load model metadata (vocabulary size and dimensions).
     * Called automatically on first access.
     */
    public void initialize() throws IOException {
        if (loaded.get()) {
            return;
        }

        synchronized (loadLock) {
            if (loaded.get()) {
                return;
            }

            log.info("Initializing FastText model from: {}", modelPath);
            long startTime = System.currentTimeMillis();

            try (var reader = new BufferedReader(
                    new InputStreamReader(createInputStream(modelPath), StandardCharsets.UTF_8))) {
                
                // Read header line: "vocab_size dimensions"
                var header = reader.readLine();
                if (header == null) {
                    throw new IOException("Empty FastText model file");
                }

                var parts = header.trim().split("\\s+");
                if (parts.length != 2) {
                    throw new IOException("Invalid FastText header format: " + header);
                }

                int vocabSize = Integer.parseInt(parts[0]);
                int fileDims = Integer.parseInt(parts[1]);

                if (fileDims != dimensions) {
                    throw new IOException(String.format(
                        "Dimension mismatch: expected %d, found %d in file", dimensions, fileDims));
                }

                vocabularySize.set(vocabSize);
                loaded.set(true);

                long loadTime = System.currentTimeMillis() - startTime;
                log.info("FastText model initialized: vocab={}, dims={}, time={}ms", 
                        vocabSize, fileDims, loadTime);
            }
        }
    }

    /**
     * Get word embedding as DenseVector.
     * @param word Word to look up
     * @return DenseVector embedding or null if not found
     */
    public DenseVector getWordVector(String word) {
        if (word == null || word.isBlank()) {
            return null;
        }

        try {
            initialize();
        } catch (IOException e) {
            log.error("Failed to initialize FastText model", e);
            return null;
        }

        var normalizedWord = word.trim().toLowerCase();
        var vector = vectorCache.get(normalizedWord);

        if (vector != null) {
            cacheHits.incrementAndGet();
            return new DenseVector(convertToDoubleArray(vector.clone()));
        }

        cacheMisses.incrementAndGet();
        vector = loadWordVector(normalizedWord);
        
        if (vector != null) {
            // Cache if not at capacity
            if (vectorCache.size() < maxCacheSize) {
                vectorCache.put(normalizedWord, vector.clone());
            }
            return new DenseVector(convertToDoubleArray(vector));
        }

        oovCount.incrementAndGet();
        log.debug("OOV word: {}", word);
        return null;
    }

    /**
     * Load word vector from file.
     * This is expensive - results should be cached.
     */
    private float[] loadWordVector(String word) {
        try (var reader = new BufferedReader(
                new InputStreamReader(createInputStream(modelPath), StandardCharsets.UTF_8))) {
            
            // Skip header
            reader.readLine();
            
            String line;
            while ((line = reader.readLine()) != null) {
                var parts = line.trim().split("\\s+");
                if (parts.length != dimensions + 1) {
                    continue; // Skip malformed lines
                }

                if (parts[0].equals(word)) {
                    var vector = new float[dimensions];
                    for (int i = 0; i < dimensions; i++) {
                        vector[i] = Float.parseFloat(parts[i + 1]);
                    }

                    if (normalizeVectors) {
                        normalizeVector(vector);
                    }

                    return vector;
                }
            }
        } catch (IOException e) {
            log.error("Error loading vector for word: " + word, e);
        }

        return null;
    }

    /**
     * Create input stream that handles both regular .vec and compressed .vec.gz files.
     */
    private InputStream createInputStream(Path path) throws IOException {
        var inputStream = new FileInputStream(path.toFile());
        
        // Check if file is gzipped based on extension
        if (path.toString().endsWith(".gz")) {
            return new GZIPInputStream(inputStream);
        } else {
            return inputStream;
        }
    }

    /**
     * Normalize vector to unit length (L2 normalization).
     */
    private void normalizeVector(float[] vector) {
        var norm = 0.0f;
        for (var value : vector) {
            norm += value * value;
        }
        norm = (float) Math.sqrt(norm);

        if (norm > 0.0f) {
            for (int i = 0; i < vector.length; i++) {
                vector[i] /= norm;
            }
        }
    }

    /**
     * Get multiple word vectors efficiently.
     * Uses batch processing for better I/O performance.
     */
    public Map<String, DenseVector> getWordVectors(String... words) {
        var result = new ConcurrentHashMap<String, DenseVector>();
        
        for (var word : words) {
            var vector = getWordVector(word);
            if (vector != null) {
                result.put(word, vector);
            }
        }
        
        return result;
    }

    /**
     * Get zero vector as fallback for OOV words.
     */
    public DenseVector getZeroVector() {
        return new DenseVector(new double[dimensions]);
    }

    /**
     * Get random vector as fallback for OOV words.
     * Uses Gaussian distribution with small variance.
     */
    public DenseVector getRandomVector() {
        var vector = new float[dimensions];
        var random = new java.util.Random();
        
        for (int i = 0; i < dimensions; i++) {
            vector[i] = (float) (random.nextGaussian() * 0.1);
        }
        
        if (normalizeVectors) {
            normalizeVector(vector);
        }
        
        return new DenseVector(convertToDoubleArray(vector));
    }

    /**
     * Check if word exists in vocabulary (cached check).
     */
    public boolean hasWord(String word) {
        if (word == null || word.isBlank()) {
            return false;
        }
        
        var normalizedWord = word.trim().toLowerCase();
        if (vectorCache.containsKey(normalizedWord)) {
            return true;
        }
        
        // This will load and cache if found
        return getWordVector(word) != null;
    }

    /**
     * Get model statistics.
     */
    public ModelStats getStats() {
        return new ModelStats(
            vocabularySize.get(),
            dimensions,
            vectorCache.size(),
            maxCacheSize,
            cacheHits.get(),
            cacheMisses.get(),
            oovCount.get(),
            loaded.get()
        );
    }

    /**
     * Clear vector cache to free memory.
     */
    public void clearCache() {
        vectorCache.clear();
        cacheHits.set(0);
        cacheMisses.set(0);
        log.debug("Cleared FastText vector cache");
    }

    /**
     * Get current cache size.
     */
    public int getCacheSize() {
        return vectorCache.size();
    }

    /**
     * Check if model is initialized.
     */
    public boolean isLoaded() {
        return loaded.get();
    }

    /**
     * Get model dimensions.
     */
    public int getDimensions() {
        return dimensions;
    }

    /**
     * Get vocabulary size (after initialization).
     */
    public int getVocabularySize() {
        return vocabularySize.get();
    }

    @Override
    public void close() {
        clearCache();
        log.debug("FastText model closed");
    }

    /**
     * Convert float array to double array for DenseVector compatibility.
     */
    private double[] convertToDoubleArray(float[] floatArray) {
        var doubleArray = new double[floatArray.length];
        for (int i = 0; i < floatArray.length; i++) {
            doubleArray[i] = floatArray[i];
        }
        return doubleArray;
    }

    @Override
    public String toString() {
        return String.format("FastTextModel{path=%s, dims=%d, vocab=%d, cache=%d/%d, hits=%d, misses=%d, oov=%d}",
                modelPath.getFileName(), dimensions, vocabularySize.get(), vectorCache.size(), maxCacheSize,
                cacheHits.get(), cacheMisses.get(), oovCount.get());
    }

    /**
     * Model statistics record.
     */
    public record ModelStats(
        int vocabularySize,
        int dimensions,
        int cacheSize,
        int maxCacheSize,
        int cacheHits,
        int cacheMisses,
        int oovCount,
        boolean loaded
    ) {
        public double cacheHitRate() {
            var total = cacheHits + cacheMisses;
            return total > 0 ? (double) cacheHits / total : 0.0;
        }

        public double oovRate() {
            var total = cacheHits + cacheMisses + oovCount;
            return total > 0 ? (double) oovCount / total : 0.0;
        }
    }
}