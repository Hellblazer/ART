package com.hellblazer.art.nlp.channels;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.algorithms.FuzzyART;
import com.hellblazer.art.core.parameters.FuzzyParameters;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.nlp.channels.base.BaseChannel;
import com.hellblazer.art.nlp.fasttext.FastTextModel;
import com.hellblazer.art.nlp.fasttext.VectorPreprocessor;
import com.hellblazer.art.nlp.fasttext.VectorPreprocessor.PreprocessingPipeline;

/**
 * Semantic channel using FastText word embeddings with ART clustering.
 * Processes text into semantic vector representations for pattern recognition.
 */
public final class FastTextChannel extends BaseChannel {
    private static final Logger log = LoggerFactory.getLogger(FastTextChannel.class);

    // FastText integration
    private final FastTextModel fastTextModel;
    private final PreprocessingPipeline preprocessingPipeline;
    
    // Text processing configuration
    private final OOVStrategy oovStrategy;
    private final boolean useSubwordFallback;
    private final int maxTokensPerInput;
    
    // ART algorithm for semantic clustering
    private final FuzzyART fuzzyART;
    private final FuzzyParameters artParameters;
    
    // Performance tracking
    private final AtomicInteger totalTokens = new AtomicInteger(0);
    private final AtomicInteger oovTokens = new AtomicInteger(0);
    private final AtomicInteger successfulClassifications = new AtomicInteger(0);

    /**
     * Strategy for handling out-of-vocabulary words.
     */
    public enum OOVStrategy {
        /** Skip OOV words entirely */
        SKIP,
        /** Use zero vector for OOV words */
        ZERO_VECTOR,
        /** Use random vector for OOV words */
        RANDOM_VECTOR,
        /** Use average of known words as fallback */
        AVERAGE_FALLBACK
    }

    /**
     * Create FastText channel with default configuration.
     */
    public FastTextChannel(String channelName, double vigilance, Path fastTextModelPath) throws IOException {
        this(channelName, vigilance, fastTextModelPath, 300,
             OOVStrategy.RANDOM_VECTOR, true, 100,
             VectorPreprocessor.pipeline()
                 .normalize(VectorPreprocessor.NormalizationType.L2)
                 .complementCode()
                 .build());
    }

    /**
     * Create FastText channel with specified dimensions.
     */
    public FastTextChannel(String channelName, double vigilance, Path fastTextModelPath, int dimensions) throws IOException {
        this(channelName, vigilance, fastTextModelPath, dimensions,
             OOVStrategy.RANDOM_VECTOR, true, 100,
             VectorPreprocessor.pipeline()
                 .normalize(VectorPreprocessor.NormalizationType.L2)
                 .complementCode()
                 .build());
    }

    /**
     * Create FastText channel with custom configuration.
     */
    public FastTextChannel(String channelName, double vigilance, Path fastTextModelPath, int dimensions,
                          OOVStrategy oovStrategy, boolean useSubwordFallback, int maxTokensPerInput,
                          PreprocessingPipeline preprocessingPipeline) throws IOException {
        super(channelName, vigilance);
        
        this.oovStrategy = Objects.requireNonNull(oovStrategy, "oovStrategy cannot be null");
        this.useSubwordFallback = useSubwordFallback;
        this.maxTokensPerInput = maxTokensPerInput;
        this.preprocessingPipeline = Objects.requireNonNull(preprocessingPipeline, "preprocessingPipeline cannot be null");
        
        // Initialize FastText model
        this.fastTextModel = new FastTextModel(fastTextModelPath, dimensions, true, 1000);
        this.fastTextModel.initialize();
        
        // Initialize ART algorithm for semantic clustering
        this.fuzzyART = new FuzzyART();
        this.artParameters = FuzzyParameters.of(vigilance, 0.001, 0.6); // vigilance, choice, learning rate
        
        log.info("FastText channel '{}' created: oov={}, subword={}, maxTokens={}, pipeline={}, vigilance={}",
                channelName, oovStrategy, useSubwordFallback, maxTokensPerInput, 
                preprocessingPipeline.getStepCount(), vigilance);
    }

    @Override
    public int classify(DenseVector input) {
        if (input == null) {
            return -1;
        }
        
        var startTime = System.currentTimeMillis();
        try {
            getReadLock().lock();
            
            // Apply preprocessing (normalization + complement coding)
            var processedInput = preprocessInput(input);
            
            // Use FuzzyART to classify the vector
            var result = fuzzyART.stepFit(processedInput, artParameters);
            
            if (result instanceof ActivationResult.Success success) {
                var processingTime = System.currentTimeMillis() - startTime;
                recordClassification(processingTime, false); // Assume existing category for now
                successfulClassifications.incrementAndGet();
                return success.categoryIndex();
            } else {
                log.debug("ART classification failed for input vector of size {}", input.dimension());
                recordError();
                return -1;
            }
            
        } catch (Exception e) {
            log.error("Error classifying vector in channel '{}': {}", getChannelName(), e.getMessage());
            recordError();
            return -1;
        } finally {
            getReadLock().unlock();
        }
    }

    /**
     * Process text input into semantic vector for classification.
     * 
     * @param text Input text to process
     * @return Semantic category ID
     */
    public int classifyText(String text) {
        if (text == null || text.isBlank()) {
            return -1; // Invalid input
        }

        // Tokenize and convert to vectors
        var tokens = tokenize(text);
        if (tokens.isEmpty()) {
            return -1;
        }

        // Get word vectors
        var vectors = new ArrayList<DenseVector>();
        var knownTokenCount = 0;
        
        for (var token : tokens) {
            var vector = fastTextModel.getWordVector(token);
            totalTokens.incrementAndGet();
            
            if (vector != null) {
                vectors.add(vector);
                knownTokenCount++;
            } else {
                oovTokens.incrementAndGet();
                var fallbackVector = handleOOV(token, vectors);
                if (fallbackVector != null) {
                    vectors.add(fallbackVector);
                }
            }
        }

        if (vectors.isEmpty()) {
            log.debug("No valid vectors found for text: {}", text);
            return -1;
        }

        // Aggregate vectors into single representation
        var aggregatedVector = VectorPreprocessor.aggregateMean(vectors);
        if (aggregatedVector == null) {
            return -1;
        }

        // Apply preprocessing pipeline
        var processedVector = preprocessingPipeline.apply(aggregatedVector);
        if (processedVector == null) {
            log.debug("Preprocessing failed for text: {}", text);
            return -1;
        }

        // Classify using base ART algorithm
        var category = classify(processedVector);
        
        if (category >= 0) {
            successfulClassifications.incrementAndGet();
        }
        
        return category;
    }

    /**
     * Handle out-of-vocabulary words based on configured strategy.
     */
    private DenseVector handleOOV(String token, List<DenseVector> existingVectors) {
        return switch (oovStrategy) {
            case SKIP -> null;
            case ZERO_VECTOR -> fastTextModel.getZeroVector();
            case RANDOM_VECTOR -> fastTextModel.getRandomVector();
            case AVERAGE_FALLBACK -> {
                if (existingVectors.isEmpty()) {
                    yield fastTextModel.getRandomVector();
                } else {
                    yield VectorPreprocessor.aggregateMean(existingVectors);
                }
            }
        };
    }

    /**
     * Simple whitespace tokenization with basic cleanup.
     * Override for more sophisticated tokenization.
     */
    protected List<String> tokenize(String text) {
        var tokens = text.toLowerCase()
                        .replaceAll("[^a-zA-Z0-9\\s]", " ")
                        .trim()
                        .split("\\s+");
        
        var result = new ArrayList<String>();
        for (var token : tokens) {
            if (!token.isBlank() && result.size() < maxTokensPerInput) {
                result.add(token);
            }
        }
        
        return result;
    }

    /**
     * Batch classify multiple texts efficiently.
     */
    public List<Integer> classifyTexts(List<String> texts) {
        return texts.stream()
                   .map(this::classifyText)
                   .toList();
    }

    /**
     * Get semantic similarity between two texts.
     * Returns cosine similarity of their aggregated vectors.
     */
    public double getTextSimilarity(String text1, String text2) {
        var vector1 = getTextVector(text1);
        var vector2 = getTextVector(text2);
        
        if (vector1 == null || vector2 == null) {
            return 0.0;
        }
        
        return cosineSimilarity(vector1, vector2);
    }

    /**
     * Get processed vector representation of text.
     */
    public DenseVector getTextVector(String text) {
        if (text == null || text.isBlank()) {
            return null;
        }

        var tokens = tokenize(text);
        if (tokens.isEmpty()) {
            return null;
        }

        var vectors = new ArrayList<DenseVector>();
        for (var token : tokens) {
            var vector = fastTextModel.getWordVector(token);
            if (vector != null) {
                vectors.add(vector);
            } else {
                var fallbackVector = handleOOV(token, vectors);
                if (fallbackVector != null) {
                    vectors.add(fallbackVector);
                }
            }
        }

        if (vectors.isEmpty()) {
            return null;
        }

        var aggregated = VectorPreprocessor.aggregateMean(vectors);
        return preprocessingPipeline.apply(aggregated);
    }

    /**
     * Calculate cosine similarity between two vectors.
     */
    private double cosineSimilarity(DenseVector v1, DenseVector v2) {
        if (v1.dimension() != v2.dimension()) {
            return 0.0;
        }

        var values1 = v1.values();
        var values2 = v2.values();
        
        var dotProduct = 0.0;
        var norm1 = 0.0;
        var norm2 = 0.0;

        for (int i = 0; i < values1.length; i++) {
            dotProduct += values1[i] * values2[i];
            norm1 += values1[i] * values1[i];
            norm2 += values2[i] * values2[i];
        }

        var denominator = Math.sqrt(norm1) * Math.sqrt(norm2);
        return denominator > 0 ? dotProduct / denominator : 0.0;
    }

    /**
     * Get channel-specific performance metrics.
     */
    public FastTextMetrics getFastTextMetrics() {
        var baseMetrics = getMetrics();
        var modelStats = fastTextModel.getStats();
        
        return new FastTextMetrics(
            baseMetrics.getTotalClassifications(),
            baseMetrics.getTotalClassifications(), // No separate successful count in ChannelMetrics
            baseMetrics.getCurrentCategoryCount(),
            baseMetrics.getAverageProcessingTimeMs(),
            totalTokens.get(),
            oovTokens.get(),
            successfulClassifications.get(),
            modelStats.cacheHitRate(),
            modelStats.oovRate(),
            modelStats.cacheSize()
        );
    }

    @Override
    protected void performInitialization() {
        try {
            fastTextModel.initialize();
            log.info("FastText model initialized for channel '{}'", getChannelName());
        } catch (Exception e) {
            log.error("Failed to initialize FastText model for channel '{}'", getChannelName(), e);
            throw new RuntimeException("FastText initialization failed", e);
        }
    }

    @Override
    protected void performCleanup() {
        try {
            fastTextModel.close();
        } catch (Exception e) {
            log.warn("Error closing FastText model for channel '{}'", getChannelName(), e);
        }
    }

    /**
     * Get FastText channel description.
     */
    public String getDescription() {
        return String.format("FastTextChannel{name='%s', vigilance=%.3f, categories=%d, oov=%s, tokens=%d/%d}",
                getChannelName(), getVigilance(), getCategoryCount(), oovStrategy,
                oovTokens.get(), totalTokens.get());
    }

    @Override
    public void saveState() {
        getWriteLock().lock();
        try {
            // Create a serializable state object
            var stateFile = Path.of("state", "channels", getChannelName() + ".state");
            Files.createDirectories(stateFile.getParent());
            
            var stateData = Map.<String, Object>of(
                "totalTokens", totalTokens.get(),
                "oovTokens", oovTokens.get(),
                "successfulClassifications", successfulClassifications.get(),
                "categoryCount", fuzzyART.getCategoryCount(),
                "timestamp", System.currentTimeMillis()
            );
            
            // Write state to file using simple serialization
            try (var fos = Files.newOutputStream(stateFile);
                 var oos = new ObjectOutputStream(fos)) {
                oos.writeObject(stateData);
            }
            
            log.debug("Saved FastText channel '{}' state: {} tokens processed, {} categories, {} OOV tokens", 
                     getChannelName(), totalTokens.get(), fuzzyART.getCategoryCount(), oovTokens.get());
            
        } catch (Exception e) {
            log.error("Failed to save state for FastText channel '{}': {}", getChannelName(), e.getMessage());
        } finally {
            getWriteLock().unlock();
        }
    }

    @Override
    public void loadState() {
        getWriteLock().lock();
        try {
            var stateFile = Path.of("state", "channels", getChannelName() + ".state");
            
            if (Files.exists(stateFile)) {
                // Load state from file using simple deserialization
                try (var fis = Files.newInputStream(stateFile);
                     var ois = new ObjectInputStream(fis)) {
                    
                    @SuppressWarnings("unchecked")
                    var stateData = (Map<String, Object>) ois.readObject();
                    
                    // Restore performance counters if available
                    if (stateData.containsKey("totalTokens")) {
                        totalTokens.set((Integer) stateData.get("totalTokens"));
                    }
                    if (stateData.containsKey("oovTokens")) {
                        oovTokens.set((Integer) stateData.get("oovTokens"));
                    }
                    if (stateData.containsKey("successfulClassifications")) {
                        successfulClassifications.set((Integer) stateData.get("successfulClassifications"));
                    }
                    
                    log.info("Loaded state for FastText channel '{}': {} tokens, {} OOV tokens, {} categories", 
                            getChannelName(), totalTokens.get(), oovTokens.get(), 
                            stateData.getOrDefault("categoryCount", 0));
                } catch (ClassNotFoundException e) {
                    log.warn("Failed to deserialize state for FastText channel '{}', starting fresh", getChannelName());
                    initializeCleanState();
                }
            } else {
                // Initialize with clean state if no saved state exists
                initializeCleanState();
                log.debug("No saved state found for FastText channel '{}', starting fresh", getChannelName());
            }
            
        } catch (Exception e) {
            log.error("Failed to load state for FastText channel '{}': {}", getChannelName(), e.getMessage());
            initializeCleanState();
        } finally {
            getWriteLock().unlock();
        }
    }
    
    private void initializeCleanState() {
        totalTokens.set(0);
        oovTokens.set(0);
        successfulClassifications.set(0);
    }

    @Override
    public int getCategoryCount() {
        getReadLock().lock();
        try {
            return fuzzyART.getCategoryCount();
        } finally {
            getReadLock().unlock();
        }
    }

    @Override
    public int pruneCategories(double threshold) {
        getWriteLock().lock();
        try {
            // Use threshold as minimum usage ratio for pruning
            var pruned = fuzzyART.pruneByUsageFrequency(threshold);
            log.debug("Pruned {} categories from FastText channel '{}' with threshold {}", 
                     pruned, getChannelName(), threshold);
            return pruned;
        } finally {
            getWriteLock().unlock();
        }
    }

    /**
     * FastText-specific performance metrics.
     */
    public record FastTextMetrics(
        long totalClassifications,
        long successfulClassifications,
        int categoryCount,
        double averageProcessingTime,
        int totalTokens,
        int oovTokens,
        int successfulTextClassifications,
        double cacheHitRate,
        double oovRate,
        int cacheSize
    ) {
        public double successRate() {
            return totalClassifications > 0 ? 
                (double) successfulClassifications / totalClassifications : 0.0;
        }

        public double textSuccessRate() {
            var textAttempts = successfulTextClassifications + 
                              (totalClassifications - successfulClassifications);
            return textAttempts > 0 ? 
                (double) successfulTextClassifications / textAttempts : 0.0;
        }

        public double oovRateByTokens() {
            return totalTokens > 0 ? (double) oovTokens / totalTokens : 0.0;
        }
    }
}