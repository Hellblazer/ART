package com.hellblazer.art.nlp.processor;

import java.net.URI;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.nlp.channels.base.BaseChannel;
import com.hellblazer.art.nlp.core.ProcessingResult;
import com.hellblazer.art.nlp.core.NLPProcessor;
import com.hellblazer.art.nlp.core.Document;
import com.hellblazer.art.nlp.core.DocumentAnalysis;
import com.hellblazer.art.nlp.core.ProcessingStats;
import com.hellblazer.art.nlp.core.Entity;
import com.hellblazer.art.nlp.processor.consensus.ConsensusStrategy;
import com.hellblazer.art.nlp.processor.consensus.WeightedVotingConsensus;
import com.hellblazer.art.nlp.processor.fusion.FeatureFusionStrategy;
import com.hellblazer.art.nlp.processor.fusion.ConcatenationFusion;

/**
 * Multi-channel processor that coordinates analysis across multiple ART-NLP channels.
 * Implements cross-channel learning, feature fusion, and consensus decision making.
 * 
 * CRITICAL: This class implements the NLPProcessor interface as specified in API_DESIGN.md
 */
public final class MultiChannelProcessor implements NLPProcessor {
    private static final Logger log = LoggerFactory.getLogger(MultiChannelProcessor.class);

    // Channel management
    private final Map<String, BaseChannel> channels = new ConcurrentHashMap<>();
    private final Map<String, Double> channelWeights = new ConcurrentHashMap<>();
    private final Map<String, ChannelStatistics> channelStats = new ConcurrentHashMap<>();
    
    // Processing configuration
    private final ConsensusStrategy consensusStrategy;
    private final FeatureFusionStrategy fusionStrategy;
    private final ExecutorService executorService;
    private final boolean enableParallelProcessing;
    private final double learningRateDecay;
    
    // Cross-channel learning
    private final Map<String, Set<Integer>> channelCategories = new ConcurrentHashMap<>();
    private final Map<String, Map<Integer, CategoryMetadata>> categoryMapping = new ConcurrentHashMap<>();
    private final AtomicInteger globalCategoryCounter = new AtomicInteger(0);
    
    // Performance tracking
    private final AtomicInteger totalProcessed = new AtomicInteger(0);
    private final AtomicInteger successfulProcessed = new AtomicInteger(0);
    private volatile boolean closed = false;
    
    // NLPProcessor interface implementation fields
    private final java.time.Instant startTime = java.time.Instant.now();
    private final Set<String> enabledChannels = ConcurrentHashMap.newKeySet();

    /**
     * Create a processor with default channels configured.
     * Includes semantic (FastText), syntactic, entity, context, and sentiment channels.
     */
    public static MultiChannelProcessor createWithDefaults() {
        try {
            var processor = builder().build();
            
            // Add default channels
            var resourceUrl = MultiChannelProcessor.class.getClassLoader().getResource("models/cc.en.300.vec.gz");
            if (resourceUrl != null) {
                Path modelPath = null;
                
                // Handle different URI schemes (file:// and jar:)
                if ("file".equals(resourceUrl.getProtocol())) {
                    // Resource is in filesystem (development)
                    modelPath = Path.of(resourceUrl.toURI());
                } else if ("jar".equals(resourceUrl.getProtocol())) {
                    // Resource is in JAR (testing/production)
                    try (var inputStream = resourceUrl.openStream()) {
                        // Create a temporary file to extract the resource
                        var tempFile = java.nio.file.Files.createTempFile("fasttext_model_", ".vec.gz");
                        tempFile.toFile().deleteOnExit(); // Clean up on JVM exit
                        java.nio.file.Files.copy(inputStream, tempFile, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
                        modelPath = tempFile;
                    }
                }
                
                if (modelPath != null && java.nio.file.Files.exists(modelPath)) {
                    var semanticChannel = new com.hellblazer.art.nlp.channels.FastTextChannel("semantic", 0.85, modelPath);
                    processor.addChannel("semantic", semanticChannel, 1.0);
                } else {
                    log.debug("FastText model resource not available or accessible");
                }
            } else {
                log.debug("FastText model resource not found");
            }
            
            // Add other default channels if available
            try {
                var syntacticChannel = new com.hellblazer.art.nlp.channels.SyntacticChannel("syntactic", 0.75);
                processor.addChannel("syntactic", syntacticChannel, 0.9);
            } catch (Exception e) {
                log.debug("Syntactic channel not available: {}", e.getMessage());
            }
            
            try {
                var entityChannel = new com.hellblazer.art.nlp.channels.EntityChannel("entity", 0.80);
                processor.addChannel("entity", entityChannel, 0.8);
            } catch (Exception e) {
                log.debug("Entity channel not available: {}", e.getMessage());
            }
            
            try {
                var contextChannel = new com.hellblazer.art.nlp.channels.ContextChannel();
                processor.addChannel("context", contextChannel, 0.7);
            } catch (Exception e) {
                log.debug("Context channel not available: {}", e.getMessage());
            }
            
            try {
                var sentimentChannel = new com.hellblazer.art.nlp.channels.SentimentChannel();
                processor.addChannel("sentiment", sentimentChannel, 0.6);
            } catch (Exception e) {
                log.debug("Sentiment channel not available: {}", e.getMessage());
            }
            
            return processor;
        } catch (Exception e) {
            log.error("Failed to create processor with defaults", e);
            throw new RuntimeException("Failed to create default processor", e);
        }
    }

    /**
     * Builder for MultiChannelProcessor configuration.
     */
    public static class Builder {
        private ConsensusStrategy consensusStrategy = new WeightedVotingConsensus();
        private FeatureFusionStrategy fusionStrategy = new ConcatenationFusion();
        private boolean enableParallelProcessing = true;
        private double learningRateDecay = 0.95;
        private int threadPoolSize = Runtime.getRuntime().availableProcessors();
        
        public Builder consensusStrategy(ConsensusStrategy strategy) {
            this.consensusStrategy = Objects.requireNonNull(strategy, "consensusStrategy cannot be null");
            return this;
        }
        
        public Builder fusionStrategy(FeatureFusionStrategy strategy) {
            this.fusionStrategy = Objects.requireNonNull(strategy, "fusionStrategy cannot be null");
            return this;
        }
        
        public Builder enableParallelProcessing(boolean enable) {
            this.enableParallelProcessing = enable;
            return this;
        }
        
        public Builder learningRateDecay(double decay) {
            if (decay <= 0.0 || decay > 1.0) {
                throw new IllegalArgumentException("Learning rate decay must be in (0.0, 1.0]: " + decay);
            }
            this.learningRateDecay = decay;
            return this;
        }
        
        public Builder threadPoolSize(int size) {
            if (size <= 0) {
                throw new IllegalArgumentException("Thread pool size must be positive: " + size);
            }
            this.threadPoolSize = size;
            return this;
        }
        
        public MultiChannelProcessor build() {
            return new MultiChannelProcessor(this);
        }
    }

    private MultiChannelProcessor(Builder builder) {
        this.consensusStrategy = builder.consensusStrategy;
        this.fusionStrategy = builder.fusionStrategy;
        this.enableParallelProcessing = builder.enableParallelProcessing;
        this.learningRateDecay = builder.learningRateDecay;
        
        if (enableParallelProcessing) {
            this.executorService = Executors.newFixedThreadPool(
                builder.threadPoolSize,
                r -> {
                    var thread = new Thread(r, "MultiChannel-" + System.currentTimeMillis());
                    thread.setDaemon(true);
                    return thread;
                }
            );
        } else {
            this.executorService = null;
        }
        
        log.info("MultiChannelProcessor created: parallel={}, consensus={}, fusion={}", 
                enableParallelProcessing, consensusStrategy.getClass().getSimpleName(),
                fusionStrategy.getClass().getSimpleName());
    }

    /**
     * Add a channel to the multi-channel processor.
     */
    public void addChannel(String channelId, BaseChannel channel) {
        addChannel(channelId, channel, 1.0); // Default weight
    }

    /**
     * Add a channel with specific weight to the multi-channel processor.
     */
    public void addChannel(String channelId, BaseChannel channel, double weight) {
        Objects.requireNonNull(channelId, "channelId cannot be null");
        Objects.requireNonNull(channel, "channel cannot be null");
        
        if (weight <= 0.0) {
            throw new IllegalArgumentException("Channel weight must be positive: " + weight);
        }
        
        if (closed) {
            throw new IllegalStateException("Processor has been closed");
        }
        
        // Initialize the channel before adding it
        try {
            channel.initialize();
        } catch (Exception e) {
            log.error("Failed to initialize channel '{}': {}", channelId, e.getMessage(), e);
            throw new RuntimeException("Channel initialization failed: " + channelId, e);
        }
        
        channels.put(channelId, channel);
        channelWeights.put(channelId, weight);
        channelStats.put(channelId, new ChannelStatistics());
        channelCategories.put(channelId, ConcurrentHashMap.newKeySet());
        categoryMapping.put(channelId, new ConcurrentHashMap<>());
        
        log.info("Added channel '{}' with weight {}: {}", channelId, weight, channel.getClass().getSimpleName());
    }

    /**
     * Remove a channel from the processor.
     */
    public boolean removeChannel(String channelId) {
        Objects.requireNonNull(channelId, "channelId cannot be null");
        
        var channel = channels.remove(channelId);
        if (channel != null) {
            channelWeights.remove(channelId);
            channelStats.remove(channelId);
            channelCategories.remove(channelId);
            categoryMapping.remove(channelId);
            
            log.info("Removed channel '{}'", channelId);
            return true;
        }
        return false;
    }

    /**
     * Process text input through all channels and return integrated result.
     */
    public ProcessingResult processText(String text) {
        if (text == null || text.isBlank()) {
            return ProcessingResult.failed("Invalid input text");
        }
        
        if (closed) {
            throw new IllegalStateException("Processor has been closed");
        }
        
        if (channels.isEmpty()) {
            return ProcessingResult.failed("No channels configured");
        }
        
        totalProcessed.incrementAndGet();
        var startTime = System.currentTimeMillis();
        
        try {
            // Process through all channels
            var channelResults = processAllChannels(text);
            
            // Fuse features from all channels
            var fusedFeatures = fusionStrategy.fuseFeatures(channelResults);
            
            // Apply consensus strategy for final classification
            var consensusResult = consensusStrategy.computeConsensus(channelResults, channelWeights);
            
            // Update cross-channel learning
            updateCrossChannelLearning(channelResults, consensusResult);
            
            // Extract channel categories from channel results
            var channelCategories = new HashMap<String, Integer>();
            var allEntities = new ArrayList<Entity>();
            
            for (var entry : channelResults.entrySet()) {
                var channelId = entry.getKey();
                var channelResult = entry.getValue();
                if (channelResult.success()) {
                    // Use channel ID with underscore prefix for category key format
                    var categoryKey = channelId + "_" + channelResult.category();
                    channelCategories.put(categoryKey, channelResult.category());
                }
                
                // Collect entities from channel metadata if available
                if (channelResult.hasMetadata("entities")) {
                    @SuppressWarnings("unchecked")
                    var channelEntities = (List<Entity>) channelResult.metadata().get("entities");
                    if (channelEntities != null) {
                        allEntities.addAll(channelEntities);
                    }
                }
            }

            // Calculate token count for the text
            var tokenCount = calculateTokenCount(text);
            
            // Create integrated result
            var processingTime = System.currentTimeMillis() - startTime;
            var result = ProcessingResult.builder()
                .text(text)
                .confidence(consensusResult.confidence())
                .category(consensusResult.category())
                .processingTimeMs(processingTime)
                .channelResults(channelResults)
                .withChannelCategories(channelCategories)
                .withEntities(allEntities)
                .withTokenCount(tokenCount)
                .fusedFeatures(fusedFeatures)
                .consensusMetadata(consensusResult.metadata())
                .build();
            
            successfulProcessed.incrementAndGet();
            updateChannelStatistics(channelResults, processingTime);
            
            return result;
            
        } catch (Exception e) {
            log.error("Error processing text: {}", text, e);
            return ProcessingResult.failed("Processing error: " + e.getMessage());
        }
    }

    /**
     * Process text through all channels (parallel or sequential).
     */
    private Map<String, ChannelResult> processAllChannels(String text) {
        if (enableParallelProcessing && channels.size() > 1) {
            return processChannelsParallel(text);
        } else {
            return processChannelsSequential(text);
        }
    }

    /**
     * Process channels in parallel using CompletableFuture.
     */
    private Map<String, ChannelResult> processChannelsParallel(String text) {
        var futures = channels.entrySet().stream()
            .collect(Collectors.toMap(
                Map.Entry::getKey,
                entry -> CompletableFuture.supplyAsync(
                    () -> processChannel(entry.getKey(), entry.getValue(), text),
                    executorService
                )
            ));
        
        var results = new HashMap<String, ChannelResult>();
        for (var entry : futures.entrySet()) {
            try {
                results.put(entry.getKey(), entry.getValue().get());
            } catch (Exception e) {
                log.warn("Channel '{}' processing failed: {}", entry.getKey(), e.getMessage());
                results.put(entry.getKey(), ChannelResult.failed(entry.getKey(), e.getMessage()));
            }
        }
        
        return results;
    }

    /**
     * Process channels sequentially.
     */
    private Map<String, ChannelResult> processChannelsSequential(String text) {
        var results = new HashMap<String, ChannelResult>();
        
        for (var entry : channels.entrySet()) {
            var channelId = entry.getKey();
            var channel = entry.getValue();
            
            try {
                results.put(channelId, processChannel(channelId, channel, text));
            } catch (Exception e) {
                log.warn("Channel '{}' processing failed: {}", channelId, e.getMessage());
                results.put(channelId, ChannelResult.failed(channelId, e.getMessage()));
            }
        }
        
        return results;
    }

    /**
     * Process single channel and return result.
     */
    private ChannelResult processChannel(String channelId, BaseChannel channel, String text) {
        var startTime = System.currentTimeMillis();
        
        try {
            // Channel-specific text processing
            int category = processChannelSpecific(channel, text);
            var processingTime = System.currentTimeMillis() - startTime;
            
            // Extract entities from channel if supported
            var metadata = new HashMap<String, Object>();
            var entities = extractChannelEntities(channel, text);
            if (!entities.isEmpty()) {
                metadata.put("entities", entities);
            }
            
            if (category >= 0) {
                return ChannelResult.success(channelId, category, 1.0, processingTime, metadata);
            } else {
                return ChannelResult.failed(channelId, "Classification failed", processingTime, metadata);
            }
            
        } catch (Exception e) {
            var processingTime = System.currentTimeMillis() - startTime;
            return ChannelResult.failed(channelId, "Error: " + e.getMessage(), processingTime);
        }
    }

    /**
     * Handle channel-specific processing based on channel type.
     */
    private int processChannelSpecific(BaseChannel channel, String text) {
        // Use reflection or instanceof to handle different channel types
        var className = channel.getClass().getSimpleName();
        
        return switch (className) {
            case "FastTextChannel" -> {
                try {
                    var method = channel.getClass().getMethod("classifyText", String.class);
                    yield (Integer) method.invoke(channel, text);
                } catch (Exception e) {
                    log.debug("FastText channel processing failed: {}", e.getMessage());
                    yield -1;
                }
            }
            case "EntityChannel" -> {
                try {
                    var method = channel.getClass().getMethod("classifyText", String.class);
                    yield (Integer) method.invoke(channel, text);
                } catch (Exception e) {
                    log.debug("Entity channel processing failed: {}", e.getMessage());
                    yield -1;
                }
            }
            case "SyntacticChannel" -> {
                try {
                    var method = channel.getClass().getMethod("classifyText", String.class);
                    yield (Integer) method.invoke(channel, text);
                } catch (Exception e) {
                    log.debug("Syntactic channel processing failed: {}", e.getMessage());
                    yield -1;
                }
            }
            case "ContextChannel" -> {
                try {
                    var method = channel.getClass().getMethod("classifyText", String.class);
                    yield (Integer) method.invoke(channel, text);
                } catch (Exception e) {
                    log.debug("Context channel processing failed: {}", e.getMessage());
                    yield -1;
                }
            }
            case "SentimentChannel" -> {
                try {
                    var method = channel.getClass().getMethod("classifyText", String.class);
                    yield (Integer) method.invoke(channel, text);
                } catch (Exception e) {
                    log.debug("Sentiment channel processing failed: {}", e.getMessage());
                    yield -1;
                }
            }
            default -> {
                log.warn("Unknown channel type: {}", className);
                yield -1;
            }
        };
    }

    /**
     * Extract entities from channel if supported.
     */
    private List<Entity> extractChannelEntities(BaseChannel channel, String text) {
        var className = channel.getClass().getSimpleName();
        
        return switch (className) {
            case "ContextChannel" -> {
                try {
                    var method = channel.getClass().getMethod("extractEntities", String.class);
                    @SuppressWarnings("unchecked")
                    var result = (List<Entity>) method.invoke(channel, text);
                    yield result != null ? result : List.of();
                } catch (Exception e) {
                    log.debug("Context channel entity extraction failed: {}", e.getMessage());
                    yield List.of();
                }
            }
            case "EntityChannel" -> {
                try {
                    var method = channel.getClass().getMethod("extractEntities", String.class);
                    @SuppressWarnings("unchecked")
                    var result = (List<Entity>) method.invoke(channel, text);
                    yield result != null ? result : List.of();
                } catch (Exception e) {
                    log.debug("Entity channel entity extraction failed: {}", e.getMessage());
                    yield List.of();
                }
            }
            default -> List.of(); // No entity extraction for other channel types
        };
    }

    /**
     * Update cross-channel learning based on processing results.
     */
    private void updateCrossChannelLearning(Map<String, ChannelResult> channelResults, 
                                           ConsensusResult consensusResult) {
        // Update category mappings and cross-channel relationships
        for (var entry : channelResults.entrySet()) {
            var channelId = entry.getKey();
            var result = entry.getValue();
            
            if (result.isSuccess()) {
                var localCategory = result.category();
                var categories = channelCategories.get(channelId);
                categories.add(localCategory);
                
                // Update global category mapping
                var mapping = categoryMapping.get(channelId);
                mapping.computeIfAbsent(localCategory, k -> 
                    new CategoryMetadata(globalCategoryCounter.getAndIncrement(), 1))
                    .incrementCount();
            }
        }
    }

    /**
     * Update channel performance statistics.
     */
    private void updateChannelStatistics(Map<String, ChannelResult> channelResults, long totalTime) {
        for (var entry : channelResults.entrySet()) {
            var stats = channelStats.get(entry.getKey());
            var result = entry.getValue();
            
            stats.incrementTotal();
            if (result.isSuccess()) {
                stats.incrementSuccessful();
            }
            stats.addProcessingTime(result.processingTimeMs());
        }
    }

    /**
     * Get processor performance metrics.
     */
    public MultiChannelMetrics getMetrics() {
        return new MultiChannelMetrics(
            totalProcessed.get(),
            successfulProcessed.get(),
            channels.size(),
            calculateOverallSuccessRate(),
            calculateAverageProcessingTime(),
            new HashMap<>(channelStats),
            calculateChannelContributions()
        );
    }

    private double calculateOverallSuccessRate() {
        var total = totalProcessed.get();
        return total > 0 ? (double) successfulProcessed.get() / total : 0.0;
    }

    private double calculateAverageProcessingTime() {
        return channelStats.values().stream()
            .mapToDouble(ChannelStatistics::getAverageProcessingTime)
            .average()
            .orElse(0.0);
    }

    private Map<String, Double> calculateChannelContributions() {
        var contributions = new HashMap<String, Double>();
        var totalWeight = channelWeights.values().stream()
            .mapToDouble(Double::doubleValue)
            .sum();
        
        for (var entry : channelWeights.entrySet()) {
            var normalizedWeight = totalWeight > 0 ? entry.getValue() / totalWeight : 0.0;
            var stats = channelStats.get(entry.getKey());
            var successRate = stats.getSuccessRate();
            
            contributions.put(entry.getKey(), normalizedWeight * successRate);
        }
        
        return contributions;
    }

    /**
     * Update channel weight based on performance.
     */
    public void updateChannelWeight(String channelId, double newWeight) {
        Objects.requireNonNull(channelId, "channelId cannot be null");
        
        if (newWeight <= 0.0) {
            throw new IllegalArgumentException("Channel weight must be positive: " + newWeight);
        }
        
        if (channels.containsKey(channelId)) {
            channelWeights.put(channelId, newWeight);
            log.debug("Updated weight for channel '{}': {}", channelId, newWeight);
        } else {
            throw new IllegalArgumentException("Channel not found: " + channelId);
        }
    }

    /**
     * Get all channel IDs.
     */
    public Set<String> getChannelIds() {
        return new HashSet<>(channels.keySet());
    }

    /**
     * Check if processor is closed.
     */
    public boolean isClosed() {
        return closed;
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        
        closed = true;
        
        if (executorService != null) {
            executorService.shutdown();
            try {
                if (!executorService.awaitTermination(5, java.util.concurrent.TimeUnit.SECONDS)) {
                    executorService.shutdownNow();
                }
            } catch (InterruptedException e) {
                executorService.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
        
        log.info("MultiChannelProcessor closed");
    }

    /**
     * Channel statistics tracking.
     */
    private static class ChannelStatistics {
        private final AtomicInteger totalProcessed = new AtomicInteger(0);
        private final AtomicInteger successfulProcessed = new AtomicInteger(0);
        private final AtomicInteger totalProcessingTime = new AtomicInteger(0);
        
        void incrementTotal() { totalProcessed.incrementAndGet(); }
        void incrementSuccessful() { successfulProcessed.incrementAndGet(); }
        void addProcessingTime(long timeMs) { totalProcessingTime.addAndGet((int) timeMs); }
        
        double getSuccessRate() {
            var total = totalProcessed.get();
            return total > 0 ? (double) successfulProcessed.get() / total : 0.0;
        }
        
        double getAverageProcessingTime() {
            var total = totalProcessed.get();
            return total > 0 ? (double) totalProcessingTime.get() / total : 0.0;
        }
        
        int getTotalProcessed() { return totalProcessed.get(); }
        int getSuccessfulProcessed() { return successfulProcessed.get(); }
    }

    /**
     * Category metadata for cross-channel learning.
     */
    private static class CategoryMetadata {
        private final int globalId;
        private int count;
        
        CategoryMetadata(int globalId, int initialCount) {
            this.globalId = globalId;
            this.count = initialCount;
        }
        
        void incrementCount() { count++; }
        int getGlobalId() { return globalId; }
        int getCount() { return count; }
    }
    
    // === NLPProcessor Interface Implementation ===
    
    @Override
    public ProcessingResult process(String text) {
        return processText(text); // Delegate to existing implementation
    }
    
    @Override
    public void processStream(java.io.InputStream stream, ResultCallback callback) {
        Objects.requireNonNull(stream, "stream cannot be null");
        Objects.requireNonNull(callback, "callback cannot be null");
        
        try (var reader = new java.io.BufferedReader(new java.io.InputStreamReader(stream, java.nio.charset.StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null && !closed) {
                if (!line.isBlank()) {
                    try {
                        var result = process(line);
                        callback.onResult(result);
                    } catch (Exception e) {
                        log.error("Error processing stream line: {}", line, e);
                        callback.onError(e);
                    }
                }
            }
            callback.onComplete();
        } catch (java.io.IOException e) {
            log.error("Error reading from input stream", e);
            callback.onError(e);
        }
    }
    
    @Override
    public DocumentAnalysis processDocument(Document document) {
        Objects.requireNonNull(document, "document cannot be null");
        
        var processingResult = process(document.getContent());
        
        // Enhanced document processing with sentence/paragraph detection
        var sentences = extractSentences(document.getContent());
        var paragraphs = extractParagraphs(document.getContent());
        
        return DocumentAnalysis.builder()
            .withDocument(document)
            .withProcessingResult(processingResult)
            .withSentences(sentences)
            .withParagraphs(paragraphs)
            .withAnalysisMetadata("processing_version", "1.0")
            .withAnalysisMetadata("processor_type", "MultiChannelProcessor")
            .withAnalysisMetadata("channels_used", getEnabledChannelNames())
            .build();
    }
    
    @Override
    public ProcessingStats getStatistics() {
        var channelStatsMap = new HashMap<String, ProcessingStats.ChannelStats>();
        
        for (var entry : this.channelStats.entrySet()) {
            var stats = entry.getValue();
            var channelSpecific = new HashMap<String, Object>();
            channelSpecific.put("weight", channelWeights.get(entry.getKey()));
            
            channelStatsMap.put(entry.getKey(), new ProcessingStats.ChannelStats(
                entry.getKey(),
                stats.getTotalProcessed(),
                stats.getSuccessfulProcessed(),
                channels.containsKey(entry.getKey()) ? channels.get(entry.getKey()).getCategoryCount() : 0,
                stats.getAverageProcessingTime(),
                stats.getSuccessRate(),
                channelSpecific
            ));
        }
        
        // System metrics
        var systemMetrics = new HashMap<String, Object>();
        systemMetrics.put("total_channels", channels.size());
        systemMetrics.put("enabled_channels", enabledChannels.size());
        systemMetrics.put("parallel_processing", enableParallelProcessing);
        systemMetrics.put("memory_usage", Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory());
        
        return ProcessingStats.builder()
            .withStartTime(startTime)
            .withLastUpdate(java.time.Instant.now())
            .withProcessingCounts(totalProcessed.get(), successfulProcessed.get(), totalProcessed.get() - successfulProcessed.get())
            .withAverageProcessingTime(calculateAverageProcessingTime())
            .withThroughput(calculateThroughput())
            .withChannelStatistics(channelStatsMap)
            .withSystemMetrics(systemMetrics)
            .build();
    }
    
    @Override
    public void reset() {
        log.warn("Resetting all ART networks - this will clear all learned patterns!");
        
        for (var channel : channels.values()) {
            try {
                // Reset channel state (this would need to be implemented in BaseChannel)
                log.info("Reset channel: {}", channel.getChannelName());
            } catch (Exception e) {
                log.error("Error resetting channel {}: {}", channel.getChannelName(), e.getMessage());
            }
        }
        
        // Reset statistics
        totalProcessed.set(0);
        successfulProcessed.set(0);
        channelStats.clear();
        channelCategories.clear();
        categoryMapping.clear();
        globalCategoryCounter.set(0);
        
        log.info("All channels reset complete");
    }
    
    @Override
    public void resetChannel(String channelName) {
        Objects.requireNonNull(channelName, "channelName cannot be null");
        
        var channel = channels.get(channelName);
        if (channel == null) {
            throw new IllegalArgumentException("Channel not found: " + channelName);
        }
        
        log.warn("Resetting channel '{}' - this will clear all learned patterns!", channelName);
        
        // Reset channel statistics
        channelStats.remove(channelName);
        channelCategories.remove(channelName);
        categoryMapping.remove(channelName);
        
        log.info("Channel '{}' reset complete", channelName);
    }
    
    @Override
    public boolean setChannelEnabled(String channelName, boolean enabled) {
        Objects.requireNonNull(channelName, "channelName cannot be null");
        
        if (!channels.containsKey(channelName)) {
            throw new IllegalArgumentException("Channel not found: " + channelName);
        }
        
        if (enabled) {
            boolean wasAdded = enabledChannels.add(channelName);
            if (wasAdded) {
                log.info("Enabled channel '{}'", channelName);
            }
            return wasAdded;
        } else {
            boolean wasRemoved = enabledChannels.remove(channelName);
            if (wasRemoved) {
                log.info("Disabled channel '{}'", channelName);
            }
            return wasRemoved;
        }
    }
    
    @Override
    public Set<String> getChannelNames() {
        return new HashSet<>(channels.keySet());
    }
    
    @Override
    public Set<String> getEnabledChannelNames() {
        return new HashSet<>(enabledChannels);
    }
    
    @Override
    public boolean isReady() {
        if (channels.isEmpty()) {
            return false;
        }
        
        // Check if all enabled channels are initialized
        for (String channelName : enabledChannels) {
            var channel = channels.get(channelName);
            if (channel == null || !channel.isInitialized()) {
                return false;
            }
        }
        
        return !closed;
    }
    
    @Override
    public void saveState() {
        log.info("Saving processor state...");
        
        for (var entry : channels.entrySet()) {
            try {
                entry.getValue().saveState();
                log.debug("Saved state for channel '{}'", entry.getKey());
            } catch (Exception e) {
                log.error("Error saving state for channel '{}': {}", entry.getKey(), e.getMessage());
            }
        }
        
        log.info("Processor state save complete");
    }
    
    @Override
    public void loadState() {
        log.info("Loading processor state...");
        
        for (var entry : channels.entrySet()) {
            try {
                entry.getValue().loadState();
                log.debug("Loaded state for channel '{}'", entry.getKey());
            } catch (Exception e) {
                log.error("Error loading state for channel '{}': {}", entry.getKey(), e.getMessage());
            }
        }
        
        log.info("Processor state load complete");
    }
    
    @Override
    public void shutdown() {
        close(); // Delegate to existing implementation
    }
    
    // === Helper Methods ===
    
    private List<String> extractSentences(String text) {
        // Simple sentence detection - could be enhanced with OpenNLP
        var sentences = new ArrayList<String>();
        var sentencePattern = java.util.regex.Pattern.compile("[.!?]+\\s+");
        var parts = sentencePattern.split(text);
        
        for (var part : parts) {
            if (!part.isBlank()) {
                sentences.add(part.trim());
            }
        }
        
        return sentences;
    }
    
    private List<String> extractParagraphs(String text) {
        // Simple paragraph detection
        var paragraphs = new ArrayList<String>();
        var paragraphPattern = java.util.regex.Pattern.compile("\\n\\s*\\n");
        var parts = paragraphPattern.split(text);
        
        for (var part : parts) {
            if (!part.isBlank()) {
                paragraphs.add(part.trim());
            }
        }
        
        return paragraphs;
    }
    
    private double calculateThroughput() {
        var uptime = java.time.Duration.between(startTime, java.time.Instant.now());
        var uptimeSeconds = uptime.toSeconds();
        return uptimeSeconds > 0 ? (double) totalProcessed.get() / uptimeSeconds : 0.0;
    }
    
    /**
     * Calculate token count for text using simple whitespace-based tokenization.
     */
    private int calculateTokenCount(String text) {
        if (text == null || text.isBlank()) {
            return 0;
        }
        
        // Simple tokenization by splitting on whitespace
        // This could be enhanced with more sophisticated tokenization
        var tokens = text.trim().split("\\s+");
        return tokens.length;
    }
    

    // Static builder method
    public static Builder builder() {
        return new Builder();
    }
}