package com.hellblazer.art.hartcq.spatial;

import com.hellblazer.art.hartcq.HARTCQConfig;
import com.hellblazer.art.hartcq.Token;
import org.joml.Vector2f;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Main spatial processing coordinator for HART-CQ system.
 * 
 * Orchestrates 2D spatial mapping of tokens, extracts spatial relationships,
 * and performs distance-based processing to maintain spatial coherence across
 * processing windows.
 * 
 * The spatial processor leverages JOML for efficient vector math operations
 * and supports parallel processing for high-throughput scenarios.
 * 
 * @author Claude Code
 */
public class SpatialProcessor implements AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(SpatialProcessor.class);
    
    private final HARTCQConfig config;
    private final SpatialMap spatialMap;
    private final ProximityAnalyzer proximityAnalyzer;
    private final TopologicalProcessor topologicalProcessor;
    private final Executor executor;
    
    // State management
    private final AtomicReference<ProcessingState> currentState;
    private final AtomicInteger processingCycle;
    private final Map<String, SpatialContext> contextCache;
    
    // Performance tracking
    private final SpatialProcessorStats stats;
    
    /**
     * Represents the current state of spatial processing.
     */
    public enum ProcessingState {
        INITIALIZING,
        READY,
        MAPPING_TOKENS,
        ANALYZING_PROXIMITY,
        DETECTING_PATTERNS,
        PROCESSING_TOPOLOGY,
        COMPLETED,
        ERROR
    }
    
    /**
     * Result of spatial processing operations.
     */
    public static class SpatialResult {
        private final Token[] inputTokens;
        private final Map<Token, Vector2f> tokenPositions;
        private final List<SpatialPattern> detectedPatterns;
        private final ProximityAnalyzer.ProximityResult proximityResult;
        private final TopologicalProcessor.TopologyResult topologyResult;
        private final double spatialCoherence;
        private final int processingTimeMs;
        
        public SpatialResult(Token[] inputTokens,
                           Map<Token, Vector2f> tokenPositions,
                           List<SpatialPattern> detectedPatterns,
                           ProximityAnalyzer.ProximityResult proximityResult,
                           TopologicalProcessor.TopologyResult topologyResult,
                           double spatialCoherence,
                           int processingTimeMs) {
            this.inputTokens = Objects.requireNonNull(inputTokens);
            this.tokenPositions = new HashMap<>(tokenPositions);
            this.detectedPatterns = new ArrayList<>(detectedPatterns);
            this.proximityResult = proximityResult;
            this.topologyResult = topologyResult;
            this.spatialCoherence = spatialCoherence;
            this.processingTimeMs = processingTimeMs;
        }
        
        public Token[] getInputTokens() { return inputTokens.clone(); }
        public Map<Token, Vector2f> getTokenPositions() { return new HashMap<>(tokenPositions); }
        public List<SpatialPattern> getDetectedPatterns() { return new ArrayList<>(detectedPatterns); }
        public ProximityAnalyzer.ProximityResult getProximityResult() { return proximityResult; }
        public TopologicalProcessor.TopologyResult getTopologyResult() { return topologyResult; }
        public double getSpatialCoherence() { return spatialCoherence; }
        public int getProcessingTimeMs() { return processingTimeMs; }
        
        @Override
        public String toString() {
            return "SpatialResult{tokens=%d, patterns=%d, coherence=%.3f, time=%dms}"
                .formatted(inputTokens.length, detectedPatterns.size(), spatialCoherence, processingTimeMs);
        }
    }
    
    /**
     * Spatial processing context for maintaining state across windows.
     */
    private static class SpatialContext {
        private final Map<String, Vector2f> tokenTypePositions;
        private final Set<SpatialPattern> persistentPatterns;
        private final double[] spatialWeights;
        private final long lastUpdateTime;
        
        public SpatialContext() {
            this.tokenTypePositions = new ConcurrentHashMap<>();
            this.persistentPatterns = ConcurrentHashMap.newKeySet();
            this.spatialWeights = new double[4]; // x, y, proximity, topology
            this.lastUpdateTime = System.currentTimeMillis();
            
            // Initialize default spatial weights
            this.spatialWeights[0] = 0.25; // x-coordinate weight
            this.spatialWeights[1] = 0.25; // y-coordinate weight
            this.spatialWeights[2] = 0.30; // proximity weight
            this.spatialWeights[3] = 0.20; // topology weight
        }
        
        public Map<String, Vector2f> getTokenTypePositions() { return tokenTypePositions; }
        public Set<SpatialPattern> getPersistentPatterns() { return persistentPatterns; }
        public double[] getSpatialWeights() { return spatialWeights.clone(); }
        public long getLastUpdateTime() { return lastUpdateTime; }
    }
    
    /**
     * Creates a spatial processor with the given configuration.
     * 
     * @param config HART-CQ configuration
     * @param executor Executor for asynchronous processing
     */
    public SpatialProcessor(HARTCQConfig config, Executor executor) {
        this.config = Objects.requireNonNull(config, "Configuration cannot be null");
        this.executor = Objects.requireNonNull(executor, "Executor cannot be null");
        
        // Initialize spatial components
        this.spatialMap = new SpatialMap(config);
        this.proximityAnalyzer = new ProximityAnalyzer(config);
        this.topologicalProcessor = new TopologicalProcessor(config);
        
        // Initialize state management
        this.currentState = new AtomicReference<>(ProcessingState.INITIALIZING);
        this.processingCycle = new AtomicInteger(0);
        this.contextCache = new ConcurrentHashMap<>();
        this.stats = new SpatialProcessorStats();
        
        logger.info("SpatialProcessor initialized with windowSize={}", config.getWindowSize());
        this.currentState.set(ProcessingState.READY);
    }
    
    /**
     * Processes tokens through the spatial system synchronously.
     * 
     * @param tokens Input token array
     * @return Spatial processing result
     */
    public SpatialResult process(Token[] tokens) {
        return processAsync(tokens).join();
    }
    
    /**
     * Processes tokens through the spatial system with context.
     * 
     * @param tokens Input token array
     * @param contextId Context identifier for maintaining spatial coherence
     * @return Spatial processing result
     */
    public SpatialResult processWithContext(Token[] tokens, String contextId) {
        return processAsyncWithContext(tokens, contextId).join();
    }
    
    /**
     * Processes tokens asynchronously.
     * 
     * @param tokens Input token array
     * @return CompletableFuture with spatial result
     */
    public CompletableFuture<SpatialResult> processAsync(Token[] tokens) {
        return processAsyncWithContext(tokens, "default");
    }
    
    /**
     * Processes tokens asynchronously with context.
     * 
     * @param tokens Input token array
     * @param contextId Context identifier
     * @return CompletableFuture with spatial result
     */
    public CompletableFuture<SpatialResult> processAsyncWithContext(Token[] tokens, String contextId) {
        if (tokens == null || tokens.length == 0) {
            return CompletableFuture.completedFuture(createErrorResult("Invalid input tokens"));
        }
        
        return CompletableFuture.supplyAsync(() -> {
            var startTime = System.currentTimeMillis();
            var cycle = processingCycle.incrementAndGet();
            
            try {
                logger.debug("Starting spatial processing cycle {} with {} tokens", cycle, tokens.length);
                
                // Get or create spatial context
                var context = contextCache.computeIfAbsent(contextId, k -> new SpatialContext());
                
                // Step 1: Map tokens to 2D space
                currentState.set(ProcessingState.MAPPING_TOKENS);
                var tokenPositions = spatialMap.mapTokensToSpace(tokens, context.getTokenTypePositions());
                
                // Step 2: Analyze proximity relationships
                currentState.set(ProcessingState.ANALYZING_PROXIMITY);
                var proximityResult = proximityAnalyzer.analyzeProximity(tokenPositions);
                
                // Step 3: Detect spatial patterns
                currentState.set(ProcessingState.DETECTING_PATTERNS);
                var detectedPatterns = detectSpatialPatterns(tokenPositions, proximityResult, context);
                
                // Step 4: Process topological relationships
                currentState.set(ProcessingState.PROCESSING_TOPOLOGY);
                var topologyResult = topologicalProcessor.processTopology(tokenPositions, proximityResult);
                
                // Calculate spatial coherence
                var spatialCoherence = calculateSpatialCoherence(tokenPositions, proximityResult, 
                                                               detectedPatterns, topologyResult);
                
                // Update context
                updateSpatialContext(context, tokenPositions, detectedPatterns);
                
                var processingTime = (int) (System.currentTimeMillis() - startTime);
                
                var result = new SpatialResult(
                    tokens, tokenPositions, detectedPatterns,
                    proximityResult, topologyResult, spatialCoherence, processingTime
                );
                
                // Update statistics
                stats.recordProcessing(tokens.length, processingTime, spatialCoherence);
                
                currentState.set(ProcessingState.COMPLETED);
                logger.debug("Completed spatial processing cycle {}: {}", cycle, result);
                
                return result;
                
            } catch (Exception e) {
                logger.error("Error in spatial processing cycle " + cycle, e);
                currentState.set(ProcessingState.ERROR);
                return createErrorResult("Processing error: " + e.getMessage());
            }
        }, executor);
    }
    
    /**
     * Detects spatial patterns in the mapped token space.
     */
    private List<SpatialPattern> detectSpatialPatterns(Map<Token, Vector2f> tokenPositions,
                                                      ProximityAnalyzer.ProximityResult proximityResult,
                                                      SpatialContext context) {
        var patterns = new ArrayList<SpatialPattern>();
        
        // Extract clusters from proximity analysis
        for (var cluster : proximityResult.getClusters()) {
            if (cluster.getTokens().size() >= 2) {
                var pattern = SpatialPattern.fromCluster(cluster.getTokens(), tokenPositions);
                patterns.add(pattern);
            }
        }
        
        // Check for persistent patterns from context
        for (var persistentPattern : context.getPersistentPatterns()) {
            var matchingTokens = findMatchingTokens(persistentPattern, tokenPositions);
            if (!matchingTokens.isEmpty()) {
                var updatedPattern = persistentPattern.updateWithTokens(matchingTokens, tokenPositions);
                patterns.add(updatedPattern);
            }
        }
        
        return patterns;
    }
    
    /**
     * Finds tokens that match a persistent spatial pattern.
     */
    private List<Token> findMatchingTokens(SpatialPattern pattern, Map<Token, Vector2f> tokenPositions) {
        var matchingTokens = new ArrayList<Token>();
        
        for (var entry : tokenPositions.entrySet()) {
            if (pattern.matches(entry.getKey(), entry.getValue())) {
                matchingTokens.add(entry.getKey());
            }
        }
        
        return matchingTokens;
    }
    
    /**
     * Calculates overall spatial coherence score.
     */
    private double calculateSpatialCoherence(Map<Token, Vector2f> tokenPositions,
                                          ProximityAnalyzer.ProximityResult proximityResult,
                                          List<SpatialPattern> patterns,
                                          TopologicalProcessor.TopologyResult topologyResult) {
        // Combine multiple coherence measures
        var proximityCoherence = proximityResult.getOverallCoherence();
        var patternCoherence = calculatePatternCoherence(patterns);
        var topologyCoherence = topologyResult.getTopologyCoherence();
        var spatialDistribution = calculateSpatialDistribution(tokenPositions);
        
        // Weighted average
        return 0.3 * proximityCoherence +
               0.3 * patternCoherence +
               0.2 * topologyCoherence +
               0.2 * spatialDistribution;
    }
    
    /**
     * Calculates coherence based on detected patterns.
     */
    private double calculatePatternCoherence(List<SpatialPattern> patterns) {
        if (patterns.isEmpty()) {
            return 0.0;
        }
        
        return patterns.stream()
            .mapToDouble(SpatialPattern::getConfidence)
            .average()
            .orElse(0.0);
    }
    
    /**
     * Calculates spatial distribution quality.
     */
    private double calculateSpatialDistribution(Map<Token, Vector2f> tokenPositions) {
        if (tokenPositions.size() < 2) {
            return 1.0;
        }
        
        var positions = tokenPositions.values();
        var center = new Vector2f();
        
        // Calculate centroid
        for (var pos : positions) {
            center.add(pos);
        }
        center.div(positions.size());
        
        // Calculate average distance from center
        var totalDistance = 0.0;
        for (var pos : positions) {
            totalDistance += center.distance(pos);
        }
        var avgDistance = totalDistance / positions.size();
        
        // Normalize to [0,1] range - higher values indicate better distribution
        return Math.min(1.0, avgDistance / Math.sqrt(2.0)); // Assuming unit square space
    }
    
    /**
     * Updates spatial context with new processing results.
     */
    private void updateSpatialContext(SpatialContext context,
                                    Map<Token, Vector2f> tokenPositions,
                                    List<SpatialPattern> detectedPatterns) {
        // Update token type positions with exponential moving average
        var alpha = 0.3; // Learning rate
        for (var entry : tokenPositions.entrySet()) {
            var tokenType = entry.getKey().getType().name();
            var newPos = entry.getValue();
            
            context.getTokenTypePositions().compute(tokenType, (k, oldPos) -> {
                if (oldPos == null) {
                    return new Vector2f(newPos);
                } else {
                    return new Vector2f(
                        (float) (alpha * newPos.x + (1 - alpha) * oldPos.x),
                        (float) (alpha * newPos.y + (1 - alpha) * oldPos.y)
                    );
                }
            });
        }
        
        // Update persistent patterns
        for (var pattern : detectedPatterns) {
            if (pattern.getConfidence() > 0.7) { // High confidence patterns become persistent
                context.getPersistentPatterns().add(pattern);
            }
        }
        
        // Cleanup old patterns (keep only recent high-confidence patterns)
        context.getPersistentPatterns().removeIf(pattern -> 
            System.currentTimeMillis() - pattern.getLastSeenTime() > 60000); // 1 minute
    }
    
    /**
     * Creates an error result for failed processing.
     */
    private SpatialResult createErrorResult(String message) {
        logger.error("Spatial processing error: {}", message);
        return new SpatialResult(
            new Token[0], new HashMap<>(), new ArrayList<>(),
            ProximityAnalyzer.ProximityResult.empty(),
            TopologicalProcessor.TopologyResult.empty(),
            0.0, 0
        );
    }
    
    /**
     * Gets the current processing state.
     * 
     * @return current processing state
     */
    public ProcessingState getCurrentState() {
        return currentState.get();
    }
    
    /**
     * Gets spatial processing statistics.
     * 
     * @return processing statistics
     */
    public SpatialProcessorStats getStatistics() {
        return stats.copy();
    }
    
    /**
     * Resets the spatial processor to initial state.
     */
    public void reset() {
        logger.info("Resetting spatial processor");
        
        currentState.set(ProcessingState.INITIALIZING);
        processingCycle.set(0);
        contextCache.clear();
        stats.reset();
        
        // Reset components
        spatialMap.reset();
        proximityAnalyzer.reset();
        topologicalProcessor.reset();
        
        currentState.set(ProcessingState.READY);
        logger.info("Spatial processor reset completed");
    }
    
    /**
     * Closes the spatial processor and releases resources.
     */
    @Override
    public void close() {
        logger.info("Closing spatial processor");
        currentState.set(ProcessingState.INITIALIZING);
        
        try {
            spatialMap.close();
        } catch (Exception e) {
            logger.warn("Error closing spatial map", e);
        }
        
        try {
            proximityAnalyzer.close();
        } catch (Exception e) {
            logger.warn("Error closing proximity analyzer", e);
        }
        
        try {
            topologicalProcessor.close();
        } catch (Exception e) {
            logger.warn("Error closing topological processor", e);
        }
        
        contextCache.clear();
        logger.info("Spatial processor closed");
    }
    
    /**
     * Statistics for spatial processing performance.
     */
    public static class SpatialProcessorStats {
        private int totalProcessings = 0;
        private int totalTokensProcessed = 0;
        private long totalProcessingTimeMs = 0;
        private double averageSpatialCoherence = 0.0;
        private long lastResetTime = System.currentTimeMillis();
        
        synchronized void recordProcessing(int tokenCount, int processingTimeMs, double spatialCoherence) {
            totalProcessings++;
            totalTokensProcessed += tokenCount;
            totalProcessingTimeMs += processingTimeMs;
            
            // Update average coherence using exponential moving average
            var alpha = 0.1;
            averageSpatialCoherence = alpha * spatialCoherence + (1 - alpha) * averageSpatialCoherence;
        }
        
        public synchronized int getTotalProcessings() { return totalProcessings; }
        public synchronized int getTotalTokensProcessed() { return totalTokensProcessed; }
        public synchronized long getTotalProcessingTimeMs() { return totalProcessingTimeMs; }
        public synchronized double getAverageProcessingTimeMs() {
            return totalProcessings > 0 ? (double) totalProcessingTimeMs / totalProcessings : 0.0;
        }
        public synchronized double getAverageSpatialCoherence() { return averageSpatialCoherence; }
        public synchronized double getThroughputTokensPerSecond() {
            var elapsedSeconds = (System.currentTimeMillis() - lastResetTime) / 1000.0;
            return elapsedSeconds > 0 ? totalTokensProcessed / elapsedSeconds : 0.0;
        }
        
        synchronized void reset() {
            totalProcessings = 0;
            totalTokensProcessed = 0;
            totalProcessingTimeMs = 0;
            averageSpatialCoherence = 0.0;
            lastResetTime = System.currentTimeMillis();
        }
        
        synchronized SpatialProcessorStats copy() {
            var copy = new SpatialProcessorStats();
            copy.totalProcessings = this.totalProcessings;
            copy.totalTokensProcessed = this.totalTokensProcessed;
            copy.totalProcessingTimeMs = this.totalProcessingTimeMs;
            copy.averageSpatialCoherence = this.averageSpatialCoherence;
            copy.lastResetTime = this.lastResetTime;
            return copy;
        }
        
        @Override
        public String toString() {
            return "SpatialProcessorStats{processings=%d, tokens=%d, avgTime=%.2fms, coherence=%.3f, throughput=%.1f tokens/s}"
                .formatted(totalProcessings, totalTokensProcessed, getAverageProcessingTimeMs(), 
                         averageSpatialCoherence, getThroughputTokensPerSecond());
        }
    }
}