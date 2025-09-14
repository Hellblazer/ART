package com.hellblazer.art.hartcq.spatial;

import com.hellblazer.art.hartcq.HARTCQConfig;
import com.hellblazer.art.hartcq.Token;
import org.joml.Vector2f;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * 2D representation of token relationships with spatial mapping capabilities.
 * 
 * Maps tokens to 2D coordinate space based on linguistic properties, positional
 * information, and semantic relationships. Supports proximity-based clustering
 * and spatial pattern detection.
 * 
 * Uses JOML Vector2f for efficient 2D vector operations and maintains spatial
 * coherence across processing windows.
 * 
 * @author Claude Code
 */
public class SpatialMap implements AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(SpatialMap.class);
    
    private final HARTCQConfig config;
    private final Map<String, Vector2f> tokenTypeHistory;
    private final Map<Token.TokenType, SpatialRegion> typeRegions;
    private final Random spatialRandom;
    private final SpatialMapStats stats;
    
    // Spatial mapping parameters
    private final float mapWidth;
    private final float mapHeight;
    private final float noiseLevel;
    private final float historicalWeight;
    
    /**
     * Represents a spatial region for a specific token type.
     */
    private static class SpatialRegion {
        private final Vector2f center;
        private final float radius;
        private final Token.TokenType tokenType;
        private final List<Vector2f> recentPositions;
        
        public SpatialRegion(Token.TokenType tokenType, Vector2f center, float radius) {
            this.tokenType = tokenType;
            this.center = new Vector2f(center);
            this.radius = radius;
            this.recentPositions = new ArrayList<>();
        }
        
        public Vector2f getCenter() { return new Vector2f(center); }
        public float getRadius() { return radius; }
        public Token.TokenType getTokenType() { return tokenType; }
        
        public Vector2f generatePosition(Random random) {
            // Generate position within circular region with some variation
            var angle = random.nextFloat() * 2 * Math.PI;
            var distance = random.nextFloat() * radius * 0.8f; // Stay within 80% of radius
            
            var position = new Vector2f(
                center.x + (float) Math.cos(angle) * distance,
                center.y + (float) Math.sin(angle) * distance
            );
            
            // Track recent positions for region updates
            recentPositions.add(new Vector2f(position));
            if (recentPositions.size() > 50) {
                recentPositions.remove(0);
            }
            
            return position;
        }
        
        public void updateCenter() {
            if (recentPositions.isEmpty()) return;
            
            var newCenter = new Vector2f();
            for (var pos : recentPositions) {
                newCenter.add(pos);
            }
            newCenter.div(recentPositions.size());
            
            // Exponential moving average update
            center.lerp(newCenter, 0.1f);
        }
    }
    
    /**
     * Result of spatial clustering analysis.
     */
    public static class ClusterResult {
        private final List<TokenCluster> clusters;
        private final Map<Token, Vector2f> positions;
        private final double clusterQuality;
        
        public ClusterResult(List<TokenCluster> clusters, Map<Token, Vector2f> positions, double clusterQuality) {
            this.clusters = new ArrayList<>(clusters);
            this.positions = new HashMap<>(positions);
            this.clusterQuality = clusterQuality;
        }
        
        public List<TokenCluster> getClusters() { return new ArrayList<>(clusters); }
        public Map<Token, Vector2f> getPositions() { return new HashMap<>(positions); }
        public double getClusterQuality() { return clusterQuality; }
    }
    
    /**
     * Represents a cluster of spatially related tokens.
     */
    public static class TokenCluster {
        private final List<Token> tokens;
        private final Vector2f centroid;
        private final float radius;
        private final double density;
        
        public TokenCluster(List<Token> tokens, Vector2f centroid, float radius, double density) {
            this.tokens = new ArrayList<>(tokens);
            this.centroid = new Vector2f(centroid);
            this.radius = radius;
            this.density = density;
        }
        
        public List<Token> getTokens() { return new ArrayList<>(tokens); }
        public Vector2f getCentroid() { return new Vector2f(centroid); }
        public float getRadius() { return radius; }
        public double getDensity() { return density; }
        public int size() { return tokens.size(); }
    }
    
    /**
     * Creates a spatial map with the given configuration.
     * 
     * @param config HART-CQ configuration
     */
    public SpatialMap(HARTCQConfig config) {
        this.config = Objects.requireNonNull(config, "Configuration cannot be null");
        this.tokenTypeHistory = new ConcurrentHashMap<>();
        this.typeRegions = new ConcurrentHashMap<>();
        this.spatialRandom = new Random(42); // Deterministic for testing
        this.stats = new SpatialMapStats();
        
        // Initialize spatial parameters
        this.mapWidth = 10.0f;
        this.mapHeight = 10.0f;
        this.noiseLevel = 0.1f;
        this.historicalWeight = 0.3f;
        
        initializeTypeRegions();
        logger.info("SpatialMap initialized with {}x{} coordinate space", mapWidth, mapHeight);
    }
    
    /**
     * Initialize spatial regions for different token types.
     */
    private void initializeTypeRegions() {
        // Define spatial regions for different token types
        typeRegions.put(Token.TokenType.WORD, 
            new SpatialRegion(Token.TokenType.WORD, new Vector2f(2.5f, 2.5f), 1.5f));
        typeRegions.put(Token.TokenType.PUNCTUATION, 
            new SpatialRegion(Token.TokenType.PUNCTUATION, new Vector2f(7.5f, 2.5f), 1.0f));
        typeRegions.put(Token.TokenType.NUMBER, 
            new SpatialRegion(Token.TokenType.NUMBER, new Vector2f(2.5f, 7.5f), 1.2f));
        typeRegions.put(Token.TokenType.SYMBOL, 
            new SpatialRegion(Token.TokenType.SYMBOL, new Vector2f(7.5f, 7.5f), 0.8f));
        typeRegions.put(Token.TokenType.WHITESPACE, 
            new SpatialRegion(Token.TokenType.WHITESPACE, new Vector2f(5.0f, 5.0f), 0.5f));
        typeRegions.put(Token.TokenType.SPECIAL, 
            new SpatialRegion(Token.TokenType.SPECIAL, new Vector2f(1.0f, 8.0f), 0.7f));
        typeRegions.put(Token.TokenType.UNKNOWN, 
            new SpatialRegion(Token.TokenType.UNKNOWN, new Vector2f(8.0f, 1.0f), 0.6f));
    }
    
    /**
     * Maps tokens to 2D coordinate space.
     * 
     * @param tokens Input tokens to map
     * @param contextPositions Historical position context
     * @return Map of tokens to their 2D positions
     */
    public Map<Token, Vector2f> mapTokensToSpace(Token[] tokens, Map<String, Vector2f> contextPositions) {
        var positions = new HashMap<Token, Vector2f>();
        
        for (var i = 0; i < tokens.length; i++) {
            var token = tokens[i];
            var position = calculateTokenPosition(token, i, tokens.length, contextPositions);
            positions.put(token, position);
        }
        
        // Update statistics
        stats.recordMapping(tokens.length);
        
        // Update type regions based on new positions
        updateTypeRegions();
        
        logger.debug("Mapped {} tokens to spatial coordinates", tokens.length);
        return positions;
    }
    
    /**
     * Calculate position for a single token.
     */
    private Vector2f calculateTokenPosition(Token token, int index, int totalTokens, 
                                          Map<String, Vector2f> contextPositions) {
        var basePosition = calculateBasePosition(token, index, totalTokens);
        
        // Apply historical context if available
        var tokenTypeKey = token.getType().name();
        var historicalPosition = contextPositions.get(tokenTypeKey);
        
        if (historicalPosition != null) {
            // Blend base position with historical position
            var blended = new Vector2f(basePosition);
            blended.lerp(historicalPosition, historicalWeight);
            basePosition = blended;
        }
        
        // Add controlled noise for variation
        basePosition.x += (spatialRandom.nextFloat() - 0.5f) * noiseLevel;
        basePosition.y += (spatialRandom.nextFloat() - 0.5f) * noiseLevel;
        
        // Ensure position is within map bounds
        basePosition.x = Math.max(0, Math.min(mapWidth, basePosition.x));
        basePosition.y = Math.max(0, Math.min(mapHeight, basePosition.y));
        
        return basePosition;
    }
    
    /**
     * Calculate base position using token properties.
     */
    private Vector2f calculateBasePosition(Token token, int index, int totalTokens) {
        var region = typeRegions.get(token.getType());
        
        if (region != null) {
            var regionPosition = region.generatePosition(spatialRandom);
            
            // Adjust based on sequence position
            var sequenceWeight = (float) index / totalTokens;
            regionPosition.x += sequenceWeight * 0.5f - 0.25f; // Small positional influence
            
            return regionPosition;
        }
        
        // Fallback: linear mapping based on position and type
        var x = (float) index / totalTokens * mapWidth;
        var y = token.getType().ordinal() / (float) Token.TokenType.values().length * mapHeight;
        
        return new Vector2f(x, y);
    }
    
    /**
     * Update type region centers based on recent token positions.
     */
    private void updateTypeRegions() {
        typeRegions.values().forEach(SpatialRegion::updateCenter);
    }
    
    /**
     * Performs spatial clustering of token positions.
     * 
     * @param tokenPositions Map of tokens to positions
     * @return Clustering result
     */
    public ClusterResult performClustering(Map<Token, Vector2f> tokenPositions) {
        var clusters = new ArrayList<TokenCluster>();
        var processed = new HashSet<Token>();
        
        for (var entry : tokenPositions.entrySet()) {
            var token = entry.getKey();
            if (processed.contains(token)) continue;
            
            var cluster = buildClusterAroundToken(token, tokenPositions, processed);
            if (cluster.size() >= 2) { // Only keep clusters with multiple tokens
                clusters.add(cluster);
            }
        }
        
        var clusterQuality = calculateClusteringQuality(clusters, tokenPositions);
        return new ClusterResult(clusters, tokenPositions, clusterQuality);
    }
    
    /**
     * Build a cluster around a seed token using proximity.
     */
    private TokenCluster buildClusterAroundToken(Token seedToken, Map<Token, Vector2f> positions, Set<Token> processed) {
        var clusterTokens = new ArrayList<Token>();
        var seedPosition = positions.get(seedToken);
        var clusterRadius = 1.5f; // Clustering radius
        
        clusterTokens.add(seedToken);
        processed.add(seedToken);
        
        // Find nearby tokens
        for (var entry : positions.entrySet()) {
            var token = entry.getKey();
            var position = entry.getValue();
            
            if (!processed.contains(token) && seedPosition.distance(position) <= clusterRadius) {
                clusterTokens.add(token);
                processed.add(token);
            }
        }
        
        // Calculate cluster properties
        var centroid = calculateCentroid(clusterTokens, positions);
        var maxDistance = calculateMaxDistanceFromCentroid(clusterTokens, positions, centroid);
        var density = calculateClusterDensity(clusterTokens, positions, maxDistance);
        
        return new TokenCluster(clusterTokens, centroid, maxDistance, density);
    }
    
    /**
     * Calculate centroid of a cluster.
     */
    private Vector2f calculateCentroid(List<Token> tokens, Map<Token, Vector2f> positions) {
        var centroid = new Vector2f();
        for (var token : tokens) {
            centroid.add(positions.get(token));
        }
        centroid.div(tokens.size());
        return centroid;
    }
    
    /**
     * Calculate maximum distance from centroid.
     */
    private float calculateMaxDistanceFromCentroid(List<Token> tokens, Map<Token, Vector2f> positions, Vector2f centroid) {
        var maxDistance = 0.0f;
        for (var token : tokens) {
            var distance = centroid.distance(positions.get(token));
            maxDistance = Math.max(maxDistance, distance);
        }
        return maxDistance;
    }
    
    /**
     * Calculate cluster density (tokens per unit area).
     */
    private double calculateClusterDensity(List<Token> tokens, Map<Token, Vector2f> positions, float radius) {
        if (radius <= 0) return Double.MAX_VALUE;
        var area = Math.PI * radius * radius;
        return tokens.size() / area;
    }
    
    /**
     * Calculate overall clustering quality metric.
     */
    private double calculateClusteringQuality(List<TokenCluster> clusters, Map<Token, Vector2f> positions) {
        if (clusters.isEmpty()) return 0.0;
        
        // Calculate average cluster density
        var avgDensity = clusters.stream()
            .mapToDouble(TokenCluster::getDensity)
            .average()
            .orElse(0.0);
        
        // Calculate silhouette-like score
        var totalTokens = positions.size();
        var clusteredTokens = clusters.stream().mapToInt(TokenCluster::size).sum();
        var coverageRatio = (double) clusteredTokens / totalTokens;
        
        // Combine metrics
        return 0.7 * Math.min(1.0, avgDensity / 10.0) + 0.3 * coverageRatio;
    }
    
    /**
     * Finds tokens within a specified distance from a point.
     * 
     * @param center Center point
     * @param radius Search radius
     * @param tokenPositions All token positions
     * @return List of tokens within the radius
     */
    public List<Token> findTokensInRadius(Vector2f center, float radius, Map<Token, Vector2f> tokenPositions) {
        return tokenPositions.entrySet().stream()
            .filter(entry -> center.distance(entry.getValue()) <= radius)
            .map(Map.Entry::getKey)
            .collect(Collectors.toList());
    }
    
    /**
     * Gets spatial processing statistics.
     * 
     * @return spatial map statistics
     */
    public SpatialMapStats getStatistics() {
        return stats.copy();
    }
    
    /**
     * Resets the spatial map to initial state.
     */
    public void reset() {
        logger.info("Resetting spatial map");
        tokenTypeHistory.clear();
        initializeTypeRegions();
        stats.reset();
    }
    
    /**
     * Closes the spatial map and releases resources.
     */
    @Override
    public void close() {
        logger.info("Closing spatial map");
        tokenTypeHistory.clear();
        typeRegions.clear();
    }
    
    /**
     * Statistics for spatial mapping performance.
     */
    public static class SpatialMapStats {
        private int totalMappings = 0;
        private int totalTokensMapped = 0;
        private long lastResetTime = System.currentTimeMillis();
        
        synchronized void recordMapping(int tokenCount) {
            totalMappings++;
            totalTokensMapped += tokenCount;
        }
        
        public synchronized int getTotalMappings() { return totalMappings; }
        public synchronized int getTotalTokensMapped() { return totalTokensMapped; }
        public synchronized double getAverageTokensPerMapping() {
            return totalMappings > 0 ? (double) totalTokensMapped / totalMappings : 0.0;
        }
        
        synchronized void reset() {
            totalMappings = 0;
            totalTokensMapped = 0;
            lastResetTime = System.currentTimeMillis();
        }
        
        synchronized SpatialMapStats copy() {
            var copy = new SpatialMapStats();
            copy.totalMappings = this.totalMappings;
            copy.totalTokensMapped = this.totalTokensMapped;
            copy.lastResetTime = this.lastResetTime;
            return copy;
        }
        
        @Override
        public String toString() {
            return "SpatialMapStats{mappings=%d, tokens=%d, avgTokensPerMapping=%.1f}"
                .formatted(totalMappings, totalTokensMapped, getAverageTokensPerMapping());
        }
    }
}