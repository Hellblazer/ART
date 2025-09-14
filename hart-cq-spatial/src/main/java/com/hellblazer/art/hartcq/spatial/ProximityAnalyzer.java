package com.hellblazer.art.hartcq.spatial;

import com.hellblazer.art.hartcq.HARTCQConfig;
import com.hellblazer.art.hartcq.Token;
import org.joml.Vector2f;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Analyzes token proximity relationships using multiple distance metrics.
 * 
 * Provides comprehensive proximity analysis including Euclidean, Manhattan,
 * and Cosine distance measurements. Performs neighborhood analysis and
 * proximity-based weighting for spatial processing.
 * 
 * Leverages JOML for efficient vector operations and supports parallel
 * processing for large token sets.
 * 
 * @author Claude Code
 */
public class ProximityAnalyzer implements AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(ProximityAnalyzer.class);
    
    private final HARTCQConfig config;
    private final ProximityAnalyzerStats stats;
    
    // Analysis parameters
    private final float proximityThreshold;
    private final int maxNeighbors;
    private final boolean enableParallelProcessing;
    
    /**
     * Distance metric types for proximity analysis.
     */
    public enum DistanceMetric {
        EUCLIDEAN,
        MANHATTAN,
        COSINE,
        CHEBYSHEV
    }
    
    /**
     * Neighborhood relationship between tokens.
     */
    public static class TokenNeighborhood {
        private final Token centerToken;
        private final Vector2f centerPosition;
        private final List<NeighborInfo> neighbors;
        private final double averageDistance;
        private final double density;
        
        public TokenNeighborhood(Token centerToken, Vector2f centerPosition, List<NeighborInfo> neighbors) {
            this.centerToken = centerToken;
            this.centerPosition = new Vector2f(centerPosition);
            this.neighbors = new ArrayList<>(neighbors);
            this.averageDistance = neighbors.stream().mapToDouble(n -> n.distance).average().orElse(0.0);
            this.density = calculateDensity();
        }
        
        private double calculateDensity() {
            if (neighbors.isEmpty()) return 0.0;
            var maxDistance = neighbors.stream().mapToDouble(n -> n.distance).max().orElse(1.0);
            var area = Math.PI * maxDistance * maxDistance;
            return neighbors.size() / area;
        }
        
        public Token getCenterToken() { return centerToken; }
        public Vector2f getCenterPosition() { return new Vector2f(centerPosition); }
        public List<NeighborInfo> getNeighbors() { return new ArrayList<>(neighbors); }
        public double getAverageDistance() { return averageDistance; }
        public double getDensity() { return density; }
        public int getNeighborCount() { return neighbors.size(); }
    }
    
    /**
     * Information about a neighboring token.
     */
    public static class NeighborInfo {
        private final Token token;
        private final Vector2f position;
        private final double distance;
        private final double weight;
        private final DistanceMetric metric;
        
        public NeighborInfo(Token token, Vector2f position, double distance, double weight, DistanceMetric metric) {
            this.token = token;
            this.position = new Vector2f(position);
            this.distance = distance;
            this.weight = weight;
            this.metric = metric;
        }
        
        public Token getToken() { return token; }
        public Vector2f getPosition() { return new Vector2f(position); }
        public double getDistance() { return distance; }
        public double getWeight() { return weight; }
        public DistanceMetric getMetric() { return metric; }
    }
    
    /**
     * Result of proximity analysis.
     */
    public static class ProximityResult {
        private final Map<Token, TokenNeighborhood> neighborhoods;
        private final List<TokenCluster> clusters;
        private final Map<Token, List<NeighborInfo>> proximityGraph;
        private final double overallCoherence;
        private final DistanceMetric primaryMetric;
        
        public ProximityResult(Map<Token, TokenNeighborhood> neighborhoods,
                              List<TokenCluster> clusters,
                              Map<Token, List<NeighborInfo>> proximityGraph,
                              double overallCoherence,
                              DistanceMetric primaryMetric) {
            this.neighborhoods = new HashMap<>(neighborhoods);
            this.clusters = new ArrayList<>(clusters);
            this.proximityGraph = new HashMap<>(proximityGraph);
            this.overallCoherence = overallCoherence;
            this.primaryMetric = primaryMetric;
        }
        
        public Map<Token, TokenNeighborhood> getNeighborhoods() { return new HashMap<>(neighborhoods); }
        public List<TokenCluster> getClusters() { return new ArrayList<>(clusters); }
        public Map<Token, List<NeighborInfo>> getProximityGraph() { return new HashMap<>(proximityGraph); }
        public double getOverallCoherence() { return overallCoherence; }
        public DistanceMetric getPrimaryMetric() { return primaryMetric; }
        
        public static ProximityResult empty() {
            return new ProximityResult(new HashMap<>(), new ArrayList<>(), new HashMap<>(), 0.0, DistanceMetric.EUCLIDEAN);
        }
        
        @Override
        public String toString() {
            return "ProximityResult{neighborhoods=%d, clusters=%d, coherence=%.3f, metric=%s}"
                .formatted(neighborhoods.size(), clusters.size(), overallCoherence, primaryMetric);
        }
    }
    
    /**
     * Cluster of spatially proximate tokens.
     */
    public static class TokenCluster {
        private final List<Token> tokens;
        private final Vector2f centroid;
        private final double compactness;
        private final double separation;
        private final DistanceMetric clusterMetric;
        
        public TokenCluster(List<Token> tokens, Vector2f centroid, double compactness, double separation, DistanceMetric metric) {
            this.tokens = new ArrayList<>(tokens);
            this.centroid = new Vector2f(centroid);
            this.compactness = compactness;
            this.separation = separation;
            this.clusterMetric = metric;
        }
        
        public List<Token> getTokens() { return new ArrayList<>(tokens); }
        public Vector2f getCentroid() { return new Vector2f(centroid); }
        public double getCompactness() { return compactness; }
        public double getSeparation() { return separation; }
        public DistanceMetric getClusterMetric() { return clusterMetric; }
        public int size() { return tokens.size(); }
        
        @Override
        public String toString() {
            return "TokenCluster{tokens=%d, compactness=%.3f, separation=%.3f}"
                .formatted(tokens.size(), compactness, separation);
        }
    }
    
    /**
     * Creates a proximity analyzer with the given configuration.
     * 
     * @param config HART-CQ configuration
     */
    public ProximityAnalyzer(HARTCQConfig config) {
        this.config = Objects.requireNonNull(config, "Configuration cannot be null");
        this.stats = new ProximityAnalyzerStats();
        
        // Initialize analysis parameters
        this.proximityThreshold = 2.0f; // Maximum distance for proximity
        this.maxNeighbors = 8; // Maximum neighbors per token
        this.enableParallelProcessing = config.getPerformanceConfig().getMaxConcurrentProcessors() > 2;
        
        logger.info("ProximityAnalyzer initialized with threshold={}, maxNeighbors={}", 
                   proximityThreshold, maxNeighbors);
    }
    
    /**
     * Analyzes proximity relationships between tokens.
     * 
     * @param tokenPositions Map of tokens to their 2D positions
     * @return Proximity analysis result
     */
    public ProximityResult analyzeProximity(Map<Token, Vector2f> tokenPositions) {
        return analyzeProximity(tokenPositions, DistanceMetric.EUCLIDEAN);
    }
    
    /**
     * Analyzes proximity relationships using specified distance metric.
     * 
     * @param tokenPositions Map of tokens to their 2D positions
     * @param metric Distance metric to use
     * @return Proximity analysis result
     */
    public ProximityResult analyzeProximity(Map<Token, Vector2f> tokenPositions, DistanceMetric metric) {
        if (tokenPositions == null || tokenPositions.isEmpty()) {
            return ProximityResult.empty();
        }
        
        var startTime = System.currentTimeMillis();
        
        try {
            // Build neighborhoods for each token
            var neighborhoods = buildNeighborhoods(tokenPositions, metric);
            
            // Create proximity graph
            var proximityGraph = buildProximityGraph(neighborhoods);
            
            // Perform clustering
            var clusters = performProximityClustering(tokenPositions, neighborhoods, metric);
            
            // Calculate overall coherence
            var coherence = calculateOverallCoherence(neighborhoods, clusters);
            
            var result = new ProximityResult(neighborhoods, clusters, proximityGraph, coherence, metric);
            
            // Update statistics
            var processingTime = (int) (System.currentTimeMillis() - startTime);
            stats.recordAnalysis(tokenPositions.size(), processingTime, coherence);
            
            logger.debug("Proximity analysis completed: {}", result);
            return result;
            
        } catch (Exception e) {
            logger.error("Error in proximity analysis", e);
            return ProximityResult.empty();
        }
    }
    
    /**
     * Build neighborhood relationships for all tokens.
     */
    private Map<Token, TokenNeighborhood> buildNeighborhoods(Map<Token, Vector2f> tokenPositions, DistanceMetric metric) {
        var neighborhoods = new HashMap<Token, TokenNeighborhood>();
        
        var tokenStream = enableParallelProcessing ? 
            tokenPositions.entrySet().parallelStream() : 
            tokenPositions.entrySet().stream();
        
        var neighborhoodList = tokenStream
            .map(entry -> buildNeighborhoodForToken(entry.getKey(), entry.getValue(), tokenPositions, metric))
            .collect(Collectors.toList());
        
        // Convert to map
        for (var neighborhood : neighborhoodList) {
            neighborhoods.put(neighborhood.getCenterToken(), neighborhood);
        }
        
        return neighborhoods;
    }
    
    /**
     * Build neighborhood for a single token.
     */
    private TokenNeighborhood buildNeighborhoodForToken(Token centerToken, Vector2f centerPosition,
                                                       Map<Token, Vector2f> allPositions, DistanceMetric metric) {
        var neighbors = new ArrayList<NeighborInfo>();
        
        for (var entry : allPositions.entrySet()) {
            var token = entry.getKey();
            var position = entry.getValue();
            
            if (token.equals(centerToken)) continue;
            
            var distance = calculateDistance(centerPosition, position, metric);
            if (distance <= proximityThreshold) {
                var weight = calculateProximityWeight(distance, metric);
                neighbors.add(new NeighborInfo(token, position, distance, weight, metric));
            }
        }
        
        // Sort by distance and limit to maxNeighbors
        neighbors.sort(Comparator.comparingDouble(n -> n.distance));
        if (neighbors.size() > maxNeighbors) {
            neighbors = new ArrayList<>(neighbors.subList(0, maxNeighbors));
        }
        
        return new TokenNeighborhood(centerToken, centerPosition, neighbors);
    }
    
    /**
     * Calculate distance between two points using specified metric.
     */
    private double calculateDistance(Vector2f pos1, Vector2f pos2, DistanceMetric metric) {
        return switch (metric) {
            case EUCLIDEAN -> pos1.distance(pos2);
            case MANHATTAN -> Math.abs(pos1.x - pos2.x) + Math.abs(pos1.y - pos2.y);
            case COSINE -> 1.0 - calculateCosineDistance(pos1, pos2);
            case CHEBYSHEV -> Math.max(Math.abs(pos1.x - pos2.x), Math.abs(pos1.y - pos2.y));
        };
    }
    
    /**
     * Calculate cosine distance between two vectors.
     */
    private double calculateCosineDistance(Vector2f pos1, Vector2f pos2) {
        var dot = pos1.dot(pos2);
        var mag1 = pos1.length();
        var mag2 = pos2.length();
        
        if (mag1 == 0 || mag2 == 0) return 0.0;
        return dot / (mag1 * mag2);
    }
    
    /**
     * Calculate proximity weight based on distance.
     */
    private double calculateProximityWeight(double distance, DistanceMetric metric) {
        // Gaussian weighting function
        var sigma = proximityThreshold / 3.0; // 99.7% of weight within threshold
        return Math.exp(-0.5 * Math.pow(distance / sigma, 2));
    }
    
    /**
     * Build proximity graph from neighborhoods.
     */
    private Map<Token, List<NeighborInfo>> buildProximityGraph(Map<Token, TokenNeighborhood> neighborhoods) {
        var graph = new HashMap<Token, List<NeighborInfo>>();
        
        for (var entry : neighborhoods.entrySet()) {
            graph.put(entry.getKey(), entry.getValue().getNeighbors());
        }
        
        return graph;
    }
    
    /**
     * Perform clustering based on proximity relationships.
     */
    private List<TokenCluster> performProximityClustering(Map<Token, Vector2f> tokenPositions,
                                                         Map<Token, TokenNeighborhood> neighborhoods,
                                                         DistanceMetric metric) {
        var clusters = new ArrayList<TokenCluster>();
        var processed = new HashSet<Token>();
        
        for (var token : tokenPositions.keySet()) {
            if (processed.contains(token)) continue;
            
            var cluster = buildClusterFromSeed(token, neighborhoods, tokenPositions, processed, metric);
            if (cluster.size() >= 2) {
                clusters.add(cluster);
            }
        }
        
        return clusters;
    }
    
    /**
     * Build a cluster starting from a seed token.
     */
    private TokenCluster buildClusterFromSeed(Token seedToken,
                                            Map<Token, TokenNeighborhood> neighborhoods,
                                            Map<Token, Vector2f> positions,
                                            Set<Token> processed,
                                            DistanceMetric metric) {
        var clusterTokens = new ArrayList<Token>();
        var toProcess = new ArrayDeque<Token>();
        
        toProcess.add(seedToken);
        
        while (!toProcess.isEmpty()) {
            var currentToken = toProcess.poll();
            if (processed.contains(currentToken)) continue;
            
            processed.add(currentToken);
            clusterTokens.add(currentToken);
            
            var neighborhood = neighborhoods.get(currentToken);
            if (neighborhood != null) {
                for (var neighbor : neighborhood.getNeighbors()) {
                    if (!processed.contains(neighbor.getToken()) && neighbor.getWeight() > 0.5) {
                        toProcess.add(neighbor.getToken());
                    }
                }
            }
        }
        
        // Calculate cluster properties
        var centroid = calculateClusterCentroid(clusterTokens, positions);
        var compactness = calculateClusterCompactness(clusterTokens, positions, centroid, metric);
        var separation = calculateClusterSeparation(clusterTokens, positions, processed, metric);
        
        return new TokenCluster(clusterTokens, centroid, compactness, separation, metric);
    }
    
    /**
     * Calculate cluster centroid.
     */
    private Vector2f calculateClusterCentroid(List<Token> clusterTokens, Map<Token, Vector2f> positions) {
        var centroid = new Vector2f();
        for (var token : clusterTokens) {
            centroid.add(positions.get(token));
        }
        centroid.div(clusterTokens.size());
        return centroid;
    }
    
    /**
     * Calculate cluster compactness (average distance from centroid).
     */
    private double calculateClusterCompactness(List<Token> clusterTokens, Map<Token, Vector2f> positions,
                                             Vector2f centroid, DistanceMetric metric) {
        var totalDistance = 0.0;
        for (var token : clusterTokens) {
            var position = positions.get(token);
            totalDistance += calculateDistance(centroid, position, metric);
        }
        return clusterTokens.isEmpty() ? 0.0 : totalDistance / clusterTokens.size();
    }
    
    /**
     * Calculate cluster separation (minimum distance to nearest other cluster).
     */
    private double calculateClusterSeparation(List<Token> clusterTokens, Map<Token, Vector2f> positions,
                                            Set<Token> allProcessed, DistanceMetric metric) {
        var minSeparation = Double.MAX_VALUE;
        
        for (var token : clusterTokens) {
            var position = positions.get(token);
            for (var otherEntry : positions.entrySet()) {
                var otherToken = otherEntry.getKey();
                if (!clusterTokens.contains(otherToken) && allProcessed.contains(otherToken)) {
                    var distance = calculateDistance(position, otherEntry.getValue(), metric);
                    minSeparation = Math.min(minSeparation, distance);
                }
            }
        }
        
        return minSeparation == Double.MAX_VALUE ? 0.0 : minSeparation;
    }
    
    /**
     * Calculate overall coherence from neighborhoods and clusters.
     */
    private double calculateOverallCoherence(Map<Token, TokenNeighborhood> neighborhoods, List<TokenCluster> clusters) {
        // Average neighborhood density
        var avgNeighborhoodDensity = neighborhoods.values().stream()
            .mapToDouble(TokenNeighborhood::getDensity)
            .average()
            .orElse(0.0);
        
        // Cluster quality (compactness vs separation)
        var clusterQuality = 0.0;
        if (!clusters.isEmpty()) {
            clusterQuality = clusters.stream()
                .mapToDouble(c -> c.getSeparation() / Math.max(c.getCompactness(), 0.01))
                .average()
                .orElse(0.0);
        }
        
        // Normalize cluster quality to [0,1]
        clusterQuality = Math.min(1.0, clusterQuality / 10.0);
        
        // Combine metrics
        return 0.6 * Math.min(1.0, avgNeighborhoodDensity) + 0.4 * clusterQuality;
    }
    
    /**
     * Gets proximity analysis statistics.
     * 
     * @return proximity analyzer statistics
     */
    public ProximityAnalyzerStats getStatistics() {
        return stats.copy();
    }
    
    /**
     * Resets the proximity analyzer to initial state.
     */
    public void reset() {
        logger.info("Resetting proximity analyzer");
        stats.reset();
    }
    
    /**
     * Closes the proximity analyzer and releases resources.
     */
    @Override
    public void close() {
        logger.info("Closing proximity analyzer");
        // No resources to clean up currently
    }
    
    /**
     * Statistics for proximity analysis performance.
     */
    public static class ProximityAnalyzerStats {
        private int totalAnalyses = 0;
        private int totalTokensAnalyzed = 0;
        private long totalAnalysisTimeMs = 0;
        private double averageCoherence = 0.0;
        private long lastResetTime = System.currentTimeMillis();
        
        synchronized void recordAnalysis(int tokenCount, int analysisTimeMs, double coherence) {
            totalAnalyses++;
            totalTokensAnalyzed += tokenCount;
            totalAnalysisTimeMs += analysisTimeMs;
            
            // Update average coherence using exponential moving average
            var alpha = 0.1;
            averageCoherence = alpha * coherence + (1 - alpha) * averageCoherence;
        }
        
        public synchronized int getTotalAnalyses() { return totalAnalyses; }
        public synchronized int getTotalTokensAnalyzed() { return totalTokensAnalyzed; }
        public synchronized long getTotalAnalysisTimeMs() { return totalAnalysisTimeMs; }
        public synchronized double getAverageAnalysisTimeMs() {
            return totalAnalyses > 0 ? (double) totalAnalysisTimeMs / totalAnalyses : 0.0;
        }
        public synchronized double getAverageCoherence() { return averageCoherence; }
        
        synchronized void reset() {
            totalAnalyses = 0;
            totalTokensAnalyzed = 0;
            totalAnalysisTimeMs = 0;
            averageCoherence = 0.0;
            lastResetTime = System.currentTimeMillis();
        }
        
        synchronized ProximityAnalyzerStats copy() {
            var copy = new ProximityAnalyzerStats();
            copy.totalAnalyses = this.totalAnalyses;
            copy.totalTokensAnalyzed = this.totalTokensAnalyzed;
            copy.totalAnalysisTimeMs = this.totalAnalysisTimeMs;
            copy.averageCoherence = this.averageCoherence;
            copy.lastResetTime = this.lastResetTime;
            return copy;
        }
        
        @Override
        public String toString() {
            return "ProximityAnalyzerStats{analyses=%d, tokens=%d, avgTime=%.2fms, coherence=%.3f}"
                .formatted(totalAnalyses, totalTokensAnalyzed, getAverageAnalysisTimeMs(), averageCoherence);
        }
    }
}