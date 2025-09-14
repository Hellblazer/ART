package com.hellblazer.art.hartcq.spatial;

import com.hellblazer.art.hartcq.Token;
import org.joml.Vector2f;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Represents spatial patterns in token arrangements.
 * 
 * Provides pattern templates, pattern matching capabilities, and confidence
 * scoring for spatial relationships between tokens. Supports dynamic pattern
 * evolution and persistent pattern storage.
 * 
 * Uses JOML for efficient 2D vector operations and geometric calculations.
 * 
 * @author Claude Code
 */
public class SpatialPattern {
    private static final Logger logger = LoggerFactory.getLogger(SpatialPattern.class);
    
    private final String patternId;
    private final PatternType type;
    private final List<PatternElement> elements;
    private final Vector2f centroid;
    private final double confidence;
    private final long creationTime;
    private final long lastSeenTime;
    private final int occurrenceCount;
    private final Map<String, Object> properties;
    
    // Pattern matching parameters
    private static final double DEFAULT_POSITION_TOLERANCE = 0.5;
    private static final double DEFAULT_ANGLE_TOLERANCE = Math.PI / 8; // 22.5 degrees
    private static final double DEFAULT_SCALE_TOLERANCE = 0.3;
    
    /**
     * Types of spatial patterns.
     */
    public enum PatternType {
        LINEAR,      // Tokens arranged in a line
        CIRCULAR,    // Tokens arranged in a circle/arc
        CLUSTER,     // Dense grouping of tokens
        GRID,        // Regular grid-like arrangement
        SCATTERED,   // Sparse, irregular arrangement
        CUSTOM       // User-defined pattern
    }
    
    /**
     * Element within a spatial pattern.
     */
    public static class PatternElement {
        private final Vector2f relativePosition;
        private final Token.TokenType expectedType;
        private final double weight;
        private final Map<String, Object> constraints;
        
        public PatternElement(Vector2f relativePosition, Token.TokenType expectedType, double weight) {
            this.relativePosition = new Vector2f(relativePosition);
            this.expectedType = expectedType;
            this.weight = weight;
            this.constraints = new HashMap<>();
        }
        
        public PatternElement(Vector2f relativePosition, Token.TokenType expectedType, double weight, 
                            Map<String, Object> constraints) {
            this.relativePosition = new Vector2f(relativePosition);
            this.expectedType = expectedType;
            this.weight = weight;
            this.constraints = new HashMap<>(constraints);
        }
        
        public Vector2f getRelativePosition() { return new Vector2f(relativePosition); }
        public Token.TokenType getExpectedType() { return expectedType; }
        public double getWeight() { return weight; }
        public Map<String, Object> getConstraints() { return new HashMap<>(constraints); }
        
        /**
         * Checks if a token matches this pattern element.
         */
        public boolean matches(Token token, Vector2f position, Vector2f patternCentroid) {
            // Check token type
            if (expectedType != null && !expectedType.equals(token.getType())) {
                return false;
            }
            
            // Check relative position
            var expectedPos = new Vector2f(patternCentroid).add(relativePosition);
            var distance = expectedPos.distance(position);
            
            return distance <= DEFAULT_POSITION_TOLERANCE;
        }
        
        /**
         * Calculate match score for a token at a given position.
         */
        public double calculateMatchScore(Token token, Vector2f position, Vector2f patternCentroid) {
            var score = 0.0;
            
            // Type match score
            if (expectedType == null || expectedType.equals(token.getType())) {
                score += 0.5;
            }
            
            // Position match score
            var expectedPos = new Vector2f(patternCentroid).add(relativePosition);
            var distance = expectedPos.distance(position);
            var positionScore = Math.exp(-distance * distance / (2 * DEFAULT_POSITION_TOLERANCE * DEFAULT_POSITION_TOLERANCE));
            score += 0.5 * positionScore;
            
            return score * weight;
        }
    }
    
    /**
     * Result of pattern matching operation.
     */
    public static class PatternMatchResult {
        private final SpatialPattern pattern;
        private final List<Token> matchedTokens;
        private final Map<Token, Vector2f> matchedPositions;
        private final double overallScore;
        private final double coverage;
        private final List<PatternElement> unmatchedElements;
        
        public PatternMatchResult(SpatialPattern pattern, List<Token> matchedTokens, 
                                Map<Token, Vector2f> matchedPositions, double overallScore, 
                                double coverage, List<PatternElement> unmatchedElements) {
            this.pattern = pattern;
            this.matchedTokens = new ArrayList<>(matchedTokens);
            this.matchedPositions = new HashMap<>(matchedPositions);
            this.overallScore = overallScore;
            this.coverage = coverage;
            this.unmatchedElements = new ArrayList<>(unmatchedElements);
        }
        
        public SpatialPattern getPattern() { return pattern; }
        public List<Token> getMatchedTokens() { return new ArrayList<>(matchedTokens); }
        public Map<Token, Vector2f> getMatchedPositions() { return new HashMap<>(matchedPositions); }
        public double getOverallScore() { return overallScore; }
        public double getCoverage() { return coverage; }
        public List<PatternElement> getUnmatchedElements() { return new ArrayList<>(unmatchedElements); }
        
        public boolean isGoodMatch() {
            return overallScore > 0.7 && coverage > 0.6;
        }
        
        @Override
        public String toString() {
            return "PatternMatchResult{pattern=%s, score=%.3f, coverage=%.3f, matched=%d}"
                .formatted(pattern.getPatternId(), overallScore, coverage, matchedTokens.size());
        }
    }
    
    /**
     * Creates a spatial pattern.
     */
    private SpatialPattern(String patternId, PatternType type, List<PatternElement> elements, 
                          Vector2f centroid, double confidence, long creationTime, 
                          long lastSeenTime, int occurrenceCount, Map<String, Object> properties) {
        this.patternId = patternId;
        this.type = type;
        this.elements = new ArrayList<>(elements);
        this.centroid = new Vector2f(centroid);
        this.confidence = confidence;
        this.creationTime = creationTime;
        this.lastSeenTime = lastSeenTime;
        this.occurrenceCount = occurrenceCount;
        this.properties = new HashMap<>(properties);
    }
    
    // Getters
    public String getPatternId() { return patternId; }
    public PatternType getType() { return type; }
    public List<PatternElement> getElements() { return new ArrayList<>(elements); }
    public Vector2f getCentroid() { return new Vector2f(centroid); }
    public double getConfidence() { return confidence; }
    public long getCreationTime() { return creationTime; }
    public long getLastSeenTime() { return lastSeenTime; }
    public int getOccurrenceCount() { return occurrenceCount; }
    public Map<String, Object> getProperties() { return new HashMap<>(properties); }
    
    /**
     * Creates a spatial pattern from a cluster of tokens.
     */
    public static SpatialPattern fromCluster(List<Token> tokens, Map<Token, Vector2f> positions) {
        if (tokens.isEmpty()) {
            throw new IllegalArgumentException("Cannot create pattern from empty token list");
        }
        
        var patternId = generatePatternId(tokens);
        var centroid = calculateCentroid(tokens, positions);
        var elements = createElementsFromTokens(tokens, positions, centroid);
        var type = determinePatternType(elements, positions);
        var confidence = calculateInitialConfidence(elements, positions);
        var currentTime = System.currentTimeMillis();
        
        return new SpatialPattern(
            patternId, type, elements, centroid, confidence, 
            currentTime, currentTime, 1, new HashMap<>()
        );
    }
    
    /**
     * Creates a linear pattern from tokens.
     */
    public static SpatialPattern createLinearPattern(List<Token> tokens, Map<Token, Vector2f> positions) {
        var patternId = "linear_" + generatePatternId(tokens);
        var centroid = calculateCentroid(tokens, positions);
        var elements = createLinearElements(tokens, positions, centroid);
        var confidence = calculateLinearConfidence(elements, positions);
        var currentTime = System.currentTimeMillis();
        
        return new SpatialPattern(
            patternId, PatternType.LINEAR, elements, centroid, confidence,
            currentTime, currentTime, 1, new HashMap<>()
        );
    }
    
    /**
     * Creates a circular pattern from tokens.
     */
    public static SpatialPattern createCircularPattern(List<Token> tokens, Map<Token, Vector2f> positions) {
        var patternId = "circular_" + generatePatternId(tokens);
        var centroid = calculateCentroid(tokens, positions);
        var elements = createCircularElements(tokens, positions, centroid);
        var confidence = calculateCircularConfidence(elements, positions);
        var currentTime = System.currentTimeMillis();
        
        return new SpatialPattern(
            patternId, PatternType.CIRCULAR, elements, centroid, confidence,
            currentTime, currentTime, 1, new HashMap<>()
        );
    }
    
    /**
     * Generate a unique pattern ID based on tokens.
     */
    private static String generatePatternId(List<Token> tokens) {
        var tokenTypes = tokens.stream()
            .map(t -> t.getType().name())
            .sorted()
            .collect(Collectors.joining("_"));
        return "pattern_" + Math.abs(tokenTypes.hashCode());
    }
    
    /**
     * Calculate centroid of token positions.
     */
    private static Vector2f calculateCentroid(List<Token> tokens, Map<Token, Vector2f> positions) {
        var centroid = new Vector2f();
        for (var token : tokens) {
            var position = positions.get(token);
            if (position != null) {
                centroid.add(position);
            }
        }
        centroid.div(tokens.size());
        return centroid;
    }
    
    /**
     * Create pattern elements from tokens and positions.
     */
    private static List<PatternElement> createElementsFromTokens(List<Token> tokens, 
                                                               Map<Token, Vector2f> positions, 
                                                               Vector2f centroid) {
        var elements = new ArrayList<PatternElement>();
        
        for (var token : tokens) {
            var position = positions.get(token);
            if (position != null) {
                var relativePos = new Vector2f(position).sub(centroid);
                var weight = 1.0 / tokens.size(); // Equal weight for all elements
                elements.add(new PatternElement(relativePos, token.getType(), weight));
            }
        }
        
        return elements;
    }
    
    /**
     * Create linear arrangement elements.
     */
    private static List<PatternElement> createLinearElements(List<Token> tokens, 
                                                           Map<Token, Vector2f> positions, 
                                                           Vector2f centroid) {
        // Sort tokens by their position along the line
        var sortedTokens = new ArrayList<>(tokens);
        sortedTokens.sort((t1, t2) -> {
            var p1 = positions.get(t1);
            var p2 = positions.get(t2);
            return Double.compare(p1.x + p1.y, p2.x + p2.y); // Simple diagonal sort
        });
        
        return createElementsFromTokens(sortedTokens, positions, centroid);
    }
    
    /**
     * Create circular arrangement elements.
     */
    private static List<PatternElement> createCircularElements(List<Token> tokens, 
                                                             Map<Token, Vector2f> positions, 
                                                             Vector2f centroid) {
        // Sort tokens by their angle around the centroid
        var sortedTokens = new ArrayList<>(tokens);
        sortedTokens.sort((t1, t2) -> {
            var p1 = new Vector2f(positions.get(t1)).sub(centroid);
            var p2 = new Vector2f(positions.get(t2)).sub(centroid);
            var angle1 = Math.atan2(p1.y, p1.x);
            var angle2 = Math.atan2(p2.y, p2.x);
            return Double.compare(angle1, angle2);
        });
        
        return createElementsFromTokens(sortedTokens, positions, centroid);
    }
    
    /**
     * Determine the pattern type from elements and positions.
     */
    private static PatternType determinePatternType(List<PatternElement> elements, Map<Token, Vector2f> positions) {
        if (elements.size() < 2) {
            return PatternType.CLUSTER;
        }
        
        // Analyze spatial arrangement
        var positions2D = elements.stream()
            .map(PatternElement::getRelativePosition)
            .collect(Collectors.toList());
        
        var linearity = calculateLinearity(positions2D);
        var circularity = calculateCircularity(positions2D);
        
        if (linearity > 0.8) {
            return PatternType.LINEAR;
        } else if (circularity > 0.7) {
            return PatternType.CIRCULAR;
        } else if (calculateDensity(positions2D) > 0.5) {
            return PatternType.CLUSTER;
        } else {
            return PatternType.SCATTERED;
        }
    }
    
    /**
     * Calculate how linear the positions are.
     */
    private static double calculateLinearity(List<Vector2f> positions) {
        if (positions.size() < 3) return 1.0;
        
        // Fit a line and calculate R-squared
        var n = positions.size();
        var sumX = 0.0f;
        var sumY = 0.0f;
        var sumXY = 0.0f;
        var sumXX = 0.0f;
        
        for (var pos : positions) {
            sumX += pos.x;
            sumY += pos.y;
            sumXY += pos.x * pos.y;
            sumXX += pos.x * pos.x;
        }
        
        var meanX = sumX / n;
        var meanY = sumY / n;
        
        var slope = (sumXY - n * meanX * meanY) / (sumXX - n * meanX * meanX);
        var intercept = meanY - slope * meanX;
        
        // Calculate R-squared
        var ssRes = 0.0;
        var ssTot = 0.0;
        
        for (var pos : positions) {
            var predicted = slope * pos.x + intercept;
            var residual = pos.y - predicted;
            ssRes += residual * residual;
            
            var totalVariation = pos.y - meanY;
            ssTot += totalVariation * totalVariation;
        }
        
        return ssTot > 0 ? 1.0 - (ssRes / ssTot) : 0.0;
    }
    
    /**
     * Calculate how circular the positions are.
     */
    private static double calculateCircularity(List<Vector2f> positions) {
        if (positions.size() < 3) return 0.0;
        
        // Calculate average distance from center
        var avgDistance = positions.stream()
            .mapToDouble(Vector2f::length)
            .average()
            .orElse(0.0);
        
        if (avgDistance == 0) return 0.0;
        
        // Calculate variance in distances
        var variance = positions.stream()
            .mapToDouble(pos -> {
                var distance = pos.length();
                var diff = distance - avgDistance;
                return diff * diff;
            })
            .average()
            .orElse(0.0);
        
        var coefficient = Math.sqrt(variance) / avgDistance;
        return Math.max(0.0, 1.0 - coefficient); // Lower variance = higher circularity
    }
    
    /**
     * Calculate density of positions.
     */
    private static double calculateDensity(List<Vector2f> positions) {
        if (positions.size() < 2) return 1.0;
        
        var totalDistance = 0.0;
        var count = 0;
        
        for (var i = 0; i < positions.size(); i++) {
            for (var j = i + 1; j < positions.size(); j++) {
                totalDistance += positions.get(i).distance(positions.get(j));
                count++;
            }
        }
        
        var avgDistance = totalDistance / count;
        return Math.max(0.0, 1.0 - avgDistance / 5.0); // Normalize to [0,1]
    }
    
    /**
     * Calculate initial confidence for the pattern.
     */
    private static double calculateInitialConfidence(List<PatternElement> elements, Map<Token, Vector2f> positions) {
        if (elements.isEmpty()) return 0.0;
        
        // Base confidence on pattern regularity and element count
        var elementCount = elements.size();
        var countScore = Math.min(1.0, elementCount / 5.0); // Better with more elements
        
        // Calculate regularity score
        var regularityScore = calculateRegularityScore(elements);
        
        return 0.6 * countScore + 0.4 * regularityScore;
    }
    
    /**
     * Calculate linear pattern confidence.
     */
    private static double calculateLinearConfidence(List<PatternElement> elements, Map<Token, Vector2f> positions) {
        var baseConfidence = calculateInitialConfidence(elements, positions);
        var positions2D = elements.stream().map(PatternElement::getRelativePosition).collect(Collectors.toList());
        var linearity = calculateLinearity(positions2D);
        
        return 0.7 * baseConfidence + 0.3 * linearity;
    }
    
    /**
     * Calculate circular pattern confidence.
     */
    private static double calculateCircularConfidence(List<PatternElement> elements, Map<Token, Vector2f> positions) {
        var baseConfidence = calculateInitialConfidence(elements, positions);
        var positions2D = elements.stream().map(PatternElement::getRelativePosition).collect(Collectors.toList());
        var circularity = calculateCircularity(positions2D);
        
        return 0.7 * baseConfidence + 0.3 * circularity;
    }
    
    /**
     * Calculate regularity score for pattern elements.
     */
    private static double calculateRegularityScore(List<PatternElement> elements) {
        if (elements.size() < 2) return 1.0;
        
        // Calculate variance in distances between consecutive elements
        var distances = new ArrayList<Double>();
        for (var i = 0; i < elements.size() - 1; i++) {
            var pos1 = elements.get(i).getRelativePosition();
            var pos2 = elements.get(i + 1).getRelativePosition();
            distances.add((double) pos1.distance(pos2));
        }
        
        var meanDistance = distances.stream().mapToDouble(d -> d).average().orElse(0.0);
        if (meanDistance == 0) return 1.0;
        
        var variance = distances.stream()
            .mapToDouble(d -> {
                var diff = d - meanDistance;
                return diff * diff;
            })
            .average()
            .orElse(0.0);
        
        var coefficient = Math.sqrt(variance) / meanDistance;
        return Math.max(0.0, 1.0 - coefficient); // Lower variance = higher regularity
    }
    
    /**
     * Check if a token at a position matches this pattern.
     */
    public boolean matches(Token token, Vector2f position) {
        return getBestMatch(List.of(token), Map.of(token, position)).isGoodMatch();
    }
    
    /**
     * Find the best match for this pattern against a set of tokens.
     */
    public PatternMatchResult getBestMatch(List<Token> tokens, Map<Token, Vector2f> positions) {
        var bestScore = 0.0;
        List<Token> bestMatchedTokens = new ArrayList<>();
        Map<Token, Vector2f> bestMatchedPositions = new HashMap<>();
        List<PatternElement> bestUnmatchedElements = new ArrayList<>(elements);
        
        // Try different centroid positions
        for (var candidateToken : tokens) {
            var candidatePosition = positions.get(candidateToken);
            if (candidatePosition == null) continue;
            
            var match = matchWithCentroid(candidatePosition, tokens, positions);
            if (match.getOverallScore() > bestScore) {
                bestScore = match.getOverallScore();
                bestMatchedTokens = match.getMatchedTokens();
                bestMatchedPositions = match.getMatchedPositions();
                bestUnmatchedElements = match.getUnmatchedElements();
            }
        }
        
        var coverage = elements.isEmpty() ? 1.0 : 
            (double) (elements.size() - bestUnmatchedElements.size()) / elements.size();
        
        return new PatternMatchResult(this, bestMatchedTokens, bestMatchedPositions, 
                                    bestScore, coverage, bestUnmatchedElements);
    }
    
    /**
     * Match pattern with a specific centroid position.
     */
    private PatternMatchResult matchWithCentroid(Vector2f candidateCentroid, List<Token> tokens, 
                                               Map<Token, Vector2f> positions) {
        var matchedTokens = new ArrayList<Token>();
        var matchedPositions = new HashMap<Token, Vector2f>();
        var totalScore = 0.0;
        var unmatchedElements = new ArrayList<PatternElement>();
        
        for (var element : elements) {
            var bestElementScore = 0.0;
            Token bestToken = null;
            Vector2f bestPosition = null;
            
            for (var token : tokens) {
                var position = positions.get(token);
                if (position == null || matchedTokens.contains(token)) continue;
                
                var score = element.calculateMatchScore(token, position, candidateCentroid);
                if (score > bestElementScore) {
                    bestElementScore = score;
                    bestToken = token;
                    bestPosition = position;
                }
            }
            
            if (bestElementScore > 0.3) { // Minimum threshold for match
                matchedTokens.add(bestToken);
                matchedPositions.put(bestToken, bestPosition);
                totalScore += bestElementScore;
            } else {
                unmatchedElements.add(element);
            }
        }
        
        var avgScore = elements.isEmpty() ? 0.0 : totalScore / elements.size();
        var coverage = elements.isEmpty() ? 1.0 : (double) matchedTokens.size() / elements.size();
        
        return new PatternMatchResult(this, matchedTokens, matchedPositions, 
                                    avgScore, coverage, unmatchedElements);
    }
    
    /**
     * Update pattern with new token observations.
     */
    public SpatialPattern updateWithTokens(List<Token> tokens, Map<Token, Vector2f> positions) {
        var newCentroid = calculateCentroid(tokens, positions);
        var newElements = createElementsFromTokens(tokens, positions, newCentroid);
        var newConfidence = updateConfidence(newElements, positions);
        var newOccurrenceCount = occurrenceCount + 1;
        
        return new SpatialPattern(
            patternId, type, newElements, newCentroid, newConfidence,
            creationTime, System.currentTimeMillis(), newOccurrenceCount, properties
        );
    }
    
    /**
     * Update confidence based on new observations.
     */
    private double updateConfidence(List<PatternElement> newElements, Map<Token, Vector2f> positions) {
        var newConfidence = calculateInitialConfidence(newElements, positions);
        
        // Exponential moving average with occurrence count influence
        var alpha = Math.min(0.3, 1.0 / occurrenceCount);
        return alpha * newConfidence + (1 - alpha) * confidence;
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof SpatialPattern pattern)) return false;
        return Objects.equals(patternId, pattern.patternId);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(patternId);
    }
    
    @Override
    public String toString() {
        return "SpatialPattern{id='%s', type=%s, elements=%d, confidence=%.3f, occurrences=%d}"
            .formatted(patternId, type, elements.size(), confidence, occurrenceCount);
    }
}