package com.hellblazer.art.nlp.processor.consensus;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.nlp.processor.ChannelResult;
import com.hellblazer.art.nlp.processor.ConsensusResult;

import java.util.*;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Attention-based consensus strategy that computes dynamic attention weights
 * for each channel based on result quality, confidence, and cross-channel agreement.
 * Implements scaled dot-product attention mechanism for channel weighting.
 */
public class AttentionConsensus implements ConsensusStrategy {
    private static final Logger log = LoggerFactory.getLogger(AttentionConsensus.class);
    
    private final double confidenceThreshold;
    private final boolean normalizeAttentionWeights;
    private final double temperatureScaling;
    private final int attentionDimensions;
    private final boolean usePositionalEncoding;
    
    /**
     * Create attention consensus with default parameters.
     */
    public AttentionConsensus() {
        this(0.5, true, 1.0, 64, false);
    }
    
    /**
     * Create attention consensus with custom parameters.
     * 
     * @param confidenceThreshold Minimum confidence for channel participation
     * @param normalizeAttentionWeights Whether to normalize attention weights to sum to 1
     * @param temperatureScaling Temperature parameter for attention softmax
     * @param attentionDimensions Dimensionality for attention key/query vectors
     * @param usePositionalEncoding Whether to include positional encoding
     */
    public AttentionConsensus(double confidenceThreshold,
                             boolean normalizeAttentionWeights,
                             double temperatureScaling,
                             int attentionDimensions,
                             boolean usePositionalEncoding) {
        if (confidenceThreshold < 0.0 || confidenceThreshold > 1.0) {
            throw new IllegalArgumentException("Confidence threshold must be in [0.0, 1.0]: " + confidenceThreshold);
        }
        if (temperatureScaling <= 0.0) {
            throw new IllegalArgumentException("Temperature scaling must be positive: " + temperatureScaling);
        }
        if (attentionDimensions <= 0) {
            throw new IllegalArgumentException("Attention dimensions must be positive: " + attentionDimensions);
        }
        
        this.confidenceThreshold = confidenceThreshold;
        this.normalizeAttentionWeights = normalizeAttentionWeights;
        this.temperatureScaling = temperatureScaling;
        this.attentionDimensions = attentionDimensions;
        this.usePositionalEncoding = usePositionalEncoding;
    }
    
    @Override
    public ConsensusResult computeConsensus(Map<String, ChannelResult> channelResults,
                                          Map<String, Double> channelWeights) {
        Objects.requireNonNull(channelResults, "channelResults cannot be null");
        Objects.requireNonNull(channelWeights, "channelWeights cannot be null");
        
        // Filter successful results
        var validResults = channelResults.entrySet().stream()
            .filter(entry -> entry.getValue().isSuccess())
            .filter(entry -> entry.getValue().confidence() >= confidenceThreshold)
            .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
        
        if (validResults.isEmpty()) {
            log.debug("No valid channel results for attention consensus");
            return ConsensusResult.create(-1, 0.0, getStrategyName(),
                                        calculateContributions(channelResults, channelWeights));
        }
        
        if (validResults.size() == 1) {
            // Single channel - direct decision
            var entry = validResults.entrySet().iterator().next();
            var result = entry.getValue();
            var contributions = Map.of(entry.getKey(), 1.0);
            
            log.debug("Single channel attention consensus: {}", entry.getKey());
            return ConsensusResult.create(result.category(), result.confidence(), getStrategyName(),
                                        contributions);
        }
        
        // Compute attention weights
        var attentionWeights = computeAttentionWeights(validResults, channelWeights);
        
        // Apply attention to category voting
        var categoryVotes = new HashMap<Integer, Double>();
        var categoryChannels = new HashMap<Integer, Set<String>>();
        var contributions = new HashMap<String, Double>();
        var totalAttentionWeight = 0.0;
        
        for (var entry : validResults.entrySet()) {
            var channelId = entry.getKey();
            var result = entry.getValue();
            var attentionWeight = attentionWeights.get(channelId);
            
            // Combine attention weight with base channel weight and confidence
            var baseWeight = channelWeights.getOrDefault(channelId, 1.0);
            var effectiveWeight = attentionWeight * baseWeight * result.confidence();
            
            var category = result.category();
            categoryVotes.merge(category, effectiveWeight, Double::sum);
            categoryChannels.computeIfAbsent(category, k -> new HashSet<>()).add(channelId);
            
            contributions.put(channelId, effectiveWeight);
            totalAttentionWeight += effectiveWeight;
        }
        
        // Normalize contributions
        final var finalTotalAttentionWeight = totalAttentionWeight;
        if (finalTotalAttentionWeight > 0) {
            contributions.replaceAll((k, v) -> v / finalTotalAttentionWeight);
        }
        
        // Find winning category
        var winningEntry = categoryVotes.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .orElseThrow(() -> new IllegalStateException("No votes found"));
        
        var winningCategory = winningEntry.getKey();
        var winningWeight = winningEntry.getValue();
        var confidence = totalAttentionWeight > 0 ? winningWeight / totalAttentionWeight : 0.0;
        
        // Create comprehensive metadata
        var metadata = createAttentionMetadata(validResults, attentionWeights, categoryVotes, 
                                              categoryChannels.get(winningCategory));
        
        log.debug("Attention consensus: category={}, confidence={:.3f}, attended channels={}",
                 winningCategory, confidence, validResults.keySet());
        
        return ConsensusResult.create(winningCategory, confidence, getStrategyName(),
                                    contributions, metadata);
    }
    
    /**
     * Compute attention weights for channels using scaled dot-product attention.
     */
    private Map<String, Double> computeAttentionWeights(Map<String, ChannelResult> validResults,
                                                       Map<String, Double> channelWeights) {
        var channels = new ArrayList<>(validResults.keySet());
        var channelCount = channels.size();
        
        // Create feature representations for each channel (queries and keys)
        var channelFeatures = new HashMap<String, DenseVector>();
        for (var channelId : channels) {
            var result = validResults.get(channelId);
            var features = createChannelFeatures(channelId, result, channelWeights.get(channelId));
            channelFeatures.put(channelId, features);
        }
        
        // Compute attention matrix: attention[i][j] = attention from channel i to channel j
        var attentionMatrix = new double[channelCount][channelCount];
        
        for (var i = 0; i < channelCount; i++) {
            var queryChannel = channels.get(i);
            var queryFeatures = channelFeatures.get(queryChannel);
            
            for (var j = 0; j < channelCount; j++) {
                var keyChannel = channels.get(j);
                var keyFeatures = channelFeatures.get(keyChannel);
                
                // Scaled dot-product attention: Q·K / sqrt(d_k)
                var dotProduct = computeDotProduct(queryFeatures, keyFeatures);
                var scaledAttention = dotProduct / Math.sqrt(attentionDimensions);
                
                // Apply temperature scaling
                attentionMatrix[i][j] = scaledAttention / temperatureScaling;
            }
        }
        
        // Apply softmax to each row to get normalized attention weights
        var attentionWeights = new HashMap<String, Double>();
        
        for (var i = 0; i < channelCount; i++) {
            var queryChannel = channels.get(i);
            var weights = softmax(attentionMatrix[i]);
            
            // Aggregate attention weights (self-attention + cross-attention)
            var totalAttention = 0.0;
            for (var j = 0; j < channelCount; j++) {
                totalAttention += weights[j];
            }
            
            attentionWeights.put(queryChannel, totalAttention);
        }
        
        // Normalize attention weights if requested
        if (normalizeAttentionWeights) {
            var totalWeight = attentionWeights.values().stream().mapToDouble(Double::doubleValue).sum();
            if (totalWeight > 0) {
                attentionWeights.replaceAll((k, v) -> v / totalWeight);
            }
        }
        
        log.debug("Computed attention weights: {}", attentionWeights);
        return attentionWeights;
    }
    
    /**
     * Create feature vector representation for a channel result.
     */
    private DenseVector createChannelFeatures(String channelId, ChannelResult result, Double channelWeight) {
        var features = new double[attentionDimensions];
        
        // Base features
        features[0] = result.confidence();
        features[1] = result.category() >= 0 ? 1.0 : 0.0;  // Has category
        features[2] = Math.log(1.0 + result.processingTimeMs()) / 10.0;  // Normalized processing time
        features[3] = channelWeight != null ? channelWeight : 1.0;
        features[4] = result.metadata().size() / 10.0;  // Normalized metadata count
        
        // Channel type encoding (one-hot style)
        var channelTypeIndex = getChannelTypeIndex(channelId);
        if (channelTypeIndex >= 0 && channelTypeIndex + 5 < features.length) {
            features[5 + channelTypeIndex] = 1.0;
        }
        
        // Category encoding (distributed representation)
        if (result.category() >= 0) {
            var categoryBase = 10;
            for (var i = categoryBase; i < Math.min(categoryBase + 10, features.length); i++) {
                features[i] = Math.sin((result.category() + i - categoryBase) * Math.PI / 10.0);
            }
        }
        
        // Confidence-based features
        var confidenceBase = 20;
        for (var i = confidenceBase; i < Math.min(confidenceBase + 10, features.length); i++) {
            features[i] = result.confidence() * Math.cos((i - confidenceBase) * Math.PI / 10.0);
        }
        
        // Positional encoding if enabled
        if (usePositionalEncoding && features.length > 30) {
            var position = Math.abs(channelId.hashCode()) % 100;  // Pseudo-position
            for (var i = 30; i < Math.min(40, features.length); i++) {
                if (i % 2 == 0) {
                    features[i] = Math.sin(position / Math.pow(10000, 2.0 * i / attentionDimensions));
                } else {
                    features[i] = Math.cos(position / Math.pow(10000, 2.0 * i / attentionDimensions));
                }
            }
        }
        
        // Fill remaining dimensions with channel-specific patterns
        for (var i = 40; i < features.length; i++) {
            features[i] = result.confidence() * Math.tanh((i - 40.0) / features.length);
        }
        
        return new DenseVector(features);
    }
    
    /**
     * Get channel type index for one-hot encoding.
     */
    private int getChannelTypeIndex(String channelId) {
        return switch (channelId.toLowerCase()) {
            case "semantic", "fasttext" -> 0;
            case "entity", "ner" -> 1;
            case "syntactic", "pos" -> 2;
            case "lexical" -> 3;
            default -> 4; // Unknown type
        };
    }
    
    /**
     * Compute dot product between two feature vectors.
     */
    private double computeDotProduct(DenseVector a, DenseVector b) {
        if (a.dimension() != b.dimension()) {
            throw new IllegalArgumentException("Vector dimensions must match");
        }
        
        var sum = 0.0;
        for (var i = 0; i < a.dimension(); i++) {
            sum += a.get(i) * b.get(i);
        }
        
        return sum;
    }
    
    /**
     * Apply softmax function to array of logits.
     */
    private double[] softmax(double[] logits) {
        // Find max for numerical stability
        var max = Arrays.stream(logits).max().orElse(0.0);
        
        // Compute exponentials
        var exp = new double[logits.length];
        var sum = 0.0;
        
        for (var i = 0; i < logits.length; i++) {
            exp[i] = Math.exp(logits[i] - max);
            sum += exp[i];
        }
        
        // Normalize
        if (sum > 0) {
            for (var i = 0; i < exp.length; i++) {
                exp[i] /= sum;
            }
        }
        
        return exp;
    }
    
    /**
     * Calculate fallback contributions when no valid results.
     */
    private Map<String, Double> calculateContributions(Map<String, ChannelResult> channelResults,
                                                      Map<String, Double> channelWeights) {
        var contributions = new HashMap<String, Double>();
        
        for (var channelId : channelResults.keySet()) {
            var result = channelResults.get(channelId);
            if (result.isSuccess() && result.confidence() >= confidenceThreshold) {
                contributions.put(channelId, result.confidence());
            } else {
                contributions.put(channelId, 0.0);
            }
        }
        
        return contributions;
    }
    
    /**
     * Create comprehensive attention metadata.
     */
    private Map<String, Object> createAttentionMetadata(Map<String, ChannelResult> validResults,
                                                       Map<String, Double> attentionWeights,
                                                       Map<Integer, Double> categoryVotes,
                                                       Set<String> winningChannels) {
        var metadata = new HashMap<String, Object>();
        
        metadata.put("attentionWeights", new HashMap<>(attentionWeights));
        metadata.put("categoryVotes", new HashMap<>(categoryVotes));
        metadata.put("winningChannels", new HashSet<>(winningChannels));
        metadata.put("validChannels", validResults.keySet());
        metadata.put("temperatureScaling", temperatureScaling);
        metadata.put("attentionDimensions", attentionDimensions);
        metadata.put("normalizedWeights", normalizeAttentionWeights);
        metadata.put("positionalEncoding", usePositionalEncoding);
        
        // Attention statistics
        var maxAttention = attentionWeights.values().stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
        var minAttention = attentionWeights.values().stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
        var avgAttention = attentionWeights.values().stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        
        metadata.put("attentionStats", Map.of(
            "max", maxAttention,
            "min", minAttention,
            "average", avgAttention,
            "variance", calculateVariance(attentionWeights.values(), avgAttention)
        ));
        
        return metadata;
    }
    
    /**
     * Calculate variance of attention weights.
     */
    private double calculateVariance(Collection<Double> values, double mean) {
        return values.stream()
            .mapToDouble(x -> Math.pow(x - mean, 2))
            .average()
            .orElse(0.0);
    }
    
    @Override
    public String getStrategyName() {
        return "AttentionConsensus";
    }
    
    @Override
    public int getMinimumRequiredChannels() {
        return 1;
    }
    
    /**
     * Get confidence threshold.
     */
    public double getConfidenceThreshold() {
        return confidenceThreshold;
    }
    
    /**
     * Check if attention weights are normalized.
     */
    public boolean isNormalizeAttentionWeights() {
        return normalizeAttentionWeights;
    }
    
    /**
     * Get temperature scaling parameter.
     */
    public double getTemperatureScaling() {
        return temperatureScaling;
    }
    
    /**
     * Get attention dimensions.
     */
    public int getAttentionDimensions() {
        return attentionDimensions;
    }
    
    /**
     * Check if positional encoding is used.
     */
    public boolean isUsePositionalEncoding() {
        return usePositionalEncoding;
    }
    
    @Override
    public String toString() {
        return String.format("AttentionConsensus{threshold=%.2f, normalized=%s, temp=%.1f, dims=%d, pos=%s}",
                           confidenceThreshold, normalizeAttentionWeights, temperatureScaling,
                           attentionDimensions, usePositionalEncoding);
    }
}