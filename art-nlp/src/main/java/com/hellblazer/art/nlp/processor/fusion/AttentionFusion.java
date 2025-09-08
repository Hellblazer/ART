package com.hellblazer.art.nlp.processor.fusion;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.nlp.processor.ChannelResult;

import java.util.*;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Attention-based feature fusion strategy that dynamically weights channel features
 * based on their relevance and quality using multi-head attention mechanisms.
 */
public class AttentionFusion implements FeatureFusionStrategy {
    private static final Logger log = LoggerFactory.getLogger(AttentionFusion.class);
    
    private final int attentionDimensions;
    private final int numAttentionHeads;
    private final double temperatureScaling;
    private final boolean usePositionalEncoding;
    private final boolean normalizeOutputs;
    private final int maxFeatureDimensions;
    
    /**
     * Create attention fusion with default settings.
     */
    public AttentionFusion() {
        this(64, 4, 1.0, false, true, 256);
    }
    
    /**
     * Create attention fusion with custom parameters.
     * 
     * @param attentionDimensions Dimensionality of attention mechanism
     * @param numAttentionHeads Number of attention heads
     * @param temperatureScaling Temperature parameter for attention softmax
     * @param usePositionalEncoding Whether to use positional encoding
     * @param normalizeOutputs Whether to normalize final outputs
     * @param maxFeatureDimensions Maximum output feature dimensions
     */
    public AttentionFusion(int attentionDimensions, int numAttentionHeads, 
                          double temperatureScaling, boolean usePositionalEncoding,
                          boolean normalizeOutputs, int maxFeatureDimensions) {
        if (attentionDimensions <= 0) {
            throw new IllegalArgumentException("Attention dimensions must be positive: " + attentionDimensions);
        }
        if (numAttentionHeads <= 0) {
            throw new IllegalArgumentException("Number of attention heads must be positive: " + numAttentionHeads);
        }
        if (temperatureScaling <= 0.0) {
            throw new IllegalArgumentException("Temperature scaling must be positive: " + temperatureScaling);
        }
        if (maxFeatureDimensions <= 0) {
            throw new IllegalArgumentException("Max feature dimensions must be positive: " + maxFeatureDimensions);
        }
        
        this.attentionDimensions = attentionDimensions;
        this.numAttentionHeads = numAttentionHeads;
        this.temperatureScaling = temperatureScaling;
        this.usePositionalEncoding = usePositionalEncoding;
        this.normalizeOutputs = normalizeOutputs;
        this.maxFeatureDimensions = maxFeatureDimensions;
    }
    
    @Override
    public DenseVector fuseFeatures(Map<String, ChannelResult> channelResults) {
        Objects.requireNonNull(channelResults, "channelResults cannot be null");
        
        // Filter successful channel results
        var successfulResults = channelResults.entrySet().stream()
            .filter(entry -> entry.getValue().isSuccess())
            .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
        
        if (successfulResults.isEmpty()) {
            log.debug("No successful channel results for attention fusion");
            return null;
        }
        
        if (successfulResults.size() == 1) {
            // Single channel - return features directly
            var entry = successfulResults.entrySet().iterator().next();
            var features = extractChannelFeatures(entry.getKey(), entry.getValue());
            
            if (features != null && features.length > 0) {
                log.debug("Single channel attention fusion: {}", entry.getKey());
                return new DenseVector(Arrays.copyOf(features, Math.min(features.length, maxFeatureDimensions)));
            }
            return null;
        }
        
        // Extract and prepare channel features
        var channelFeatures = extractAllChannelFeatures(successfulResults);
        
        if (channelFeatures.isEmpty()) {
            log.debug("No features extracted for attention fusion");
            return null;
        }
        
        // Perform multi-head attention fusion
        var fusedFeatures = performMultiHeadAttention(channelFeatures, successfulResults);
        
        if (fusedFeatures == null || fusedFeatures.length == 0) {
            log.debug("Attention fusion produced no output");
            return null;
        }
        
        // Apply final normalization if requested
        if (normalizeOutputs) {
            fusedFeatures = normalizeVector(fusedFeatures);
        }
        
        log.debug("Attention fusion: {} channels → {} dimensions using {} heads",
                 channelFeatures.size(), fusedFeatures.length, numAttentionHeads);
        
        return new DenseVector(fusedFeatures);
    }
    
    /**
     * Extract features from all successful channels.
     */
    private Map<String, double[]> extractAllChannelFeatures(Map<String, ChannelResult> successfulResults) {
        var channelFeatures = new LinkedHashMap<String, double[]>();
        
        // Process channels in sorted order for consistency
        var sortedChannelIds = successfulResults.keySet().stream().sorted().toList();
        
        for (var channelId : sortedChannelIds) {
            var result = successfulResults.get(channelId);
            var features = extractChannelFeatures(channelId, result);
            
            if (features != null && features.length > 0) {
                // Add positional encoding if enabled
                if (usePositionalEncoding) {
                    features = addPositionalEncoding(features, channelFeatures.size());
                }
                
                channelFeatures.put(channelId, features);
            }
        }
        
        return channelFeatures;
    }
    
    /**
     * Perform multi-head attention fusion.
     */
    private double[] performMultiHeadAttention(Map<String, double[]> channelFeatures, 
                                             Map<String, ChannelResult> channelResults) {
        var channelIds = new ArrayList<>(channelFeatures.keySet());
        var numChannels = channelIds.size();
        
        if (numChannels == 0) {
            return null;
        }
        
        // Determine feature dimensions (use maximum across channels)
        var maxFeatureLength = channelFeatures.values().stream()
            .mapToInt(f -> f.length)
            .max().orElse(0);
        
        // Pad all feature vectors to same length
        var normalizedFeatures = new HashMap<String, double[]>();
        for (var entry : channelFeatures.entrySet()) {
            var features = entry.getValue();
            var padded = Arrays.copyOf(features, maxFeatureLength);
            normalizedFeatures.put(entry.getKey(), padded);
        }
        
        // Multi-head attention computation
        var headOutputs = new ArrayList<double[]>();
        
        for (var head = 0; head < numAttentionHeads; head++) {
            var headOutput = computeSingleHeadAttention(normalizedFeatures, channelResults, head);
            if (headOutput != null) {
                headOutputs.add(headOutput);
            }
        }
        
        if (headOutputs.isEmpty()) {
            log.warn("No attention heads produced output");
            return createFallbackOutput(normalizedFeatures);
        }
        
        // Combine multi-head outputs
        return combineAttentionHeads(headOutputs);
    }
    
    /**
     * Compute attention for a single head.
     */
    private double[] computeSingleHeadAttention(Map<String, double[]> channelFeatures,
                                              Map<String, ChannelResult> channelResults,
                                              int headIndex) {
        var channelIds = new ArrayList<>(channelFeatures.keySet());
        var numChannels = channelIds.size();
        
        if (numChannels == 0) {
            return null;
        }
        
        // Create queries, keys, and values for this head
        var headDim = attentionDimensions / numAttentionHeads;
        var queries = new double[numChannels][headDim];
        var keys = new double[numChannels][headDim];
        var values = new double[numChannels][];
        
        // Project features to Q, K, V spaces
        for (var i = 0; i < numChannels; i++) {
            var channelId = channelIds.get(i);
            var features = channelFeatures.get(channelId);
            var result = channelResults.get(channelId);
            
            // Create query and key vectors (head-specific projections)
            projectToQK(features, result, queries[i], keys[i], headIndex);
            
            // Values are the original features (possibly projected)
            values[i] = projectToValue(features, result, headIndex);
        }
        
        // Compute attention weights
        var attentionWeights = computeAttentionWeights(queries, keys);
        
        // Apply attention to values
        return applyAttention(attentionWeights, values);
    }
    
    /**
     * Project features to query and key vectors.
     */
    private void projectToQK(double[] features, ChannelResult result, 
                           double[] query, double[] key, int headIndex) {
        var headDim = query.length;
        
        // Simple linear projection with head-specific weights
        for (var i = 0; i < headDim; i++) {
            var featureIdx = (i + headIndex * headDim) % features.length;
            var weight = Math.sin((headIndex + 1) * Math.PI / numAttentionHeads + i);
            
            query[i] = features[featureIdx] * weight + result.confidence() * 0.1;
            key[i] = features[featureIdx] * weight * result.confidence();
        }
    }
    
    /**
     * Project features to value vector.
     */
    private double[] projectToValue(double[] features, ChannelResult result, int headIndex) {
        var valueLength = Math.min(features.length, maxFeatureDimensions / numAttentionHeads);
        var values = new double[valueLength];
        
        // Value projection with confidence weighting
        for (var i = 0; i < valueLength; i++) {
            values[i] = features[i] * result.confidence();
        }
        
        return values;
    }
    
    /**
     * Compute attention weights using scaled dot-product attention.
     */
    private double[][] computeAttentionWeights(double[][] queries, double[][] keys) {
        var numChannels = queries.length;
        var attentionMatrix = new double[numChannels][numChannels];
        
        // Compute attention scores: Q * K^T / sqrt(d_k)
        for (var i = 0; i < numChannels; i++) {
            for (var j = 0; j < numChannels; j++) {
                var score = dotProduct(queries[i], keys[j]);
                attentionMatrix[i][j] = score / Math.sqrt(queries[i].length) / temperatureScaling;
            }
        }
        
        // Apply softmax to each row
        for (var i = 0; i < numChannels; i++) {
            attentionMatrix[i] = softmax(attentionMatrix[i]);
        }
        
        return attentionMatrix;
    }
    
    /**
     * Apply attention weights to values.
     */
    private double[] applyAttention(double[][] attentionWeights, double[][] values) {
        var numChannels = attentionWeights.length;
        
        if (numChannels == 0 || values.length == 0) {
            return new double[0];
        }
        
        // Determine output dimension
        var outputDim = Arrays.stream(values).mapToInt(v -> v.length).max().orElse(0);
        var output = new double[outputDim];
        
        // Weighted sum of values
        for (var i = 0; i < numChannels; i++) {
            for (var j = 0; j < numChannels; j++) {
                var weight = attentionWeights[i][j];
                var value = values[j];
                
                for (var k = 0; k < Math.min(outputDim, value.length); k++) {
                    output[k] += weight * value[k] / numChannels;
                }
            }
        }
        
        return output;
    }
    
    /**
     * Combine outputs from multiple attention heads.
     */
    private double[] combineAttentionHeads(List<double[]> headOutputs) {
        if (headOutputs.isEmpty()) {
            return new double[0];
        }
        
        // Determine output dimension
        var maxDim = headOutputs.stream().mapToInt(h -> h.length).max().orElse(0);
        var combinedDim = Math.min(maxDim * numAttentionHeads, maxFeatureDimensions);
        var combined = new double[combinedDim];
        
        // Concatenate head outputs
        var offset = 0;
        for (var headOutput : headOutputs) {
            var copyLength = Math.min(headOutput.length, combinedDim - offset);
            System.arraycopy(headOutput, 0, combined, offset, copyLength);
            offset += copyLength;
            
            if (offset >= combinedDim) {
                break;
            }
        }
        
        return combined;
    }
    
    /**
     * Extract feature vector from channel result.
     */
    private double[] extractChannelFeatures(String channelId, ChannelResult result) {
        return switch (channelId.toLowerCase()) {
            case "fasttext", "semantic" -> createSemanticFeatures(result);
            case "entity", "ner" -> createEntityFeatures(result);
            case "syntactic", "pos" -> createSyntacticFeatures(result);
            default -> createGenericFeatures(result);
        };
    }
    
    private double[] createSemanticFeatures(ChannelResult result) {
        var features = new double[32]; // Rich feature set for attention
        
        // Base features
        features[0] = result.confidence();
        features[1] = result.category() >= 0 ? 1.0 : 0.0;
        features[2] = Math.log(1.0 + result.processingTimeMs()) / 10.0;
        features[3] = result.metadata().size() / 10.0;
        
        // Semantic patterns
        for (var i = 4; i < features.length; i++) {
            features[i] = result.confidence() * Math.sin(i * Math.PI / features.length) +
                         0.1 * Math.cos(2 * i * Math.PI / features.length);
        }
        
        return features;
    }
    
    private double[] createEntityFeatures(ChannelResult result) {
        var features = new double[24];
        
        features[0] = result.confidence();
        features[1] = result.category() >= 0 ? 1.0 : 0.0;
        features[2] = Math.log(1.0 + result.processingTimeMs()) / 10.0;
        
        // Entity-specific patterns
        for (var i = 3; i < features.length; i++) {
            features[i] = result.confidence() * Math.cos(i * Math.PI / features.length) +
                         0.05 * Math.sin(3 * i * Math.PI / features.length);
        }
        
        return features;
    }
    
    private double[] createSyntacticFeatures(ChannelResult result) {
        var features = new double[28];
        
        features[0] = result.confidence();
        features[1] = result.category() >= 0 ? 1.0 : 0.0;
        features[2] = Math.log(1.0 + result.processingTimeMs()) / 10.0;
        
        // Syntactic patterns
        for (var i = 3; i < features.length; i++) {
            features[i] = result.confidence() * Math.tanh((i - features.length / 2.0) / 5.0) +
                         0.1 * Math.exp(-Math.abs(i - features.length / 2.0) / 10.0);
        }
        
        return features;
    }
    
    private double[] createGenericFeatures(ChannelResult result) {
        return new double[] {
            result.confidence(),
            result.category() >= 0 ? 1.0 : 0.0,
            Math.log(1.0 + result.processingTimeMs()),
            result.metadata().size() / 5.0,
            result.confidence() * 0.5,
            Math.sqrt(result.confidence())
        };
    }
    
    /**
     * Add positional encoding to features.
     */
    private double[] addPositionalEncoding(double[] features, int position) {
        var encoded = Arrays.copyOf(features, features.length);
        
        // Add sinusoidal positional encoding
        for (var i = 0; i < encoded.length; i++) {
            if (i % 2 == 0) {
                encoded[i] += 0.1 * Math.sin(position / Math.pow(10000, 2.0 * i / encoded.length));
            } else {
                encoded[i] += 0.1 * Math.cos(position / Math.pow(10000, 2.0 * i / encoded.length));
            }
        }
        
        return encoded;
    }
    
    /**
     * Compute dot product of two vectors.
     */
    private double dotProduct(double[] a, double[] b) {
        var sum = 0.0;
        var minLength = Math.min(a.length, b.length);
        
        for (var i = 0; i < minLength; i++) {
            sum += a[i] * b[i];
        }
        
        return sum;
    }
    
    /**
     * Apply softmax to array.
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
     * Normalize vector to unit length.
     */
    private double[] normalizeVector(double[] vector) {
        var norm = Math.sqrt(Arrays.stream(vector).map(x -> x * x).sum());
        
        if (norm > 1e-10) {
            for (var i = 0; i < vector.length; i++) {
                vector[i] /= norm;
            }
        }
        
        return vector;
    }
    
    /**
     * Create fallback output when attention fails.
     */
    private double[] createFallbackOutput(Map<String, double[]> channelFeatures) {
        var fallbackLength = Math.min(maxFeatureDimensions, 
                                    channelFeatures.values().stream()
                                                  .mapToInt(f -> f.length)
                                                  .max().orElse(10));
        var fallback = new double[fallbackLength];
        
        // Simple average of all channel features
        var numChannels = channelFeatures.size();
        for (var features : channelFeatures.values()) {
            for (var i = 0; i < Math.min(fallback.length, features.length); i++) {
                fallback[i] += features[i] / numChannels;
            }
        }
        
        return fallback;
    }
    
    @Override
    public String getStrategyName() {
        return "AttentionFusion";
    }
    
    @Override
    public int getOutputDimension() {
        return maxFeatureDimensions;
    }
    
    @Override
    public int getMinimumRequiredChannels() {
        return 1;
    }
    
    /**
     * Get attention dimensions.
     */
    public int getAttentionDimensions() {
        return attentionDimensions;
    }
    
    /**
     * Get number of attention heads.
     */
    public int getNumAttentionHeads() {
        return numAttentionHeads;
    }
    
    /**
     * Get temperature scaling.
     */
    public double getTemperatureScaling() {
        return temperatureScaling;
    }
    
    /**
     * Check if positional encoding is used.
     */
    public boolean isUsePositionalEncoding() {
        return usePositionalEncoding;
    }
    
    /**
     * Check if outputs are normalized.
     */
    public boolean isNormalizeOutputs() {
        return normalizeOutputs;
    }
    
    /**
     * Get maximum feature dimensions.
     */
    public int getMaxFeatureDimensions() {
        return maxFeatureDimensions;
    }
    
    @Override
    public String toString() {
        return String.format("AttentionFusion{attentionDims=%d, heads=%d, temp=%.1f, pos=%s, norm=%s, maxDims=%d}",
                           attentionDimensions, numAttentionHeads, temperatureScaling,
                           usePositionalEncoding, normalizeOutputs, maxFeatureDimensions);
    }
}