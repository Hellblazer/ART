package com.hellblazer.art.nlp.processor.fusion;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.nlp.processor.ChannelResult;

import java.util.ArrayList;
import java.util.Map;
import java.util.Objects;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Feature fusion strategy that concatenates channel feature vectors.
 * Creates a single vector by placing channel features end-to-end.
 */
public class ConcatenationFusion implements FeatureFusionStrategy {
    private static final Logger log = LoggerFactory.getLogger(ConcatenationFusion.class);
    
    private final boolean normalizeFeatures;
    private final boolean includeChannelWeights;
    private final int maxDimensions;
    
    /**
     * Create concatenation fusion with default settings.
     */
    public ConcatenationFusion() {
        this(true, false, 1000);
    }
    
    /**
     * Create concatenation fusion with custom settings.
     * 
     * @param normalizeFeatures Whether to normalize individual channel features
     * @param includeChannelWeights Whether to include channel metadata
     * @param maxDimensions Maximum allowed output dimensions
     */
    public ConcatenationFusion(boolean normalizeFeatures, boolean includeChannelWeights, int maxDimensions) {
        if (maxDimensions <= 0) {
            throw new IllegalArgumentException("maxDimensions must be positive: " + maxDimensions);
        }
        
        this.normalizeFeatures = normalizeFeatures;
        this.includeChannelWeights = includeChannelWeights;
        this.maxDimensions = maxDimensions;
    }
    
    @Override
    public DenseVector fuseFeatures(Map<String, ChannelResult> channelResults) {
        Objects.requireNonNull(channelResults, "channelResults cannot be null");
        
        // Filter successful channel results
        var successfulResults = channelResults.entrySet().stream()
            .filter(entry -> entry.getValue().isSuccess())
            .toList();
        
        if (successfulResults.isEmpty()) {
            log.debug("No successful channel results for feature fusion");
            return null;
        }
        
        var fusedFeatures = new ArrayList<Double>();
        var channelCount = 0;
        
        // Process each channel in consistent order (sorted by channel ID)
        var sortedChannels = successfulResults.stream()
            .sorted(Map.Entry.comparingByKey())
            .toList();
        
        for (var entry : sortedChannels) {
            var channelId = entry.getKey();
            var result = entry.getValue();
            
            try {
                // Extract features from channel result
                var channelFeatures = extractChannelFeatures(channelId, result);
                
                if (channelFeatures != null && channelFeatures.length > 0) {
                    // Normalize if requested
                    if (normalizeFeatures) {
                        channelFeatures = normalizeVector(channelFeatures);
                    }
                    
                    // Add to fused vector
                    for (var feature : channelFeatures) {
                        fusedFeatures.add(feature);
                        
                        // Check dimension limit
                        if (fusedFeatures.size() >= maxDimensions) {
                            log.warn("Reached maximum dimensions limit: {}", maxDimensions);
                            break;
                        }
                    }
                    
                    // Include channel metadata if requested
                    if (includeChannelWeights && fusedFeatures.size() < maxDimensions) {
                        fusedFeatures.add(result.confidence());
                        fusedFeatures.add((double) result.category());
                    }
                    
                    channelCount++;
                }
                
                if (fusedFeatures.size() >= maxDimensions) {
                    break;
                }
                
            } catch (Exception e) {
                log.warn("Failed to extract features from channel '{}': {}", channelId, e.getMessage());
            }
        }
        
        if (fusedFeatures.isEmpty()) {
            log.debug("No features extracted for fusion");
            return null;
        }
        
        // Convert to array and create DenseVector
        var featuresArray = fusedFeatures.stream()
            .mapToDouble(Double::doubleValue)
            .toArray();
        
        log.debug("Fused features from {} channels: {} dimensions", channelCount, featuresArray.length);
        return new DenseVector(featuresArray);
    }
    
    /**
     * Extract feature vector from channel result.
     * Creates meaningful feature representations based on channel type and result data.
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
        // Create semantic feature representation based on ART category activation
        var features = new double[16]; // Larger dimension for semantic richness
        
        // Core semantic features
        features[0] = result.confidence(); // Raw confidence score
        features[1] = result.success() ? 1.0 : 0.0; // Success indicator
        features[2] = result.category() >= 0 ? 1.0 : 0.0; // Has valid category
        features[3] = Math.log(1.0 + result.processingTimeMs() / 1000.0); // Log processing time
        
        // Category embedding - distributed representation
        if (result.category() >= 0) {
            var categoryNorm = result.category() % (features.length - 4);
            for (int i = 4; i < features.length; i++) {
                var dist = Math.abs(i - 4 - categoryNorm);
                features[i] = result.confidence() * Math.exp(-0.5 * dist * dist); // Gaussian distribution
            }
        }
        
        // Extract metadata-based semantic indicators
        extractMetadataFeatures(result.metadata(), features, 8, 8);
        
        return features;
    }
    
    private double[] createEntityFeatures(ChannelResult result) {
        // Create entity-specific feature representation
        var features = new double[12]; // Entity features tend to be categorical
        
        // Core entity features
        features[0] = result.confidence(); // Entity confidence
        features[1] = result.success() ? 1.0 : 0.0; // Recognition success
        features[2] = result.category() >= 0 ? 1.0 : 0.0; // Has entity type
        features[3] = Math.min(result.processingTimeMs() / 100.0, 1.0); // Normalized processing time
        
        // Entity type encoding (one-hot style based on category)
        if (result.category() >= 0) {
            var entityType = result.category() % 4; // Common entity types: PERSON, ORG, LOC, MISC
            features[4 + entityType] = result.confidence();
        }
        
        // Extract entity-specific metadata (count, positions, etc.)
        extractMetadataFeatures(result.metadata(), features, 8, 4);
        
        return features;
    }
    
    private double[] createSyntacticFeatures(ChannelResult result) {
        // Create syntactic pattern feature representation
        var features = new double[14]; // Syntactic complexity requires more dimensions
        
        // Core syntactic features
        features[0] = result.confidence(); // Syntactic pattern confidence
        features[1] = result.success() ? 1.0 : 0.0; // Parse success
        features[2] = result.category() >= 0 ? 1.0 : 0.0; // Has syntactic category
        features[3] = Math.log(1.0 + result.processingTimeMs() / 10.0); // Log processing complexity
        
        // Syntactic pattern encoding - distributed across common patterns
        if (result.category() >= 0) {
            var patternType = result.category() % 6; // Common patterns: NP, VP, PP, ADJP, ADVP, S
            for (int i = 4; i < 10; i++) {
                var affinity = Math.exp(-Math.abs(i - 4 - patternType) / 2.0);
                features[i] = result.confidence() * affinity;
            }
        }
        
        // Extract syntactic metadata (sentence count, token count, etc.)
        extractMetadataFeatures(result.metadata(), features, 10, 4);
        
        return features;
    }
    
    private double[] createGenericFeatures(ChannelResult result) {
        // Generic feature representation for unknown channel types
        return new double[] {
            result.confidence(),
            result.category() >= 0 ? 1.0 : 0.0,
            Math.log(1.0 + result.processingTimeMs()),
            result.metadata().size()
        };
    }
    
    /**
     * Extract features from result metadata into the feature vector.
     * 
     * @param metadata The metadata map to extract features from
     * @param features The feature array to fill
     * @param startIndex Starting index in features array
     * @param count Number of metadata features to extract
     */
    private void extractMetadataFeatures(Map<String, Object> metadata, double[] features, int startIndex, int count) {
        if (metadata == null || startIndex >= features.length) {
            return;
        }
        
        var endIndex = Math.min(startIndex + count, features.length);
        var metadataKeys = metadata.keySet().toArray(new String[0]);
        
        for (int i = startIndex; i < endIndex; i++) {
            var keyIndex = (i - startIndex) % Math.max(metadataKeys.length, 1);
            
            if (keyIndex < metadataKeys.length) {
                var key = metadataKeys[keyIndex];
                var value = metadata.get(key);
                
                // Convert metadata value to double feature
                if (value instanceof Number number) {
                    features[i] = Math.tanh(number.doubleValue()); // Bounded to [-1,1]
                } else if (value instanceof Boolean bool) {
                    features[i] = bool ? 1.0 : 0.0;
                } else if (value instanceof String str) {
                    features[i] = Math.tanh(str.length() / 10.0); // String length feature
                } else {
                    features[i] = 0.1; // Default for unknown types
                }
            } else {
                features[i] = 0.0; // No more metadata available
            }
        }
    }
    
    /**
     * Normalize feature vector to unit length.
     */
    private double[] normalizeVector(double[] vector) {
        var norm = Math.sqrt(java.util.Arrays.stream(vector)
            .map(x -> x * x)
            .sum());
        
        if (norm > 1e-10) { // Avoid division by zero
            for (int i = 0; i < vector.length; i++) {
                vector[i] /= norm;
            }
        }
        
        return vector;
    }
    
    @Override
    public String getStrategyName() {
        return "Concatenation";
    }
    
    @Override
    public int getOutputDimension() {
        return -1; // Variable depending on channels
    }
    
    @Override
    public int getMinimumRequiredChannels() {
        return 1;
    }
    
    /**
     * Get maximum output dimensions.
     */
    public int getMaxDimensions() {
        return maxDimensions;
    }
    
    @Override
    public String toString() {
        return String.format("ConcatenationFusion{normalize=%s, includeWeights=%s, maxDims=%d}",
                           normalizeFeatures, includeChannelWeights, maxDimensions);
    }
}