package com.hellblazer.art.nlp.channels;

import com.hellblazer.art.nlp.config.ChannelConfig;
import com.hellblazer.art.nlp.core.Entity;
import com.hellblazer.art.core.algorithms.TopoART;
import com.hellblazer.art.core.parameters.TopoARTParameters;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.nlp.util.DistanceMetric;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Context processing channel using TopoART for contextual relationship learning.
 * Specializes in understanding semantic context, co-occurrence patterns, and
 * topological relationships between text features.
 */
public class ContextChannel extends com.hellblazer.art.nlp.channels.base.BaseChannel {
    
    private static final String CHANNEL_NAME = "context";
    private static final Logger logger = LoggerFactory.getLogger(ContextChannel.class);
    
    private final ChannelConfig config;
    private final TopoART topoART;
    private final DistanceMetric distanceMetric;
    private final Map<String, Integer> contextTypeMap;
    private final AtomicInteger categoryCounter;
    private final double neighbourhoodRadius;
    
    /**
     * Create ContextChannel with default configuration.
     * Uses vigilance=0.5, learning rate=0.8, neighbourhood radius=0.2
     */
    public ContextChannel() {
        this(ChannelConfig.builder()
            .channelName(CHANNEL_NAME)
            .vigilance(0.5)
            .learningRate(0.8)
            .build(), 0.2);
    }
    
    /**
     * Create ContextChannel with custom configuration.
     * 
     * @param config Channel configuration
     * @param neighbourhoodRadius Radius for topological neighbourhood in TopoART
     */
    public ContextChannel(ChannelConfig config, double neighbourhoodRadius) {
        super(config.getChannelName(), config.getVigilance());
        this.config = config;
        
        var dimensions = 100; // Default dimension for context vectors
        this.neighbourhoodRadius = neighbourhoodRadius;
        this.distanceMetric = DistanceMetric.EUCLIDEAN;
        this.contextTypeMap = new HashMap<>();
        this.categoryCounter = new AtomicInteger(0);
        
        // Initialize TopoART with specified parameters
        var topoParams = new TopoARTParameters(
            dimensions,                    // inputDimension
            config.getVigilance(),        // vigilanceA
            config.getLearningRate(),     // learningRateSecond  
            10,                          // phi (permanence threshold)
            100,                         // tau (cleanup cycle period)
            0.1                          // alpha (choice parameter)
        );
        this.topoART = new TopoART(topoParams);
        
        initializeContextTypes();
    }
    
    private void initializeContextTypes() {
        // Common context types for natural language processing
        contextTypeMap.put("SEQUENCE", 1);
        contextTypeMap.put("DEPENDENCY", 2);
        contextTypeMap.put("CO_OCCURRENCE", 3);
        contextTypeMap.put("SEMANTIC_FIELD", 4);
        contextTypeMap.put("DISCOURSE_MARKER", 5);
        contextTypeMap.put("TEMPORAL", 6);
        contextTypeMap.put("SPATIAL", 7);
        contextTypeMap.put("CAUSAL", 8);
        contextTypeMap.put("COMPARATIVE", 9);
        contextTypeMap.put("MODAL", 10);
    }
    
    protected double[] extractFeatures(String text) {
        if (text == null || text.trim().isEmpty()) {
            return new double[100]; // Default context dimension
        }
        
        var features = new double[100]; // Default context dimension
        var cleanText = text.toLowerCase().trim();
        var words = cleanText.split("\\s+");
        
        // Extract contextual features
        extractSequentialFeatures(words, features);
        extractCoOccurrenceFeatures(words, features);
        extractDiscourseFeatures(cleanText, features);
        extractModalFeatures(cleanText, features);
        extractTemporalFeatures(cleanText, features);
        extractSpatialFeatures(cleanText, features);
        extractCausalFeatures(cleanText, features);
        
        // Normalize features to [0,1] range for ART processing
        normalizeFeatures(features);
        
        return features;
    }
    
    private void extractSequentialFeatures(String[] words, double[] features) {
        var baseIdx = 0;
        var maxFeatures = Math.min(words.length - 1, features.length / 10);
        
        // Capture sequential patterns (bigrams and trigrams)
        for (int i = 0; i < maxFeatures && i < words.length - 1; i++) {
            var bigram = words[i] + "_" + words[i + 1];
            var hash = Math.abs(bigram.hashCode()) % (features.length / 10);
            features[baseIdx + hash] += 0.1;
            
            if (i < words.length - 2) {
                var trigram = words[i] + "_" + words[i + 1] + "_" + words[i + 2];
                hash = Math.abs(trigram.hashCode()) % (features.length / 10);
                features[baseIdx + hash] += 0.15;
            }
        }
    }
    
    private void extractCoOccurrenceFeatures(String[] words, double[] features) {
        var baseIdx = features.length / 10;
        var windowSize = 5; // Context window
        
        for (int i = 0; i < words.length; i++) {
            var centerWord = words[i];
            var start = Math.max(0, i - windowSize);
            var end = Math.min(words.length, i + windowSize + 1);
            
            for (int j = start; j < end; j++) {
                if (i != j) {
                    var coOccurrence = centerWord + "_WITH_" + words[j];
                    var hash = Math.abs(coOccurrence.hashCode()) % (features.length / 10);
                    var distance = Math.abs(i - j);
                    var weight = 1.0 / (distance + 1.0); // Closer words have higher weight
                    features[baseIdx + hash] += weight * 0.05;
                }
            }
        }
    }
    
    private void extractDiscourseFeatures(String text, double[] features) {
        var baseIdx = features.length / 5;
        var discourseMarkers = Arrays.asList(
            "however", "therefore", "furthermore", "moreover", "meanwhile",
            "consequently", "nevertheless", "although", "because", "since",
            "while", "whereas", "instead", "otherwise", "thus", "hence"
        );
        
        for (var marker : discourseMarkers) {
            if (text.contains(marker)) {
                var hash = Math.abs(marker.hashCode()) % (features.length / 10);
                features[baseIdx + hash] += 0.2;
            }
        }
    }
    
    private void extractModalFeatures(String text, double[] features) {
        var baseIdx = features.length * 3 / 10;
        var modalVerbs = Arrays.asList(
            "can", "could", "may", "might", "shall", "should",
            "will", "would", "must", "ought"
        );
        
        var words = text.split("\\s+");
        for (var word : words) {
            if (modalVerbs.contains(word)) {
                var hash = Math.abs(word.hashCode()) % (features.length / 20);
                features[baseIdx + hash] += 0.3;
            }
        }
    }
    
    private void extractTemporalFeatures(String text, double[] features) {
        var baseIdx = features.length * 2 / 5;
        
        // Temporal markers and sequences
        var temporalMarkers = Arrays.asList(
            "before", "after", "during", "while", "when", "then", "next", 
            "previously", "later", "simultaneously", "meanwhile", "until",
            "since", "by the time", "as soon as", "once", "whenever",
            "first", "second", "third", "finally", "last", "initially"
        );
        
        var words = text.split("\\s+");
        
        // Detect temporal sequence indicators
        for (var marker : temporalMarkers) {
            if (text.contains(marker)) {
                var hash = Math.abs(marker.hashCode()) % (features.length / 15);
                features[baseIdx + hash] += 0.25;
            }
        }
        
        // Detect temporal ordering patterns
        for (int i = 0; i < words.length - 1; i++) {
            var bigram = words[i] + " " + words[i + 1];
            
            // Past-to-present patterns
            if (bigram.matches(".*(was|were|had|did)\\s+(now|currently|today).*")) {
                features[baseIdx + (features.length / 20)] += 0.4;
            }
            
            // Sequential patterns (first...then, before...after)
            if (words[i].matches("(first|initially|originally)") && 
                i + 2 < words.length && words[i + 2].matches("(then|next|later|subsequently)")) {
                features[baseIdx + (features.length / 25)] += 0.5;
            }
        }
        
        // Time expressions (years, months, days, hours)
        var timePattern = "\\b(\\d{4}|\\d{1,2}:\\d{2}|january|february|march|april|may|june|july|august|september|october|november|december|monday|tuesday|wednesday|thursday|friday|saturday|sunday|yesterday|today|tomorrow|morning|afternoon|evening|night)\\b";
        var timeMatches = text.toLowerCase().split("\\s+");
        var timeCount = 0;
        for (var word : timeMatches) {
            if (word.matches(timePattern)) {
                timeCount++;
            }
        }
        if (timeCount > 0) {
            features[baseIdx + (features.length / 30)] += Math.min(timeCount * 0.15, 1.0);
        }
    }
    
    private void extractSpatialFeatures(String text, double[] features) {
        var baseIdx = features.length / 2;
        
        // Spatial markers and relationships
        var spatialMarkers = Arrays.asList(
            "above", "below", "beside", "near", "far", "inside", "outside",
            "within", "beyond", "across", "through", "around", "over", "under",
            "in front of", "behind", "left", "right", "north", "south", "east", "west",
            "here", "there", "everywhere", "nowhere", "somewhere", "anywhere"
        );
        
        // Detect spatial relationships
        for (var marker : spatialMarkers) {
            if (text.contains(marker)) {
                var hash = Math.abs(marker.hashCode()) % (features.length / 15);
                features[baseIdx + hash] += 0.2;
            }
        }
        
        // Geographic locations pattern
        var locationPattern = "\\b[A-Z][a-z]+\\s+(City|Street|Avenue|Road|Boulevard|Plaza|Square|Park|Lake|River|Mountain|Valley|Beach)\\b";
        if (text.matches(".*" + locationPattern + ".*")) {
            features[baseIdx + (features.length / 20)] += 0.3;
        }
    }
    
    private void extractCausalFeatures(String text, double[] features) {
        var baseIdx = features.length * 3 / 5;
        
        // Causal markers and relationships
        var causalMarkers = Arrays.asList(
            "because", "since", "due to", "owing to", "as a result", "consequently",
            "therefore", "thus", "hence", "so", "leads to", "causes", "results in",
            "brings about", "triggers", "produces", "generates", "creates",
            "if", "when", "unless", "provided that", "given that"
        );
        
        // Detect causal relationships
        for (var marker : causalMarkers) {
            if (text.contains(marker)) {
                var hash = Math.abs(marker.hashCode()) % (features.length / 15);
                features[baseIdx + hash] += 0.3;
            }
        }
        
        // Causal chain patterns (A causes B, B leads to C)
        var words = text.split("\\s+");
        for (int i = 0; i < words.length - 2; i++) {
            var trigram = words[i] + " " + words[i + 1] + " " + words[i + 2];
            
            // Direct causation patterns
            if (trigram.matches(".*(cause|lead|result)\\s+to.*") ||
                trigram.matches(".*(because|since|due)\\s+to.*")) {
                features[baseIdx + (features.length / 25)] += 0.4;
            }
        }
        
        // Conditional causation (if-then patterns)
        if (text.contains("if") && (text.contains("then") || text.contains("will"))) {
            features[baseIdx + (features.length / 30)] += 0.35;
        }
    }
    
    private void normalizeFeatures(double[] features) {
        var sum = Arrays.stream(features).sum();
        if (sum > 0) {
            for (int i = 0; i < features.length; i++) {
                features[i] /= sum;
            }
        }
    }
    
    protected Map<String, Integer> classifyFeatures(double[] features) {
        var categories = new HashMap<String, Integer>();
        
        if (features.length == 0) {
            return categories;
        }
        
        try {
            // Store count before learning
            var countBefore = topoART.getCategoryCount();
            
            // Learn/classify with TopoART
            topoART.learn(features);
            
            var countAfter = topoART.getCategoryCount();
            logger.debug("TopoART categories: before={}, after={}", countBefore, countAfter);
            
            // If a new category was created, use it; otherwise use the last existing category
            var categoryId = countAfter > 0 ? (countAfter - 1) : 0;
            
            if (countAfter > 0) {
                var contextType = determineContextType(features);
                var categoryName = CHANNEL_NAME + "_" + contextType + "_" + categoryId;
                categories.put(categoryName, categoryId);
                logger.debug("Created category: {}", categoryName);
                
                // Skip neighbourhood for now - getNeighbours method not available
                // var neighbours = topoART.getNeighbours(categoryId, neighbourhoodRadius);
            } else {
                logger.warn("TopoART failed to create any categories");
            }
        } catch (Exception e) {
            logger.warn("Context classification failed", e);
        }
        
        return categories;
    }
    
    private String determineContextType(double[] features) {
        var maxValue = 0.0;
        var dominantType = "GENERAL";
        var sectionSize = features.length / 10;
        
        // Analyze which section has the highest activation
        for (int section = 0; section < 10 && section * sectionSize < features.length; section++) {
            var start = section * sectionSize;
            var end = Math.min(start + sectionSize, features.length);
            var sectionSum = 0.0;
            
            for (int i = start; i < end; i++) {
                sectionSum += features[i];
            }
            
            if (sectionSum > maxValue) {
                maxValue = sectionSum;
                dominantType = getContextTypeForSection(section);
            }
        }
        
        return dominantType;
    }
    
    private String getContextTypeForSection(int section) {
        return switch (section) {
            case 0 -> "SEQUENCE";
            case 1 -> "CO_OCCURRENCE";
            case 2 -> "DISCOURSE_MARKER";
            case 3 -> "MODAL";
            case 4 -> "TEMPORAL";
            case 5 -> "SPATIAL";
            case 6 -> "CAUSAL";
            case 7 -> "COMPARATIVE";
            case 8 -> "SEMANTIC_FIELD";
            default -> "GENERAL";
        };
    }
    
    public List<Entity> extractEntities(String text) {
        logger.debug("ContextChannel: extractEntities called with text: {}", 
                    text != null ? text.substring(0, Math.min(50, text.length())) + "..." : "null");
        
        var entities = new ArrayList<Entity>();
        
        if (text == null || text.trim().isEmpty()) {
            logger.debug("ContextChannel: Empty text provided to extractEntities");
            return entities;
        }
        
        // Extract contextual entities (relationships, discourse elements)
        extractDiscourseEntities(text, entities);
        extractRelationshipEntities(text, entities);
        
        logger.debug("ContextChannel: extractEntities found {} entities: {}", 
                    entities.size(), entities.stream().map(e -> e.getType() + ":" + e.getText()).toList());
        
        return entities;
    }
    
    private void extractDiscourseEntities(String text, List<Entity> entities) {
        var discoursePatterns = Map.of(
            "CONTRAST", Arrays.asList("however", "but", "although", "whereas", "instead"),
            "CAUSE", Arrays.asList("because", "since", "due to", "as a result"),
            "ADDITION", Arrays.asList("furthermore", "moreover", "additionally", "also"),
            "CONCLUSION", Arrays.asList("therefore", "thus", "consequently", "hence")
        );
        
        for (var entry : discoursePatterns.entrySet()) {
            var type = entry.getKey();
            var patterns = entry.getValue();
            
            for (var pattern : patterns) {
                var index = text.toLowerCase().indexOf(pattern);
                if (index >= 0) {
                    entities.add(new Entity(
                        pattern,
                        "DISCOURSE_" + type,
                        index,
                        index + pattern.length(),
                        0.8
                    ));
                }
            }
        }
    }
    
    private void extractRelationshipEntities(String text, List<Entity> entities) {
        var relationshipPatterns = Arrays.asList(
            "related to", "associated with", "connected to", "linked with",
            "similar to", "different from", "compared to", "in contrast to"
        );
        
        for (var pattern : relationshipPatterns) {
            var index = text.toLowerCase().indexOf(pattern);
            if (index >= 0) {
                entities.add(new Entity(
                    pattern,
                    "RELATIONSHIP",
                    index,
                    index + pattern.length(),
                    0.7
                ));
            }
        }
    }
    
    public void reset() {
        // Note: BaseChannel doesn't have reset() method
        // topoART.reset(); // Method doesn't exist in TopoART
        categoryCounter.set(0);
        contextTypeMap.clear();
        initializeContextTypes();
        
        logger.info("ContextChannel reset completed");
    }
    
    @Override
    public int getCategoryCount() {
        return topoART.getCategoryCount();
    }
    
    /**
     * Get the neighbourhood radius used by TopoART.
     */
    public double getNeighbourhoodRadius() {
        return neighbourhoodRadius;
    }
    
    /**
     * Get topological neighbours of a specific category.
     * 
     * @param categoryId Category to find neighbours for
     * @return Set of neighbour category IDs
     */
    public Set<Integer> getNeighbours(int categoryId) {
        // TopoART doesn't expose getNeighbours method directly
        // Return empty set for now - in full implementation would access topology
        logger.debug("Getting neighbours for category {} (placeholder implementation)", categoryId);
        return new HashSet<>();
    }
    
    /**
     * Get the distance metric used by this channel.
     */
    public DistanceMetric getDistanceMetric() {
        return distanceMetric;
    }
    
    /**
     * Classify text and return category ID.
     * This is the main entry point for text processing in this channel.
     */
    public int classifyText(String text) {
        if (text == null || text.trim().isEmpty()) {
            logger.debug("ContextChannel: Empty text provided");
            return -1;
        }
        
        try {
            logger.debug("ContextChannel: Processing text: {}", text.substring(0, Math.min(50, text.length())));
            
            // Extract features from text
            var features = extractFeatures(text);
            logger.debug("ContextChannel: Extracted {} features", features.length);
            
            // Classify features and get categories
            var categories = classifyFeatures(features);
            logger.debug("ContextChannel: Found {} categories: {}", categories.size(), categories.keySet());
            
            // If we got categories, return the first one's ID
            if (!categories.isEmpty()) {
                var categoryId = categories.values().iterator().next();
                logger.debug("ContextChannel: Returning category ID: {}", categoryId);
                return categoryId;
            }
            
            // No categories found
            logger.debug("ContextChannel: No categories found");
            return -1;
            
        } catch (Exception e) {
            logger.error("Failed to classify text: {}", e.getMessage(), e);
            recordError();
            return -1;
        }
    }
    
    // ===== Abstract method implementations =====
    
    @Override
    public int classify(DenseVector input) {
        try {
            // Preprocess the input vector (normalization + complement coding)
            var preprocessedInput = preprocessInput(input);
            
            // Use TopoART to classify the input
            // Note: In a real implementation, we'd use the actual TopoART API
            // For now, return a placeholder category based on input properties
            var category = Math.abs(Arrays.hashCode(input.data())) % 10;
            categoryCounter.set(Math.max(categoryCounter.get(), category + 1));
            
            logger.debug("Classified input vector to context category: {}", category);
            return category;
            
        } catch (Exception e) {
            logger.error("Failed to classify input: {}", e.getMessage());
            recordError();
            return -1; // Error category
        }
    }
    
    @Override
    protected void performInitialization() {
        // Initialize context type mappings
        initializeContextTypes();
        logger.info("Context channel initialized with {} context types", contextTypeMap.size());
    }
    
    @Override
    protected void performCleanup() {
        // Clear context mappings to free memory
        contextTypeMap.clear();
        logger.info("Context channel cleaned up successfully");
    }
    
    
    @Override
    public int pruneCategories(double threshold) {
        // For now, return 0 as pruning would require access to TopoART internals
        logger.debug("Pruning categories with threshold {}, but not implemented yet", threshold);
        return 0;
    }
    
    @Override
    public void saveState() {
        try {
            // In a full implementation, this would save:
            // - TopoART weights and topology
            // - Context type mappings
            // - Channel metrics and statistics
            logger.debug("Saving context channel state (placeholder implementation)");
        } catch (Exception e) {
            logger.error("Failed to save context channel state: {}", e.getMessage());
            throw new RuntimeException("State save failed", e);
        }
    }
    
    @Override
    public void loadState() {
        try {
            // In a full implementation, this would load:
            // - Previously saved TopoART weights and topology
            // - Context type mappings
            // - Channel metrics and statistics
            logger.debug("Loading context channel state (placeholder implementation)");
        } catch (Exception e) {
            logger.warn("Failed to load context channel state: {}", e.getMessage());
            // Don't throw exception on load failure - start fresh instead
        }
    }
}