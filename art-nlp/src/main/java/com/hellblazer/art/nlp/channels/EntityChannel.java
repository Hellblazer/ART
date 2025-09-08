package com.hellblazer.art.nlp.channels;

import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.namefind.TokenNameFinderModel;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.util.Span;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.artmap.FuzzyARTMAP;
import com.hellblazer.art.core.parameters.FuzzyARTMAPParameters;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.nlp.channels.base.BaseChannel;
import com.hellblazer.art.nlp.core.Entity;

/**
 * Entity channel using OpenNLP for Named Entity Recognition (NER).
 * Processes text to identify and classify entities into feature vectors.
 */
public final class EntityChannel extends BaseChannel {
    private static final Logger log = LoggerFactory.getLogger(EntityChannel.class);

    // OpenNLP models and tools
    private TokenizerME tokenizer;
    private SentenceDetectorME sentenceDetector;
    private final Map<EntityType, NameFinderME> entityFinders = new EnumMap<>(EntityType.class);
    
    // Model paths
    private final Path tokenizerModelPath;
    private final Path sentenceModelPath;
    private final Map<EntityType, Path> nerModelPaths;
    
    // Configuration
    private final Set<EntityType> enabledEntityTypes;
    private final EntityFeatureMode featureMode;
    private final boolean useNormalization;
    private final int maxEntitiesPerText;
    
    // ARTMAP algorithm for entity classification
    private final FuzzyARTMAP fuzzyARTMAP;
    private final FuzzyARTMAPParameters artmapParameters;
    
    // Entity type encoding
    private final Map<String, Integer> entityTypeToIndex = new ConcurrentHashMap<>();
    private final AtomicInteger nextEntityIndex = new AtomicInteger(0);
    
    // Performance metrics
    private final AtomicInteger totalSentences = new AtomicInteger(0);
    private final AtomicInteger totalEntities = new AtomicInteger(0);
    private final AtomicInteger successfulClassifications = new AtomicInteger(0);

    /**
     * Supported entity types for NER.
     */
    public enum EntityType {
        PERSON("en-ner-person.bin"),
        LOCATION("en-ner-location.bin"),
        ORGANIZATION("en-ner-organization.bin");
        
        private final String modelFileName;
        
        EntityType(String modelFileName) {
            this.modelFileName = modelFileName;
        }
        
        public String getModelFileName() {
            return modelFileName;
        }
    }

    /**
     * Feature extraction modes for entities.
     */
    public enum EntityFeatureMode {
        /** Count entities by type */
        COUNT_BASED,
        /** Entity density and distribution */
        DENSITY_BASED,
        /** Full entity feature extraction */
        COMPREHENSIVE
    }

    /**
     * Create entity channel with default configuration.
     */
    public EntityChannel(String channelName, double vigilance) {
        this(channelName, vigilance,
             EnumSet.allOf(EntityType.class),
             EntityFeatureMode.COUNT_BASED,
             true, 50);
    }

    /**
     * Create entity channel with custom configuration.
     */
    public EntityChannel(String channelName, double vigilance,
                        Set<EntityType> enabledEntityTypes,
                        EntityFeatureMode featureMode,
                        boolean useNormalization,
                        int maxEntitiesPerText) {
        super(channelName, vigilance);
        
        this.enabledEntityTypes = EnumSet.copyOf(enabledEntityTypes);
        this.featureMode = Objects.requireNonNull(featureMode, "featureMode cannot be null");
        this.useNormalization = useNormalization;
        this.maxEntitiesPerText = maxEntitiesPerText;
        
        // Set up model paths
        this.tokenizerModelPath = getDefaultModelPath("en-token.bin");
        this.sentenceModelPath = getDefaultModelPath("en-sent.bin");
        this.nerModelPaths = new EnumMap<>(EntityType.class);
        
        for (var entityType : this.enabledEntityTypes) {
            this.nerModelPaths.put(entityType, getDefaultModelPath(entityType.getModelFileName()));
        }
        
        // Initialize FuzzyARTMAP algorithm for entity classification
        this.fuzzyARTMAP = new FuzzyARTMAP();
        this.artmapParameters = new FuzzyARTMAPParameters(vigilance, 0.001, 0.7, 0.1); // vigilance, choice, learning rate, epsilon
        
        log.info("Entity channel '{}' created: types={}, mode={}, normalize={}, maxEntities={}, vigilance={}", 
                channelName, enabledEntityTypes, featureMode, useNormalization, maxEntitiesPerText, vigilance);
    }

    @Override
    protected void performInitialization() {
        try {
            log.debug("EntityChannel initialization starting - tokenizerModelPath: {}, sentenceModelPath: {}", 
                     tokenizerModelPath, sentenceModelPath);
            
            // Load tokenizer
            try (var tokenizerStream = getModelStream(tokenizerModelPath)) {
                log.debug("Loading tokenizer model...");
                var tokenizerModel = new TokenizerModel(tokenizerStream);
                this.tokenizer = new TokenizerME(tokenizerModel);
                log.debug("Tokenizer loaded successfully");
            }
            
            // Load sentence detector
            try (var sentStream = getModelStream(sentenceModelPath)) {
                log.debug("Loading sentence detector model...");
                var sentModel = new SentenceModel(sentStream);
                this.sentenceDetector = new SentenceDetectorME(sentModel);
                log.debug("Sentence detector loaded successfully");
            }
            
            // Load NER models for enabled entity types
            for (var entityType : enabledEntityTypes) {
                var modelPath = nerModelPaths.get(entityType);
                try (var nerStream = getModelStream(modelPath)) {
                    var nerModel = new TokenNameFinderModel(nerStream);
                    var nameFinder = new NameFinderME(nerModel);
                    entityFinders.put(entityType, nameFinder);
                    log.debug("Loaded NER model for {}", entityType);
                }
            }
            
            // Initialize entity type indices
            initializeEntityTypes();
            
            log.info("OpenNLP NER models initialized for entity channel '{}'", getChannelName());
            
        } catch (IOException e) {
            log.error("Failed to initialize OpenNLP NER models for channel '{}'", getChannelName(), e);
            throw new RuntimeException("OpenNLP NER initialization failed", e);
        }
    }

    @Override
    protected void performCleanup() {
        // Clear NER models
        for (var nameFinder : entityFinders.values()) {
            nameFinder.clearAdaptiveData();
        }
        entityFinders.clear();
        
        tokenizer = null;
        sentenceDetector = null;
        entityTypeToIndex.clear();
        log.debug("Entity channel '{}' cleanup complete", getChannelName());
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
            
            // Use FuzzyARTMAP for supervised entity classification
            // Convert to single-element array for prediction
            var inputPattern = Pattern.of(processedInput.data());
            var predictResult = fuzzyARTMAP.predict(new Pattern[]{inputPattern});
            
            if (predictResult.length > 0 && predictResult[0] >= 0) {
                var processingTime = System.currentTimeMillis() - startTime;
                recordClassification(processingTime, false); // Existing category
                successfulClassifications.incrementAndGet();
                return predictResult[0];
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
     * Process text input into entity vector for classification.
     * 
     * @param text Input text to process
     * @return Entity category ID
     */
    public int classifyText(String text) {
        if (text == null || text.isBlank()) {
            return -1; // Invalid input
        }

        try {
            // Extract entities from text to get both features AND labels for supervised learning
            var entities = extractEntities(text);
            
            if (entities.isEmpty()) {
                log.debug("No entities found in text: {}", text);
                return -1;
            }
            
            // Generate entity features
            var features = generateEntityFeatures(entities, text);
            if (features == null || features.length == 0) {
                return -1;
            }
            
            // Determine dominant entity type for supervised learning label
            var dominantEntityType = getDominantEntityType(entities);
            var entityLabel = getEntityTypeIndex(dominantEntityType.name());
            
            // Create input pattern for FuzzyARTMAP
            var inputPattern = Pattern.of(features);
            
            // Perform supervised learning with FuzzyARTMAP
            var trainResult = fuzzyARTMAP.trainSingle(inputPattern, entityLabel);
            
            var category = trainResult.category();
            
            if (category >= 0) {
                successfulClassifications.incrementAndGet();
            }
            
            return category;
            
        } catch (Exception e) {
            log.debug("Error processing entity features for text: {}", text, e);
            return -1;
        }
    }

    /**
     * Extract entities from text using NER models.
     */
    public List<Entity> extractEntities(String text) {
        if (text == null || text.isBlank()) {
            return Collections.emptyList();
        }
        
        // Check if components are initialized
        if (sentenceDetector == null || tokenizer == null) {
            log.error("EntityChannel not properly initialized - sentenceDetector: {}, tokenizer: {}", 
                     sentenceDetector != null, tokenizer != null);
            return Collections.emptyList();
        }

        var allEntities = new ArrayList<Entity>();
        
        try {
            // Detect sentences
            var sentences = sentenceDetector.sentDetect(text);
            totalSentences.addAndGet(sentences.length);
            
            for (var sentence : sentences) {
                // Tokenize sentence
                var tokens = tokenizer.tokenize(sentence);
                if (tokens.length == 0) continue;
                
                // Extract entities for each enabled type
                for (var entityType : enabledEntityTypes) {
                    var nameFinder = entityFinders.get(entityType);
                    if (nameFinder == null) continue;
                    
                    var nameSpans = nameFinder.find(tokens);
                    
                    for (var span : nameSpans) {
                        var entityText = String.join(" ", 
                            Arrays.copyOfRange(tokens, span.getStart(), span.getEnd()));
                        var confidence = span.getProb();
                        
                        var entity = new Entity(
                            entityText,
                            entityType.name(),
                            span.getStart(),
                            span.getEnd() - 1,
                            confidence
                        );
                        
                        allEntities.add(entity);
                    }
                    
                    // Clear adaptive data to avoid interference between sentences
                    nameFinder.clearAdaptiveData();
                }
            }
            
            // Limit total entities
            if (allEntities.size() > maxEntitiesPerText) {
                // Sort by confidence and keep top entities
                allEntities.sort((e1, e2) -> Double.compare(e2.getConfidence(), e1.getConfidence()));
                allEntities = new ArrayList<>(allEntities.subList(0, maxEntitiesPerText));
            }
            
            totalEntities.addAndGet(allEntities.size());
            
        } catch (Exception e) {
            log.debug("Error extracting entities from text: {}", text, e);
            return Collections.emptyList();
        }
        
        return allEntities;
    }

    /**
     * Generate feature vector from extracted entities.
     */
    private double[] generateEntityFeatures(List<Entity> entities, String text) {
        return switch (featureMode) {
            case COUNT_BASED -> generateCountFeatures(entities);
            case DENSITY_BASED -> generateDensityFeatures(entities, text);
            case COMPREHENSIVE -> generateComprehensiveFeatures(entities, text);
        };
    }
    
    /**
     * Generate count-based features (entity counts by type).
     */
    private double[] generateCountFeatures(List<Entity> entities) {
        var features = new double[enabledEntityTypes.size()];
        var typeToFeatureIndex = new HashMap<EntityType, Integer>();
        
        var index = 0;
        for (var entityType : enabledEntityTypes) {
            typeToFeatureIndex.put(entityType, index++);
        }
        
        // Count entities by type
        for (var entity : entities) {
            try {
                var entityType = EntityType.valueOf(entity.getType());
                var featureIndex = typeToFeatureIndex.get(entityType);
                if (featureIndex != null) {
                    features[featureIndex] += 1.0;
                }
            } catch (IllegalArgumentException e) {
                // Unknown entity type, skip
            }
        }
        
        // Normalize if requested
        if (useNormalization && entities.size() > 0) {
            var totalEntities = entities.size();
            for (int i = 0; i < features.length; i++) {
                features[i] /= totalEntities;
            }
        }
        
        return features;
    }
    
    /**
     * Generate density-based features (entity density and distribution).
     */
    private double[] generateDensityFeatures(List<Entity> entities, String text) {
        var countFeatures = generateCountFeatures(entities);
        var features = new ArrayList<Double>();
        
        // Add count features
        for (var feature : countFeatures) {
            features.add(feature);
        }
        
        // Add density features
        var textLength = text.length();
        var totalEntityLength = entities.stream()
            .mapToInt(e -> e.getText().length())
            .sum();
        
        // Entity density (entity characters / total characters)
        var density = textLength > 0 ? (double) totalEntityLength / textLength : 0.0;
        features.add(density);
        
        // Average entity confidence
        var avgConfidence = entities.stream()
            .mapToDouble(Entity::getConfidence)
            .average()
            .orElse(0.0);
        features.add(avgConfidence);
        
        // Entity diversity (unique types / total types)
        var uniqueTypes = entities.stream()
            .map(Entity::getType)
            .collect(HashSet::new, Set::add, Set::addAll)
            .size();
        var diversity = enabledEntityTypes.size() > 0 ? 
            (double) uniqueTypes / enabledEntityTypes.size() : 0.0;
        features.add(diversity);
        
        return features.stream().mapToDouble(Double::doubleValue).toArray();
    }
    
    /**
     * Generate comprehensive entity features.
     */
    private double[] generateComprehensiveFeatures(List<Entity> entities, String text) {
        var densityFeatures = generateDensityFeatures(entities, text);
        var features = new ArrayList<Double>();
        
        // Add density features
        for (var feature : densityFeatures) {
            features.add(feature);
        }
        
        // Add positional features
        if (!entities.isEmpty()) {
            // First entity position (normalized)
            var firstEntityPos = entities.get(0).getStartToken();
            var firstPosNorm = text.length() > 0 ? (double) firstEntityPos / text.length() : 0.0;
            features.add(firstPosNorm);
            
            // Entity spread (distance between first and last entity)
            if (entities.size() > 1) {
                var lastEntity = entities.get(entities.size() - 1);
                var spread = lastEntity.getEndToken() - firstEntityPos;
                var spreadNorm = text.length() > 0 ? (double) spread / text.length() : 0.0;
                features.add(spreadNorm);
            } else {
                features.add(0.0);
            }
        } else {
            features.add(0.0); // first position
            features.add(0.0); // spread
        }
        
        // Average entity length
        var avgLength = entities.stream()
            .mapToInt(e -> e.getText().length())
            .average()
            .orElse(0.0);
        features.add(useNormalization && text.length() > 0 ? avgLength / text.length() : avgLength);
        
        return features.stream().mapToDouble(Double::doubleValue).toArray();
    }
    
    /**
     * Determine the dominant entity type from a list of entities.
     * Uses frequency analysis to find the most common entity type.
     * 
     * @param entities List of extracted entities
     * @return The most frequent EntityType, or the first valid type if tied
     */
    private EntityType getDominantEntityType(List<Entity> entities) {
        if (entities.isEmpty()) {
            // Return first enabled entity type as default
            return enabledEntityTypes.iterator().next();
        }
        
        // Count entity types
        var typeCounts = new EnumMap<EntityType, Integer>(EntityType.class);
        
        for (var entity : entities) {
            try {
                var entityType = EntityType.valueOf(entity.getType());
                if (enabledEntityTypes.contains(entityType)) {
                    typeCounts.merge(entityType, 1, Integer::sum);
                }
            } catch (IllegalArgumentException e) {
                log.debug("Unknown entity type: {}", entity.getType());
                // Skip unknown entity types
            }
        }
        
        if (typeCounts.isEmpty()) {
            // No valid entities found, return default
            return enabledEntityTypes.iterator().next();
        }
        
        // Find the most frequent entity type
        return typeCounts.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse(enabledEntityTypes.iterator().next());
    }
    
    /**
     * Initialize entity type indices.
     */
    private void initializeEntityTypes() {
        for (var entityType : EntityType.values()) {
            getEntityTypeIndex(entityType.name());
        }
    }
    
    /**
     * Get or assign index for entity type.
     */
    private int getEntityTypeIndex(String entityType) {
        return entityTypeToIndex.computeIfAbsent(entityType, type -> nextEntityIndex.getAndIncrement());
    }

    /**
     * Get input stream for model file.
     */
    private InputStream getModelStream(Path modelPath) throws IOException {
        if (modelPath.toString().startsWith("classpath:")) {
            var resourcePath = modelPath.toString().substring(10);
            var stream = getClass().getClassLoader().getResourceAsStream(resourcePath);
            if (stream == null) {
                throw new IOException("Model resource not found: " + resourcePath);
            }
            return stream;
        } else {
            return modelPath.toUri().toURL().openStream();
        }
    }

    /**
     * Get default model path in classpath.
     */
    private static Path getDefaultModelPath(String modelName) {
        return Path.of("classpath:models/opennlp/" + modelName);
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
     * Batch extract entities from multiple texts.
     */
    public List<List<Entity>> extractEntitiesFromTexts(List<String> texts) {
        return texts.stream()
                   .map(this::extractEntities)
                   .toList();
    }

    /**
     * Get entity channel performance metrics.
     */
    public EntityMetrics getEntityMetrics() {
        var baseMetrics = getMetrics();
        
        return new EntityMetrics(
            baseMetrics.getTotalClassifications(),
            successfulClassifications.get(),
            baseMetrics.getCurrentCategoryCount(),
            baseMetrics.getAverageProcessingTimeMs(),
            totalSentences.get(),
            totalEntities.get(),
            enabledEntityTypes.size(),
            featureMode.name()
        );
    }

    @Override
    public void saveState() {
        getWriteLock().lock();
        try {
            var stateFile = Path.of("state", "channels", getChannelName() + ".state");
            Files.createDirectories(stateFile.getParent());
            
            var stateData = Map.<String, Object>of(
                "totalSentences", totalSentences.get(),
                "totalEntities", totalEntities.get(),
                "successfulClassifications", successfulClassifications.get(),
                "categoryCount", fuzzyARTMAP.getCategoryCount(),
                "timestamp", System.currentTimeMillis()
            );
            
            // Write state to file using simple serialization
            try (var fos = Files.newOutputStream(stateFile);
                 var oos = new ObjectOutputStream(fos)) {
                oos.writeObject(stateData);
            }
            
            log.debug("Saved entity channel '{}' state: {} sentences, {} entities, {} categories", 
                     getChannelName(), totalSentences.get(), totalEntities.get(), fuzzyARTMAP.getCategoryCount());
            
        } catch (Exception e) {
            log.error("Failed to save state for entity channel '{}': {}", getChannelName(), e.getMessage());
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
                    if (stateData.containsKey("totalSentences")) {
                        totalSentences.set((Integer) stateData.get("totalSentences"));
                    }
                    if (stateData.containsKey("totalEntities")) {
                        totalEntities.set((Integer) stateData.get("totalEntities"));
                    }
                    if (stateData.containsKey("successfulClassifications")) {
                        successfulClassifications.set((Integer) stateData.get("successfulClassifications"));
                    }
                    
                    log.info("Loaded state for entity channel '{}': {} sentences, {} entities, {} categories", 
                            getChannelName(), totalSentences.get(), totalEntities.get(), 
                            stateData.getOrDefault("categoryCount", 0));
                } catch (ClassNotFoundException e) {
                    log.warn("Failed to deserialize state for entity channel '{}', starting fresh", getChannelName());
                    initializeCleanState();
                }
            } else {
                // Initialize with clean state if no saved state exists
                initializeCleanState();
                log.debug("No saved state found for entity channel '{}', starting fresh", getChannelName());
            }
            
        } catch (Exception e) {
            log.error("Failed to load state for entity channel '{}': {}", getChannelName(), e.getMessage());
            initializeCleanState();
        } finally {
            getWriteLock().unlock();
        }
    }
    
    private void initializeCleanState() {
        totalSentences.set(0);
        totalEntities.set(0);
        successfulClassifications.set(0);
    }

    @Override
    public int getCategoryCount() {
        getReadLock().lock();
        try {
            return fuzzyARTMAP.getCategoryCount();
        } finally {
            getReadLock().unlock();
        }
    }

    @Override
    public int pruneCategories(double threshold) {
        getWriteLock().lock();
        try {
            // For now, return 0 as pruning is not directly available on FuzzyARTMAP
            // This method is meant for category management which FuzzyARTMAP handles internally
            log.debug("Pruning not implemented for entity channel '{}' with threshold {}", 
                     getChannelName(), threshold);
            return 0;
        } finally {
            getWriteLock().unlock();
        }
    }

    /**
     * Entity channel performance metrics.
     */
    public record EntityMetrics(
        long totalClassifications,
        int successfulClassifications,
        int categoryCount,
        double averageProcessingTime,
        int totalSentences,
        int totalEntities,
        int enabledEntityTypes,
        String featureModeName
    ) {
        public double successRate() {
            return totalClassifications > 0 ? 
                (double) successfulClassifications / totalClassifications : 0.0;
        }

        public double averageEntitiesPerSentence() {
            return totalSentences > 0 ? (double) totalEntities / totalSentences : 0.0;
        }

        public double averageEntitiesPerClassification() {
            return totalClassifications > 0 ? (double) totalEntities / totalClassifications : 0.0;
        }

        public double entityRecognitionRate() {
            return totalSentences > 0 ? (double) totalEntities / totalSentences : 0.0;
        }
    }
}