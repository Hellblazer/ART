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

import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.salience.SalienceAwareART;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.nlp.channels.base.BaseChannel;

/**
 * Syntactic channel using OpenNLP for POS tagging and syntactic analysis.
 * Processes text into syntactic feature vectors for pattern recognition.
 */
public final class SyntacticChannel extends BaseChannel {
    private static final Logger log = LoggerFactory.getLogger(SyntacticChannel.class);

    // OpenNLP models and tools
    private TokenizerME tokenizer;
    private POSTaggerME posTagger;
    private SentenceDetectorME sentenceDetector;
    
    // Model paths
    private final Path tokenizerModelPath;
    private final Path posModelPath;
    private final Path sentenceModelPath;
    
    // Configuration
    private final int maxTokensPerSentence;
    private final boolean useNormalization;
    private final SyntacticFeatureSet featureSet;
    
    // ART algorithm for syntactic clustering
    private final SalienceAwareART salienceART;
    
    // POS tag encoding
    private final Map<String, Integer> posTagToIndex = new ConcurrentHashMap<>();
    private final AtomicInteger nextPosIndex = new AtomicInteger(0);
    
    // Performance metrics
    private final AtomicInteger totalSentences = new AtomicInteger(0);
    private final AtomicInteger totalTokens = new AtomicInteger(0);
    private final AtomicInteger successfulClassifications = new AtomicInteger(0);

    /**
     * Feature sets for syntactic analysis.
     */
    public enum SyntacticFeatureSet {
        /** Basic POS tag distribution */
        POS_DISTRIBUTION,
        /** POS tags + syntactic patterns */
        POS_PATTERNS,
        /** Full syntactic feature extraction */
        FULL_SYNTAX
    }

    /**
     * Create syntactic channel with default configuration.
     */
    public SyntacticChannel(String channelName, double vigilance) {
        this(channelName, vigilance, 
             getDefaultModelPath("en-token.bin"),
             getDefaultModelPath("en-pos-maxent.bin"),
             getDefaultModelPath("en-sent.bin"),
             SyntacticFeatureSet.POS_DISTRIBUTION, 100, true);
    }

    /**
     * Create syntactic channel with custom configuration.
     */
    public SyntacticChannel(String channelName, double vigilance,
                           Path tokenizerModelPath, Path posModelPath, Path sentenceModelPath,
                           SyntacticFeatureSet featureSet, int maxTokensPerSentence, boolean useNormalization) {
        super(channelName, vigilance);
        
        this.tokenizerModelPath = Objects.requireNonNull(tokenizerModelPath, "tokenizerModelPath cannot be null");
        this.posModelPath = Objects.requireNonNull(posModelPath, "posModelPath cannot be null");  
        this.sentenceModelPath = Objects.requireNonNull(sentenceModelPath, "sentenceModelPath cannot be null");
        this.featureSet = Objects.requireNonNull(featureSet, "featureSet cannot be null");
        this.maxTokensPerSentence = maxTokensPerSentence;
        this.useNormalization = useNormalization;
        
        // Initialize SalienceAwareART algorithm for syntactic clustering
        this.salienceART = new SalienceAwareART.Builder()
            .vigilance(vigilance)
            .learningRate(0.8)
            .alpha(0.001)
            .salienceUpdateRate(0.01)
            .useSparseMode(true)
            .sparsityThreshold(0.01)
            .build();
        
        log.info("Syntactic channel '{}' created: features={}, maxTokens={}, normalize={}, vigilance={}", 
                channelName, featureSet, maxTokensPerSentence, useNormalization, vigilance);
    }

    @Override
    protected void performInitialization() {
        try {
            // Load tokenizer
            try (var tokenizerStream = getModelStream(tokenizerModelPath)) {
                var tokenizerModel = new TokenizerModel(tokenizerStream);
                this.tokenizer = new TokenizerME(tokenizerModel);
            }
            
            // Load POS tagger
            try (var posStream = getModelStream(posModelPath)) {
                var posModel = new POSModel(posStream);
                this.posTagger = new POSTaggerME(posModel);
            }
            
            // Load sentence detector
            try (var sentStream = getModelStream(sentenceModelPath)) {
                var sentModel = new SentenceModel(sentStream);
                this.sentenceDetector = new SentenceDetectorME(sentModel);
            }
            
            // Initialize common POS tags
            initializeCommonPosTags();
            
            log.info("OpenNLP models initialized for syntactic channel '{}'", getChannelName());
            
        } catch (IOException e) {
            log.error("Failed to initialize OpenNLP models for channel '{}'", getChannelName(), e);
            throw new RuntimeException("OpenNLP initialization failed", e);
        }
    }

    @Override
    protected void performCleanup() {
        // OpenNLP tools don't require explicit cleanup
        tokenizer = null;
        posTagger = null;
        sentenceDetector = null;
        posTagToIndex.clear();
        log.debug("Syntactic channel '{}' cleanup complete", getChannelName());
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
            
            // Use SalienceAwareART to classify the vector
            var result = salienceART.stepFit(processedInput);
            
            if (result instanceof ActivationResult.Success success) {
                var processingTime = System.currentTimeMillis() - startTime;
                recordClassification(processingTime, false); // Remove wasNewCategoryCreated() - not available
                successfulClassifications.incrementAndGet();
                return success.categoryIndex();
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
     * Process text input into syntactic vector for classification.
     * 
     * @param text Input text to process
     * @return Syntactic category ID
     */
    public int classifyText(String text) {
        if (text == null || text.isBlank()) {
            return -1; // Invalid input
        }
        
        // Check if components are initialized
        if (sentenceDetector == null || tokenizer == null || posTagger == null) {
            log.error("SyntacticChannel not properly initialized - sentenceDetector: {}, tokenizer: {}, posTagger: {}", 
                     sentenceDetector != null, tokenizer != null, posTagger != null);
            return -1;
        }

        try {
            // Detect sentences
            var sentences = sentenceDetector.sentDetect(text);
            totalSentences.addAndGet(sentences.length);
            
            var allFeatures = new ArrayList<double[]>();
            
            for (var sentence : sentences) {
                // Sanitize sentence to prevent tokenization errors
                var cleanSentence = sentence.trim();
                if (cleanSentence.isEmpty()) continue;
                
                // Tokenize sentence with error handling
                String[] tokens;
                try {
                    tokens = tokenizer.tokenize(cleanSentence);
                } catch (IllegalArgumentException e) {
                    log.debug("Tokenization failed for sentence '{}': {}", cleanSentence, e.getMessage());
                    continue; // Skip problematic sentences
                }
                if (tokens.length == 0) continue;
                
                // Limit tokens per sentence
                var processTokens = Arrays.copyOf(tokens, Math.min(tokens.length, maxTokensPerSentence));
                totalTokens.addAndGet(processTokens.length);
                
                // Get POS tags
                var posTags = posTagger.tag(processTokens);
                
                // Extract syntactic features
                var features = extractSyntacticFeatures(processTokens, posTags);
                if (features != null && features.length > 0) {
                    allFeatures.add(features);
                }
            }
            
            if (allFeatures.isEmpty()) {
                log.debug("No syntactic features extracted for text: {}", text);
                return -1;
            }
            
            // Aggregate features across sentences
            var aggregatedFeatures = aggregateFeatures(allFeatures);
            var featureVector = new DenseVector(aggregatedFeatures);
            
            // Classify using base ART algorithm
            var category = classify(featureVector);
            
            if (category >= 0) {
                successfulClassifications.incrementAndGet();
            }
            
            return category;
            
        } catch (Exception e) {
            log.debug("Error processing syntactic features for text: {}", text, e);
            return -1;
        }
    }

    /**
     * Extract syntactic features from tokens and POS tags.
     */
    private double[] extractSyntacticFeatures(String[] tokens, String[] posTags) {
        return switch (featureSet) {
            case POS_DISTRIBUTION -> extractPosDistribution(posTags);
            case POS_PATTERNS -> extractPosPatterns(tokens, posTags);
            case FULL_SYNTAX -> extractFullSyntacticFeatures(tokens, posTags);
        };
    }
    
    /**
     * Extract POS tag distribution features.
     */
    private double[] extractPosDistribution(String[] posTags) {
        // Create feature vector based on POS tag frequencies
        var posCount = new HashMap<String, Integer>();
        for (var tag : posTags) {
            posCount.merge(tag, 1, Integer::sum);
        }
        
        // Convert to normalized distribution vector
        var features = new double[50]; // Fixed size for common POS tags
        var totalTags = posTags.length;
        
        for (var entry : posCount.entrySet()) {
            var index = getPosTagIndex(entry.getKey());
            if (index < features.length) {
                features[index] = useNormalization ? 
                    (double) entry.getValue() / totalTags : entry.getValue();
            }
        }
        
        return features;
    }
    
    /**
     * Extract POS patterns and transitions.
     */
    private double[] extractPosPatterns(String[] tokens, String[] posTags) {
        var posDistribution = extractPosDistribution(posTags);
        var patternFeatures = new ArrayList<Double>();
        
        // Add base POS distribution
        for (var feature : posDistribution) {
            patternFeatures.add(feature);
        }
        
        // Add bigram POS patterns with proper syntactic analysis
        var bigramCounts = new HashMap<String, Integer>();
        for (int i = 0; i < posTags.length - 1; i++) {
            var bigram = posTags[i] + "_" + posTags[i + 1];
            bigramCounts.merge(bigram, 1, Integer::sum);
        }
        
        // Convert bigram patterns to feature vector
        var totalBigrams = Math.max(1, bigramCounts.values().stream().mapToInt(Integer::intValue).sum());
        for (var entry : bigramCounts.entrySet()) {
            var bigramPattern = entry.getKey();
            var count = entry.getValue();
            
            // Analyze syntactic significance of bigram
            var significance = analyzeSyntacticSignificance(bigramPattern);
            var normalizedCount = useNormalization ? (double) count / totalBigrams : count;
            var featureValue = normalizedCount * significance;
            
            // Add to pattern features
            patternFeatures.add(featureValue);
        }
        
        // Add trigram patterns separately
        var trigramCounts = new HashMap<String, Integer>();
        for (int i = 0; i < posTags.length - 2; i++) {
            var trigram = posTags[i] + "_" + posTags[i + 1] + "_" + posTags[i + 2];
            trigramCounts.merge(trigram, 1, Integer::sum);
        }
        
        var totalTrigrams = Math.max(1, trigramCounts.values().stream().mapToInt(Integer::intValue).sum());
        for (var entry : trigramCounts.entrySet()) {
            var trigramPattern = entry.getKey();
            var count = entry.getValue();
            var trigramSignificance = analyzeSyntacticSignificance(trigramPattern);
            var normalizedCount = useNormalization ? (double) count / totalTrigrams : count;
            patternFeatures.add(normalizedCount * trigramSignificance * 0.8); // Weight trigrams slightly less
        }
        
        return patternFeatures.stream().mapToDouble(Double::doubleValue).toArray();
    }
    
    /**
     * Extract comprehensive syntactic features.
     */
    private double[] extractFullSyntacticFeatures(String[] tokens, String[] posTags) {
        var features = new ArrayList<Double>();
        
        // Base POS distribution
        var posFeatures = extractPosDistribution(posTags);
        for (var feature : posFeatures) {
            features.add(feature);
        }
        
        // Sentence length features
        features.add(useNormalization ? Math.log(tokens.length + 1) / 10.0 : tokens.length);
        
        // POS diversity (number of unique POS tags)
        var uniquePosCount = Arrays.stream(posTags).collect(HashSet::new, Set::add, Set::addAll).size();
        features.add(useNormalization ? (double) uniquePosCount / posTags.length : uniquePosCount);
        
        // Simple syntactic complexity (transitions)
        var transitions = 0;
        for (int i = 0; i < posTags.length - 1; i++) {
            if (!posTags[i].equals(posTags[i + 1])) {
                transitions++;
            }
        }
        features.add(useNormalization ? (double) transitions / (posTags.length - 1) : transitions);
        
        return features.stream().mapToDouble(Double::doubleValue).toArray();
    }
    
    /**
     * Analyze the syntactic significance of POS patterns.
     * 
     * @param pattern POS pattern (bigram or trigram)
     * @return Significance score (0.0-1.0)
     */
    private double analyzeSyntacticSignificance(String pattern) {
        // Define syntactically significant patterns using HashMap
        var significantPatterns = new HashMap<String, Double>();
        
        // Noun phrases
        significantPatterns.put("DT_NN", 0.9);     // the cat
        significantPatterns.put("DT_JJ", 0.8);     // the big
        significantPatterns.put("JJ_NN", 0.9);     // big cat
        significantPatterns.put("NN_NN", 0.7);     // compound nouns
        significantPatterns.put("NNP_NNP", 0.8);   // proper noun sequences
        
        // Verb phrases
        significantPatterns.put("VBZ_VBG", 0.8);   // is running
        significantPatterns.put("VB_DT", 0.7);     // verb determiner
        significantPatterns.put("MD_VB", 0.9);     // modal verb (can run)
        significantPatterns.put("RB_VB", 0.8);     // adverb verb
        
        // Prepositional phrases
        significantPatterns.put("IN_DT", 0.8);     // in the
        significantPatterns.put("IN_NN", 0.7);     // in house
        
        // Question patterns
        significantPatterns.put("WP_VBZ", 0.9);    // what is
        significantPatterns.put("WRB_VBZ", 0.9);   // where is
        
        // Adjective patterns
        significantPatterns.put("RB_JJ", 0.7);     // very big
        significantPatterns.put("JJ_CC", 0.6);     // big and
        
        // Common trigrams
        significantPatterns.put("DT_JJ_NN", 0.95); // the big cat
        significantPatterns.put("DT_NN_VBZ", 0.9); // the cat runs
        significantPatterns.put("MD_VB_DT", 0.8);  // can see the
        significantPatterns.put("IN_DT_NN", 0.85); // in the house
        
        // Check for exact pattern match
        if (significantPatterns.containsKey(pattern)) {
            return significantPatterns.get(pattern);
        }
        
        // Analyze pattern components for partial significance
        var parts = pattern.split("_");
        var score = 0.0;
        
        // Weight by part-of-speech significance
        for (var part : parts) {
            score += getPosSignificance(part);
        }
        
        // Normalize by number of parts and cap at 1.0
        return Math.min(score / parts.length, 1.0);
    }
    
    /**
     * Get significance score for individual POS tag.
     */
    private double getPosSignificance(String posTag) {
        return switch (posTag.substring(0, Math.min(2, posTag.length()))) {
            case "NN", "VB", "JJ", "RB" -> 0.8; // Content words
            case "DT", "IN", "CC", "TO" -> 0.4; // Function words
            case "WP", "WR", "WD" -> 0.9;       // Question words
            case "MD" -> 0.7;                   // Modals
            case "PD", "PO", "PR" -> 0.5;       // Pronouns/determiners
            default -> 0.3;                     // Other tags
        };
    }
    
    /**
     * Aggregate features from multiple sentences.
     */
    private double[] aggregateFeatures(List<double[]> allFeatures) {
        if (allFeatures.isEmpty()) return new double[0];
        
        var featureLength = allFeatures.get(0).length;
        var aggregated = new double[featureLength];
        
        // Mean aggregation across sentences
        for (var features : allFeatures) {
            for (int i = 0; i < Math.min(features.length, featureLength); i++) {
                aggregated[i] += features[i];
            }
        }
        
        // Normalize by sentence count
        var sentenceCount = allFeatures.size();
        for (int i = 0; i < aggregated.length; i++) {
            aggregated[i] /= sentenceCount;
        }
        
        return aggregated;
    }
    
    /**
     * Get or assign index for POS tag.
     */
    private int getPosTagIndex(String posTag) {
        return posTagToIndex.computeIfAbsent(posTag, tag -> nextPosIndex.getAndIncrement());
    }
    
    /**
     * Initialize common POS tags with consistent indices.
     */
    private void initializeCommonPosTags() {
        // Penn Treebank POS tags
        var commonTags = Arrays.asList(
            "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS",
            "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB",
            "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN",
            "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"
        );
        
        for (var tag : commonTags) {
            getPosTagIndex(tag);
        }
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
     * Get syntactic channel performance metrics.
     */
    public SyntacticMetrics getSyntacticMetrics() {
        var baseMetrics = getMetrics();
        
        return new SyntacticMetrics(
            baseMetrics.getTotalClassifications(),
            successfulClassifications.get(),
            baseMetrics.getCurrentCategoryCount(),
            baseMetrics.getAverageProcessingTimeMs(),
            totalSentences.get(),
            totalTokens.get(),
            posTagToIndex.size(),
            featureSet.name()
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
                "totalTokens", totalTokens.get(),
                "successfulClassifications", successfulClassifications.get(),
                "categoryCount", salienceART.getNumberOfCategories(),
                "posTagCount", posTagToIndex.size(),
                "timestamp", System.currentTimeMillis()
            );
            
            // Write state to file using simple serialization
            try (var fos = Files.newOutputStream(stateFile);
                 var oos = new ObjectOutputStream(fos)) {
                oos.writeObject(stateData);
            }
            
            log.debug("Saved syntactic channel '{}' state: {} sentences, {} tokens, {} categories", 
                     getChannelName(), totalSentences.get(), totalTokens.get(), salienceART.getNumberOfCategories());
            
        } catch (Exception e) {
            log.error("Failed to save state for syntactic channel '{}': {}", getChannelName(), e.getMessage());
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
                    if (stateData.containsKey("totalTokens")) {
                        totalTokens.set((Integer) stateData.get("totalTokens"));
                    }
                    if (stateData.containsKey("successfulClassifications")) {
                        successfulClassifications.set((Integer) stateData.get("successfulClassifications"));
                    }
                    
                    log.info("Loaded state for syntactic channel '{}': {} sentences, {} tokens, {} categories", 
                            getChannelName(), totalSentences.get(), totalTokens.get(), 
                            stateData.getOrDefault("categoryCount", 0));
                } catch (ClassNotFoundException e) {
                    log.warn("Failed to deserialize state for syntactic channel '{}', starting fresh", getChannelName());
                    initializeCleanState();
                }
            } else {
                // Initialize with clean state if no saved state exists
                initializeCleanState();
                log.debug("No saved state found for syntactic channel '{}', starting fresh", getChannelName());
            }
            
        } catch (Exception e) {
            log.error("Failed to load state for syntactic channel '{}': {}", getChannelName(), e.getMessage());
            initializeCleanState();
        } finally {
            getWriteLock().unlock();
        }
    }
    
    private void initializeCleanState() {
        totalSentences.set(0);
        totalTokens.set(0);
        successfulClassifications.set(0);
    }

    @Override
    public int getCategoryCount() {
        getReadLock().lock();
        try {
            return salienceART.getNumberOfCategories();
        } finally {
            getReadLock().unlock();
        }
    }

    @Override
    public int pruneCategories(double threshold) {
        getWriteLock().lock();
        try {
            // Return 0 as placeholder since pruneByUsageFrequency method is not available
            log.debug("Pruning not implemented for syntactic channel '{}' with threshold {}", 
                     getChannelName(), threshold);
            return 0;
        } finally {
            getWriteLock().unlock();
        }
    }

    /**
     * Syntactic channel performance metrics.
     */
    public record SyntacticMetrics(
        long totalClassifications,
        int successfulClassifications,
        int categoryCount,
        double averageProcessingTime,
        int totalSentences,
        int totalTokens,
        int uniquePosTagCount,
        String featureSetName
    ) {
        public double successRate() {
            return totalClassifications > 0 ? 
                (double) successfulClassifications / totalClassifications : 0.0;
        }

        public double averageTokensPerSentence() {
            return totalSentences > 0 ? (double) totalTokens / totalSentences : 0.0;
        }

        public double averageTokensPerClassification() {
            return totalClassifications > 0 ? (double) totalTokens / totalClassifications : 0.0;
        }
    }
}