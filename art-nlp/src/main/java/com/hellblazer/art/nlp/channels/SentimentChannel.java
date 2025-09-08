package com.hellblazer.art.nlp.channels;

import com.hellblazer.art.nlp.config.ChannelConfig;
import com.hellblazer.art.nlp.core.Entity;
import com.hellblazer.art.nlp.core.SentimentScore;
import com.hellblazer.art.core.algorithms.FuzzyART;
import com.hellblazer.art.core.parameters.FuzzyParameters;
import com.hellblazer.art.core.DenseVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Pattern;

/**
 * Sentiment analysis channel using FuzzyART for emotion and opinion detection.
 * Specializes in identifying sentiment polarity, emotional tone, and subjective content.
 */
public class SentimentChannel extends com.hellblazer.art.nlp.channels.base.BaseChannel {
    
    private static final String CHANNEL_NAME = "sentiment";
    private static final Logger logger = LoggerFactory.getLogger(SentimentChannel.class);
    
    private final ChannelConfig config;
    private final FuzzyART fuzzyART;
    private final Map<String, Double> sentimentLexicon;
    private final Map<String, String> emotionLexicon;
    private final AtomicInteger categoryCounter;
    private final Pattern punctuationPattern;
    private final Pattern intensifierPattern;
    private final Pattern negationPattern;
    
    /**
     * Create SentimentChannel with default configuration.
     * Uses vigilance=0.5, learning rate=0.7, alpha=0.1, beta=1.0
     */
    public SentimentChannel() {
        this(ChannelConfig.builder()
            .channelName(CHANNEL_NAME)
            .vigilance(0.5)
            .learningRate(0.7)
            .build());
    }
    
    /**
     * Create SentimentChannel with custom configuration.
     * 
     * @param config Channel configuration
     */
    public SentimentChannel(ChannelConfig config) {
        super(config.getChannelName(), config.getVigilance());
        this.config = config;
        
        var dimensions = 80; // Default dimension for sentiment vectors
        this.categoryCounter = new AtomicInteger(0);
        
        // Initialize FuzzyART with complement coding
        this.fuzzyART = new FuzzyART();
        
        this.sentimentLexicon = new HashMap<>();
        this.emotionLexicon = new HashMap<>();
        
        // Regex patterns for sentiment analysis
        this.punctuationPattern = Pattern.compile("[!]{2,}|[?]{2,}|[.]{3,}");
        this.intensifierPattern = Pattern.compile("\\b(very|extremely|really|quite|rather|totally|completely|absolutely|incredibly|amazingly)\\b");
        this.negationPattern = Pattern.compile("\\b(not|no|never|none|nobody|nothing|neither|nowhere|hardly|scarcely|barely)\\b");
        
        initializeLexicons();
    }
    
    private void initializeLexicons() {
        // Positive sentiment words
        var positiveWords = Arrays.asList(
            "amazing", "awesome", "beautiful", "brilliant", "excellent", "fantastic",
            "good", "great", "happy", "incredible", "love", "perfect", "wonderful",
            "outstanding", "superb", "magnificent", "delightful", "pleased", "satisfied",
            "enjoy", "like", "appreciate", "admire", "praise", "recommend", "impressive"
        );
        
        // Negative sentiment words
        var negativeWords = Arrays.asList(
            "awful", "bad", "horrible", "terrible", "disgusting", "hate", "dislike",
            "disappointing", "frustrated", "angry", "sad", "depressed", "annoying",
            "irritating", "boring", "ugly", "stupid", "wrong", "failed", "worst",
            "pathetic", "useless", "worthless", "ridiculous", "absurd", "nonsense"
        );
        
        // Initialize sentiment lexicon
        for (var word : positiveWords) {
            sentimentLexicon.put(word, 1.0);
        }
        for (var word : negativeWords) {
            sentimentLexicon.put(word, -1.0);
        }
        
        // Emotion categories
        emotionLexicon.put("joy", "POSITIVE");
        emotionLexicon.put("happiness", "POSITIVE");
        emotionLexicon.put("excitement", "POSITIVE");
        emotionLexicon.put("love", "POSITIVE");
        emotionLexicon.put("anger", "NEGATIVE");
        emotionLexicon.put("rage", "NEGATIVE");
        emotionLexicon.put("sadness", "NEGATIVE");
        emotionLexicon.put("fear", "NEGATIVE");
        emotionLexicon.put("anxiety", "NEGATIVE");
        emotionLexicon.put("disgust", "NEGATIVE");
        emotionLexicon.put("surprise", "NEUTRAL");
        emotionLexicon.put("confusion", "NEUTRAL");
    }
    
    protected double[] extractFeatures(String text) {
        if (text == null || text.trim().isEmpty()) {
            return new double[80];
        }
        
        var features = new double[80];
        var cleanText = text.toLowerCase().trim();
        var words = cleanText.split("\\s+");
        
        // Extract sentiment features
        extractSentimentScores(words, features);
        extractEmotionalFeatures(words, features);
        extractLinguisticFeatures(cleanText, features);
        extractIntensityFeatures(cleanText, features);
        
        // Apply complement coding for FuzzyART
        applyComplementCoding(features);
        
        return features;
    }
    
    private void extractSentimentScores(String[] words, double[] features) {
        var baseIdx = 0;
        var positiveScore = 0.0;
        var negativeScore = 0.0;
        var objectiveScore = 0.0;
        var negationActive = false;
        
        for (var word : words) {
            // Clean word by removing punctuation
            var cleanWord = word.replaceAll("[^a-zA-Z]", "").toLowerCase();
            
            // Check for negation
            if (negationPattern.matcher(cleanWord).find()) {
                negationActive = !negationActive; // Toggle negation state
                continue;
            }
            
            // Calculate sentiment scores
            var sentiment = sentimentLexicon.getOrDefault(cleanWord, 0.0);
            
            // Apply negation effect to current word if negation is active
            if (negationActive && sentiment != 0.0) {
                sentiment = -sentiment; // Flip the sentiment
            }
            
            if (sentiment > 0) {
                positiveScore += sentiment;
            } else if (sentiment < 0) {
                negativeScore += Math.abs(sentiment);
            } else {
                objectiveScore += 0.1;
            }
            
            // Reset negation after processing a sentiment word (negation typically affects next word)
            if (negationActive && sentiment != 0.0) {
                negationActive = false;
            }
        }
        
        // Normalize and assign to features
        var totalScore = positiveScore + negativeScore + objectiveScore;
        if (totalScore > 0) {
            features[baseIdx] = positiveScore / totalScore;
            features[baseIdx + 1] = negativeScore / totalScore;
            features[baseIdx + 2] = objectiveScore / totalScore;
        }
    }
    
    private void extractEmotionalFeatures(String[] words, double[] features) {
        var baseIdx = 10;
        var emotionCounts = new HashMap<String, Double>();
        
        for (var word : words) {
            for (var entry : emotionLexicon.entrySet()) {
                if (word.contains(entry.getKey())) {
                    var emotion = entry.getValue();
                    emotionCounts.merge(emotion, 1.0, Double::sum);
                }
            }
        }
        
        // Map emotions to features
        var totalEmotions = emotionCounts.values().stream().mapToDouble(Double::doubleValue).sum();
        if (totalEmotions > 0) {
            features[baseIdx] = emotionCounts.getOrDefault("POSITIVE", 0.0) / totalEmotions;
            features[baseIdx + 1] = emotionCounts.getOrDefault("NEGATIVE", 0.0) / totalEmotions;
            features[baseIdx + 2] = emotionCounts.getOrDefault("NEUTRAL", 0.0) / totalEmotions;
        }
    }
    
    private void extractLinguisticFeatures(String text, double[] features) {
        var baseIdx = 20;
        
        // Punctuation intensity
        var exclamationCount = text.length() - text.replace("!", "").length();
        var questionCount = text.length() - text.replace("?", "").length();
        var capsCount = text.length() - text.replaceAll("[A-Z]", "").length();
        
        features[baseIdx] = Math.min(exclamationCount / 10.0, 1.0);
        features[baseIdx + 1] = Math.min(questionCount / 10.0, 1.0);
        features[baseIdx + 2] = Math.min(capsCount / text.length(), 1.0);
        
        // Sentence structure indicators
        features[baseIdx + 3] = punctuationPattern.matcher(text).find() ? 1.0 : 0.0;
    }
    
    private void extractIntensityFeatures(String text, double[] features) {
        var baseIdx = 30;
        
        // Intensifiers
        var intensifierMatcher = intensifierPattern.matcher(text);
        var intensifierCount = 0;
        while (intensifierMatcher.find()) {
            intensifierCount++;
        }
        
        features[baseIdx] = Math.min(intensifierCount / 5.0, 1.0);
        
        // Negation density
        var negationMatcher = negationPattern.matcher(text);
        var negationCount = 0;
        while (negationMatcher.find()) {
            negationCount++;
        }
        
        features[baseIdx + 1] = Math.min(negationCount / 5.0, 1.0);
    }
    
    private void applyComplementCoding(double[] features) {
        var halfSize = features.length / 2;
        
        // Apply complement coding: [x, 1-x]
        for (int i = 0; i < halfSize; i++) {
            features[halfSize + i] = 1.0 - features[i];
        }
    }
    
    protected Map<String, Integer> classifyFeatures(double[] features) {
        var categories = new HashMap<String, Integer>();
        
        if (features.length == 0) {
            return categories;
        }
        
        try {
            // Learn/classify with FuzzyART (placeholder - actual API would be used)
            var categoryId = Math.abs(Arrays.hashCode(features)) % 10;
            
            if (categoryId >= 0) {
                var sentimentType = determineSentimentType(features);
                var categoryName = CHANNEL_NAME + "_" + sentimentType + "_" + categoryId;
                categories.put(categoryName, categoryId);
                
                // Add specific sentiment categories
                addSentimentCategories(features, categories);
            }
        } catch (Exception e) {
            logger.warn("Sentiment classification failed", e);
        }
        
        return categories;
    }
    
    private String determineSentimentType(double[] features) {
        var positiveScore = features[0];
        var negativeScore = features[1];
        var objectiveScore = features[2];
        
        var maxScore = Math.max(positiveScore, Math.max(negativeScore, objectiveScore));
        
        if (maxScore == positiveScore && positiveScore > 0.4) {
            return "POSITIVE";
        } else if (maxScore == negativeScore && negativeScore > 0.4) {
            return "NEGATIVE";
        } else {
            return "NEUTRAL";
        }
    }
    
    private void addSentimentCategories(double[] features, Map<String, Integer> categories) {
        // Emotional intensity
        var emotionIntensity = features[30]; // Intensifier feature
        if (emotionIntensity > 0.5) {
            categories.put(CHANNEL_NAME + "_HIGH_INTENSITY", 1);
        } else if (emotionIntensity > 0.2) {
            categories.put(CHANNEL_NAME + "_MEDIUM_INTENSITY", 1);
        } else {
            categories.put(CHANNEL_NAME + "_LOW_INTENSITY", 1);
        }
        
        // Subjectivity
        var objectiveScore = features[2];
        if (objectiveScore > 0.6) {
            categories.put(CHANNEL_NAME + "_OBJECTIVE", 1);
        } else {
            categories.put(CHANNEL_NAME + "_SUBJECTIVE", 1);
        }
    }
    
    protected List<Entity> extractEntities(String text) {
        var entities = new ArrayList<Entity>();
        
        if (text == null || text.trim().isEmpty()) {
            return entities;
        }
        
        // Extract sentiment-bearing phrases and words
        extractSentimentEntities(text, entities);
        extractEmotionalEntities(text, entities);
        
        return entities;
    }
    
    private void extractSentimentEntities(String text, List<Entity> entities) {
        var lowerText = text.toLowerCase();
        var words = lowerText.split("\\s+");
        var currentPos = 0;
        
        for (var word : words) {
            var index = text.toLowerCase().indexOf(word, currentPos);
            var sentiment = sentimentLexicon.get(word);
            
            if (sentiment != null && index >= 0) {
                var type = sentiment > 0 ? "POSITIVE_SENTIMENT" : "NEGATIVE_SENTIMENT";
                var confidence = Math.abs(sentiment);
                
                entities.add(new Entity(
                    word,
                    type,
                    index,
                    index + word.length(),
                    confidence
                ));
            }
            currentPos = index + word.length();
        }
    }
    
    private void extractEmotionalEntities(String text, List<Entity> entities) {
        for (var entry : emotionLexicon.entrySet()) {
            var emotion = entry.getKey();
            var category = entry.getValue();
            var index = text.toLowerCase().indexOf(emotion);
            
            if (index >= 0) {
                entities.add(new Entity(
                    emotion,
                    "EMOTION_" + category,
                    index,
                    index + emotion.length(),
                    0.8
                ));
            }
        }
    }
    
    /**
     * Calculate sentiment score for given text.
     * 
     * @param text Input text
     * @return Sentiment score with polarity and confidence
     */
    public SentimentScore calculateSentimentScore(String text) {
        if (text == null || text.trim().isEmpty()) {
            return new SentimentScore(SentimentScore.Sentiment.NEUTRAL, 0.0, 0.0, 1.0, 0.5);
        }
        
        var features = extractFeatures(text);
        var positiveScore = features[0];
        var negativeScore = features[1];
        var objectiveScore = features[2];
        
        var sentiment = determineSentiment(positiveScore, negativeScore, objectiveScore);
        var polarity = calculatePolarity(positiveScore, negativeScore);
        var confidence = calculateConfidence(positiveScore, negativeScore, objectiveScore);
        
        return new SentimentScore(sentiment, positiveScore, negativeScore, objectiveScore, confidence);
    }
    
    private SentimentScore.Sentiment determineSentiment(double positive, double negative, double objective) {
        var maxScore = Math.max(positive, Math.max(negative, objective));
        
        if (maxScore == positive && positive > 0.4) {
            return SentimentScore.Sentiment.POSITIVE;
        } else if (maxScore == negative && negative > 0.4) {
            return SentimentScore.Sentiment.NEGATIVE;
        } else {
            return SentimentScore.Sentiment.NEUTRAL;
        }
    }
    
    private double calculatePolarity(double positive, double negative) {
        var total = positive + negative;
        if (total == 0) return 0.0;
        return (positive - negative) / total;
    }
    
    private double calculateConfidence(double positive, double negative, double objective) {
        var maxScore = Math.max(positive, Math.max(negative, objective));
        var total = positive + negative + objective;
        return total > 0 ? maxScore / total : 0.0;
    }
    
    public void reset() {
        // Note: BaseChannel doesn't have reset() method
        // fuzzyART.reset(); // Method doesn't exist in FuzzyART
        categoryCounter.set(0);
        
        logger.info("SentimentChannel reset completed");
    }
    
    @Override
    public int getCategoryCount() {
        return fuzzyART.getCategoryCount();
    }
    
    /**
     * Get the sentiment lexicon used by this channel.
     */
    public Map<String, Double> getSentimentLexicon() {
        return new HashMap<>(sentimentLexicon);
    }
    
    /**
     * Add custom sentiment word to lexicon.
     * 
     * @param word Word to add
     * @param sentiment Sentiment score (-1.0 to 1.0)
     */
    public void addSentimentWord(String word, double sentiment) {
        if (word != null && !word.trim().isEmpty() && sentiment >= -1.0 && sentiment <= 1.0) {
            sentimentLexicon.put(word.toLowerCase().trim(), sentiment);
        }
    }
    
    /**
     * Remove word from sentiment lexicon.
     * 
     * @param word Word to remove
     */
    public void removeSentimentWord(String word) {
        if (word != null) {
            sentimentLexicon.remove(word.toLowerCase().trim());
        }
    }
    
    // ===== Abstract method implementations =====
    
    @Override
    public int classify(DenseVector input) {
        try {
            // Preprocess the input vector (normalization + complement coding)
            var preprocessedInput = preprocessInput(input);
            
            // Use FuzzyART to classify the input
            // Note: In a real implementation, we'd use the actual FuzzyART API
            // For now, return a placeholder category based on input properties
            var category = Math.abs(Arrays.hashCode(input.data())) % 10;
            categoryCounter.set(Math.max(categoryCounter.get(), category + 1));
            
            logger.debug("Classified input vector to sentiment category: {}", category);
            return category;
            
        } catch (Exception e) {
            logger.error("Failed to classify input: {}", e.getMessage());
            recordError();
            return -1; // Error category
        }
    }
    
    @Override
    protected void performInitialization() {
        // Initialize sentiment lexicon with basic positive/negative words
        sentimentLexicon.put("good", 0.7);
        sentimentLexicon.put("great", 0.8);
        sentimentLexicon.put("excellent", 0.9);
        sentimentLexicon.put("bad", -0.7);
        sentimentLexicon.put("terrible", -0.8);
        sentimentLexicon.put("awful", -0.9);
        sentimentLexicon.put("love", 0.8);
        sentimentLexicon.put("hate", -0.8);
        sentimentLexicon.put("like", 0.5);
        sentimentLexicon.put("dislike", -0.5);
        
        // Initialize emotion lexicon
        emotionLexicon.put("happy", "joy");
        emotionLexicon.put("sad", "sadness");
        emotionLexicon.put("angry", "anger");
        emotionLexicon.put("scared", "fear");
        emotionLexicon.put("surprised", "surprise");
        
        logger.info("Sentiment channel initialized with {} sentiment words and {} emotion mappings", 
                   sentimentLexicon.size(), emotionLexicon.size());
    }
    
    @Override
    protected void performCleanup() {
        // Clear lexicons to free memory
        sentimentLexicon.clear();
        emotionLexicon.clear();
        logger.info("Sentiment channel cleaned up successfully");
    }
    
    
    @Override
    public int pruneCategories(double threshold) {
        // For now, return 0 as pruning would require access to FuzzyART internals
        // In a full implementation, this would prune low-usage categories
        logger.debug("Pruning categories with threshold {}, but not implemented yet", threshold);
        return 0;
    }
    
    @Override
    public void saveState() {
        try {
            // In a full implementation, this would save:
            // - FuzzyART weights and categories
            // - Custom sentiment lexicon additions
            // - Channel metrics and statistics
            logger.debug("Saving sentiment channel state (placeholder implementation)");
        } catch (Exception e) {
            logger.error("Failed to save sentiment channel state: {}", e.getMessage());
            throw new RuntimeException("State save failed", e);
        }
    }
    
    @Override
    public void loadState() {
        try {
            // In a full implementation, this would load:
            // - Previously saved FuzzyART weights and categories  
            // - Custom sentiment lexicon additions
            // - Channel metrics and statistics
            logger.debug("Loading sentiment channel state (placeholder implementation)");
        } catch (Exception e) {
            logger.warn("Failed to load sentiment channel state: {}", e.getMessage());
            // Don't throw exception on load failure - start fresh instead
        }
    }
}