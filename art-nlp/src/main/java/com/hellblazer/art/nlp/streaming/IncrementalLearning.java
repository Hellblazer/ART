package com.hellblazer.art.nlp.streaming;

import java.time.Instant;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;

import com.hellblazer.art.nlp.core.ProcessingResult;
import com.hellblazer.art.nlp.processor.ChannelResult;
import com.hellblazer.art.core.DenseVector;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Incremental learning extensions for ART channels enabling online learning
 * without full model retraining. Supports continuous adaptation to new data
 * patterns while preserving existing knowledge.
 */
public class IncrementalLearning {
    private static final Logger log = LoggerFactory.getLogger(IncrementalLearning.class);
    
    /**
     * Configuration for incremental learning behavior.
     */
    public record IncrementalConfig(
        double learningRate,
        double vigilanceDecay,
        int maxCategories,
        boolean enableForgetting,
        double forgettingFactor,
        int minCategorySupport
    ) {
        public static IncrementalConfig defaultConfig() {
            return new IncrementalConfig(
                0.1,    // learningRate
                0.995,  // vigilanceDecay
                1000,   // maxCategories
                true,   // enableForgetting
                0.001,  // forgettingFactor
                5       // minCategorySupport
            );
        }
        
        public IncrementalConfig withLearningRate(double rate) {
            return new IncrementalConfig(rate, vigilanceDecay, maxCategories, 
                                       enableForgetting, forgettingFactor, minCategorySupport);
        }
        
        public IncrementalConfig withVigilanceDecay(double decay) {
            return new IncrementalConfig(learningRate, decay, maxCategories,
                                       enableForgetting, forgettingFactor, minCategorySupport);
        }
    }
    
    /**
     * Incremental learning state for a specific channel.
     */
    public static class ChannelLearningState {
        private final String channelName;
        private final IncrementalConfig config;
        private final Map<Integer, CategoryState> categories = new ConcurrentHashMap<>();
        private final AtomicLong learningCount = new AtomicLong(0);
        private volatile double currentVigilance;
        private volatile Instant lastLearningTime;
        
        public ChannelLearningState(String channelName, IncrementalConfig config, double initialVigilance) {
            this.channelName = channelName;
            this.config = config;
            this.currentVigilance = initialVigilance;
            this.lastLearningTime = Instant.now();
            
            log.debug("Created incremental learning state for channel '{}': vigilance={}, learningRate={}", 
                     channelName, initialVigilance, config.learningRate());
        }
        
        /**
         * Updates learning state based on new processing result.
         */
        public synchronized LearningUpdate learn(ProcessingResult result) {
            var startTime = Instant.now();
            var channelResult = result.getChannelResults().get(channelName);
            
            if (channelResult == null || !channelResult.success()) {
                log.debug("Skipping learning for channel '{}': no valid result", channelName);
                return LearningUpdate.noUpdate(channelName, "No valid channel result");
            }
            
            var category = channelResult.category();
            var confidence = channelResult.confidence();
            var features = result.getFusedFeatures();
            
            // Get or create category state
            var categoryState = categories.computeIfAbsent(category, 
                k -> new CategoryState(k, features, confidence, startTime));
            
            // Perform incremental update
            var updateResult = updateCategory(categoryState, features, confidence);
            
            // Apply vigilance decay
            currentVigilance *= config.vigilanceDecay();
            
            // Update counters and timestamps
            learningCount.incrementAndGet();
            lastLearningTime = startTime;
            
            // Check for category pruning
            if (config.enableForgetting()) {
                pruneWeakCategories();
            }
            
            log.debug("Incremental learning update for channel '{}': category={}, confidence={:.3f}, vigilance={:.3f}", 
                     channelName, category, confidence, currentVigilance);
            
            return updateResult;
        }
        
        /**
         * Updates a category with new features using incremental learning.
         */
        private LearningUpdate updateCategory(CategoryState categoryState, DenseVector newFeatures, double confidence) {
            var oldPrototype = categoryState.prototype;
            var oldActivation = categoryState.averageActivation;
            
            // Compute learning rate based on category age and confidence
            var adaptiveLearningRate = computeAdaptiveLearningRate(categoryState, confidence);
            
            // Update prototype using exponential moving average
            var updatedPrototype = updatePrototype(oldPrototype, newFeatures, adaptiveLearningRate);
            
            // Update category statistics
            categoryState.prototype = updatedPrototype;
            categoryState.supportCount++;
            categoryState.averageActivation = updateMovingAverage(oldActivation, confidence, adaptiveLearningRate);
            categoryState.lastUpdateTime = Instant.now();
            
            // Compute update magnitude for monitoring
            var updateMagnitude = computeUpdateMagnitude(oldPrototype, updatedPrototype);
            
            return new LearningUpdate(
                channelName,
                categoryState.categoryId,
                updateMagnitude,
                adaptiveLearningRate,
                categoryState.supportCount,
                true,
                null
            );
        }
        
        /**
         * Computes adaptive learning rate based on category state and confidence.
         */
        private double computeAdaptiveLearningRate(CategoryState categoryState, double confidence) {
            // Base learning rate from config
            var baseLearningRate = config.learningRate();
            
            // Adjust based on confidence (higher confidence = slower learning)
            var confidenceAdjustment = 1.0 - (confidence * 0.5);
            
            // Adjust based on category support (more support = slower learning)
            var supportAdjustment = 1.0 / (1.0 + Math.log(categoryState.supportCount + 1));
            
            return baseLearningRate * confidenceAdjustment * supportAdjustment;
        }
        
        /**
         * Updates prototype vector using exponential moving average.
         */
        private DenseVector updatePrototype(DenseVector oldPrototype, DenseVector newFeatures, double learningRate) {
            var dimension = oldPrototype.dimension();
            var updatedValues = new double[dimension];
            
            for (int i = 0; i < dimension; i++) {
                var oldValue = oldPrototype.get(i);
                var newValue = newFeatures.get(i);
                updatedValues[i] = oldValue * (1.0 - learningRate) + newValue * learningRate;
            }
            
            return new DenseVector(updatedValues);
        }
        
        /**
         * Updates moving average with new value.
         */
        private double updateMovingAverage(double oldAverage, double newValue, double learningRate) {
            return oldAverage * (1.0 - learningRate) + newValue * learningRate;
        }
        
        /**
         * Computes magnitude of prototype update for monitoring.
         */
        private double computeUpdateMagnitude(DenseVector oldPrototype, DenseVector newPrototype) {
            var sumSquaredDiffs = 0.0;
            var dimension = oldPrototype.dimension();
            
            for (int i = 0; i < dimension; i++) {
                var diff = newPrototype.get(i) - oldPrototype.get(i);
                sumSquaredDiffs += diff * diff;
            }
            
            return Math.sqrt(sumSquaredDiffs);
        }
        
        /**
         * Prunes categories with insufficient support.
         */
        private void pruneWeakCategories() {
            var currentTime = Instant.now();
            var prunedCount = 0;
            
            var iterator = categories.entrySet().iterator();
            while (iterator.hasNext()) {
                var entry = iterator.next();
                var categoryState = entry.getValue();
                
                // Check if category should be forgotten
                if (shouldForgetCategory(categoryState, currentTime)) {
                    iterator.remove();
                    prunedCount++;
                    log.debug("Pruned category {} from channel '{}': support={}, lastUpdate={}", 
                             entry.getKey(), channelName, categoryState.supportCount, categoryState.lastUpdateTime);
                }
            }
            
            if (prunedCount > 0) {
                log.info("Pruned {} weak categories from channel '{}'", prunedCount, channelName);
            }
        }
        
        /**
         * Determines if a category should be forgotten based on support and recency.
         */
        private boolean shouldForgetCategory(CategoryState categoryState, Instant currentTime) {
            // Check minimum support threshold
            if (categoryState.supportCount < config.minCategorySupport()) {
                return true;
            }
            
            // Check if category is too old without recent updates
            var timeSinceUpdate = java.time.Duration.between(categoryState.lastUpdateTime, currentTime);
            var maxAge = java.time.Duration.ofHours(24); // Configurable threshold
            
            if (timeSinceUpdate.compareTo(maxAge) > 0 && categoryState.averageActivation < 0.1) {
                return true;
            }
            
            return false;
        }
        
        // Getters for monitoring
        public String getChannelName() { return channelName; }
        public double getCurrentVigilance() { return currentVigilance; }
        public long getLearningCount() { return learningCount.get(); }
        public int getCategoryCount() { return categories.size(); }
        public Instant getLastLearningTime() { return lastLearningTime; }
        public Map<Integer, CategoryState> getCategories() { return Map.copyOf(categories); }
    }
    
    /**
     * State information for a single category in incremental learning.
     */
    public static class CategoryState {
        public final int categoryId;
        public volatile DenseVector prototype;
        public volatile int supportCount;
        public volatile double averageActivation;
        public volatile Instant lastUpdateTime;
        
        public CategoryState(int categoryId, DenseVector initialPrototype, double initialActivation, Instant createTime) {
            this.categoryId = categoryId;
            this.prototype = initialPrototype;
            this.supportCount = 1;
            this.averageActivation = initialActivation;
            this.lastUpdateTime = createTime;
        }
        
        @Override
        public String toString() {
            return String.format("CategoryState[id=%d, support=%d, activation=%.3f, lastUpdate=%s]",
                                categoryId, supportCount, averageActivation, lastUpdateTime);
        }
    }
    
    /**
     * Result of an incremental learning update.
     */
    public record LearningUpdate(
        String channelName,
        int categoryId,
        double updateMagnitude,
        double learningRate,
        int supportCount,
        boolean success,
        String errorMessage
    ) {
        public static LearningUpdate noUpdate(String channelName, String reason) {
            return new LearningUpdate(channelName, -1, 0.0, 0.0, 0, false, reason);
        }
        
        public boolean isSignificant() {
            return success && updateMagnitude > 0.001;
        }
    }
    
    /**
     * Manager for incremental learning across multiple channels.
     */
    public static class IncrementalLearningManager {
        private final Map<String, ChannelLearningState> channelStates = new ConcurrentHashMap<>();
        private final IncrementalConfig defaultConfig;
        private final AtomicLong totalUpdates = new AtomicLong(0);
        
        public IncrementalLearningManager(IncrementalConfig defaultConfig) {
            this.defaultConfig = defaultConfig;
            log.info("Created incremental learning manager with config: {}", defaultConfig);
        }
        
        /**
         * Registers a channel for incremental learning.
         */
        public void registerChannel(String channelName, double initialVigilance) {
            registerChannel(channelName, initialVigilance, defaultConfig);
        }
        
        /**
         * Registers a channel for incremental learning with specific configuration.
         */
        public void registerChannel(String channelName, double initialVigilance, IncrementalConfig config) {
            var state = new ChannelLearningState(channelName, config, initialVigilance);
            channelStates.put(channelName, state);
            log.info("Registered channel '{}' for incremental learning: vigilance={}", 
                    channelName, initialVigilance);
        }
        
        /**
         * Performs incremental learning update for all channels in the result.
         */
        public Map<String, LearningUpdate> learn(ProcessingResult result) {
            var updates = new ConcurrentHashMap<String, LearningUpdate>();
            
            // Process each registered channel
            channelStates.entrySet().parallelStream().forEach(entry -> {
                var channelName = entry.getKey();
                var channelState = entry.getValue();
                
                try {
                    var update = channelState.learn(result);
                    updates.put(channelName, update);
                    
                    if (update.success()) {
                        totalUpdates.incrementAndGet();
                    }
                } catch (Exception e) {
                    log.error("Error during incremental learning for channel '{}'", channelName, e);
                    updates.put(channelName, LearningUpdate.noUpdate(channelName, e.getMessage()));
                }
            });
            
            return updates;
        }
        
        /**
         * Gets learning state for a specific channel.
         */
        public ChannelLearningState getChannelState(String channelName) {
            return channelStates.get(channelName);
        }
        
        /**
         * Gets summary of incremental learning statistics.
         */
        public IncrementalLearningSummary getSummary() {
            var channelSummaries = channelStates.entrySet().stream()
                .collect(java.util.stream.Collectors.toMap(
                    Map.Entry::getKey,
                    entry -> new ChannelSummary(
                        entry.getKey(),
                        entry.getValue().getCategoryCount(),
                        entry.getValue().getLearningCount(),
                        entry.getValue().getCurrentVigilance(),
                        entry.getValue().getLastLearningTime()
                    )
                ));
            
            return new IncrementalLearningSummary(
                channelStates.size(),
                totalUpdates.get(),
                channelSummaries
            );
        }
    }
    
    /**
     * Summary of channel learning state.
     */
    public record ChannelSummary(
        String channelName,
        int categoryCount,
        long learningCount,
        double currentVigilance,
        Instant lastLearningTime
    ) {}
    
    /**
     * Overall summary of incremental learning system.
     */
    public record IncrementalLearningSummary(
        int registeredChannels,
        long totalUpdates,
        Map<String, ChannelSummary> channelSummaries
    ) {}
}