package com.hellblazer.art.hartcq.hierarchical;

import com.hellblazer.art.core.BaseART;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.algorithms.FuzzyART;
import com.hellblazer.art.core.artmap.DeepARTMAP;
import com.hellblazer.art.core.artmap.DeepARTMAPParameters;
import com.hellblazer.art.core.artmap.DeepARTMAPResult;
import com.hellblazer.art.core.parameters.FuzzyParameters;
import com.hellblazer.art.hartcq.core.MultiChannelProcessor;
import com.hellblazer.art.hartcq.Token;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Hierarchical ART processor using DeepARTMAP from art-core.
 * Integrates multi-channel processing with hierarchical ART learning.
 */
public class HierarchicalProcessor {
    private static final Logger logger = LoggerFactory.getLogger(HierarchicalProcessor.class);
    
    private final MultiChannelProcessor channelProcessor;
    private final DeepARTMAP deepARTMAP;
    private final AtomicInteger patternCounter;
    private final ConcurrentHashMap<Integer, String> categoryLabels;
    private boolean supervisedMode = false;
    
    // Hierarchical levels
    private static final int NUM_LEVELS = 3;
    private static final double[] LEVEL_VIGILANCE = {0.7, 0.8, 0.9}; // Increasing vigilance
    
    public HierarchicalProcessor() {
        this.channelProcessor = new MultiChannelProcessor();
        this.deepARTMAP = createDeepARTMAP();
        this.patternCounter = new AtomicInteger(0);
        this.categoryLabels = new ConcurrentHashMap<>();
        
        logger.info("Initialized HierarchicalProcessor with {} levels", NUM_LEVELS);
    }
    
    /**
     * Creates and configures DeepARTMAP for hierarchical processing.
     */
    private DeepARTMAP createDeepARTMAP() {
        // Configure hierarchical parameters
        var params = new DeepARTMAPParameters();
        
        // Create FuzzyART modules for each hierarchical level
        var modules = new ArrayList<BaseART>();
        for (int i = 0; i < NUM_LEVELS; i++) {
            // FuzzyART uses parameters passed during fit, not constructor
            modules.add(new FuzzyART());
        }
        
        // Create DeepARTMAP with modules and parameters
        return new DeepARTMAP(modules, params);
    }
    
    /**
     * Processes a token window through channels and hierarchical ART.
     * @param tokens The token window
     * @return Hierarchical processing result
     */
    public HierarchicalResult processWindow(Token[] tokens) {
        // Validate input
        if (tokens == null) {
            throw new IllegalArgumentException("Token array cannot be null");
        }
        if (tokens.length == 0) {
            throw new IllegalArgumentException("Token array cannot be empty");
        }

        // Process through multi-channel architecture
        var channelOutput = channelProcessor.processWindow(tokens);

        // Convert to Pattern for ART processing
        var patterns = createPatternsForLevels(channelOutput);

        // Process through DeepARTMAP hierarchy (unsupervised only if not in supervised mode)
        if (supervisedMode || deepARTMAP.isTrained()) {
            // In supervised mode or if already trained, use predict instead of fitUnsupervised
            if (!deepARTMAP.isTrained()) {
                // Not trained yet, return empty result
                var result = new HierarchicalResult();
                result.setPatternId(patternCounter.incrementAndGet());
                result.setChannelFeatures(channelOutput);
                result.setHierarchicalCategories(new int[]{0, 0, 0}); // 3 levels
                result.setVigilanceLevels(LEVEL_VIGILANCE);
                return result;
            }
            var predictions = deepARTMAP.predictDeep(patterns);
            // Create result from predictions
            var result = new HierarchicalResult();
            result.setPatternId(patternCounter.incrementAndGet());
            result.setChannelFeatures(channelOutput);

            // Ensure we have 3 levels of categories
            int[] categories = new int[NUM_LEVELS];
            if (predictions.length > 0 && predictions[0] != null) {
                // Copy available predictions
                System.arraycopy(predictions[0], 0, categories, 0, Math.min(predictions[0].length, NUM_LEVELS));
                // Fill remaining levels with defaults if needed
                for (int i = predictions[0].length; i < NUM_LEVELS; i++) {
                    categories[i] = 0;
                }
            }

            result.setHierarchicalCategories(categories);
            result.setVigilanceLevels(LEVEL_VIGILANCE);
            return result;
        } else {
            // Unsupervised mode
            var artResult = deepARTMAP.fitUnsupervised(patterns);

            // Create hierarchical result
            var result = new HierarchicalResult();
            result.setPatternId(patternCounter.incrementAndGet());
            result.setChannelFeatures(channelOutput);
            result.setHierarchicalCategories(extractCategories(artResult));
            result.setVigilanceLevels(LEVEL_VIGILANCE);

            return result;
        }
    }
    
    /**
     * Trains the hierarchical system with labeled data.
     * @param tokens Token window
     * @param label Category label
     */
    public void train(Token[] tokens, String label) {
        // If switching from unsupervised to supervised mode, clear DeepARTMAP
        if (!supervisedMode) {
            deepARTMAP.clearDeepARTMAP();
            supervisedMode = true;  // Set to supervised mode
        }

        var channelOutput = channelProcessor.processWindow(tokens);
        var patterns = createPatternsForLevels(channelOutput);

        // Create labels array for supervised training
        int labelId = categoryLabels.size();
        int[] labels = new int[]{labelId};

        // Train DeepARTMAP using supervised learning
        var result = deepARTMAP.fitSupervised(patterns, labels);

        // Store category label mapping
        if (result instanceof DeepARTMAPResult.Success success) {
            categoryLabels.put(labelId, label);
            patternCounter.incrementAndGet(); // Increment pattern counter for training
            logger.debug("Trained pattern with label: {} (id: {})", label, labelId);
        } else {
            logger.warn("Training failed for label: {}", label);
        }
    }
    
    /**
     * Predicts the category for a token window.
     * @param tokens Token window
     * @return Predicted label
     */
    public String predict(Token[] tokens) {
        // If in supervised mode or DeepARTMAP is trained, use predict
        if (supervisedMode || deepARTMAP.isTrained()) {
            var channelOutput = channelProcessor.processWindow(tokens);
            var patterns = createPatternsForLevels(channelOutput);

            // If trained, use predict; otherwise need to handle untrained case
            if (deepARTMAP.isTrained()) {
                var predictions = deepARTMAP.predict(patterns);
                if (predictions.length > 0) {
                    return categoryLabels.getOrDefault(predictions[0], "UNKNOWN");
                }
            } else {
                // Not yet trained, return default
                return "UNKNOWN";
            }
        } else {
            // Unsupervised mode - but check if DeepARTMAP is trained
            if (deepARTMAP.isTrained()) {
                // DeepARTMAP is trained, can't call processWindow which uses fitUnsupervised
                // Use predict directly
                var channelOutput = channelProcessor.processWindow(tokens);
                var patterns = createPatternsForLevels(channelOutput);
                var predictions = deepARTMAP.predict(patterns);
                if (predictions.length > 0) {
                    return categoryLabels.getOrDefault(predictions[0], "UNKNOWN");
                }
                return "UNKNOWN";
            } else {
                // Safe to use processWindow for unsupervised
                var result = processWindow(tokens);
                var topCategory = result.getTopCategory();
                return categoryLabels.getOrDefault(topCategory, "UNKNOWN");
            }
        }
        return "UNKNOWN";
    }
    
    /**
     * Creates a Pattern from channel output features.
     */
    private Pattern createPattern(float[] features) {
        // Convert float array to double array for Pattern
        var doubleFeatures = new double[features.length];
        for (int i = 0; i < features.length; i++) {
            doubleFeatures[i] = features[i];
        }
        
        return new DenseVector(doubleFeatures);
    }
    
    /**
     * Creates patterns for each hierarchical level.
     */
    private List<Pattern[]> createPatternsForLevels(float[] features) {
        var patterns = new ArrayList<Pattern[]>();
        var pattern = createPattern(features);
        
        // Create pattern array for each level
        for (int i = 0; i < NUM_LEVELS; i++) {
            patterns.add(new Pattern[]{pattern});
        }
        
        return patterns;
    }
    
    /**
     * Extracts category information from DeepARTMAP result.
     */
    private int[] extractCategories(DeepARTMAPResult artResult) {
        // Extract categories from result if successful
        if (artResult instanceof DeepARTMAPResult.Success success) {
            // Extract deep labels from the successful result
            var deepLabels = success.deepLabels();
            if (deepLabels != null && deepLabels.length > 0 && deepLabels[0] != null) {
                // Ensure we always return NUM_LEVELS categories
                int[] categories = new int[NUM_LEVELS];
                System.arraycopy(deepLabels[0], 0, categories, 0, Math.min(deepLabels[0].length, NUM_LEVELS));
                // Fill remaining levels with defaults if needed
                for (int i = deepLabels[0].length; i < NUM_LEVELS; i++) {
                    categories[i] = 0;
                }
                return categories;
            }
            return new int[]{0, 0, 0};
        }
        return new int[]{-1, -1, -1};
    }
    
    /**
     * Resets the hierarchical processor.
     */
    public void reset() {
        channelProcessor.resetChannels();
        // DeepARTMAP doesn't have a reset method, so recreate it
        // Note: This is a workaround - in production, we might want to keep state
        patternCounter.set(0);
        categoryLabels.clear();
        supervisedMode = false;  // Reset supervised mode flag
        deepARTMAP.clearDeepARTMAP();  // Clear DeepARTMAP state
        logger.info("Hierarchical processor reset");
    }
    
    /**
     * Gets statistics about the hierarchical processing.
     * @return Processing statistics
     */
    public HierarchicalStats getStats() {
        var stats = new HierarchicalStats();
        stats.setPatternsProcessed(patternCounter.get());
        stats.setNumCategories(categoryLabels.size());
        stats.setNumLevels(NUM_LEVELS);
        stats.setChannelDimension(channelProcessor.getTotalOutputDimension());
        
        return stats;
    }
    
    /**
     * Result of hierarchical processing.
     */
    public static class HierarchicalResult {
        private int patternId;
        private float[] channelFeatures;
        private int[] hierarchicalCategories;
        private double[] vigilanceLevels;
        
        public int getPatternId() {
            return patternId;
        }
        
        public void setPatternId(int patternId) {
            this.patternId = patternId;
        }
        
        public float[] getChannelFeatures() {
            return channelFeatures;
        }
        
        public void setChannelFeatures(float[] channelFeatures) {
            this.channelFeatures = channelFeatures;
        }
        
        public int[] getHierarchicalCategories() {
            return hierarchicalCategories;
        }
        
        public void setHierarchicalCategories(int[] hierarchicalCategories) {
            this.hierarchicalCategories = hierarchicalCategories;
        }
        
        public double[] getVigilanceLevels() {
            return vigilanceLevels;
        }
        
        public void setVigilanceLevels(double[] vigilanceLevels) {
            this.vigilanceLevels = vigilanceLevels;
        }
        
        public int getTopCategory() {
            return hierarchicalCategories != null && hierarchicalCategories.length > 0 
                ? hierarchicalCategories[0] : -1;
        }
    }
    
    /**
     * Statistics about hierarchical processing.
     */
    public static class HierarchicalStats {
        private int patternsProcessed;
        private int numCategories;
        private int numLevels;
        private int channelDimension;
        
        public int getPatternsProcessed() {
            return patternsProcessed;
        }
        
        public void setPatternsProcessed(int patternsProcessed) {
            this.patternsProcessed = patternsProcessed;
        }
        
        public int getNumCategories() {
            return numCategories;
        }
        
        public void setNumCategories(int numCategories) {
            this.numCategories = numCategories;
        }
        
        public int getNumLevels() {
            return numLevels;
        }
        
        public void setNumLevels(int numLevels) {
            this.numLevels = numLevels;
        }
        
        public int getChannelDimension() {
            return channelDimension;
        }
        
        public void setChannelDimension(int channelDimension) {
            this.channelDimension = channelDimension;
        }
    }
}