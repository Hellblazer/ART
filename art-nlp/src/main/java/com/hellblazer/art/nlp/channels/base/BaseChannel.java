package com.hellblazer.art.nlp.channels.base;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.preprocessing.DataPreprocessor;
import com.hellblazer.art.nlp.metrics.ChannelMetrics;
import com.hellblazer.art.nlp.persistence.CategoryPersistence;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * CANONICAL BaseChannel - Abstract base class for all ART-NLP channels.
 * 
 * CRITICAL REQUIREMENTS:
 * - MUST implement classify(DenseVector input) method
 * - MUST use DenseVector (never Pattern interface)
 * - MUST be thread-safe using ReadWriteLock
 * - MUST provide metrics and persistence
 */
public abstract class BaseChannel {
    private static final Logger logger = LoggerFactory.getLogger(BaseChannel.class);
    
    protected final String channelName;
    protected final double vigilance;
    protected final DataPreprocessor preprocessor;
    protected final CategoryPersistence persistence;
    protected final ChannelMetrics metrics;
    protected final ReadWriteLock lock;
    
    // Thread-safe state
    private volatile boolean initialized = false;
    private volatile boolean learningEnabled = true;

    protected BaseChannel(String channelName, double vigilance) {
        if (channelName == null || channelName.trim().isEmpty()) {
            throw new IllegalArgumentException("Channel name cannot be null or empty");
        }
        if (vigilance < 0.0 || vigilance > 1.0) {
            throw new IllegalArgumentException("Vigilance must be in [0.0, 1.0]: " + vigilance);
        }
        
        this.channelName = channelName.trim();
        this.vigilance = vigilance;
        this.preprocessor = DataPreprocessor.builder()
            .addNormalization()
            .addComplementCoding()
            .build();
        this.persistence = new CategoryPersistence();
        this.metrics = new ChannelMetrics(this.channelName);
        this.lock = new ReentrantReadWriteLock();
        
        logger.debug("Created channel '{}' with vigilance {}", this.channelName, this.vigilance);
    }

    /**
     * CRITICAL: Main classification method - EXACT SIGNATURE REQUIRED
     * All channels must implement this method to classify input vectors.
     * 
     * @param input DenseVector to classify (NEVER use Pattern interface)
     * @return Category ID (0-based integer)
     */
    public abstract int classify(DenseVector input);

    /**
     * Process word through embeddings - SEMANTIC CHANNEL ONLY
     * Other channels should throw UnsupportedOperationException.
     * 
     * @param word Word to classify
     * @return Category ID
     * @throws UnsupportedOperationException for non-semantic channels
     */
    public int classifyWord(String word) {
        throw new UnsupportedOperationException(
            "Word classification only supported in SemanticChannel, not in " + channelName);
    }

    /**
     * Get channel performance metrics.
     * 
     * @return Current metrics snapshot
     */
    public final ChannelMetrics getMetrics() {
        return metrics;
    }

    /**
     * Save channel state to persistent storage.
     * Implementation must be thread-safe.
     */
    public abstract void saveState();

    /**
     * Load channel state from persistent storage.
     * Implementation must be thread-safe.
     */
    public abstract void loadState();
    
    /**
     * Get channel name.
     */
    public final String getChannelName() {
        return channelName;
    }
    
    /**
     * Get vigilance parameter.
     */
    public final double getVigilance() {
        return vigilance;
    }
    
    /**
     * Check if channel is initialized.
     */
    public final boolean isInitialized() {
        return initialized;
    }
    
    /**
     * Set initialized state - protected for subclasses.
     */
    protected final void setInitialized(boolean initialized) {
        this.initialized = initialized;
    }
    
    /**
     * Check if learning is enabled.
     */
    public final boolean isLearningEnabled() {
        return learningEnabled;
    }
    
    /**
     * Enable or disable learning.
     * When disabled, classify() will only match existing categories.
     */
    public final void setLearningEnabled(boolean enabled) {
        lock.writeLock().lock();
        try {
            this.learningEnabled = enabled;
            logger.debug("Learning {} for channel '{}'", enabled ? "enabled" : "disabled", channelName);
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Get current number of categories.
     * Implementation should be thread-safe.
     */
    public abstract int getCategoryCount();
    
    /**
     * Prune categories below threshold.
     * Implementation should be thread-safe.
     * 
     * @param threshold Minimum usage threshold for category retention
     * @return Number of categories pruned
     */
    public abstract int pruneCategories(double threshold);
    
    /**
     * Initialize channel (load state, setup resources).
     * Should be called before first use.
     */
    public final void initialize() {
        lock.writeLock().lock();
        try {
            if (initialized) {
                logger.debug("Channel '{}' already initialized", channelName);
                return;
            }
            
            logger.info("Initializing channel '{}'", channelName);
            
            // Attempt to load saved state
            try {
                loadState();
                logger.debug("Loaded saved state for channel '{}'", channelName);
            } catch (Exception e) {
                logger.warn("Failed to load state for channel '{}', starting fresh: {}", 
                          channelName, e.getMessage());
            }
            
            // Perform channel-specific initialization
            performInitialization();
            
            setInitialized(true);
            logger.info("Channel '{}' initialized successfully", channelName);
            
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Shutdown channel (save state, cleanup resources).
     * Should be called when channel is no longer needed.
     */
    public final void shutdown() {
        lock.writeLock().lock();
        try {
            if (!initialized) {
                logger.debug("Channel '{}' not initialized, nothing to shutdown", channelName);
                return;
            }
            
            logger.info("Shutting down channel '{}'", channelName);
            
            // Save current state
            try {
                saveState();
                logger.debug("Saved state for channel '{}'", channelName);
            } catch (Exception e) {
                logger.error("Failed to save state for channel '{}': {}", channelName, e.getMessage());
            }
            
            // Perform channel-specific cleanup
            performCleanup();
            
            setInitialized(false);
            logger.info("Channel '{}' shutdown complete", channelName);
            
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Perform channel-specific initialization.
     * Called during initialize() while holding write lock.
     */
    protected abstract void performInitialization();
    
    /**
     * Perform channel-specific cleanup.
     * Called during shutdown() while holding write lock.
     */
    protected abstract void performCleanup();
    
    /**
     * Preprocess input vector for this channel.
     * Uses configured data preprocessor with complement coding.
     * 
     * @param input Input vector
     * @return Preprocessed vector ready for ART algorithm
     */
    protected final DenseVector preprocessInput(DenseVector input) {
        if (input == null) {
            throw new IllegalArgumentException("Input cannot be null");
        }
        
        // Apply preprocessing pipeline (normalization + complement coding)
        var patterns = preprocessor.transform(new com.hellblazer.art.core.Pattern[]{input});
        return (DenseVector) patterns[0];
    }
    
    /**
     * Record successful classification in metrics.
     * 
     * @param processingTimeMs Time taken for classification
     * @param categoryCreated Whether a new category was created
     */
    protected final void recordClassification(long processingTimeMs, boolean categoryCreated) {
        metrics.recordClassification(processingTimeMs);
        if (categoryCreated) {
            metrics.recordCategoryCreated();
        }
    }
    
    /**
     * Record error in metrics.
     */
    protected final void recordError() {
        metrics.recordError();
    }
    
    /**
     * Get read lock for thread-safe read operations.
     */
    protected final java.util.concurrent.locks.Lock getReadLock() {
        return lock.readLock();
    }
    
    /**
     * Get write lock for thread-safe write operations.
     */
    protected final java.util.concurrent.locks.Lock getWriteLock() {
        return lock.writeLock();
    }

    @Override
    public final String toString() {
        return String.format("%s{name='%s', vigilance=%.3f, categories=%d, initialized=%b}",
                           getClass().getSimpleName(), channelName, vigilance, 
                           getCategoryCount(), initialized);
    }
}