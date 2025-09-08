package com.hellblazer.art.nlp.persistence;

import com.hellblazer.art.core.DenseVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Thread-safe category persistence for ART channels.
 * Handles saving/loading of category weight vectors and metadata.
 */
public final class CategoryPersistence {
    private static final Logger logger = LoggerFactory.getLogger(CategoryPersistence.class);
    private static final String CATEGORIES_DIR = "models/categories";
    private static final String CATEGORY_EXTENSION = ".dat";
    private static final String METADATA_EXTENSION = ".meta";
    
    private final ReadWriteLock lock = new ReentrantReadWriteLock();
    
    /**
     * Save channel categories to disk.
     * Format: [count][weight_vector_1][weight_vector_2]...
     */
    public void saveCategories(String channelName, List<DenseVector> categories) {
        if (channelName == null || channelName.trim().isEmpty()) {
            throw new IllegalArgumentException("Channel name cannot be null or empty");
        }
        if (categories == null) {
            throw new IllegalArgumentException("Categories list cannot be null");
        }
        
        lock.writeLock().lock();
        try {
            ensureDirectoryExists();
            var filename = getCategoryFileName(channelName);
            
            try (var oos = new ObjectOutputStream(new BufferedOutputStream(
                    new FileOutputStream(filename)))) {
                
                // Save category count
                oos.writeInt(categories.size());
                
                // Save each weight vector
                for (var vector : categories) {
                    oos.writeObject(vector.values());
                }
                
                oos.flush();
                logger.info("Saved {} categories for channel '{}'", categories.size(), channelName);
                
            } catch (IOException e) {
                logger.error("Failed to save categories for channel '{}'", channelName, e);
                throw new RuntimeException("Failed to save categories for channel: " + channelName, e);
            }
            
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Load channel categories from disk.
     */
    public List<DenseVector> loadCategories(String channelName) {
        if (channelName == null || channelName.trim().isEmpty()) {
            throw new IllegalArgumentException("Channel name cannot be null or empty");
        }
        
        lock.readLock().lock();
        try {
            var filename = getCategoryFileName(channelName);
            var file = new File(filename);
            
            if (!file.exists()) {
                logger.debug("No saved categories found for channel '{}'", channelName);
                return new ArrayList<>();
            }
            
            try (var ois = new ObjectInputStream(new BufferedInputStream(
                    new FileInputStream(filename)))) {
                
                var numCategories = ois.readInt();
                var categories = new ArrayList<DenseVector>(numCategories);
                
                for (int i = 0; i < numCategories; i++) {
                    var weights = (double[]) ois.readObject();
                    categories.add(new DenseVector(weights));
                }
                
                logger.info("Loaded {} categories for channel '{}'", numCategories, channelName);
                return categories;
                
            } catch (IOException | ClassNotFoundException e) {
                logger.error("Failed to load categories for channel '{}'", channelName, e);
                throw new RuntimeException("Failed to load categories for channel: " + channelName, e);
            }
            
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Save channel metadata (vigilance, learning rate, etc.).
     */
    public void saveMetadata(String channelName, ChannelMetadata metadata) {
        if (channelName == null || channelName.trim().isEmpty()) {
            throw new IllegalArgumentException("Channel name cannot be null or empty");
        }
        if (metadata == null) {
            throw new IllegalArgumentException("Metadata cannot be null");
        }
        
        lock.writeLock().lock();
        try {
            ensureDirectoryExists();
            var filename = getMetadataFileName(channelName);
            
            try (var oos = new ObjectOutputStream(new BufferedOutputStream(
                    new FileOutputStream(filename)))) {
                
                oos.writeObject(metadata);
                oos.flush();
                logger.debug("Saved metadata for channel '{}'", channelName);
                
            } catch (IOException e) {
                logger.error("Failed to save metadata for channel '{}'", channelName, e);
                throw new RuntimeException("Failed to save metadata for channel: " + channelName, e);
            }
            
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    /**
     * Load channel metadata.
     */
    public ChannelMetadata loadMetadata(String channelName) {
        if (channelName == null || channelName.trim().isEmpty()) {
            throw new IllegalArgumentException("Channel name cannot be null or empty");
        }
        
        lock.readLock().lock();
        try {
            var filename = getMetadataFileName(channelName);
            var file = new File(filename);
            
            if (!file.exists()) {
                logger.debug("No saved metadata found for channel '{}'", channelName);
                return null;
            }
            
            try (var ois = new ObjectInputStream(new BufferedInputStream(
                    new FileInputStream(filename)))) {
                
                var metadata = (ChannelMetadata) ois.readObject();
                logger.debug("Loaded metadata for channel '{}'", channelName);
                return metadata;
                
            } catch (IOException | ClassNotFoundException e) {
                logger.error("Failed to load metadata for channel '{}'", channelName, e);
                throw new RuntimeException("Failed to load metadata for channel: " + channelName, e);
            }
            
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Check if saved state exists for channel.
     */
    public boolean hasState(String channelName) {
        if (channelName == null || channelName.trim().isEmpty()) {
            return false;
        }
        
        lock.readLock().lock();
        try {
            var categoryFile = new File(getCategoryFileName(channelName));
            return categoryFile.exists();
        } finally {
            lock.readLock().unlock();
        }
    }
    
    /**
     * Delete all saved state for channel.
     */
    public void deleteState(String channelName) {
        if (channelName == null || channelName.trim().isEmpty()) {
            throw new IllegalArgumentException("Channel name cannot be null or empty");
        }
        
        lock.writeLock().lock();
        try {
            var categoryFile = new File(getCategoryFileName(channelName));
            var metadataFile = new File(getMetadataFileName(channelName));
            
            boolean deletedCategory = !categoryFile.exists() || categoryFile.delete();
            boolean deletedMetadata = !metadataFile.exists() || metadataFile.delete();
            
            if (deletedCategory && deletedMetadata) {
                logger.info("Deleted saved state for channel '{}'", channelName);
            } else {
                logger.warn("Failed to fully delete state for channel '{}'", channelName);
            }
            
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    private void ensureDirectoryExists() {
        try {
            var path = Paths.get(CATEGORIES_DIR);
            Files.createDirectories(path);
        } catch (IOException e) {
            throw new RuntimeException("Failed to create categories directory", e);
        }
    }
    
    private String getCategoryFileName(String channelName) {
        return CATEGORIES_DIR + "/" + channelName + CATEGORY_EXTENSION;
    }
    
    private String getMetadataFileName(String channelName) {
        return CATEGORIES_DIR + "/" + channelName + METADATA_EXTENSION;
    }
    
    /**
     * Serializable metadata for channel persistence.
     */
    public static class ChannelMetadata implements Serializable {
        private static final long serialVersionUID = 1L;
        
        public final double vigilance;
        public final double learningRate;
        public final String algorithmType;
        public final long saveTimestamp;
        public final int version;
        
        public ChannelMetadata(double vigilance, double learningRate, String algorithmType) {
            this.vigilance = vigilance;
            this.learningRate = learningRate;
            this.algorithmType = algorithmType;
            this.saveTimestamp = System.currentTimeMillis();
            this.version = 1;
        }
        
        @Override
        public String toString() {
            return String.format("ChannelMetadata{vigilance=%.3f, learningRate=%.3f, type='%s', version=%d}",
                               vigilance, learningRate, algorithmType, version);
        }
    }
}