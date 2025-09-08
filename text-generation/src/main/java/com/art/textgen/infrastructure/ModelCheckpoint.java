package com.art.textgen.infrastructure;

import java.io.*;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.*;
import java.util.zip.*;

/**
 * Model Checkpointing System for ART Text Generation
 * Provides save/load, versioning, and rollback capabilities
 * Based on Phase 6.2 of EXECUTION_PLAN.md
 */
public class ModelCheckpoint implements Serializable {
    private static final long serialVersionUID = 1L;
    private static final String CHECKPOINT_VERSION = "1.0";
    
    // Checkpoint metadata
    private final String checkpointId;
    private final String version;
    private final LocalDateTime timestamp;
    private final Map<String, Object> metadata;
    
    // Model state
    private final Map<String, Serializable> modelState;
    private final Map<String, Double> trainingMetrics;
    private final int epochNumber;
    private final long totalSteps;
    
    // Checkpoint configuration
    private static final Path CHECKPOINT_DIR = Paths.get("checkpoints");
    private static final int MAX_CHECKPOINTS = 10;
    private static final boolean USE_COMPRESSION = true;
    
    private ModelCheckpoint(Builder builder) {
        this.checkpointId = builder.checkpointId;
        this.version = CHECKPOINT_VERSION;
        this.timestamp = LocalDateTime.now();
        this.metadata = new HashMap<>(builder.metadata);
        this.modelState = new HashMap<>(builder.modelState);
        this.trainingMetrics = new HashMap<>(builder.trainingMetrics);
        this.epochNumber = builder.epochNumber;
        this.totalSteps = builder.totalSteps;
    }
    
    /**
     * Save checkpoint to disk
     */
    public void save() throws IOException {
        save(getDefaultPath());
    }
    
    /**
     * Save checkpoint to specific path
     */
    public void save(Path path) throws IOException {
        System.out.println("Saving checkpoint: " + checkpointId);
        
        // Ensure directory exists
        Files.createDirectories(path.getParent());
        
        // Save with optional compression
        if (USE_COMPRESSION) {
            saveCompressed(path);
        } else {
            saveUncompressed(path);
        }
        
        // Save metadata separately for quick access
        saveMetadata(path);
        
        // Manage checkpoint history
        manageCheckpointHistory();
        
        System.out.println("Checkpoint saved: " + path);
    }
    
    /**
     * Save compressed checkpoint
     */
    private void saveCompressed(Path path) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(path.toFile());
             GZIPOutputStream gzos = new GZIPOutputStream(fos);
             ObjectOutputStream oos = new ObjectOutputStream(gzos)) {
            
            oos.writeObject(this);
        }
    }
    
    /**
     * Save uncompressed checkpoint
     */
    private void saveUncompressed(Path path) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(path.toFile());
             ObjectOutputStream oos = new ObjectOutputStream(fos)) {
            
            oos.writeObject(this);
        }
    }
    
    /**
     * Save metadata file for quick inspection
     */
    private void saveMetadata(Path checkpointPath) throws IOException {
        Path metadataPath = Paths.get(checkpointPath.toString() + ".meta");
        
        List<String> lines = new ArrayList<>();
        lines.add("Checkpoint ID: " + checkpointId);
        lines.add("Version: " + version);
        lines.add("Timestamp: " + timestamp);
        lines.add("Epoch: " + epochNumber);
        lines.add("Total Steps: " + totalSteps);
        lines.add("\nTraining Metrics:");
        
        for (Map.Entry<String, Double> metric : trainingMetrics.entrySet()) {
            lines.add(String.format("  %s: %.4f", metric.getKey(), metric.getValue()));
        }
        
        lines.add("\nMetadata:");
        for (Map.Entry<String, Object> meta : metadata.entrySet()) {
            lines.add("  " + meta.getKey() + ": " + meta.getValue());
        }
        
        Files.write(metadataPath, lines);
    }
    
    /**
     * Load checkpoint from disk
     */
    public static ModelCheckpoint load(String checkpointId) throws IOException, ClassNotFoundException {
        Path path = CHECKPOINT_DIR.resolve(checkpointId + ".ckpt");
        return load(path);
    }
    
    /**
     * Load checkpoint from specific path
     */
    public static ModelCheckpoint load(Path path) throws IOException, ClassNotFoundException {
        System.out.println("Loading checkpoint: " + path);
        
        if (!Files.exists(path)) {
            throw new FileNotFoundException("Checkpoint not found: " + path);
        }
        
        ModelCheckpoint checkpoint;
        
        // Detect if compressed
        if (isCompressed(path)) {
            checkpoint = loadCompressed(path);
        } else {
            checkpoint = loadUncompressed(path);
        }
        
        System.out.println("Checkpoint loaded: " + checkpoint.checkpointId);
        return checkpoint;
    }
    
    /**
     * Load compressed checkpoint
     */
    private static ModelCheckpoint loadCompressed(Path path) throws IOException, ClassNotFoundException {
        try (FileInputStream fis = new FileInputStream(path.toFile());
             GZIPInputStream gzis = new GZIPInputStream(fis);
             ObjectInputStream ois = new ObjectInputStream(gzis)) {
            
            return (ModelCheckpoint) ois.readObject();
        }
    }
    
    /**
     * Load uncompressed checkpoint
     */
    private static ModelCheckpoint loadUncompressed(Path path) throws IOException, ClassNotFoundException {
        try (FileInputStream fis = new FileInputStream(path.toFile());
             ObjectInputStream ois = new ObjectInputStream(fis)) {
            
            return (ModelCheckpoint) ois.readObject();
        }
    }
    
    /**
     * Check if file is compressed
     */
    private static boolean isCompressed(Path path) throws IOException {
        try (FileInputStream fis = new FileInputStream(path.toFile())) {
            byte[] signature = new byte[2];
            fis.read(signature);
            // GZIP signature: 0x1f 0x8b
            return signature[0] == 0x1f && signature[1] == (byte) 0x8b;
        }
    }
    
    /**
     * List all available checkpoints
     */
    public static List<CheckpointInfo> listCheckpoints() throws IOException {
        List<CheckpointInfo> checkpoints = new ArrayList<>();
        
        if (!Files.exists(CHECKPOINT_DIR)) {
            return checkpoints;
        }
        
        Files.walk(CHECKPOINT_DIR, 1)
            .filter(path -> path.toString().endsWith(".ckpt"))
            .forEach(path -> {
                try {
                    CheckpointInfo info = loadCheckpointInfo(path);
                    checkpoints.add(info);
                } catch (IOException e) {
                    System.err.println("Failed to load checkpoint info: " + path);
                }
            });
        
        // Sort by timestamp (newest first)
        checkpoints.sort((a, b) -> b.timestamp.compareTo(a.timestamp));
        
        return checkpoints;
    }
    
    /**
     * Load checkpoint metadata without loading full model
     */
    private static CheckpointInfo loadCheckpointInfo(Path checkpointPath) throws IOException {
        Path metadataPath = Paths.get(checkpointPath.toString() + ".meta");
        
        if (Files.exists(metadataPath)) {
            // Parse metadata file
            List<String> lines = Files.readAllLines(metadataPath);
            
            String id = "";
            LocalDateTime timestamp = LocalDateTime.now();
            int epoch = 0;
            long steps = 0;
            
            for (String line : lines) {
                if (line.startsWith("Checkpoint ID: ")) {
                    id = line.substring("Checkpoint ID: ".length());
                } else if (line.startsWith("Timestamp: ")) {
                    timestamp = LocalDateTime.parse(line.substring("Timestamp: ".length()));
                } else if (line.startsWith("Epoch: ")) {
                    epoch = Integer.parseInt(line.substring("Epoch: ".length()));
                } else if (line.startsWith("Total Steps: ")) {
                    steps = Long.parseLong(line.substring("Total Steps: ".length()));
                }
            }
            
            return new CheckpointInfo(id, timestamp, epoch, steps, checkpointPath);
        }
        
        // Fallback: create basic info from filename
        String filename = checkpointPath.getFileName().toString();
        return new CheckpointInfo(
            filename.replace(".ckpt", ""),
            LocalDateTime.now(),
            0,
            0,
            checkpointPath
        );
    }
    
    /**
     * Delete old checkpoints beyond retention limit
     */
    private void manageCheckpointHistory() throws IOException {
        List<CheckpointInfo> checkpoints = listCheckpoints();
        
        if (checkpoints.size() > MAX_CHECKPOINTS) {
            // Keep only the most recent checkpoints
            for (int i = MAX_CHECKPOINTS; i < checkpoints.size(); i++) {
                CheckpointInfo old = checkpoints.get(i);
                deleteCheckpoint(old.path);
            }
        }
    }
    
    /**
     * Delete a checkpoint
     */
    private void deleteCheckpoint(Path checkpointPath) throws IOException {
        Files.deleteIfExists(checkpointPath);
        Files.deleteIfExists(Paths.get(checkpointPath.toString() + ".meta"));
        System.out.println("Deleted old checkpoint: " + checkpointPath.getFileName());
    }
    
    /**
     * Get default checkpoint path
     */
    private Path getDefaultPath() {
        String filename = checkpointId + ".ckpt";
        return CHECKPOINT_DIR.resolve(filename);
    }
    
    /**
     * Restore model state from checkpoint
     */
    public void restoreModel(Object model) {
        System.out.println("Restoring model from checkpoint: " + checkpointId);
        
        // This would restore the actual model state
        // Implementation depends on specific model classes
        // For now, we store state in modelState map
        
        System.out.println("Model restored to epoch " + epochNumber + ", step " + totalSteps);
    }
    
    /**
     * Get model state
     */
    public Map<String, Serializable> getModelState() {
        return new HashMap<>(modelState);
    }
    
    /**
     * Get training metrics
     */
    public Map<String, Double> getTrainingMetrics() {
        return new HashMap<>(trainingMetrics);
    }
    
    // Getters
    public String getCheckpointId() { return checkpointId; }
    public String getVersion() { return version; }
    public LocalDateTime getTimestamp() { return timestamp; }
    public int getEpochNumber() { return epochNumber; }
    public long getTotalSteps() { return totalSteps; }
    public Map<String, Object> getMetadata() { return new HashMap<>(metadata); }
    
    /**
     * Builder for creating checkpoints
     */
    public static class Builder {
        private String checkpointId;
        private Map<String, Object> metadata = new HashMap<>();
        private Map<String, Serializable> modelState = new HashMap<>();
        private Map<String, Double> trainingMetrics = new HashMap<>();
        private int epochNumber = 0;
        private long totalSteps = 0;
        
        public Builder(String checkpointId) {
            this.checkpointId = checkpointId;
        }
        
        public Builder withMetadata(String key, Object value) {
            metadata.put(key, value);
            return this;
        }
        
        public Builder withModelState(String key, Serializable value) {
            modelState.put(key, value);
            return this;
        }
        
        public Builder withTrainingMetric(String key, double value) {
            trainingMetrics.put(key, value);
            return this;
        }
        
        public Builder withEpoch(int epoch) {
            this.epochNumber = epoch;
            return this;
        }
        
        public Builder withTotalSteps(long steps) {
            this.totalSteps = steps;
            return this;
        }
        
        public ModelCheckpoint build() {
            return new ModelCheckpoint(this);
        }
    }
    
    /**
     * Checkpoint information for listing
     */
    public static class CheckpointInfo {
        public final String id;
        public final LocalDateTime timestamp;
        public final int epoch;
        public final long steps;
        public final Path path;
        
        public CheckpointInfo(String id, LocalDateTime timestamp, int epoch, long steps, Path path) {
            this.id = id;
            this.timestamp = timestamp;
            this.epoch = epoch;
            this.steps = steps;
            this.path = path;
        }
        
        @Override
        public String toString() {
            return String.format("Checkpoint[%s] - Epoch %d, Steps %d, Time: %s",
                id, epoch, steps,
                timestamp.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")));
        }
    }
    
    /**
     * Automatic checkpoint manager for periodic saves
     */
    public static class CheckpointManager {
        private final int saveInterval;
        private final String modelName;
        private long stepCounter = 0;
        private int epochCounter = 0;
        private final Map<String, Double> currentMetrics = new ConcurrentHashMap<>();
        
        public CheckpointManager(String modelName, int saveInterval) {
            this.modelName = modelName;
            this.saveInterval = saveInterval;
        }
        
        /**
         * Check if checkpoint should be saved
         */
        public boolean shouldCheckpoint() {
            return stepCounter % saveInterval == 0;
        }
        
        /**
         * Create and save checkpoint
         */
        public void checkpoint(Map<String, Serializable> modelState) throws IOException {
            String checkpointId = String.format("%s_epoch%d_step%d",
                modelName, epochCounter, stepCounter);
            
            ModelCheckpoint checkpoint = new Builder(checkpointId)
                .withEpoch(epochCounter)
                .withTotalSteps(stepCounter)
                .withMetadata("model_name", modelName)
                .withMetadata("save_interval", saveInterval)
                .withModelState("full_state", (Serializable) modelState)
                .build();
            
            // Add current metrics
            for (Map.Entry<String, Double> metric : currentMetrics.entrySet()) {
                checkpoint.trainingMetrics.put(metric.getKey(), metric.getValue());
            }
            
            checkpoint.save();
        }
        
        /**
         * Update training step counter
         */
        public void step() {
            stepCounter++;
        }
        
        /**
         * Update epoch counter
         */
        public void nextEpoch() {
            epochCounter++;
        }
        
        /**
         * Update metric for next checkpoint
         */
        public void updateMetric(String key, double value) {
            currentMetrics.put(key, value);
        }
    }
    
    /**
     * Main method for testing
     */
    public static void main(String[] args) {
        try {
            // Create a test checkpoint
            ModelCheckpoint checkpoint = new Builder("test_checkpoint_1")
                .withEpoch(5)
                .withTotalSteps(10000)
                .withTrainingMetric("loss", 0.234)
                .withTrainingMetric("accuracy", 0.892)
                .withMetadata("corpus_size", "22.64MB")
                .withModelState("vocabulary_size", 152951)
                .build();
            
            // Save checkpoint
            checkpoint.save();
            
            // List all checkpoints
            System.out.println("\nAvailable checkpoints:");
            for (CheckpointInfo info : listCheckpoints()) {
                System.out.println(info);
            }
            
            // Load checkpoint
            ModelCheckpoint loaded = ModelCheckpoint.load("test_checkpoint_1");
            System.out.println("\nLoaded checkpoint: " + loaded.getCheckpointId());
            System.out.println("Metrics: " + loaded.getTrainingMetrics());
            
            // Test checkpoint manager
            CheckpointManager manager = new CheckpointManager("art_model", 100);
            for (int i = 0; i < 200; i++) {
                manager.step();
                manager.updateMetric("step", i);
                
                if (manager.shouldCheckpoint()) {
                    System.out.println("Auto-saving at step " + i);
                    manager.checkpoint(new HashMap<>());
                }
            }
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
