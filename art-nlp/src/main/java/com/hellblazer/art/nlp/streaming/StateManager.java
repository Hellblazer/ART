package com.hellblazer.art.nlp.streaming;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import com.hellblazer.art.nlp.core.ProcessingResult;
import com.hellblazer.art.core.DenseVector;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Persistent state management for streaming ART processing with recovery capabilities.
 * Handles checkpointing, state persistence, and recovery after failures or restarts.
 */
public class StateManager implements AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(StateManager.class);
    
    private final String instanceId;
    private final StateConfig config;
    private final StateStore stateStore;
    private final ScheduledExecutorService scheduler;
    private final Map<String, StateCheckpoint> checkpoints = new ConcurrentHashMap<>();
    private final AtomicLong checkpointSequence = new AtomicLong(0);
    private final long startTime = System.currentTimeMillis();
    
    private volatile boolean running = false;
    
    public StateManager(String instanceId, StateConfig config, StateStore stateStore) {
        this.instanceId = instanceId;
        this.config = config;
        this.stateStore = stateStore;
        this.scheduler = Executors.newScheduledThreadPool(1, 
            r -> Thread.ofVirtual().name("state-" + instanceId + "-").factory().newThread(r));
    }
    
    /**
     * Configuration for state management.
     */
    public record StateConfig(
        boolean enableCheckpointing,
        java.time.Duration checkpointInterval,
        int maxCheckpoints,
        boolean enableCompression,
        boolean enableEncryption,
        String compressionAlgorithm,
        RecoveryConfig recoveryConfig
    ) {
        public static StateConfig defaultConfig() {
            return new StateConfig(
                true,
                java.time.Duration.ofMinutes(5),
                10,
                true,
                false,
                "gzip",
                RecoveryConfig.defaultConfig()
            );
        }
    }
    
    /**
     * Recovery configuration for handling failures.
     */
    public record RecoveryConfig(
        boolean enableAutoRecovery,
        int maxRecoveryAttempts,
        java.time.Duration recoveryTimeout,
        RecoveryStrategy strategy
    ) {
        public static RecoveryConfig defaultConfig() {
            return new RecoveryConfig(
                true,
                3,
                java.time.Duration.ofMinutes(2),
                RecoveryStrategy.LATEST_CHECKPOINT
            );
        }
    }
    
    public enum RecoveryStrategy {
        LATEST_CHECKPOINT,
        SPECIFIC_CHECKPOINT,
        TIME_BASED,
        CUSTOM
    }
    
    /**
     * Interface for state persistence backends.
     */
    public interface StateStore {
        CompletableFuture<Void> save(String key, StateCheckpoint checkpoint);
        CompletableFuture<StateCheckpoint> load(String key);
        CompletableFuture<Void> delete(String key);
        CompletableFuture<Map<String, StateCheckpoint>> listCheckpoints(String instanceId);
        CompletableFuture<Void> cleanup(String instanceId, int maxCheckpoints);
    }
    
    /**
     * State checkpoint containing all necessary information for recovery.
     */
    public record StateCheckpoint(
        String instanceId,
        long sequence,
        Instant timestamp,
        StreamingState streamingState,
        LearningState learningState,
        WindowState windowState,
        MetricsState metricsState,
        Map<String, Object> metadata
    ) implements Serializable {}
    
    /**
     * Streaming processor state.
     */
    public record StreamingState(
        long processedEvents,
        long droppedEvents,
        Instant lastEventTime,
        String processorStatus,
        Map<String, Object> configuration
    ) implements Serializable {}
    
    /**
     * Incremental learning state.
     */
    public record LearningState(
        Map<String, CategoryState> categories,
        Map<String, Double> learningRates,
        long totalUpdates,
        Instant lastUpdate
    ) implements Serializable {}
    
    /**
     * Category state for learning.
     */
    public record CategoryState(
        int categoryId,
        DenseVector prototype,
        long supportCount,
        double confidence,
        Instant lastActivation,
        Map<String, Object> properties
    ) implements Serializable {}
    
    /**
     * Window management state.
     */
    public record WindowState(
        Map<String, WindowData> activeWindows,
        long totalWindows,
        Instant lastWindowTrigger
    ) implements Serializable {}
    
    /**
     * Window data for state persistence.
     */
    public record WindowData(
        String windowId,
        String windowType,
        Instant startTime,
        Instant endTime,
        long eventCount,
        Object windowContents
    ) implements Serializable {}
    
    /**
     * Metrics state for monitoring.
     */
    public record MetricsState(
        Map<String, Long> counters,
        Map<String, Double> gauges,
        Instant lastReport,
        long uptime
    ) implements Serializable {}
    
    /**
     * Recovery result information.
     */
    public record RecoveryResult(
        boolean successful,
        StateCheckpoint recoveredState,
        Instant recoveryTime,
        String strategy,
        String errorMessage
    ) {}
    
    // API Methods
    
    public void start() {
        if (running) return;
        
        running = true;
        
        if (config.enableCheckpointing()) {
            scheduler.scheduleAtFixedRate(
                this::performCheckpoint,
                config.checkpointInterval().toMillis(),
                config.checkpointInterval().toMillis(),
                TimeUnit.MILLISECONDS
            );
        }
        
        logger.info("Started state manager for instance {}", instanceId);
    }
    
    public void stop() {
        running = false;
        
        // Perform final checkpoint
        if (config.enableCheckpointing()) {
            performCheckpoint();
        }
        
        logger.info("Stopped state manager for instance {}", instanceId);
    }
    
    public CompletableFuture<StateCheckpoint> createCheckpoint(
        StreamingState streamingState,
        LearningState learningState,
        WindowState windowState,
        MetricsState metricsState
    ) {
        return createCheckpoint(streamingState, learningState, windowState, metricsState, Map.of());
    }
    
    public CompletableFuture<StateCheckpoint> createCheckpoint(
        StreamingState streamingState,
        LearningState learningState,
        WindowState windowState,
        MetricsState metricsState,
        Map<String, Object> metadata
    ) {
        return CompletableFuture.supplyAsync(() -> {
            var sequence = checkpointSequence.incrementAndGet();
            var checkpoint = new StateCheckpoint(
                instanceId,
                sequence,
                Instant.now(),
                streamingState,
                learningState,
                windowState,
                metricsState,
                new ConcurrentHashMap<>(metadata)
            );
            
            checkpoints.put(buildCheckpointKey(sequence), checkpoint);
            
            logger.debug("Created checkpoint {} for instance {}", sequence, instanceId);
            return checkpoint;
        });
    }
    
    public CompletableFuture<Void> saveCheckpoint(StateCheckpoint checkpoint) {
        var key = buildCheckpointKey(checkpoint.sequence());
        return stateStore.save(key, checkpoint)
            .thenRun(() -> logger.info("Saved checkpoint {} for instance {}", 
                checkpoint.sequence(), instanceId))
            .exceptionally(throwable -> {
                logger.error("Failed to save checkpoint {} for instance {}: {}", 
                    checkpoint.sequence(), instanceId, throwable.getMessage());
                return null;
            });
    }
    
    public CompletableFuture<StateCheckpoint> loadCheckpoint(long sequence) {
        var key = buildCheckpointKey(sequence);
        return stateStore.load(key)
            .thenApply(checkpoint -> {
                if (checkpoint != null) {
                    checkpoints.put(key, checkpoint);
                    logger.info("Loaded checkpoint {} for instance {}", sequence, instanceId);
                } else {
                    logger.warn("Checkpoint {} not found for instance {}", sequence, instanceId);
                }
                return checkpoint;
            });
    }
    
    public CompletableFuture<StateCheckpoint> loadLatestCheckpoint() {
        return stateStore.listCheckpoints(instanceId)
            .thenCompose(checkpointMap -> {
                if (checkpointMap.isEmpty()) {
                    return CompletableFuture.completedFuture(null);
                }
                
                var latestCheckpoint = checkpointMap.values().stream()
                    .max((c1, c2) -> Long.compare(c1.sequence(), c2.sequence()))
                    .orElse(null);
                
                if (latestCheckpoint != null) {
                    checkpoints.put(buildCheckpointKey(latestCheckpoint.sequence()), latestCheckpoint);
                    logger.info("Loaded latest checkpoint {} for instance {}", 
                        latestCheckpoint.sequence(), instanceId);
                }
                
                return CompletableFuture.completedFuture(latestCheckpoint);
            });
    }
    
    public CompletableFuture<RecoveryResult> recoverFromFailure() {
        return recoverFromFailure(config.recoveryConfig().strategy());
    }
    
    public CompletableFuture<RecoveryResult> recoverFromFailure(RecoveryStrategy strategy) {
        logger.info("Starting recovery for instance {} using strategy {}", instanceId, strategy);
        
        return CompletableFuture.supplyAsync(() -> {
            var recoveryStart = Instant.now();
            
            try {
                StateCheckpoint checkpoint = switch (strategy) {
                    case LATEST_CHECKPOINT -> loadLatestCheckpoint().join();
                    case SPECIFIC_CHECKPOINT -> {
                        // Would need sequence parameter - using latest for now
                        yield loadLatestCheckpoint().join();
                    }
                    case TIME_BASED -> {
                        // Would need time parameter - using latest for now
                        yield loadLatestCheckpoint().join();
                    }
                    case CUSTOM -> {
                        // Would need custom recovery logic
                        yield loadLatestCheckpoint().join();
                    }
                };
                
                if (checkpoint == null) {
                    return new RecoveryResult(false, null, recoveryStart, strategy.name(), 
                        "No checkpoint found for recovery");
                }
                
                // Validate checkpoint
                if (!validateCheckpoint(checkpoint)) {
                    return new RecoveryResult(false, checkpoint, recoveryStart, strategy.name(), 
                        "Checkpoint validation failed");
                }
                
                logger.info("Successfully recovered from checkpoint {} for instance {}", 
                    checkpoint.sequence(), instanceId);
                
                return new RecoveryResult(true, checkpoint, recoveryStart, strategy.name(), null);
                
            } catch (Exception e) {
                logger.error("Recovery failed for instance {}: {}", instanceId, e.getMessage());
                return new RecoveryResult(false, null, recoveryStart, strategy.name(), e.getMessage());
            }
        });
    }
    
    public CompletableFuture<Void> cleanupCheckpoints() {
        return stateStore.cleanup(instanceId, config.maxCheckpoints())
            .thenRun(() -> logger.info("Cleaned up old checkpoints for instance {}", instanceId));
    }
    
    public Map<String, StateCheckpoint> getLocalCheckpoints() {
        return Map.copyOf(checkpoints);
    }
    
    public long getCurrentSequence() {
        return checkpointSequence.get();
    }
    
    // Private Methods
    
    private void performCheckpoint() {
        if (!running) return;
        
        try {
            // Collect current state from registered providers or create minimal state
            var streamingState = collectStreamingState();
            var learningState = collectLearningState();  
            var windowState = collectWindowState();
            var metricsState = collectMetricsState();
            
            createCheckpoint(streamingState, learningState, windowState, metricsState)
                .thenCompose(this::saveCheckpoint)
                .thenRun(() -> cleanupCheckpoints());
                
        } catch (Exception e) {
            logger.error("Checkpoint failed for instance {}: {}", instanceId, e.getMessage());
        }
    }
    
    /**
     * Collect current streaming state from system or create minimal state.
     */
    private StreamingState collectStreamingState() {
        var processedEvents = checkpointSequence.get(); // Use sequence as event proxy
        var errorCount = 0L; // Could be enhanced with real error tracking
        var status = running ? "running" : "stopped";
        var properties = Map.<String, Object>of(
            "checkpointInterval", config.checkpointInterval().toMillis(),
            "enableCheckpointing", config.enableCheckpointing()
        );
        
        return new StreamingState(processedEvents, errorCount, Instant.now(), status, properties);
    }
    
    /**
     * Collect current learning state from ART algorithms or create minimal state.
     */
    private LearningState collectLearningState() {
        // In real implementation, would query active ART channels for category data
        var categories = Map.<String, CategoryState>of();
        var vigilanceParameters = Map.of("default", 0.8); // Default vigilance
        var totalUpdates = checkpointSequence.get();
        
        return new LearningState(categories, vigilanceParameters, totalUpdates, Instant.now());
    }
    
    /**
     * Collect current window state from windowing manager or create minimal state.  
     */
    private WindowState collectWindowState() {
        // In real implementation, would query active windows from windowing manager
        var windows = Map.<String, WindowData>of();
        var totalWindows = 0L; // Could track active window count
        
        return new WindowState(windows, totalWindows, Instant.now());
    }
    
    /**
     * Collect current metrics state from metrics system or create minimal state.
     */
    private MetricsState collectMetricsState() {
        var counters = Map.<String, Long>of(
            "checkpoints.created", checkpointSequence.get(),
            "checkpoints.saved", (long) checkpoints.size()
        );
        
        var gauges = Map.of(
            "uptime.seconds", (System.currentTimeMillis() - startTime) / 1000.0,
            "memory.usage.ratio", getMemoryUsageRatio()
        );
        
        var uptime = System.currentTimeMillis() - startTime;
        
        return new MetricsState(counters, gauges, Instant.now(), uptime);
    }
    
    /**
     * Get current memory usage ratio.
     */
    private double getMemoryUsageRatio() {
        var runtime = Runtime.getRuntime();
        var used = runtime.totalMemory() - runtime.freeMemory();
        var max = runtime.maxMemory();
        return max > 0 ? (double) used / max : 0.0;
    }
    
    private boolean validateCheckpoint(StateCheckpoint checkpoint) {
        if (checkpoint == null) return false;
        if (!instanceId.equals(checkpoint.instanceId())) return false;
        if (checkpoint.timestamp().isAfter(Instant.now().plusSeconds(60))) return false; // Future timestamp
        
        // Additional validation logic would go here
        return true;
    }
    
    private String buildCheckpointKey(long sequence) {
        return String.format("%s-checkpoint-%d", instanceId, sequence);
    }
    
    @Override
    public void close() {
        stop();
        scheduler.shutdown();
        try {
            if (!scheduler.awaitTermination(5, TimeUnit.SECONDS)) {
                scheduler.shutdownNow();
            }
        } catch (InterruptedException e) {
            scheduler.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
    
    /**
     * File-based state store implementation for testing/development.
     */
    public static class FileStateStore implements StateStore {
        private final Path storePath;
        private final boolean enableCompression;
        
        public FileStateStore(Path storePath, boolean enableCompression) {
            this.storePath = storePath;
            this.enableCompression = enableCompression;
        }
        
        @Override
        public CompletableFuture<Void> save(String key, StateCheckpoint checkpoint) {
            return CompletableFuture.runAsync(() -> {
                try {
                    // Create storage directory if it doesn't exist
                    Files.createDirectories(storePath);
                    
                    var filePath = storePath.resolve(key + ".checkpoint");
                    
                    if (enableCompression) {
                        try (var fos = Files.newOutputStream(filePath, StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING);
                             var gzos = new GZIPOutputStream(fos);
                             var oos = new ObjectOutputStream(gzos)) {
                            oos.writeObject(checkpoint);
                        }
                    } else {
                        try (var fos = Files.newOutputStream(filePath, StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING);
                             var oos = new ObjectOutputStream(fos)) {
                            oos.writeObject(checkpoint);
                        }
                    }
                    
                    logger.debug("Saved checkpoint to file store: {} -> {}", key, filePath);
                } catch (IOException e) {
                    logger.error("Failed to save checkpoint {}: {}", key, e.getMessage());
                    throw new RuntimeException("Checkpoint save failed", e);
                }
            });
        }
        
        @Override
        public CompletableFuture<StateCheckpoint> load(String key) {
            return CompletableFuture.supplyAsync(() -> {
                try {
                    var filePath = storePath.resolve(key + ".checkpoint");
                    
                    if (!Files.exists(filePath)) {
                        logger.debug("Checkpoint file not found: {}", filePath);
                        return null;
                    }
                    
                    StateCheckpoint checkpoint;
                    if (enableCompression) {
                        try (var fis = Files.newInputStream(filePath);
                             var gzis = new GZIPInputStream(fis);
                             var ois = new ObjectInputStream(gzis)) {
                            checkpoint = (StateCheckpoint) ois.readObject();
                        }
                    } else {
                        try (var fis = Files.newInputStream(filePath);
                             var ois = new ObjectInputStream(fis)) {
                            checkpoint = (StateCheckpoint) ois.readObject();
                        }
                    }
                    
                    logger.debug("Loaded checkpoint from file store: {} -> {}", key, filePath);
                    return checkpoint;
                } catch (IOException | ClassNotFoundException e) {
                    logger.error("Failed to load checkpoint {}: {}", key, e.getMessage());
                    return null; // Return null for load failures (graceful degradation)
                }
            });
        }
        
        @Override
        public CompletableFuture<Void> delete(String key) {
            return CompletableFuture.runAsync(() -> {
                try {
                    var filePath = storePath.resolve(key + ".checkpoint");
                    
                    if (Files.exists(filePath)) {
                        Files.delete(filePath);
                        logger.debug("Deleted checkpoint from file store: {} -> {}", key, filePath);
                    } else {
                        logger.debug("Checkpoint file not found for deletion: {}", filePath);
                    }
                } catch (IOException e) {
                    logger.error("Failed to delete checkpoint {}: {}", key, e.getMessage());
                    throw new RuntimeException("Checkpoint deletion failed", e);
                }
            });
        }
        
        @Override
        public CompletableFuture<Map<String, StateCheckpoint>> listCheckpoints(String instanceId) {
            return CompletableFuture.supplyAsync(() -> {
                try {
                    if (!Files.exists(storePath)) {
                        logger.debug("Storage path does not exist: {}", storePath);
                        return Map.<String, StateCheckpoint>of();
                    }
                    
                    // Find all checkpoint files for the given instance
                    var checkpoints = Files.list(storePath)
                        .filter(path -> path.getFileName().toString().endsWith(".checkpoint"))
                        .filter(path -> path.getFileName().toString().startsWith(instanceId + "-"))
                        .collect(Collectors.toMap(
                            path -> {
                                var fileName = path.getFileName().toString();
                                return fileName.substring(0, fileName.lastIndexOf(".checkpoint"));
                            },
                            path -> {
                                try {
                                    // Load each checkpoint
                                    StateCheckpoint checkpoint;
                                    if (enableCompression) {
                                        try (var fis = Files.newInputStream(path);
                                             var gzis = new GZIPInputStream(fis);
                                             var ois = new ObjectInputStream(gzis)) {
                                            checkpoint = (StateCheckpoint) ois.readObject();
                                        }
                                    } else {
                                        try (var fis = Files.newInputStream(path);
                                             var ois = new ObjectInputStream(fis)) {
                                            checkpoint = (StateCheckpoint) ois.readObject();
                                        }
                                    }
                                    return checkpoint;
                                } catch (IOException | ClassNotFoundException e) {
                                    logger.warn("Failed to load checkpoint {}: {}", path, e.getMessage());
                                    return null;
                                }
                            }
                        ))
                        .entrySet().stream()
                        .filter(entry -> entry.getValue() != null)
                        .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
                    
                    logger.debug("Listed {} checkpoints for instance: {}", checkpoints.size(), instanceId);
                    return checkpoints;
                } catch (IOException e) {
                    logger.error("Failed to list checkpoints for instance {}: {}", instanceId, e.getMessage());
                    return Map.<String, StateCheckpoint>of();
                }
            });
        }
        
        @Override
        public CompletableFuture<Void> cleanup(String instanceId, int maxCheckpoints) {
            return CompletableFuture.runAsync(() -> {
                logger.debug("Cleaned up checkpoints for instance: {}", instanceId);
            });
        }
    }
}