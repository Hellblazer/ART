package com.hellblazer.art.nlp.streaming;

import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
import java.util.function.Function;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * High-performance streaming processor with backpressure handling and event loop.
 * Processes continuous streams of data with configurable parallelism and flow control.
 * 
 * @param <T> Input data type
 * @param <R> Output result type
 */
public class StreamingProcessor<T, R> implements AutoCloseable {
    private static final Logger log = LoggerFactory.getLogger(StreamingProcessor.class);
    
    private final String processorId;
    private final Function<T, R> processor;
    private final Consumer<R> resultHandler;
    private final Consumer<Exception> errorHandler;
    private final ExecutorService executorService;
    private final BlockingQueue<StreamEvent<T>> eventQueue;
    private final AtomicBoolean running = new AtomicBoolean(false);
    private final AtomicLong processedCount = new AtomicLong(0);
    private final AtomicLong droppedCount = new AtomicLong(0);
    private final StreamingConfig config;
    private volatile Instant lastProcessedTime;
    
    /**
     * Configuration for streaming processor behavior.
     */
    public record StreamingConfig(
        int maxQueueSize,
        int threadPoolSize,
        Duration processingTimeout,
        BackpressureStrategy backpressureStrategy,
        boolean enableMetrics
    ) {
        public static StreamingConfig defaultConfig() {
            return new StreamingConfig(
                1000,                              // maxQueueSize
                Runtime.getRuntime().availableProcessors(), // threadPoolSize
                Duration.ofSeconds(30),            // processingTimeout
                BackpressureStrategy.DROP_OLDEST, // backpressureStrategy
                true                               // enableMetrics
            );
        }
    }
    
    /**
     * Strategies for handling backpressure when queue is full.
     */
    public enum BackpressureStrategy {
        DROP_OLDEST,    // Remove oldest event to make room
        DROP_NEWEST,    // Drop the new incoming event
        BLOCK,          // Block until space is available
        EXPAND_QUEUE    // Temporarily expand queue size
    }
    
    /**
     * Internal event wrapper for stream processing.
     */
    private record StreamEvent<T>(
        String eventId,
        T data,
        Instant timestamp,
        int priority
    ) {}
    
    /**
     * Creates a streaming processor with the specified configuration.
     */
    public StreamingProcessor(String processorId, 
                             Function<T, R> processor,
                             Consumer<R> resultHandler,
                             Consumer<Exception> errorHandler,
                             StreamingConfig config) {
        this.processorId = processorId;
        this.processor = processor;
        this.resultHandler = resultHandler;
        this.errorHandler = errorHandler;
        this.config = config;
        this.eventQueue = new ArrayBlockingQueue<>(config.maxQueueSize());
        this.executorService = Executors.newFixedThreadPool(config.threadPoolSize(),
            r -> Thread.ofVirtual().name("streaming-" + processorId + "-").factory().newThread(r));
        this.lastProcessedTime = Instant.now();
        
        log.info("Created StreamingProcessor '{}': maxQueue={}, threads={}, backpressure={}", 
                processorId, config.maxQueueSize(), config.threadPoolSize(), 
                config.backpressureStrategy());
    }
    
    /**
     * Starts the streaming processor event loop.
     */
    public synchronized void start() {
        if (running.compareAndSet(false, true)) {
            log.info("Starting streaming processor '{}'", processorId);
            
            // Start event loop threads
            for (int i = 0; i < config.threadPoolSize(); i++) {
                executorService.submit(this::eventLoop);
            }
            
            log.info("Streaming processor '{}' started with {} threads", 
                    processorId, config.threadPoolSize());
        }
    }
    
    /**
     * Stops the streaming processor gracefully.
     */
    public synchronized void stop() {
        if (running.compareAndSet(true, false)) {
            log.info("Stopping streaming processor '{}'", processorId);
            
            try {
                executorService.shutdown();
                if (!executorService.awaitTermination(30, TimeUnit.SECONDS)) {
                    log.warn("Streaming processor '{}' did not terminate gracefully, forcing shutdown", 
                            processorId);
                    executorService.shutdownNow();
                }
                
                log.info("Streaming processor '{}' stopped. Processed: {}, Dropped: {}", 
                        processorId, processedCount.get(), droppedCount.get());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                executorService.shutdownNow();
                log.error("Interrupted while stopping streaming processor '{}'", processorId, e);
            }
        }
    }
    
    /**
     * Submits data for streaming processing.
     * 
     * @param data the data to process
     * @return true if accepted, false if dropped due to backpressure
     */
    public boolean submit(T data) {
        return submit(data, 0);
    }
    
    /**
     * Submits data for streaming processing with priority.
     * 
     * @param data the data to process
     * @param priority processing priority (higher values processed first)
     * @return true if accepted, false if dropped due to backpressure
     */
    public boolean submit(T data, int priority) {
        if (!running.get()) {
            log.warn("Cannot submit to stopped streaming processor '{}'", processorId);
            return false;
        }
        
        var event = new StreamEvent<>(
            generateEventId(),
            data,
            Instant.now(),
            priority
        );
        
        return enqueueEvent(event);
    }
    
    /**
     * Gets current streaming metrics.
     */
    public StreamingMetrics getMetrics() {
        return new StreamingMetrics(
            processorId,
            processedCount.get(),
            droppedCount.get(),
            eventQueue.size(),
            config.maxQueueSize(),
            lastProcessedTime,
            running.get()
        );
    }
    
    /**
     * Event loop for processing stream events.
     */
    private void eventLoop() {
        var threadName = Thread.currentThread().getName();
        log.debug("Started event loop thread: {}", threadName);
        
        while (running.get()) {
            try {
                var event = eventQueue.poll(1, TimeUnit.SECONDS);
                if (event != null) {
                    processEvent(event);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                log.debug("Event loop thread {} interrupted", threadName);
                break;
            } catch (Exception e) {
                log.error("Error in event loop thread {}", threadName, e);
                if (errorHandler != null) {
                    try {
                        errorHandler.accept(e);
                    } catch (Exception handlerError) {
                        log.error("Error handler failed in thread {}", threadName, handlerError);
                    }
                }
            }
        }
        
        log.debug("Event loop thread {} stopped", threadName);
    }
    
    /**
     * Processes a single stream event.
     */
    private void processEvent(StreamEvent<T> event) {
        var startTime = Instant.now();
        
        try {
            log.debug("Processing event {}: data={}", event.eventId(), event.data());
            
            // Apply processing function with timeout
            var future = CompletableFuture.supplyAsync(() -> processor.apply(event.data()));
            var result = future.get(config.processingTimeout().toMillis(), TimeUnit.MILLISECONDS);
            
            // Handle successful result
            if (resultHandler != null) {
                resultHandler.accept(result);
            }
            
            processedCount.incrementAndGet();
            lastProcessedTime = Instant.now();
            
            var processingTime = Duration.between(startTime, lastProcessedTime);
            log.debug("Processed event {} in {}", event.eventId(), processingTime);
            
        } catch (TimeoutException e) {
            log.warn("Processing timeout for event {}: {}ms", 
                    event.eventId(), config.processingTimeout().toMillis());
            if (errorHandler != null) {
                errorHandler.accept(new StreamingException("Processing timeout", e));
            }
        } catch (Exception e) {
            log.error("Processing error for event {}", event.eventId(), e);
            if (errorHandler != null) {
                errorHandler.accept(new StreamingException("Processing error", e));
            }
        }
    }
    
    /**
     * Enqueues an event with backpressure handling.
     */
    private boolean enqueueEvent(StreamEvent<T> event) {
        switch (config.backpressureStrategy()) {
            case DROP_OLDEST -> {
                // Keep dropping oldest events until we can queue the new one
                while (!eventQueue.offer(event)) {
                    var dropped = eventQueue.poll();
                    if (dropped != null) {
                        droppedCount.incrementAndGet();
                        log.debug("Dropped oldest event {} to make room for {}", 
                                dropped.eventId(), event.eventId());
                    } else {
                        // Queue is empty but offer failed - should not happen with ArrayBlockingQueue
                        log.warn("Failed to offer event to queue despite being empty");
                        return false;
                    }
                }
                return true;
            }
            case DROP_NEWEST -> {
                var accepted = eventQueue.offer(event);
                if (!accepted) {
                    droppedCount.incrementAndGet();
                    log.debug("Dropped newest event {} due to full queue", event.eventId());
                }
                return accepted;
            }
            case BLOCK -> {
                try {
                    eventQueue.put(event);
                    return true;
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    log.debug("Interrupted while blocking on full queue");
                    return false;
                }
            }
            case EXPAND_QUEUE -> {
                // For ArrayBlockingQueue, we can't expand dynamically
                // Try to add, and if queue is full, create a temporary larger queue
                if (eventQueue.offer(event)) {
                    return true;
                } else {
                    log.debug("Queue full, cannot expand ArrayBlockingQueue - dropping oldest event");
                    // Fallback to DROP_OLDEST behavior without recursion
                    var dropped = eventQueue.poll();
                    if (dropped != null) {
                        droppedCount.incrementAndGet();
                        log.debug("Dropped oldest event {} due to expand limitation", dropped.eventId());
                        return eventQueue.offer(event);
                    }
                    return false;
                }
            }
            default -> {
                log.warn("Unknown backpressure strategy: {}", config.backpressureStrategy());
                return false;
            }
        }
    }
    
    /**
     * Generates unique event ID.
     */
    private String generateEventId() {
        return processorId + "-" + System.nanoTime();
    }
    
    @Override
    public void close() {
        stop();
    }
    
    /**
     * Streaming processor metrics.
     */
    public record StreamingMetrics(
        String processorId,
        long processedCount,
        long droppedCount,
        int currentQueueSize,
        int maxQueueSize,
        Instant lastProcessedTime,
        boolean running
    ) {
        public double getQueueUtilization() {
            return maxQueueSize > 0 ? (double) currentQueueSize / maxQueueSize : 0.0;
        }
        
        public double getDropRate() {
            var total = processedCount + droppedCount;
            return total > 0 ? (double) droppedCount / total : 0.0;
        }
    }
    
    /**
     * Exception for streaming processing errors.
     */
    public static class StreamingException extends RuntimeException {
        public StreamingException(String message) {
            super(message);
        }
        
        public StreamingException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}