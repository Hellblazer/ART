package com.hellblazer.art.hartcq;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;

/**
 * Competitive queuing mechanism for HART-CQ processing.
 * Implements priority-based processing with thread-safe operations
 * and support for different priority strategies.
 */
public class CompetitiveQueue<T> implements AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(CompetitiveQueue.class);
    
    private final PriorityBlockingQueue<QueuedItem<T>> queue;
    private final ExecutorService processingPool;
    private final AtomicBoolean isProcessing;
    private final AtomicInteger activeProcessors;
    private final AtomicLong processedItemCount;
    private final List<CompletableFuture<Void>> activeProcessingTasks;
    
    private final PriorityStrategy<T> priorityStrategy;
    private final int maxConcurrentProcessors;
    private volatile boolean shutdown;
    
    /**
     * Creates a CompetitiveQueue with default priority strategy.
     * 
     * @param maxConcurrentProcessors maximum number of concurrent processors
     */
    public CompetitiveQueue(int maxConcurrentProcessors) {
        this(maxConcurrentProcessors, new DefaultPriorityStrategy<>());
    }
    
    /**
     * Creates a CompetitiveQueue with custom priority strategy.
     * 
     * @param maxConcurrentProcessors maximum number of concurrent processors
     * @param priorityStrategy strategy for determining item priorities
     */
    public CompetitiveQueue(int maxConcurrentProcessors, PriorityStrategy<T> priorityStrategy) {
        this.maxConcurrentProcessors = maxConcurrentProcessors;
        this.priorityStrategy = Objects.requireNonNull(priorityStrategy, "Priority strategy cannot be null");
        this.queue = new PriorityBlockingQueue<>();
        this.processingPool = Executors.newFixedThreadPool(maxConcurrentProcessors,
            r -> {
                var thread = new Thread(r, "HART-CQ-Processor");
                thread.setDaemon(true);
                return thread;
            });
        this.isProcessing = new AtomicBoolean(false);
        this.activeProcessors = new AtomicInteger(0);
        this.processedItemCount = new AtomicLong(0);
        this.activeProcessingTasks = new ArrayList<>();
        this.shutdown = false;
    }
    
    /**
     * Enqueues an item for competitive processing.
     * 
     * @param item the item to process
     * @return CompletableFuture that completes when the item is processed
     */
    public CompletableFuture<ProcessingResult<T>> enqueue(T item) {
        return enqueue(item, priorityStrategy.calculatePriority(item));
    }
    
    /**
     * Enqueues an item with explicit priority.
     * 
     * @param item the item to process
     * @param priority the priority for this item (higher = more priority)
     * @return CompletableFuture that completes when the item is processed
     */
    public CompletableFuture<ProcessingResult<T>> enqueue(T item, double priority) {
        if (shutdown) {
            return CompletableFuture.failedFuture(new IllegalStateException("Queue is shutdown"));
        }
        
        var result = new CompletableFuture<ProcessingResult<T>>();
        var queuedItem = new QueuedItem<>(item, priority, result, System.nanoTime());
        
        queue.offer(queuedItem);
        
        // Start processing if not already active
        if (!isProcessing.get() && activeProcessors.get() < maxConcurrentProcessors) {
            startProcessor();
        }
        
        return result;
    }
    
    /**
     * Starts a new processor thread if capacity allows.
     */
    private synchronized void startProcessor() {
        if (shutdown || activeProcessors.get() >= maxConcurrentProcessors) {
            return;
        }
        
        var processingTask = CompletableFuture.runAsync(() -> {
            activeProcessors.incrementAndGet();
            isProcessing.set(true);
            
            try {
                processItems();
            } finally {
                activeProcessors.decrementAndGet();
                if (activeProcessors.get() == 0) {
                    isProcessing.set(false);
                }
            }
        }, processingPool);
        
        activeProcessingTasks.add(processingTask);
    }
    
    /**
     * Main processing loop for competitive queue items.
     */
    private void processItems() {
        while (!shutdown && !Thread.currentThread().isInterrupted()) {
            try {
                var queuedItem = queue.poll(100, TimeUnit.MILLISECONDS);
                if (queuedItem == null) {
                    continue; // Timeout, check for shutdown
                }
                
                // Process the item
                var startTime = System.nanoTime();
                var result = processItem(queuedItem);
                var endTime = System.nanoTime();
                
                result.setProcessingTimeNanos(endTime - startTime);
                result.setQueueTimeNanos(startTime - queuedItem.getEnqueueTime());
                
                processedItemCount.incrementAndGet();
                queuedItem.getResultFuture().complete(result);
                
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                logger.error("Error processing queue item", e);
            }
        }
    }
    
    /**
     * Processes an individual queued item.
     * 
     * @param queuedItem the item to process
     * @return processing result
     */
    private ProcessingResult<T> processItem(QueuedItem<T> queuedItem) {
        var result = new ProcessingResult<T>();
        result.setOriginalItem(queuedItem.getItem());
        result.setPriority(queuedItem.getPriority());
        
        try {
            // Here would be the actual HART-CQ processing logic
            // For now, we simulate successful processing
            result.setSuccessful(true);
            result.setConfidence(priorityStrategy.calculateConfidence(queuedItem.getItem()));
            
            logger.debug("Processed item with priority {}: {}", 
                        queuedItem.getPriority(), queuedItem.getItem());
                        
        } catch (Exception e) {
            result.setSuccessful(false);
            result.setErrorMessage(e.getMessage());
            logger.error("Failed to process item: {}", queuedItem.getItem(), e);
        }
        
        return result;
    }
    
    /**
     * Gets the current queue size.
     * 
     * @return number of items waiting in queue
     */
    public int getQueueSize() {
        return queue.size();
    }
    
    /**
     * Gets the number of active processors.
     * 
     * @return number of currently active processors
     */
    public int getActiveProcessors() {
        return activeProcessors.get();
    }
    
    /**
     * Gets the total number of processed items.
     * 
     * @return total processed item count
     */
    public long getProcessedItemCount() {
        return processedItemCount.get();
    }
    
    /**
     * Checks if the queue is currently processing items.
     * 
     * @return true if processing is active
     */
    public boolean isProcessing() {
        return isProcessing.get();
    }
    
    /**
     * Gets processing statistics.
     * 
     * @return current processing statistics
     */
    public ProcessingStats getProcessingStats() {
        return new ProcessingStats(
            getQueueSize(),
            getActiveProcessors(),
            getProcessedItemCount(),
            isProcessing()
        );
    }
    
    @Override
    public void close() {
        shutdown = true;
        isProcessing.set(false);
        
        // Cancel all active processing tasks
        for (var task : activeProcessingTasks) {
            task.cancel(true);
        }
        
        processingPool.shutdown();
        try {
            if (!processingPool.awaitTermination(5, TimeUnit.SECONDS)) {
                processingPool.shutdownNow();
            }
        } catch (InterruptedException e) {
            processingPool.shutdownNow();
            Thread.currentThread().interrupt();
        }
        
        // Complete any remaining futures with cancellation
        QueuedItem<T> remaining;
        while ((remaining = queue.poll()) != null) {
            remaining.getResultFuture().cancel(true);
        }
        
        logger.info("CompetitiveQueue closed. Processed {} items total.", processedItemCount.get());
    }
    
    /**
     * Queued item wrapper with priority and timing information.
     */
    private static class QueuedItem<T> implements Comparable<QueuedItem<T>> {
        private final T item;
        private final double priority;
        private final CompletableFuture<ProcessingResult<T>> resultFuture;
        private final long enqueueTime;
        
        public QueuedItem(T item, double priority, CompletableFuture<ProcessingResult<T>> resultFuture, long enqueueTime) {
            this.item = item;
            this.priority = priority;
            this.resultFuture = resultFuture;
            this.enqueueTime = enqueueTime;
        }
        
        @Override
        public int compareTo(QueuedItem<T> other) {
            // Higher priority items come first
            return Double.compare(other.priority, this.priority);
        }
        
        public T getItem() { return item; }
        public double getPriority() { return priority; }
        public CompletableFuture<ProcessingResult<T>> getResultFuture() { return resultFuture; }
        public long getEnqueueTime() { return enqueueTime; }
    }
    
    /**
     * Strategy for determining item priorities.
     */
    public interface PriorityStrategy<T> {
        double calculatePriority(T item);
        default double calculateConfidence(T item) { return 0.8; }
    }
    
    /**
     * Default priority strategy using hashCode-based priority.
     */
    public static class DefaultPriorityStrategy<T> implements PriorityStrategy<T> {
        @Override
        public double calculatePriority(T item) {
            return Math.abs(item.hashCode() % 100) / 100.0;
        }
    }
    
    /**
     * Processing result for queued items.
     */
    public static class ProcessingResult<T> {
        private T originalItem;
        private double priority;
        private boolean successful;
        private double confidence;
        private String errorMessage;
        private long processingTimeNanos;
        private long queueTimeNanos;
        
        public T getOriginalItem() { return originalItem; }
        public void setOriginalItem(T originalItem) { this.originalItem = originalItem; }
        
        public double getPriority() { return priority; }
        public void setPriority(double priority) { this.priority = priority; }
        
        public boolean isSuccessful() { return successful; }
        public void setSuccessful(boolean successful) { this.successful = successful; }
        
        public double getConfidence() { return confidence; }
        public void setConfidence(double confidence) { this.confidence = confidence; }
        
        public String getErrorMessage() { return errorMessage; }
        public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }
        
        public long getProcessingTimeNanos() { return processingTimeNanos; }
        public void setProcessingTimeNanos(long processingTimeNanos) { this.processingTimeNanos = processingTimeNanos; }
        
        public long getQueueTimeNanos() { return queueTimeNanos; }
        public void setQueueTimeNanos(long queueTimeNanos) { this.queueTimeNanos = queueTimeNanos; }
        
        public double getProcessingTimeMillis() { return processingTimeNanos / 1_000_000.0; }
        public double getQueueTimeMillis() { return queueTimeNanos / 1_000_000.0; }
    }
    
    /**
     * Processing statistics snapshot.
     */
    public static class ProcessingStats {
        private final int queueSize;
        private final int activeProcessors;
        private final long processedItemCount;
        private final boolean isProcessing;
        
        public ProcessingStats(int queueSize, int activeProcessors, long processedItemCount, boolean isProcessing) {
            this.queueSize = queueSize;
            this.activeProcessors = activeProcessors;
            this.processedItemCount = processedItemCount;
            this.isProcessing = isProcessing;
        }
        
        public int getQueueSize() { return queueSize; }
        public int getActiveProcessors() { return activeProcessors; }
        public long getProcessedItemCount() { return processedItemCount; }
        public boolean isProcessing() { return isProcessing; }
        
        @Override
        public String toString() {
            return String.format("ProcessingStats[queue=%d, active=%d, processed=%d, processing=%s]",
                               queueSize, activeProcessors, processedItemCount, isProcessing);
        }
    }
}