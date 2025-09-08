package com.hellblazer.art.nlp.streaming;

import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
import java.util.function.Predicate;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * High-performance event-driven architecture with publisher-subscriber pattern.
 * Supports typed events, filtering, priority handling, and fault tolerance.
 */
public class EventBus implements AutoCloseable {
    private static final Logger log = LoggerFactory.getLogger(EventBus.class);
    
    private final String busId;
    private final EventBusConfig config;
    private final ExecutorService executorService;
    private final Map<Class<?>, Set<Subscription<?>>> subscriptions = new ConcurrentHashMap<>();
    private final BlockingQueue<EventDelivery<?>> deliveryQueue;
    private final AtomicLong eventCounter = new AtomicLong(0);
    private final AtomicLong deliveredCounter = new AtomicLong(0);
    private final AtomicLong failedCounter = new AtomicLong(0);
    private volatile boolean running = false;
    
    /**
     * Configuration for event bus behavior.
     */
    public record EventBusConfig(
        int maxQueueSize,
        int deliveryThreads,
        boolean enableDeadLetterQueue,
        int maxRetryAttempts,
        long retryDelayMs,
        boolean enableMetrics,
        boolean enableEventOrdering
    ) {
        public static EventBusConfig defaultConfig() {
            return new EventBusConfig(
                10000,  // maxQueueSize
                4,      // deliveryThreads
                true,   // enableDeadLetterQueue
                3,      // maxRetryAttempts
                100,    // retryDelayMs
                true,   // enableMetrics
                false   // enableEventOrdering (faster without ordering)
            );
        }
        
        public EventBusConfig withQueueSize(int size) {
            return new EventBusConfig(size, deliveryThreads, enableDeadLetterQueue, 
                                    maxRetryAttempts, retryDelayMs, enableMetrics, enableEventOrdering);
        }
        
        public EventBusConfig withDeliveryThreads(int threads) {
            return new EventBusConfig(maxQueueSize, threads, enableDeadLetterQueue,
                                    maxRetryAttempts, retryDelayMs, enableMetrics, enableEventOrdering);
        }
    }
    
    /**
     * Base class for all events in the system.
     */
    public abstract static class Event {
        private final String eventId;
        private final Instant timestamp;
        private final int priority;
        private final Map<String, Object> metadata;
        
        protected Event(int priority) {
            this.eventId = generateEventId();
            this.timestamp = Instant.now();
            this.priority = priority;
            this.metadata = new ConcurrentHashMap<>();
        }
        
        protected Event() {
            this(0);
        }
        
        public String eventId() { return eventId; }
        public Instant timestamp() { return timestamp; }
        public int priority() { return priority; }
        public Map<String, Object> metadata() { return Collections.unmodifiableMap(metadata); }
        
        public void addMetadata(String key, Object value) {
            metadata.put(key, value);
        }
        
        private static String generateEventId() {
            return "event-" + System.nanoTime() + "-" + Thread.currentThread().getId();
        }
        
        @Override
        public String toString() {
            return String.format("%s[id=%s, timestamp=%s, priority=%d]", 
                               getClass().getSimpleName(), eventId, timestamp, priority);
        }
    }
    
    /**
     * Processing result event for streaming updates.
     */
    public static class ProcessingResultEvent extends Event {
        private final String processorId;
        private final Object result;
        private final boolean success;
        private final String errorMessage;
        
        public ProcessingResultEvent(String processorId, Object result, boolean success, String errorMessage) {
            super(success ? 0 : 1); // Higher priority for errors
            this.processorId = processorId;
            this.result = result;
            this.success = success;
            this.errorMessage = errorMessage;
        }
        
        public static ProcessingResultEvent success(String processorId, Object result) {
            return new ProcessingResultEvent(processorId, result, true, null);
        }
        
        public static ProcessingResultEvent error(String processorId, String errorMessage) {
            return new ProcessingResultEvent(processorId, null, false, errorMessage);
        }
        
        public String processorId() { return processorId; }
        public Object result() { return result; }
        public boolean success() { return success; }
        public String errorMessage() { return errorMessage; }
    }
    
    /**
     * Learning update event for incremental learning notifications.
     */
    public static class LearningUpdateEvent extends Event {
        private final String channelName;
        private final int categoryId;
        private final double updateMagnitude;
        private final boolean significant;
        
        public LearningUpdateEvent(String channelName, int categoryId, double updateMagnitude, boolean significant) {
            super(significant ? 1 : 0); // Higher priority for significant updates
            this.channelName = channelName;
            this.categoryId = categoryId;
            this.updateMagnitude = updateMagnitude;
            this.significant = significant;
        }
        
        public String channelName() { return channelName; }
        public int categoryId() { return categoryId; }
        public double updateMagnitude() { return updateMagnitude; }
        public boolean significant() { return significant; }
    }
    
    /**
     * System monitoring event for performance tracking.
     */
    public static class SystemMonitoringEvent extends Event {
        private final String componentId;
        private final Map<String, Number> metrics;
        private final Instant measurementTime;
        
        public SystemMonitoringEvent(String componentId, Map<String, Number> metrics) {
            super(-1); // Low priority for monitoring events
            this.componentId = componentId;
            this.metrics = Map.copyOf(metrics);
            this.measurementTime = Instant.now();
        }
        
        public String componentId() { return componentId; }
        public Map<String, Number> metrics() { return metrics; }
        public Instant measurementTime() { return measurementTime; }
    }
    
    /**
     * Event subscription with filtering and error handling.
     */
    public static class Subscription<T extends Event> {
        private final String subscriptionId;
        private final Class<T> eventType;
        private final Consumer<T> handler;
        private final Predicate<T> filter;
        private final Consumer<Exception> errorHandler;
        private final AtomicLong processedCount = new AtomicLong(0);
        private final AtomicLong errorCount = new AtomicLong(0);
        private volatile boolean active = true;
        
        public Subscription(Class<T> eventType, Consumer<T> handler, Predicate<T> filter, Consumer<Exception> errorHandler) {
            this.subscriptionId = "sub-" + System.nanoTime();
            this.eventType = eventType;
            this.handler = handler;
            this.filter = filter != null ? filter : event -> true;
            this.errorHandler = errorHandler;
        }
        
        public Subscription(Class<T> eventType, Consumer<T> handler) {
            this(eventType, handler, null, null);
        }
        
        public boolean matches(Event event) {
            return active && eventType.isAssignableFrom(event.getClass());
        }
        
        @SuppressWarnings("unchecked")
        public boolean deliver(Event event) {
            if (!matches(event)) {
                return false;
            }
            
            var typedEvent = (T) event;
            if (!filter.test(typedEvent)) {
                return false;
            }
            
            try {
                handler.accept(typedEvent);
                processedCount.incrementAndGet();
                return true;
            } catch (Exception e) {
                errorCount.incrementAndGet();
                log.warn("Error in event handler for subscription {}: {}", subscriptionId, e.getMessage());
                
                if (errorHandler != null) {
                    try {
                        errorHandler.accept(e);
                    } catch (Exception handlerError) {
                        log.error("Error in error handler for subscription {}", subscriptionId, handlerError);
                    }
                }
                return false;
            }
        }
        
        public void deactivate() {
            active = false;
        }
        
        // Getters
        public String subscriptionId() { return subscriptionId; }
        public Class<T> eventType() { return eventType; }
        public long processedCount() { return processedCount.get(); }
        public long errorCount() { return errorCount.get(); }
        public boolean active() { return active; }
    }
    
    /**
     * Internal event delivery wrapper with retry support.
     */
    private record EventDelivery<T extends Event>(
        T event,
        Set<Subscription<? super T>> targetSubscriptions,
        int attemptCount,
        Instant scheduleTime
    ) {
        public EventDelivery<T> withRetry() {
            var retryTime = Instant.now().plusMillis(100L * attemptCount); // Exponential backoff
            return new EventDelivery<>(event, targetSubscriptions, attemptCount + 1, retryTime);
        }
        
        public boolean isReady() {
            return Instant.now().isAfter(scheduleTime);
        }
    }
    
    /**
     * Creates an event bus with the specified configuration.
     */
    public EventBus(String busId, EventBusConfig config) {
        this.busId = busId;
        this.config = config;
        this.deliveryQueue = config.enableEventOrdering() 
            ? new PriorityBlockingQueue<>(config.maxQueueSize(), this::compareDeliveries)
            : new ArrayBlockingQueue<>(config.maxQueueSize());
        this.executorService = Executors.newFixedThreadPool(config.deliveryThreads(),
            r -> Thread.ofVirtual().name("eventbus-" + busId + "-").factory().newThread(r));
        
        log.info("Created EventBus '{}': maxQueue={}, threads={}, ordering={}", 
                busId, config.maxQueueSize(), config.deliveryThreads(), config.enableEventOrdering());
    }
    
    /**
     * Creates an event bus with default configuration.
     */
    public EventBus(String busId) {
        this(busId, EventBusConfig.defaultConfig());
    }
    
    /**
     * Starts the event bus delivery system.
     */
    public synchronized void start() {
        if (!running) {
            running = true;
            
            // Start delivery threads
            for (int i = 0; i < config.deliveryThreads(); i++) {
                executorService.submit(this::deliveryLoop);
            }
            
            log.info("Started EventBus '{}' with {} delivery threads", busId, config.deliveryThreads());
        }
    }
    
    /**
     * Stops the event bus gracefully.
     */
    public synchronized void stop() {
        if (running) {
            running = false;
            
            try {
                executorService.shutdown();
                if (!executorService.awaitTermination(10, TimeUnit.SECONDS)) {
                    log.warn("EventBus '{}' did not terminate gracefully, forcing shutdown", busId);
                    executorService.shutdownNow();
                }
                
                log.info("Stopped EventBus '{}'. Events: {}, Delivered: {}, Failed: {}", 
                        busId, eventCounter.get(), deliveredCounter.get(), failedCounter.get());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                executorService.shutdownNow();
                log.error("Interrupted while stopping EventBus '{}'", busId, e);
            }
        }
    }
    
    /**
     * Publishes an event to all matching subscribers.
     */
    public <T extends Event> boolean publish(T event) {
        if (!running) {
            log.warn("Cannot publish to stopped EventBus '{}'", busId);
            return false;
        }
        
        eventCounter.incrementAndGet();
        
        // Find matching subscriptions
        var matchingSubscriptions = findMatchingSubscriptions(event);
        if (matchingSubscriptions.isEmpty()) {
            log.debug("No subscribers for event: {}", event);
            return true;
        }
        
        // Queue for delivery
        var delivery = new EventDelivery<>(event, matchingSubscriptions, 0, Instant.now());
        var queued = deliveryQueue.offer(delivery);
        
        if (!queued) {
            log.warn("EventBus '{}' queue full, dropping event: {}", busId, event);
            failedCounter.incrementAndGet();
            return false;
        }
        
        log.debug("Published event to EventBus '{}': {} -> {} subscribers", 
                 busId, event, matchingSubscriptions.size());
        return true;
    }
    
    /**
     * Subscribes to events of a specific type.
     */
    public <T extends Event> Subscription<T> subscribe(Class<T> eventType, Consumer<T> handler) {
        return subscribe(eventType, handler, null, null);
    }
    
    /**
     * Subscribes to events of a specific type with filtering.
     */
    public <T extends Event> Subscription<T> subscribe(Class<T> eventType, Consumer<T> handler, Predicate<T> filter) {
        return subscribe(eventType, handler, filter, null);
    }
    
    /**
     * Subscribes to events of a specific type with filtering and error handling.
     */
    public <T extends Event> Subscription<T> subscribe(Class<T> eventType, Consumer<T> handler, 
                                                      Predicate<T> filter, Consumer<Exception> errorHandler) {
        var subscription = new Subscription<>(eventType, handler, filter, errorHandler);
        
        subscriptions.computeIfAbsent(eventType, k -> ConcurrentHashMap.newKeySet()).add(subscription);
        
        log.info("Added subscription to EventBus '{}': {} -> {}", 
                busId, eventType.getSimpleName(), subscription.subscriptionId());
        
        return subscription;
    }
    
    /**
     * Unsubscribes from events.
     */
    public <T extends Event> boolean unsubscribe(Subscription<T> subscription) {
        subscription.deactivate();
        
        var eventSubscriptions = subscriptions.get(subscription.eventType());
        if (eventSubscriptions != null) {
            var removed = eventSubscriptions.remove(subscription);
            if (removed) {
                log.info("Removed subscription from EventBus '{}': {}", busId, subscription.subscriptionId());
            }
            return removed;
        }
        
        return false;
    }
    
    /**
     * Gets current event bus metrics.
     */
    public EventBusMetrics getMetrics() {
        var subscriptionCounts = subscriptions.entrySet().stream()
            .collect(java.util.stream.Collectors.toMap(
                entry -> entry.getKey().getSimpleName(),
                entry -> entry.getValue().size()
            ));
        
        return new EventBusMetrics(
            busId,
            eventCounter.get(),
            deliveredCounter.get(),
            failedCounter.get(),
            deliveryQueue.size(),
            config.maxQueueSize(),
            subscriptionCounts,
            running
        );
    }
    
    /**
     * Main delivery loop for processing events.
     */
    private void deliveryLoop() {
        var threadName = Thread.currentThread().getName();
        log.debug("Started delivery loop: {}", threadName);
        
        while (running) {
            try {
                var delivery = deliveryQueue.poll(1, TimeUnit.SECONDS);
                if (delivery != null && delivery.isReady()) {
                    deliverEvent(delivery);
                } else if (delivery != null) {
                    // Put back delayed delivery
                    deliveryQueue.offer(delivery);
                    Thread.sleep(10); // Small delay to prevent tight loop
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                log.debug("Delivery loop {} interrupted", threadName);
                break;
            } catch (Exception e) {
                log.error("Error in delivery loop {}", threadName, e);
                failedCounter.incrementAndGet();
            }
        }
        
        log.debug("Delivery loop {} stopped", threadName);
    }
    
    /**
     * Delivers an event to its target subscriptions.
     */
    @SuppressWarnings("unchecked")
    private <T extends Event> void deliverEvent(EventDelivery<T> delivery) {
        var event = delivery.event();
        var successCount = 0;
        var failureCount = 0;
        
        for (var subscription : delivery.targetSubscriptions()) {
            try {
                var delivered = ((Subscription<T>) subscription).deliver(event);
                if (delivered) {
                    successCount++;
                } else {
                    failureCount++;
                }
            } catch (Exception e) {
                failureCount++;
                log.error("Delivery error for event {}: {}", event.eventId(), e.getMessage());
            }
        }
        
        if (successCount > 0) {
            deliveredCounter.addAndGet(successCount);
            log.debug("Delivered event {} to {} subscribers", event.eventId(), successCount);
        }
        
        if (failureCount > 0) {
            failedCounter.addAndGet(failureCount);
            
            // Retry logic
            if (delivery.attemptCount() < config.maxRetryAttempts()) {
                var retryDelivery = delivery.withRetry();
                deliveryQueue.offer(retryDelivery);
                log.debug("Retrying delivery for event {} (attempt {})", 
                         event.eventId(), retryDelivery.attemptCount());
            } else {
                log.warn("Failed to deliver event {} after {} attempts", 
                        event.eventId(), config.maxRetryAttempts());
            }
        }
    }
    
    /**
     * Finds subscriptions that match the given event.
     */
    @SuppressWarnings("unchecked")
    private <T extends Event> Set<Subscription<? super T>> findMatchingSubscriptions(T event) {
        var matches = new HashSet<Subscription<? super T>>();
        
        // Check direct type matches and supertypes
        for (var entry : subscriptions.entrySet()) {
            var subscriptionType = entry.getKey();
            if (subscriptionType.isAssignableFrom(event.getClass())) {
                entry.getValue().forEach(sub -> matches.add((Subscription<? super T>) sub));
            }
        }
        
        return matches;
    }
    
    /**
     * Compares event deliveries for priority ordering.
     */
    private int compareDeliveries(EventDelivery<?> d1, EventDelivery<?> d2) {
        // Higher priority events first (reverse order)
        var priorityCompare = Integer.compare(d2.event().priority(), d1.event().priority());
        if (priorityCompare != 0) {
            return priorityCompare;
        }
        
        // Earlier scheduled time first
        return d1.scheduleTime().compareTo(d2.scheduleTime());
    }
    
    @Override
    public void close() {
        stop();
    }
    
    /**
     * Event bus performance metrics.
     */
    public record EventBusMetrics(
        String busId,
        long totalEvents,
        long deliveredEvents,
        long failedEvents,
        int currentQueueSize,
        int maxQueueSize,
        Map<String, Integer> subscriptionCounts,
        boolean running
    ) {
        public double getDeliveryRate() {
            var total = totalEvents;
            return total > 0 ? (double) deliveredEvents / total : 0.0;
        }
        
        public double getQueueUtilization() {
            return maxQueueSize > 0 ? (double) currentQueueSize / maxQueueSize : 0.0;
        }
        
        public int getTotalSubscriptions() {
            return subscriptionCounts.values().stream().mapToInt(Integer::intValue).sum();
        }
    }
}