package com.hellblazer.art.nlp.streaming;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
import java.util.function.Function;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Advanced windowing manager for stream processing supporting time-based, count-based,
 * session-based, and sliding window operations with efficient memory management.
 */
public class WindowingManager<T> implements AutoCloseable {
    private static final Logger log = LoggerFactory.getLogger(WindowingManager.class);
    
    private final String managerId;
    private final WindowingConfig config;
    private final Map<String, Window<T>> activeWindows = new ConcurrentHashMap<>();
    private final ScheduledExecutorService scheduler;
    private final ExecutorService processingExecutor;
    private final AtomicLong windowCounter = new AtomicLong(0);
    private final AtomicLong eventCounter = new AtomicLong(0);
    private volatile boolean running = false;
    
    /**
     * Configuration for windowing behavior.
     */
    public record WindowingConfig(
        Duration defaultTimeWindow,
        int defaultCountWindow,
        Duration sessionTimeout,
        int maxActiveWindows,
        boolean enableWatermarks,
        Duration watermarkLateness,
        int processingThreads
    ) {
        public static WindowingConfig defaultConfig() {
            return new WindowingConfig(
                Duration.ofMinutes(5),    // defaultTimeWindow
                1000,                     // defaultCountWindow
                Duration.ofMinutes(30),   // sessionTimeout
                10000,                    // maxActiveWindows
                true,                     // enableWatermarks
                Duration.ofSeconds(10),   // watermarkLateness
                4                         // processingThreads
            );
        }
        
        public WindowingConfig withTimeWindow(Duration window) {
            return new WindowingConfig(window, defaultCountWindow, sessionTimeout,
                                     maxActiveWindows, enableWatermarks, watermarkLateness, processingThreads);
        }
        
        public WindowingConfig withCountWindow(int count) {
            return new WindowingConfig(defaultTimeWindow, count, sessionTimeout,
                                     maxActiveWindows, enableWatermarks, watermarkLateness, processingThreads);
        }
    }
    
    /**
     * Types of windows supported by the manager.
     */
    public enum WindowType {
        TUMBLING_TIME,    // Fixed time intervals
        SLIDING_TIME,     // Overlapping time intervals
        TUMBLING_COUNT,   // Fixed count intervals
        SLIDING_COUNT,    // Overlapping count intervals
        SESSION,          // Activity-based sessions
        GLOBAL            // Single global window
    }
    
    /**
     * Window specification for creating windows.
     */
    public record WindowSpec<T>(
        String windowId,
        WindowType type,
        Duration timeSize,
        Duration slideInterval,
        int countSize,
        int slideCount,
        Function<T, String> keyExtractor,
        Duration sessionTimeout
    ) {
        // Factory methods for common window types
        public static <T> WindowSpec<T> tumblingTime(String windowId, Duration size, Function<T, String> keyExtractor) {
            return new WindowSpec<>(windowId, WindowType.TUMBLING_TIME, size, size, 0, 0, keyExtractor, null);
        }
        
        public static <T> WindowSpec<T> slidingTime(String windowId, Duration size, Duration slide, Function<T, String> keyExtractor) {
            return new WindowSpec<>(windowId, WindowType.SLIDING_TIME, size, slide, 0, 0, keyExtractor, null);
        }
        
        public static <T> WindowSpec<T> tumblingCount(String windowId, int size, Function<T, String> keyExtractor) {
            return new WindowSpec<>(windowId, WindowType.TUMBLING_COUNT, null, null, size, size, keyExtractor, null);
        }
        
        public static <T> WindowSpec<T> slidingCount(String windowId, int size, int slide, Function<T, String> keyExtractor) {
            return new WindowSpec<>(windowId, WindowType.SLIDING_COUNT, null, null, size, slide, keyExtractor, null);
        }
        
        public static <T> WindowSpec<T> session(String windowId, Duration timeout, Function<T, String> keyExtractor) {
            return new WindowSpec<>(windowId, WindowType.SESSION, null, null, 0, 0, keyExtractor, timeout);
        }
        
        public static <T> WindowSpec<T> global(String windowId) {
            return new WindowSpec<>(windowId, WindowType.GLOBAL, null, null, 0, 0, t -> "global", null);
        }
    }
    
    /**
     * Window state containing events and metadata.
     */
    public static class Window<T> {
        private final String windowId;
        private final String key;
        private final WindowType type;
        private final Instant startTime;
        private final Instant endTime;
        private final List<WindowEvent<T>> events = new CopyOnWriteArrayList<>();
        private final AtomicLong eventCount = new AtomicLong(0);
        private volatile Instant lastEventTime;
        private volatile boolean triggered = false;
        private volatile boolean expired = false;
        
        public Window(String windowId, String key, WindowType type, Instant startTime, Instant endTime) {
            this.windowId = windowId;
            this.key = key;
            this.type = type;
            this.startTime = startTime;
            this.endTime = endTime;
            this.lastEventTime = startTime;
        }
        
        public synchronized boolean addEvent(T data, Instant eventTime) {
            if (expired || (endTime != null && eventTime.isAfter(endTime))) {
                return false;
            }
            
            var windowEvent = new WindowEvent<>(data, eventTime);
            events.add(windowEvent);
            eventCount.incrementAndGet();
            lastEventTime = eventTime.isAfter(lastEventTime) ? eventTime : lastEventTime;
            
            return true;
        }
        
        public synchronized List<T> getEvents() {
            return events.stream()
                        .map(WindowEvent::data)
                        .toList();
        }
        
        public synchronized List<WindowEvent<T>> getWindowEvents() {
            return List.copyOf(events);
        }
        
        public boolean shouldTrigger(WindowSpec<?> spec, Instant currentTime) {
            if (triggered || expired) {
                return false;
            }
            
            return switch (type) {
                case TUMBLING_TIME, SLIDING_TIME -> 
                    endTime != null && currentTime.isAfter(endTime);
                case TUMBLING_COUNT, SLIDING_COUNT -> 
                    eventCount.get() >= spec.countSize();
                case SESSION -> {
                    var timeout = spec.sessionTimeout() != null ? spec.sessionTimeout() : Duration.ofMinutes(30);
                    yield Duration.between(lastEventTime, currentTime).compareTo(timeout) > 0;
                }
                case GLOBAL -> false; // Global windows don't auto-trigger
            };
        }
        
        public void markTriggered() {
            triggered = true;
        }
        
        public void markExpired() {
            expired = true;
        }
        
        // Getters
        public String windowId() { return windowId; }
        public String key() { return key; }
        public WindowType type() { return type; }
        public Instant startTime() { return startTime; }
        public Instant endTime() { return endTime; }
        public long eventCount() { return eventCount.get(); }
        public Instant lastEventTime() { return lastEventTime; }
        public boolean isTriggered() { return triggered; }
        public boolean isExpired() { return expired; }
        
        @Override
        public String toString() {
            return String.format("Window[id=%s, key=%s, type=%s, events=%d, start=%s, end=%s]",
                               windowId, key, type, eventCount.get(), startTime, endTime);
        }
    }
    
    /**
     * Event wrapper with timestamp for windowing.
     */
    public record WindowEvent<T>(T data, Instant eventTime) {}
    
    /**
     * Window trigger result containing the triggered window and events.
     */
    public record WindowTrigger<T>(
        Window<T> window,
        List<T> events,
        Instant triggerTime,
        String reason
    ) {}
    
    /**
     * Creates a windowing manager with the specified configuration.
     */
    public WindowingManager(String managerId, WindowingConfig config) {
        this.managerId = managerId;
        this.config = config;
        this.scheduler = Executors.newScheduledThreadPool(2,
            r -> Thread.ofVirtual().name("windowing-scheduler-" + managerId + "-").factory().newThread(r));
        this.processingExecutor = Executors.newFixedThreadPool(config.processingThreads(),
            r -> Thread.ofVirtual().name("windowing-processor-" + managerId + "-").factory().newThread(r));
        
        log.info("Created WindowingManager '{}': timeWindow={}, countWindow={}, maxWindows={}", 
                managerId, config.defaultTimeWindow(), config.defaultCountWindow(), config.maxActiveWindows());
    }
    
    /**
     * Creates a windowing manager with default configuration.
     */
    public WindowingManager(String managerId) {
        this(managerId, WindowingConfig.defaultConfig());
    }
    
    /**
     * Starts the windowing manager.
     */
    public synchronized void start() {
        if (!running) {
            running = true;
            
            // Start window cleanup task
            scheduler.scheduleAtFixedRate(this::cleanupExpiredWindows, 1, 1, TimeUnit.MINUTES);
            
            // Start watermark advancement task if enabled
            if (config.enableWatermarks()) {
                scheduler.scheduleAtFixedRate(this::advanceWatermarks, 100, 100, TimeUnit.MILLISECONDS);
            }
            
            log.info("Started WindowingManager '{}'", managerId);
        }
    }
    
    /**
     * Stops the windowing manager gracefully.
     */
    public synchronized void stop() {
        if (running) {
            running = false;
            
            try {
                scheduler.shutdown();
                processingExecutor.shutdown();
                
                if (!scheduler.awaitTermination(10, TimeUnit.SECONDS)) {
                    scheduler.shutdownNow();
                }
                if (!processingExecutor.awaitTermination(10, TimeUnit.SECONDS)) {
                    processingExecutor.shutdownNow();
                }
                
                log.info("Stopped WindowingManager '{}'. Windows: {}, Events: {}", 
                        managerId, windowCounter.get(), eventCounter.get());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                scheduler.shutdownNow();
                processingExecutor.shutdownNow();
                log.error("Interrupted while stopping WindowingManager '{}'", managerId, e);
            }
        }
    }
    
    /**
     * Adds an event to appropriate windows based on specification.
     */
    public CompletableFuture<List<WindowTrigger<T>>> addEvent(WindowSpec<T> spec, T data, Instant eventTime) {
        if (!running) {
            return CompletableFuture.completedFuture(Collections.emptyList());
        }
        
        return CompletableFuture.supplyAsync(() -> {
            eventCounter.incrementAndGet();
            var key = spec.keyExtractor().apply(data);
            var triggers = new ArrayList<WindowTrigger<T>>();
            
            // Find or create windows for this event
            var targetWindows = getOrCreateWindows(spec, key, eventTime);
            
            // Add event to each target window
            for (var window : targetWindows) {
                if (window.addEvent(data, eventTime)) {
                    log.debug("Added event to window {}: {}", window.windowId(), data);
                    
                    // Check if window should trigger
                    if (window.shouldTrigger(spec, eventTime)) {
                        var trigger = triggerWindow(window, eventTime, "Event threshold reached");
                        if (trigger != null) {
                            triggers.add(trigger);
                        }
                    }
                }
            }
            
            return triggers;
        }, processingExecutor);
    }
    
    /**
     * Manually triggers a window by ID.
     */
    public Optional<WindowTrigger<T>> triggerWindow(String windowId) {
        var window = activeWindows.get(windowId);
        if (window != null) {
            var trigger = triggerWindow(window, Instant.now(), "Manual trigger");
            return Optional.ofNullable(trigger);
        }
        return Optional.empty();
    }
    
    /**
     * Registers a window trigger handler.
     */
    public void onWindowTrigger(Consumer<WindowTrigger<T>> handler) {
        // This would be implemented with the EventBus for loose coupling
        log.info("Window trigger handler registered for WindowingManager '{}'", managerId);
    }
    
    /**
     * Gets current windowing metrics.
     */
    public WindowingMetrics getMetrics() {
        var windowCounts = activeWindows.values().stream()
            .collect(java.util.stream.Collectors.groupingBy(
                Window::type,
                java.util.stream.Collectors.counting()
            ));
        
        return new WindowingMetrics(
            managerId,
            activeWindows.size(),
            windowCounter.get(),
            eventCounter.get(),
            windowCounts,
            running
        );
    }
    
    /**
     * Gets or creates windows for an event based on window specification.
     */
    private List<Window<T>> getOrCreateWindows(WindowSpec<T> spec, String key, Instant eventTime) {
        var windows = new ArrayList<Window<T>>();
        
        switch (spec.type()) {
            case TUMBLING_TIME -> {
                var window = getOrCreateTumblingTimeWindow(spec, key, eventTime);
                if (window != null) windows.add(window);
            }
            case SLIDING_TIME -> {
                windows.addAll(getOrCreateSlidingTimeWindows(spec, key, eventTime));
            }
            case TUMBLING_COUNT -> {
                var window = getOrCreateTumblingCountWindow(spec, key);
                if (window != null) windows.add(window);
            }
            case SLIDING_COUNT -> {
                windows.addAll(getOrCreateSlidingCountWindows(spec, key));
            }
            case SESSION -> {
                var window = getOrCreateSessionWindow(spec, key, eventTime);
                if (window != null) windows.add(window);
            }
            case GLOBAL -> {
                var window = getOrCreateGlobalWindow(spec, key);
                if (window != null) windows.add(window);
            }
        }
        
        return windows;
    }
    
    /**
     * Gets or creates a tumbling time window.
     */
    private Window<T> getOrCreateTumblingTimeWindow(WindowSpec spec, String key, Instant eventTime) {
        var windowSize = spec.timeSize();
        var windowStart = eventTime.truncatedTo(java.time.temporal.ChronoUnit.SECONDS)
            .minusNanos(eventTime.getNano() % windowSize.toNanos());
        var windowEnd = windowStart.plus(windowSize);
        
        var windowId = String.format("%s-%s-%s", spec.windowId(), key, windowStart);
        
        return activeWindows.computeIfAbsent(windowId, id -> {
            windowCounter.incrementAndGet();
            var window = new Window<T>(id, key, WindowType.TUMBLING_TIME, windowStart, windowEnd);
            log.debug("Created tumbling time window: {}", window);
            return window;
        });
    }
    
    /**
     * Gets or creates sliding time windows.
     */
    private List<Window<T>> getOrCreateSlidingTimeWindows(WindowSpec spec, String key, Instant eventTime) {
        var windows = new ArrayList<Window<T>>();
        var windowSize = spec.timeSize();
        var slideInterval = spec.slideInterval();
        
        // Calculate how many windows this event should belong to
        var numWindows = (int) (windowSize.toMillis() / slideInterval.toMillis());
        
        for (int i = 0; i < numWindows; i++) {
            var windowStart = eventTime.minus(slideInterval.multipliedBy(i))
                .truncatedTo(java.time.temporal.ChronoUnit.SECONDS);
            var windowEnd = windowStart.plus(windowSize);
            
            if (eventTime.isBefore(windowStart) || !eventTime.isBefore(windowEnd)) {
                continue; // Event doesn't belong to this window
            }
            
            var windowId = String.format("%s-%s-%s", spec.windowId(), key, windowStart);
            var window = activeWindows.computeIfAbsent(windowId, id -> {
                windowCounter.incrementAndGet();
                var w = new Window<T>(id, key, WindowType.SLIDING_TIME, windowStart, windowEnd);
                log.debug("Created sliding time window: {}", w);
                return w;
            });
            
            windows.add(window);
        }
        
        return windows;
    }
    
    /**
     * Gets or creates a tumbling count window.
     */
    private Window<T> getOrCreateTumblingCountWindow(WindowSpec spec, String key) {
        var windowId = String.format("%s-%s-current", spec.windowId(), key);
        
        return activeWindows.computeIfAbsent(windowId, id -> {
            windowCounter.incrementAndGet();
            var window = new Window<T>(id, key, WindowType.TUMBLING_COUNT, Instant.now(), null);
            log.debug("Created tumbling count window: {}", window);
            return window;
        });
    }
    
    /**
     * Gets or creates sliding count windows.
     */
    private List<Window<T>> getOrCreateSlidingCountWindows(WindowSpec spec, String key) {
        // Simplified implementation - would need more sophisticated sliding logic
        var window = getOrCreateTumblingCountWindow(spec, key);
        return List.of(window);
    }
    
    /**
     * Gets or creates a session window.
     */
    private Window<T> getOrCreateSessionWindow(WindowSpec spec, String key, Instant eventTime) {
        // Find existing session window within timeout
        var timeout = spec.sessionTimeout();
        var existingWindow = activeWindows.values().stream()
            .filter(w -> w.type() == WindowType.SESSION && w.key().equals(key))
            .filter(w -> Duration.between(w.lastEventTime(), eventTime).compareTo(timeout) <= 0)
            .findFirst();
        
        if (existingWindow.isPresent()) {
            return existingWindow.get();
        }
        
        // Create new session window
        var windowId = String.format("%s-%s-%s", spec.windowId(), key, eventTime);
        var window = new Window<T>(windowId, key, WindowType.SESSION, eventTime, null);
        windowCounter.incrementAndGet();
        activeWindows.put(windowId, window);
        
        log.debug("Created session window: {}", window);
        return window;
    }
    
    /**
     * Gets or creates a global window.
     */
    private Window<T> getOrCreateGlobalWindow(WindowSpec spec, String key) {
        var windowId = String.format("%s-%s-global", spec.windowId(), key);
        
        return activeWindows.computeIfAbsent(windowId, id -> {
            windowCounter.incrementAndGet();
            var window = new Window<T>(id, key, WindowType.GLOBAL, Instant.now(), null);
            log.debug("Created global window: {}", window);
            return window;
        });
    }
    
    /**
     * Triggers a window and creates a trigger result.
     */
    private WindowTrigger<T> triggerWindow(Window<T> window, Instant triggerTime, String reason) {
        if (window.isTriggered()) {
            return null;
        }
        
        window.markTriggered();
        var events = window.getEvents();
        var trigger = new WindowTrigger<>(window, events, triggerTime, reason);
        
        log.info("Triggered window {}: {} events, reason: {}", 
                window.windowId(), events.size(), reason);
        
        return trigger;
    }
    
    /**
     * Cleans up expired windows to manage memory.
     */
    private void cleanupExpiredWindows() {
        if (!running) return;
        
        var currentTime = Instant.now();
        var expiredWindows = activeWindows.entrySet().stream()
            .filter(entry -> isWindowExpired(entry.getValue(), currentTime))
            .map(Map.Entry::getKey)
            .toList();
        
        for (var windowId : expiredWindows) {
            var window = activeWindows.remove(windowId);
            if (window != null) {
                window.markExpired();
                log.debug("Cleaned up expired window: {}", window);
            }
        }
        
        if (!expiredWindows.isEmpty()) {
            log.info("Cleaned up {} expired windows from WindowingManager '{}'", 
                    expiredWindows.size(), managerId);
        }
    }
    
    /**
     * Determines if a window has expired and should be cleaned up.
     */
    private boolean isWindowExpired(Window<T> window, Instant currentTime) {
        // Window is expired if it's been triggered and no activity for cleanup period
        if (window.isTriggered()) {
            var inactivityPeriod = Duration.ofMinutes(5); // Configurable
            return Duration.between(window.lastEventTime(), currentTime).compareTo(inactivityPeriod) > 0;
        }
        
        // Session windows expire based on session timeout
        if (window.type() == WindowType.SESSION) {
            var timeout = config.sessionTimeout();
            return Duration.between(window.lastEventTime(), currentTime).compareTo(timeout) > 0;
        }
        
        // Time windows expire after their end time + lateness allowance
        if (window.endTime() != null) {
            var expireTime = window.endTime().plus(config.watermarkLateness());
            return currentTime.isAfter(expireTime);
        }
        
        return false;
    }
    
    /**
     * Advances watermarks for late event handling.
     */
    private void advanceWatermarks() {
        if (!running || !config.enableWatermarks()) return;
        
        var currentTime = Instant.now();
        var watermark = currentTime.minus(config.watermarkLateness());
        
        // Check for windows that should trigger based on watermark
        var triggeredWindows = new ArrayList<WindowTrigger<T>>();
        
        for (var window : activeWindows.values()) {
            if (!window.isTriggered() && window.endTime() != null && 
                watermark.isAfter(window.endTime())) {
                
                var trigger = triggerWindow(window, currentTime, "Watermark advance");
                if (trigger != null) {
                    triggeredWindows.add(trigger);
                }
            }
        }
        
        if (!triggeredWindows.isEmpty()) {
            log.debug("Watermark advanced, triggered {} windows", triggeredWindows.size());
        }
    }
    
    @Override
    public void close() {
        stop();
    }
    
    /**
     * Windowing performance metrics.
     */
    public record WindowingMetrics(
        String managerId,
        int activeWindows,
        long totalWindows,
        long totalEvents,
        Map<WindowType, Long> windowTypeDistribution,
        boolean running
    ) {
        public double getAverageEventsPerWindow() {
            return totalWindows > 0 ? (double) totalEvents / totalWindows : 0.0;
        }
    }
}