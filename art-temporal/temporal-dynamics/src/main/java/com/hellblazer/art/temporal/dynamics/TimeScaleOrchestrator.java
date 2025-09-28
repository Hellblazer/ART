package com.hellblazer.art.temporal.dynamics;

import com.hellblazer.art.temporal.core.*;
import org.jctools.queues.MpscArrayQueue;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Orchestrates multiple dynamical systems operating at different time scales.
 * Uses virtual threads for efficient concurrent execution while maintaining
 * causal consistency and synchronization points.
 *
 * Based on the time scale hierarchy from Kazerounian & Grossberg (2014):
 * - FAST (10-100ms): Working memory / Shunting dynamics
 * - MEDIUM (50-500ms): Masking field dynamics
 * - SLOW (500-5000ms): Transmitter dynamics
 * - VERY_SLOW (1000-10000ms): Weight learning dynamics
 */
public class TimeScaleOrchestrator implements AutoCloseable {

    private final Map<DynamicalSystem.TimeScale, List<SystemRunner<?, ?>>> systemsByTimeScale;
    private final ExecutorService virtualThreadExecutor;
    private final ScheduledExecutorService scheduler;
    private final EventBus eventBus;
    private final AtomicBoolean running;
    private final ReentrantLock synchronizationLock;

    public TimeScaleOrchestrator() {
        this.systemsByTimeScale = new EnumMap<>(DynamicalSystem.TimeScale.class);
        this.virtualThreadExecutor = Executors.newVirtualThreadPerTaskExecutor();
        this.scheduler = Executors.newScheduledThreadPool(1);
        this.eventBus = new EventBus();
        this.running = new AtomicBoolean(false);
        this.synchronizationLock = new ReentrantLock();

        // Initialize time scale groups
        for (var timeScale : DynamicalSystem.TimeScale.values()) {
            systemsByTimeScale.put(timeScale, new CopyOnWriteArrayList<>());
        }
    }

    /**
     * Register a dynamical system to be orchestrated.
     */
    public <S extends State, P extends Parameters> void registerSystem(
            DynamicalSystem<S, P> system,
            S initialState,
            P parameters,
            NumericalIntegrator<S, P> integrator) {

        var runner = new SystemRunner<>(system, initialState, parameters, integrator);
        systemsByTimeScale.get(system.getTimeScale()).add(runner);
    }

    /**
     * Start orchestrated execution of all registered systems.
     */
    public void start() {
        if (running.compareAndSet(false, true)) {
            // Schedule update tasks for each time scale
            for (var entry : systemsByTimeScale.entrySet()) {
                var timeScale = entry.getKey();
                var systems = entry.getValue();

                if (!systems.isEmpty()) {
                    var period = (long) timeScale.getTypicalMillis();
                    scheduler.scheduleAtFixedRate(
                        () -> updateSystemsAtTimeScale(timeScale),
                        0, period, TimeUnit.MILLISECONDS
                    );
                }
            }
        }
    }

    /**
     * Stop orchestrated execution.
     */
    public void stop() {
        if (running.compareAndSet(true, false)) {
            scheduler.shutdown();
            try {
                if (!scheduler.awaitTermination(5, TimeUnit.SECONDS)) {
                    scheduler.shutdownNow();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                scheduler.shutdownNow();
            }
        }
    }

    /**
     * Update all systems at a given time scale.
     */
    private void updateSystemsAtTimeScale(DynamicalSystem.TimeScale timeScale) {
        var systems = systemsByTimeScale.get(timeScale);
        if (systems.isEmpty()) return;

        // Create futures for parallel execution
        var futures = new ArrayList<CompletableFuture<Void>>(systems.size());

        for (var system : systems) {
            futures.add(CompletableFuture.runAsync(
                system::update,
                virtualThreadExecutor
            ));
        }

        // Wait for all systems at this time scale to complete
        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
            .thenRun(() -> synchronizeAcrossTimeScales(timeScale))
            .exceptionally(throwable -> {
                handleError(timeScale, throwable);
                return null;
            });
    }

    /**
     * Synchronize state across different time scales when needed.
     */
    private void synchronizeAcrossTimeScales(DynamicalSystem.TimeScale completedScale) {
        synchronizationLock.lock();
        try {
            // Publish synchronization event
            eventBus.publish(new SynchronizationEvent(completedScale, System.currentTimeMillis()));

            // Handle cross-scale coupling based on completed scale
            switch (completedScale) {
                case FAST -> {
                    // Working memory updates may trigger masking field changes
                    propagateToMaskingField();
                }
                case MEDIUM -> {
                    // Masking field updates affect learning
                    propagateToLearning();
                }
                case SLOW -> {
                    // Transmitter updates modulate learning strength
                    modulateLearning();
                }
                case VERY_SLOW -> {
                    // Weight updates may affect working memory dynamics
                    updateWorkingMemoryParameters();
                }
            }
        } finally {
            synchronizationLock.unlock();
        }
    }

    private void propagateToMaskingField() {
        // Extract working memory states and update masking field inputs
        var fastSystems = systemsByTimeScale.get(DynamicalSystem.TimeScale.FAST);
        var mediumSystems = systemsByTimeScale.get(DynamicalSystem.TimeScale.MEDIUM);

        for (var source : fastSystems) {
            var state = source.getCurrentState();
            for (var target : mediumSystems) {
                target.receiveInput("working_memory", state);
            }
        }
    }

    private void propagateToLearning() {
        // Extract masking field activations for learning
        var mediumSystems = systemsByTimeScale.get(DynamicalSystem.TimeScale.MEDIUM);
        var verySlowSystems = systemsByTimeScale.get(DynamicalSystem.TimeScale.VERY_SLOW);

        for (var source : mediumSystems) {
            var state = source.getCurrentState();
            for (var target : verySlowSystems) {
                target.receiveInput("category_activation", state);
            }
        }
    }

    private void modulateLearning() {
        // Transmitter levels modulate learning strength
        var slowSystems = systemsByTimeScale.get(DynamicalSystem.TimeScale.SLOW);
        var verySlowSystems = systemsByTimeScale.get(DynamicalSystem.TimeScale.VERY_SLOW);

        for (var source : slowSystems) {
            var state = source.getCurrentState();
            for (var target : verySlowSystems) {
                target.receiveInput("transmitter_modulation", state);
            }
        }
    }

    private void updateWorkingMemoryParameters() {
        // Learned weights may affect working memory dynamics
        var verySlowSystems = systemsByTimeScale.get(DynamicalSystem.TimeScale.VERY_SLOW);
        var fastSystems = systemsByTimeScale.get(DynamicalSystem.TimeScale.FAST);

        for (var source : verySlowSystems) {
            var state = source.getCurrentState();
            for (var target : fastSystems) {
                target.receiveInput("learned_weights", state);
            }
        }
    }

    private void handleError(DynamicalSystem.TimeScale timeScale, Throwable error) {
        System.err.println("Error in time scale " + timeScale + ": " + error.getMessage());
        eventBus.publish(new ErrorEvent(timeScale, error, System.currentTimeMillis()));
    }

    @Override
    public void close() {
        stop();
        virtualThreadExecutor.shutdown();
        try {
            if (!virtualThreadExecutor.awaitTermination(10, TimeUnit.SECONDS)) {
                virtualThreadExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            virtualThreadExecutor.shutdownNow();
        }
    }

    /**
     * Runner for individual dynamical systems.
     */
    private static class SystemRunner<S extends State, P extends Parameters> {
        private final DynamicalSystem<S, P> system;
        private final P parameters;
        private final NumericalIntegrator<S, P> integrator;
        private final MpscArrayQueue<InputEvent> inputQueue;
        private volatile S currentState;
        private double currentTime;

        SystemRunner(DynamicalSystem<S, P> system, S initialState, P parameters,
                    NumericalIntegrator<S, P> integrator) {
            this.system = system;
            this.currentState = initialState;
            this.parameters = parameters;
            this.integrator = integrator;
            this.inputQueue = new MpscArrayQueue<>(1024);
            this.currentTime = 0.0;
        }

        void update() {
            // Process input events
            processInputs();

            // Advance dynamics by one time step
            var dt = system.getTimeScale().getTypicalMillis() / 1000.0;
            currentState = integrator.step(currentState, parameters, currentTime, dt).state();
            currentTime += dt;
        }

        void receiveInput(String type, State input) {
            inputQueue.offer(new InputEvent(type, input));
        }

        S getCurrentState() {
            return currentState;
        }

        private void processInputs() {
            InputEvent event;
            while ((event = inputQueue.poll()) != null) {
                // Process input based on type
                // This would be system-specific
            }
        }
    }

    /**
     * Event bus for cross-scale communication.
     */
    private static class EventBus {
        private final List<EventListener> listeners = new CopyOnWriteArrayList<>();

        void subscribe(EventListener listener) {
            listeners.add(listener);
        }

        void publish(Event event) {
            for (var listener : listeners) {
                listener.onEvent(event);
            }
        }
    }

    // Event types
    record SynchronizationEvent(DynamicalSystem.TimeScale scale, long timestamp) implements Event {}
    record ErrorEvent(DynamicalSystem.TimeScale scale, Throwable error, long timestamp) implements Event {}
    record InputEvent(String type, State data) implements Event {}

    interface Event {}
    interface EventListener {
        void onEvent(Event event);
    }
}