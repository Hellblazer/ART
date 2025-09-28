package com.hellblazer.art.hybrid.pan.performance;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

/**
 * Performance profiler for PAN operations.
 */
public class PANProfiler {

    private static final PANProfiler INSTANCE = new PANProfiler();

    private final ConcurrentHashMap<String, TimingStats> timings = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, LongAdder> counters = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, MemorySnapshot> memorySnapshots = new ConcurrentHashMap<>();

    private boolean enabled = false;

    private PANProfiler() {}

    public static PANProfiler getInstance() {
        return INSTANCE;
    }

    /**
     * Enable or disable profiling.
     */
    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
        if (!enabled) {
            clear();
        }
    }

    /**
     * Start timing an operation.
     */
    public Timer startTimer(String operation) {
        if (!enabled) {
            return Timer.NOOP;
        }
        return new Timer(operation, System.nanoTime(), this);
    }

    /**
     * Increment a counter.
     */
    public void incrementCounter(String name) {
        if (!enabled) return;
        counters.computeIfAbsent(name, k -> new LongAdder()).increment();
    }

    /**
     * Increment a counter by amount.
     */
    public void incrementCounter(String name, long amount) {
        if (!enabled) return;
        counters.computeIfAbsent(name, k -> new LongAdder()).add(amount);
    }

    /**
     * Take a memory snapshot.
     */
    public void takeMemorySnapshot(String label) {
        if (!enabled) return;

        Runtime runtime = Runtime.getRuntime();
        runtime.gc(); // Suggest GC for more accurate measurement

        long totalMemory = runtime.totalMemory();
        long freeMemory = runtime.freeMemory();
        long usedMemory = totalMemory - freeMemory;
        long maxMemory = runtime.maxMemory();

        memorySnapshots.put(label, new MemorySnapshot(
            usedMemory, totalMemory, maxMemory, System.currentTimeMillis()
        ));
    }

    /**
     * Get profiling report.
     */
    public String getReport() {
        if (!enabled) {
            return "Profiling is disabled";
        }

        StringBuilder sb = new StringBuilder();
        sb.append("\n=== PAN Performance Profile ===\n");

        // Timing statistics
        sb.append("\nOperation Timings:\n");
        sb.append(String.format("%-30s %10s %10s %10s %10s %10s\n",
            "Operation", "Count", "Total(ms)", "Avg(ms)", "Min(ms)", "Max(ms)"));
        sb.append("-".repeat(80)).append("\n");

        timings.entrySet().stream()
            .sorted((a, b) -> Long.compare(b.getValue().totalTime.get(), a.getValue().totalTime.get()))
            .forEach(entry -> {
                var stats = entry.getValue();
                long count = stats.count.get();
                if (count > 0) {
                    double totalMs = stats.totalTime.get() / 1_000_000.0;
                    double avgMs = totalMs / count;
                    double minMs = stats.minTime.get() / 1_000_000.0;
                    double maxMs = stats.maxTime.get() / 1_000_000.0;

                    sb.append(String.format("%-30s %10d %10.2f %10.2f %10.2f %10.2f\n",
                        entry.getKey(), count, totalMs, avgMs, minMs, maxMs));
                }
            });

        // Counters
        sb.append("\nCounters:\n");
        counters.entrySet().stream()
            .sorted((a, b) -> b.getKey().compareTo(a.getKey()))
            .forEach(entry -> {
                sb.append(String.format("  %-40s: %,d\n",
                    entry.getKey(), entry.getValue().sum()));
            });

        // Memory snapshots
        if (!memorySnapshots.isEmpty()) {
            sb.append("\nMemory Snapshots:\n");
            memorySnapshots.forEach((label, snapshot) -> {
                sb.append(String.format("  %s: %.2f MB used (%.2f MB total, %.2f MB max)\n",
                    label,
                    snapshot.usedMemory / (1024.0 * 1024.0),
                    snapshot.totalMemory / (1024.0 * 1024.0),
                    snapshot.maxMemory / (1024.0 * 1024.0)));
            });
        }

        // Top time consumers
        sb.append("\nTop Time Consumers:\n");
        timings.entrySet().stream()
            .sorted((a, b) -> Long.compare(b.getValue().totalTime.get(), a.getValue().totalTime.get()))
            .limit(5)
            .forEach(entry -> {
                double percentage = 100.0 * entry.getValue().totalTime.get() / getTotalTime();
                sb.append(String.format("  %-30s: %.1f%%\n", entry.getKey(), percentage));
            });

        return sb.toString();
    }

    /**
     * Clear all profiling data.
     */
    public void clear() {
        timings.clear();
        counters.clear();
        memorySnapshots.clear();
    }

    private long getTotalTime() {
        return timings.values().stream()
            .mapToLong(stats -> stats.totalTime.get())
            .sum();
    }

    /**
     * Timer for measuring operation duration.
     */
    public static class Timer implements AutoCloseable {
        public static final Timer NOOP = new Timer(null, 0, null);

        private final String operation;
        private final long startTime;
        private final PANProfiler profiler;

        private Timer(String operation, long startTime, PANProfiler profiler) {
            this.operation = operation;
            this.startTime = startTime;
            this.profiler = profiler;
        }

        @Override
        public void close() {
            if (this == NOOP || profiler == null || !profiler.enabled) return;

            long duration = System.nanoTime() - startTime;
            var stats = profiler.timings.computeIfAbsent(operation, k -> new TimingStats());
            stats.record(duration);
        }
    }

    /**
     * Timing statistics for an operation.
     */
    private static class TimingStats {
        private final AtomicLong count = new AtomicLong();
        private final AtomicLong totalTime = new AtomicLong();
        private final AtomicLong minTime = new AtomicLong(Long.MAX_VALUE);
        private final AtomicLong maxTime = new AtomicLong();

        void record(long nanos) {
            count.incrementAndGet();
            totalTime.addAndGet(nanos);
            minTime.updateAndGet(current -> Math.min(current, nanos));
            maxTime.updateAndGet(current -> Math.max(current, nanos));
        }
    }

    /**
     * Memory snapshot at a point in time.
     */
    private record MemorySnapshot(
        long usedMemory,
        long totalMemory,
        long maxMemory,
        long timestamp
    ) {}
}