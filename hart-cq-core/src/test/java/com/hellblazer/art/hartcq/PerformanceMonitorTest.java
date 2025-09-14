package com.hellblazer.art.hartcq;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.AfterEach;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.*;
import static java.util.concurrent.TimeUnit.MILLISECONDS;
import static java.util.concurrent.TimeUnit.SECONDS;

/**
 * Comprehensive test suite for PerformanceMonitor.
 * Tests throughput calculation, latency monitoring, resource tracking, and reporting functionality.
 */
class PerformanceMonitorTest {

    private PerformanceMonitor monitor;

    @BeforeEach
    void setUp() {
        monitor = new PerformanceMonitor(150, 1, null); // 150 sentences/sec target, 1 sec reports
    }

    @AfterEach
    void tearDown() {
        if (monitor != null) {
            monitor.close();
        }
    }

    @Nested
    @DisplayName("Basic Performance Recording")
    class BasicPerformanceRecording {

        @Test
        @DisplayName("Should record sentences processed correctly")
        void shouldRecordSentencesProcessed() {
            // Given
            monitor.startMonitoring();

            // When
            monitor.recordSentencesProcessed(10);
            monitor.recordSentencesProcessed(5);

            // Then
            var report = monitor.generatePerformanceReport();
            assertThat(report.getTotalSentencesProcessed()).isEqualTo(15);
        }

        @Test
        @DisplayName("Should record tokens processed correctly")
        void shouldRecordTokensProcessed() {
            // Given
            monitor.startMonitoring();

            // When
            monitor.recordTokensProcessed(100);
            monitor.recordTokensProcessed(50);

            // Then
            var report = monitor.generatePerformanceReport();
            assertThat(report.getTotalTokensProcessed()).isEqualTo(150);
        }

        @Test
        @DisplayName("Should record windows processed correctly")
        void shouldRecordWindowsProcessed() {
            // Given
            monitor.startMonitoring();

            // When
            monitor.recordWindowsProcessed(3);
            monitor.recordWindowsProcessed(2);

            // Then
            var report = monitor.generatePerformanceReport();
            assertThat(report.getTotalWindowsProcessed()).isEqualTo(5);
        }

        @Test
        @DisplayName("Should record errors correctly")
        void shouldRecordErrors() {
            // Given
            monitor.startMonitoring();

            // When
            monitor.recordError();
            monitor.recordError();
            monitor.recordError();

            // Then
            var report = monitor.generatePerformanceReport();
            assertThat(report.getTotalErrors()).isEqualTo(3);
        }

        @Test
        @DisplayName("Should record processing times correctly")
        void shouldRecordProcessingTimes() {
            // Given
            monitor.startMonitoring();

            // When
            monitor.recordProcessingTime(1_000_000L); // 1ms
            monitor.recordProcessingTime(2_000_000L); // 2ms
            monitor.recordProcessingTime(500_000L);   // 0.5ms

            // Then
            var report = monitor.generatePerformanceReport();
            assertThat(report.getMinProcessingTimeMs()).isEqualTo(0.5, within(0.01));
            assertThat(report.getMaxProcessingTimeMs()).isEqualTo(2.0, within(0.01));
        }

        @Test
        @DisplayName("Should record complete window processing")
        void shouldRecordCompleteWindowProcessing() {
            // Given
            monitor.startMonitoring();

            // When
            monitor.recordWindowProcessing(5, 25, 1_500_000L); // 5 sentences, 25 tokens, 1.5ms

            // Then
            var report = monitor.generatePerformanceReport();
            assertThat(report.getTotalSentencesProcessed()).isEqualTo(5);
            assertThat(report.getTotalTokensProcessed()).isEqualTo(25);
            assertThat(report.getTotalWindowsProcessed()).isEqualTo(1);
            assertThat(report.getMinProcessingTimeMs()).isEqualTo(1.5, within(0.01));
        }
    }

    @Nested
    @DisplayName("Throughput Calculation")
    class ThroughputCalculation {

        @Test
        @DisplayName("Should calculate sentences per second correctly")
        @Timeout(value = 3, unit = SECONDS)
        void shouldCalculateSentencesPerSecond() throws InterruptedException {
            // Given
            monitor.startMonitoring();

            // When - simulate processing over time
            monitor.recordSentencesProcessed(50);
            Thread.sleep(500); // Wait 0.5 seconds
            monitor.recordSentencesProcessed(50);

            // Then
            var throughput = monitor.getCurrentThroughputSentencesPerSecond();
            assertThat(throughput).isBetween(150.0, 250.0); // Should be around 200 sentences/sec
        }

        @Test
        @DisplayName("Should calculate tokens per second correctly")
        @Timeout(value = 3, unit = SECONDS)
        void shouldCalculateTokensPerSecond() throws InterruptedException {
            // Given
            monitor.startMonitoring();

            // When
            monitor.recordTokensProcessed(500);
            Thread.sleep(500);
            monitor.recordTokensProcessed(500);

            // Then
            var throughput = monitor.getCurrentThroughputTokensPerSecond();
            assertThat(throughput).isBetween(1500.0, 2500.0); // Should be around 2000 tokens/sec
        }

        @Test
        @DisplayName("Should handle zero elapsed time gracefully")
        void shouldHandleZeroElapsedTime() {
            // Given
            monitor.startMonitoring();

            // When - record immediately without any time passing
            monitor.recordSentencesProcessed(10);

            // Then - should not throw and return 0 or very high number
            var throughput = monitor.getCurrentThroughputSentencesPerSecond();
            assertThat(throughput).isGreaterThanOrEqualTo(0.0);
        }

        @Test
        @DisplayName("Should check throughput target correctly")
        @Timeout(value = 3, unit = SECONDS)
        void shouldCheckThroughputTarget() throws InterruptedException {
            // Given - monitor with 100 sentences/sec target
            try (var lowTargetMonitor = new PerformanceMonitor(100, 1, null)) {
                lowTargetMonitor.startMonitoring();

                // When - achieve high throughput
                lowTargetMonitor.recordSentencesProcessed(200);
                Thread.sleep(500);

                // Then
                assertThat(lowTargetMonitor.isMeetingThroughputTarget()).isTrue();
            }
        }

        @Test
        @DisplayName("Should detect when not meeting throughput target")
        @Timeout(value = 3, unit = SECONDS)
        void shouldDetectNotMeetingTarget() throws InterruptedException {
            // Given - monitor with very high target
            try (var highTargetMonitor = new PerformanceMonitor(10000, 1, null)) {
                highTargetMonitor.startMonitoring();

                // When - achieve low throughput
                highTargetMonitor.recordSentencesProcessed(10);
                Thread.sleep(1000);

                // Then
                assertThat(highTargetMonitor.isMeetingThroughputTarget()).isFalse();
            }
        }
    }

    @Nested
    @DisplayName("Latency Monitoring")
    class LatencyMonitoring {

        @Test
        @DisplayName("Should calculate average processing time correctly")
        void shouldCalculateAverageProcessingTime() {
            // Given
            monitor.startMonitoring();

            // When
            monitor.recordSentencesProcessed(1);
            monitor.recordProcessingTime(1_000_000L); // 1ms
            monitor.recordSentencesProcessed(1);
            monitor.recordProcessingTime(3_000_000L); // 3ms

            // Then
            var avgTime = monitor.getAverageProcessingTimeMs();
            assertThat(avgTime).isEqualTo(2.0, within(0.01)); // (1 + 3) / 2 = 2ms
        }

        @Test
        @DisplayName("Should track minimum processing time correctly")
        void shouldTrackMinimumProcessingTime() {
            // Given
            monitor.startMonitoring();

            // When
            monitor.recordProcessingTime(5_000_000L); // 5ms
            monitor.recordProcessingTime(2_000_000L); // 2ms (minimum)
            monitor.recordProcessingTime(8_000_000L); // 8ms

            // Then
            var report = monitor.generatePerformanceReport();
            assertThat(report.getMinProcessingTimeMs()).isEqualTo(2.0, within(0.01));
        }

        @Test
        @DisplayName("Should track maximum processing time correctly")
        void shouldTrackMaximumProcessingTime() {
            // Given
            monitor.startMonitoring();

            // When
            monitor.recordProcessingTime(3_000_000L); // 3ms
            monitor.recordProcessingTime(7_000_000L); // 7ms (maximum)
            monitor.recordProcessingTime(1_000_000L); // 1ms

            // Then
            var report = monitor.generatePerformanceReport();
            assertThat(report.getMaxProcessingTimeMs()).isEqualTo(7.0, within(0.01));
        }

        @Test
        @DisplayName("Should handle no processing times gracefully")
        void shouldHandleNoProcessingTimes() {
            // Given
            monitor.startMonitoring();

            // When - no processing times recorded

            // Then
            var avgTime = monitor.getAverageProcessingTimeMs();
            assertThat(avgTime).isEqualTo(0.0);

            var report = monitor.generatePerformanceReport();
            assertThat(report.getMinProcessingTimeMs()).isEqualTo(0.0);
            assertThat(report.getMaxProcessingTimeMs()).isEqualTo(0.0);
        }
    }

    @Nested
    @DisplayName("Resource Tracking")
    class ResourceTracking {

        @Test
        @DisplayName("Should get CPU usage")
        void shouldGetCpuUsage() {
            // Given
            monitor.startMonitoring();

            // When
            var cpuUsage = monitor.getCpuUsage();

            // Then
            assertThat(cpuUsage).isBetween(-1.0, 1.0); // CPU load can be -1 if not available
        }

        @Test
        @DisplayName("Should get memory usage")
        void shouldGetMemoryUsage() {
            // Given
            monitor.startMonitoring();

            // When
            var memoryUsage = monitor.getMemoryUsage();

            // Then
            assertThat(memoryUsage).isBetween(0.0, 1.0);
        }

        @Test
        @DisplayName("Should include system metrics in report")
        void shouldIncludeSystemMetricsInReport() {
            // Given
            monitor.startMonitoring();

            // When
            var report = monitor.generatePerformanceReport();

            // Then
            assertThat(report.getCpuUsage()).isBetween(-1.0, 1.0);
            assertThat(report.getMemoryUsage()).isBetween(0.0, 1.0);
        }
    }

    @Nested
    @DisplayName("Performance Reporting")
    class PerformanceReporting {

        @Test
        @DisplayName("Should generate comprehensive performance report")
        void shouldGenerateComprehensiveReport() {
            // Given
            monitor.startMonitoring();
            monitor.recordSentencesProcessed(100);
            monitor.recordTokensProcessed(500);
            monitor.recordWindowsProcessed(10);
            monitor.recordError();
            monitor.recordProcessingTime(2_000_000L);

            // When
            var report = monitor.generatePerformanceReport();

            // Then
            assertThat(report).isNotNull();
            assertThat(report.getTimestamp()).isCloseTo(System.currentTimeMillis(), within(1000L));
            assertThat(report.getTotalSentencesProcessed()).isEqualTo(100);
            assertThat(report.getTotalTokensProcessed()).isEqualTo(500);
            assertThat(report.getTotalWindowsProcessed()).isEqualTo(10);
            assertThat(report.getTotalErrors()).isEqualTo(1);
            assertThat(report.getTargetThroughputSentencesPerSecond()).isEqualTo(150);
            assertThat(report.getElapsedTimeSeconds()).isGreaterThan(0.0);
        }

        @Test
        @DisplayName("Should calculate error rate correctly")
        void shouldCalculateErrorRate() {
            // Given
            monitor.startMonitoring();
            monitor.recordSentencesProcessed(50);
            monitor.recordWindowsProcessed(50);
            monitor.recordError();
            monitor.recordError();

            // When
            var report = monitor.generatePerformanceReport();

            // Then
            assertThat(report.getErrorRate()).isEqualTo(0.02, within(0.001)); // 2 errors / 100 operations = 2%
        }

        @Test
        @DisplayName("Should handle zero operations for error rate")
        void shouldHandleZeroOperationsForErrorRate() {
            // Given
            monitor.startMonitoring();
            monitor.recordError();

            // When
            var report = monitor.generatePerformanceReport();

            // Then
            assertThat(report.getErrorRate()).isEqualTo(0.0);
        }

        @Test
        @DisplayName("Should format report string correctly")
        void shouldFormatReportString() {
            // Given
            monitor.startMonitoring();
            monitor.recordSentencesProcessed(100);
            monitor.recordTokensProcessed(500);

            // When
            var report = monitor.generatePerformanceReport();
            var reportString = report.toString();

            // Then
            assertThat(reportString).contains("sentences=100");
            assertThat(reportString).contains("tokens=500");
            assertThat(reportString).contains("/sec");
        }

        @Test
        @DisplayName("Should call report consumer when provided")
        @Timeout(value = 5, unit = SECONDS)
        void shouldCallReportConsumer() throws InterruptedException {
            // Given
            var reportReceived = new CountDownLatch(1);
            var receivedReport = new AtomicReference<PerformanceMonitor.PerformanceReport>();

            try (var monitorWithConsumer = new PerformanceMonitor(100, 1, report -> {
                receivedReport.set(report);
                reportReceived.countDown();
            })) {

                // When
                monitorWithConsumer.startMonitoring();
                monitorWithConsumer.recordSentencesProcessed(10);

                // Wait for report to be generated (should happen within 1 second)
                assertThat(reportReceived.await(2, SECONDS)).isTrue();

                // Then
                assertThat(receivedReport.get()).isNotNull();
                assertThat(receivedReport.get().getTotalSentencesProcessed()).isEqualTo(10);
            }
        }
    }

    @Nested
    @DisplayName("Monitoring Lifecycle")
    class MonitoringLifecycle {

        @Test
        @DisplayName("Should start monitoring correctly")
        void shouldStartMonitoring() {
            // Given
            assertThat(monitor.getCurrentThroughputSentencesPerSecond()).isEqualTo(0.0);

            // When
            monitor.startMonitoring();
            monitor.recordSentencesProcessed(10);

            // Then
            assertThat(monitor.getCurrentThroughputSentencesPerSecond()).isGreaterThanOrEqualTo(0.0);
        }

        @Test
        @DisplayName("Should stop monitoring correctly")
        @Timeout(value = 5, unit = SECONDS)
        void shouldStopMonitoring() {
            // Given
            monitor.startMonitoring();
            monitor.recordSentencesProcessed(10);

            // When
            monitor.stopMonitoring();

            // Then - should still be able to get final report
            var report = monitor.generatePerformanceReport();
            assertThat(report.getTotalSentencesProcessed()).isEqualTo(10);
        }

        @Test
        @DisplayName("Should handle multiple start calls gracefully")
        void shouldHandleMultipleStartCalls() {
            // Given
            monitor.startMonitoring();
            var firstStartTime = System.nanoTime();

            // When
            monitor.startMonitoring(); // Second call should be ignored
            var secondStartTime = System.nanoTime();

            // Then - should not restart timing
            monitor.recordSentencesProcessed(10);
            var throughput = monitor.getCurrentThroughputSentencesPerSecond();
            assertThat(throughput).isGreaterThanOrEqualTo(0.0);
        }

        @Test
        @DisplayName("Should handle multiple stop calls gracefully")
        void shouldHandleMultipleStopCalls() {
            // Given
            monitor.startMonitoring();
            monitor.recordSentencesProcessed(10);

            // When
            monitor.stopMonitoring();
            monitor.stopMonitoring(); // Second call should be safe

            // Then - should not throw
            var report = monitor.generatePerformanceReport();
            assertThat(report).isNotNull();
        }

        @Test
        @DisplayName("Should reset counters correctly")
        void shouldResetCounters() {
            // Given
            monitor.startMonitoring();
            monitor.recordSentencesProcessed(100);
            monitor.recordTokensProcessed(500);
            monitor.recordError();
            monitor.recordProcessingTime(1_000_000L);

            // When
            monitor.reset();

            // Then
            var report = monitor.generatePerformanceReport();
            assertThat(report.getTotalSentencesProcessed()).isEqualTo(0);
            assertThat(report.getTotalTokensProcessed()).isEqualTo(0);
            assertThat(report.getTotalErrors()).isEqualTo(0);
            assertThat(report.getMinProcessingTimeMs()).isEqualTo(0.0);
            assertThat(report.getMaxProcessingTimeMs()).isEqualTo(0.0);
        }

        @Test
        @DisplayName("Should close resources properly")
        void shouldCloseResourcesProperly() {
            // Given
            monitor.startMonitoring();
            monitor.recordSentencesProcessed(10);

            // When
            monitor.close();

            // Then - should be able to get final report
            var report = monitor.generatePerformanceReport();
            assertThat(report.getTotalSentencesProcessed()).isEqualTo(10);
        }
    }

    @Nested
    @DisplayName("Thread Safety")
    class ThreadSafety {

        @Test
        @DisplayName("Should handle concurrent sentence recording")
        @Timeout(value = 10, unit = SECONDS)
        void shouldHandleConcurrentSentenceRecording() throws InterruptedException {
            // Given
            monitor.startMonitoring();
            var numThreads = 10;
            var sentencesPerThread = 100;
            var executor = Executors.newFixedThreadPool(numThreads);

            try {
                // When
                var futures = new CompletableFuture[numThreads];
                for (int i = 0; i < numThreads; i++) {
                    futures[i] = CompletableFuture.runAsync(() -> {
                        for (int j = 0; j < sentencesPerThread; j++) {
                            monitor.recordSentencesProcessed(1);
                        }
                    }, executor);
                }

                CompletableFuture.allOf(futures).join();

                // Then
                var report = monitor.generatePerformanceReport();
                assertThat(report.getTotalSentencesProcessed()).isEqualTo(numThreads * sentencesPerThread);
            } finally {
                executor.shutdown();
            }
        }

        @Test
        @DisplayName("Should handle concurrent processing time recording")
        @Timeout(value = 10, unit = SECONDS)
        void shouldHandleConcurrentProcessingTimeRecording() throws InterruptedException {
            // Given
            monitor.startMonitoring();
            var numThreads = 10;
            var timesPerThread = 100;
            var executor = Executors.newFixedThreadPool(numThreads);

            try {
                // When
                var futures = new CompletableFuture[numThreads];
                for (int i = 0; i < numThreads; i++) {
                    var threadId = i;
                    futures[i] = CompletableFuture.runAsync(() -> {
                        for (int j = 0; j < timesPerThread; j++) {
                            // Use different times per thread to test min/max tracking
                            var processingTime = ((long) threadId + 1) * 1_000_000L; // 1ms, 2ms, 3ms, etc.
                            monitor.recordProcessingTime(processingTime);
                            monitor.recordSentencesProcessed(1); // For average calculation
                        }
                    }, executor);
                }

                CompletableFuture.allOf(futures).join();

                // Then
                var report = monitor.generatePerformanceReport();
                assertThat(report.getMinProcessingTimeMs()).isEqualTo(1.0, within(0.01));
                assertThat(report.getMaxProcessingTimeMs()).isEqualTo(10.0, within(0.01));
                assertThat(report.getTotalSentencesProcessed()).isEqualTo(numThreads * timesPerThread);
            } finally {
                executor.shutdown();
            }
        }

        @Test
        @DisplayName("Should handle concurrent error recording")
        @Timeout(value = 10, unit = SECONDS)
        void shouldHandleConcurrentErrorRecording() throws InterruptedException {
            // Given
            monitor.startMonitoring();
            var numThreads = 10;
            var errorsPerThread = 50;
            var executor = Executors.newFixedThreadPool(numThreads);

            try {
                // When
                var futures = new CompletableFuture[numThreads];
                for (int i = 0; i < numThreads; i++) {
                    futures[i] = CompletableFuture.runAsync(() -> {
                        for (int j = 0; j < errorsPerThread; j++) {
                            monitor.recordError();
                        }
                    }, executor);
                }

                CompletableFuture.allOf(futures).join();

                // Then
                var report = monitor.generatePerformanceReport();
                assertThat(report.getTotalErrors()).isEqualTo(numThreads * errorsPerThread);
            } finally {
                executor.shutdown();
            }
        }

        @Test
        @DisplayName("Should handle concurrent report generation")
        @Timeout(value = 10, unit = SECONDS)
        void shouldHandleConcurrentReportGeneration() throws InterruptedException {
            // Given
            monitor.startMonitoring();
            monitor.recordSentencesProcessed(100);
            var numThreads = 20;
            var executor = Executors.newFixedThreadPool(numThreads);

            try {
                // When - generate reports concurrently
                var futures = new CompletableFuture[numThreads];
                var reports = new PerformanceMonitor.PerformanceReport[numThreads];

                for (int i = 0; i < numThreads; i++) {
                    var threadId = i;
                    futures[i] = CompletableFuture.runAsync(() -> {
                        reports[threadId] = monitor.generatePerformanceReport();
                    }, executor);
                }

                CompletableFuture.allOf(futures).join();

                // Then - all reports should be valid and consistent
                for (var report : reports) {
                    assertThat(report).isNotNull();
                    assertThat(report.getTotalSentencesProcessed()).isEqualTo(100);
                }
            } finally {
                executor.shutdown();
            }
        }
    }

    @Nested
    @DisplayName("Performance Benchmarks")
    class PerformanceBenchmarks {

        @Test
        @DisplayName("Should handle high-frequency recording efficiently")
        @Timeout(value = 5, unit = SECONDS)
        void shouldHandleHighFrequencyRecording() {
            // Given
            monitor.startMonitoring();
            var numOperations = 100_000;

            // When
            var startTime = System.nanoTime();

            for (int i = 0; i < numOperations; i++) {
                monitor.recordSentencesProcessed(1);
                monitor.recordProcessingTime(1_000_000L); // 1ms
            }

            var endTime = System.nanoTime();
            var durationMs = (endTime - startTime) / 1_000_000.0;

            // Then
            var report = monitor.generatePerformanceReport();
            assertThat(report.getTotalSentencesProcessed()).isEqualTo(numOperations);

            // Should complete in reasonable time (less than 1 second for 100k operations)
            assertThat(durationMs).isLessThan(1000.0);

            // Performance should be good (> 100k operations/second)
            var operationsPerSecond = numOperations / (durationMs / 1000.0);
            assertThat(operationsPerSecond).isGreaterThan(100_000.0);
        }

        @Test
        @DisplayName("Should demonstrate target throughput validation")
        @Timeout(value = 3, unit = SECONDS)
        void shouldDemonstrateTargetThroughputValidation() throws InterruptedException {
            // Given - monitor targeting >100 sentences/second
            try (var testMonitor = new PerformanceMonitor(100, 1, null)) {
                testMonitor.startMonitoring();

                // When - simulate achieving target throughput
                for (int i = 0; i < 10; i++) {
                    testMonitor.recordSentencesProcessed(15); // 150 total over ~1 second
                    Thread.sleep(100); // 100ms intervals
                }

                // Then
                assertThat(testMonitor.isMeetingThroughputTarget()).isTrue();

                var report = testMonitor.generatePerformanceReport();
                assertThat(report.getCurrentThroughputSentencesPerSecond()).isGreaterThan(100.0);
                assertThat(report.isMeetingTarget()).isTrue();
            }
        }

        @Test
        @DisplayName("Should measure memory efficiency during intensive operations")
        @Timeout(value = 10, unit = SECONDS)
        void shouldMeasureMemoryEfficiency() {
            // Given
            monitor.startMonitoring();
            var initialMemoryUsage = monitor.getMemoryUsage();

            // When - perform many operations
            for (int i = 0; i < 10_000; i++) {
                monitor.recordWindowProcessing(5, 25, 1_000_000L);
            }

            // Then - memory usage should not grow excessively
            var finalMemoryUsage = monitor.getMemoryUsage();
            var memoryGrowth = finalMemoryUsage - initialMemoryUsage;

            assertThat(memoryGrowth).isLessThan(0.1); // Less than 10% memory growth

            var report = monitor.generatePerformanceReport();
            assertThat(report.getTotalSentencesProcessed()).isEqualTo(50_000);
            assertThat(report.getTotalTokensProcessed()).isEqualTo(250_000);
            assertThat(report.getTotalWindowsProcessed()).isEqualTo(10_000);
        }
    }

    @Nested
    @DisplayName("Edge Cases and Error Handling")
    class EdgeCasesAndErrorHandling {

        @Test
        @DisplayName("Should handle negative processing times gracefully")
        void shouldHandleNegativeProcessingTimes() {
            // Given
            monitor.startMonitoring();

            // When - record negative time (shouldn't happen but test robustness)
            monitor.recordProcessingTime(-1_000_000L);
            monitor.recordSentencesProcessed(1);

            // Then - should not crash and handle gracefully
            var report = monitor.generatePerformanceReport();
            assertThat(report).isNotNull();
            assertThat(report.getMinProcessingTimeMs()).isLessThanOrEqualTo(0.0);
        }

        @Test
        @DisplayName("Should handle zero and negative counts gracefully")
        void shouldHandleZeroAndNegativeCounts() {
            // Given
            monitor.startMonitoring();

            // When
            monitor.recordSentencesProcessed(0);
            monitor.recordTokensProcessed(-5); // Shouldn't happen but test robustness
            monitor.recordWindowsProcessed(0);

            // Then
            var report = monitor.generatePerformanceReport();
            assertThat(report).isNotNull();
            // Values might be weird but shouldn't crash
        }

        @Test
        @DisplayName("Should handle very large numbers gracefully")
        void shouldHandleVeryLargeNumbers() {
            // Given
            monitor.startMonitoring();

            // When
            monitor.recordSentencesProcessed(Integer.MAX_VALUE);
            monitor.recordTokensProcessed(Integer.MAX_VALUE);
            monitor.recordProcessingTime(Long.MAX_VALUE / 2);

            // Then
            var report = monitor.generatePerformanceReport();
            assertThat(report).isNotNull();
            assertThat(report.getTotalSentencesProcessed()).isGreaterThan(0);
        }

        @Test
        @DisplayName("Should create monitor with minimum valid parameters")
        void shouldCreateMonitorWithMinimumValidParameters() {
            // When
            try (var minMonitor = new PerformanceMonitor(1, 1, null)) {
                // Then
                assertThat(minMonitor).isNotNull();

                minMonitor.startMonitoring();
                var report = minMonitor.generatePerformanceReport();
                assertThat(report.getTargetThroughputSentencesPerSecond()).isEqualTo(1);
            }
        }

        @Test
        @DisplayName("Should handle zero/negative constructor parameters gracefully")
        void shouldHandleZeroNegativeConstructorParameters() {
            // When
            try (var zeroMonitor = new PerformanceMonitor(0, 0, null)) {
                // Then - should use minimum values
                zeroMonitor.startMonitoring();
                var report = zeroMonitor.generatePerformanceReport();
                assertThat(report.getTargetThroughputSentencesPerSecond()).isGreaterThan(0);
            }
        }
    }
}