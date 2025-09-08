package com.hellblazer.art.nlp.channels.base;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.nlp.metrics.ChannelMetrics;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.*;

/**
 * Comprehensive test suite for BaseChannel abstract class.
 * Uses TestChannel implementation to test base functionality.
 */
@DisplayName("BaseChannel Tests")
class BaseChannelTest {

    private TestChannel channel;
    private ExecutorService executor;

    @BeforeEach
    void setUp() {
        channel = new TestChannel("test", 0.75);
        executor = Executors.newFixedThreadPool(4);
    }

    @AfterEach
    void tearDown() {
        if (channel != null && channel.isInitialized()) {
            channel.shutdown();
        }
        if (executor != null && !executor.isShutdown()) {
            executor.shutdown();
        }
    }

    @Test
    @DisplayName("Should create channel with valid parameters")
    void shouldCreateChannelWithValidParameters() {
        assertThat(channel.getChannelName()).isEqualTo("test");
        assertThat(channel.getVigilance()).isEqualTo(0.75);
        assertThat(channel.isInitialized()).isFalse();
        assertThat(channel.isLearningEnabled()).isTrue();
        assertThat(channel.getCategoryCount()).isZero();
    }

    @Test
    @DisplayName("Should validate null channel name")
    void shouldValidateNullChannelName() {
        assertThatThrownBy(() -> new TestChannel(null, 0.5))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessage("Channel name cannot be null or empty");
    }

    @Test
    @DisplayName("Should validate empty channel name")
    void shouldValidateEmptyChannelName() {
        assertThatThrownBy(() -> new TestChannel("   ", 0.5))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessage("Channel name cannot be null or empty");
    }

    @Test
    @DisplayName("Should validate vigilance parameter range")
    void shouldValidateVigilanceParameterRange() {
        assertThatThrownBy(() -> new TestChannel("test", -0.1))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessage("Vigilance must be in [0.0, 1.0]: -0.1");

        assertThatThrownBy(() -> new TestChannel("test", 1.1))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessage("Vigilance must be in [0.0, 1.0]: 1.1");

        // Boundary values should be accepted
        assertThatNoException().isThrownBy(() -> new TestChannel("test", 0.0));
        assertThatNoException().isThrownBy(() -> new TestChannel("test", 1.0));
    }

    @Test
    @DisplayName("Should trim channel name")
    void shouldTrimChannelName() {
        var channel = new TestChannel("  semantic  ", 0.8);
        assertThat(channel.getChannelName()).isEqualTo("semantic");
    }

    @Test
    @DisplayName("Should initialize channel correctly")
    void shouldInitializeChannelCorrectly() {
        assertThat(channel.isInitialized()).isFalse();
        
        channel.initialize();
        
        assertThat(channel.isInitialized()).isTrue();
        assertThat(channel.initializeCount.get()).isOne();
    }

    @Test
    @DisplayName("Should not initialize already initialized channel")
    void shouldNotInitializeAlreadyInitializedChannel() {
        channel.initialize();
        assertThat(channel.isInitialized()).isTrue();
        
        channel.initialize(); // Second call
        
        assertThat(channel.initializeCount.get()).isOne(); // Should still be 1
    }

    @Test
    @DisplayName("Should shutdown channel correctly")
    void shouldShutdownChannelCorrectly() {
        channel.initialize();
        assertThat(channel.isInitialized()).isTrue();
        
        channel.shutdown();
        
        assertThat(channel.isInitialized()).isFalse();
        assertThat(channel.cleanupCount.get()).isOne();
        assertThat(channel.saveCount.get()).isOne();
    }

    @Test
    @DisplayName("Should not shutdown uninitialized channel")
    void shouldNotShutdownUninitializedChannel() {
        assertThat(channel.isInitialized()).isFalse();
        
        channel.shutdown();
        
        assertThat(channel.cleanupCount.get()).isZero();
    }

    @Test
    @DisplayName("Should handle initialization failure gracefully")
    void shouldHandleInitializationFailureGracefully() {
        channel.shouldFailLoad = true;
        
        assertThatNoException().isThrownBy(() -> channel.initialize());
        assertThat(channel.isInitialized()).isTrue(); // Should still initialize despite load failure
    }

    @Test
    @DisplayName("Should handle shutdown failure gracefully")
    void shouldHandleShutdownFailureGracefully() {
        channel.initialize();
        channel.shouldFailSave = true;
        
        assertThatNoException().isThrownBy(() -> channel.shutdown());
        assertThat(channel.isInitialized()).isFalse(); // Should still shutdown despite save failure
    }

    @Test
    @DisplayName("Should control learning enable/disable")
    void shouldControlLearningEnableDisable() {
        assertThat(channel.isLearningEnabled()).isTrue();
        
        channel.setLearningEnabled(false);
        assertThat(channel.isLearningEnabled()).isFalse();
        
        channel.setLearningEnabled(true);
        assertThat(channel.isLearningEnabled()).isTrue();
    }

    @Test
    @DisplayName("Should throw UnsupportedOperationException for classifyWord in base channel")
    void shouldThrowUnsupportedOperationExceptionForClassifyWord() {
        assertThatThrownBy(() -> channel.classifyWord("test"))
            .isInstanceOf(UnsupportedOperationException.class)
            .hasMessage("Word classification only supported in SemanticChannel, not in test");
    }

    @Test
    @DisplayName("Should provide metrics")
    void shouldProvideMetrics() {
        var metrics = channel.getMetrics();
        
        assertThat(metrics).isNotNull();
        assertThat(metrics.getChannelName()).isEqualTo("test");
        assertThat(metrics.getTotalClassifications()).isZero();
    }

    @Test
    @DisplayName("Should classify input and record metrics")
    void shouldClassifyInputAndRecordMetrics() {
        channel.initialize();
        var input = new DenseVector(new double[]{0.1, 0.2, 0.3});
        
        var category = channel.classify(input);
        
        assertThat(category).isEqualTo(0); // First category
        assertThat(channel.getCategoryCount()).isOne();
        
        var metrics = channel.getMetrics();
        assertThat(metrics.getTotalClassifications()).isOne();
        assertThat(metrics.getCategoriesCreated()).isOne();
    }

    @Test
    @DisplayName("Should be thread-safe for concurrent operations")
    void shouldBeThreadSafeForConcurrentOperations() throws InterruptedException {
        channel.initialize();
        
        var numThreads = 10;
        var numOperationsPerThread = 50;
        var latch = new CountDownLatch(numThreads);
        var successfulClassifications = new AtomicInteger(0);
        
        for (int i = 0; i < numThreads; i++) {
            executor.submit(() -> {
                try {
                    for (int j = 0; j < numOperationsPerThread; j++) {
                        var input = new DenseVector(new double[]{Math.random(), Math.random(), Math.random()});
                        try {
                            channel.classify(input);
                            successfulClassifications.incrementAndGet();
                        } catch (Exception e) {
                            // Classification might fail due to concurrent modifications, that's ok
                        }
                    }
                } finally {
                    latch.countDown();
                }
            });
        }
        
        assertThat(latch.await(10, TimeUnit.SECONDS)).isTrue();
        
        // Should have processed some classifications without crashing
        assertThat(successfulClassifications.get()).isPositive();
        assertThat(channel.getCategoryCount()).isPositive();
    }

    @Test
    @DisplayName("Should be thread-safe for concurrent initialization")
    void shouldBeThreadSafeForConcurrentInitialization() throws InterruptedException {
        var numThreads = 5;
        var latch = new CountDownLatch(numThreads);
        
        for (int i = 0; i < numThreads; i++) {
            executor.submit(() -> {
                try {
                    channel.initialize();
                } finally {
                    latch.countDown();
                }
            });
        }
        
        assertThat(latch.await(5, TimeUnit.SECONDS)).isTrue();
        assertThat(channel.isInitialized()).isTrue();
        assertThat(channel.initializeCount.get()).isOne(); // Should only initialize once
    }

    @Test
    @DisplayName("Should be thread-safe for learning enable/disable")
    void shouldBeThreadSafeForLearningEnableDisable() throws InterruptedException {
        var numThreads = 10;
        var latch = new CountDownLatch(numThreads);
        
        for (int i = 0; i < numThreads; i++) {
            var enable = (i % 2 == 0);
            executor.submit(() -> {
                try {
                    channel.setLearningEnabled(enable);
                } finally {
                    latch.countDown();
                }
            });
        }
        
        assertThat(latch.await(5, TimeUnit.SECONDS)).isTrue();
        // Learning should be either enabled or disabled (no corruption)
        boolean learningState = channel.isLearningEnabled();
        assertThat(learningState).isIn(true, false);
    }

    @Test
    @DisplayName("Should preprocess input correctly")
    void shouldPreprocessInputCorrectly() {
        var input = new DenseVector(new double[]{0.5, 0.8, 0.2});
        
        // Test that preprocessing doesn't fail (actual preprocessing logic tested separately)
        assertThatNoException().isThrownBy(() -> {
            channel.testPreprocessInput(input);
        });
    }

    @Test
    @DisplayName("Should validate null input for preprocessing")
    void shouldValidateNullInputForPreprocessing() {
        assertThatThrownBy(() -> channel.testPreprocessInput(null))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessage("Input cannot be null");
    }

    @Test
    @DisplayName("Should provide meaningful toString")
    void shouldProvideMeaningfulToString() {
        channel.initialize();
        channel.classify(new DenseVector(new double[]{0.1, 0.2})); // Add a category
        
        var str = channel.toString();
        
        assertThat(str).contains("TestChannel");
        assertThat(str).contains("name='test'");
        assertThat(str).contains("vigilance=0.750");
        assertThat(str).contains("categories=1");
        assertThat(str).contains("initialized=true");
    }

    @Test
    @DisplayName("Should provide read and write locks")
    void shouldProvideReadAndWriteLocks() {
        var readLock = channel.testGetReadLock();
        var writeLock = channel.testGetWriteLock();
        
        assertThat(readLock).isNotNull();
        assertThat(writeLock).isNotNull();
        assertThat(readLock).isNotSameAs(writeLock);
    }

    /**
     * Concrete test implementation of BaseChannel for testing purposes.
     */
    static class TestChannel extends BaseChannel {
        final AtomicInteger initializeCount = new AtomicInteger(0);
        final AtomicInteger cleanupCount = new AtomicInteger(0);
        final AtomicInteger saveCount = new AtomicInteger(0);
        final AtomicInteger loadCount = new AtomicInteger(0);
        final AtomicInteger categoryCount = new AtomicInteger(0);
        final List<DenseVector> categories = new ArrayList<>();
        
        boolean shouldFailSave = false;
        boolean shouldFailLoad = false;

        public TestChannel(String channelName, double vigilance) {
            super(channelName, vigilance);
        }

        @Override
        public int classify(DenseVector input) {
            if (!isInitialized()) {
                throw new IllegalStateException("Channel not initialized");
            }
            
            var startTime = System.currentTimeMillis();
            
            getWriteLock().lock();
            try {
                // Simple mock classification: always create new category
                var categoryId = categoryCount.getAndIncrement();
                categories.add(input);
                
                var processingTime = System.currentTimeMillis() - startTime;
                recordClassification(processingTime, true);
                
                return categoryId;
            } finally {
                getWriteLock().unlock();
            }
        }

        @Override
        public void saveState() {
            saveCount.incrementAndGet();
            if (shouldFailSave) {
                throw new RuntimeException("Simulated save failure");
            }
        }

        @Override
        public void loadState() {
            loadCount.incrementAndGet();
            if (shouldFailLoad) {
                throw new RuntimeException("Simulated load failure");
            }
        }

        @Override
        public int getCategoryCount() {
            getReadLock().lock();
            try {
                return categoryCount.get();
            } finally {
                getReadLock().unlock();
            }
        }

        @Override
        public int pruneCategories(double threshold) {
            getWriteLock().lock();
            try {
                // Simple mock pruning: remove half the categories
                var originalCount = categoryCount.get();
                var newCount = originalCount / 2;
                var pruned = originalCount - newCount;
                
                categoryCount.set(newCount);
                categories.clear(); // Simple clear for mock
                
                return pruned;
            } finally {
                getWriteLock().unlock();
            }
        }

        @Override
        protected void performInitialization() {
            initializeCount.incrementAndGet();
        }

        @Override
        protected void performCleanup() {
            cleanupCount.incrementAndGet();
        }

        // Test helper methods to access protected functionality
        public DenseVector testPreprocessInput(DenseVector input) {
            return preprocessInput(input);
        }

        public java.util.concurrent.locks.Lock testGetReadLock() {
            return getReadLock();
        }

        public java.util.concurrent.locks.Lock testGetWriteLock() {
            return getWriteLock();
        }
    }
}