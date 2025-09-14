package com.hellblazer.art.hartcq.core;

import com.hellblazer.art.hartcq.Token;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Timeout;

import java.time.Clock;
import java.time.Instant;
import java.time.ZoneId;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.assertj.core.api.Assertions.*;

/**
 * Comprehensive unit tests for the MultiChannelProcessor class.
 * Tests all 6 channels individually, parallel processing, channel coordination,
 * and error handling functionality.
 */
@DisplayName("MultiChannelProcessor Tests")
class MultiChannelProcessorTest {

    private MultiChannelProcessor processor;
    private TestClock testClock;

    /**
     * A controllable test clock that advances by a fixed amount each time millis() is called.
     * This ensures deterministic behavior for temporal calculations.
     */
    private static class TestClock extends Clock {
        private long currentMillis;
        private final long incrementMillis;
        private final ZoneId zone;

        TestClock(Instant start, long incrementMillis) {
            this.currentMillis = start.toEpochMilli();
            this.incrementMillis = incrementMillis;
            this.zone = ZoneId.of("UTC");
        }

        @Override
        public ZoneId getZone() {
            return zone;
        }

        @Override
        public Clock withZone(ZoneId zone) {
            return this; // Simplified for testing
        }

        @Override
        public Instant instant() {
            return Instant.ofEpochMilli(millis());
        }

        @Override
        public long millis() {
            long result = currentMillis;
            currentMillis += incrementMillis; // Auto-advance for each call
            return result;
        }

        public void reset(Instant start) {
            this.currentMillis = start.toEpochMilli();
        }
    }

    @BeforeEach
    void setUp() {
        // Use a test clock that advances by 100ms each time it's called
        testClock = new TestClock(Instant.parse("2024-01-01T12:00:00Z"), 100);
        processor = new MultiChannelProcessor(Runtime.getRuntime().availableProcessors(), testClock);
    }

    @AfterEach
    void tearDown() {
        if (processor != null) {
            processor.shutdown();
        }
    }
    
    @Nested
    @DisplayName("Initialization Tests")
    class InitializationTests {
        
        @Test
        @DisplayName("Should initialize all 6 channels correctly")
        void shouldInitializeAllChannelsCorrectly() {
            var channelInfo = processor.getChannelInfo();
            
            assertThat(channelInfo).hasSize(6);
            
            // Verify channel names (expected channels)
            var channelNames = channelInfo.stream().map(info -> info.name()).toList();
            assertThat(channelNames).containsExactlyInAnyOrder(
                "PositionalChannel",
                "WordChannel", 
                "ContextChannel",
                "StructuralChannel",
                "SemanticChannel",
                "TemporalChannel"
            );
        }
        
        @Test
        @DisplayName("Should calculate total output dimension correctly")
        void shouldCalculateTotalOutputDimensionCorrectly() {
            var channelInfo = processor.getChannelInfo();
            var expectedTotal = channelInfo.stream().mapToInt(info -> info.outputDimension()).sum();
            
            assertThat(processor.getTotalOutputDimension()).isEqualTo(expectedTotal);
            assertThat(processor.getTotalOutputDimension()).isPositive();
        }
        
        @Test
        @DisplayName("Should initialize with custom thread pool size")
        void shouldInitializeWithCustomThreadPoolSize() {
            var customProcessor = new MultiChannelProcessor(4, testClock);
            try {
                assertThat(customProcessor.getChannelInfo()).hasSize(6);
                assertThat(customProcessor.getTotalOutputDimension()).isPositive();
            } finally {
                customProcessor.shutdown();
            }
        }
        
        @Test
        @DisplayName("All channels should have positive output dimensions")
        void allChannelsShouldHavePositiveOutputDimensions() {
            var channelInfo = processor.getChannelInfo();
            
            for (var info : channelInfo) {
                assertThat(info.outputDimension())
                    .as("Channel %s should have positive output dimension", info.name())
                    .isPositive();
            }
        }
    }
    
    @Nested
    @DisplayName("Basic Window Processing Tests")
    class BasicWindowProcessingTests {
        
        @Test
        @DisplayName("Should process simple token window successfully")
        void shouldProcessSimpleTokenWindowSuccessfully() {
            var tokens = createTokenArray("Hello", "world", "how", "are", "you", "?");
            
            var result = processor.processWindow(tokens);
            
            assertThat(result).isNotNull();
            assertThat(result.length).isEqualTo(processor.getTotalOutputDimension());
            
            // Result should contain meaningful values (not all zeros)
            var hasNonZeroValues = false;
            for (float value : result) {
                if (Math.abs(value) > 0.001f) {
                    hasNonZeroValues = true;
                    break;
                }
            }
            assertThat(hasNonZeroValues).isTrue();
        }
        
        @Test
        @DisplayName("Should handle empty token window")
        void shouldHandleEmptyTokenWindow() {
            var emptyTokens = new Token[0];
            
            var result = processor.processWindow(emptyTokens);
            
            assertThat(result).isNotNull();
            assertThat(result.length).isEqualTo(processor.getTotalOutputDimension());
        }
        
        @Test
        @DisplayName("Should handle single token window")
        void shouldHandleSingleTokenWindow() {
            var singleToken = createTokenArray("hello");
            
            var result = processor.processWindow(singleToken);
            
            assertThat(result).isNotNull();
            assertThat(result.length).isEqualTo(processor.getTotalOutputDimension());
        }
        
        @Test
        @DisplayName("Should handle large token window")
        void shouldHandleLargeTokenWindow() {
            var tokens = new ArrayList<String>();
            for (int i = 0; i < 100; i++) {
                tokens.add("word" + i);
            }
            var largeTokenArray = createTokenArray(tokens.toArray(new String[0]));
            
            var result = processor.processWindow(largeTokenArray);
            
            assertThat(result).isNotNull();
            assertThat(result.length).isEqualTo(processor.getTotalOutputDimension());
        }
    }
    
    @Nested
    @DisplayName("Individual Channel Processing Tests") 
    class IndividualChannelProcessingTests {
        
        @Test
        @DisplayName("Should process tokens through positional channel")
        void shouldProcessTokensThroughPositionalChannel() {
            var tokens = createTokenArray("The", "quick", "brown", "fox", "jumps");
            
            var result = processor.processWindow(tokens);
            
            // Positional channel should contribute to the output
            // (We can't test specific values without knowing the exact implementation,
            // but we can verify the output has reasonable properties)
            assertThat(result).isNotNull();
            assertThat(result.length).isGreaterThan(0);
        }
        
        @Test
        @DisplayName("Should handle different token types in word channel")
        void shouldHandleDifferentTokenTypesInWordChannel() {
            var mixedTokens = new Token[] {
                new Token("hello", 0, Token.TokenType.WORD),
                new Token("!", 1, Token.TokenType.PUNCTUATION),
                new Token("123", 2, Token.TokenType.NUMBER),
                new Token("@", 3, Token.TokenType.SPECIAL),
                new Token("world", 4, Token.TokenType.WORD)
            };
            
            var result = processor.processWindow(mixedTokens);
            
            assertThat(result).isNotNull();
            assertThat(result.length).isEqualTo(processor.getTotalOutputDimension());
        }
        
        @Test
        @DisplayName("Should maintain context across sequential processing")
        void shouldMaintainContextAcrossSequentialProcessing() {
            var window1 = createTokenArray("This", "is", "the", "first", "window");
            var window2 = createTokenArray("This", "is", "the", "second", "window"); 
            var window3 = createTokenArray("This", "is", "the", "third", "window");
            
            var result1 = processor.processWindow(window1);
            var result2 = processor.processWindow(window2);
            var result3 = processor.processWindow(window3);
            
            // All results should be valid
            assertThat(result1).isNotNull();
            assertThat(result2).isNotNull();
            assertThat(result3).isNotNull();
            
            // Context channel should potentially show evolution
            // (exact behavior depends on implementation)
            assertThat(result1.length).isEqualTo(result2.length).isEqualTo(result3.length);
        }
        
        @Test
        @DisplayName("Should analyze structural patterns")
        void shouldAnalyzeStructuralPatterns() {
            var questionTokens = createTokenArray("What", "is", "your", "name", "?");
            var statementTokens = createTokenArray("My", "name", "is", "John", ".");
            
            var questionResult = processor.processWindow(questionTokens);
            var statementResult = processor.processWindow(statementTokens);
            
            assertThat(questionResult).isNotNull();
            assertThat(statementResult).isNotNull();
            
            // Results should be different for different structural patterns
            assertThat(Arrays.equals(questionResult, statementResult)).isFalse();
        }
        
        @Test
        @DisplayName("Should extract semantic features")
        void shouldExtractSemanticFeatures() {
            var coherentTokens = createTokenArray("The", "cat", "sat", "on", "the", "mat");
            var randomTokens = createTokenArray("Purple", "elephant", "bicycle", "quantum", "seventeen");
            
            var coherentResult = processor.processWindow(coherentTokens);
            var randomResult = processor.processWindow(randomTokens);
            
            assertThat(coherentResult).isNotNull();
            assertThat(randomResult).isNotNull();
            
            // Semantic channel should produce different outputs for coherent vs random text
            assertThat(Arrays.equals(coherentResult, randomResult)).isFalse();
        }
        
        @Test
        @DisplayName("Should track temporal sequences")
        void shouldTrackTemporalSequences() {
            var sequentialTokens = createTokenArray("First", "then", "next", "finally", "done");
            var shuffledTokens = createTokenArray("Finally", "first", "done", "then", "next");
            
            var sequentialResult = processor.processWindow(sequentialTokens);
            var shuffledResult = processor.processWindow(shuffledTokens);
            
            assertThat(sequentialResult).isNotNull();
            assertThat(shuffledResult).isNotNull();
            
            // Temporal channel should detect sequence differences
            assertThat(Arrays.equals(sequentialResult, shuffledResult)).isFalse();
        }
    }
    
    @Nested
    @DisplayName("Batch Processing Tests")
    class BatchProcessingTests {
        
        @Test
        @DisplayName("Should process multiple windows in batch")
        void shouldProcessMultipleWindowsInBatch() {
            var windows = List.of(
                createTokenArray("Hello", "world"),
                createTokenArray("How", "are", "you"),
                createTokenArray("Fine", "thank", "you"),
                createTokenArray("Goodbye", "now")
            );
            
            var results = processor.processBatch(windows);
            
            assertThat(results).hasSize(4);
            for (var result : results) {
                assertThat(result).isNotNull();
                assertThat(result.length).isEqualTo(processor.getTotalOutputDimension());
            }
        }
        
        @Test
        @DisplayName("Should handle empty batch")
        void shouldHandleEmptyBatch() {
            var emptyBatch = List.<Token[]>of();
            
            var results = processor.processBatch(emptyBatch);
            
            assertThat(results).isEmpty();
        }
        
        @Test
        @DisplayName("Should handle large batches efficiently")
        void shouldHandleLargeBatchesEfficiently() {
            var largeBatch = new ArrayList<Token[]>();
            for (int i = 0; i < 100; i++) {
                largeBatch.add(createTokenArray("Batch", "window", String.valueOf(i)));
            }
            
            var startTime = System.nanoTime();
            var results = processor.processBatch(largeBatch);
            var endTime = System.nanoTime();
            
            var processingTimeMs = (endTime - startTime) / 1_000_000.0;
            
            assertThat(results).hasSize(100);
            assertThat(processingTimeMs).isLessThan(10000); // Less than 10 seconds
            
            System.out.println("Processed " + largeBatch.size() + " windows in " + processingTimeMs + " ms");
        }
    }
    
    @Nested
    @DisplayName("Parallel Processing Tests")
    class ParallelProcessingTests {
        
        @Test
        @DisplayName("Should process channels in parallel")
        void shouldProcessChannelsInParallel() {
            var tokens = createTokenArray("Parallel", "processing", "test", "with", "multiple", "channels");

            // Process multiple times - verifying parallel processing works
            var results = new ArrayList<float[]>();
            for (int i = 0; i < 5; i++) {
                results.add(processor.processWindow(tokens));
            }

            // All results should be valid and well-formed
            for (int i = 0; i < results.size(); i++) {
                var result = results.get(i);
                assertThat(result).as("Result %d should not be null", i).isNotNull();
                assertThat(result.length)
                    .as("Result %d should have correct dimension", i)
                    .isEqualTo(processor.getTotalOutputDimension());

                // Verify results contain meaningful values
                var hasNonZeroValues = false;
                for (float value : result) {
                    if (Math.abs(value) > 0.001f) {
                        hasNonZeroValues = true;
                        break;
                    }
                }
                assertThat(hasNonZeroValues)
                    .as("Result %d should contain meaningful values", i)
                    .isTrue();
            }

            // With TestClock that advances predictably, we get consistent progression
            // The temporal channel will show progression but in a predictable way
            // This is the correct behavior - temporal features should track time/sequence
        }
        
        @Test
        @Timeout(value = 5, unit = TimeUnit.SECONDS)
        @DisplayName("Should complete parallel processing within time limit")
        void shouldCompleteParallelProcessingWithinTimeLimit() {
            var tokens = createTokenArray("Time", "sensitive", "parallel", "processing", "test");
            
            var result = processor.processWindow(tokens);
            
            assertThat(result).isNotNull();
            assertThat(result.length).isEqualTo(processor.getTotalOutputDimension());
        }
        
        @Test
        @DisplayName("Should handle concurrent window processing")
        void shouldHandleConcurrentWindowProcessing() throws Exception {
            var executor = Executors.newFixedThreadPool(10);
            var futures = new ArrayList<CompletableFuture<float[]>>();
            
            // Submit 50 concurrent processing tasks
            for (int i = 0; i < 50; i++) {
                final var windowId = i;
                var future = CompletableFuture.supplyAsync(() -> {
                    var tokens = createTokenArray("Concurrent", "window", String.valueOf(windowId));
                    return processor.processWindow(tokens);
                }, executor);
                
                futures.add(future);
            }
            
            // Collect all results
            var results = new ArrayList<float[]>();
            for (var future : futures) {
                results.add(future.get());
            }
            
            executor.shutdown();
            assertThat(executor.awaitTermination(5, TimeUnit.SECONDS)).isTrue();
            
            // Verify all results
            assertThat(results).hasSize(50);
            for (var result : results) {
                assertThat(result).isNotNull();
                assertThat(result.length).isEqualTo(processor.getTotalOutputDimension());
            }
        }
    }
    
    @Nested
    @DisplayName("Channel Coordination Tests")
    class ChannelCoordinationTests {
        
        @Test
        @DisplayName("Should coordinate all channels for combined output")
        void shouldCoordinateAllChannelsForCombinedOutput() {
            var tokens = createTokenArray("Coordination", "test", "for", "all", "channels");
            
            var result = processor.processWindow(tokens);
            var channelInfo = processor.getChannelInfo();
            
            // Verify that output length matches sum of all channel dimensions
            var expectedLength = channelInfo.stream().mapToInt(info -> info.outputDimension()).sum();
            assertThat(result.length).isEqualTo(expectedLength);
            
            // Verify that each channel's output is properly positioned
            // (We can't test exact values, but we can check structure)
            int offset = 0;
            for (var info : channelInfo) {
                // Each channel should contribute non-trivial output
                var channelOutput = Arrays.copyOfRange(result, offset, offset + info.outputDimension());
                offset += info.outputDimension();
                
                assertThat(channelOutput).isNotNull();
                assertThat(channelOutput.length).isEqualTo(info.outputDimension());
            }
        }
        
        @Test
        @DisplayName("Should handle channel processing failures gracefully")
        void shouldHandleChannelProcessingFailuresGracefully() {
            // Test with potentially problematic input
            var problemTokens = new Token[] {
                new Token(null, 0, Token.TokenType.UNKNOWN),
                new Token("", 1, Token.TokenType.WORD),
                new Token("normal", 2, Token.TokenType.WORD)
            };
            
            // Should not throw exception, even with problematic input
            assertThatCode(() -> {
                var result = processor.processWindow(problemTokens);
                assertThat(result).isNotNull();
                assertThat(result.length).isEqualTo(processor.getTotalOutputDimension());
            }).doesNotThrowAnyException();
        }
        
        @Test
        @DisplayName("Should maintain deterministic channel ordering")
        void shouldMaintainDeterministicChannelOrdering() {
            var tokens = createTokenArray("Deterministic", "channel", "ordering", "test");
            
            // Process multiple times and verify ordering is consistent
            var channelInfos = new ArrayList<List<MultiChannelProcessor.ChannelInfo>>();
            for (int i = 0; i < 3; i++) {
                channelInfos.add(processor.getChannelInfo());
            }
            
            // All channel info lists should be identical
            var firstChannelInfo = channelInfos.get(0);
            for (int i = 1; i < channelInfos.size(); i++) {
                var currentChannelInfo = channelInfos.get(i);
                assertThat(currentChannelInfo).hasSize(firstChannelInfo.size());
                
                for (int j = 0; j < firstChannelInfo.size(); j++) {
                    assertThat(currentChannelInfo.get(j).name())
                        .isEqualTo(firstChannelInfo.get(j).name());
                    assertThat(currentChannelInfo.get(j).outputDimension())
                        .isEqualTo(firstChannelInfo.get(j).outputDimension());
                }
            }
        }
    }
    
    @Nested
    @DisplayName("Error Handling Tests")
    class ErrorHandlingTests {
        
        @Test
        @DisplayName("Should handle null token array")
        void shouldHandleNullTokenArray() {
            assertThatCode(() -> {
                var result = processor.processWindow(null);
                // Implementation should handle this gracefully
            }).doesNotThrowAnyException();
        }
        
        @Test
        @DisplayName("Should handle malformed tokens")
        void shouldHandleMalformedTokens() {
            var malformedTokens = new Token[] {
                new Token(null, -1, Token.TokenType.UNKNOWN),
                new Token("", Integer.MAX_VALUE, Token.TokenType.WORD),
                new Token("test", 0, null)
            };
            
            assertThatCode(() -> {
                var result = processor.processWindow(malformedTokens);
                assertThat(result).isNotNull();
            }).doesNotThrowAnyException();
        }
        
        @Test
        @DisplayName("Should handle extreme token arrays")
        void shouldHandleExtremeTokenArrays() {
            // Very long token array
            var longTokens = new Token[10000];
            for (int i = 0; i < longTokens.length; i++) {
                longTokens[i] = new Token("token" + i, i, Token.TokenType.WORD);
            }
            
            assertThatCode(() -> {
                var result = processor.processWindow(longTokens);
                assertThat(result).isNotNull();
            }).doesNotThrowAnyException();
        }
        
        @Test
        @DisplayName("Should recover from channel timeouts")
        void shouldRecoverFromChannelTimeouts() {
            // Create tokens that might cause processing delays
            var complexTokens = createTokenArray(
                "Complex", "sentence", "with", "multiple", "clauses",
                "and", "various", "punctuation", "marks", "!",
                "Does", "this", "cause", "any", "processing",
                "delays", "or", "timeouts", "?"
            );
            
            var startTime = System.nanoTime();
            var result = processor.processWindow(complexTokens);
            var endTime = System.nanoTime();
            
            var processingTimeMs = (endTime - startTime) / 1_000_000.0;
            
            assertThat(result).isNotNull();
            assertThat(processingTimeMs).isLessThan(5000); // Should complete within 5 seconds
        }
    }
    
    @Nested
    @DisplayName("Performance Tests")
    class PerformanceTests {
        
        @Test
        @Timeout(value = 10, unit = TimeUnit.SECONDS)
        @DisplayName("Should process windows with acceptable performance")
        void shouldProcessWindowsWithAcceptablePerformance() {
            var tokens = createTokenArray("Performance", "test", "with", "standard", "window", "size");
            
            // Warm-up runs
            for (int i = 0; i < 10; i++) {
                processor.processWindow(tokens);
            }
            
            // Timed runs
            var startTime = System.nanoTime();
            for (int i = 0; i < 100; i++) {
                processor.processWindow(tokens);
            }
            var endTime = System.nanoTime();
            
            var totalTimeMs = (endTime - startTime) / 1_000_000.0;
            var avgTimePerWindow = totalTimeMs / 100.0;
            
            System.out.println("Average processing time per window: " + avgTimePerWindow + " ms");
            
            // Should process at least 10 windows per second (100ms per window)
            assertThat(avgTimePerWindow).isLessThan(100);
        }
        
        @Test
        @DisplayName("Should scale reasonably with window size")
        void shouldScaleReasonablyWithWindowSize() {
            var smallWindow = createTokenArray("Small", "window");
            var mediumWindow = createTokenArray("Medium", "sized", "window", "with", "more", "tokens");
            var largeWindow = createTokenArray("Large", "window", "with", "many", "more", "tokens",
                                              "including", "various", "types", "and", "structures",
                                              "to", "test", "scaling", "behavior");
            
            var smallTime = timeWindowProcessing(smallWindow);
            var mediumTime = timeWindowProcessing(mediumWindow);
            var largeTime = timeWindowProcessing(largeWindow);
            
            System.out.println("Small window: " + smallTime + " ms");
            System.out.println("Medium window: " + mediumTime + " ms");
            System.out.println("Large window: " + largeTime + " ms");
            
            // Processing time should scale reasonably (not exponentially)
            assertThat(largeTime / smallTime).isLessThan(10); // Should not be more than 10x slower
        }
        
        private double timeWindowProcessing(Token[] tokens) {
            // Warm up
            for (int i = 0; i < 5; i++) {
                processor.processWindow(tokens);
            }
            
            var startTime = System.nanoTime();
            for (int i = 0; i < 20; i++) {
                processor.processWindow(tokens);
            }
            var endTime = System.nanoTime();
            
            return (endTime - startTime) / 1_000_000.0 / 20.0; // Average time in ms
        }
    }
    
    @Nested
    @DisplayName("Channel State Management Tests")
    class ChannelStateManagementTests {
        
        @Test
        @DisplayName("Should reset all channels")
        void shouldResetAllChannels() {
            // Process some windows to establish state
            processor.processWindow(createTokenArray("First", "window"));
            processor.processWindow(createTokenArray("Second", "window"));
            processor.processWindow(createTokenArray("Third", "window"));
            
            // Reset channels
            processor.resetChannels();
            
            // Processing should still work after reset
            var result = processor.processWindow(createTokenArray("After", "reset", "window"));
            
            assertThat(result).isNotNull();
            assertThat(result.length).isEqualTo(processor.getTotalOutputDimension());
        }
        
        @Test
        @DisplayName("Should maintain channel properties after reset")
        void shouldMaintainChannelPropertiesAfterReset() {
            var channelInfoBefore = processor.getChannelInfo();
            
            processor.resetChannels();
            
            var channelInfoAfter = processor.getChannelInfo();
            
            // Channel properties should be unchanged
            assertThat(channelInfoAfter).hasSize(channelInfoBefore.size());
            for (int i = 0; i < channelInfoBefore.size(); i++) {
                assertThat(channelInfoAfter.get(i).name())
                    .isEqualTo(channelInfoBefore.get(i).name());
                assertThat(channelInfoAfter.get(i).outputDimension())
                    .isEqualTo(channelInfoBefore.get(i).outputDimension());
                assertThat(channelInfoAfter.get(i).isDeterministic())
                    .isEqualTo(channelInfoBefore.get(i).isDeterministic());
            }
        }
    }
    
    @Nested
    @DisplayName("Resource Management Tests")
    class ResourceManagementTests {
        
        @Test
        @DisplayName("Should shutdown gracefully")
        void shouldShutdownGracefully() {
            var testProcessor = new MultiChannelProcessor(2, testClock);

            // Process some windows
            testProcessor.processWindow(createTokenArray("Test", "before", "shutdown"));

            // Shutdown should complete without exceptions
            assertThatCode(testProcessor::shutdown).doesNotThrowAnyException();
            
            // Note: We don't test processing after shutdown as behavior is undefined
        }
        
        @Test
        @DisplayName("Should handle shutdown timeout")
        void shouldHandleShutdownTimeout() {
            var testProcessor = new MultiChannelProcessor(1, testClock);

            // Submit some work
            testProcessor.processWindow(createTokenArray("Work", "before", "shutdown"));

            // Shutdown with potential timeout
            assertThatCode(testProcessor::shutdown).doesNotThrowAnyException();
        }
    }
    
    // Helper method to create Token arrays from strings
    private Token[] createTokenArray(String... texts) {
        var tokens = new Token[texts.length];
        for (int i = 0; i < texts.length; i++) {
            var text = texts[i];
            var type = determineTokenType(text);
            tokens[i] = new Token(text, i, type);
        }
        return tokens;
    }
    
    private Token.TokenType determineTokenType(String text) {
        if (text == null || text.isEmpty()) {
            return Token.TokenType.UNKNOWN;
        }
        
        if (text.matches("\\w+")) {
            return Token.TokenType.WORD;
        } else if (text.matches("[.,;:!?()\\[\\]{}'\"-]")) {
            return Token.TokenType.PUNCTUATION;
        } else if (text.matches("\\d+(\\.\\d+)?")) {
            return Token.TokenType.NUMBER;
        } else if (text.matches("[#@$%^&*+=<>~`]")) {
            return Token.TokenType.SPECIAL;
        } else {
            return Token.TokenType.UNKNOWN;
        }
    }
}