package com.hellblazer.art.hartcq;

import com.hellblazer.art.hartcq.core.StreamProcessor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Basic tests for HART-CQ stream processing functionality.
 * Tests the integration of various components including tokenization,
 * windowing, performance monitoring, and competitive queuing.
 */
class StreamProcessorTest {
    
    private StreamProcessor streamProcessor;
    private Tokenizer tokenizer;
    private PerformanceMonitor performanceMonitor;
    private CompetitiveQueue<String> competitiveQueue;
    private HARTCQConfig config;
    
    @BeforeEach
    void setUp() {
        streamProcessor = new StreamProcessor();
        tokenizer = new Tokenizer();
        performanceMonitor = new PerformanceMonitor();
        competitiveQueue = new CompetitiveQueue<>(2);
        config = new HARTCQConfig();
    }
    
    @AfterEach
    void tearDown() throws Exception {
        if (streamProcessor != null) {
            streamProcessor.close();
        }
        if (performanceMonitor != null) {
            performanceMonitor.close();
        }
        if (competitiveQueue != null) {
            competitiveQueue.close();
        }
    }
    
    @Test
    void testBasicTokenization() {
        var text = "Hello world! This is a test.";
        var tokens = tokenizer.tokenize(text);
        
        assertNotNull(tokens);
        assertTrue(tokens.size() > 0);
        
        // Check that we have word tokens
        var wordTokens = tokens.stream()
            .filter(token -> token.getType() == Token.TokenType.WORD)
            .toList();
        
        assertTrue(wordTokens.size() >= 5); // At least "Hello", "world", "This", "is", "a", "test"
    }
    
    @Test
    void testTokenizeWords() {
        var text = "The quick brown fox jumps over the lazy dog.";
        var wordTokens = tokenizer.tokenizeWords(text);
        
        assertNotNull(wordTokens);
        assertEquals(9, wordTokens.size()); // All words except punctuation
        
        // Verify all are word tokens
        for (Token token : wordTokens) {
            assertEquals(Token.TokenType.WORD, token.getType());
        }
    }
    
    @Test
    void testTokenizeWithCategories() {
        var text = "I have 5 cats and 3 dogs! Amazing.";
        var result = tokenizer.tokenizeWithCategories(text);
        
        assertNotNull(result);
        assertTrue(result.getWordCount() > 0);
        assertTrue(result.getNumberCount() > 0);
        assertTrue(result.getPunctuationCount() > 0);
        
        assertEquals(2, result.getNumberCount()); // "5" and "3"
        assertTrue(result.getWordCount() >= 6); // "I", "have", "cats", "and", "dogs", "Amazing"
    }
    
    @Test
    void testWindowFeatures() {
        var text = "This is a simple test sentence with some words.";
        var features = WindowFeatures.fromText(text);
        
        assertNotNull(features);
        assertTrue(features.getWordCount() > 0);
        assertTrue(features.getAvgWordLength() > 0);
        assertTrue(features.getUniqueWordCount() > 0);
        assertTrue(features.getQualityScore() >= 0.0 && features.getQualityScore() <= 1.0);
        assertTrue(features.isProcessable());
    }
    
    @Test
    void testWindowFeaturesEmpty() {
        var features = WindowFeatures.fromText("");
        
        assertNotNull(features);
        assertEquals(0, features.getWordCount());
        assertEquals(0.0, features.getAvgWordLength());
        assertEquals(0, features.getUniqueWordCount());
        assertEquals(0.0, features.getQualityScore());
        assertFalse(features.isProcessable());
    }
    
    @Test
    void testWindowResult() {
        var features = WindowFeatures.fromText("Test sentence for window processing.");
        var patterns = List.of(
            new WindowResult.Pattern("WORD", "test", 0.8, 0),
            new WindowResult.Pattern("SYNTAX", "noun-phrase", 0.6, 1)
        );
        
        var result = new WindowResult(1L, features, patterns, 0.75, 1_000_000L);
        
        assertEquals(1L, result.getWindowId());
        assertEquals(features, result.getFeatures());
        assertEquals(patterns, result.getPatterns());
        assertEquals(0.75, result.getConfidence(), 0.001);
        assertEquals(1_000_000L, result.getProcessingTimeNanos());
        assertEquals(1.0, result.getProcessingTimeMillis(), 0.001);
        assertTrue(result.isSuccessful());
        assertEquals(2, result.getPatternCount());
    }
    
    @Test
    @Timeout(value = 5, unit = TimeUnit.SECONDS)
    void testStreamProcessorBasic() throws Exception {
        var text = "The quick brown fox jumps over the lazy dog. This is another sentence.";
        var future = streamProcessor.processStream(text);
        
        var result = future.get(3, TimeUnit.SECONDS);
        
        assertNotNull(result);
        assertTrue(result.isSuccessful());
        assertTrue(result.getTotalTokens() > 0);
        assertTrue(result.getWindowsProcessed() > 0);
    }
    
    @Test
    void testPerformanceMonitorBasic() {
        performanceMonitor.recordSentencesProcessed(10);
        performanceMonitor.recordTokensProcessed(100);
        performanceMonitor.recordWindowsProcessed(5);
        performanceMonitor.recordProcessingTime(1_000_000L); // 1ms
        
        var report = performanceMonitor.generatePerformanceReport();
        
        assertNotNull(report);
        assertEquals(10, report.getTotalSentencesProcessed());
        assertEquals(100, report.getTotalTokensProcessed());
        assertEquals(5, report.getTotalWindowsProcessed());
        assertTrue(report.getAverageProcessingTimeMs() > 0);
    }
    
    @Test
    void testPerformanceMonitorThroughputTarget() {
        // Test with a low target to ensure it's met quickly
        var monitor = new PerformanceMonitor(1, 1, null);
        
        try {
            monitor.recordSentencesProcessed(10);
            // Give it a small delay to calculate throughput
            Thread.sleep(10);
            
            assertTrue(monitor.getCurrentThroughputSentencesPerSecond() > 0);
            assertTrue(monitor.isMeetingThroughputTarget());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            fail("Test interrupted");
        } finally {
            monitor.close();
        }
    }
    
    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void testCompetitiveQueue() throws Exception {
        var futures = new ArrayList<CompletableFuture<CompetitiveQueue.ProcessingResult<String>>>();
        
        // Enqueue several items
        futures.add(competitiveQueue.enqueue("High priority item", 0.9));
        futures.add(competitiveQueue.enqueue("Medium priority item", 0.5));
        futures.add(competitiveQueue.enqueue("Low priority item", 0.1));
        
        // Wait for all to complete
        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
            .get(5, TimeUnit.SECONDS);
        
        // Verify all completed successfully
        for (var future : futures) {
            var result = future.get();
            assertNotNull(result);
            assertTrue(result.isSuccessful());
            assertTrue(result.getConfidence() > 0);
        }
        
        assertEquals(3, competitiveQueue.getProcessedItemCount());
    }
    
    @Test
    void testHARTCQConfigDefaults() {
        assertEquals(20, config.getWindowSize());
        assertTrue(config.isEnableSlidingWindow());
        assertTrue(config.getChannelConfig().isEnablePositionalChannel());
        assertTrue(config.getChannelConfig().isEnableSyntaxChannel());
        assertTrue(config.getChannelConfig().isEnableSemanticChannel());
        assertEquals(100, config.getPerformanceConfig().getTargetThroughputSentencesPerSecond());
        assertTrue(config.getPerformanceConfig().isEnablePerformanceMonitoring());
        assertEquals(0.7, config.getTemplateConfig().getVigilanceParameter(), 0.001);
    }
    
    @Test
    void testHARTCQConfigBuilder() {
        var customConfig = new HARTCQConfig.Builder()
            .windowSize(15)
            .enableSlidingWindow(false)
            .targetThroughput(200)
            .vigilanceParameter(0.8)
            .maxTemplates(50)
            .build();
        
        assertEquals(15, customConfig.getWindowSize());
        assertFalse(customConfig.isEnableSlidingWindow());
        assertEquals(200, customConfig.getPerformanceConfig().getTargetThroughputSentencesPerSecond());
        assertEquals(0.8, customConfig.getTemplateConfig().getVigilanceParameter(), 0.001);
        assertEquals(50, customConfig.getTemplateConfig().getMaxTemplates());
    }
    
    @Test
    void testHARTCQConfigPresets() {
        var highThroughputConfig = HARTCQConfig.forHighThroughput();
        assertEquals(200, highThroughputConfig.getPerformanceConfig().getTargetThroughputSentencesPerSecond());
        assertEquals(5000, highThroughputConfig.getPerformanceConfig().getQueueCapacity());
        
        var lowLatencyConfig = HARTCQConfig.forLowLatency();
        assertEquals(50, lowLatencyConfig.getPerformanceConfig().getTargetThroughputSentencesPerSecond());
        assertEquals(10, lowLatencyConfig.getWindowSize());
        
        var memoryConstrainedConfig = HARTCQConfig.forMemoryConstrained();
        assertEquals(2, memoryConstrainedConfig.getPerformanceConfig().getMaxConcurrentProcessors());
        assertEquals(25, memoryConstrainedConfig.getTemplateConfig().getMaxTemplates());
    }
    
    @Test
    void testTokenStats() {
        var text = "I have 5 cats, 3 dogs, and 1 bird! They are amazing.";
        var tokens = tokenizer.tokenize(text);
        var stats = tokenizer.getTokenStats(tokens);
        
        assertNotNull(stats);
        assertTrue(stats.getWordCount() > 0);
        assertTrue(stats.getNumberCount() > 0);
        assertTrue(stats.getPunctuationCount() > 0);
        assertTrue(stats.getTotalCount() > 0);
        
        assertEquals(3, stats.getNumberCount()); // "5", "3", "1"
        assertTrue(stats.getWordCount() >= 8); // At least 8 words
    }
    
    @Test
    void testTokenizeForWindows() {
        var text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10 Word11 Word12";
        var windows = tokenizer.tokenizeForWindows(text, 5);
        
        assertNotNull(windows);
        assertTrue(windows.size() > 0);
        
        // Each window should have 5 tokens (except possibly the last)
        for (int i = 0; i < windows.size() - 1; i++) {
            assertEquals(5, windows.get(i).size());
        }
    }
    
    @Test
    void testWindowFeaturesCombination() {
        var features1 = WindowFeatures.fromText("Hello world");
        var features2 = WindowFeatures.fromText("Goodbye universe");
        
        var combined = features1.combineWith(features2);
        
        assertNotNull(combined);
        assertEquals(features1.getWordCount() + features2.getWordCount(), combined.getWordCount());
        assertEquals(features1.getPunctuationCount() + features2.getPunctuationCount(), combined.getPunctuationCount());
        assertTrue(combined.getQualityScore() > 0);
    }
    
    @Test
    void testIntegrationScenario() throws Exception {
        // Test a complete processing scenario
        performanceMonitor.startMonitoring();
        
        try {
            var text = "This is a comprehensive test of the HART-CQ system. " +
                      "It processes text through multiple stages including tokenization, " +
                      "windowing, feature extraction, and performance monitoring.";
            
            // Tokenize the text
            var tokens = tokenizer.tokenize(text);
            assertTrue(tokens.size() > 20);
            
            // Extract features
            var features = WindowFeatures.fromText(text);
            assertTrue(features.isProcessable());
            
            // Process through stream processor
            var streamResult = streamProcessor.processStream(text).get(3, TimeUnit.SECONDS);
            assertTrue(streamResult.isSuccessful());
            
            // Record performance metrics
            performanceMonitor.recordWindowProcessing(
                3, // sentence count
                tokens.size(),
                streamResult.getWindowsProcessed() * 1_000_000L // simulate 1ms per window
            );
            
            // Generate performance report
            var report = performanceMonitor.generatePerformanceReport();
            assertEquals(3, report.getTotalSentencesProcessed());
            assertEquals(tokens.size(), report.getTotalTokensProcessed());
            
        } finally {
            performanceMonitor.stopMonitoring();
        }
    }
    
    @Test
    void testErrorHandling() {
        // Test null input handling
        var emptyTokens = tokenizer.tokenize(null);
        assertNotNull(emptyTokens);
        assertTrue(emptyTokens.isEmpty());
        
        var emptyTokens2 = tokenizer.tokenize("");
        assertNotNull(emptyTokens2);
        assertTrue(emptyTokens2.isEmpty());
        
        // Test features with null input
        var emptyFeatures = WindowFeatures.fromText(null);
        assertNotNull(emptyFeatures);
        assertEquals(0, emptyFeatures.getWordCount());
        assertFalse(emptyFeatures.isProcessable());
    }
}