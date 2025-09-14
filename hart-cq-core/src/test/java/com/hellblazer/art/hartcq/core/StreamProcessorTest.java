package com.hellblazer.art.hartcq.core;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;

class StreamProcessorTest {
    
    private StreamProcessor processor;
    
    @BeforeEach
    void setUp() {
        processor = new StreamProcessor();
    }
    
    @AfterEach
    void tearDown() {
        processor.close();
    }
    
    @Test
    void testBasicStreamProcessing() throws Exception {
        var text = "This is a test sentence with exactly twenty tokens to fill the sliding window completely for processing.";
        
        var future = processor.processStream(text);
        var result = future.get();
        
        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getTotalTokens()).isGreaterThan(0);
        assertThat(result.getWindowsProcessed()).isGreaterThan(0);
    }
    
    @Test
    void testSlidingWindowSize() {
        var tokenizer = new Tokenizer();
        var window = new SlidingWindow();
        
        var text = "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty";
        var tokens = tokenizer.tokenize(text);
        
        for (Token token : tokens) {
            window.addToken(token);
        }
        
        assertThat(window.size()).isEqualTo(SlidingWindow.WINDOW_SIZE);
    }
    
    @Test
    void testWindowProcessorCallback() throws Exception {
        var windowCount = new AtomicInteger(0);
        
        processor.registerWindowProcessor(tokens -> {
            windowCount.incrementAndGet();
            assertThat(tokens).hasSize(SlidingWindow.WINDOW_SIZE);
        });
        
        var text = "The quick brown fox jumps over the lazy dog. " +
                   "The quick brown fox jumps over the lazy dog. " +
                   "The quick brown fox jumps over the lazy dog.";
        
        var future = processor.processStream(text);
        var result = future.get();
        
        assertThat(result.isSuccessful()).isTrue();
        assertThat(windowCount.get()).isGreaterThan(0);
    }
    
    @Test
    void testTokenizer() {
        var tokenizer = new Tokenizer();
        
        var text = "Hello, world! This is test 123.";
        var tokens = tokenizer.tokenize(text);
        
        assertThat(tokens).isNotEmpty();
        
        // Check for different token types
        var hasWord = tokens.stream().anyMatch(t -> t.getType() == Token.TokenType.WORD);
        var hasPunctuation = tokens.stream().anyMatch(t -> t.getType() == Token.TokenType.PUNCTUATION);
        var hasNumber = tokens.stream().anyMatch(t -> t.getType() == Token.TokenType.NUMBER);
        
        assertThat(hasWord).isTrue();
        assertThat(hasPunctuation).isTrue();
        assertThat(hasNumber).isTrue();
    }
    
    @Test
    void testBatchProcessing() {
        var text = "This is a longer text that will be processed in batches. " +
                   "Each batch will contain a specific number of tokens. " +
                   "This allows for more efficient processing of large texts.";
        
        var results = processor.processBatches(text, 10);
        
        assertThat(results).isNotEmpty();
        for (var result : results) {
            assertThat(result.isSuccessful()).isTrue();
        }
    }
}