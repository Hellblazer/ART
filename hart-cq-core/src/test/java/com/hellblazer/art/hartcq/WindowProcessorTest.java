package com.hellblazer.art.hartcq;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Timeout;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutionException;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.Mockito.*;

/**
 * Comprehensive unit tests for the WindowProcessor class.
 * Tests window processing functionality including feature extraction,
 * pattern matching, confidence calculation, and thread safety.
 */
@DisplayName("WindowProcessor Tests")
class WindowProcessorTest {
    
    private WindowProcessor processor;
    
    @Mock
    private ProcessingWindow mockWindow;
    
    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        processor = new WindowProcessor();
    }
    
    @Nested
    @DisplayName("Basic Window Processing Tests")
    class BasicWindowProcessingTests {
        
        @Test
        @DisplayName("Should process simple window successfully")
        void shouldProcessSimpleWindowSuccessfully() {
            // Create a real window with tokens
            var tokens = createTokenList("Hello", "world", "how", "are", "you", "?");
            var window = createProcessingWindow(1L, tokens);
            
            var result = processor.process(window);
            
            assertThat(result).isNotNull();
            assertThat(result.getWindowId()).isEqualTo(1L);
            assertThat(result.getFeatures()).isNotNull();
            assertThat(result.getPatterns()).isNotNull();
            assertThat(result.getConfidence()).isBetween(0.0, 1.0);
            assertThat(result.getProcessingTimeNanos()).isPositive();
        }
        
        @Test
        @DisplayName("Should extract meaningful features from window")
        void shouldExtractMeaningfulFeaturesFromWindow() {
            var tokens = createTokenList("The", "quick", "brown", "fox", "jumps", ".");
            var window = createProcessingWindow(2L, tokens);
            
            var result = processor.process(window);
            
            var features = result.getFeatures();
            assertThat(features.getWordCount()).isEqualTo(5); // Excluding punctuation
            assertThat(features.getPunctuationCount()).isEqualTo(1);
            assertThat(features.getAvgWordLength()).isPositive();
            assertThat(features.getUniqueWordCount()).isEqualTo(5);
            assertThat(features.getQualityScore()).isBetween(0.0, 1.0);
        }
        
        @Test
        @DisplayName("Should calculate confidence based on features and patterns")
        void shouldCalculateConfidenceBasedOnFeaturesAndPatterns() {
            // Create a high-quality window with clear patterns
            var tokens = createTokenList("What", "is", "your", "name", "?");
            var window = createProcessingWindow(3L, tokens);
            
            var result = processor.process(window);
            
            // Should have reasonable confidence for a question pattern
            assertThat(result.getConfidence()).isGreaterThan(0.3);
            assertThat(result.getPatterns()).hasSizeGreaterThan(0);
            
            // Check that question pattern was detected
            var hasQuestionPattern = result.getPatterns().stream()
                .anyMatch(p -> p.getType().equalsIgnoreCase("INTERROGATIVE") || 
                              p.getType().contains("question"));
            assertThat(hasQuestionPattern).isTrue();
        }
        
        @Test
        @DisplayName("Should detect different pattern types")
        void shouldDetectDifferentPatternTypes() {
            // Test question pattern
            var questionTokens = createTokenList("How", "are", "you", "?");
            var questionWindow = createProcessingWindow(4L, questionTokens);
            var questionResult = processor.process(questionWindow);
            
            // Test statement pattern
            var statementTokens = createTokenList("This", "is", "a", "test", ".");
            var statementWindow = createProcessingWindow(5L, statementTokens);
            var statementResult = processor.process(statementWindow);
            
            // Test exclamation pattern
            var exclamationTokens = createTokenList("Great", "job", "!");
            var exclamationWindow = createProcessingWindow(6L, exclamationTokens);
            var exclamationResult = processor.process(exclamationWindow);
            
            // Each should detect appropriate patterns
            assertThat(questionResult.getPatterns()).isNotEmpty();
            assertThat(statementResult.getPatterns()).isNotEmpty();
            assertThat(exclamationResult.getPatterns()).isNotEmpty();
            
            // Patterns should be different types
            var questionType = questionResult.getPatterns().get(0).getType();
            var statementType = statementResult.getPatterns().get(0).getType();
            var exclamationType = exclamationResult.getPatterns().get(0).getType();
            
            assertThat(questionType).isNotEqualTo(statementType);
            assertThat(statementType).isNotEqualTo(exclamationType);
        }
    }
    
    @Nested
    @DisplayName("Feature Extraction Tests")
    class FeatureExtractionTests {
        
        @Test
        @DisplayName("Should handle empty window gracefully")
        void shouldHandleEmptyWindowGracefully() {
            var emptyTokens = new ArrayList<Token>();
            var window = createProcessingWindow(7L, emptyTokens);
            
            var result = processor.process(window);
            
            assertThat(result).isNotNull();
            assertThat(result.getFeatures().getWordCount()).isZero();
            assertThat(result.getFeatures().getQualityScore()).isLessThanOrEqualTo(0.1);
            assertThat(result.getConfidence()).isLessThan(0.5);
        }
        
        @Test
        @DisplayName("Should calculate word statistics correctly")
        void shouldCalculateWordStatisticsCorrectly() {
            var tokens = createTokenList("The", "very", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", ".");
            var window = createProcessingWindow(8L, tokens);
            
            var result = processor.process(window);
            var features = result.getFeatures();
            
            assertThat(features.getWordCount()).isEqualTo(10); // Excluding punctuation
            assertThat(features.getUniqueWordCount()).isEqualTo(9); // "the" appears twice
            assertThat(features.getLexicalDiversity()).isEqualTo(9.0 / 10.0);
            assertThat(features.getPunctuationDensity()).isEqualTo(1.0 / 10.0);
        }
        
        @Test
        @DisplayName("Should handle punctuation-heavy text")
        void shouldHandlePunctuationHeavyText() {
            var tokens = createTokenList("Hello", "!", "How", "are", "you", "?", "I", "'", "m", "fine", ".");
            var window = createProcessingWindow(9L, tokens);
            
            var result = processor.process(window);
            var features = result.getFeatures();
            
            assertThat(features.getPunctuationCount()).isGreaterThan(3);
            assertThat(features.getPunctuationDensity()).isPositive();
            assertThat(features.getWordCount()).isGreaterThan(0);
        }
        
        @Test
        @DisplayName("Should calculate average word length accurately")
        void shouldCalculateAverageWordLengthAccurately() {
            // Known word lengths: "cat"=3, "dog"=3, "elephant"=8
            var tokens = createTokenList("cat", "dog", "elephant");
            var window = createProcessingWindow(10L, tokens);
            
            var result = processor.process(window);
            var features = result.getFeatures();
            
            var expectedAvgLength = (3.0 + 3.0 + 8.0) / 3.0;
            assertThat(features.getAvgWordLength()).isEqualTo(expectedAvgLength, within(0.01));
        }
    }
    
    @Nested
    @DisplayName("Pattern Matching Tests")
    class PatternMatchingTests {
        
        @Test
        @DisplayName("Should match interrogative patterns correctly")
        void shouldMatchInterrogativePatternsCorrectly() {
            var patterns = List.of(
                createTokenList("What", "is", "this", "?"),
                createTokenList("How", "do", "you", "do", "?"),
                createTokenList("Where", "are", "you", "going", "?"),
                createTokenList("Why", "did", "you", "leave", "?")
            );
            
            for (int i = 0; i < patterns.size(); i++) {
                var window = createProcessingWindow(11L + i, patterns.get(i));
                var result = processor.process(window);
                
                var hasInterrogativePattern = result.getPatterns().stream()
                    .anyMatch(p -> p.getType().contains("INTERROGATIVE") || 
                                  p.getValue().contains("question"));
                
                assertThat(hasInterrogativePattern)
                    .as("Should detect interrogative pattern for: %s", 
                        patterns.get(i).stream().map(Token::getText).toList())
                    .isTrue();
            }
        }
        
        @Test
        @DisplayName("Should match declarative patterns correctly")
        void shouldMatchDeclarativePatternsCorrectly() {
            var patterns = List.of(
                createTokenList("This", "is", "a", "statement", "."),
                createTokenList("The", "sky", "is", "blue", "."),
                createTokenList("I", "like", "programming", ".")
            );
            
            for (int i = 0; i < patterns.size(); i++) {
                var window = createProcessingWindow(15L + i, patterns.get(i));
                var result = processor.process(window);
                
                var hasDeclarativePattern = result.getPatterns().stream()
                    .anyMatch(p -> p.getType().contains("DECLARATIVE") || 
                                  p.getValue().contains("statement"));
                
                assertThat(hasDeclarativePattern)
                    .as("Should detect declarative pattern for: %s", 
                        patterns.get(i).stream().map(Token::getText).toList())
                    .isTrue();
            }
        }
        
        @Test
        @DisplayName("Should match exclamatory patterns correctly")
        void shouldMatchExclamatoryPatternsCorrectly() {
            var patterns = List.of(
                createTokenList("Great", "job", "!"),
                createTokenList("Amazing", "!"),
                createTokenList("Oh", "no", "!")
            );
            
            for (int i = 0; i < patterns.size(); i++) {
                var window = createProcessingWindow(18L + i, patterns.get(i));
                var result = processor.process(window);
                
                var hasExclamatoryPattern = result.getPatterns().stream()
                    .anyMatch(p -> p.getType().contains("EXCLAMATORY") || 
                                  p.getValue().contains("exclamation"));
                
                assertThat(hasExclamatoryPattern)
                    .as("Should detect exclamatory pattern for: %s", 
                        patterns.get(i).stream().map(Token::getText).toList())
                    .isTrue();
            }
        }
        
        @Test
        @DisplayName("Should match imperative patterns correctly")
        void shouldMatchImperativePatternsCorrectly() {
            var patterns = List.of(
                createTokenList("Please", "help", "me", "."),
                createTokenList("Could", "you", "assist", "?")
            );
            
            for (int i = 0; i < patterns.size(); i++) {
                var window = createProcessingWindow(21L + i, patterns.get(i));
                var result = processor.process(window);
                
                var hasImperativePattern = result.getPatterns().stream()
                    .anyMatch(p -> p.getType().contains("IMPERATIVE") || 
                                  p.getValue().contains("command"));
                
                assertThat(hasImperativePattern)
                    .as("Should detect imperative pattern for: %s", 
                        patterns.get(i).stream().map(Token::getText).toList())
                    .isTrue();
            }
        }
        
        @Test
        @DisplayName("Should handle ambiguous patterns gracefully")
        void shouldHandleAmbiguousPatternsGracefully() {
            // Text that could match multiple patterns
            var ambiguousTokens = createTokenList("Well", "hello", "there", "!");
            var window = createProcessingWindow(23L, ambiguousTokens);
            
            var result = processor.process(window);
            
            // Should still produce some patterns
            assertThat(result.getPatterns()).isNotEmpty();
            assertThat(result.getConfidence()).isPositive();
            
            // All patterns should have valid confidence scores
            for (var pattern : result.getPatterns()) {
                assertThat(pattern.getStrength()).isBetween(0.0, 1.0);
            }
        }
    }
    
    @Nested
    @DisplayName("Confidence Calculation Tests")
    class ConfidenceCalculationTests {
        
        @Test
        @DisplayName("Should calculate higher confidence for high-quality features")
        void shouldCalculateHigherConfidenceForHighQualityFeatures() {
            // High-quality window: good word diversity, reasonable length, clear pattern
            var highQualityTokens = createTokenList("The", "brilliant", "scientist", "discovered", 
                                                   "amazing", "results", "yesterday", ".");
            var highQualityWindow = createProcessingWindow(24L, highQualityTokens);
            
            // Low-quality window: repetitive, short
            var lowQualityTokens = createTokenList("test", "test", "test");
            var lowQualityWindow = createProcessingWindow(25L, lowQualityTokens);
            
            var highQualityResult = processor.process(highQualityWindow);
            var lowQualityResult = processor.process(lowQualityWindow);
            
            assertThat(highQualityResult.getConfidence())
                .isGreaterThan(lowQualityResult.getConfidence());
        }
        
        @Test
        @DisplayName("Should factor in pattern strength for confidence")
        void shouldFactorInPatternStrengthForConfidence() {
            // Strong pattern: clear question with question mark
            var strongPatternTokens = createTokenList("What", "time", "is", "it", "?");
            var strongPatternWindow = createProcessingWindow(26L, strongPatternTokens);
            
            // Weak pattern: ambiguous structure
            var weakPatternTokens = createTokenList("maybe", "perhaps", "sometimes");
            var weakPatternWindow = createProcessingWindow(27L, weakPatternTokens);
            
            var strongResult = processor.process(strongPatternWindow);
            var weakResult = processor.process(weakPatternWindow);
            
            assertThat(strongResult.getConfidence())
                .isGreaterThan(weakResult.getConfidence());
        }
        
        @Test
        @DisplayName("Confidence should be bounded between 0 and 1")
        void confidenceShouldBeBoundedBetweenZeroAndOne() {
            // Test various window types
            var testCases = List.of(
                createTokenList(""), // empty
                createTokenList("a"), // single word
                createTokenList("Hello", "world", "!"), // simple
                createTokenList("The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", ".") // complex
            );
            
            for (int i = 0; i < testCases.size(); i++) {
                var window = createProcessingWindow(28L + i, testCases.get(i));
                var result = processor.process(window);
                
                assertThat(result.getConfidence())
                    .as("Confidence should be between 0 and 1 for case %d", i)
                    .isBetween(0.0, 1.0);
            }
        }
    }
    
    @Nested
    @DisplayName("Performance Tests")
    class PerformanceTests {
        
        @Test
        @Timeout(value = 5, unit = TimeUnit.SECONDS)
        @DisplayName("Should process windows within reasonable time")
        void shouldProcessWindowsWithinReasonableTime() {
            var tokens = createTokenList("The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", ".");
            var window = createProcessingWindow(32L, tokens);
            
            var startTime = System.nanoTime();
            var result = processor.process(window);
            var endTime = System.nanoTime();
            
            var processingTimeMs = (endTime - startTime) / 1_000_000.0;
            
            assertThat(result).isNotNull();
            assertThat(processingTimeMs).isLessThan(100); // Should process in under 100ms
            
            // Processing time in result should be approximately accurate
            var recordedTimeMs = result.getProcessingTimeNanos() / 1_000_000.0;
            assertThat(recordedTimeMs).isLessThan(processingTimeMs + 10); // Allow for some overhead
        }
        
        @Test
        @Timeout(value = 10, unit = TimeUnit.SECONDS)
        @DisplayName("Should handle batch processing efficiently")
        void shouldHandleBatchProcessingEfficiently() {
            var windows = new ArrayList<ProcessingWindow>();
            
            // Create 100 different windows
            for (int i = 0; i < 100; i++) {
                var tokens = createTokenList("Window", "number", String.valueOf(i), "test", ".");
                windows.add(createProcessingWindow(33L + i, tokens));
            }
            
            var startTime = System.nanoTime();
            var results = new ArrayList<WindowResult>();
            
            for (var window : windows) {
                results.add(processor.process(window));
            }
            
            var endTime = System.nanoTime();
            var totalTimeMs = (endTime - startTime) / 1_000_000.0;
            var avgTimePerWindow = totalTimeMs / windows.size();
            
            assertThat(results).hasSize(100);
            assertThat(avgTimePerWindow).isLessThan(10); // Less than 10ms per window on average
            
            System.out.println("Processed " + windows.size() + " windows in " + totalTimeMs + 
                             " ms (avg: " + avgTimePerWindow + " ms/window)");
        }
    }
    
    @Nested
    @DisplayName("Thread Safety Tests")
    class ThreadSafetyTests {
        
        @Test
        @Timeout(value = 15, unit = TimeUnit.SECONDS)
        @DisplayName("WindowProcessor should be thread-safe")
        void windowProcessorShouldBeThreadSafe() throws InterruptedException, ExecutionException {
            var executor = Executors.newFixedThreadPool(10);
            var results = new java.util.concurrent.ConcurrentHashMap<String, WindowResult>();
            var exceptions = new java.util.concurrent.ConcurrentLinkedQueue<Exception>();
            
            // Submit 200 processing tasks across multiple threads
            var futures = new ArrayList<CompletableFuture<Void>>();
            
            for (int i = 0; i < 200; i++) {
                final var windowId = (long) i;
                var future = CompletableFuture.runAsync(() -> {
                    try {
                        var tokens = createTokenList("Thread", "test", String.valueOf(windowId), "processing", ".");
                        var window = createProcessingWindow(windowId, tokens);
                        
                        var result = processor.process(window);
                        results.put("window_" + windowId, result);
                        
                    } catch (Exception e) {
                        exceptions.add(e);
                    }
                }, executor);
                
                futures.add(future);
            }
            
            // Wait for all tasks to complete
            CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).get();
            
            executor.shutdown();
            assertThat(executor.awaitTermination(5, TimeUnit.SECONDS)).isTrue();
            
            // Verify results
            assertThat(exceptions).isEmpty();
            assertThat(results).hasSize(200);
            
            // All results should be valid
            for (var result : results.values()) {
                assertThat(result).isNotNull();
                assertThat(result.getConfidence()).isBetween(0.0, 1.0);
                assertThat(result.getFeatures()).isNotNull();
                assertThat(result.getPatterns()).isNotNull();
            }
        }
    }
    
    @Nested
    @DisplayName("Edge Cases and Error Handling")
    class EdgeCasesAndErrorHandlingTests {
        
        @Test
        @DisplayName("Should handle malformed tokens gracefully")
        void shouldHandleMalformedTokensGracefully() {
            // Create tokens with unusual properties
            var tokens = new ArrayList<Token>();
            tokens.add(new Token("", 0, Token.TokenType.UNKNOWN));
            tokens.add(new Token(null, 1, Token.TokenType.WORD)); // This might cause issues
            tokens.add(new Token("normal", 2, Token.TokenType.WORD));
            
            var window = createProcessingWindow(999L, tokens);
            
            // Should not throw exception
            assertThatCode(() -> {
                var result = processor.process(window);
                assertThat(result).isNotNull();
            }).doesNotThrowAnyException();
        }
        
        @Test
        @DisplayName("Should handle very large windows")
        void shouldHandleVeryLargeWindows() {
            var tokens = new ArrayList<Token>();
            
            // Create a very large window (1000 tokens)
            for (int i = 0; i < 1000; i++) {
                tokens.add(new Token("word" + i, i, Token.TokenType.WORD));
                if (i % 10 == 9) {
                    tokens.add(new Token(".", i, Token.TokenType.PUNCTUATION));
                }
            }
            
            var window = createProcessingWindow(1000L, tokens);
            
            var result = processor.process(window);
            
            assertThat(result).isNotNull();
            assertThat(result.getFeatures().getWordCount()).isGreaterThan(900);
            assertThat(result.getProcessingTimeNanos()).isPositive();
        }
        
        @Test
        @DisplayName("Should handle windows with only punctuation")
        void shouldHandleWindowsWithOnlyPunctuation() {
            var tokens = createTokenList(".", ",", ";", ":", "!", "?");
            var window = createProcessingWindow(1001L, tokens);
            
            var result = processor.process(window);
            
            assertThat(result).isNotNull();
            assertThat(result.getFeatures().getWordCount()).isZero();
            assertThat(result.getFeatures().getPunctuationCount()).isEqualTo(6);
            assertThat(result.getConfidence()).isLessThan(0.5); // Low confidence for no words
        }
    }
    
    // Helper methods
    private List<Token> createTokenList(String... texts) {
        var tokens = new ArrayList<Token>();
        for (int i = 0; i < texts.length; i++) {
            var text = texts[i];
            var type = determineTokenType(text);
            tokens.add(new Token(text, i, type));
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
    
    private ProcessingWindow createProcessingWindow(long windowId, List<Token> tokens) {
        var window = mock(ProcessingWindow.class);
        when(window.getWindowId()).thenReturn(windowId);
        when(window.getTokens()).thenReturn(tokens);
        when(window.getCreationTime()).thenReturn(System.nanoTime() - 1000L); // Created 1000ns ago
        
        // Create text from tokens for pattern matching
        var text = tokens.stream()
            .map(Token::getText)
            .reduce("", (a, b) -> a.isEmpty() ? b : a + " " + b);
        when(window.getText()).thenReturn(text);
        
        return window;
    }
}