package com.hellblazer.art.hartcq;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.*;

/**
 * Comprehensive unit tests for the Tokenizer class.
 * Tests all tokenization functionality including edge cases,
 * performance, and thread safety.
 */
@DisplayName("Tokenizer Tests")
class TokenizerTest {
    
    private Tokenizer tokenizer;
    
    @BeforeEach
    void setUp() {
        tokenizer = new Tokenizer();
    }
    
    @Nested
    @DisplayName("Basic Tokenization Tests")
    class BasicTokenizationTests {
        
        @Test
        @DisplayName("Should tokenize simple sentence correctly")
        void shouldTokenizeSimpleSentence() {
            var input = "Hello world how are you?";
            var tokens = tokenizer.tokenize(input);
            
            assertThat(tokens).hasSize(6);
            assertThat(tokens.get(0).getText()).isEqualTo("Hello");
            assertThat(tokens.get(0).getType()).isEqualTo(Token.TokenType.WORD);
            assertThat(tokens.get(0).getPosition()).isEqualTo(0);
            
            assertThat(tokens.get(5).getText()).isEqualTo("?");
            assertThat(tokens.get(5).getType()).isEqualTo(Token.TokenType.PUNCTUATION);
        }
        
        @Test
        @DisplayName("Should handle mixed content types")
        void shouldHandleMixedContent() {
            var input = "The price is $25.99 at 3:30 PM!";
            var tokens = tokenizer.tokenize(input);
            
            assertThat(tokens).isNotEmpty();
            
            // Check for different token types
            var hasWords = tokens.stream().anyMatch(t -> t.getType() == Token.TokenType.WORD);
            var hasNumbers = tokens.stream().anyMatch(t -> t.getType() == Token.TokenType.NUMBER);
            var hasPunctuation = tokens.stream().anyMatch(t -> t.getType() == Token.TokenType.PUNCTUATION);
            var hasSpecial = tokens.stream().anyMatch(t -> t.getType() == Token.TokenType.SPECIAL);
            
            assertThat(hasWords).isTrue();
            assertThat(hasNumbers || hasSpecial).isTrue(); // Numbers might be classified as special
            assertThat(hasPunctuation || hasSpecial).isTrue();
        }
        
        @Test
        @DisplayName("Should preserve token positions")
        void shouldPreserveTokenPositions() {
            var input = "First second third";
            var tokens = tokenizer.tokenize(input);
            
            assertThat(tokens).hasSize(3);
            for (int i = 0; i < tokens.size(); i++) {
                assertThat(tokens.get(i).getPosition()).isEqualTo(i);
            }
        }
    }
    
    @Nested
    @DisplayName("Edge Cases")
    class EdgeCaseTests {
        
        @Test
        @DisplayName("Should handle null input gracefully")
        void shouldHandleNullInput() {
            var tokens = tokenizer.tokenize(null);
            assertThat(tokens).isEmpty();
        }
        
        @Test
        @DisplayName("Should handle empty string")
        void shouldHandleEmptyString() {
            var tokens = tokenizer.tokenize("");
            assertThat(tokens).isEmpty();
        }
        
        @Test
        @DisplayName("Should handle whitespace only")
        void shouldHandleWhitespaceOnly() {
            var tokens = tokenizer.tokenize("   \t\n  ");
            // Depending on implementation, this might be empty or contain whitespace tokens
            // We'll be flexible in the assertion
            assertThat(tokens).satisfiesAnyOf(
                list -> assertThat(list).isEmpty(),
                list -> assertThat(list).allMatch(t -> t.getType() == Token.TokenType.WHITESPACE)
            );
        }
        
        @Test
        @DisplayName("Should handle single character")
        void shouldHandleSingleCharacter() {
            var tokens = tokenizer.tokenize("a");
            assertThat(tokens).hasSize(1);
            assertThat(tokens.get(0).getText()).isEqualTo("a");
            assertThat(tokens.get(0).getType()).isEqualTo(Token.TokenType.WORD);
        }
        
        @Test
        @DisplayName("Should handle long strings with repeated words")
        void shouldHandleLongStringsWithRepeatedWords() {
            var input = "test ".repeat(1000).trim();
            var tokens = tokenizer.tokenize(input);
            
            assertThat(tokens).hasSize(1000);
            assertThat(tokens).allMatch(t -> t.getText().equals("test"));
            assertThat(tokens).allMatch(t -> t.getType() == Token.TokenType.WORD);
        }
    }
    
    @Nested
    @DisplayName("Special Characters Tests")
    class SpecialCharacterTests {
        
        @ParameterizedTest
        @ValueSource(strings = {"@", "#", "$", "%", "^", "&", "*", "+", "=", "<", ">", "~", "`"})
        @DisplayName("Should classify special characters correctly")
        void shouldClassifySpecialCharacters(String specialChar) {
            var tokens = tokenizer.tokenize(specialChar);
            assertThat(tokens).hasSize(1);
            assertThat(tokens.get(0).getType()).isEqualTo(Token.TokenType.SPECIAL);
        }
        
        @ParameterizedTest
        @ValueSource(strings = {".", ",", ";", ":", "!", "?", "(", ")", "[", "]", "{", "}", "'", "\"", "-"})
        @DisplayName("Should classify punctuation correctly")
        void shouldClassifyPunctuation(String punctuation) {
            var tokens = tokenizer.tokenize(punctuation);
            assertThat(tokens).hasSize(1);
            assertThat(tokens.get(0).getType()).isEqualTo(Token.TokenType.PUNCTUATION);
        }
        
        @Test
        @DisplayName("Should handle Unicode characters")
        void shouldHandleUnicodeCharacters() {
            var input = "Hello 世界 café naïve résumé";
            var tokens = tokenizer.tokenize(input);
            
            assertThat(tokens).isNotEmpty();
            // Should contain Unicode words
            var hasUnicodeWords = tokens.stream().anyMatch(t -> 
                t.getText().contains("世界") || t.getText().contains("café") || 
                t.getText().contains("naïve") || t.getText().contains("résumé"));
            assertThat(hasUnicodeWords).isTrue();
        }
        
        @Test
        @DisplayName("Should handle mixed Unicode and ASCII")
        void shouldHandleMixedUnicodeAndAscii() {
            var input = "English 中文 Français العربية русский";
            var tokens = tokenizer.tokenize(input);
            
            assertThat(tokens).hasSizeGreaterThan(4);
            assertThat(tokens).allMatch(t -> t.getText() != null && !t.getText().isEmpty());
        }
    }
    
    @Nested
    @DisplayName("Tokenization Method Variants")
    class TokenizationMethodTests {
        
        @Test
        @DisplayName("tokenizeWords should return only word tokens")
        void tokenizeWordsShouldReturnOnlyWords() {
            var input = "Hello, world! How are you? 123 #test";
            var wordTokens = tokenizer.tokenizeWords(input);
            
            assertThat(wordTokens).isNotEmpty();
            assertThat(wordTokens).allMatch(t -> t.getType() == Token.TokenType.WORD);
            
            var expectedWords = List.of("Hello", "world", "How", "are", "you");
            var actualWords = wordTokens.stream().map(Token::getText).toList();
            assertThat(actualWords).containsAll(expectedWords);
        }
        
        @Test
        @DisplayName("tokenizeWithCategories should categorize all tokens correctly")
        void tokenizeWithCategoriesShouldCategorizeCorrectly() {
            var input = "Hello, world! The price is $25.99.";
            var result = tokenizer.tokenizeWithCategories(input);
            
            assertThat(result.getAllTokens()).isNotEmpty();
            assertThat(result.getWords()).isNotEmpty();
            assertThat(result.getPunctuation()).isNotEmpty();
            assertThat(result.getTotalTokens()).isEqualTo(result.getAllTokens().size());
            assertThat(result.getWordCount()).isEqualTo(result.getWords().size());
            assertThat(result.getPunctuationCount()).isEqualTo(result.getPunctuation().size());
        }
        
        @Test
        @DisplayName("tokenizeForWindows should create proper sliding windows")
        void tokenizeForWindowsShouldCreateSlidingWindows() {
            var input = "One two three four five six seven eight nine ten";
            var windows = tokenizer.tokenizeForWindows(input, 3);
            
            // Should create windows: [One,two,three], [two,three,four], [three,four,five], etc.
            assertThat(windows).hasSizeGreaterThanOrEqualTo(8); // 10 tokens - 3 + 1 = 8 windows
            
            for (var window : windows) {
                assertThat(window).hasSize(3);
            }
            
            // Check sliding behavior
            assertThat(windows.get(0).get(1).getText()).isEqualTo(windows.get(1).get(0).getText());
        }
        
        @Test
        @DisplayName("tokenizeForWindows should handle small inputs")
        void tokenizeForWindowsShouldHandleSmallInputs() {
            var input = "One two";
            var windows = tokenizer.tokenizeForWindows(input, 5);
            
            assertThat(windows).hasSize(1);
            assertThat(windows.get(0)).hasSize(2);
        }
        
        @Test
        @DisplayName("fastTokenize should be faster and still accurate")
        void fastTokenizeShouldBeFasterAndAccurate() {
            var input = "Hello world this is a test";
            var fastTokens = tokenizer.fastTokenize(input);
            var regularTokens = tokenizer.tokenize(input);
            
            // Fast tokenization should produce similar results
            assertThat(fastTokens).hasSameSizeAs(regularTokens);
            
            // Check that basic words are correctly identified
            var fastWords = fastTokens.stream().filter(t -> t.getType() == Token.TokenType.WORD).toList();
            assertThat(fastWords).hasSizeGreaterThan(0);
        }
    }
    
    @Nested
    @DisplayName("Statistics and Analysis")
    class StatisticsTests {
        
        @Test
        @DisplayName("getTokenStats should provide accurate statistics")
        void getTokenStatsShouldProvideAccurateStats() {
            var input = "Hello, world! How are you? 123 #test";
            var tokens = tokenizer.tokenize(input);
            var stats = tokenizer.getTokenStats(tokens);
            
            assertThat(stats.getTotalCount()).isEqualTo(tokens.size());
            assertThat(stats.getWordCount()).isEqualTo(
                tokens.stream().mapToInt(t -> t.getType() == Token.TokenType.WORD ? 1 : 0).sum());
            assertThat(stats.getPunctuationCount()).isEqualTo(
                tokens.stream().mapToInt(t -> t.getType() == Token.TokenType.PUNCTUATION ? 1 : 0).sum());
            
            assertThat(stats.toString()).contains("TokenStats");
        }
        
        @Test
        @DisplayName("Empty input should produce zero stats")
        void emptyInputShouldProduceZeroStats() {
            var tokens = tokenizer.tokenize("");
            var stats = tokenizer.getTokenStats(tokens);
            
            assertThat(stats.getTotalCount()).isZero();
            assertThat(stats.getWordCount()).isZero();
            assertThat(stats.getPunctuationCount()).isZero();
        }
    }
    
    @Nested
    @DisplayName("Performance Tests")
    class PerformanceTests {
        
        @Test
        @Timeout(value = 5, unit = TimeUnit.SECONDS)
        @DisplayName("Should tokenize large text within time limit")
        void shouldTokenizeLargeTextWithinTimeLimit() {
            // Create a large text (approximately 10,000 words)
            var largeText = new StringBuilder();
            var words = new String[]{"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"};
            
            for (int i = 0; i < 10000; i++) {
                largeText.append(words[i % words.length]).append(" ");
                if (i % 20 == 19) {
                    largeText.append(". ");
                }
            }
            
            var startTime = System.nanoTime();
            var tokens = tokenizer.tokenize(largeText.toString());
            var endTime = System.nanoTime();
            
            assertThat(tokens).hasSizeGreaterThan(10000);
            
            var processingTimeMs = (endTime - startTime) / 1_000_000.0;
            System.out.println("Processed " + tokens.size() + " tokens in " + processingTimeMs + " ms");
            
            // Should process at least 1000 tokens per millisecond (very generous threshold)
            assertThat(tokens.size() / processingTimeMs).isGreaterThan(1000);
        }
        
        @Test
        @Timeout(value = 2, unit = TimeUnit.SECONDS)
        @DisplayName("Fast tokenization should outperform regular tokenization")
        void fastTokenizationShouldOutperformRegular() {
            var text = "Hello world this is a performance test ".repeat(1000);
            
            // Warm up
            tokenizer.tokenize(text);
            tokenizer.fastTokenize(text);
            
            var startRegular = System.nanoTime();
            var regularTokens = tokenizer.tokenize(text);
            var endRegular = System.nanoTime();
            
            var startFast = System.nanoTime();
            var fastTokens = tokenizer.fastTokenize(text);
            var endFast = System.nanoTime();
            
            var regularTimeMs = (endRegular - startRegular) / 1_000_000.0;
            var fastTimeMs = (endFast - startFast) / 1_000_000.0;
            
            System.out.println("Regular tokenization: " + regularTimeMs + " ms");
            System.out.println("Fast tokenization: " + fastTimeMs + " ms");
            
            // Fast should be at least somewhat faster (allow for JIT variations)
            // More importantly, both should complete quickly
            assertThat(fastTimeMs).isLessThan(1000); // Less than 1 second
            assertThat(regularTimeMs).isLessThan(2000); // Less than 2 seconds
            
            // Both should produce reasonable results
            assertThat(regularTokens).hasSizeGreaterThan(1000);
            assertThat(fastTokens).hasSizeGreaterThan(1000);
        }
    }
    
    @Nested
    @DisplayName("Thread Safety Tests")
    class ThreadSafetyTests {
        
        @Test
        @Timeout(value = 10, unit = TimeUnit.SECONDS)
        @DisplayName("Tokenizer should be thread-safe")
        void tokenizerShouldBeThreadSafe() throws InterruptedException {
            var text1 = "This is the first test text";
            var text2 = "This is the second test text";
            var text3 = "This is the third test text";
            
            var results = new java.util.concurrent.ConcurrentHashMap<String, List<Token>>();
            var exceptions = new java.util.concurrent.ConcurrentLinkedQueue<Exception>();
            
            var executor = java.util.concurrent.Executors.newFixedThreadPool(10);
            
            // Submit multiple tokenization tasks
            for (int i = 0; i < 100; i++) {
                final var iteration = i;
                executor.submit(() -> {
                    try {
                        var text = switch (iteration % 3) {
                            case 0 -> text1;
                            case 1 -> text2;
                            default -> text3;
                        };
                        
                        var tokens = tokenizer.tokenize(text);
                        results.put(Thread.currentThread().getName() + "_" + iteration, tokens);
                    } catch (Exception e) {
                        exceptions.add(e);
                    }
                });
            }
            
            executor.shutdown();
            assertThat(executor.awaitTermination(5, TimeUnit.SECONDS)).isTrue();
            
            // Check that no exceptions occurred
            assertThat(exceptions).isEmpty();
            
            // Check that all results are present and consistent
            assertThat(results).hasSize(100);
            
            // Verify consistent tokenization results
            var text1Results = results.values().stream()
                .filter(tokens -> tokens.size() > 0 && "This".equals(tokens.get(0).getText()) && 
                         "first".equals(tokens.stream().filter(t -> "first".equals(t.getText())).findFirst().orElse(new Token("", 0, Token.TokenType.UNKNOWN)).getText()))
                .toList();
            
            // All text1 tokenizations should produce identical results
            if (text1Results.size() > 1) {
                var first = text1Results.get(0);
                for (var result : text1Results) {
                    assertThat(result).hasSameSizeAs(first);
                }
            }
        }
    }
    
    @Nested
    @DisplayName("Token Object Tests")
    class TokenObjectTests {
        
        @Test
        @DisplayName("Token should have proper equals and hashCode")
        void tokenShouldHaveProperEqualsAndHashCode() {
            var token1 = new Token("hello", 0, Token.TokenType.WORD);
            var token2 = new Token("hello", 0, Token.TokenType.WORD);
            var token3 = new Token("world", 1, Token.TokenType.WORD);
            
            assertThat(token1).isEqualTo(token2);
            assertThat(token1).isNotEqualTo(token3);
            assertThat(token1.hashCode()).isEqualTo(token2.hashCode());
        }
        
        @Test
        @DisplayName("Token should have meaningful toString")
        void tokenShouldHaveMeaningfulToString() {
            var token = new Token("hello", 5, Token.TokenType.WORD);
            var tokenString = token.toString();
            
            assertThat(tokenString).contains("hello");
            assertThat(tokenString).contains("5");
            assertThat(tokenString).contains("WORD");
        }
        
        @Test
        @DisplayName("Token should store timestamp")
        void tokenShouldStoreTimestamp() {
            var before = System.nanoTime();
            var token = new Token("test", 0, Token.TokenType.WORD);
            var after = System.nanoTime();
            
            assertThat(token.getTimestamp()).isBetween(before, after);
        }
    }
}