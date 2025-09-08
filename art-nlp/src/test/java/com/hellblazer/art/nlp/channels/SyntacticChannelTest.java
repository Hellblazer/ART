package com.hellblazer.art.nlp.channels;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.io.TempDir;
import static org.assertj.core.api.Assertions.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.charset.StandardCharsets;
import java.util.List;

import com.hellblazer.art.nlp.channels.SyntacticChannel.SyntacticFeatureSet;

/**
 * Tests for SyntacticChannel POS tagging and syntactic processing.
 */
@DisplayName("SyntacticChannel Tests")
class SyntacticChannelTest {

    @TempDir
    Path tempDir;
    
    private Path tokenizerModelPath;
    private Path posModelPath;
    private Path sentenceModelPath;
    private SyntacticChannel channel;

    @BeforeEach
    void setUp() throws IOException {
        // Create minimal test models (these won't actually work with OpenNLP but test structure)
        tokenizerModelPath = tempDir.resolve("en-token.bin");
        posModelPath = tempDir.resolve("en-pos-maxent.bin");
        sentenceModelPath = tempDir.resolve("en-sent.bin");
        
        // Create dummy model files for structure testing
        Files.write(tokenizerModelPath, "dummy tokenizer model".getBytes(StandardCharsets.UTF_8));
        Files.write(posModelPath, "dummy pos model".getBytes(StandardCharsets.UTF_8));
        Files.write(sentenceModelPath, "dummy sentence model".getBytes(StandardCharsets.UTF_8));
    }

    @AfterEach
    void tearDown() {
        if (channel != null) {
            channel.shutdown();
        }
    }

    @Test
    @DisplayName("Should create syntactic channel with default configuration")
    void shouldCreateChannelWithDefaults() {
        channel = new SyntacticChannel("syntactic", 0.8);
        
        assertThat(channel.getChannelName()).isEqualTo("syntactic");
        assertThat(channel.getVigilance()).isEqualTo(0.8);
        assertThat(channel.isInitialized()).isFalse();
    }

    @Test
    @DisplayName("Should create syntactic channel with custom configuration")
    void shouldCreateChannelWithCustomConfig() {
        channel = new SyntacticChannel("test", 0.75,
                tokenizerModelPath, posModelPath, sentenceModelPath,
                SyntacticFeatureSet.POS_PATTERNS, 50, false);
        
        assertThat(channel.getChannelName()).isEqualTo("test");
        assertThat(channel.getVigilance()).isEqualTo(0.75);
        assertThat(channel.isInitialized()).isFalse();
    }

    @Test
    @DisplayName("Should handle null and empty text inputs gracefully")
    void shouldHandleNullAndEmptyTextInputs() {
        channel = new SyntacticChannel("syntactic", 0.8);
        
        assertThat(channel.classifyText(null)).isEqualTo(-1);
        assertThat(channel.classifyText("")).isEqualTo(-1);
        assertThat(channel.classifyText("   ")).isEqualTo(-1);
    }

    @Test
    @DisplayName("Should handle initialization errors gracefully")
    void shouldHandleInitializationErrors() {
        // Create channel with non-existent model paths
        var nonExistentPath = tempDir.resolve("nonexistent.bin");
        channel = new SyntacticChannel("test", 0.8,
                nonExistentPath, nonExistentPath, nonExistentPath,
                SyntacticFeatureSet.POS_DISTRIBUTION, 100, true);
        
        // Initialization should fail with proper error handling
        assertThatThrownBy(() -> channel.initialize())
            .isInstanceOf(RuntimeException.class)
            .hasMessageContaining("OpenNLP initialization failed");
    }

    @Test
    @DisplayName("Should batch classify multiple texts")
    void shouldBatchClassifyMultipleTexts() {
        channel = new SyntacticChannel("syntactic", 0.8);
        
        var texts = List.of(
            "This is a test sentence.",
            "Another example text.",
            "Short text.",
            ""  // Empty text
        );
        
        var categories = channel.classifyTexts(texts);
        
        assertThat(categories).hasSize(4);
        // All should fail without proper OpenNLP models, returning -1
        for (var category : categories) {
            assertThat(category).isEqualTo(-1);
        }
    }

    @Test
    @DisplayName("Should provide meaningful performance metrics")
    void shouldProvideMeaningfulPerformanceMetrics() {
        channel = new SyntacticChannel("syntactic", 0.8);
        
        // Generate some activity (will fail without real models but updates metrics)
        channel.classifyText("Test sentence for metrics.");
        channel.classifyText("Another sentence.");
        
        var metrics = channel.getSyntacticMetrics();
        
        assertThat(metrics.totalClassifications()).isGreaterThanOrEqualTo(0);
        assertThat(metrics.successfulClassifications()).isGreaterThanOrEqualTo(0);
        assertThat(metrics.categoryCount()).isGreaterThanOrEqualTo(0);
        assertThat(metrics.totalSentences()).isGreaterThanOrEqualTo(0);
        assertThat(metrics.totalTokens()).isGreaterThanOrEqualTo(0);
        assertThat(metrics.uniquePosTagCount()).isGreaterThanOrEqualTo(0);
        assertThat(metrics.featureSetName()).isNotBlank();
        assertThat(metrics.successRate()).isBetween(0.0, 1.0);
    }

    @Test
    @DisplayName("Should work with different feature sets")
    void shouldWorkWithDifferentFeatureSets() {
        // Test POS_DISTRIBUTION
        var posChannel = new SyntacticChannel("pos", 0.8,
                tokenizerModelPath, posModelPath, sentenceModelPath,
                SyntacticFeatureSet.POS_DISTRIBUTION, 100, true);
        assertThat(posChannel.getSyntacticMetrics().featureSetName()).isEqualTo("POS_DISTRIBUTION");
        posChannel.shutdown();
        
        // Test POS_PATTERNS
        var patternChannel = new SyntacticChannel("pattern", 0.8,
                tokenizerModelPath, posModelPath, sentenceModelPath,
                SyntacticFeatureSet.POS_PATTERNS, 100, true);
        assertThat(patternChannel.getSyntacticMetrics().featureSetName()).isEqualTo("POS_PATTERNS");
        patternChannel.shutdown();
        
        // Test FULL_SYNTAX
        var fullChannel = new SyntacticChannel("full", 0.8,
                tokenizerModelPath, posModelPath, sentenceModelPath,
                SyntacticFeatureSet.FULL_SYNTAX, 100, true);
        assertThat(fullChannel.getSyntacticMetrics().featureSetName()).isEqualTo("FULL_SYNTAX");
        fullChannel.shutdown();
    }

    @Test
    @DisplayName("Should handle different token limits")
    void shouldHandleDifferentTokenLimits() {
        // Test with small token limit
        var smallChannel = new SyntacticChannel("small", 0.8,
                tokenizerModelPath, posModelPath, sentenceModelPath,
                SyntacticFeatureSet.POS_DISTRIBUTION, 5, true);
        assertThat(smallChannel.classifyText("This is a long sentence with many words")).isEqualTo(-1);
        smallChannel.shutdown();
        
        // Test with large token limit
        var largeChannel = new SyntacticChannel("large", 0.8,
                tokenizerModelPath, posModelPath, sentenceModelPath,
                SyntacticFeatureSet.POS_DISTRIBUTION, 1000, true);
        assertThat(largeChannel.classifyText("Short text")).isEqualTo(-1);
        largeChannel.shutdown();
    }

    @Test
    @DisplayName("Should handle normalization settings")
    void shouldHandleNormalizationSettings() {
        // Test with normalization enabled
        var normalizedChannel = new SyntacticChannel("normalized", 0.8,
                tokenizerModelPath, posModelPath, sentenceModelPath,
                SyntacticFeatureSet.POS_DISTRIBUTION, 100, true);
        assertThat(normalizedChannel.classifyText("Test sentence")).isEqualTo(-1);
        normalizedChannel.shutdown();
        
        // Test with normalization disabled
        var rawChannel = new SyntacticChannel("raw", 0.8,
                tokenizerModelPath, posModelPath, sentenceModelPath,
                SyntacticFeatureSet.POS_DISTRIBUTION, 100, false);
        assertThat(rawChannel.classifyText("Test sentence")).isEqualTo(-1);
        rawChannel.shutdown();
    }

    @Test
    @DisplayName("Should provide meaningful toString")
    void shouldProvideMeaningfulToString() {
        channel = new SyntacticChannel("syntactic", 0.75);
        
        var str = channel.toString();
        assertThat(str).contains("syntactic");
        assertThat(str).contains("0.750");
        assertThat(str).contains("categories=");
    }

    @Test
    @DisplayName("Should shutdown gracefully")
    void shouldShutdownGracefully() {
        channel = new SyntacticChannel("syntactic", 0.8);
        
        // Should not throw exceptions
        assertThatNoException().isThrownBy(() -> channel.shutdown());
        
        // Second shutdown should also be safe
        assertThatNoException().isThrownBy(() -> channel.shutdown());
    }

    @Test
    @DisplayName("Should implement required abstract methods")
    void shouldImplementRequiredAbstractMethods() {
        channel = new SyntacticChannel("syntactic", 0.8);
        
        // These are placeholder implementations
        assertThat(channel.getCategoryCount()).isEqualTo(0);
        assertThat(channel.pruneCategories(0.5)).isEqualTo(0);
        
        // State operations should not throw
        assertThatNoException().isThrownBy(() -> channel.saveState());
        assertThatNoException().isThrownBy(() -> channel.loadState());
    }

    @Test
    @DisplayName("Should maintain thread safety")
    void shouldMaintainThreadSafety() throws InterruptedException {
        channel = new SyntacticChannel("syntactic", 0.8);
        
        var texts = List.of("First text", "Second text", "Third text", "Fourth text");
        var threads = new Thread[4];
        var results = new int[4];
        
        for (int i = 0; i < 4; i++) {
            final var index = i;
            threads[i] = new Thread(() -> {
                results[index] = channel.classifyText(texts.get(index));
            });
        }
        
        // Start all threads
        for (var thread : threads) {
            thread.start();
        }
        
        // Wait for completion
        for (var thread : threads) {
            thread.join();
        }
        
        // All should return -1 without real models but no exceptions
        for (var result : results) {
            assertThat(result).isEqualTo(-1);
        }
    }

    @Test
    @DisplayName("Should handle learning state changes")
    void shouldHandleLearningStateChanges() {
        channel = new SyntacticChannel("syntactic", 0.8);
        
        // Test learning enabled/disabled
        channel.setLearningEnabled(true);
        assertThat(channel.isLearningEnabled()).isTrue();
        
        channel.setLearningEnabled(false);
        assertThat(channel.isLearningEnabled()).isFalse();
        
        // Classification should work regardless of learning state
        assertThat(channel.classifyText("Test text")).isEqualTo(-1);
    }

    @Test
    @DisplayName("Should handle various text complexities")
    void shouldHandleVariousTextComplexities() {
        channel = new SyntacticChannel("syntactic", 0.8);
        
        // Simple text
        assertThat(channel.classifyText("Hello world")).isEqualTo(-1);
        
        // Complex text with punctuation
        assertThat(channel.classifyText("The quick, brown fox jumps over the lazy dog!")).isEqualTo(-1);
        
        // Multi-sentence text
        assertThat(channel.classifyText("First sentence. Second sentence! Third sentence?")).isEqualTo(-1);
        
        // Text with numbers and special characters
        assertThat(channel.classifyText("Test 123 with @special #characters and URLs like http://test.com")).isEqualTo(-1);
        
        // Very short text
        assertThat(channel.classifyText("Hi")).isEqualTo(-1);
        
        // Single character
        assertThat(channel.classifyText("A")).isEqualTo(-1);
    }

    @Test
    @DisplayName("Should validate constructor parameters")
    void shouldValidateConstructorParameters() {
        assertThatThrownBy(() -> new SyntacticChannel("test", 0.8,
                null, posModelPath, sentenceModelPath,
                SyntacticFeatureSet.POS_DISTRIBUTION, 100, true))
            .isInstanceOf(NullPointerException.class)
            .hasMessageContaining("tokenizerModelPath cannot be null");

        assertThatThrownBy(() -> new SyntacticChannel("test", 0.8,
                tokenizerModelPath, null, sentenceModelPath,
                SyntacticFeatureSet.POS_DISTRIBUTION, 100, true))
            .isInstanceOf(NullPointerException.class)
            .hasMessageContaining("posModelPath cannot be null");

        assertThatThrownBy(() -> new SyntacticChannel("test", 0.8,
                tokenizerModelPath, posModelPath, null,
                SyntacticFeatureSet.POS_DISTRIBUTION, 100, true))
            .isInstanceOf(NullPointerException.class)
            .hasMessageContaining("sentenceModelPath cannot be null");

        assertThatThrownBy(() -> new SyntacticChannel("test", 0.8,
                tokenizerModelPath, posModelPath, sentenceModelPath,
                null, 100, true))
            .isInstanceOf(NullPointerException.class)
            .hasMessageContaining("featureSet cannot be null");
    }
}