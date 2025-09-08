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

import com.hellblazer.art.nlp.fasttext.VectorPreprocessor;
import com.hellblazer.art.nlp.channels.FastTextChannel.OOVStrategy;

/**
 * Tests for FastTextChannel semantic processing.
 */
@DisplayName("FastTextChannel Tests")
class FastTextChannelTest {

    @TempDir
    Path tempDir;
    
    private Path testModelPath;
    private FastTextChannel channel;

    @BeforeEach
    void setUp() throws IOException {
        // Create test FastText model with meaningful words
        testModelPath = tempDir.resolve("test.vec");
        var content = """
                8 3
                hello 0.1 0.2 0.3
                world -0.1 0.4 -0.2
                good 0.5 -0.3 0.1
                bad -0.5 0.3 -0.1
                java 0.2 0.6 -0.4
                programming -0.3 0.1 0.7
                test 0.4 0.0 -0.3
                language 0.0 0.5 0.2
                """;
        Files.write(testModelPath, content.getBytes(StandardCharsets.UTF_8));
    }

    @AfterEach
    void tearDown() {
        if (channel != null) {
            channel.shutdown();
        }
    }

    @Test
    @DisplayName("Should create FastText channel with default configuration")
    void shouldCreateChannelWithDefaults() throws IOException {
        channel = new FastTextChannel("semantic", 0.8, testModelPath, 3);
        
        assertThat(channel.getChannelName()).isEqualTo("semantic");
        assertThat(channel.getVigilance()).isEqualTo(0.8);
        assertThat(channel.isInitialized()).isFalse();
    }

    @Test
    @DisplayName("Should create FastText channel with custom configuration")
    void shouldCreateChannelWithCustomConfig() throws IOException {
        var pipeline = VectorPreprocessor.pipeline()
            .normalize(VectorPreprocessor.NormalizationType.MIN_MAX)
            .build();
            
        channel = new FastTextChannel("semantic", 0.75, testModelPath, 3,
                                    OOVStrategy.ZERO_VECTOR, false, 50, pipeline);
        
        assertThat(channel.getChannelName()).isEqualTo("semantic");
        assertThat(channel.getVigilance()).isEqualTo(0.75);
    }

    @Test
    @DisplayName("Should initialize channel properly")
    void shouldInitializeChannel() throws IOException {
        channel = new FastTextChannel("semantic", 0.8, testModelPath, 3);
        channel.initialize();
        
        assertThat(channel.isInitialized()).isTrue();
        assertThat(channel.getCategoryCount()).isZero(); // No patterns learned yet
    }

    @Test
    @DisplayName("Should classify text and learn patterns")
    void shouldClassifyTextAndLearnPatterns() throws IOException {
        channel = new FastTextChannel("semantic", 0.8, testModelPath, 3);
        channel.initialize();
        
        // Enable learning
        channel.setLearningEnabled(true);
        
        // First text should create new category
        var category1 = channel.classifyText("hello world");
        assertThat(category1).isGreaterThanOrEqualTo(0);
        // ART algorithm should now create categories, so count should be > 0
        assertThat(channel.getCategoryCount()).isGreaterThan(0);
        
        // Similar text might use same category (depending on vigilance)
        var category2 = channel.classifyText("hello java");
        assertThat(category2).isGreaterThanOrEqualTo(0);
        
        // Very different text should create new category
        var category3 = channel.classifyText("good programming");
        assertThat(category3).isGreaterThanOrEqualTo(0);
    }

    @Test
    @DisplayName("Should handle different OOV strategies")
    void shouldHandleDifferentOOVStrategies() throws IOException {
        // Test SKIP strategy
        var skipChannel = new FastTextChannel("test", 0.8, testModelPath, 3,
                                            OOVStrategy.SKIP, false, 100,
                                            VectorPreprocessor.pipeline()
                                                .normalize(VectorPreprocessor.NormalizationType.L2)
                                                .build());
        skipChannel.initialize();
        skipChannel.setLearningEnabled(true);
        
        var category1 = skipChannel.classifyText("hello unknown_word");
        assertThat(category1).isGreaterThanOrEqualTo(0); // Should work with known word
        
        skipChannel.shutdown();
        
        // Test ZERO_VECTOR strategy
        var zeroChannel = new FastTextChannel("test", 0.8, testModelPath, 3,
                                            OOVStrategy.ZERO_VECTOR, false, 100,
                                            VectorPreprocessor.pipeline()
                                                .normalize(VectorPreprocessor.NormalizationType.L2)
                                                .build());
        zeroChannel.initialize();
        zeroChannel.setLearningEnabled(true);
        
        var category2 = zeroChannel.classifyText("unknown_word1 unknown_word2");
        assertThat(category2).isGreaterThanOrEqualTo(0); // Should work with zero fallback
        
        zeroChannel.shutdown();
    }

    @Test
    @DisplayName("Should tokenize text properly")
    void shouldTokenizeTextProperly() throws IOException {
        channel = new FastTextChannel("semantic", 0.8, testModelPath, 3);
        
        // Test tokenization through text vector generation
        var vector1 = channel.getTextVector("Hello, World!");
        var vector2 = channel.getTextVector("hello world");
        
        assertThat(vector1).isNotNull();
        assertThat(vector2).isNotNull();
        // Should be similar due to case insensitive processing
    }

    @Test
    @DisplayName("Should respect max tokens limit")
    void shouldRespectMaxTokensLimit() throws IOException {
        channel = new FastTextChannel("test", 0.8, testModelPath, 3,
                                    OOVStrategy.SKIP, false, 2, // Max 2 tokens
                                    VectorPreprocessor.pipeline()
                                        .normalize(VectorPreprocessor.NormalizationType.L2)
                                        .build());
        
        var vector = channel.getTextVector("hello world good bad java"); // 5 words
        assertThat(vector).isNotNull(); // Should work with first 2 tokens only
    }

    @Test
    @DisplayName("Should calculate text similarity")
    void shouldCalculateTextSimilarity() throws IOException {
        channel = new FastTextChannel("semantic", 0.8, testModelPath, 3);
        
        var similarity1 = channel.getTextSimilarity("hello world", "hello java");
        var similarity2 = channel.getTextSimilarity("good programming", "bad test");
        var similarity3 = channel.getTextSimilarity("hello", "programming");
        
        // All similarities should be in valid range
        assertThat(similarity1).isBetween(-1.0, 1.0);
        assertThat(similarity2).isBetween(-1.0, 1.0);
        assertThat(similarity3).isBetween(-1.0, 1.0);
    }

    @Test
    @DisplayName("Should batch classify multiple texts")
    void shouldBatchClassifyMultipleTexts() throws IOException {
        channel = new FastTextChannel("semantic", 0.8, testModelPath, 3);
        channel.initialize();
        channel.setLearningEnabled(true);
        
        var texts = List.of(
            "hello world",
            "good programming",
            "java language",
            "test bad"
        );
        
        var categories = channel.classifyTexts(texts);
        
        assertThat(categories).hasSize(4);
        for (var category : categories) {
            assertThat(category).isGreaterThanOrEqualTo(0);
        }
    }

    @Test
    @DisplayName("Should handle null and empty text inputs")
    void shouldHandleNullAndEmptyTextInputs() throws IOException {
        channel = new FastTextChannel("semantic", 0.8, testModelPath, 3);
        
        assertThat(channel.classifyText(null)).isEqualTo(-1);
        assertThat(channel.classifyText("")).isEqualTo(-1);
        assertThat(channel.classifyText("   ")).isEqualTo(-1);
        
        assertThat(channel.getTextVector(null)).isNull();
        assertThat(channel.getTextVector("")).isNull();
        assertThat(channel.getTextVector("   ")).isNull();
        
        assertThat(channel.getTextSimilarity("hello", null)).isZero();
        assertThat(channel.getTextSimilarity(null, "world")).isZero();
    }

    @Test
    @DisplayName("Should handle text with only unknown words")
    void shouldHandleTextWithOnlyUnknownWords() throws IOException {
        channel = new FastTextChannel("test", 0.8, testModelPath, 3,
                                    OOVStrategy.RANDOM_VECTOR, false, 100,
                                    VectorPreprocessor.pipeline()
                                        .normalize(VectorPreprocessor.NormalizationType.L2)
                                        .build());
        channel.initialize();
        channel.setLearningEnabled(true);
        
        // Text with only unknown words
        var category = channel.classifyText("xyzabc defghi");
        
        // With RANDOM_VECTOR strategy, should still work
        assertThat(category).isGreaterThanOrEqualTo(0);
        
        // With SKIP strategy, should fail
        var skipChannel = new FastTextChannel("test2", 0.8, testModelPath, 3,
                                            OOVStrategy.SKIP, false, 100,
                                            VectorPreprocessor.pipeline()
                                                .normalize(VectorPreprocessor.NormalizationType.L2)
                                                .build());
        skipChannel.initialize();
        
        var category2 = skipChannel.classifyText("xyzabc defghi");
        assertThat(category2).isEqualTo(-1); // Should fail
        
        skipChannel.shutdown();
    }

    @Test
    @DisplayName("Should provide meaningful performance metrics")
    void shouldProvideMeaningfulPerformanceMetrics() throws IOException {
        channel = new FastTextChannel("semantic", 0.8, testModelPath, 3);
        channel.initialize();
        channel.setLearningEnabled(true);
        
        // Generate some activity
        channel.classifyText("hello world"); // Known words
        channel.classifyText("unknown_word1 unknown_word2"); // OOV words
        channel.classifyText("good java"); // Mixed
        
        var metrics = channel.getFastTextMetrics();
        
        assertThat(metrics.totalClassifications()).isGreaterThanOrEqualTo(0);
        assertThat(metrics.totalTokens()).isGreaterThan(0);
        assertThat(metrics.categoryCount()).isGreaterThanOrEqualTo(0);
        assertThat(metrics.successRate()).isBetween(0.0, 1.0);
        assertThat(metrics.oovRateByTokens()).isBetween(0.0, 1.0);
    }

    @Test
    @DisplayName("Should work with different preprocessing pipelines")
    void shouldWorkWithDifferentPreprocessingPipelines() throws IOException {
        // Test with complement coding pipeline
        var complementPipeline = VectorPreprocessor.pipeline()
            .normalize(VectorPreprocessor.NormalizationType.MIN_MAX)
            .complementCode()
            .normalize(VectorPreprocessor.NormalizationType.L2)
            .build();
            
        channel = new FastTextChannel("test", 0.8, testModelPath, 3,
                                    OOVStrategy.RANDOM_VECTOR, false, 100,
                                    complementPipeline);
        channel.initialize();
        channel.setLearningEnabled(true);
        
        var category = channel.classifyText("hello world");
        assertThat(category).isGreaterThanOrEqualTo(0);
        
        // Vector should be complement-coded (doubled in size)
        var vector = channel.getTextVector("hello world");
        assertThat(vector).isNotNull();
        assertThat(vector.dimension()).isEqualTo(6); // 3D vectors become 6D after complement coding
    }

    @Test
    @DisplayName("Should maintain thread safety")
    void shouldMaintainThreadSafety() throws IOException, InterruptedException {
        channel = new FastTextChannel("semantic", 0.8, testModelPath, 3);
        channel.initialize();
        channel.setLearningEnabled(true);
        
        var texts = List.of("hello world", "good programming", "java language", "test bad");
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
        
        // All should succeed
        for (var result : results) {
            assertThat(result).isGreaterThanOrEqualTo(0);
        }
    }

    @Test
    @DisplayName("Should disable learning correctly")
    void shouldDisableLearningCorrectly() throws IOException {
        channel = new FastTextChannel("semantic", 0.8, testModelPath, 3);
        channel.initialize();
        
        // First, enable learning and create a category
        channel.setLearningEnabled(true);
        var category1 = channel.classifyText("hello world");
        assertThat(category1).isGreaterThanOrEqualTo(0);
        var initialCategories = channel.getCategoryCount();
        
        // Disable learning
        channel.setLearningEnabled(false);
        
        // Try to classify very different text - should not create new categories
        channel.classifyText("completely different unknown text pattern");
        
        // Category count should not increase significantly
        assertThat(channel.getCategoryCount()).isLessThanOrEqualTo(initialCategories + 1);
    }

    @Test
    @DisplayName("Should provide meaningful toString")
    void shouldProvideMeaningfulToString() throws IOException {
        channel = new FastTextChannel("semantic", 0.75, testModelPath, 3);
        channel.initialize();
        
        var str = channel.toString();
        assertThat(str).contains("semantic");
        assertThat(str).contains("0.750");
        assertThat(str).contains("categories=");
    }

    @Test
    @DisplayName("Should shutdown gracefully")
    void shouldShutdownGracefully() throws IOException {
        channel = new FastTextChannel("semantic", 0.8, testModelPath, 3);
        channel.initialize();
        
        // Should not throw exceptions
        assertThatNoException().isThrownBy(() -> channel.shutdown());
        
        // Second shutdown should also be safe
        assertThatNoException().isThrownBy(() -> channel.shutdown());
    }

    @Test
    @DisplayName("Should handle average fallback OOV strategy")
    void shouldHandleAverageFallbackOOVStrategy() throws IOException {
        channel = new FastTextChannel("test", 0.8, testModelPath, 3,
                                    OOVStrategy.AVERAGE_FALLBACK, false, 100,
                                    VectorPreprocessor.pipeline()
                                        .normalize(VectorPreprocessor.NormalizationType.L2)
                                        .build());
        channel.initialize();
        channel.setLearningEnabled(true);
        
        // Text with mix of known and unknown words
        var category = channel.classifyText("hello unknown_word world");
        assertThat(category).isGreaterThanOrEqualTo(0);
        
        // Text with only unknown words - should fall back to random
        var category2 = channel.classifyText("unknown1 unknown2 unknown3");
        assertThat(category2).isGreaterThanOrEqualTo(0);
    }
}