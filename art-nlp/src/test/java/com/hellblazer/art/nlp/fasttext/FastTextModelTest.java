package com.hellblazer.art.nlp.fasttext;

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

/**
 * Tests for FastTextModel word embedding functionality.
 */
@DisplayName("FastTextModel Tests")
class FastTextModelTest {

    @TempDir
    Path tempDir;
    
    private Path testModelPath;
    private FastTextModel model;

    @BeforeEach
    void setUp() throws IOException {
        // Create a small test FastText model file
        testModelPath = tempDir.resolve("test.vec");
        var content = """
                5 3
                hello 0.1 0.2 0.3
                world -0.1 0.4 -0.2
                test 0.5 -0.3 0.1
                java 0.2 0.6 -0.4
                art -0.3 0.1 0.7
                """;
        Files.write(testModelPath, content.getBytes(StandardCharsets.UTF_8));
    }

    @AfterEach
    void tearDown() {
        if (model != null) {
            model.close();
        }
    }

    @Test
    @DisplayName("Should create model with default configuration")
    void shouldCreateModelWithDefaults() {
        model = new FastTextModel(testModelPath);
        
        assertThat(model.getDimensions()).isEqualTo(300); // Default
        assertThat(model.isLoaded()).isFalse();
        assertThat(model.getCacheSize()).isZero();
    }

    @Test
    @DisplayName("Should create model with custom configuration")
    void shouldCreateModelWithCustomConfig() {
        model = new FastTextModel(testModelPath, 3, false, 100);
        
        assertThat(model.getDimensions()).isEqualTo(3);
        assertThat(model.isLoaded()).isFalse();
        assertThat(model.getCacheSize()).isZero();
    }

    @Test
    @DisplayName("Should initialize model and load metadata")
    void shouldInitializeModel() throws IOException {
        model = new FastTextModel(testModelPath, 3, true, 100);
        model.initialize();
        
        assertThat(model.isLoaded()).isTrue();
        assertThat(model.getVocabularySize()).isEqualTo(5);
        assertThat(model.getDimensions()).isEqualTo(3);
    }

    @Test
    @DisplayName("Should handle dimension mismatch")
    void shouldHandleDimensionMismatch() {
        model = new FastTextModel(testModelPath, 100, true, 100); // Wrong dimensions
        
        assertThatThrownBy(() -> model.initialize())
            .isInstanceOf(IOException.class)
            .hasMessageContaining("Dimension mismatch");
    }

    @Test
    @DisplayName("Should load word vectors correctly")
    void shouldLoadWordVectors() throws IOException {
        model = new FastTextModel(testModelPath, 3, false, 100); // No normalization
        model.initialize();
        
        var helloVector = model.getWordVector("hello");
        assertThat(helloVector).isNotNull();
        assertThat(helloVector.dimension()).isEqualTo(3);
        var values = helloVector.values();
        assertThat(values[0]).isCloseTo(0.1, offset(1e-6));
        assertThat(values[1]).isCloseTo(0.2, offset(1e-6));
        assertThat(values[2]).isCloseTo(0.3, offset(1e-6));
        
        var worldVector = model.getWordVector("world");
        assertThat(worldVector).isNotNull();
        var worldValues = worldVector.values();
        assertThat(worldValues[0]).isCloseTo(-0.1, offset(1e-6));
        assertThat(worldValues[1]).isCloseTo(0.4, offset(1e-6));
        assertThat(worldValues[2]).isCloseTo(-0.2, offset(1e-6));
    }

    @Test
    @DisplayName("Should normalize vectors when enabled")
    void shouldNormalizeVectors() throws IOException {
        model = new FastTextModel(testModelPath, 3, true, 100); // With normalization
        model.initialize();
        
        var testVector = model.getWordVector("test");
        assertThat(testVector).isNotNull();
        
        // Calculate expected L2 norm: sqrt(0.5^2 + (-0.3)^2 + 0.1^2) = sqrt(0.35) ≈ 0.5916
        var values = testVector.values();
        var norm = Math.sqrt(values[0]*values[0] + values[1]*values[1] + values[2]*values[2]);
        assertThat(norm).isCloseTo(1.0, offset(1e-6)); // Should be unit vector
    }

    @Test
    @DisplayName("Should handle case insensitive lookup")
    void shouldHandleCaseInsensitiveLookup() throws IOException {
        model = new FastTextModel(testModelPath, 3, false, 100);
        model.initialize();
        
        var vector1 = model.getWordVector("HELLO");
        var vector2 = model.getWordVector("hello");
        var vector3 = model.getWordVector("Hello");
        
        assertThat(vector1).isNotNull();
        assertThat(vector2).isNotNull();
        assertThat(vector3).isNotNull();
        
        // All should be the same (case insensitive)
        assertThat(vector1.values()).isEqualTo(vector2.values());
        assertThat(vector2.values()).isEqualTo(vector3.values());
    }

    @Test
    @DisplayName("Should return null for OOV words")
    void shouldReturnNullForOOVWords() throws IOException {
        model = new FastTextModel(testModelPath, 3, false, 100);
        model.initialize();
        
        var oovVector = model.getWordVector("nonexistent");
        assertThat(oovVector).isNull();
        
        var stats = model.getStats();
        assertThat(stats.oovCount()).isEqualTo(1);
    }

    @Test
    @DisplayName("Should cache vectors effectively")
    void shouldCacheVectors() throws IOException {
        model = new FastTextModel(testModelPath, 3, false, 2); // Small cache
        model.initialize();
        
        // First access - cache miss
        var vector1 = model.getWordVector("hello");
        assertThat(vector1).isNotNull();
        assertThat(model.getCacheSize()).isEqualTo(1);
        
        // Second access - cache hit
        var vector2 = model.getWordVector("hello");
        assertThat(vector2).isNotNull();
        assertThat(vector1.values()).isEqualTo(vector2.values());
        
        var stats = model.getStats();
        assertThat(stats.cacheHits()).isEqualTo(1);
        assertThat(stats.cacheMisses()).isEqualTo(1);
        assertThat(stats.cacheHitRate()).isEqualTo(0.5);
    }

    @Test
    @DisplayName("Should respect cache size limit")
    void shouldRespectCacheSizeLimit() throws IOException {
        model = new FastTextModel(testModelPath, 3, false, 2); // Cache limit = 2
        model.initialize();
        
        // Fill cache
        model.getWordVector("hello");
        model.getWordVector("world");
        assertThat(model.getCacheSize()).isEqualTo(2);
        
        // Adding third word should not increase cache size
        model.getWordVector("test");
        assertThat(model.getCacheSize()).isEqualTo(2); // Still at limit
    }

    @Test
    @DisplayName("Should load multiple vectors efficiently")
    void shouldLoadMultipleVectors() throws IOException {
        model = new FastTextModel(testModelPath, 3, false, 100);
        model.initialize();
        
        var vectors = model.getWordVectors("hello", "world", "nonexistent", "test");
        
        assertThat(vectors).hasSize(3); // Only found words
        assertThat(vectors).containsKey("hello");
        assertThat(vectors).containsKey("world");
        assertThat(vectors).containsKey("test");
        assertThat(vectors).doesNotContainKey("nonexistent");
    }

    @Test
    @DisplayName("Should provide zero vector fallback")
    void shouldProvideZeroVector() throws IOException {
        model = new FastTextModel(testModelPath, 3, false, 100);
        model.initialize();
        
        var zeroVector = model.getZeroVector();
        assertThat(zeroVector).isNotNull();
        assertThat(zeroVector.dimension()).isEqualTo(3);
        assertThat(zeroVector.values()).containsOnly(0.0);
    }

    @Test
    @DisplayName("Should provide random vector fallback")
    void shouldProvideRandomVector() throws IOException {
        model = new FastTextModel(testModelPath, 3, true, 100); // With normalization
        model.initialize();
        
        var randomVector1 = model.getRandomVector();
        var randomVector2 = model.getRandomVector();
        
        assertThat(randomVector1).isNotNull();
        assertThat(randomVector2).isNotNull();
        assertThat(randomVector1.dimension()).isEqualTo(3);
        assertThat(randomVector2.dimension()).isEqualTo(3);
        
        // Vectors should be different
        assertThat(randomVector1.values()).isNotEqualTo(randomVector2.values());
        
        // Should be normalized (unit vectors)
        var norm1 = calculateNorm(randomVector1.values());
        var norm2 = calculateNorm(randomVector2.values());
        assertThat(norm1).isCloseTo(1.0, offset(1e-5));
        assertThat(norm2).isCloseTo(1.0, offset(1e-5));
    }

    @Test
    @DisplayName("Should check word existence")
    void shouldCheckWordExistence() throws IOException {
        model = new FastTextModel(testModelPath, 3, false, 100);
        model.initialize();
        
        assertThat(model.hasWord("hello")).isTrue();
        assertThat(model.hasWord("HELLO")).isTrue(); // Case insensitive
        assertThat(model.hasWord("nonexistent")).isFalse();
    }

    @Test
    @DisplayName("Should handle null and blank inputs")
    void shouldHandleNullAndBlankInputs() throws IOException {
        model = new FastTextModel(testModelPath, 3, false, 100);
        model.initialize();
        
        assertThat(model.getWordVector(null)).isNull();
        assertThat(model.getWordVector("")).isNull();
        assertThat(model.getWordVector("   ")).isNull();
        
        assertThat(model.hasWord(null)).isFalse();
        assertThat(model.hasWord("")).isFalse();
        assertThat(model.hasWord("   ")).isFalse();
    }

    @Test
    @DisplayName("Should provide meaningful statistics")
    void shouldProvideMeaningfulStatistics() throws IOException {
        model = new FastTextModel(testModelPath, 3, false, 100);
        model.initialize();
        
        // Generate some activity
        model.getWordVector("hello"); // Cache miss + hit
        model.getWordVector("hello"); // Cache hit
        model.getWordVector("world"); // Cache miss
        model.getWordVector("nonexistent"); // OOV
        
        var stats = model.getStats();
        assertThat(stats.vocabularySize()).isEqualTo(5);
        assertThat(stats.dimensions()).isEqualTo(3);
        assertThat(stats.loaded()).isTrue();
        assertThat(stats.cacheHits()).isEqualTo(1);
        assertThat(stats.cacheMisses()).isEqualTo(3);
        assertThat(stats.oovCount()).isEqualTo(1);
        assertThat(stats.cacheSize()).isEqualTo(2);
        assertThat(stats.cacheHitRate()).isEqualTo(1.0/4.0);
        assertThat(stats.oovRate()).isEqualTo(0.2);
    }

    @Test
    @DisplayName("Should clear cache properly")
    void shouldClearCache() throws IOException {
        model = new FastTextModel(testModelPath, 3, false, 100);
        model.initialize();
        
        // Fill cache
        model.getWordVector("hello");
        model.getWordVector("world");
        assertThat(model.getCacheSize()).isEqualTo(2);
        
        // Clear cache
        model.clearCache();
        assertThat(model.getCacheSize()).isZero();
        
        var stats = model.getStats();
        assertThat(stats.cacheHits()).isZero();
        assertThat(stats.cacheMisses()).isZero();
    }

    @Test
    @DisplayName("Should handle malformed model file")
    void shouldHandleMalformedModelFile() throws IOException {
        var malformedPath = tempDir.resolve("malformed.vec");
        Files.write(malformedPath, "invalid header\n".getBytes(StandardCharsets.UTF_8));
        
        model = new FastTextModel(malformedPath, 3, false, 100);
        
        assertThatThrownBy(() -> model.initialize())
            .isInstanceOfAny(IOException.class, NumberFormatException.class);
    }

    @Test
    @DisplayName("Should handle empty model file")
    void shouldHandleEmptyModelFile() throws IOException {
        var emptyPath = tempDir.resolve("empty.vec");
        Files.write(emptyPath, "".getBytes(StandardCharsets.UTF_8));
        
        model = new FastTextModel(emptyPath, 3, false, 100);
        
        assertThatThrownBy(() -> model.initialize())
            .isInstanceOf(IOException.class)
            .hasMessageContaining("Empty FastText model file");
    }

    @Test
    @DisplayName("Should provide meaningful toString")
    void shouldProvideMeaningfulToString() throws IOException {
        model = new FastTextModel(testModelPath, 3, false, 100);
        model.initialize();
        
        var str = model.toString();
        assertThat(str).contains("test.vec");
        assertThat(str).contains("dims=3");
        assertThat(str).contains("vocab=5");
    }

    private double calculateNorm(double[] values) {
        var sum = 0.0;
        for (var value : values) {
            sum += value * value;
        }
        return Math.sqrt(sum);
    }
}