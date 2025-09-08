package com.hellblazer.art.nlp.core;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import java.util.List;
import java.util.Map;
import static org.assertj.core.api.Assertions.*;

/**
 * Comprehensive test suite for ProcessingResult class.
 * Tests the CRITICAL requirement: Map<String, Integer> channelCategories
 */
@DisplayName("ProcessingResult Tests")
class ProcessingResultTest {

    @Test
    @DisplayName("Should create empty result")
    void shouldCreateEmptyResult() {
        var result = ProcessingResult.empty();
        
        assertThat(result.getAllCategories()).isEmpty();
        assertThat(result.getEntities()).isEmpty();
        assertThat(result.getSentiment()).isNull();
        assertThat(result.getProcessingTimeMs()).isZero();
        assertThat(result.getTokenCount()).isZero();
        assertThat(result.getAllMetadata()).isEmpty();
        assertThat(result.getChannelNames()).isEmpty();
        assertThat(result.isDegraded()).isFalse();
    }

    @Test
    @DisplayName("Should create result with channel categories")
    void shouldCreateResultWithChannelCategories() {
        var result = ProcessingResult.builder()
            .withChannelCategory("semantic", 5)
            .withChannelCategory("syntactic", 12)
            .withChannelCategory("context", 3)
            .build();
        
        // Test CRITICAL requirement: Map<String, Integer> channelCategories
        assertThat(result.getCategoryForChannel("semantic")).isEqualTo(5);
        assertThat(result.getCategoryForChannel("syntactic")).isEqualTo(12);
        assertThat(result.getCategoryForChannel("context")).isEqualTo(3);
        assertThat(result.getCategoryForChannel("nonexistent")).isNull();
        
        var allCategories = result.getAllCategories();
        assertThat(allCategories).hasSize(3);
        assertThat(allCategories).containsEntry("semantic", 5);
        assertThat(allCategories).containsEntry("syntactic", 12);
        assertThat(allCategories).containsEntry("context", 3);
        
        assertThat(result.getChannelNames()).containsExactlyInAnyOrder("semantic", "syntactic", "context");
        assertThat(result.hasChannel("semantic")).isTrue();
        assertThat(result.hasChannel("missing")).isFalse();
    }

    @Test
    @DisplayName("Should create result with multiple channel categories at once")
    void shouldCreateResultWithMultipleChannelCategories() {
        var categories = Map.of(
            "semantic", 10,
            "entity", 7,
            "sentiment", 2
        );
        
        var result = ProcessingResult.builder()
            .withChannelCategories(categories)
            .build();
        
        assertThat(result.getAllCategories()).isEqualTo(categories);
        assertThat(result.getCategoryForChannel("semantic")).isEqualTo(10);
        assertThat(result.getCategoryForChannel("entity")).isEqualTo(7);
        assertThat(result.getCategoryForChannel("sentiment")).isEqualTo(2);
    }

    @Test
    @DisplayName("Should create result with entities")
    void shouldCreateResultWithEntities() {
        var entity1 = new Entity("Apple Inc", "ORG", 0, 1, 0.95);
        var entity2 = new Entity("New York", "LOC", 5, 6, 0.87);
        
        var result = ProcessingResult.builder()
            .withEntity(entity1)
            .withEntity(entity2)
            .build();
        
        var entities = result.getEntities();
        assertThat(entities).hasSize(2);
        assertThat(entities).contains(entity1, entity2);
        
        // Test filtering by type
        var orgs = result.getEntitiesByType("ORG");
        var locs = result.getEntitiesByType("LOC");
        var missing = result.getEntitiesByType("PERSON");
        
        assertThat(orgs).hasSize(1).contains(entity1);
        assertThat(locs).hasSize(1).contains(entity2);
        assertThat(missing).isEmpty();
    }

    @Test
    @DisplayName("Should create result with entity list")
    void shouldCreateResultWithEntityList() {
        var entities = List.of(
            new Entity("John", "PERSON", 0),
            new Entity("Microsoft", "ORG", 2)
        );
        
        var result = ProcessingResult.builder()
            .withEntities(entities)
            .build();
        
        assertThat(result.getEntities()).hasSize(2);
        assertThat(result.getEntitiesByType("PERSON")).hasSize(1);
        assertThat(result.getEntitiesByType("ORG")).hasSize(1);
    }

    @Test
    @DisplayName("Should create result with sentiment")
    void shouldCreateResultWithSentiment() {
        var sentiment = SentimentScore.positive(0.85);
        
        var result = ProcessingResult.builder()
            .withSentiment(sentiment)
            .build();
        
        assertThat(result.getSentiment()).isEqualTo(sentiment);
    }

    @Test
    @DisplayName("Should create result with processing metrics")
    void shouldCreateResultWithProcessingMetrics() {
        var result = ProcessingResult.builder()
            .withProcessingTime(150)
            .withTokenCount(42)
            .build();
        
        assertThat(result.getProcessingTimeMs()).isEqualTo(150);
        assertThat(result.getTokenCount()).isEqualTo(42);
    }

    @Test
    @DisplayName("Should create result with metadata")
    void shouldCreateResultWithMetadata() {
        var result = ProcessingResult.builder()
            .withMetadata("version", "1.0")
            .withMetadata("model", "fasttext")
            .withMetadata("accuracy", 0.92)
            .build();
        
        assertThat(result.getMetadata("version")).isEqualTo("1.0");
        assertThat(result.getMetadata("model")).isEqualTo("fasttext");
        assertThat(result.getMetadata("accuracy")).isEqualTo(0.92);
        assertThat(result.getMetadata("missing")).isNull();
        
        var allMetadata = result.getAllMetadata();
        assertThat(allMetadata).hasSize(3);
        assertThat(allMetadata).containsEntry("version", "1.0");
        assertThat(allMetadata).containsEntry("model", "fasttext");
        assertThat(allMetadata).containsEntry("accuracy", 0.92);
    }

    @Test
    @DisplayName("Should create result with metadata map")
    void shouldCreateResultWithMetadataMap() {
        Map<String, Object> metadata = Map.of(
            "source", "keyboard",
            "confidence", 0.78,
            "channels_processed", 5
        );
        
        var result = ProcessingResult.builder()
            .withMetadata(metadata)
            .build();
        
        assertThat(result.getAllMetadata()).isEqualTo(metadata);
        assertThat(result.getMetadata("source")).isEqualTo("keyboard");
    }

    @Test
    @DisplayName("Should create comprehensive processing result")
    void shouldCreateComprehensiveProcessingResult() {
        var entities = List.of(
            new Entity("Apple", "ORG", 0),
            new Entity("California", "LOC", 2)
        );
        var sentiment = SentimentScore.negative(0.73);
        
        var result = ProcessingResult.builder()
            .withChannelCategory("semantic", 15)
            .withChannelCategory("syntactic", 8)
            .withChannelCategory("context", 22)
            .withChannelCategory("entity", 4)
            .withChannelCategory("sentiment", 1)
            .withEntities(entities)
            .withSentiment(sentiment)
            .withProcessingTime(89)
            .withTokenCount(156)
            .withMetadata("input_source", "pdf")
            .withMetadata("total_chars", 743)
            .build();
        
        // Validate all components
        assertThat(result.getAllCategories()).hasSize(5);
        assertThat(result.getCategoryForChannel("semantic")).isEqualTo(15);
        assertThat(result.getCategoryForChannel("sentiment")).isEqualTo(1);
        
        assertThat(result.getEntities()).hasSize(2);
        assertThat(result.getSentiment()).isEqualTo(sentiment);
        
        assertThat(result.getProcessingTimeMs()).isEqualTo(89);
        assertThat(result.getTokenCount()).isEqualTo(156);
        
        assertThat(result.getMetadata("input_source")).isEqualTo("pdf");
        assertThat(result.getMetadata("total_chars")).isEqualTo(743);
        
        assertThat(result.isDegraded()).isFalse();
    }

    @Test
    @DisplayName("Should create degraded result with error info")
    void shouldCreateDegradedResultWithErrorInfo() {
        var partialResults = Map.of("semantic", 5, "syntactic", 12);
        
        var result = ProcessingResult.degraded("entity", "Model failed to load", partialResults);
        
        assertThat(result.isDegraded()).isTrue();
        assertThat(result.getFailedChannel()).isEqualTo("entity");
        assertThat(result.getErrorMessage()).isEqualTo("Model failed to load");
        assertThat(result.getAllCategories()).isEqualTo(partialResults);
        
        // Metadata should contain degraded info
        assertThat(result.getMetadata("degraded")).isEqualTo(true);
        assertThat(result.getMetadata("failed_channel")).isEqualTo("entity");
        assertThat(result.getMetadata("error")).isEqualTo("Model failed to load");
    }

    @Test
    @DisplayName("Should create degraded result with null partial results")
    void shouldCreateDegradedResultWithNullPartialResults() {
        var result = ProcessingResult.degraded("context", "Connection timeout", null);
        
        assertThat(result.isDegraded()).isTrue();
        assertThat(result.getFailedChannel()).isEqualTo("context");
        assertThat(result.getErrorMessage()).isEqualTo("Connection timeout");
        assertThat(result.getAllCategories()).isEmpty();
    }

    @Test
    @DisplayName("Should mark result as degraded using builder")
    void shouldMarkResultAsDegradedUsingBuilder() {
        var result = ProcessingResult.builder()
            .withChannelCategory("semantic", 10)
            .markDegraded("sentiment", "Lexicon not found")
            .withProcessingTime(250)
            .build();
        
        assertThat(result.isDegraded()).isTrue();
        assertThat(result.getFailedChannel()).isEqualTo("sentiment");
        assertThat(result.getErrorMessage()).isEqualTo("Lexicon not found");
        assertThat(result.getCategoryForChannel("semantic")).isEqualTo(10);
        assertThat(result.getProcessingTimeMs()).isEqualTo(250);
    }

    @Test
    @DisplayName("Should validate builder parameters")
    void shouldValidateBuilderParameters() {
        var builder = ProcessingResult.builder();
        
        // Null channel name
        assertThatThrownBy(() -> builder.withChannelCategory(null, 5))
            .isInstanceOf(NullPointerException.class)
            .hasMessage("channel cannot be null");
        
        // Null entity
        assertThatThrownBy(() -> builder.withEntity(null))
            .isInstanceOf(NullPointerException.class)
            .hasMessage("entity cannot be null");
        
        // Null metadata key
        assertThatThrownBy(() -> builder.withMetadata(null, "value"))
            .isInstanceOf(NullPointerException.class)
            .hasMessage("metadata key cannot be null");
        
        // Negative processing time
        assertThatThrownBy(() -> builder.withProcessingTime(-1))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessage("Processing time cannot be negative: -1");
        
        // Negative token count
        assertThatThrownBy(() -> builder.withTokenCount(-5))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessage("Token count cannot be negative: -5");
    }

    @Test
    @DisplayName("Should handle null collections gracefully")
    void shouldHandleNullCollectionsGracefully() {
        var result = ProcessingResult.builder()
            .withChannelCategories(null)
            .withEntities(null)
            .withMetadata((Map<String, Object>) null)
            .build();
        
        assertThat(result.getAllCategories()).isEmpty();
        assertThat(result.getEntities()).isEmpty();
        assertThat(result.getAllMetadata()).isEmpty();
    }

    @Test
    @DisplayName("Should maintain immutability")
    void shouldMaintainImmutability() {
        var builder = ProcessingResult.builder()
            .withChannelCategory("semantic", 5)
            .withEntity(new Entity("Test", "MISC", 0));
        
        var result1 = builder.build();
        var result2 = builder.withChannelCategory("syntactic", 10).build();
        
        // First result should not be affected by changes to builder
        assertThat(result1.getAllCategories()).hasSize(1);
        assertThat(result1.hasChannel("syntactic")).isFalse();
        
        assertThat(result2.getAllCategories()).hasSize(2);
        assertThat(result2.hasChannel("syntactic")).isTrue();
        
        // Direct modifications to returned collections should not affect result
        var categories = result1.getAllCategories();
        assertThatThrownBy(() -> categories.put("new", 99))
            .isInstanceOf(UnsupportedOperationException.class);
        
        var entities = result1.getEntities();
        assertThatThrownBy(() -> entities.add(new Entity("hack", "MISC", 0)))
            .isInstanceOf(UnsupportedOperationException.class);
    }

    @Test
    @DisplayName("Should implement equals correctly")
    void shouldImplementEqualsCorrectly() {
        var entity = new Entity("Test", "MISC", 0);
        var sentiment = SentimentScore.neutral();
        
        var result1 = ProcessingResult.builder()
            .withChannelCategory("semantic", 5)
            .withEntity(entity)
            .withSentiment(sentiment)
            .withProcessingTime(100)
            .withTokenCount(20)
            .withMetadata("test", "value")
            .build();
        
        var result2 = ProcessingResult.builder()
            .withChannelCategory("semantic", 5)
            .withEntity(entity)
            .withSentiment(sentiment)
            .withProcessingTime(100)
            .withTokenCount(20)
            .withMetadata("test", "value")
            .build();
        
        var result3 = ProcessingResult.builder()
            .withChannelCategory("semantic", 10) // Different category
            .withEntity(entity)
            .withSentiment(sentiment)
            .withProcessingTime(100)
            .withTokenCount(20)
            .withMetadata("test", "value")
            .build();
        
        // Same reference
        assertThat(result1).isEqualTo(result1);
        
        // Same values
        assertThat(result1).isEqualTo(result2);
        assertThat(result2).isEqualTo(result1);
        
        // Different values
        assertThat(result1).isNotEqualTo(result3);
        
        // Null and different class
        assertThat(result1).isNotEqualTo(null);
        assertThat(result1).isNotEqualTo("not a result");
    }

    @Test
    @DisplayName("Should implement hashCode correctly")
    void shouldImplementHashCodeCorrectly() {
        var entity = new Entity("Test", "MISC", 0);
        
        var result1 = ProcessingResult.builder()
            .withChannelCategory("semantic", 5)
            .withEntity(entity)
            .withProcessingTime(100)
            .build();
        
        var result2 = ProcessingResult.builder()
            .withChannelCategory("semantic", 5)
            .withEntity(entity)
            .withProcessingTime(100)
            .build();
        
        var result3 = ProcessingResult.builder()
            .withChannelCategory("semantic", 10)
            .withEntity(entity)
            .withProcessingTime(100)
            .build();
        
        // Equal objects should have equal hash codes
        assertThat(result1.hashCode()).isEqualTo(result2.hashCode());
        
        // Different objects should preferably have different hash codes
        assertThat(result1.hashCode()).isNotEqualTo(result3.hashCode());
    }

    @Test
    @DisplayName("Should provide meaningful toString")
    void shouldProvideMeaningfulToString() {
        var result = ProcessingResult.builder()
            .withChannelCategory("semantic", 15)
            .withChannelCategory("entity", 7)
            .withEntity(new Entity("Apple", "ORG", 0))
            .withEntity(new Entity("John", "PERSON", 2))
            .withSentiment(SentimentScore.positive(0.8))
            .withProcessingTime(125)
            .withTokenCount(47)
            .build();
        
        var str = result.toString();
        
        assertThat(str).contains("channels=2");
        assertThat(str).contains("entities=2");
        assertThat(str).contains("POSITIVE");
        assertThat(str).contains("125ms");
        assertThat(str).contains("tokens=47");
        assertThat(str).startsWith("ProcessingResult{");
    }

    @Test
    @DisplayName("Should handle builder reuse correctly")
    void shouldHandleBuilderReuseCorrectly() {
        var builder = ProcessingResult.builder()
            .withChannelCategory("base", 1);
        
        var result1 = builder
            .withChannelCategory("semantic", 5)
            .build();
        
        var result2 = builder
            .withChannelCategory("syntactic", 10)
            .build();
        
        // Both results should have all categories added to the builder
        assertThat(result1.getAllCategories()).hasSize(2);
        assertThat(result1.hasChannel("base")).isTrue();
        assertThat(result1.hasChannel("semantic")).isTrue();
        
        assertThat(result2.getAllCategories()).hasSize(3);
        assertThat(result2.hasChannel("base")).isTrue();
        assertThat(result2.hasChannel("semantic")).isTrue();
        assertThat(result2.hasChannel("syntactic")).isTrue();
    }

    @Test
    @DisplayName("Should validate channel categories data type")
    void shouldValidateChannelCategoriesDataType() {
        var result = ProcessingResult.builder()
            .withChannelCategory("semantic", 42)
            .withChannelCategory("entity", 0)
            .withChannelCategory("sentiment", 999)
            .build();
        
        // CRITICAL: Verify Map<String, Integer> type requirement
        var categories = result.getAllCategories();
        assertThat(categories).isInstanceOf(Map.class);
        
        for (var entry : categories.entrySet()) {
            assertThat(entry.getKey()).isInstanceOf(String.class);
            assertThat(entry.getValue()).isInstanceOf(Integer.class);
        }
        
        // Test specific values
        assertThat(result.getCategoryForChannel("semantic")).isEqualTo(42);
        assertThat(result.getCategoryForChannel("entity")).isEqualTo(0);
        assertThat(result.getCategoryForChannel("sentiment")).isEqualTo(999);
    }
}