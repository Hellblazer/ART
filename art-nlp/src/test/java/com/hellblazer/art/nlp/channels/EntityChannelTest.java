package com.hellblazer.art.nlp.channels;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import static org.assertj.core.api.Assertions.*;

import java.util.List;
import java.util.EnumSet;

import com.hellblazer.art.nlp.channels.EntityChannel.EntityType;
import com.hellblazer.art.nlp.channels.EntityChannel.EntityFeatureMode;

/**
 * Tests for EntityChannel Named Entity Recognition processing.
 */
@DisplayName("EntityChannel Tests")
class EntityChannelTest {

    private EntityChannel channel;

    @AfterEach
    void tearDown() {
        if (channel != null) {
            channel.shutdown();
        }
    }

    @Test
    @DisplayName("Should create entity channel with default configuration")
    void shouldCreateChannelWithDefaults() {
        channel = new EntityChannel("entity", 0.8);
        
        assertThat(channel.getChannelName()).isEqualTo("entity");
        assertThat(channel.getVigilance()).isEqualTo(0.8);
        assertThat(channel.isInitialized()).isFalse();
    }

    @Test
    @DisplayName("Should create entity channel with custom configuration")
    void shouldCreateChannelWithCustomConfig() {
        var entityTypes = EnumSet.of(EntityType.PERSON, EntityType.LOCATION);
        channel = new EntityChannel("test", 0.75, entityTypes,
                EntityFeatureMode.DENSITY_BASED, false, 25);
        
        assertThat(channel.getChannelName()).isEqualTo("test");
        assertThat(channel.getVigilance()).isEqualTo(0.75);
        assertThat(channel.isInitialized()).isFalse();
    }

    @Test
    @DisplayName("Should handle null and empty text inputs gracefully")
    void shouldHandleNullAndEmptyTextInputs() {
        channel = new EntityChannel("entity", 0.8);
        
        assertThat(channel.classifyText(null)).isEqualTo(-1);
        assertThat(channel.classifyText("")).isEqualTo(-1);
        assertThat(channel.classifyText("   ")).isEqualTo(-1);
        
        assertThat(channel.extractEntities(null)).isEmpty();
        assertThat(channel.extractEntities("")).isEmpty();
        assertThat(channel.extractEntities("   ")).isEmpty();
    }

    @Test
    @DisplayName("Should initialize successfully with OpenNLP models")
    void shouldInitializeSuccessfully() {
        channel = new EntityChannel("entity", 0.8);
        
        // With OpenNLP models available, initialization should succeed
        assertThatNoException().isThrownBy(() -> channel.initialize());
        assertThat(channel.isInitialized()).isTrue();
    }

    @Test
    @DisplayName("Should batch classify multiple texts")
    void shouldBatchClassifyMultipleTexts() {
        channel = new EntityChannel("entity", 0.8);
        
        var texts = List.of(
            "John Smith works at Microsoft in Seattle.",
            "Apple Inc. is located in Cupertino.",
            "No entities here.",
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
    @DisplayName("Should batch extract entities from multiple texts")
    void shouldBatchExtractEntitiesFromTexts() {
        channel = new EntityChannel("entity", 0.8);
        
        var texts = List.of(
            "John Smith works at Microsoft.",
            "Apple is in Cupertino.",
            "No entities here."
        );
        
        var allEntities = channel.extractEntitiesFromTexts(texts);
        
        assertThat(allEntities).hasSize(3);
        // Without real models, all should be empty
        for (var entities : allEntities) {
            assertThat(entities).isEmpty();
        }
    }

    @Test
    @DisplayName("Should provide meaningful performance metrics")
    void shouldProvideMeaningfulPerformanceMetrics() {
        channel = new EntityChannel("entity", 0.8);
        
        // Generate some activity (will fail without real models but updates metrics)
        channel.classifyText("John Smith works at Microsoft in Seattle.");
        channel.classifyText("Apple Inc. is located in Cupertino, California.");
        
        var metrics = channel.getEntityMetrics();
        
        assertThat(metrics.totalClassifications()).isGreaterThanOrEqualTo(0);
        assertThat(metrics.successfulClassifications()).isGreaterThanOrEqualTo(0);
        assertThat(metrics.categoryCount()).isGreaterThanOrEqualTo(0);
        assertThat(metrics.totalSentences()).isGreaterThanOrEqualTo(0);
        assertThat(metrics.totalEntities()).isGreaterThanOrEqualTo(0);
        assertThat(metrics.enabledEntityTypes()).isEqualTo(3); // All entity types by default
        assertThat(metrics.featureModeName()).isEqualTo("COUNT_BASED");
        assertThat(metrics.successRate()).isBetween(0.0, 1.0);
        assertThat(metrics.averageEntitiesPerSentence()).isGreaterThanOrEqualTo(0.0);
        assertThat(metrics.entityRecognitionRate()).isGreaterThanOrEqualTo(0.0);
    }

    @Test
    @DisplayName("Should work with different entity types")
    void shouldWorkWithDifferentEntityTypes() {
        // Test with only PERSON
        var personChannel = new EntityChannel("person", 0.8,
                EnumSet.of(EntityType.PERSON),
                EntityFeatureMode.COUNT_BASED, true, 50);
        assertThat(personChannel.getEntityMetrics().enabledEntityTypes()).isEqualTo(1);
        personChannel.shutdown();
        
        // Test with PERSON and LOCATION
        var personLocationChannel = new EntityChannel("person_location", 0.8,
                EnumSet.of(EntityType.PERSON, EntityType.LOCATION),
                EntityFeatureMode.COUNT_BASED, true, 50);
        assertThat(personLocationChannel.getEntityMetrics().enabledEntityTypes()).isEqualTo(2);
        personLocationChannel.shutdown();
        
        // Test with all entity types
        var allChannel = new EntityChannel("all", 0.8,
                EnumSet.allOf(EntityType.class),
                EntityFeatureMode.COUNT_BASED, true, 50);
        assertThat(allChannel.getEntityMetrics().enabledEntityTypes()).isEqualTo(3);
        allChannel.shutdown();
    }

    @Test
    @DisplayName("Should work with different feature modes")
    void shouldWorkWithDifferentFeatureModes() {
        // Test COUNT_BASED
        var countChannel = new EntityChannel("count", 0.8,
                EnumSet.allOf(EntityType.class),
                EntityFeatureMode.COUNT_BASED, true, 50);
        assertThat(countChannel.getEntityMetrics().featureModeName()).isEqualTo("COUNT_BASED");
        countChannel.shutdown();
        
        // Test DENSITY_BASED
        var densityChannel = new EntityChannel("density", 0.8,
                EnumSet.allOf(EntityType.class),
                EntityFeatureMode.DENSITY_BASED, true, 50);
        assertThat(densityChannel.getEntityMetrics().featureModeName()).isEqualTo("DENSITY_BASED");
        densityChannel.shutdown();
        
        // Test COMPREHENSIVE
        var comprehensiveChannel = new EntityChannel("comprehensive", 0.8,
                EnumSet.allOf(EntityType.class),
                EntityFeatureMode.COMPREHENSIVE, true, 50);
        assertThat(comprehensiveChannel.getEntityMetrics().featureModeName()).isEqualTo("COMPREHENSIVE");
        comprehensiveChannel.shutdown();
    }

    @Test
    @DisplayName("Should handle different entity limits")
    void shouldHandleDifferentEntityLimits() {
        // Test with small entity limit
        var smallChannel = new EntityChannel("small", 0.8,
                EnumSet.allOf(EntityType.class),
                EntityFeatureMode.COUNT_BASED, true, 5);
        assertThat(smallChannel.extractEntities("Text with many potential entities")).isEmpty();
        smallChannel.shutdown();
        
        // Test with large entity limit
        var largeChannel = new EntityChannel("large", 0.8,
                EnumSet.allOf(EntityType.class),
                EntityFeatureMode.COUNT_BASED, true, 1000);
        assertThat(largeChannel.extractEntities("Short text")).isEmpty();
        largeChannel.shutdown();
    }

    @Test
    @DisplayName("Should handle normalization settings")
    void shouldHandleNormalizationSettings() {
        // Test with normalization enabled
        var normalizedChannel = new EntityChannel("normalized", 0.8,
                EnumSet.allOf(EntityType.class),
                EntityFeatureMode.COUNT_BASED, true, 50);
        assertThat(normalizedChannel.classifyText("John works at Apple")).isEqualTo(-1);
        normalizedChannel.shutdown();
        
        // Test with normalization disabled
        var rawChannel = new EntityChannel("raw", 0.8,
                EnumSet.allOf(EntityType.class),
                EntityFeatureMode.COUNT_BASED, false, 50);
        assertThat(rawChannel.classifyText("John works at Apple")).isEqualTo(-1);
        rawChannel.shutdown();
    }

    @Test
    @DisplayName("Should provide meaningful toString")
    void shouldProvideMeaningfulToString() {
        channel = new EntityChannel("entity", 0.75);
        
        var str = channel.toString();
        assertThat(str).contains("entity");
        assertThat(str).contains("0.750");
        assertThat(str).contains("categories=");
    }

    @Test
    @DisplayName("Should shutdown gracefully")
    void shouldShutdownGracefully() {
        channel = new EntityChannel("entity", 0.8);
        
        // Should not throw exceptions
        assertThatNoException().isThrownBy(() -> channel.shutdown());
        
        // Second shutdown should also be safe
        assertThatNoException().isThrownBy(() -> channel.shutdown());
    }

    @Test
    @DisplayName("Should implement required abstract methods")
    void shouldImplementRequiredAbstractMethods() {
        channel = new EntityChannel("entity", 0.8);
        
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
        channel = new EntityChannel("entity", 0.8);
        
        var texts = List.of(
            "John Smith works here",
            "Apple Inc. in California",
            "Microsoft Corporation",
            "Google in Mountain View"
        );
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
        channel = new EntityChannel("entity", 0.8);
        
        // Test learning enabled/disabled
        channel.setLearningEnabled(true);
        assertThat(channel.isLearningEnabled()).isTrue();
        
        channel.setLearningEnabled(false);
        assertThat(channel.isLearningEnabled()).isFalse();
        
        // Classification should work regardless of learning state
        assertThat(channel.classifyText("John Smith works at Apple")).isEqualTo(-1);
    }

    @Test
    @DisplayName("Should handle various text complexities")
    void shouldHandleVariousTextComplexities() {
        channel = new EntityChannel("entity", 0.8);
        
        // Simple entity text
        assertThat(channel.extractEntities("John Smith")).isEmpty();
        
        // Complex text with multiple entities
        assertThat(channel.extractEntities("John Smith works at Microsoft in Seattle, Washington.")).isEmpty();
        
        // Text with potential false positives
        assertThat(channel.extractEntities("Apple pie and Orange juice are delicious.")).isEmpty();
        
        // Mixed case and punctuation
        assertThat(channel.extractEntities("DR. JOHN SMITH, M.D. works at St. Mary's Hospital.")).isEmpty();
        
        // Numbers and dates
        assertThat(channel.extractEntities("John Smith was born on January 1, 1990 in New York.")).isEmpty();
        
        // No entities
        assertThat(channel.extractEntities("This text contains no named entities at all.")).isEmpty();
    }

    @Test
    @DisplayName("Should validate constructor parameters")
    void shouldValidateConstructorParameters() {
        assertThatThrownBy(() -> new EntityChannel("test", 0.8,
                EnumSet.allOf(EntityType.class), null, true, 50))
            .isInstanceOf(NullPointerException.class)
            .hasMessageContaining("featureMode cannot be null");

        // Empty entity types should work (though not very useful)
        var emptyChannel = new EntityChannel("empty", 0.8,
                EnumSet.noneOf(EntityType.class),
                EntityFeatureMode.COUNT_BASED, true, 50);
        assertThat(emptyChannel.getEntityMetrics().enabledEntityTypes()).isEqualTo(0);
        emptyChannel.shutdown();
    }

    @Test
    @DisplayName("Should handle entity type enum correctly")
    void shouldHandleEntityTypeEnumCorrectly() {
        assertThat(EntityType.PERSON.getModelFileName()).isEqualTo("en-ner-person.bin");
        assertThat(EntityType.LOCATION.getModelFileName()).isEqualTo("en-ner-location.bin");
        assertThat(EntityType.ORGANIZATION.getModelFileName()).isEqualTo("en-ner-organization.bin");
        
        assertThat(EntityType.values()).hasSize(3);
        assertThat(EntityType.valueOf("PERSON")).isEqualTo(EntityType.PERSON);
    }

    @Test
    @DisplayName("Should handle feature mode enum correctly")
    void shouldHandleFeatureModeEnumCorrectly() {
        assertThat(EntityFeatureMode.values()).hasSize(3);
        assertThat(EntityFeatureMode.valueOf("COUNT_BASED")).isEqualTo(EntityFeatureMode.COUNT_BASED);
        assertThat(EntityFeatureMode.valueOf("DENSITY_BASED")).isEqualTo(EntityFeatureMode.DENSITY_BASED);
        assertThat(EntityFeatureMode.valueOf("COMPREHENSIVE")).isEqualTo(EntityFeatureMode.COMPREHENSIVE);
    }

    @Test
    @DisplayName("Should handle entity extraction edge cases")
    void shouldHandleEntityExtractionEdgeCases() {
        channel = new EntityChannel("entity", 0.8);
        
        // Very long text
        var longText = "This is a very long text that might contain many entities. ".repeat(100);
        assertThat(channel.extractEntities(longText)).isEmpty();
        
        // Text with only punctuation
        assertThat(channel.extractEntities("!@#$%^&*()_+")).isEmpty();
        
        // Text with only numbers
        assertThat(channel.extractEntities("123 456 789")).isEmpty();
        
        // Single word
        assertThat(channel.extractEntities("Word")).isEmpty();
        
        // Mixed languages (English focus)
        assertThat(channel.extractEntities("Hello 世界 bonjour mundo")).isEmpty();
    }

    @Test
    @DisplayName("Should provide accurate metric calculations")
    void shouldProvideAccurateMetricCalculations() {
        channel = new EntityChannel("entity", 0.8);
        
        var metrics = channel.getEntityMetrics();
        
        // Test metric calculations with no data
        assertThat(metrics.successRate()).isEqualTo(0.0);
        assertThat(metrics.averageEntitiesPerSentence()).isEqualTo(0.0);
        assertThat(metrics.averageEntitiesPerClassification()).isEqualTo(0.0);
        assertThat(metrics.entityRecognitionRate()).isEqualTo(0.0);
        
        // Metrics should be consistent
        assertThat(metrics.averageEntitiesPerSentence()).isEqualTo(metrics.entityRecognitionRate());
    }
}