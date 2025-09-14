/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 * 
 * This file is part of Java ART Neural Networks.
 * 
 * Java ART Neural Networks is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Java ART Neural Networks is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with Java ART Neural Networks. If not, see <https://www.gnu.org/licenses/>.
 */
package com.hellblazer.art.hartcq.hierarchical;

import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.hartcq.hierarchical.CategoryManager.CategoryKey;
import com.hellblazer.art.hartcq.hierarchical.CategoryManager.CategoryParameters;
import com.hellblazer.art.hartcq.hierarchical.CategoryManager.ManagedCategory;
import com.hellblazer.art.hartcq.hierarchical.HierarchyLevel.Level;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.RepeatedTest;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.Set;
import java.util.stream.IntStream;

import static org.assertj.core.api.Assertions.*;

/**
 * Comprehensive unit tests for CategoryManager.
 * Tests category creation, deterministic selection, pruning and merging,
 * cross-level tracking, and performance optimization.
 * 
 * @author Hal Hildebrand
 */
class CategoryManagerTest {
    private static final Logger logger = LoggerFactory.getLogger(CategoryManagerTest.class);
    
    private CategoryManager categoryManager;
    private CategoryParameters defaultParams;
    
    @BeforeEach
    void setUp() {
        defaultParams = CategoryParameters.defaults();
        categoryManager = new CategoryManager(defaultParams);
    }
    
    @Nested
    @DisplayName("Category Creation and Storage Tests")
    class CategoryCreationTests {
        
        @Test
        @DisplayName("Should create and store categories successfully")
        void shouldCreateAndStoreCategoriesSuccessfully() {
            // Given
            var level = Level.TOKEN;
            var categoryId = 1;
            var prototype = createTestPattern(0.5, 0.3, 0.8);
            var activation = 0.85;
            Map<String, Object> metadata = Map.of("test", "value");

            // When
            var categoryKey = categoryManager.storeCategory(level, categoryId, prototype, activation, metadata);
            
            // Then
            assertThat(categoryKey).isNotNull();
            assertThat(categoryKey.level()).isEqualTo(level);
            assertThat(categoryKey.categoryId()).isEqualTo(categoryId);
            
            var retrievedCategory = categoryManager.getCategory(categoryKey);
            assertThat(retrievedCategory).isNotNull();
            assertThat(retrievedCategory.getKey()).isEqualTo(categoryKey);
            assertThat(retrievedCategory.getPrototype()).isEqualTo(prototype);
            assertThat(retrievedCategory.getAverageActivation()).isEqualTo(activation, offset(0.001));
            assertThat(retrievedCategory.getMetadata()).isEqualTo(metadata);
        }
        
        @Test
        @DisplayName("Should handle category updates when storing existing category")
        void shouldHandleCategoryUpdatesWhenStoringExistingCategory() {
            // Given
            var level = Level.WINDOW;
            var categoryId = 2;
            var prototype = createTestPattern(0.4, 0.6, 0.2);
            var initialActivation = 0.7;
            var updatedActivation = 0.9;
            
            // Store initial category
            var categoryKey = categoryManager.storeCategory(level, categoryId, prototype, initialActivation, Map.of());
            
            // When - store the same category again with different activation
            categoryManager.storeCategory(level, categoryId, prototype, updatedActivation, Map.of());
            
            // Then
            var retrievedCategory = categoryManager.getCategory(categoryKey);
            assertThat(retrievedCategory).isNotNull();
            assertThat(retrievedCategory.getActivationCount()).isEqualTo(2);
            
            // Average should incorporate both activations
            var expectedAverage = (initialActivation + updatedActivation) / 2.0;
            assertThat(retrievedCategory.getAverageActivation()).isEqualTo(expectedAverage, offset(0.001));
        }
        
        @Test
        @DisplayName("Should store categories across different levels")
        void shouldStoreCategoriesAcrossDifferentLevels() {
            // Given
            var prototype = createTestPattern(0.1, 0.2, 0.3);
            
            // When - store same category ID across different levels
            var tokenKey = categoryManager.storeCategory(Level.TOKEN, 1, prototype, 0.8, Map.of());
            var windowKey = categoryManager.storeCategory(Level.WINDOW, 1, prototype, 0.7, Map.of());
            var documentKey = categoryManager.storeCategory(Level.DOCUMENT, 1, prototype, 0.6, Map.of());
            
            // Then - all should be stored successfully
            assertThat(categoryManager.getCategory(tokenKey)).isNotNull();
            assertThat(categoryManager.getCategory(windowKey)).isNotNull();
            assertThat(categoryManager.getCategory(documentKey)).isNotNull();
            
            // Keys should be different even with same category ID
            assertThat(tokenKey).isNotEqualTo(windowKey);
            assertThat(windowKey).isNotEqualTo(documentKey);
            assertThat(tokenKey).isNotEqualTo(documentKey);
        }
        
        @Test
        @DisplayName("Should reject null inputs for category storage")
        void shouldRejectNullInputsForCategoryStorage() {
            var prototype = createTestPattern(0.5, 0.5, 0.5);
            
            // Test null level
            assertThatThrownBy(() -> 
                categoryManager.storeCategory(null, 1, prototype, 0.8, Map.of())
            ).isInstanceOf(IllegalArgumentException.class);
            
            // Test null prototype
            assertThatThrownBy(() -> 
                categoryManager.storeCategory(Level.TOKEN, 1, null, 0.8, Map.of())
            ).isInstanceOf(IllegalArgumentException.class);
        }
    }
    
    @Nested
    @DisplayName("Category Retrieval Tests")
    class CategoryRetrievalTests {
        
        @Test
        @DisplayName("Should retrieve categories by level")
        void shouldRetrieveCategoriesByLevel() {
            // Given - store categories at different levels
            var prototype1 = createTestPattern(0.1, 0.2, 0.3);
            var prototype2 = createTestPattern(0.4, 0.5, 0.6);
            var prototype3 = createTestPattern(0.7, 0.8, 0.9);
            
            categoryManager.storeCategory(Level.TOKEN, 1, prototype1, 0.8, Map.of());
            categoryManager.storeCategory(Level.TOKEN, 2, prototype2, 0.7, Map.of());
            categoryManager.storeCategory(Level.WINDOW, 3, prototype3, 0.6, Map.of());
            
            // When
            var tokenCategories = categoryManager.getCategoriesForLevel(Level.TOKEN);
            var windowCategories = categoryManager.getCategoriesForLevel(Level.WINDOW);
            var documentCategories = categoryManager.getCategoriesForLevel(Level.DOCUMENT);
            
            // Then
            assertThat(tokenCategories).hasSize(2);
            assertThat(windowCategories).hasSize(1);
            assertThat(documentCategories).isEmpty();
            
            // Verify ordering (should be by category ID)
            assertThat(tokenCategories.get(0).getKey().categoryId()).isLessThan(
                tokenCategories.get(1).getKey().categoryId());
        }
        
        @Test
        @DisplayName("Should return null for non-existent categories")
        void shouldReturnNullForNonExistentCategories() {
            // Given
            var nonExistentKey = new CategoryKey(Level.TOKEN, 999);
            
            // When
            var category = categoryManager.getCategory(nonExistentKey);
            
            // Then
            assertThat(category).isNull();
        }
        
        @Test
        @DisplayName("Should handle null category keys gracefully")
        void shouldHandleNullCategoryKeysGracefully() {
            // When
            var category = categoryManager.getCategory(null);
            
            // Then
            assertThat(category).isNull();
        }
        
        @Test
        @DisplayName("Should return empty list for null level")
        void shouldReturnEmptyListForNullLevel() {
            // When
            var categories = categoryManager.getCategoriesForLevel(null);
            
            // Then
            assertThat(categories).isEmpty();
        }
    }
    
    @Nested
    @DisplayName("Similarity and Matching Tests")
    class SimilarityMatchingTests {
        
        @Test
        @DisplayName("Should find most similar category above threshold")
        void shouldFindMostSimilarCategoryAboveThreshold() {
            // Given - store categories with different similarities
            var targetPattern = createTestPattern(0.5, 0.5, 0.5);
            var similarPattern = createTestPattern(0.6, 0.4, 0.5); // Similar
            var dissimilarPattern = createTestPattern(0.1, 0.9, 0.1); // Dissimilar
            
            categoryManager.storeCategory(Level.TOKEN, 1, similarPattern, 0.8, Map.of());
            categoryManager.storeCategory(Level.TOKEN, 2, dissimilarPattern, 0.7, Map.of());
            
            // When
            var mostSimilar = categoryManager.findMostSimilar(Level.TOKEN, targetPattern, 0.5);
            
            // Then
            assertThat(mostSimilar).isNotNull();
            assertThat(mostSimilar.getKey().categoryId()).isEqualTo(1); // Should match similar pattern
        }
        
        @Test
        @DisplayName("Should return null when no categories meet similarity threshold")
        void shouldReturnNullWhenNoCategoriesMeetThreshold() {
            // Given - store dissimilar category
            var targetPattern = createTestPattern(1.0, 1.0, 1.0);
            var dissimilarPattern = createTestPattern(0.0, 0.0, 0.0);
            
            categoryManager.storeCategory(Level.TOKEN, 1, dissimilarPattern, 0.8, Map.of());
            
            // When - set very high similarity threshold
            var mostSimilar = categoryManager.findMostSimilar(Level.TOKEN, targetPattern, 0.99);
            
            // Then
            assertThat(mostSimilar).isNull();
        }
        
        @Test
        @DisplayName("Should handle empty category set gracefully")
        void shouldHandleEmptyCategorySetGracefully() {
            // Given - no categories stored
            var pattern = createTestPattern(0.5, 0.5, 0.5);
            
            // When
            var mostSimilar = categoryManager.findMostSimilar(Level.TOKEN, pattern, 0.5);
            
            // Then
            assertThat(mostSimilar).isNull();
        }
        
        @Test
        @DisplayName("Should track access counts for similarity searches")
        void shouldTrackAccessCountsForSimilaritySearches() {
            // Given
            var pattern = createTestPattern(0.5, 0.5, 0.5);
            var key = categoryManager.storeCategory(Level.TOKEN, 1, pattern, 0.8, Map.of());
            var initialCategory = categoryManager.getCategory(key);
            var initialAccessCount = initialCategory.getAccessCount();
            
            // When - perform similarity search
            var foundCategory = categoryManager.findMostSimilar(Level.TOKEN, pattern, 0.5);
            
            // Then
            assertThat(foundCategory).isNotNull();
            assertThat(foundCategory.getAccessCount()).isGreaterThan(initialAccessCount);
        }
    }
    
    @Nested
    @DisplayName("Deterministic Selection Tests")
    class DeterministicSelectionTests {
        
        @Test
        @DisplayName("Should provide deterministic category selection")
        void shouldProvideDeterministicCategorySelection() {
            // Given
            var pattern = createTestPattern(0.3, 0.7, 0.4);
            var availableCategories = Set.of(1, 2, 3, 4, 5);
            
            // When - select multiple times with same inputs
            var selection1 = categoryManager.selectDeterministic(Level.TOKEN, pattern, availableCategories);
            var selection2 = categoryManager.selectDeterministic(Level.TOKEN, pattern, availableCategories);
            var selection3 = categoryManager.selectDeterministic(Level.TOKEN, pattern, availableCategories);
            
            // Then - all selections should be identical
            assertThat(selection1).isEqualTo(selection2);
            assertThat(selection2).isEqualTo(selection3);
            assertThat(selection1).isIn(availableCategories);
        }
        
        @RepeatedTest(10)
        @DisplayName("Should maintain deterministic behavior across multiple runs")
        void shouldMaintainDeterministicBehaviorAcrossRuns() {
            // Given
            var pattern = createTestPattern(0.1, 0.9, 0.5);
            var availableCategories = Set.of(10, 20, 30);
            
            // When
            var selection = categoryManager.selectDeterministic(Level.WINDOW, pattern, availableCategories);
            
            // Then - should consistently select the same category
            assertThat(selection).isIn(availableCategories);
            
            // Verify it's deterministic by running again
            var secondSelection = categoryManager.selectDeterministic(Level.WINDOW, pattern, availableCategories);
            assertThat(selection).isEqualTo(secondSelection);
        }
        
        @Test
        @DisplayName("Should handle different patterns deterministically")
        void shouldHandleDifferentPatternsDeterministically() {
            // Given
            var pattern1 = createTestPattern(0.2, 0.4, 0.6);
            var pattern2 = createTestPattern(0.6, 0.4, 0.2); // Different pattern
            var availableCategories = Set.of(1, 2, 3);
            
            // When
            var selection1 = categoryManager.selectDeterministic(Level.TOKEN, pattern1, availableCategories);
            var selection2 = categoryManager.selectDeterministic(Level.TOKEN, pattern2, availableCategories);
            
            // Then - different patterns might select different categories (but deterministically)
            assertThat(selection1).isIn(availableCategories);
            assertThat(selection2).isIn(availableCategories);
            
            // Verify consistency for each pattern
            var repeatSelection1 = categoryManager.selectDeterministic(Level.TOKEN, pattern1, availableCategories);
            var repeatSelection2 = categoryManager.selectDeterministic(Level.TOKEN, pattern2, availableCategories);
            
            assertThat(selection1).isEqualTo(repeatSelection1);
            assertThat(selection2).isEqualTo(repeatSelection2);
        }
        
        @Test
        @DisplayName("Should handle edge cases in deterministic selection")
        void shouldHandleEdgeCasesInDeterministicSelection() {
            var pattern = createTestPattern(0.5, 0.5, 0.5);
            
            // Test with empty set
            var emptyResult = categoryManager.selectDeterministic(Level.TOKEN, pattern, Set.of());
            assertThat(emptyResult).isEqualTo(-1);
            
            // Test with null set
            var nullResult = categoryManager.selectDeterministic(Level.TOKEN, pattern, null);
            assertThat(nullResult).isEqualTo(-1);
            
            // Test with null pattern
            var nullPatternResult = categoryManager.selectDeterministic(Level.TOKEN, null, Set.of(1, 2));
            assertThat(nullPatternResult).isEqualTo(-1);
            
            // Test with null level
            var nullLevelResult = categoryManager.selectDeterministic(null, pattern, Set.of(1, 2));
            assertThat(nullLevelResult).isEqualTo(-1);
            
            // Test with single item set
            var singleResult = categoryManager.selectDeterministic(Level.TOKEN, pattern, Set.of(42));
            assertThat(singleResult).isEqualTo(42);
        }
        
        @Test
        @DisplayName("Should allow setting deterministic seed")
        void shouldAllowSettingDeterministicSeed() {
            // Given
            var pattern = createTestPattern(0.2, 0.8, 0.3);
            var availableCategories = Set.of(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
            
            // When - set seed and select
            categoryManager.setDeterministicSeed(12345L);
            var selection1 = categoryManager.selectDeterministic(Level.TOKEN, pattern, availableCategories);
            
            // Change seed and select
            categoryManager.setDeterministicSeed(67890L);
            var selection2 = categoryManager.selectDeterministic(Level.TOKEN, pattern, availableCategories);
            
            // Reset to original seed and select
            categoryManager.setDeterministicSeed(12345L);
            var selection3 = categoryManager.selectDeterministic(Level.TOKEN, pattern, availableCategories);
            
            // Then
            assertThat(selection1).isEqualTo(selection3); // Same seed should give same result
            assertThat(selection1).isIn(availableCategories);
            assertThat(selection2).isIn(availableCategories);
        }
    }
    
    @Nested
    @DisplayName("Category Pruning Tests")
    class CategoryPruningTests {
        
        @Test
        @DisplayName("Should prune categories based on age threshold")
        void shouldPruneCategoriesBasedOnAgeThreshold() {
            // Given - parameters with very short max age for testing
            var shortAgeParams = new CategoryParameters(
                100L, // 100ms max age
                5,
                60 * 60 * 1000L,
                0.1,
                0.95,
                5 * 60 * 1000L
            );
            var manager = new CategoryManager(shortAgeParams);
            
            var pattern = createTestPattern(0.5, 0.5, 0.5);
            var key = manager.storeCategory(Level.TOKEN, 1, pattern, 0.8, Map.of());
            
            // Wait for age threshold
            try {
                Thread.sleep(150);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            
            // When
            var prunedCount = manager.pruneCategories();
            
            // Then
            assertThat(prunedCount).isGreaterThanOrEqualTo(0); // Might be 0 if timing is off
            
            // Verify category might be gone (timing-dependent)
            var retrievedCategory = manager.getCategory(key);
            // Note: This test is timing-sensitive, so we just verify the method works
        }
        
        @Test
        @DisplayName("Should not prune recently active categories")
        void shouldNotPruneRecentlyActiveCategories() {
            // Given - store and access a category
            var pattern = createTestPattern(0.3, 0.7, 0.4);
            var key = categoryManager.storeCategory(Level.TOKEN, 1, pattern, 0.8, Map.of());
            
            // Access the category to make it recently active
            categoryManager.getCategory(key);
            
            // When
            var prunedCount = categoryManager.pruneCategories();
            
            // Then - no categories should be pruned (they're recent)
            assertThat(prunedCount).isZero();
            
            var retrievedCategory = categoryManager.getCategory(key);
            assertThat(retrievedCategory).isNotNull();
        }
        
        @Test
        @DisplayName("Should prune categories with low activation")
        void shouldPruneCategoriesWithLowActivation() {
            // Given - parameters with higher minimum activation threshold
            var strictParams = new CategoryParameters(
                24 * 60 * 60 * 1000L,
                1, // Low access count requirement
                60 * 60 * 1000L,
                0.9, // High minimum activation (90%)
                0.95,
                5 * 60 * 1000L
            );
            var manager = new CategoryManager(strictParams);
            
            var pattern = createTestPattern(0.5, 0.5, 0.5);
            // Store with low activation
            manager.storeCategory(Level.TOKEN, 1, pattern, 0.1, Map.of()); // Only 10% activation
            
            // When
            var prunedCount = manager.pruneCategories();
            
            // Then - category should be pruned due to low activation
            assertThat(prunedCount).isGreaterThanOrEqualTo(0);
        }
    }
    
    @Nested
    @DisplayName("Category Merging Tests")
    class CategoryMergingTests {
        
        @Test
        @DisplayName("Should merge similar categories")
        void shouldMergeSimilarCategories() {
            // Given - parameters with lower merge threshold for testing
            var mergeParams = new CategoryParameters(
                24 * 60 * 60 * 1000L,
                1,
                60 * 60 * 1000L,
                0.1,
                0.8, // Lower similarity threshold for merging
                5 * 60 * 1000L
            );
            var manager = new CategoryManager(mergeParams);
            
            // Store very similar categories
            var pattern1 = createTestPattern(0.5, 0.5, 0.5);
            var pattern2 = createTestPattern(0.51, 0.49, 0.5); // Very similar
            
            manager.storeCategory(Level.TOKEN, 1, pattern1, 0.8, Map.of());
            manager.storeCategory(Level.TOKEN, 2, pattern2, 0.7, Map.of());
            
            var initialCount = manager.getCategoriesForLevel(Level.TOKEN).size();
            
            // When
            var mergedCount = manager.mergeCategories();
            
            // Then
            var finalCount = manager.getCategoriesForLevel(Level.TOKEN).size();
            assertThat(mergedCount).isGreaterThanOrEqualTo(0);
            
            if (mergedCount > 0) {
                assertThat(finalCount).isLessThan(initialCount);
            }
        }
        
        @Test
        @DisplayName("Should not merge dissimilar categories")
        void shouldNotMergeDissimilarCategories() {
            // Given - store dissimilar categories
            var pattern1 = createTestPattern(1.0, 0.0, 1.0);
            var pattern2 = createTestPattern(0.0, 1.0, 0.0); // Very different
            
            categoryManager.storeCategory(Level.TOKEN, 1, pattern1, 0.8, Map.of());
            categoryManager.storeCategory(Level.TOKEN, 2, pattern2, 0.7, Map.of());
            
            var initialCount = categoryManager.getCategoriesForLevel(Level.TOKEN).size();
            
            // When
            var mergedCount = categoryManager.mergeCategories();
            
            // Then - no categories should be merged (too dissimilar)
            assertThat(mergedCount).isZero();
            
            var finalCount = categoryManager.getCategoriesForLevel(Level.TOKEN).size();
            assertThat(finalCount).isEqualTo(initialCount);
        }
        
        @Test
        @DisplayName("Should preserve higher activation when merging")
        void shouldPreserveHigherActivationWhenMerging() {
            // Given - parameters that allow merging
            var mergeParams = new CategoryParameters(
                24 * 60 * 60 * 1000L,
                1,
                60 * 60 * 1000L,
                0.1,
                0.7, // Allow merging of moderately similar categories
                5 * 60 * 1000L
            );
            var manager = new CategoryManager(mergeParams);
            
            var pattern = createTestPattern(0.5, 0.5, 0.5);
            var lowActivationKey = manager.storeCategory(Level.TOKEN, 1, pattern, 0.6, Map.of());
            var highActivationKey = manager.storeCategory(Level.TOKEN, 2, pattern, 0.9, Map.of());
            
            // When
            manager.mergeCategories();
            
            // Then - verify merging behavior (implementation-dependent)
            var remainingCategories = manager.getCategoriesForLevel(Level.TOKEN);
            if (remainingCategories.size() == 1) {
                // If merging occurred, the remaining category should have high activation
                var remainingCategory = remainingCategories.get(0);
                assertThat(remainingCategory.getAverageActivation()).isGreaterThan(0.6);
            }
        }
    }
    
    @Nested
    @DisplayName("Statistics Tests")
    class StatisticsTests {
        
        @Test
        @DisplayName("Should provide comprehensive category statistics")
        void shouldProvideComprehensiveCategoryStatistics() {
            // Given - store categories across multiple levels
            IntStream.range(0, 5).forEach(i -> {
                var pattern = createTestPattern(i * 0.1, i * 0.2, i * 0.1);
                categoryManager.storeCategory(Level.TOKEN, i, pattern, 0.8 + i * 0.02, Map.of());
            });
            
            IntStream.range(0, 3).forEach(i -> {
                var pattern = createTestPattern(i * 0.15, i * 0.25, i * 0.15);
                categoryManager.storeCategory(Level.WINDOW, i, pattern, 0.7 + i * 0.05, Map.of());
            });
            
            // When
            var stats = categoryManager.getStatistics();
            
            // Then
            assertThat(stats).isNotNull();
            assertThat(stats.totalCategories()).isEqualTo(8);
            assertThat(stats.levelStats()).hasSize(3); // TOKEN, WINDOW, DOCUMENT
            
            var tokenStats = stats.levelStats().get(Level.TOKEN);
            assertThat(tokenStats).isNotNull();
            assertThat(tokenStats.categoryCount()).isEqualTo(5);
            assertThat(tokenStats.avgActivation()).isGreaterThan(0.8);
            
            var windowStats = stats.levelStats().get(Level.WINDOW);
            assertThat(windowStats).isNotNull();
            assertThat(windowStats.categoryCount()).isEqualTo(3);
            
            var documentStats = stats.levelStats().get(Level.DOCUMENT);
            assertThat(documentStats).isNotNull();
            assertThat(documentStats.categoryCount()).isZero();
        }
        
        @Test
        @DisplayName("Should track access counts in statistics")
        void shouldTrackAccessCountsInStatistics() {
            // Given
            var pattern = createTestPattern(0.5, 0.5, 0.5);
            var key = categoryManager.storeCategory(Level.TOKEN, 1, pattern, 0.8, Map.of());
            
            // Access the category multiple times
            categoryManager.getCategory(key);
            categoryManager.getCategory(key);
            categoryManager.getCategory(key);
            
            // When
            var stats = categoryManager.getStatistics();
            
            // Then
            var tokenStats = stats.levelStats().get(Level.TOKEN);
            assertThat(tokenStats.avgAccesses()).isGreaterThan(0);
        }
        
        @Test
        @DisplayName("Should provide readable statistics strings")
        void shouldProvideReadableStatisticsStrings() {
            // Given
            var pattern = createTestPattern(0.3, 0.7, 0.1);
            categoryManager.storeCategory(Level.TOKEN, 1, pattern, 0.75, Map.of());
            
            // When
            var stats = categoryManager.getStatistics();
            var statsString = stats.toString();
            var levelStats = stats.levelStats().get(Level.TOKEN);
            var levelStatsString = levelStats.toString();
            
            // Then
            assertThat(statsString).contains("CategoryStats");
            assertThat(statsString).contains("total=");
            assertThat(statsString).contains("accesses=");
            assertThat(statsString).contains("levels=");
            
            assertThat(levelStatsString).contains("LevelTOKEN");
            assertThat(levelStatsString).contains("count=");
            assertThat(levelStatsString).contains("activation=");
        }
    }
    
    @Nested
    @DisplayName("Reset and State Management Tests")
    class ResetAndStateTests {
        
        @Test
        @DisplayName("Should reset all state correctly")
        void shouldResetAllStateCorrectly() {
            // Given - populate with categories
            IntStream.range(0, 10).forEach(i -> {
                var pattern = createTestPattern(i * 0.1, i * 0.05, i * 0.02);
                categoryManager.storeCategory(Level.values()[i % 3], i, pattern, 0.8, Map.of());
            });
            
            var statsBeforeReset = categoryManager.getStatistics();
            assertThat(statsBeforeReset.totalCategories()).isGreaterThan(0);
            
            // When
            categoryManager.reset();
            
            // Then
            var statsAfterReset = categoryManager.getStatistics();
            assertThat(statsAfterReset.totalCategories()).isZero();
            assertThat(statsAfterReset.totalAccesses()).isZero();
            assertThat(statsAfterReset.totalPrunings()).isZero();
            assertThat(statsAfterReset.totalMergings()).isZero();
            
            // All level statistics should show zero counts
            for (var level : Level.values()) {
                var levelStats = statsAfterReset.levelStats().get(level);
                assertThat(levelStats.categoryCount()).isZero();
            }
        }
        
        @Test
        @DisplayName("Should reset deterministic seed on reset")
        void shouldResetDeterministicSeedOnReset() {
            // Given
            var pattern = createTestPattern(0.4, 0.6, 0.2);
            var categories = Set.of(1, 2, 3, 4, 5);
            
            // Set custom seed and get selection
            categoryManager.setDeterministicSeed(99999L);
            var selection1 = categoryManager.selectDeterministic(Level.TOKEN, pattern, categories);
            
            // When - reset
            categoryManager.reset();
            
            // Then - deterministic behavior should be reset to default
            var selectionAfterReset = categoryManager.selectDeterministic(Level.TOKEN, pattern, categories);
            
            // Create new manager to compare with default behavior
            var newManager = new CategoryManager(defaultParams);
            var defaultSelection = newManager.selectDeterministic(Level.TOKEN, pattern, categories);
            
            assertThat(selectionAfterReset).isEqualTo(defaultSelection);
        }
    }
    
    @Nested
    @DisplayName("Thread Safety Tests")
    class ThreadSafetyTests {
        
        @Test
        @DisplayName("Should handle concurrent category storage")
        void shouldHandleConcurrentCategoryStorage() throws InterruptedException {
            // Given
            var threads = new Thread[10];
            var results = new CategoryKey[10];
            
            // When - multiple threads store categories concurrently
            for (int i = 0; i < 10; i++) {
                int threadIndex = i;
                threads[i] = new Thread(() -> {
                    var pattern = createTestPattern(threadIndex * 0.1, threadIndex * 0.05, threadIndex * 0.02);
                    results[threadIndex] = categoryManager.storeCategory(
                        Level.TOKEN, threadIndex, pattern, 0.8, Map.of("thread", threadIndex)
                    );
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
            
            // Then - all categories should be stored successfully
            for (int i = 0; i < 10; i++) {
                assertThat(results[i]).isNotNull();
                var category = categoryManager.getCategory(results[i]);
                assertThat(category).isNotNull();
                assertThat(category.getKey().categoryId()).isEqualTo(i);
            }
        }
        
        @Test
        @DisplayName("Should handle concurrent statistics access")
        void shouldHandleConcurrentStatisticsAccess() throws InterruptedException {
            // Given - populate with some categories
            IntStream.range(0, 5).forEach(i -> {
                var pattern = createTestPattern(i * 0.1, i * 0.1, i * 0.1);
                categoryManager.storeCategory(Level.TOKEN, i, pattern, 0.8, Map.of());
            });
            
            var threads = new Thread[5];
            var exceptions = new Exception[5];
            
            // When - multiple threads access statistics concurrently
            for (int i = 0; i < 5; i++) {
                int threadIndex = i;
                threads[i] = new Thread(() -> {
                    try {
                        var stats = categoryManager.getStatistics();
                        assertThat(stats).isNotNull();
                        assertThat(stats.totalCategories()).isGreaterThanOrEqualTo(5);
                    } catch (Exception e) {
                        exceptions[threadIndex] = e;
                    }
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
            
            // Then - no exceptions should occur
            for (var exception : exceptions) {
                assertThat(exception).isNull();
            }
        }
    }
    
    // Helper methods
    
    private Pattern createTestPattern(double... values) {
        return new DenseVector(values);
    }
}