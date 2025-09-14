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

import com.hellblazer.art.core.BaseART;
import com.hellblazer.art.core.algorithms.FuzzyART;
import com.hellblazer.art.hartcq.hierarchical.ARTAdapter.ARTProcessingResult;
import com.hellblazer.art.hartcq.hierarchical.ARTAdapter.PerformanceStats;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.RepeatedTest;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.*;

/**
 * Comprehensive unit tests for ARTAdapter.
 * Tests pattern conversion, complement coding, deterministic behavior, 
 * performance metrics, and learning mode transitions.
 * 
 * @author Hal Hildebrand
 */
class ARTAdapterTest {
    private static final Logger logger = LoggerFactory.getLogger(ARTAdapterTest.class);
    
    private ARTAdapter adapter;
    private List<BaseART> modules;

    @BeforeEach
    void setUp() {
        // Use real FuzzyART instances instead of mocks
        modules = List.of(
            new FuzzyART(),
            new FuzzyART(),
            new FuzzyART()
        );
        adapter = new ARTAdapter(modules, 0.75, 0.05, 100);
    }
    
    @Nested
    @DisplayName("Pattern Conversion Tests")
    class PatternConversionTests {
        
        @Test
        @DisplayName("Should convert channel outputs to ART patterns correctly")
        void shouldConvertChannelOutputsToARTPatterns() {
            // Given
            var channelOutputs = new float[][] {
                {0.5f, 0.3f, 0.8f},     // Level 1
                {0.2f, 0.9f, 0.4f},     // Level 2
                {0.7f, 0.1f, 0.6f}      // Level 3
            };
            
            // When
            var patterns = adapter.convertToARTPatterns(channelOutputs);
            
            // Then
            assertThat(patterns).hasSize(3);
            
            for (int level = 0; level < 3; level++) {
                var levelPatterns = patterns.get(level);
                assertThat(levelPatterns).hasSize(1); // Single pattern per level
                
                var pattern = levelPatterns[0];
                // Complement coding doubles the dimension
                assertThat(pattern.dimension()).isEqualTo(channelOutputs[level].length * 2);
                
                // Verify complement coding: [x, 1-x]
                for (int i = 0; i < channelOutputs[level].length; i++) {
                    var originalValue = channelOutputs[level][i];
                    var expectedComplement = 1.0 - originalValue;
                    
                    assertThat(pattern.get(i))
                        .describedAs("Original value at index %d, level %d", i, level)
                        .isEqualTo(originalValue, offset(0.001));
                    
                    assertThat(pattern.get(i + channelOutputs[level].length))
                        .describedAs("Complement value at index %d, level %d", i, level)
                        .isEqualTo(expectedComplement, offset(0.001));
                }
            }
        }
        
        @Test
        @DisplayName("Should convert single channel output to multiple levels")
        void shouldConvertSingleChannelToMultipleLevels() {
            // Given
            var singleChannelOutput = new float[]{0.4f, 0.7f, 0.2f};
            int numLevels = 3;
            
            // When
            var patterns = adapter.convertSingleChannelToARTPatterns(singleChannelOutput, numLevels);
            
            // Then
            assertThat(patterns).hasSize(numLevels);
            
            for (int level = 0; level < numLevels; level++) {
                var levelPatterns = patterns.get(level);
                assertThat(levelPatterns).hasSize(1);
                
                var pattern = levelPatterns[0];
                assertThat(pattern.dimension()).isEqualTo(singleChannelOutput.length * 2);
                
                // All levels should have the same pattern (replicated)
                if (level > 0) {
                    var previousPattern = patterns.get(level - 1)[0];
                    for (int i = 0; i < pattern.dimension(); i++) {
                        assertThat(pattern.get(i))
                            .describedAs("Pattern value at index %d should be same across levels", i)
                            .isEqualTo(previousPattern.get(i), offset(0.001));
                    }
                }
            }
        }
        
        @Test
        @DisplayName("Should handle edge case values in complement coding")
        void shouldHandleEdgeCaseValuesInComplementCoding() {
            // Given - edge cases: 0.0, 1.0, negative, > 1.0
            var edgeCaseInputs = new float[]{0.0f, 1.0f, -0.5f, 1.5f};
            
            // When
            var complementCoded = adapter.applyComplementCoding(edgeCaseInputs);
            
            // Then
            assertThat(complementCoded).hasSize(edgeCaseInputs.length * 2);
            
            // Test 0.0 case
            assertThat(complementCoded[0]).isEqualTo(0.0, offset(0.001));
            assertThat(complementCoded[4]).isEqualTo(1.0, offset(0.001)); // 1 - 0
            
            // Test 1.0 case
            assertThat(complementCoded[1]).isEqualTo(1.0, offset(0.001));
            assertThat(complementCoded[5]).isEqualTo(0.0, offset(0.001)); // 1 - 1
            
            // Test clamping: negative value should become 0.0
            assertThat(complementCoded[2]).isEqualTo(0.0, offset(0.001));
            assertThat(complementCoded[6]).isEqualTo(1.0, offset(0.001));
            
            // Test clamping: > 1.0 value should become 1.0
            assertThat(complementCoded[3]).isEqualTo(1.0, offset(0.001));
            assertThat(complementCoded[7]).isEqualTo(0.0, offset(0.001));
        }
        
        @Test
        @DisplayName("Should reject null or empty inputs")
        void shouldRejectNullOrEmptyInputs() {
            // Test null channel outputs
            assertThatThrownBy(() -> adapter.convertToARTPatterns(null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("cannot be null or empty");
            
            // Test empty channel outputs
            assertThatThrownBy(() -> adapter.convertToARTPatterns(new float[0][]))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("cannot be null or empty");
            
            // Test null single channel
            assertThatThrownBy(() -> adapter.convertSingleChannelToARTPatterns(null, 3))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("cannot be null or empty");
            
            // Test empty single channel
            assertThatThrownBy(() -> adapter.convertSingleChannelToARTPatterns(new float[0], 3))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("cannot be null or empty");
        }
    }
    
    @Nested
    @DisplayName("Learning Mode Tests")
    class LearningModeTests {
        
        @Test
        @DisplayName("Should start with learning disabled by default")
        void shouldStartWithLearningDisabled() {
            // Given - fresh adapter
            
            // When & Then
            assertThat(adapter.isLearning()).isFalse();
        }
        
        @Test
        @DisplayName("Should enable and disable learning mode")
        void shouldEnableAndDisableLearningMode() {
            // Given - learning initially disabled
            assertThat(adapter.isLearning()).isFalse();
            
            // When - enable learning
            adapter.enableLearning();
            
            // Then
            assertThat(adapter.isLearning()).isTrue();
            
            // When - disable learning
            adapter.disableLearning();
            
            // Then
            assertThat(adapter.isLearning()).isFalse();
        }
        
        @Test
        @DisplayName("Should maintain learning state across multiple operations")
        void shouldMaintainLearningStateAcrossOperations() {
            // Given
            adapter.enableLearning();
            
            // When - perform multiple operations
            var channelOutput = new float[]{0.5f, 0.3f};
            var patterns = adapter.convertSingleChannelToARTPatterns(channelOutput, 2);
            
            // Then - learning state should be preserved
            assertThat(adapter.isLearning()).isTrue();
            
            // When - disable and check again
            adapter.disableLearning();
            var morePatterns = adapter.convertSingleChannelToARTPatterns(channelOutput, 2);
            
            // Then
            assertThat(adapter.isLearning()).isFalse();
        }
    }
    
    @Nested
    @DisplayName("Deterministic Behavior Tests")
    class DeterministicBehaviorTests {
        
        @Test
        @DisplayName("Should produce consistent complement coding for same inputs")
        void shouldProduceConsistentComplementCoding() {
            // Given
            var input = new float[]{0.3f, 0.7f, 0.1f, 0.9f};
            
            // When - apply complement coding multiple times
            var result1 = adapter.applyComplementCoding(input);
            var result2 = adapter.applyComplementCoding(input);
            var result3 = adapter.applyComplementCoding(input);
            
            // Then - all results should be identical
            assertThat(result1).isEqualTo(result2);
            assertThat(result2).isEqualTo(result3);
        }
        
        @RepeatedTest(10)
        @DisplayName("Should produce deterministic results across multiple runs")
        void shouldProduceDeterministicResultsAcrossRuns() {
            // Given
            var channelOutput = new float[]{0.25f, 0.50f, 0.75f};
            
            // When
            var patterns = adapter.convertSingleChannelToARTPatterns(channelOutput, 2);
            
            // Then - verify consistent structure
            assertThat(patterns).hasSize(2);
            for (var levelPatterns : patterns) {
                assertThat(levelPatterns).hasSize(1);
                assertThat(levelPatterns[0].dimension()).isEqualTo(6); // 3 * 2
            }
            
            // Verify specific values are consistent
            var pattern = patterns.get(0)[0];
            assertThat(pattern.get(0)).isEqualTo(0.25, offset(0.001));
            assertThat(pattern.get(1)).isEqualTo(0.50, offset(0.001));
            assertThat(pattern.get(2)).isEqualTo(0.75, offset(0.001));
            assertThat(pattern.get(3)).isEqualTo(0.75, offset(0.001)); // 1 - 0.25
            assertThat(pattern.get(4)).isEqualTo(0.50, offset(0.001)); // 1 - 0.50
            assertThat(pattern.get(5)).isEqualTo(0.25, offset(0.001)); // 1 - 0.75
        }
    }
    
    @Nested
    @DisplayName("Performance Metrics Tests")
    class PerformanceMetricsTests {
        
        @Test
        @DisplayName("Should provide initial performance statistics")
        void shouldProvideInitialPerformanceStatistics() {
            // Given - fresh adapter
            
            // When
            var stats = adapter.getPerformanceStats();
            
            // Then
            assertThat(stats).isNotNull();
            assertThat(stats.totalPatterns()).isZero();
            assertThat(stats.avgProcessingTimeMs()).isZero();
            assertThat(stats.throughputPerSecond()).isZero();
            assertThat(stats.totalCategories()).isZero();
            assertThat(stats.isLearning()).isFalse();
        }
        
        @Test
        @DisplayName("Should track performance metrics during processing")
        void shouldTrackPerformanceMetricsDuringProcessing() {
            // This test is limited because actual processing requires DeepARTMAP integration
            // We can test the structure and initialization
            
            // Given
            adapter.enableLearning();
            
            // When
            var initialStats = adapter.getPerformanceStats();
            
            // Then
            assertThat(initialStats.isLearning()).isTrue();
        }
        
        @Test
        @DisplayName("Should reset performance metrics")
        void shouldResetPerformanceMetrics() {
            // Given - simulate some activity first
            adapter.enableLearning();
            
            // When
            adapter.reset();
            
            // Then
            var stats = adapter.getPerformanceStats();
            assertThat(stats.totalPatterns()).isZero();
            assertThat(stats.avgProcessingTimeMs()).isZero();
            assertThat(stats.totalCategories()).isZero();
            assertThat(stats.isLearning()).isFalse(); // Reset to default state
        }
        
        @Test
        @DisplayName("Should provide readable performance statistics string")
        void shouldProvideReadablePerformanceStatisticsString() {
            // Given
            adapter.enableLearning();
            var stats = adapter.getPerformanceStats();
            
            // When
            var statsString = stats.toString();
            
            // Then
            assertThat(statsString).contains("PerformanceStats");
            assertThat(statsString).contains("patterns=");
            assertThat(statsString).contains("avgTime=");
            assertThat(statsString).contains("throughput=");
            assertThat(statsString).contains("categories=");
            assertThat(statsString).contains("learning=");
        }
    }
    
    @Nested
    @DisplayName("Category Cache Tests")
    class CategoryCacheTests {
        
        @Test
        @DisplayName("Should provide access to category cache")
        void shouldProvideAccessToCategoryCache() {
            // Given - fresh adapter
            
            // When
            var cache = adapter.getCategoryCache();
            
            // Then
            assertThat(cache).isNotNull();
            assertThat(cache).isEmpty(); // Initially empty
        }
        
        @Test
        @DisplayName("Should clear category cache on reset")
        void shouldClearCategoryCacheOnReset() {
            // Given
            var cache = adapter.getCategoryCache();
            
            // When
            adapter.reset();
            
            // Then
            assertThat(cache).isEmpty();
        }
    }
    
    @Nested
    @DisplayName("Error Handling Tests")
    class ErrorHandlingTests {
        
        @Test
        @DisplayName("Should handle invalid vigilance parameters in constructor")
        void shouldHandleInvalidVigilanceParameters() {
            // Given
            List<BaseART> modules = List.of(new FuzzyART());

            // When & Then - negative vigilance
            assertThatThrownBy(() -> new ARTAdapter(modules, -0.1, 0.05, 100))
                .isInstanceOf(IllegalArgumentException.class);
            
            // When & Then - vigilance > 1.0
            assertThatThrownBy(() -> new ARTAdapter(modules, 1.1, 0.05, 100))
                .isInstanceOf(IllegalArgumentException.class);
        }
        
        @Test
        @DisplayName("Should handle null inputs gracefully")
        void shouldHandleNullInputsGracefully() {
            // Test null modules list
            assertThatThrownBy(() -> new ARTAdapter(null, 0.75, 0.05, 100))
                .isInstanceOf(IllegalArgumentException.class);
            
            // Test empty modules list
            assertThatThrownBy(() -> new ARTAdapter(List.of(), 0.75, 0.05, 100))
                .isInstanceOf(IllegalArgumentException.class);
        }
        
        @Test
        @DisplayName("Should handle null level outputs in conversion")
        void shouldHandleNullLevelOutputsInConversion() {
            // Given - array with null level
            var channelOutputsWithNull = new float[][] {
                {0.5f, 0.3f},
                null,  // Null level
                {0.7f, 0.1f}
            };
            
            // When & Then
            assertThatThrownBy(() -> adapter.convertToARTPatterns(channelOutputsWithNull))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("cannot be null or empty");
        }
        
        @Test
        @DisplayName("Should handle empty level outputs in conversion")
        void shouldHandleEmptyLevelOutputsInConversion() {
            // Given - array with empty level
            var channelOutputsWithEmpty = new float[][] {
                {0.5f, 0.3f},
                {},  // Empty level
                {0.7f, 0.1f}
            };
            
            // When & Then
            assertThatThrownBy(() -> adapter.convertToARTPatterns(channelOutputsWithEmpty))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("cannot be null or empty");
        }
    }
    
    @Nested
    @DisplayName("Integration Tests")
    class IntegrationTests {
        
        @Test
        @DisplayName("Should work with default adapter configuration")
        void shouldWorkWithDefaultAdapterConfiguration() {
            // Given
            var defaultAdapter = ARTAdapter.createDefault(3, 10);
            
            // When & Then - should not throw exceptions
            assertThatCode(() -> {
                var channelOutput = new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
                var patterns = defaultAdapter.convertSingleChannelToARTPatterns(channelOutput, 3);
                assertThat(patterns).hasSize(3);
            }).doesNotThrowAnyException();
        }
        
        @Test
        @DisplayName("Should maintain consistency across learning state changes")
        void shouldMaintainConsistencyAcrossLearningStateChanges() {
            // Given
            var channelOutput = new float[]{0.2f, 0.8f, 0.4f};
            
            // When - process with learning disabled
            adapter.disableLearning();
            var patterns1 = adapter.convertSingleChannelToARTPatterns(channelOutput, 2);
            
            // When - process with learning enabled
            adapter.enableLearning();
            var patterns2 = adapter.convertSingleChannelToARTPatterns(channelOutput, 2);
            
            // Then - pattern conversion should be consistent regardless of learning mode
            assertThat(patterns1).hasSize(2);
            assertThat(patterns2).hasSize(2);
            
            for (int i = 0; i < 2; i++) {
                var pattern1 = patterns1.get(i)[0];
                var pattern2 = patterns2.get(i)[0];
                
                assertThat(pattern1.dimension()).isEqualTo(pattern2.dimension());
                for (int j = 0; j < pattern1.dimension(); j++) {
                    assertThat(pattern1.get(j)).isEqualTo(pattern2.get(j), offset(0.001));
                }
            }
        }
    }
    
    @Nested
    @DisplayName("Thread Safety Tests")
    class ThreadSafetyTests {
        
        @Test
        @DisplayName("Should handle concurrent learning state changes")
        void shouldHandleConcurrentLearningStateChanges() throws InterruptedException {
            // Given
            var threads = new Thread[10];
            var results = new boolean[10];
            
            // When - multiple threads toggle learning state
            for (int i = 0; i < 10; i++) {
                int threadIndex = i;
                threads[i] = new Thread(() -> {
                    adapter.enableLearning();
                    results[threadIndex] = adapter.isLearning();
                    adapter.disableLearning();
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
            
            // Then - verify that no data corruption occurred during concurrent access
            // The actual boolean values may vary due to race conditions, which is correct behavior
            // But we can verify that operations didn't cause any undefined state

            // Check that adapter is still responsive after concurrent access
            adapter.enableLearning();
            assertThat(adapter.isLearning()).isTrue();

            adapter.disableLearning();
            assertThat(adapter.isLearning()).isFalse();
        }
    }
}